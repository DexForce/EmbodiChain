# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Document-level matrix and native publication boundaries."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from scripts.benchmark.rendering.formal import experiment_cases, render_profile
from scripts.benchmark.rendering.backends.scene_embodichain import (
    EmbodiChainScene,
    _CameraBatch,
    _linear_u8_to_srgb_u8,
    _native_lighting,
)


def test_linear_native_bytes_are_delivered_as_display_srgb() -> None:
    import torch

    raw = torch.tensor([0, 1, 38, 128, 255], dtype=torch.uint8)
    encoded = _linear_u8_to_srgb_u8(raw, torch)
    assert encoded.dtype == torch.uint8
    assert encoded.device == raw.device
    assert encoded.tolist() == [0, 13, 108, 188, 255]
    assert raw.tolist() == [0, 1, 38, 128, 255]


def test_single_and_three_camera_matrices_match_document() -> None:
    single, triple = experiment_cases("R-04"), experiment_cases("R-05")
    assert {case.num_envs for case in single} == {1, 16, 64, 256, 1024}
    assert {case.num_envs for case in triple} == {1, 16, 64, 256, 1024}
    assert {case.cameras_per_env for case in single} == {1}
    assert {case.cameras_per_env for case in triple} == {3}
    assert {(case.width, case.height, case.modalities) for case in single + triple} == {
        (128, 128, ("rgb",))
    }


def test_quality_and_single_camera_coverage_is_not_pilot_coverage() -> None:
    static = experiment_cases("R-01")
    assert {case.view_count for case in static} == {20}
    assert {case.scene_id for case in static} == {"calibration", "workshop"}
    assert {case.num_envs for case in static} == {1, 64}
    assert {case.sequence_frames for case in experiment_cases("R-02")} == {100}
    cases = experiment_cases("R-03")
    assert len(cases) == 48
    assert {(case.width, case.height) for case in cases} == {
        (256, 256),
        (512, 512),
        (1920, 1080),
    }
    assert {case.modalities for case in cases} == {("rgb",), ("rgb", "depth", "seg")}
    assert {case.profile for case in cases} == {"realtime", "high_quality"}
    assert {case.temporal_mode for case in cases} == {"static", "moving"}


def test_profiles_are_independent_and_forbid_frame_generation() -> None:
    original = render_profile("realtime")
    changed = render_profile("realtime")
    changed["embodichain"]["spp"] = 999
    assert render_profile("realtime") == original
    assert original["frame_generation"] is False
    for name in ("realtime", "high_quality"):
        profile = render_profile(name)
        # NRD exposes jittered public geometry and FastRT range depth. Formal
        # RGBD+Seg cells require the native output-grid AOV path instead.
        assert profile["embodichain"]["denoising"] == "dlss"
        assert profile["embodichain"]["dlss_quality"] == 5
        assert profile["embodichain"]["dlss_tiled"] is True
        assert (
            profile["isaaclab"]["carb_settings"]["/rtx/post/histogram/enabled"] is False
        )


def test_native_lighting_overrides_preserve_scene_and_other_backend() -> None:
    from copy import deepcopy

    scene = {
        "lighting": {
            "dexsim_directional_intensity": 3.0,
            "dexsim_environment_intensity": 100.0,
        }
    }
    original = deepcopy(scene)
    profile = {
        "lighting_controls": {
            "embodichain": {"environment_intensity": 17.0},
            "isaaclab": {"environment_intensity": 250.0},
        }
    }
    resolved = _native_lighting(scene, profile)
    assert resolved == {
        "directional_intensity": 3.0,
        "environment_intensity": 17.0,
        "environment_color": [1.0, 1.0, 1.0],
    }
    assert scene == original
    assert profile["lighting_controls"]["isaaclab"]["environment_intensity"] == 250.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"environment_intensity": float("nan")},
        {"directional_intensity": -1},
        {"environment_color": [1, 1]},
        {"exposure": 2},
    ],
)
def test_invalid_lighting_controls_fail_before_runtime_creation(
    overrides: dict,
) -> None:
    scene = {
        "lighting": {
            "dexsim_directional_intensity": 3.0,
            "dexsim_environment_intensity": 100.0,
        }
    }
    with pytest.raises(ValueError):
        _native_lighting(scene, {"lighting_controls": {"embodichain": overrides}})


def test_native_chunks_preserve_every_environment_and_pose_slice() -> None:
    import numpy as np
    import torch
    from tensordict import TensorDict
    from types import SimpleNamespace

    class Owner:
        num_envs = 65
        device = "cpu"

        def get_env(self, index):
            return index

        def get_world(self):
            return "shared-world"

    created, destroyed = [], []

    class Camera:
        def __init__(self, config, device, owner):
            assert owner.get_world() == "shared-world"
            self.ids = np.asarray([owner.get_env(i) for i in range(owner.num_envs)])
            self.group_id = len(created)
            self.received = None
            created.append(self)

        def update(self, **kwargs):
            assert kwargs == {"fetch_only": True}

        def get_data(self):
            return TensorDict(
                {"color": torch.from_numpy(self.ids[:, None, None, None])},
                batch_size=[len(self.ids)],
            )

        def get_intrinsics(self):
            return self.ids

        def get_arena_pose(self, to_matrix=False):
            return self.ids

        def look_at(self, eye, target, up):
            self.received = (eye.copy(), target.copy(), up.copy())

        def destroy(self):
            destroyed.extend(self.ids.tolist())

    tensor_api = SimpleNamespace(
        cat=lambda values, dim: np.concatenate(values, axis=dim)
    )
    batch = _CameraBatch(Camera, None, Owner(), tensor_api, 32)
    assert [len(camera.ids) for camera in created] == [32, 32, 1]
    assert batch.group_ids == [0, 1, 2]
    batch.update(fetch_only=True)
    assert batch.get_data()["color"].ravel().tolist() == list(range(65))
    assert batch.get_intrinsics().tolist() == list(range(65))
    assert batch.get_arena_pose(True).tolist() == list(range(65))
    poses = np.arange(65 * 3).reshape(65, 3)
    batch.look_at(poses, poses + 1, poses + 2)
    for camera in created:
        for offset in range(3):
            np.testing.assert_equal(camera.received[offset], poses[camera.ids] + offset)
    batch.destroy()
    assert destroyed == list(range(65))


@pytest.mark.parametrize("published_before_render", [False, True])
def test_native_render_reuses_one_state_publication(
    published_before_render: bool,
) -> None:
    events = []

    class Sim:
        @contextmanager
        def render_frame(self, **kwargs):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        def sync_render_state(self):
            events.append("publish")

    adapter = object.__new__(EmbodiChainScene)
    adapter.sim = Sim()
    adapter._pending_frame = None
    adapter._state_publications = 0

    def render():
        events.append("render")
        return {"rgb": object()}

    adapter._render_published_device = render
    if published_before_render:
        adapter._begin_frame()
    adapter.render_device()
    assert events == ["enter", "publish", "render", "exit"]
    assert adapter._pending_frame is None
    assert adapter._state_publications == 1


def test_failed_native_render_releases_publication_scope() -> None:
    class Sim:
        @contextmanager
        def render_frame(self, **kwargs):
            yield

        def sync_render_state(self):
            pass

    adapter = object.__new__(EmbodiChainScene)
    adapter.sim = Sim()
    adapter._pending_frame = None
    adapter._state_publications = 0

    def fail():
        raise RuntimeError("native render failed")

    adapter._render_published_device = fail
    with pytest.raises(RuntimeError, match="native render failed"):
        adapter.render_device()
    assert adapter._pending_frame is None


def test_changed_transforms_flush_before_exposure_without_repeating_static_flush() -> (
    None
):
    from types import SimpleNamespace

    events = []

    class Sim:
        _world = SimpleNamespace(_flush_render_updates=lambda: events.append("flush"))

        @contextmanager
        def render_frame(self, **kwargs):
            yield

        def sync_render_state(self):
            events.append("publish")

    adapter = object.__new__(EmbodiChainScene)
    adapter.sim = Sim()
    adapter._pending_frame = None
    adapter._state_publications = adapter._state_flushes = 0
    adapter._pending_native_flush = True
    adapter._render_published_device = lambda: events.append("exposure")
    adapter.render_device()
    adapter.render_device()
    assert events == ["publish", "flush", "exposure", "publish", "exposure"]
    assert adapter._state_flushes == 1
    assert adapter._state_publications == 2


def test_freshness_probe_rejects_stale_secondary_cameras(tmp_path, monkeypatch) -> None:
    import numpy as np
    from types import SimpleNamespace
    from scripts.benchmark.rendering import formal_worker
    from scripts.benchmark.rendering.formal import RenderCase

    names = ("external_front", "external_side", "wrist")
    initial = {
        "cameras": {
            name: {
                "name": name,
                "eye": [0.0, 0.0, 1.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
            }
            for name in names
        }
    }

    class Adapter:
        moved = False

        def apply_state(self, state):
            self.moved = False

        def set_cameras(self, views):
            self.moved = views[0]["eye"][0] != 0

    def capture(adapter, case, index):
        value = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        # Only the first camera updates: the historical first-view probe passed.
        if adapter.moved:
            value[0] = 100
        return SimpleNamespace(arrays={"rgb": value})

    monkeypatch.setattr(formal_worker, "_capture", capture)
    case = RenderCase(
        "R-05", "franka_workstation", width=16, height=16, cameras_per_env=3
    )
    evidence = formal_worker._freshness_probe(Adapter(), case, initial, tmp_path)
    assert evidence["exposure_evidence"][0]["passed"]
    assert not evidence["exposure_evidence"][1]["passed"]
    assert not evidence["passed"]
