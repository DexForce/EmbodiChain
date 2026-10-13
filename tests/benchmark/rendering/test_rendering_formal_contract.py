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
from copy import deepcopy
from types import SimpleNamespace

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
        assert profile["version"] == "1.8-ea"
        assert (
            profile["embodichain"]["state_publication_policy"]
            == "on_frozen_state_change"
        )
        # NRD exposes jittered public geometry and FastRT range depth. Formal
        # RGBD+Seg cells require the native output-grid AOV path instead.
        assert profile["embodichain"]["denoising"] == "dlss"
        assert profile["embodichain"]["dlss_quality"] == 5
        assert profile["embodichain"]["dlss_tiled"] is True
        assert (
            profile["isaaclab"]["carb_settings"]["/rtx/post/histogram/enabled"] is False
        )
        assert profile["isaaclab"]["carb_settings"]["/rtx/post/tonemap/op"] == 0
        assert (
            profile["isaaclab"]["carb_settings"]["/rtx/sceneDb/ambientLightIntensity"]
            == 0.0
        )
        assert profile["isaaclab_light_contract"]["usdlux_version"] == 2505
        assert profile["isaaclab_light_contract"]["sun_normalize"] is True


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
    assert batch.assembly_operations == [
        {
            "operation": "environment_chunk_concat",
            "modality": "rgb",
            "calls": 1,
            "output_bytes": 65 * 8,
        }
    ]
    assert batch.get_intrinsics().tolist() == list(range(65))
    assert batch.get_arena_pose(True).tolist() == list(range(65))
    poses = np.arange(65 * 3).reshape(65, 3)
    batch.look_at(poses, poses + 1, poses + 2)
    for camera in created:
        for offset in range(3):
            np.testing.assert_equal(camera.received[offset], poses[camera.ids] + offset)
    batch.destroy()
    assert destroyed == list(range(65))


def _native_adapter(events: list[str]) -> EmbodiChainScene:
    """Exercise real adapter methods with CPU tensors and a strict native fake."""
    import torch

    class World:
        def __init__(self):
            self.submissions = []
            self.exposures = 0

        def _flush_render_updates(self):
            events.append("flush")
            if sim.fail_at == "flush":
                raise RuntimeError("flush failed")

        def render_camera_group(self, groups):
            events.append("render")
            if sim.fail_at == "render":
                raise RuntimeError("render failed")
            self.submissions.append(groups)
            self.exposures += 1

    class Sim:
        fail_at = None

        @contextmanager
        def render_frame(self, **kwargs):
            assert kwargs == {"capture_camera_images": False}
            events.append("enter")
            try:
                if self.fail_at == "enter":
                    raise RuntimeError("enter failed")
                yield
            finally:
                events.append("exit")

        def sync_render_state(self):
            events.append("publish")
            if self.fail_at == "publish":
                raise RuntimeError("publish failed")

        def render_camera_group(self, groups):
            raise AssertionError("Manager render wrapper would republish static state")

        def get_world(self):
            return world

    class Camera:
        def __init__(self, index):
            self.index = index
            self.group_ids = [10 + 2 * index, 11 + 2 * index]

        def get_intrinsics(self):
            return torch.eye(3).repeat(2, 1, 1)

        def update(self, **kwargs):
            assert kwargs == {"fetch_only": True}

        def get_data(self):
            return {
                "color": torch.full((2, 2, 2, 4), world.exposures, dtype=torch.uint8)
            }

        def look_at(self, eye, target, up):
            assert eye.shape == target.shape == up.shape == (2, 3)
            events.append("camera")

        def get_arena_pose(self, to_matrix=False):
            assert to_matrix
            return torch.eye(4).repeat(2, 1, 1)

    sim, world = Sim(), World()
    adapter = object.__new__(EmbodiChainScene)
    adapter.sim = sim
    adapter.torch, adapter.device = torch, "cpu"
    adapter.num_envs = 2
    adapter.modalities = ("rgb",)
    adapter._linear_rgb_source = False
    adapter._pending_frame = None
    adapter._state_publications = adapter._state_flushes = 0
    adapter._render_calls = adapter._scene_frame_id = 0
    adapter._pending_native_flush = True
    adapter.cameras = [Camera(index) for index in range(3)]
    adapter._object_poses = {"box": torch.eye(4).tolist()}
    adapter.objects = {
        "box": SimpleNamespace(set_local_pose=lambda values: events.append("object"))
    }
    adapter.scene = {
        "cameras": [
            {
                "name": name,
                "eye": [1.0, -1.0, 1.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
            }
            for name in ("external_front", "external_side", "wrist")
        ]
    }
    adapter.metadata = {}
    return adapter


@pytest.mark.parametrize("published_before_render", [False, True])
def test_static_native_captures_submit_every_group_without_republishing(
    published_before_render: bool,
) -> None:
    events = []
    adapter = _native_adapter(events)
    if published_before_render:
        adapter._begin_frame()
    for exposure in range(1, 4):
        arrays = adapter.render_device()
        assert arrays["rgb"].shape == (6, 2, 2, 3)
        assert arrays["rgb"].unique().tolist() == [exposure]
    assert events == [
        "enter",
        "publish",
        "flush",
        "render",
        "exit",
        "render",
        "render",
    ]
    assert adapter.sim.get_world().submissions == [[10, 11, 12, 13, 14, 15]] * 3
    assert adapter._pending_frame is None
    metadata = adapter.refresh_metadata()
    assert metadata["state_publications"] == metadata["changed_state_flushes"] == 1
    assert metadata["render_submissions"] == 3
    evidence = adapter.capture_evidence()
    assert evidence["render_submission_id"] == 3
    assert evidence["logical_state"]["frame_id"] == 0
    assert evidence["logical_state"]["num_envs"] == 2
    assert evidence["logical_state"]["camera_names"] == [
        "external_front",
        "external_side",
        "wrist",
    ]
    assert evidence["renderer_source_frame_id"] is None
    assert evidence["renderer_source_timestamp_s"] is None
    # One stack plus actual strided RGB packing; unchanged buffers are not
    # interpreted as stale, and view-only flattening is not counted as a copy.
    assert evidence["assembly"]["operations"] == [
        {
            "operation": "camera_stack",
            "modality": "rgb",
            "calls": 1,
            "output_bytes": 96,
        },
        {
            "operation": "contiguous_pack",
            "modality": "rgb",
            "calls": 1,
            "output_bytes": 72,
        },
    ]


@pytest.mark.parametrize("changed", ["object", "camera", "both"])
def test_changed_frozen_state_publishes_during_prepare_before_exposure(
    changed: str,
) -> None:
    events = []
    adapter = _native_adapter(events)
    adapter.render_device()
    events.clear()
    frame = {
        "frame_id": 1,
        "time_s": 1 / 30,
        "objects": deepcopy(adapter._object_poses),
        "cameras": {view["name"]: deepcopy(view) for view in adapter.scene["cameras"]},
    }
    if changed in {"object", "both"}:
        frame["objects"]["box"][0][3] = 0.25
    if changed in {"camera", "both"}:
        frame["cameras"]["wrist"]["eye"][0] = 0.5
    adapter.apply_state(frame)
    assert events[-3:] == ["enter", "publish", "flush"]
    assert "render" not in events
    adapter.render_device()
    assert events[-2:] == ["render", "exit"]
    assert adapter._state_publications == adapter._state_flushes == 2
    assert adapter.capture_evidence()["logical_state"]["time_s"] == 1 / 30
    events.clear()
    frame["frame_id"] = 2
    adapter.apply_state(frame)
    adapter.render_device()
    assert events == ["render"]
    assert adapter._scene_frame_id == 2
    assert adapter._state_publications == adapter._state_flushes == 2


def test_camera_changes_are_dirty_and_identical_specs_do_not_republish() -> None:
    events = []
    adapter = _native_adapter(events)
    adapter.render_device()
    events.clear()
    adapter.set_cameras(deepcopy(adapter.scene["cameras"]))
    assert not events
    changed = deepcopy(adapter.scene["cameras"])
    changed[1]["target"][2] = 0.5
    adapter.set_cameras(changed)
    assert adapter._pending_native_flush
    assert events == ["camera"] * 3
    adapter.set_cameras(deepcopy(changed))
    assert events == ["camera"] * 3
    adapter.render_device()
    assert events[3:] == ["enter", "publish", "flush", "render", "exit"]
    assert adapter._state_publications == adapter._state_flushes == 2


def test_last_capture_identity_survives_later_frozen_state_publication() -> None:
    adapter = _native_adapter([])
    frame = {
        "frame_id": 99,
        "time_s": 99 / 30,
        "objects": deepcopy(adapter._object_poses),
        "cameras": {view["name"]: deepcopy(view) for view in adapter.scene["cameras"]},
    }
    adapter.apply_state(frame)
    adapter.render_device()
    captured = adapter.capture_evidence()
    frame = deepcopy(frame)
    frame.update(frame_id=3, time_s=3 / 30)
    frame["objects"]["box"][0][3] = 0.5
    adapter.apply_state(frame)
    assert adapter._scene_frame_id == 3
    assert adapter.capture_evidence() == captured
    assert captured["logical_state"]["frame_id"] == 99
    assert captured["logical_state"]["time_s"] == 99 / 30
    assert adapter.refresh_metadata()["last_capture_evidence"] == captured
    adapter._end_frame()


@pytest.mark.parametrize("failure", ["enter", "publish", "flush", "render"])
def test_failed_native_publication_or_render_releases_context(failure: str) -> None:
    events = []
    adapter = _native_adapter(events)
    adapter.sim.fail_at = failure
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        adapter.render_device()
    assert events[0] == "enter" and events[-1] == "exit"
    assert adapter._pending_frame is None
    assert adapter._pending_native_flush == (failure != "render")
    assert adapter._state_publications == int(failure in {"flush", "render"})
    assert adapter._state_flushes == int(failure == "render")
    adapter.sim.fail_at = None
    adapter.render_device()
    assert adapter._pending_frame is None
    assert not adapter._pending_native_flush
    assert adapter._state_flushes == 1
    assert adapter._render_calls == 1


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
