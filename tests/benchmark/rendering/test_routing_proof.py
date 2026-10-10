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

"""CPU-fake routing controls: no simulator, CUDA context or physical quality claim."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.benchmark.rendering.routing_proof import (
    publication_callback_wall,
    publish_indexed_camera,
    run_routing_probe,
)


def binding() -> dict[str, Any]:
    return {
        "run_id": "synthetic_routing_test",
        "case_id": "synthetic_case",
        "backend": "embodichain",
        "config_sha256": "1" * 64,
        "scene_sha256": "2" * 64,
        "view_sha256": "3" * 64,
        "profile_sha256": "4" * 64,
        "hardware_id": "explicit_fake_cpu_device",
        "source_commit": "5" * 40,
        "source_files": {"fixture": "6" * 64},
        "synthetic_test_fixture": True,
    }


class FakeAdapter:
    def __init__(
        self,
        *,
        mode: str = "identity",
        modalities: tuple[str, ...] = ("rgb",),
        envs: int = 5,
        cameras: int = 2,
    ) -> None:
        self.mode, self.modalities = mode, modalities
        self.num_envs, self.camera_count = envs, cameras
        self.width, self.height = 8, 6
        self.overrides: dict[tuple[int, int], bool] = {}
        self.calls, self.capture_count = [], 0
        self.specs = [
            {
                "name": f"camera_{i}",
                "eye": [0.0, 0.0, 1.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
            }
            for i in range(cameras)
        ]
        self.fail_moved_capture = False
        self.fail_after_mutation = False
        self.moved_captures = 0

    def set(self, env: int, camera: int, spec: dict[str, Any]) -> dict[str, Any]:
        moved = spec["eye"][0] != 0
        self.calls.append((env, camera, moved))
        self.overrides[env, camera] = moved
        self.moved_captures = 0
        if self.fail_after_mutation and moved:
            self.fail_after_mutation = False
            raise RuntimeError("injected after selected camera changed")
        return {
            "sensor_index": env * self.camera_count + camera,
            "synthetic_test_fixture": True,
        }

    def capture(self, index: int) -> Any:
        if self.fail_moved_capture and any(self.overrides.values()):
            self.fail_moved_capture = False
            raise RuntimeError("injected capture error")
        arrays = {}
        for modality in self.modalities:
            values = []
            for env in range(self.num_envs):
                for camera in range(self.camera_count):
                    source = env
                    if self.mode == "broadcast" or (
                        self.mode == "depth_broadcast" and modality == "depth"
                    ):
                        source = 0
                    elif self.mode == "swap":
                        source = (
                            self.num_envs - 1
                            if env == 0
                            else 0 if env == self.num_envs - 1 else env
                        )
                    moved = self.overrides.get((source, camera), False)
                    if self.mode == "constant":
                        moved = False
                    if (
                        self.mode == "transient_leak"
                        and any(self.overrides.values())
                        and self.moved_captures == 0
                    ):
                        moved = True
                    if modality == "rgb":
                        noise = (
                            (self.capture_count % 3 - 1) * (env + 1)
                            if self.mode == "noise"
                            else 0
                        )
                        value = np.full(
                            (self.height, self.width, 3),
                            30 + camera * 10 + 100 * moved + noise,
                            dtype=np.uint8,
                        )
                    elif modality == "depth":
                        value = np.full(
                            (self.height, self.width),
                            1.0 + float(moved),
                            dtype=np.float32,
                        )
                    elif modality == "seg":
                        value = np.full(
                            (self.height, self.width), 1 + int(moved), dtype=np.int32
                        )
                    else:
                        value = np.zeros((self.height, self.width, 3), dtype=np.float32)
                        value[..., int(moved)] = 1
                    values.append(value)
            arrays[modality] = np.stack(values)
        self.capture_count += 1
        if any(self.overrides.values()):
            self.moved_captures += 1
        return SimpleNamespace(
            arrays=arrays,
            to_metadata=lambda: {
                "render_submission_id": self.capture_count,
                "synthetic_test_fixture": True,
            },
        )


def probe(adapter: FakeAdapter, output: Path, **kwargs: Any) -> dict[str, Any]:
    return run_routing_probe(
        adapter,
        backend="embodichain",
        num_envs=adapter.num_envs,
        camera_specs=adapter.specs,
        modalities=adapter.modalities,
        width=adapter.width,
        height=adapter.height,
        output=output,
        binding=binding(),
        capture=adapter.capture,
        setter=adapter.set,
        stationary_captures=3,
        settle_captures=1,
        **kwargs,
    )


def test_targeted_routes_and_every_unaffected_modality_are_recorded(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(modalities=("rgb", "depth", "seg", "normals"))
    original = deepcopy(adapter.specs)
    result = probe(adapter, tmp_path / "proof", persist_full=True, sample_edge=3)
    assert result["status"] == "completed"
    assert result["routing_status"] == "passed"
    assert result["tested_env_ids"] == [0, 2, 4]
    assert len(result["trials"]) == 6
    assert result["capture_batches"] == 6 * 7
    assert result["captured_exposures"] == 6 * 7 * 10
    assert adapter.specs == original
    assert not any(adapter.overrides.values())
    for trial in result["trials"]:
        assert trial["restore_attempted"] and trial["restore_succeeded"]
        assert len(trial["exposures"]) == 10
        assert all(
            set(row["modalities"]) == set(adapter.modalities)
            for row in trial["exposures"]
        )
        assert [obs["phase"] for obs in trial["observations"]] == [
            "baseline_00",
            "baseline_01",
            "baseline_02",
            "moved_00",
            "moved_01",
            "restored_00",
            "restored_01",
        ]
        assert all("full_arrays" in obs for obs in trial["observations"])
        path = tmp_path / "proof" / trial["observations"][0]["sampled_arrays"]["path"]
        with np.load(path) as arrays:
            assert arrays["rgb"].shape == (10, 3, 3, 3)
    for item in result["artifacts"]:
        assert (
            hashlib.sha256((tmp_path / "proof" / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    assert (
        result["publication_eligible"] is False and result["throughput_claim"] is False
    )


@pytest.mark.parametrize("mode", ["broadcast", "swap", "depth_broadcast"])
def test_identical_replica_broadcast_or_permutation_cannot_pass(
    mode: str, tmp_path: Path
) -> None:
    adapter = FakeAdapter(mode=mode, modalities=("rgb", "depth"))
    result = probe(adapter, tmp_path / "proof")
    assert result["status"] == "completed"
    assert result["routing_status"] == "failed"
    assert any(trial["unexpected_responses"] for trial in result["trials"])
    if mode == "depth_broadcast":
        trial = result["trials"][0]
        assert trial["exposures"][2]["modalities"]["rgb"]["status"] == "passed"
        assert trial["exposures"][2]["modalities"]["depth"]["status"] == "failed"
    assert not any(adapter.overrides.values())


def test_legitimate_constant_pixels_are_inconclusive_not_stale(tmp_path: Path) -> None:
    adapter = FakeAdapter(mode="constant")
    result = probe(adapter, tmp_path / "proof")
    assert result["status"] == "completed"
    assert result["routing_status"] == "inconclusive"
    assert all(not trial["unexpected_responses"] for trial in result["trials"])


def test_stationary_noise_is_specific_to_each_exposure(tmp_path: Path) -> None:
    adapter = FakeAdapter(mode="noise")
    result = probe(adapter, tmp_path / "proof")
    assert result["routing_status"] == "passed"
    exposures = result["trials"][0]["exposures"]
    low = exposures[0]["modalities"]["rgb"]["threshold"]["value_change"]
    high = exposures[-1]["modalities"]["rgb"]["threshold"]["value_change"]
    assert high > low


def test_transient_unaffected_response_is_retained_even_after_settling(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter(mode="transient_leak")
    result = probe(adapter, tmp_path / "proof")
    assert result["routing_status"] == "failed"
    value = result["trials"][0]["exposures"][2]["modalities"]["rgb"]
    assert (
        value["moved_from_baseline"][0]["value_change"]
        > value["threshold"]["value_change"]
    )
    assert value["moved_from_baseline"][1]["value_change"] == 0


@pytest.mark.parametrize("failure", ["capture", "setter"])
def test_indexed_restore_and_prior_samples_survive_exceptions(
    failure: str, tmp_path: Path
) -> None:
    adapter = FakeAdapter()
    adapter.fail_moved_capture = failure == "capture"
    adapter.fail_after_mutation = failure == "setter"
    result = probe(adapter, tmp_path / "proof")
    assert result["status"] == "failed"
    assert result["routing_status"] == "inconclusive"
    trial = result["trials"][0]
    assert trial["restore_attempted"] and trial["restore_succeeded"]
    assert len(trial["observations"]) == 5
    assert not any(adapter.overrides.values())
    persisted = json.loads((tmp_path / "proof/routing_proof.json").read_text())
    assert persisted["trials"][0]["errors"]


def test_one_environment_reports_no_cross_environment_population(
    tmp_path: Path,
) -> None:
    result = probe(FakeAdapter(envs=1, cameras=1), tmp_path / "proof")
    assert result["cross_environment_scope"] == "not_applicable_single_environment"
    assert result["tested_env_ids"] == [0]


def test_native_public_index_is_local_to_the_selected_chunk() -> None:
    calls = []
    camera0 = SimpleNamespace(
        group_id=3,
        look_at=lambda *args, **kwargs: calls.append(("first", kwargs["env_ids"])),
    )
    camera1 = SimpleNamespace(
        group_id=4,
        look_at=lambda *args, **kwargs: calls.append(("second", kwargs["env_ids"])),
    )
    adapter = SimpleNamespace(
        num_envs=64,
        scene={"cameras": [{"name": "front"}]},
        cameras=[SimpleNamespace(parts=[(0, 32, camera0), (32, 64, camera1)])],
        torch=SimpleNamespace(
            float32=np.float32,
            tensor=lambda value, **kwargs: np.asarray(value, dtype=kwargs["dtype"]),
        ),
        _end_frame=lambda: calls.append("end"),
        _begin_frame=lambda: calls.append("publish"),
    )
    spec = {"name": "front", "eye": [1, 1, 1], "target": [0, 0, 0], "up": [0, 0, 1]}
    result = publish_indexed_camera(adapter, "embodichain", 32, 0, spec)
    assert calls == ["end", ("second", [0]), "publish"]
    assert result["sensor_index"] == 0 and result["chunk_start"] == 32
    assert adapter._pending_native_flush


def test_isaac_public_index_is_sensor_row_and_offset_is_added_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.benchmark.rendering import scenes

    monkeypatch.setattr(
        scenes, "camera_world_pose", lambda spec: ([1, 2, 3], [0, 0, 0, 1])
    )
    calls = []
    torch = SimpleNamespace(
        float32=np.float32,
        tensor=lambda value, **kwargs: np.asarray(value, dtype=kwargs["dtype"]),
        cuda=SimpleNamespace(current_stream=lambda device: "selected"),
    )
    adapter = SimpleNamespace(
        num_envs=3,
        camera_specs=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
        _scene_radius=1,
        scene={"camera_model": {"far_m": 8}},
        _replica_spacing=100,
        _env_offsets=[[0, 0, 0], [100, 0, 0], [200, 0, 0]],
        torch=torch,
        device="fake",
        wp=SimpleNamespace(
            stream_from_torch=lambda stream: stream,
            ScopedStream=lambda stream: nullcontext(),
        ),
        camera=SimpleNamespace(set_world_poses=lambda **kwargs: calls.append(kwargs)),
        sim=SimpleNamespace(
            render_context=SimpleNamespace(
                reset_transform_cadence=lambda: calls.append("cadence")
            )
        ),
    )
    result = publish_indexed_camera(
        adapter,
        "isaaclab",
        2,
        1,
        {"eye": [1, 2, 3], "target": [0, 0, 0], "up": [0, 0, 1]},
    )
    assert calls[0]["env_ids"] == [7]
    assert calls[0]["positions"].tolist() == [[201, 2, 3]]
    assert calls[0]["convention"] == "opengl" and calls[1] == "cadence"
    assert result["sensor_index"] == 7


def test_publication_interval_is_derived_per_row_not_from_percentiles() -> None:
    raw = [
        {
            "frame_index": i,
            "timestamps_s": {
                "state_publish_started_s": start,
                "state_publish_callback_returned_s": end,
            },
        }
        for i, (start, end) in enumerate(((0, 9), (10, 11), (12, 14)))
    ]
    result = publication_callback_wall(raw)
    assert [row["elapsed_s"] for row in result["raw"]] == [9, 1, 2]
    assert result["p50_s"] == 2 and result["p95_s"] == 9
    assert result["total_s"] == 12
    assert result["status"] == "complete"


def test_absent_or_invalid_publication_intervals_are_not_zero() -> None:
    result = publication_callback_wall(
        [
            {"timestamps_s": {}},
            {
                "timestamps_s": {
                    "state_publish_started_s": 2,
                    "state_publish_callback_returned_s": 1,
                }
            },
        ]
    )
    assert result["p50_s"] is None and result["total_s"] is None
    assert result["missing_record_indices"] == [0] and result[
        "invalid_record_indices"
    ] == [1]
