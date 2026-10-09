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

"""CPU-only worker lifecycle and frozen sample evidence integration tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.core.records import validate_run_result
from scripts.benchmark.rendering import formal_worker as worker
from scripts.benchmark.rendering.formal import (
    RenderCase,
    render_profile,
    view_fingerprint,
)
from scripts.benchmark.rendering.replay import load_replay, prepare_replay
from scripts.benchmark.rendering.scenes import (
    build_scene,
    load_scene,
    scene_fingerprint,
)


class DeviceArray:
    is_cuda = True

    def __init__(self, data: np.ndarray) -> None:
        self.data, self.shape, self.dtype = data, data.shape, data.dtype


class Adapter:
    """Fake renderer uses explicit state/view tokens, never a CUDA context."""

    def __init__(
        self,
        config: dict[str, Any],
        case: RenderCase,
        gpu_id: int,
        runtime: dict[str, Any],
    ) -> None:
        self.case, self.runtime = case, runtime
        self.scene = load_scene(Path(config["scene_path"]))
        names = (
            ("external_front",)
            if case.cameras_per_env == 1
            else ("external_front", "external_side", "wrist")
        )
        self.views = [
            deepcopy(view) for view in self.scene["cameras"] if view["name"] in names
        ]
        self.state = {"frame_id": 0, "time_s": 0.0}
        self.device = f"cuda:{gpu_id}"
        self.metadata = {
            "physics_enabled": False,
            "backend": "fake",
            "camera_names": list(names),
        }
        self.render_calls = 0
        self.applied_ids = []
        self.closed = False

        def reset(device: str) -> None:
            runtime["events"].append(("reset_allocator", device))

        def peak(device: str) -> int:
            runtime["events"].append(("read_allocator", device))
            return 12345

        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(
                reset_peak_memory_stats=reset, max_memory_allocated=peak
            ),
            __version__="fake",
            version=SimpleNamespace(cuda="fake"),
        )

    def render_device(self) -> dict[str, DeviceArray]:
        self.render_calls += 1
        if self.runtime.get("fail_after") == self.render_calls:
            raise RuntimeError("CUDA out of memory while allocating camera batch")
        sampler = self.runtime["samplers"][-1]
        self.runtime["events"].append(
            ("render", self.state["frame_id"], sampler.active, sampler.phase)
        )
        batch = self.case.num_envs * self.case.cameras_per_env
        shape = (batch, self.case.height, self.case.width)
        arrays = {}
        for modality in self.case.modalities:
            if modality == "rgb":
                data = np.empty(shape + (3,), dtype=np.uint8)
                for index in range(batch):
                    eye_x = (
                        0
                        if self.runtime.get("stale")
                        else self.views[index % self.case.cameras_per_env]["eye"][0]
                    )
                    value = (self.state["frame_id"] + index + round(eye_x * 20)) % 256
                    data[index].fill(value)
            elif modality == "depth":
                eye_x = 0 if self.runtime.get("stale") else self.views[0]["eye"][0]
                data = np.full(shape + (1,), 2 + eye_x / 10, dtype=np.float32)
            elif modality == "seg":
                data = np.ones(shape, dtype=np.int32)
            else:
                data = np.zeros(shape + (3,), dtype=np.float32)
                data[..., 2] = 1
            arrays[modality] = DeviceArray(data)
        return arrays

    def synchronize(self) -> None:
        self.runtime["events"].append(("synchronize", self.device))

    def to_host(self, arrays: dict[str, DeviceArray]) -> dict[str, np.ndarray]:
        return {name: array.data.copy() for name, array in arrays.items()}

    def apply_state(self, state: dict[str, Any]) -> None:
        self.state = deepcopy(state)
        self.applied_ids.append(state["frame_id"])
        self.views = [
            deepcopy(state["cameras"][name]) for name in self.metadata["camera_names"]
        ]

    def set_cameras(self, views: list[dict[str, Any]]) -> None:
        assert len(views) == self.case.cameras_per_env
        self.views = deepcopy(views)

    def refresh_metadata(self) -> None:
        self.runtime["events"].append(("refresh_metadata",))

    def close(self) -> None:
        self.closed = True
        if self.runtime.get("cleanup_failure"):
            raise RuntimeError("cleanup failed")


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    data: dict[str, Any] = {"adapters": [], "samplers": [], "events": []}

    class Sampler:
        def __init__(self, gpu_id: int) -> None:
            self.gpu_id, self.active, self.phase = gpu_id, False, "setup"
            data["samplers"].append(self)

        def __enter__(self) -> Sampler:
            self.active = True
            return self

        def __exit__(self, *args: Any) -> None:
            self.active = False

        def mark_phase(self, phase: str) -> None:
            self.phase = phase
            data["events"].append(("phase", phase))

        def summary(self) -> dict[str, Any]:
            return {
                "gpu": {
                    "index": self.gpu_id,
                    "uuid": f"GPU-selected-{self.gpu_id}",
                    "name": "fake GPU",
                    "driver": "fake",
                },
                "devices": [{"uuid": "GPU-wrong-device-zero"}],
                "metrics": {
                    "gpu_process_peak_bytes": 9000,
                    "gpu_process_steady_bytes": 7000,
                    "gpu_device_peak_bytes": 15000,
                    "cpu_peak_rss_bytes": 20000,
                },
                "metric_reasons": {},
                "running": self.active,
            }

    def create(
        backend: str, config: dict[str, Any], case: RenderCase, gpu_id: int
    ) -> Adapter:
        if data.get("construction_oom"):
            raise MemoryError("camera construction out of memory")
        adapter = Adapter(config, case, gpu_id, data)
        data["adapters"].append(adapter)
        return adapter

    monkeypatch.setattr(worker, "ResourceSampler", Sampler)
    monkeypatch.setattr(worker, "_create_adapter", create)
    monkeypatch.setattr(
        worker, "software_snapshot", lambda *args: {"benchmark_commit": "test"}
    )
    monkeypatch.setattr(worker, "host_snapshot", lambda: {"platform": "CPU test"})
    monkeypatch.setattr(worker, "package_version", lambda name: "fake")
    monkeypatch.setattr(
        worker,
        "_videos",
        lambda *args: {
            "status": "unsupported",
            "reason": "encoder disabled in CPU tests",
        },
    )
    return data


def config_file(
    tmp_path: Path, case: RenderCase, *, measured: int = 3, warmup: int = 1
) -> Path:
    scene_path = build_scene("calibration", tmp_path / "scene")
    scene = load_scene(scene_path)
    scene["scene_id"] = case.scene_id
    if case.cameras_per_env == 3:
        scene["cameras"].append(
            {
                "name": "wrist",
                "eye": [0.2, 0.1, 0.4],
                "target": [0.3, 0.1, 0.1],
                "up": [0.0, 1.0, 0.0],
            }
        )
    write_json(scene_path, scene)
    replay_path = None
    replay = None
    if case.temporal_mode == "moving" and not case.deformation:
        replay_path = prepare_replay(scene_path, tmp_path / "replay.json")
        replay = load_replay(replay_path, scene_path)
    profile = render_profile(case.profile)
    fingerprint = scene_fingerprint(scene)
    return write_json(
        tmp_path / "worker-config.json",
        {
            "case": case.to_dict(),
            "scene_path": str(scene_path),
            "replay_path": str(replay_path) if replay_path is not None else None,
            "profile": profile,
            "warmup_frames": warmup,
            "measured_frames": measured,
            "scene_sha256": fingerprint,
            "view_sha256": (
                view_fingerprint(case, scene, replay)
                if not case.deformation
                else "missing_deformation"
            ),
            "profile_sha256": stable_hash(profile),
            "isaaclab_root": None,
            "approved_scene_receipt": {
                "status": "approved",
                "scope": "scene_layout_and_camera_scheme",
                "scenes": {case.scene_id: fingerprint},
            },
        },
    )


def frame_records(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((output / "frames.json").read_text())
    return manifest, [
        json.loads((output / frame["path"]).read_text()) for frame in manifest["frames"]
    ]


def test_performance_worker_preserves_boundary_samples_and_selected_gpu(
    tmp_path: Path, runtime: dict[str, Any]
) -> None:
    case = RenderCase(
        "R-05",
        "franka_workstation",
        width=8,
        height=6,
        num_envs=5,
        cameras_per_env=3,
        modalities=("rgb", "depth", "seg"),
        delivery="host_copy",
    )
    config = config_file(tmp_path, case)
    output = tmp_path / "run"
    result = worker.run_worker(config, output, "embodichain", 3, repeat=2)
    assert result["status"] == "completed", result.get("error")
    validate_run_result(result)
    assert result["quality_status"] == "not_qualified"
    assert result["hardware_id"] == "GPU-selected-3"
    assert result["metrics"]["boundary"] == "render_to_host_copy"
    assert result["metrics"]["torch_peak_allocated_bytes"] == 12345
    assert result["counts"]["exposure_count"] == 3 * 5 * 3
    assert result["counts"]["d2h_calls"] == 3 * 3
    assert result["counts"]["host_copy_calls"] == 3 * 3
    assert ("reset_allocator", "cuda:3") in runtime["events"]
    assert runtime["adapters"][0].closed
    assert not runtime["samplers"][0].active
    renders = [event for event in runtime["events"] if event[0] == "render"]
    assert renders[-1][2] is False  # Representative sample is outside resource timing.
    manifest, frames = frame_records(output)
    assert manifest["representative_env_ids"] == [0, 2, 4]
    assert manifest["captured_exposures"] == 15
    assert manifest["saved_exposures"] == 9
    assert {record["metadata"]["camera_name"] for record in frames} == {
        "external_front",
        "external_side",
        "wrist",
    }
    assert [record["metadata"]["batch_index"] for record in frames] == [
        0,
        1,
        2,
        6,
        7,
        8,
        12,
        13,
        14,
    ]
    for entry, record in zip(manifest["frames"], frames, strict=True):
        path = output / entry["arrays_path"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == record["arrays"]["sha256"]
        )
        assert record["metadata"]["rgb_encoding"] == "srgb_uint8"
        assert (
            record["metadata"]["segmentation_semantics"]
            == "canonical_instance_ids_zero_background"
        )
        assert record["metadata"]["camera"]["convention"] == "opencv"
        with np.load(path) as arrays:
            assert arrays["rgb"].shape == (6, 8, 3)
            assert arrays["depth"].shape == arrays["seg"].shape == (6, 8)


def test_dynamic_performance_sample_keeps_last_wrapped_replay_frame(
    tmp_path: Path, runtime: dict[str, Any]
) -> None:
    case = RenderCase("R-03", "calibration", width=8, height=6, temporal_mode="moving")
    config = config_file(tmp_path, case, measured=102, warmup=2)
    output = tmp_path / "run"
    result = worker.run_worker(config, output, "embodichain", 1)
    assert result["status"] == "completed", result.get("error")
    assert result["metrics"]["boundary"] == "state_publish_to_device_ready"
    assert result["sample_capture"]["replay_frame_id"] == 1
    _, frames = frame_records(output)
    assert frames[0]["metadata"]["frame_id"] == 1
    assert frames[0]["metadata"]["time_s"] == pytest.approx(1 / 30)
    assert runtime["adapters"][0].applied_ids[-1] == 1
    assert result["counts"]["state_publish_calls"] == 102


@pytest.mark.parametrize("modalities", [("rgb",), ("rgb", "depth", "seg")])
def test_dynamic_quality_sequence_is_outside_timing_and_resource_window(
    tmp_path: Path, runtime: dict[str, Any], modalities: tuple[str, ...]
) -> None:
    case = RenderCase(
        "R-03",
        "calibration",
        width=8,
        height=6,
        temporal_mode="moving",
        modalities=modalities,
    )
    config = config_file(tmp_path, case, measured=3, warmup=2)
    data = json.loads(config.read_text())
    data["post_measurement_quality_capture"] = "frozen_sequence_100"
    write_json(config, data)
    output = tmp_path / "run"
    result = worker.run_worker(config, output, "embodichain", 1)
    assert result["status"] == "completed", result.get("error")
    assert result["counts"]["batch_steps"] == 3
    assert result["counts"]["state_publish_calls"] == 3
    manifest, frames = frame_records(output)
    assert manifest["saved_exposures"] == 100
    assert [frame["metadata"]["frame_id"] for frame in frames] == list(range(100))
    assert result["quality_capture"]["outside_resource_window"] is True
    assert result["quality_capture"]["throughput_claim"] is False
    outside = [
        event
        for event in runtime["events"]
        if event[0] == "render" and event[2] is False
    ]
    assert len(outside) == 102  # quality warmup plus every sequence frame
    assert [event[1] for event in outside[2:]] == list(range(100))
    assert runtime["adapters"][0].applied_ids[-1] == 2


def test_r01_captures_twenty_full_batches_but_saves_representative_environments(
    tmp_path: Path, runtime: dict[str, Any]
) -> None:
    case = RenderCase(
        "R-01",
        "calibration",
        width=8,
        height=6,
        num_envs=64,
        modalities=("rgb", "depth", "seg"),
        delivery="host_ready",
        view_count=20,
    )
    config = config_file(tmp_path, case, measured=2)
    output = tmp_path / "run"
    result = worker.run_worker(config, output, "embodichain", 0)
    assert result["status"] == "completed", result.get("error")
    assert result["metrics"]["batch_steps_per_s"] is None
    assert result["quality_capture"]["throughput_claim"] is False
    assert result["quality_capture"]["capture_batches"] == 20
    assert result["quality_capture"]["captured_exposures"] == 1280
    manifest, frames = frame_records(output)
    assert manifest["representative_env_ids"] == [0, 32, 63]
    assert len(frames) == 60
    assert {record["metadata"]["view_id"] for record in frames} == set(range(20))
    assert len({record["metadata"]["camera_name"] for record in frames}) == 20
    assert result["quality_capture"]["warmup_batches"] == 20


def test_r02_persists_all_hundred_states_and_three_camera_streams(
    tmp_path: Path, runtime: dict[str, Any]
) -> None:
    case = RenderCase(
        "R-02",
        "franka_workstation",
        width=8,
        height=6,
        cameras_per_env=3,
        modalities=("rgb", "depth", "seg"),
        temporal_mode="moving",
        delivery="host_ready",
    )
    config = config_file(tmp_path, case, measured=2)
    output = tmp_path / "run"
    result = worker.run_worker(config, output, "isaaclab", 2)
    assert result["status"] == "completed", result.get("error")
    manifest, frames = frame_records(output)
    assert result["quality_capture"]["capture_batches"] == 100
    assert manifest["captured_exposures"] == len(frames) == 300
    assert [record["metadata"]["frame_id"] for record in frames[::3]] == list(
        range(100)
    )
    assert frames[-1]["metadata"]["time_s"] == pytest.approx(99 / 30)
    assert frames[-1]["metadata"]["physics_steps"] == 0
    assert frames[-1]["metadata"]["moving_instance_ids"]
    assert runtime["adapters"][0].applied_ids[-1] == 99
    assert result["video"]["status"] == "unsupported"
    assert result["metrics"]["camera_frames_per_s"] is None


def test_missing_deformation_exits_without_site_packages_or_gpu_imports(
    tmp_path: Path,
) -> None:
    case = RenderCase(
        "R-02",
        "franka_workstation",
        width=8,
        height=6,
        temporal_mode="moving",
        deformation=True,
    )
    config = config_file(tmp_path, case)
    output = tmp_path / "run"
    command = [
        sys.executable,
        "-S",
        "-m",
        "scripts.benchmark.rendering.formal_worker",
        "--config",
        str(config),
        "--output",
        str(output),
        "--backend",
        "embodichain",
        "--gpu-id",
        "0",
    ]
    process = subprocess.run(
        command, cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True
    )
    assert process.returncode == 1, process.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["failure_reason_code"] == "unsupported"
    assert result["quality_status"] == "unsupported"
    assert result["execution_attempted"] is False
    assert "resources" not in result
    assert result["case_id"] == case.case_id


@pytest.mark.parametrize(
    "field", ["scene_sha256", "view_sha256", "profile_sha256", "approved_scene_receipt"]
)
def test_source_changes_fail_before_backend_construction(
    tmp_path: Path, runtime: dict[str, Any], field: str
) -> None:
    case = RenderCase("R-03", "calibration", width=8, height=6)
    path = config_file(tmp_path, case)
    config = json.loads(path.read_text())
    config[field] = {} if field == "approved_scene_receipt" else "wrong"
    write_json(path, config)
    result = worker.run_worker(path, tmp_path / "run", "embodichain", 0)
    assert result["failure_reason_code"] == "source_validation_failed"
    assert result["execution_attempted"] is False
    assert not runtime["adapters"]
    assert not runtime["samplers"]


def test_stale_probe_preserves_failed_evidence_without_reducing_workload(
    tmp_path: Path, runtime: dict[str, Any]
) -> None:
    runtime["stale"] = True
    case = RenderCase("R-04", "franka_workstation", width=8, height=6, num_envs=16)
    path = config_file(tmp_path, case)
    output = tmp_path / "run"
    result = worker.run_worker(path, output, "embodichain", 0)
    assert result["failure_reason_code"] == "freshness_probe_failed"
    assert result["config"]["num_envs"] == 16
    assert result["validation"]["passed"] is False
    assert (output / "freshness_probe.npz").is_file()
    assert runtime["adapters"][0].closed
    assert "batch_steps_per_s" not in result["metrics"]


@pytest.mark.parametrize("construction", [True, False])
def test_oom_is_distinct_and_sampler_and_adapter_are_closed(
    tmp_path: Path, runtime: dict[str, Any], construction: bool
) -> None:
    if construction:
        runtime["construction_oom"] = True
    else:
        runtime["fail_after"] = (
            6  # Four probe captures + one warmup precede measurement.
        )
    path = config_file(
        tmp_path, RenderCase("R-04", "franka_workstation", width=8, height=6)
    )
    result = worker.run_worker(path, tmp_path / "run", "embodichain", 4)
    assert result["status"] == "failed"
    assert result["failure_reason_code"] == "oom"
    assert result["hardware_id"] == "GPU-selected-4"
    assert not runtime["samplers"][0].active
    assert all(adapter.closed for adapter in runtime["adapters"])
    assert result["metrics"]["gpu_process_peak_bytes"] == 9000


def test_persistence_failure_keeps_completed_measurements_and_cleans_up(
    tmp_path: Path, runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> None:
        raise OSError("sample storage full")

    monkeypatch.setattr(worker, "_save_batch", fail)
    path = config_file(tmp_path, RenderCase("R-03", "calibration", width=8, height=6))
    result = worker.run_worker(path, tmp_path / "run", "embodichain", 0)
    assert result["failure_reason_code"] == "artifact_persistence_failed"
    assert result["metrics"]["batch_steps_per_s"] > 0
    assert (tmp_path / "run" / "capture.json").is_file()
    assert runtime["adapters"][0].closed


def test_cleanup_failure_does_not_erase_existing_artifacts(
    tmp_path: Path, runtime: dict[str, Any]
) -> None:
    runtime["cleanup_failure"] = True
    path = config_file(tmp_path, RenderCase("R-03", "calibration", width=8, height=6))
    result = worker.run_worker(path, tmp_path / "run", "embodichain", 0)
    assert result["status"] == "failed"
    assert result["failure_reason_code"] == "cleanup_failed"
    assert "cleanup failed" in result["cleanup_error"]
    assert (tmp_path / "run" / "frames.json").is_file()


@pytest.mark.parametrize("capture_fails", [False, True])
def test_native_process_exit_during_close_preserves_result_checkpoint(
    tmp_path: Path, capture_fails: bool
) -> None:
    config = config_file(tmp_path, RenderCase("R-03", "calibration", width=8, height=6))
    output = tmp_path / "run"
    script = """
import os
import runpy
import sys
from pathlib import Path

fixture = runpy.run_path(sys.argv[1])
patch = fixture['pytest'].MonkeyPatch()
runtime = fixture['runtime'].__wrapped__(patch)
fixture['Adapter'].close = lambda self: os._exit(0)
if sys.argv[4] == 'fail':
    runtime['fail_after'] = 6
fixture['worker'].run_worker(Path(sys.argv[2]), Path(sys.argv[3]), 'isaaclab', 2)
raise AssertionError('Native close should have terminated this process')
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve()),
            str(config),
            str(output),
            "fail" if capture_fails else "complete",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert process.returncode == 0, process.stderr
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == ("failed" if capture_fails else "completed")
    assert result["cleanup_status"] == "pending_native_close"
    assert result["resources"]["gpu"]["uuid"] == "GPU-selected-2"
    assert (output / "frames.json").is_file()
    if capture_fails:
        assert result["failure_reason_code"] == "oom"
        assert "out of memory" in result["error"]
        assert (output / "failure.json").is_file()
    else:
        assert result["metrics"]["batch_steps_per_s"] > 0
        assert (
            result["frames_manifest"]["sha256"]
            == hashlib.sha256((output / "frames.json").read_bytes()).hexdigest()
        )


def test_r01_view_aliases_do_not_republish_obsolete_camera_names(
    tmp_path: Path, runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class StrictAdapter(Adapter):
        def set_cameras(self, views: list[dict[str, Any]]) -> None:
            super().set_cameras(views)
            self.metadata["camera_names"] = [view["name"] for view in views]

        def apply_state(self, state: dict[str, Any]) -> None:
            missing = set(self.metadata["camera_names"]) - set(state["cameras"])
            if missing:
                raise ValueError(
                    f"Replay omits selected camera views: {sorted(missing)}"
                )
            super().apply_state(state)

    def create(
        backend: str, config: dict[str, Any], case: RenderCase, gpu_id: int
    ) -> Adapter:
        adapter = StrictAdapter(config, case, gpu_id, runtime)
        runtime["adapters"].append(adapter)
        return adapter

    monkeypatch.setattr(worker, "_create_adapter", create)
    case = RenderCase(
        "R-01", "calibration", width=8, height=6, view_count=20, delivery="host_ready"
    )
    config = config_file(tmp_path, case)
    result = worker.run_worker(config, tmp_path / "run", "isaaclab", 0)
    assert result["status"] == "completed", result.get("error")
    assert result["quality_capture"]["capture_batches"] == 20
    assert runtime["adapters"][0].metadata["camera_names"] == ["r01_view_19"]
    assert runtime["adapters"][0].applied_ids == [0, 0, 0]


@pytest.fixture
def diagnostic_calls(
    tmp_path: Path, runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    from scripts.benchmark.rendering import diagnostics

    calls: dict[str, Any] = {}

    def outside() -> None:
        assert not runtime["samplers"][0].active
        assert (tmp_path / "run/frames.json").is_file()
        saved = json.loads((tmp_path / "run/result.json").read_text())
        assert saved["status"] == "completed"
        assert saved["diagnostics"]["status"] == "pending"
        calls.setdefault(
            "primary_metrics",
            json.loads((tmp_path / "run/capture.json").read_text())["metrics"],
        )

    def trace(adapter: Adapter, **kwargs: Any) -> dict[str, Any]:
        outside()
        calls["trace"] = kwargs
        path = write_json(kwargs["output"] / "cpu_ranges.json", {"traceEvents": []})
        if calls.get("fail") == "trace":
            raise ImportError("trace profiler unavailable")
        if kwargs["prepare"]:
            for index in range(kwargs["iterations"]):
                kwargs["prepare"](index)
        adapter.torch.cuda.max_memory_allocated = lambda device: 999999999
        payload = {
            "status": "completed",
            "declared_observation_transfers": {"d2h_calls": 6},
            "observed_transfers": {"actual_native_transfer_count": None},
            "profiler": {
                "activities": ["CPU"],
                "reasons": ["cuda_activity_unavailable"],
            },
            "artifacts": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            ],
            "measurement": {"metrics": {"batch_steps_per_s": 999999}},
        }
        write_json(kwargs["output"] / "delivery_diagnostics.json", payload)
        return payload

    def overhead(adapter: Adapter, **kwargs: Any) -> dict[str, Any]:
        outside()
        calls["overhead"] = kwargs
        if calls.get("fail") == "overhead":
            write_json(kwargs["output"], {"partial": True})
            raise RuntimeError("diagnostic backend unavailable")
        if kwargs["prepare"]:
            kwargs["prepare"](kwargs["iterations"] - 1)
        payload = {
            "paired_windows": 3,
            "windows": [{"repeat": index // 3} for index in range(9)],
            "overhead_ratio": 8.0,
            "overhead_fraction": 7.0,
            "equivalent_policy": True,
            "comparison_reasons": [],
        }
        write_json(kwargs["output"], payload)
        return payload

    monkeypatch.setattr(diagnostics, "trace_delivery", trace)
    monkeypatch.setattr(diagnostics, "framework_overhead", overhead)
    return calls


def test_r09_diagnostics_preserve_primary_metrics_and_last_replay_state(
    tmp_path: Path, runtime: dict[str, Any], diagnostic_calls: dict[str, Any]
) -> None:
    case = RenderCase(
        "R-09",
        "franka_workstation",
        width=8,
        height=6,
        delivery="duplicate_d2h",
        temporal_mode="moving",
        diagnostic=True,
    )
    config = config_file(tmp_path, case, measured=105, warmup=2)
    result = worker.run_worker(config, tmp_path / "run", "embodichain", 0)
    assert result["status"] == "completed", result.get("error")
    trace, overhead = diagnostic_calls["trace"], diagnostic_calls["overhead"]
    assert (trace["iterations"], trace["warmup"]) == (3, 0)
    assert (overhead["iterations"], overhead["repeats"], overhead["warmup"]) == (
        100,
        3,
        2,
    )
    assert trace["delivery"] == overhead["delivery"] == "duplicate_d2h"
    assert trace["diagnostic"] is overhead["diagnostic"] is True
    assert trace["prepare"] is overhead["prepare"]
    assert result["metrics"] == diagnostic_calls["primary_metrics"]
    assert result["metrics"]["torch_peak_allocated_bytes"] == 12345
    assert result["metrics"]["native_render_s"] is None
    assert "unqualified" in result["metric_reasons"]["native_render_s"]
    assert "overhead_ratio" not in result["metrics"]
    assert result["diagnostics"]["framework_overhead"]["sample_windows"] == 9
    assert (
        result["diagnostics"]["trace"]["declared_observation_transfers"]["d2h_calls"]
        == 6
    )
    assert result["diagnostics"]["native_timer"]["activation_requested"] is False
    assert result["sample_capture"]["replay_frame_id"] == 4
    assert runtime["adapters"][0].state["frame_id"] == 4
    assert "diagnostics/framework_overhead.json" in result["evidence"]
    assert "diagnostics/delivery_trace/cpu_ranges.json" in result["evidence"]
    assert result["software"]["packages"]["nvidia-ml-py"] == "fake"
    assert result["software"]["packages"]["pynvml"] == "fake"


@pytest.mark.parametrize("unavailable", ["trace", "overhead"])
def test_optional_diagnostic_failure_retains_success_and_partial_artifacts(
    tmp_path: Path,
    runtime: dict[str, Any],
    diagnostic_calls: dict[str, Any],
    unavailable: str,
) -> None:
    diagnostic_calls["fail"] = unavailable
    config = config_file(
        tmp_path,
        RenderCase("R-09", "franka_workstation", width=8, height=6, diagnostic=True),
    )
    result = worker.run_worker(config, tmp_path / "run", "embodichain", 0)
    assert result["status"] == "completed", result.get("error")
    assert result["metrics"] == diagnostic_calls["primary_metrics"]
    assert result["diagnostics"]["status"] == "partial"
    key = "trace" if unavailable == "trace" else "framework_overhead"
    assert key in result["diagnostics"]["missing_reasons"]
    assert (tmp_path / "run/diagnostics/summary.json").is_file()
    assert "diagnostics/delivery_trace/cpu_ranges.json" in result["evidence"]
    assert runtime["adapters"][0].closed


@pytest.mark.parametrize("experiment_id, enabled", [("R-09", False), ("R-03", True)])
def test_diagnostic_followups_are_only_for_enabled_r09(
    tmp_path: Path,
    runtime: dict[str, Any],
    diagnostic_calls: dict[str, Any],
    experiment_id: str,
    enabled: bool,
) -> None:
    config = config_file(
        tmp_path,
        RenderCase(
            experiment_id, "franka_workstation", width=8, height=6, diagnostic=enabled
        ),
    )
    result = worker.run_worker(config, tmp_path / "run", "embodichain", 0)
    assert result["status"] == "completed", result.get("error")
    assert not diagnostic_calls
    assert "diagnostics" not in result


def _capacity_log(output: Path) -> bytes:
    """Representative native errors, including ANSI/NUL output and VUID echo."""
    output.mkdir(parents=True, exist_ok=True)
    payload = (
        "\x1b[31m[DexSim Error] Invalid Vulkan multiview count: requested 64, supported 32\x00\x1b[0m"
        "[DexSim Error] Invalid Vulkan multiview count: requested 256, supported 32\n"
        "Validation Error: [ VUID-vkCreateSampler-maxSamplerAllocationCount-04110 ] | MessageID = 0xb408bc0b\n"
        "vkCreateSampler(): Number of currently valid sampler objects (4000) is not less than the maximum allowed (4000).\n"
        "The Vulkan spec states: see https://example.invalid/#VUID-vkCreateSampler-maxSamplerAllocationCount-04110\n"
    ).encode()
    (output / "worker.log").write_bytes(payload)
    return payload


def test_native_capacity_log_parsing_preserves_exact_limits_and_prefix_identity(
    tmp_path: Path,
) -> None:
    payload = _capacity_log(tmp_path)

    evidence = worker._renderer_capacity_evidence(tmp_path)

    assert evidence["status"] == "failed"
    assert evidence["counts"] == {
        "vulkan_multiview_limit": 2,
        "vulkan_sampler_allocation_limit": 1,
    }
    assert evidence["events"][0]["requested_views"] == 64
    assert evidence["events"][1]["requested_views"] == 256
    assert all(event["supported_views"] == 32 for event in evidence["events"][:2])
    sampler = evidence["events"][2]
    assert sampler["current_sampler_count"] == sampler["maximum_sampler_count"] == 4000
    assert sampler["line"] == 2 and sampler["detail_line"] == 3
    assert evidence["scanned_bytes"] == len(payload)
    assert evidence["scanned_prefix_sha256"] == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / "worker.log").read_bytes() == payload


def test_renderer_log_warnings_and_unavailable_logs_do_not_invent_capacity_failures(
    tmp_path: Path,
) -> None:
    missing = worker._renderer_capacity_evidence(tmp_path)
    assert missing["status"] == "unavailable" and not missing["events"]
    (tmp_path / "worker.log").write_text(
        "Warning: noise was high; sampler count may matter.\n"
        "https://example.invalid/VUID-vkCreateSampler-maxSamplerAllocationCount-04110\n"
        "Informational requested 64, supported 32\n"
        "Invalid Vulkan multiview count: requested 16, supported 32\n"
        "vkCreateSampler(): Number of currently valid sampler objects (3999) is not less than the maximum allowed (4000).\n"
    )
    result = worker._renderer_capacity_evidence(tmp_path)
    assert result["status"] == "no_known_capacity_errors"
    assert result["reason_codes"] == []


def test_completed_capture_with_explicit_capacity_errors_fails_without_erasing_timings(
    tmp_path: Path, runtime: dict[str, Any]
) -> None:
    config = config_file(tmp_path, RenderCase("R-03", "calibration", width=8, height=6))
    output = tmp_path / "run"
    payload = _capacity_log(output)

    result = worker.run_worker(config, output, "embodichain", 0)

    assert result["status"] == "failed"
    assert result["failure_reason_code"] == "renderer_capacity_limit"
    assert result["failure_phase"] == "renderer_validation"
    assert result["metrics"]["batch_steps_per_s"] > 0
    assert result["sample_capture"]["outside_measured_window"] is True
    assert (output / "capture.json").is_file()
    assert (output / "frames.json").is_file()
    assert "renderer_validation.json" in result["evidence"]
    assert (
        json.loads((output / "failure.json").read_text())["failure_reason_code"]
        == "renderer_capacity_limit"
    )
    assert (output / "worker.log").read_bytes() == payload
    assert runtime["adapters"][0].closed


def test_native_capacity_evidence_does_not_replace_earlier_freshness_failure(
    tmp_path: Path, runtime: dict[str, Any]
) -> None:
    runtime["stale"] = True
    config = config_file(
        tmp_path, RenderCase("R-04", "franka_workstation", width=8, height=6)
    )
    output = tmp_path / "run"
    _capacity_log(output)

    result = worker.run_worker(config, output, "embodichain", 0)

    assert result["failure_reason_code"] == "freshness_probe_failed"
    assert result["failure_phase"] == "freshness_probe"
    assert result["error"].startswith("_FreshnessFailure:")
    assert (
        result["renderer_validation"]["counts"]["vulkan_sampler_allocation_limit"] == 1
    )
    original_failure = json.loads((output / "failure.json").read_text())
    assert original_failure["failure_reason_code"] == "freshness_probe_failed"
    assert original_failure["error"] == result["error"]
    assert original_failure["traceback"] == result["traceback"]


def test_capacity_evidence_is_checkpointed_before_native_close_can_terminate(
    tmp_path: Path, runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = config_file(tmp_path, RenderCase("R-03", "calibration", width=8, height=6))
    output = tmp_path / "run"
    _capacity_log(output)

    def close(adapter: Adapter) -> None:
        checkpoint = json.loads((output / "result.json").read_text())
        assert checkpoint["cleanup_status"] == "pending_native_close"
        assert checkpoint["failure_reason_code"] == "renderer_capacity_limit"
        assert checkpoint["renderer_validation"]["status"] == "failed"
        assert checkpoint["metrics"]["batch_steps_per_s"] > 0
        assert (output / "renderer_validation.json").is_file()
        raise SystemExit(0)

    monkeypatch.setattr(Adapter, "close", close)
    with pytest.raises(SystemExit):
        worker.run_worker(config, output, "isaaclab", 0)
    assert (
        json.loads((output / "result.json").read_text())["failure_reason_code"]
        == "renderer_capacity_limit"
    )
