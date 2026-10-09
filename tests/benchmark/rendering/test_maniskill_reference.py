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

"""CPU contracts for separate ManiSkill references; never initialize a renderer."""

from __future__ import annotations

from contextlib import nullcontext
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
import torch

from scripts.benchmark.core.artifacts import read_json, stable_hash, write_json
from scripts.benchmark.rendering import maniskill_reference as runner
from scripts.benchmark.rendering import maniskill_reference_worker as worker
from scripts.benchmark.rendering.backends import scene_maniskill as backend
from scripts.benchmark.rendering.formal import view_fingerprint
from scripts.benchmark.rendering.scenes import camera_world_pose


@pytest.fixture
def approved(tmp_path: Path) -> Path:
    root = tmp_path / "approved"
    hashes = {}
    for name in ("calibration", "workshop", "franka_workstation"):
        scene = {
            "schema_version": 1,
            "scene_id": name,
            "units": "metres",
            "up_axis": "Z",
            "physics_steps": 0,
            "camera_model": {"horizontal_fov_deg": 55.0, "near_m": 0.01, "far_m": 8.0},
            "objects": [
                {
                    "uid": "cube",
                    "instance_id": 1,
                    "geometry": {"kind": "box", "size": [0.1, 0.1, 0.1]},
                    "world_from_object": np.eye(4).tolist(),
                    "material": {
                        "base_color_linear": [0.2, 0.3, 0.4],
                        "roughness": 0.7,
                        "metallic": 0.0,
                    },
                }
            ],
            "cameras": [
                {
                    "name": "external_front",
                    "eye": [2.0, -2.0, 1.0],
                    "target": [0.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                }
            ],
            "lighting": {"direction": [-0.4, 0.3, -1.0], "color": [1.0, 1.0, 1.0]},
        }
        write_json(root / name / "scene.json", scene)
        hashes[name] = stable_hash(scene)
    write_json(
        root / "approval.json",
        {
            "status": "approved",
            "scope": "scene_layout_and_camera_scheme",
            "scenes": hashes,
        },
    )
    return root


def _config(
    approved: Path, tmp_path: Path, num_envs: int = 16, delivery: str = "device_ready"
) -> Path:
    case = next(
        case
        for case in runner.reference_cases(delivery=delivery)
        if case["num_envs"] == num_envs
    )
    scene_path = approved / "franka_workstation/scene.json"
    scene = read_json(scene_path)
    profile = backend.raster_profile()
    return write_json(
        tmp_path / "worker.json",
        {
            "case": case,
            "scene_path": str(scene_path),
            "profile": profile,
            "config_sha256": stable_hash(case),
            "scene_sha256": stable_hash(scene),
            "view_sha256": view_fingerprint(SimpleNamespace(**case), scene),
            "profile_sha256": stable_hash(profile),
            "approved_scene_receipt": read_json(approved / "approval.json"),
            "warmup_frames": 2,
            "measured_frames": 4,
            "physx_gpu_library": str(tmp_path / "libPhysXGpu_64.so"),
        },
    )


class _Array:
    is_cuda = True

    def __init__(self, data: np.ndarray) -> None:
        self.data, self.shape, self.dtype = data, data.shape, data.dtype


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "events": [],
        "stale_env": None,
        "fail_capture": False,
        "close_failure": False,
    }

    class Sampler:
        def __init__(self, gpu_id: int) -> None:
            self.gpu_id, self.active, self.phase = gpu_id, False, "setup"
            state["sampler"] = self

        def __enter__(self) -> Sampler:
            self.active = True
            return self

        def __exit__(self, *args: Any) -> None:
            self.active = False

        def mark_phase(self, name: str) -> None:
            self.phase = name

        def summary(self) -> dict[str, Any]:
            return {
                "gpu": {"uuid": f"fixture-gpu-{self.gpu_id}"},
                "metrics": {"gpu_process_peak_bytes": None},
                "metric_reasons": {
                    "gpu_process_peak_bytes": "CPU fixture has no GPU memory measurement"
                },
            }

    class Adapter:
        def __init__(
            self, scene_path: Path, *, num_envs: int, gpu_id: int, **kwargs: Any
        ) -> None:
            self.scene = read_json(scene_path)
            self.views = deepcopy(self.scene["cameras"])
            self.num_envs, self.device = num_envs, f"cuda:{gpu_id}"
            self.metadata = {
                "cameras": [{"env_id": index} for index in range(num_envs)]
            }
            if state.get("torch_uuid"):
                self.metadata["torch_device"] = {"uuid": state["torch_uuid"]}
            self.torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    reset_peak_memory_stats=lambda _: None,
                    max_memory_allocated=lambda _: 123,
                )
            )
            self.closed = False
            state["adapter"] = self

        def set_cameras(self, views: list[dict[str, Any]]) -> None:
            self.views = deepcopy(views)

        def apply_state(self, initial: dict[str, Any]) -> None:
            self.set_cameras([initial["cameras"]["external_front"]])

        def render_device(self) -> dict[str, _Array]:
            sampler = state["sampler"]
            state["events"].append(("render", sampler.active, sampler.phase))
            if state["fail_capture"] and sampler.phase == "measurement":
                raise RuntimeError("CUDA out of memory")
            data = np.empty((self.num_envs, 128, 128, 3), dtype=np.uint8)
            for index in range(self.num_envs):
                eye = 2.0 if index == state["stale_env"] else self.views[0]["eye"][0]
                data[index].fill(round(eye * 30) + index)
            return {"rgb": _Array(data)}

        def synchronize(self) -> None:
            state["events"].append(("synchronize", self.device))

        def to_host(self, arrays: dict[str, _Array]) -> dict[str, np.ndarray]:
            state["events"].append(("d2h", state["sampler"].phase))
            return {name: array.data.copy() for name, array in arrays.items()}

        def refresh_metadata(self) -> None:
            assert not state["sampler"].active

        def close(self) -> None:
            self.closed = True
            if state["close_failure"]:
                raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(worker, "ResourceSampler", Sampler)
    monkeypatch.setattr(worker, "ManiSkillSceneReference", Adapter)
    monkeypatch.setattr(
        worker, "runtime_evidence", lambda _: {"fixture": "no GPU library loaded"}
    )
    return state


def test_offline_list_is_standard_library_only_and_nonmutating(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-S",
        "-m",
        "scripts.benchmark",
        "maniskill-reference",
        "--dry-run",
        "--smoke",
        "--output",
        str(tmp_path / "absent"),
    ]
    process = subprocess.run(
        command, cwd=runner.ROOT, check=True, capture_output=True, text=True
    )
    plan = json.loads(process.stdout)
    assert plan["filesystem_mutations"] is False
    assert plan["repeats"] == 1
    assert plan["measured_frames"] == 4
    assert {
        (item["case"]["num_envs"], item["case"]["width"], item["case"]["height"])
        for item in plan["cases"]
    } == {(1, 128, 128), (16, 128, 128)}
    assert not (tmp_path / "absent").exists()


def test_camera_ros_basis_matches_canonical_opengl_view() -> None:
    from scipy.spatial.transform import Rotation

    view = {"eye": [2, -2, 1], "target": [0, 0, 0], "up": [0, 0, 1]}
    matrix = backend._camera_ros_matrix(view)
    position, quaternion = camera_world_pose(view)
    gl = Rotation.from_quat(quaternion).as_matrix()
    np.testing.assert_allclose(
        matrix[:3, :3], np.column_stack((-gl[:, 2], -gl[:, 0], gl[:, 1])), atol=1e-15
    )
    np.testing.assert_allclose(matrix[:3, 3], position)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1)


def test_native_gamma_rgb_is_preserved_without_double_encoding() -> None:
    data = torch.tensor([[[[0.0, 0.25, 0.5, 1.0]]]], dtype=torch.float32)
    result = backend._rgb_device(data, torch, "default")
    assert result.tolist() == [[[[0, 64, 128]]]]
    assert result.device == data.device
    assert result.is_contiguous()
    encoded = torch.tensor([[[[10, 40, 90, 255]]]], dtype=torch.uint8)
    assert backend._rgb_device(encoded, torch, "minimal").tolist() == [[[[10, 40, 90]]]]
    with pytest.raises(ValueError, match="uint8"):
        backend._rgb_device(data, torch, "minimal")


def test_pinned_runtime_hash_check_does_not_load_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "libPhysXGpu_64.so"
    library.write_bytes(b"not executable; metadata fixture")
    monkeypatch.setattr(
        backend.metadata, "version", lambda name: backend._VERSIONS[name]
    )
    monkeypatch.setattr(
        backend.ctypes,
        "CDLL",
        lambda *args: pytest.fail("library loading is forbidden during preflight"),
    )
    with pytest.raises(ValueError, match="SHA256"):
        backend.runtime_evidence(library)
    monkeypatch.setattr(
        backend, "_PHYSX_SHA256", hashlib.sha256(library.read_bytes()).hexdigest()
    )
    assert backend.runtime_evidence(library)["automatic_downloads"] is False


def test_capture_adapter_uses_same_stream_and_never_steps_physics() -> None:
    calls = []
    adapter = backend.ManiSkillSceneReference.__new__(backend.ManiSkillSceneReference)
    adapter.shader, adapter.device, adapter.stream = (
        "default",
        torch.device("cpu"),
        "fixture-stream",
    )
    fake_torch = SimpleNamespace(
        uint8=torch.uint8,
        float32=torch.float32,
        cuda=SimpleNamespace(
            stream=lambda stream: (calls.append(("stream", stream)) or nullcontext()),
            synchronize=lambda device: calls.append(("sync", device)),
        ),
    )
    adapter.torch = fake_torch
    adapter.scene = SimpleNamespace(
        update_render=lambda **kwargs: calls.append(("render_update", kwargs)),
        step=lambda: pytest.fail("physics step"),
    )
    rgba = torch.zeros((1, 128, 128, 4))

    def get_obs(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "rgb": True,
            "depth": False,
            "position": False,
            "segmentation": False,
            "normal": False,
            "albedo": False,
            "apply_texture_transforms": False,
        }
        return {"Color": rgba}

    adapter.camera = SimpleNamespace(
        capture=lambda: calls.append(("capture",)), get_obs=get_obs
    )
    arrays = adapter.render_device()
    adapter.synchronize()
    host = adapter.to_host(arrays)
    assert host["rgb"].shape == (1, 128, 128, 3)
    assert [call[0] for call in calls] == [
        "stream",
        "render_update",
        "capture",
        "sync",
        "stream",
    ]


def test_render_only_setup_publishes_camera_without_physics_apply_or_articulations() -> (
    None
):
    calls = []
    data = torch.full((16, 13), float("nan"))
    pose = torch.tensor([2.0, -2.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    def set_pose(value: torch.Tensor) -> None:
        calls.append("publish_camera_pose")
        data[:, :7] = value

    camera = SimpleNamespace(
        px_body_type="kinematic",
        initial_pose=pose,
        _body_data_index=0,
        set_pose=set_pose,
    )

    class Physx:
        cuda_rigid_body_data = SimpleNamespace(torch=lambda: data)

        def gpu_init(self) -> None:
            calls.append("gpu_init")

        def __getattr__(self, name: str) -> Any:
            pytest.fail(f"Unnecessary simulation API called: {name}")

    scene = SimpleNamespace(px=Physx(), actors={"camera": camera}, articulations={})
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: calls.append(("sync", device)))
    )
    evidence = backend._setup_camera_pose_buffer(scene, fake_torch, data.device)
    assert calls == [
        "gpu_init",
        ("sync", data.device),
        "publish_camera_pose",
        ("sync", data.device),
    ]
    assert scene._gpu_sim_initialized
    assert scene._needs_fetch is False
    torch.testing.assert_close(data[:, :7], pose.repeat(16, 1))
    assert torch.all(data[:, 7:] == 0)
    assert evidence["simulation_apply_calls"] == 0
    assert evidence["articulation_update_calls"] == 0
    assert evidence["rigid_pose_rows"] == 16


def test_render_only_setup_rejects_hidden_articulation_before_gpu_init() -> None:
    scene = SimpleNamespace(articulations={"dummy": object()})
    with pytest.raises(ValueError, match="articulations"):
        backend._setup_camera_pose_buffer(scene, object(), object())


def test_render_only_setup_retains_warm_start_error_stage() -> None:
    def failed_init() -> None:
        raise RuntimeError("native failure")

    scene = SimpleNamespace(
        articulations={},
        actors={"camera": SimpleNamespace(px_body_type="kinematic")},
        px=SimpleNamespace(gpu_init=failed_init),
    )
    with pytest.raises(RuntimeError, match="allocation/warm-start") as caught:
        backend._setup_camera_pose_buffer(scene, object(), object())
    assert str(caught.value.__cause__) == "native failure"


@pytest.mark.parametrize("delivery", ["device_ready", "host_ready"])
def test_worker_preserves_boundaries_all_env_freshness_and_unqualified_status(
    approved: Path, tmp_path: Path, fake_runtime: dict[str, Any], delivery: str
) -> None:
    config = _config(approved, tmp_path, delivery=delivery)
    output = tmp_path / "run"
    result = worker.run_worker(config, output, gpu_id=5)
    assert result["status"] == "completed", result.get("error")
    assert result["quality_status"] == "not_qualified"
    assert result["hardware_id"] == "fixture-gpu-5"
    assert len(result["validation"]["exposure_evidence"]) == 16
    assert result["counts"]["exposure_count"] == 4 * 16
    assert result["counts"]["d2h_group_calls"] == (4 if delivery == "host_ready" else 0)
    assert result["metrics"]["camera_frames_per_s"] == pytest.approx(
        result["metrics"]["batch_steps_per_s"] * 16
    )
    assert result["metrics"]["d2h_bytes_per_capture"] == (
        16 * 128 * 128 * 3 if delivery == "host_ready" else 0
    )
    assert result["metrics"]["gpu_process_peak_bytes"] is None
    sample = read_json(output / "frames.json")
    assert sample["sampled_env_ids"] == list(range(16))
    assert sample["outside_measured_window"] is True
    renders = [event for event in fake_runtime["events"] if event[0] == "render"]
    assert renders[-1][1] is False
    assert sum(event[2] == "measurement" for event in renders) == 4
    assert fake_runtime["adapter"].closed


@pytest.mark.parametrize("failure", ["one_stale_environment", "oom", "cleanup"])
def test_worker_retains_failing_evidence(
    approved: Path, tmp_path: Path, fake_runtime: dict[str, Any], failure: str
) -> None:
    fake_runtime.update(
        stale_env=15 if failure == "one_stale_environment" else None,
        fail_capture=failure == "oom",
        close_failure=failure == "cleanup",
    )
    result = worker.run_worker(_config(approved, tmp_path), tmp_path / "run", 3)
    assert result["status"] == "failed"
    assert (
        result["failure_reason_code"]
        == {
            "one_stale_environment": "freshness_probe_failed",
            "oom": "oom",
            "cleanup": "cleanup_failed",
        }[failure]
    )
    assert result["quality_status"] == "not_qualified"
    if failure == "one_stale_environment":
        assert result["validation"]["exposure_evidence"][0]["passed"]
        assert not result["validation"]["exposure_evidence"][15]["passed"]
    assert read_json(tmp_path / "run/result.json")["status"] == "failed"


def test_missing_runtime_is_not_run_before_adapter_construction(
    approved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        worker,
        "runtime_evidence",
        lambda _: (_ for _ in ()).throw(FileNotFoundError("library absent")),
    )
    monkeypatch.setattr(
        worker,
        "ManiSkillSceneReference",
        lambda *args, **kwargs: pytest.fail("no GPU construction"),
    )
    result = worker.run_worker(_config(approved, tmp_path), tmp_path / "run", 2)
    assert result["status"] == "not_run"
    assert result["failure_reason_code"] == "dependency_missing"
    assert not result["execution_attempted"]


def test_failed_constructor_retains_pre_setup_device_checkpoint(
    approved: Path,
    tmp_path: Path,
    fake_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_constructor(
        *args: Any, construction_evidence: Path, **kwargs: Any
    ) -> None:
        write_json(
            construction_evidence,
            {
                "stage": "before_pose_buffer_initialization",
                "device_alignment": "passed",
                "torch_device": {"cuda_id": 5, "uuid": "fixture-gpu-5"},
                "physx_device": {"cuda_id": 5},
                "render_devices": [{"cuda_id": 5}],
                "articulations": 0,
                "hidden_geometry_added": False,
            },
        )
        raise RuntimeError("native setup failure")

    monkeypatch.setattr(worker, "ManiSkillSceneReference", failed_constructor)
    result = worker.run_worker(_config(approved, tmp_path), tmp_path / "run", 5)
    assert result["status"] == "failed"
    assert result["failure_reason_code"] == "construction_failed"
    assert "native setup failure" in result["error"]
    assert result["construction"]["device_alignment"] == "passed"
    assert result["hardware_binding"]["status"] == "passed"
    assert "construction.json" in result["evidence"]


def test_mismatched_gpu_identity_retains_timings_but_fails_execution(
    approved: Path, tmp_path: Path, fake_runtime: dict[str, Any]
) -> None:
    fake_runtime["torch_uuid"] = "GPU-wrong-device"
    result = worker.run_worker(_config(approved, tmp_path), tmp_path / "run", 5)
    assert result["status"] == "failed"
    assert result["failure_reason_code"] == "gpu_identity_mismatch"
    assert result["hardware_binding"]["status"] == "failed"
    assert result["metrics"]["batch_steps_per_s"] > 0


@pytest.mark.parametrize("field", ["scene_sha256", "view_sha256", "profile_sha256"])
def test_changed_frozen_identity_blocks_before_runtime_import(
    approved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    path = _config(approved, tmp_path)
    config = read_json(path)
    config[field] = "0" * 64
    write_json(path, config)
    monkeypatch.setattr(
        worker,
        "runtime_evidence",
        lambda _: pytest.fail("preflight must block runtime"),
    )
    result = worker.run_worker(path, tmp_path / "run", 0)
    assert result["status"] == "failed"
    assert not result["execution_attempted"]


def test_launcher_uses_core_plan_and_report_keeps_excluded_cells(
    approved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []

    def fake_execute(
        root: Path, runs: list[Any], **kwargs: Any
    ) -> list[dict[str, Any]]:
        seen.extend(runs)
        rows = []
        for spec in runs:
            case = read_json(Path(spec.command[spec.command.index("--config") + 1]))[
                "case"
            ]
            rows.append(
                {
                    "run_id": spec.output.name,
                    "case_id": spec.case_id,
                    "backend": "maniskill",
                    "repeat": spec.repeat,
                    "status": "failed",
                    "failure_reason_code": "fixture_unsupported",
                    "quality_status": "not_qualified",
                    "hardware_id": None,
                    "config": case,
                    "config_sha256": stable_hash(case),
                    "metrics": {},
                }
            )
        write_json(root / "runs.json", rows)
        return rows

    monkeypatch.setattr(runner, "run_experiment", fake_execute)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "99")
    output = tmp_path / "experiments"
    runner.main(
        [
            "--scene-root",
            str(approved),
            "--output",
            str(output),
            "--num-envs",
            "1",
            "--smoke",
            "--gpu-id",
            "5",
        ]
    )
    root = next(output.iterdir())
    assert len(seen) == 1
    assert "CUDA_VISIBLE_DEVICES" not in seen[0].env
    assert seen[0].env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert seen[0].env["VIRTUAL_ENV"].endswith(".venv-maniskill-reference")
    rows = read_json(root / "runs.json")
    assert {row["status"] for row in rows} == {"failed", "not_run"}
    assert len(read_json(root / "definition.json")["parameter_matrix"]["case"]) == 2
    assert "No equal-quality speedup" in (root / "report.md").read_text()
    assert "config_sha256" in (root / "reference_runs.csv").read_text()
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "scripts.benchmark.rendering.maniskill_reference",
            "--report-only",
            str(root),
        ],
        cwd=runner.ROOT,
        check=True,
        capture_output=True,
    )


def test_report_keeps_hardware_and_independent_repeats_separate(tmp_path: Path) -> None:
    case = runner.reference_cases()[0]
    rows = [
        {
            "run_id": f"r{index}",
            "case_id": case["case_id"],
            "backend": "maniskill",
            "repeat": index,
            "status": "completed",
            "quality_status": "not_qualified",
            "hardware_id": hardware,
            "config": case,
            "config_sha256": stable_hash(case),
            "metrics": {"camera_frames_per_s": rate},
        }
        for index, (hardware, rate) in enumerate(
            (("gpu-a", 10.0), ("gpu-a", 20.0), ("gpu-b", 50.0))
        )
    ]
    write_json(tmp_path / "runs.json", rows)
    runner.rebuild_report(tmp_path)
    summary = read_json(tmp_path / "summary.json")
    assert len(summary["groups"]) == 2
    metric = summary["groups"][0]["metrics"]["camera_frames_per_s"]
    assert metric["median"] == 15
    assert metric["source_runs"] == ["r0", "r1"]
    assert metric["valid_runs"] == 2
    assert metric["uncertainty"] is not None
    assert summary["groups"][1]["metrics"]["camera_frames_per_s"]["uncertainty"] is None
    assert summary["equal_quality_comparison"] is False
