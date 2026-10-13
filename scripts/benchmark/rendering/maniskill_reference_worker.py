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

"""Fresh-process capture of one unqualified ManiSkill raster reference point."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
from typing import Any

from scripts.benchmark.core.artifacts import read_json_object, stable_hash, write_json
from scripts.benchmark.core.provenance import (
    host_snapshot,
    package_version,
    software_snapshot,
)
from .backends.scene_maniskill import (
    ManiSkillSceneReference,
    raster_profile,
    runtime_evidence,
)
from .capture import capture_packet, measure_rendering, validate_capture_packet
from .formal import view_fingerprint
from .formal_worker import _freshness_probe, _initial_state, _sha256
from .maniskill_reference import EXPERIMENT_ID, ROOT, reference_cases
from .resources import ResourceSampler
from .scenes import load_scene, scene_fingerprint

__all__ = ["main", "run_worker"]


def _preflight(config: dict[str, Any]) -> dict[str, Any]:
    case = config["case"]
    if case not in reference_cases(case["shader"], case["delivery"]):
        raise ValueError(
            "Only the two frozen static RGB reference points are supported"
        )
    if config.get("config_sha256") != stable_hash(case):
        raise ValueError("Reference config SHA256 mismatch")
    profile = raster_profile(case["shader"])
    if config["profile"] != profile or config["profile_sha256"] != stable_hash(profile):
        raise ValueError("Reference profile SHA256/settings mismatch")
    scene_path = Path(config["scene_path"])
    if not scene_path.is_absolute():
        raise ValueError("Frozen scene path must be absolute")
    scene = load_scene(scene_path)
    if (
        scene["scene_id"] != case["scene_id"]
        or scene_fingerprint(scene) != config["scene_sha256"]
    ):
        raise ValueError("Reference scene identity/hash mismatch")
    if view_fingerprint(SimpleNamespace(**case), scene) != config["view_sha256"]:
        raise ValueError("Reference view SHA256 mismatch")
    approval = config["approved_scene_receipt"]
    if (
        approval.get("status") != "approved"
        or approval.get("scope") != "scene_layout_and_camera_scheme"
        or approval.get("scenes", {}).get(case["scene_id"]) != config["scene_sha256"]
    ):
        raise ValueError("Reference scene approval does not bind this scene")
    for key, minimum in (("warmup_frames", 0), ("measured_frames", 1)):
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    return scene


def _sample(adapter: Any, case: dict[str, Any], output: Path) -> dict[str, Any]:
    import numpy as np

    packet = capture_packet(
        adapter, frame_index=0, exposure_count=case["num_envs"], delivery="host_ready"
    )
    validate_capture_packet(
        packet, num_envs=case["num_envs"], cameras_per_env=1, width=128, height=128
    )
    if set(packet.arrays) != {"rgb"}:
        raise ValueError("The reference capture must deliver exactly RGB")
    rgb = packet.arrays["rgb"]
    path = output / "all_env_rgb.npz"
    np.savez_compressed(path, rgb=rgb)
    adapter.refresh_metadata()
    return {
        "schema_version": 1,
        "num_envs": case["num_envs"],
        "cameras_per_env": 1,
        "batch_order": "environment_then_camera",
        "outside_measured_window": True,
        "sampled_env_ids": list(range(case["num_envs"])),
        "camera_name": "external_front",
        "rgb": {
            "path": path.name,
            "sha256": _sha256(path),
            "shape": list(rgb.shape),
            "dtype": str(rgb.dtype),
        },
        "cameras": deepcopy(adapter.metadata.get("cameras", [])),
        "rgb_encoding": "native_gamma_2.2_approximation_not_IEC_sRGB",
        "quality_status": "not_qualified",
        "capture": packet.to_metadata(),
    }


def run_worker(
    config_path: Path, output: Path, gpu_id: int, repeat: int = 0
) -> dict[str, Any]:
    """Measure one point and preserve failure, resource and sample evidence.

    Args:
        config_path: Frozen launcher configuration.
        output: Fresh worker directory, whose result is checkpointed before close.
        gpu_id: Explicit physical GPU index in the launcher's unmasked order.
        repeat: Independent worker-process repeat identity.

    Returns:
        Standard schema-one result, always independently marked not qualified.
    """
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "definition_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "backend": "maniskill",
        "repeat": repeat,
        "status": "failed",
        "quality_status": "not_qualified",
        "hardware_id": None,
        "selected_gpu_id": gpu_id,
        "execution_attempted": False,
        "metrics": {},
        "metric_reasons": {},
        "evidence": [],
        "quality": {
            "status": "not_qualified",
            "reason": "separate_raster_profile_without_independent_calibration",
            "equal_quality_comparison": False,
        },
    }
    adapter = sampler = None
    phase = "configuration"
    started = time.perf_counter()
    try:
        if (
            type(gpu_id) is not int
            or gpu_id < 0
            or type(repeat) is not int
            or repeat < 0
        ):
            raise ValueError("GPU index/repeat must be nonnegative integers")
        config = read_json_object(config_path)
        case = config["case"]
        result.update(
            {
                key: config[key]
                for key in (
                    "config_sha256",
                    "scene_sha256",
                    "view_sha256",
                    "profile_sha256",
                )
            }
        )
        result.update(config=case, case_id=case["case_id"])
        write_json(output / "worker_config.json", config)
        result["evidence"].append("worker_config.json")
        scene = _preflight(config)
        phase = "dependency_check"
        result["runtime"] = runtime_evidence(Path(config["physx_gpu_library"]))
        result["host"] = host_snapshot()
        result["software"] = software_snapshot(
            ROOT,
            [
                Path(__file__),
                ROOT / "scripts/benchmark/rendering/backends/scene_maniskill.py",
                ROOT / "scripts/benchmark/rendering/capture.py",
            ],
        )
        result["software"]["packages"] = {
            name: package_version(name)
            for name in ("mani-skill", "sapien", "torch", "numpy", "nvidia-ml-py")
        }
        initial = _initial_state(scene)
        sampler = ResourceSampler(gpu_id=gpu_id)
        with sampler:
            phase = "construction"
            result["execution_attempted"] = True
            setup_start = time.perf_counter()
            adapter = ManiSkillSceneReference(
                Path(config["scene_path"]),
                num_envs=case["num_envs"],
                gpu_id=gpu_id,
                shader=case["shader"],
                physx_gpu_library=Path(config["physx_gpu_library"]),
                construction_evidence=output / "construction.json",
            )
            result["setup_s"] = time.perf_counter() - setup_start
            result["renderer"] = deepcopy(adapter.metadata)
            sampler.mark_phase("freshness_probe")
            phase = "freshness_probe"
            probe = _freshness_probe(adapter, SimpleNamespace(**case), initial, output)
            write_json(output / "freshness_probe.json", probe)
            result["validation"] = probe
            result["evidence"].extend(["freshness_probe.json", "freshness_probe.npz"])
            if not probe["passed"]:
                raise RuntimeError(
                    "Changed/restored camera did not establish freshness for every environment"
                )
            adapter.apply_state(initial)
            sampler.mark_phase("warmup")
            allocator_reason = None

            def measurement_start() -> None:
                nonlocal allocator_reason
                adapter.synchronize()
                try:
                    adapter.torch.cuda.reset_peak_memory_stats(adapter.device)
                except (AttributeError, RuntimeError) as exc:
                    allocator_reason = str(exc)
                sampler.mark_phase("measurement")

            phase = "capture"
            measured = measure_rendering(
                adapter,
                warmup=config["warmup_frames"],
                iterations=config["measured_frames"],
                num_envs=case["num_envs"],
                cameras_per_env=1,
                width=128,
                height=128,
                delivery=case["delivery"],
                on_measurement_start=measurement_start,
            )
            result["metrics"] = measured["metrics"]
            result["metric_reasons"] = measured["metric_reasons"]
            result["counts"], result["warmup_counts"] = (
                measured["counts"],
                measured["warmup_counts"],
            )
            try:
                peak = adapter.torch.cuda.max_memory_allocated(adapter.device)
            except (AttributeError, RuntimeError) as exc:
                peak, allocator_reason = None, str(exc)
            result["metrics"]["torch_peak_allocated_bytes"] = (
                peak if allocator_reason is None else None
            )
            if allocator_reason is not None:
                result["metric_reasons"][
                    "torch_peak_allocated_bytes"
                ] = allocator_reason
            sampler.mark_phase("measurement_complete")
        # Primary resource/timing windows have ended before file IO and samples.
        write_json(output / "capture.json", measured)
        result["evidence"].append("capture.json")
        phase = "sample_persistence"
        sample = _sample(adapter, case, output)
        write_json(output / "frames.json", sample)
        result["evidence"].extend(["all_env_rgb.npz", "frames.json"])
        result["frames_manifest"] = {
            "path": "frames.json",
            "sha256": _sha256(output / "frames.json"),
        }
        result["renderer"] = deepcopy(adapter.metadata)
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "not_run" if phase == "dependency_check" else "failed"
        result["failure_reason_code"] = (
            "dependency_missing"
            if phase == "dependency_check"
            else (
                "oom"
                if "out of memory" in str(exc).lower()
                else (
                    "freshness_probe_failed"
                    if phase == "freshness_probe"
                    else f"{phase}_failed"
                )
            )
        )
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            failure_phase=phase,
            traceback=traceback.format_exc(),
        )
        write_json(
            output / "failure.json",
            {
                key: result[key]
                for key in (
                    "failure_reason_code",
                    "failure_phase",
                    "error",
                    "traceback",
                )
            },
        )
        result["evidence"].append("failure.json")
    finally:
        if (output / "construction.json").is_file():
            result["construction"] = read_json_object(output / "construction.json")
            result["evidence"].append("construction.json")
        if sampler is not None:
            resources = sampler.summary()
            result["resources"] = resources
            result["hardware_id"] = resources["gpu"].get("uuid")
            torch_uuid = result.get("renderer", {}).get("torch_device", {}).get(
                "uuid"
            ) or result.get("construction", {}).get("torch_device", {}).get("uuid")
            nvml_uuid = result["hardware_id"]
            identities = [
                str(value).lower().removeprefix("gpu-")
                for value in (torch_uuid, nvml_uuid)
                if value
            ]
            result["hardware_binding"] = {
                "torch_uuid": torch_uuid,
                "nvml_uuid": nvml_uuid,
                "status": (
                    "missing"
                    if len(identities) != 2
                    else "passed" if identities[0] == identities[1] else "failed"
                ),
            }
            if (
                len(identities) == 2
                and identities[0] != identities[1]
                and result["status"] == "completed"
            ):
                result.update(
                    status="failed",
                    failure_reason_code="gpu_identity_mismatch",
                    error="Torch and selected NVML GPU UUIDs differ",
                )
            result["metrics"].update(resources["metrics"])
            result["metric_reasons"].update(resources.get("metric_reasons", {}))
            write_json(output / "resources.json", resources)
            result["evidence"].append("resources.json")
        result["cleanup_status"] = (
            "pending_native_close" if adapter is not None else "not_required"
        )
        result["worker_wall_s"] = time.perf_counter() - started
        write_json(output / "result.json", result)
        if adapter is not None:
            try:
                adapter.close()
                result["cleanup_status"] = "completed"
            except Exception as exc:
                result["cleanup_status"] = "failed"
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if result["status"] == "completed":
                    result.update(status="failed", failure_reason_code="cleanup_failed")
            write_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Execute the shell-free worker interface.

    Args:
        argv: Optional worker command arguments.

    Returns:
        Zero for retained completed/not-run results; one for execution failures.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_worker(args.config, args.output, args.gpu_id, args.repeat)
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
