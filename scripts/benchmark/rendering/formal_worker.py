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

"""Isolated formal rendering worker, with immutable scene/replay evidence.

The worker does not evaluate image quality or initialize reference renderers.
Its ordinary import and unsupported-deformation preflight are simulator-free.
Only the selected adapter creates a GPU context. Every capture uses frozen
object/camera state; no physics stepping is requested.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import faulthandler
import hashlib
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.core.artifacts import read_json_object, stable_hash, write_json
from scripts.benchmark.core.provenance import (
    git_revision,
    host_snapshot,
    package_version,
    software_snapshot,
)
from scripts.benchmark.rendering.capture import (
    capture_packet,
    measure_rendering,
    validate_capture_packet,
)
from scripts.benchmark.rendering.formal import RenderCase, render_profile
from scripts.benchmark.rendering.reference import (
    DEPTH_SEMANTICS,
    SEGMENTATION_SEMANTICS,
    camera_metadata,
)
from scripts.benchmark.rendering.replay import (
    frame_at,
    load_deformation_replay,
    load_replay,
)
from scripts.benchmark.rendering.resources import ResourceSampler
from scripts.benchmark.rendering.scene_cases import fixed_views
from scripts.benchmark.rendering.scenes import load_scene, scene_fingerprint

__all__ = ["main", "run_worker"]


def _renderer_capacity_evidence(output: Path) -> dict[str, Any]:
    """Read only explicit native Vulkan capacity errors from the current log."""
    multiview = re.compile(
        r"Invalid Vulkan multiview count:\s*requested\s+(\d+)\s*,\s*supported\s+(\d+)"
    )
    sampler_vuid = "VUID-vkCreateSampler-maxSamplerAllocationCount-04110"
    sampler_header = re.compile(r"Validation Error:\s*\[\s*" + sampler_vuid + r"\s*\]")
    sampler_counts = re.compile(
        r"vkCreateSampler\(\):\s*Number of currently valid sampler objects\s*\((\d+)\)"
        r"\s*is not less than the maximum allowed\s*\((\d+)\)"
    )
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    events: list[dict[str, Any]] = []
    digest, size = hashlib.sha256(), 0
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "unavailable",
        "scope": "known Vulkan multiview and sampler-capacity signatures; not full renderer validation",
        "checkpoint": "before_native_close",
        "log_path": "worker.log",
        "events": events,
        "reason_codes": [],
        "counts": {},
    }
    pending_sampler = None
    try:
        with (output / "worker.log").open("rb") as stream:
            for line_number, raw in enumerate(stream, start=1):
                digest.update(raw)
                size += len(raw)
                line = ansi.sub("", raw.decode("utf-8", errors="replace")).replace(
                    "\x00", ""
                )
                for match in multiview.finditer(line):
                    requested, supported = map(int, match.groups())
                    if requested > supported:
                        events.append(
                            {
                                "reason_code": "vulkan_multiview_limit",
                                "line": line_number,
                                "requested_views": requested,
                                "supported_views": supported,
                                "message": match.group(0),
                            }
                        )
                for match in sampler_header.finditer(line):
                    pending_sampler = {
                        "reason_code": "vulkan_sampler_allocation_limit",
                        "line": line_number,
                        "vuid": sampler_vuid,
                        "current_sampler_count": None,
                        "maximum_sampler_count": None,
                        "message": match.group(0),
                    }
                    events.append(pending_sampler)
                for match in sampler_counts.finditer(line):
                    current, maximum = map(int, match.groups())
                    if (
                        pending_sampler is not None
                        and line_number - pending_sampler["line"] <= 4
                    ):
                        pending_sampler.update(
                            current_sampler_count=current,
                            maximum_sampler_count=maximum,
                            detail_line=line_number,
                            detail=match.group(0),
                        )
                        pending_sampler = None
                    elif current >= maximum:
                        events.append(
                            {
                                "reason_code": "vulkan_sampler_allocation_limit",
                                "line": line_number,
                                "vuid": None,
                                "current_sampler_count": current,
                                "maximum_sampler_count": maximum,
                                "message": match.group(0),
                            }
                        )
    except OSError as exc:
        result["missing_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        result["status"] = "failed" if events else "no_known_capacity_errors"
        # Teardown can append to worker.log; this hash identifies only the
        # scanned prefix, not a falsely immutable final log file.
        result.update(scanned_bytes=size, scanned_prefix_sha256=digest.hexdigest())
    result["counts"] = dict(Counter(event["reason_code"] for event in events))
    result["reason_codes"] = sorted(result["counts"])
    return result


def _checkpoint_renderer_validation(output: Path, result: dict[str, Any]) -> None:
    """Preserve capacity evidence without replacing an earlier execution error."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    validation = _renderer_capacity_evidence(output)
    result["renderer_validation"] = validation
    write_json(output / "renderer_validation.json", validation)
    result["evidence"].append("renderer_validation.json")
    if validation["events"] and result["status"] == "completed":
        result.update(
            status="failed",
            failure_reason_code="renderer_capacity_limit",
            failure_phase="renderer_validation",
            error="Explicit native Vulkan capacity validation errors: "
            + ", ".join(validation["reason_codes"]),
        )
        write_json(
            output / "failure.json",
            {
                key: result[key]
                for key in ("failure_reason_code", "failure_phase", "error")
            },
        )
        result["evidence"].append("failure.json")


class _UnsupportedInput(ValueError):
    """An explicitly unavailable workload, not a fabricated successful run."""


class _FreshnessFailure(RuntimeError):
    """A changed view failed to produce distinguishable completed observations."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _camera_names(case: RenderCase) -> tuple[str, ...]:
    return (
        ("external_front",)
        if case.cameras_per_env == 1
        else ("external_front", "external_side", "wrist")
    )


def _initial_state(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_id": 0,
        "time_s": 0.0,
        "objects": {
            obj["uid"]: deepcopy(obj["world_from_object"]) for obj in scene["objects"]
        },
        "cameras": {view["name"]: deepcopy(view) for view in scene["cameras"]},
        "joint_positions": deepcopy(
            (scene.get("robot_source") or {}).get("joint_positions", {})
        ),
        "robot_links": {},
    }


def _preflight(
    config: dict[str, Any], case: RenderCase
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if case.deformation and (
        not config.get("replay_path") or not Path(config["replay_path"]).is_file()
    ):
        raise _UnsupportedInput("missing_external_frozen_deformation_sequence")
    scene_path = Path(config["scene_path"])
    if not scene_path.is_absolute():
        raise ValueError("Frozen scene_path must be absolute")
    scene = load_scene(scene_path)
    if scene["scene_id"] != case.scene_id or config[
        "scene_sha256"
    ] != scene_fingerprint(scene):
        raise ValueError("Frozen scene source hash or scene identity mismatch")
    profile = config["profile"]
    if profile != render_profile(case.profile) or config[
        "profile_sha256"
    ] != stable_hash(profile):
        raise ValueError("Frozen render profile hash or settings mismatch")
    approval = config.get("approved_scene_receipt", {})
    if (
        approval.get("status") != "approved"
        or approval.get("scope") != "scene_layout_and_camera_scheme"
        or approval.get("scenes", {}).get(case.scene_id) != config["scene_sha256"]
    ):
        raise ValueError("Approved scene receipt does not match the frozen scene")
    for field, minimum in (("warmup_frames", 0), ("measured_frames", 1)):
        if type(config.get(field)) is not int or config[field] < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    if set(_camera_names(case)) - {view["name"] for view in scene["cameras"]}:
        raise ValueError("Requested camera layout is absent from the approved scene")
    replay = None
    if case.temporal_mode == "moving":
        if not config.get("replay_path"):
            raise ValueError("Moving workload requires a frozen replay")
        replay_path = Path(config["replay_path"])
        if not replay_path.is_absolute():
            raise ValueError("Frozen replay_path must be absolute")
        replay = (
            load_deformation_replay(replay_path, scene_path)
            if case.deformation
            else load_replay(replay_path, scene_path)
        )
        if replay.get("status") == "unsupported":
            raise _UnsupportedInput(replay["reason"])
        if replay["frame_count"] != case.sequence_frames:
            raise ValueError("Frozen replay length does not match sequence_frames")
    if case.experiment_id == "R-01" and case.view_count != len(fixed_views(scene)):
        raise ValueError("R-01 requires all twenty frozen views")
    if case.experiment_id == "R-02" and (
        replay is None or replay["frame_count"] != 100
    ):
        raise ValueError("R-02 requires the complete hundred-frame frozen sequence")
    from scripts.benchmark.rendering.formal import view_fingerprint

    if config["view_sha256"] != view_fingerprint(case, scene, replay):
        raise ValueError("Frozen view/replay fingerprint mismatch")
    return scene, replay


def _create_adapter(
    backend: str, config: dict[str, Any], case: RenderCase, gpu_id: int
) -> Any:
    if backend == "embodichain":
        from scripts.benchmark.rendering.backends.scene_embodichain import (
            EmbodiChainScene as Scene,
        )
    else:
        from scripts.benchmark.rendering.backends.scene_isaaclab import (
            IsaacLabScene as Scene,
        )
    return Scene(
        Path(config["scene_path"]),
        width=case.width,
        height=case.height,
        gpu_id=gpu_id,
        num_envs=case.num_envs,
        modalities=case.modalities,
        profile=config["profile"],
        camera_names=_camera_names(case),
        diagnostic_pose_updates=False,
    )


def _capture(
    adapter: Any, case: RenderCase, frame_index: int, *, device_only: bool = False
) -> Any:
    packet = capture_packet(
        adapter,
        frame_index=frame_index,
        exposure_count=case.num_envs * case.cameras_per_env,
        delivery="device_ready" if device_only else "host_ready",
    )
    specs = validate_capture_packet(
        packet,
        num_envs=case.num_envs,
        cameras_per_env=case.cameras_per_env,
        width=case.width,
        height=case.height,
    )
    if set(specs) != set(case.modalities):
        raise ValueError("Adapter modalities do not match the frozen case")
    return packet


def _probe_values(adapter: Any, case: RenderCase) -> dict[str, Any]:
    packet = _capture(adapter, case, 0)
    # Keep a bounded pixel grid for EVERY exposure, not only the first view.
    # Sampling pixels bounds memory while preserving batch/camera coverage.
    stride_y, stride_x = max(1, (case.height + 63) // 64), max(
        1, (case.width + 63) // 64
    )
    return {
        name: array[:, ::stride_y, ::stride_x].copy()
        for name, array in packet.arrays.items()
    }


def _delta(first: Any, second: Any, modality: str) -> float:
    import numpy as np

    if modality == "seg":
        return float(np.mean(first != second))
    a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    finite = np.isfinite(a) & np.isfinite(b)
    different_validity = float(np.mean(np.isfinite(a) != np.isfinite(b)))
    delta = float(np.abs(a[finite] - b[finite]).mean()) if np.any(finite) else 0.0
    return delta / 255 if modality == "rgb" else delta + different_validity


def _freshness_probe(
    adapter: Any, case: RenderCase, initial: dict[str, Any], output: Path
) -> dict[str, Any]:
    import numpy as np

    views = [deepcopy(initial["cameras"][name]) for name in _camera_names(case)]
    adapter.apply_state(initial)
    original, stationary = _probe_values(adapter, case), _probe_values(adapter, case)
    changed = deepcopy(views)
    for camera in changed:
        camera["eye"][0] += 0.35
    try:
        adapter.set_cameras(changed)
        moved = _probe_values(adapter, case)
    finally:
        adapter.set_cameras(views)
        adapter.apply_state(initial)
    restored = _probe_values(adapter, case)
    metrics = {}
    for name in original:
        noise = _delta(original[name], stationary[name], name)
        move = _delta(original[name], moved[name], name)
        restore = _delta(moved[name], restored[name], name)
        back = _delta(original[name], restored[name], name)
        threshold = max(1 / 255 if name == "rgb" else 1e-5, 2 * noise)
        metrics[name] = {
            "stationary_delta": noise,
            "moved_delta": move,
            "moved_to_restored_delta": restore,
            "restored_to_initial_delta": back,
            "threshold": threshold,
            "passed": move > threshold
            and restore > threshold
            and back < move + threshold,
        }
    exposure_evidence = []
    for index in range(case.num_envs * case.cameras_per_env):
        modalities = {}
        for name in original:
            noise = _delta(original[name][index], stationary[name][index], name)
            move = _delta(original[name][index], moved[name][index], name)
            restore = _delta(moved[name][index], restored[name][index], name)
            back = _delta(original[name][index], restored[name][index], name)
            threshold = max(1 / 255 if name == "rgb" else 1e-5, 2 * noise)
            modalities[name] = {
                "stationary_delta": noise,
                "moved_delta": move,
                "moved_to_restored_delta": restore,
                "restored_to_initial_delta": back,
                "threshold": threshold,
                "passed": move > threshold
                and restore > threshold
                and back < move + threshold,
            }
        exposure_evidence.append(
            {
                "env_id": index // case.cameras_per_env,
                "camera_index": index % case.cameras_per_env,
                "passed": any(value["passed"] for value in modalities.values()),
                "modalities": modalities,
            }
        )
    probe_path = output / "freshness_probe.npz"
    np.savez_compressed(
        probe_path,
        **{
            f"{label}_{name}": array
            for label, arrays in (
                ("initial", original),
                ("stationary", stationary),
                ("moved", moved),
                ("restored", restored),
            )
            for name, array in arrays.items()
        },
    )
    return {
        "method": "initial_stationary_eye_x_plus_0.35m_restored",
        "passed": all(value["passed"] for value in exposure_evidence),
        "modality_evidence": metrics,
        "exposure_evidence": exposure_evidence,
        "capture_batches": 4,
        "captured_exposures": 4 * case.num_envs * case.cameras_per_env,
        "validation_scope": "every environment/camera exposure must respond in at least one modality; pixel grid sampled up to 64x64; modality correctness requires offline evaluation",
        "static_identical_pixels_are_valid": True,
        "outside_measured_window": True,
        "initial_cameras": views,
        "moved_cameras": changed,
        "sampled_arrays": {"path": probe_path.name, "sha256": _sha256(probe_path)},
    }


def _allocator(adapter: Any, *, reset: bool) -> tuple[int | None, str | None]:
    torch = getattr(adapter, "torch", None)
    device = getattr(adapter, "device", None)
    if torch is None or device is None:
        return None, "selected_device_torch_allocator_unavailable"
    try:
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
            return None, None
        return int(torch.cuda.max_memory_allocated(device)), None
    except Exception as exc:
        return (
            None,
            f"selected_device_torch_allocator_unavailable: {type(exc).__name__}: {exc}",
        )


def _moving_ids(scene: dict[str, Any], replay: dict[str, Any] | None) -> list[int]:
    if replay is None:
        return []
    initial = replay["frames"][0]["objects"]
    return [
        obj["instance_id"]
        for obj in scene["objects"]
        if any(
            frame["objects"][obj["uid"]] != initial[obj["uid"]]
            for frame in replay["frames"][1:]
        )
    ]


def _save_batch(
    output: Path,
    manifest: dict[str, Any],
    packet: Any,
    case: RenderCase,
    scene: dict[str, Any],
    config: dict[str, Any],
    backend: str,
    state: dict[str, Any],
    views: list[dict[str, Any]],
    moving_ids: list[int],
) -> None:
    import numpy as np

    directory = output / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    capture_index = manifest["capture_batches"]
    manifest["capture_batches"] += 1
    manifest["captured_exposures"] += case.num_envs * case.cameras_per_env
    for env_id in manifest["representative_env_ids"]:
        for camera_index, view in enumerate(views):
            batch_index = env_id * case.cameras_per_env + camera_index
            stem = f"capture_{capture_index:04d}_env_{env_id:04d}_camera_{camera_index:02d}"
            arrays_path = directory / f"{stem}.npz"
            arrays = {}
            for name, batch in packet.arrays.items():
                sample = batch[batch_index]
                if name in {"depth", "seg"} and sample.ndim == 3:
                    sample = sample[..., 0]
                arrays[name] = sample
            np.savez_compressed(arrays_path, **arrays)
            camera = camera_metadata(scene, view, case.width, case.height)
            metadata = {
                "backend": backend,
                "scene_sha256": config["scene_sha256"],
                "view_sha256": stable_hash(camera),
                "workload_view_sha256": config["view_sha256"],
                "profile_sha256": config["profile_sha256"],
                "frame_id": state["frame_id"],
                "frame_index": state["frame_id"],
                "time_s": state["time_s"],
                "capture_index": capture_index,
                "env_id": env_id,
                "camera_index": camera_index,
                "camera_name": view["name"],
                "batch_index": batch_index,
                "view_id": view.get("view_id"),
                "camera": camera,
                "rgb_encoding": "srgb_uint8",
                "depth_semantics": DEPTH_SEMANTICS,
                "segmentation_semantics": SEGMENTATION_SEMANTICS,
                "depth_invalid": "positive_infinity",
                "instance_ids": [obj["instance_id"] for obj in scene["objects"]],
                "world_from_instances": {
                    str(obj["instance_id"]): state["objects"][obj["uid"]]
                    for obj in scene["objects"]
                },
                "world_from_objects": state["objects"],
                "moving_instance_ids": moving_ids,
                "joint_positions": state.get("joint_positions", {}),
                "robot_links": state.get("robot_links", {}),
                "coordinate_frame": "canonical_scene_before_environment_replication",
                "physics_steps": 0,
                "physical_validation": "not_performed",
                "photometric_calibration": "not_qualified",
            }
            record_path = directory / f"{stem}.json"
            write_json(
                record_path,
                {
                    "schema_version": 1,
                    "metadata": metadata,
                    "arrays": {
                        "path": arrays_path.name,
                        "sha256": _sha256(arrays_path),
                    },
                },
            )
            manifest["frames"].append(
                {
                    "path": str(record_path.relative_to(output)),
                    "sha256": _sha256(record_path),
                    "arrays_path": str(arrays_path.relative_to(output)),
                    "arrays_sha256": _sha256(arrays_path),
                    "frame_id": state["frame_id"],
                    "time_s": state["time_s"],
                    "camera_name": view["name"],
                    "camera_index": camera_index,
                    "env_id": env_id,
                    "batch_index": batch_index,
                }
            )
    manifest["saved_exposures"] = len(manifest["frames"])
    write_json(output / "frames.json", manifest)


def _videos(output: Path, manifest: dict[str, Any], fps: float) -> dict[str, Any]:
    if "rgb" not in manifest["modalities"]:
        return {"status": "unsupported", "reason": "RGB modality is absent"}
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as exc:
        return {"status": "unsupported", "reason": f"imageio/ffmpeg unavailable: {exc}"}
    videos = []
    for name in sorted({frame["camera_name"] for frame in manifest["frames"]}):
        members = sorted(
            (
                frame
                for frame in manifest["frames"]
                if frame["camera_name"] == name and frame["env_id"] == 0
            ),
            key=lambda frame: frame["frame_id"],
        )
        path = output / f"video_camera_{members[0]['camera_index']:02d}.mp4"
        try:
            with imageio.get_writer(
                path,
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                macro_block_size=None,
            ) as writer:
                for frame in members:
                    with np.load(
                        output / frame["arrays_path"], allow_pickle=False
                    ) as arrays:
                        writer.append_data(arrays["rgb"])
            videos.append(
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "camera_name": name,
                    "env_id": 0,
                    "frame_count": len(members),
                    "fps": fps,
                }
            )
        except Exception as exc:
            path.unlink(missing_ok=True)
            return {
                "status": "unsupported",
                "reason": f"video encoder unavailable or failed: {type(exc).__name__}: {exc}",
                "completed_videos": videos,
            }
    return {
        "status": "completed",
        "videos": videos,
        "authoritative_frames": "lossless NPZ/JSON; video is a visualization only",
    }


def _is_oom(exc: BaseException) -> bool:
    label = (type(exc).__name__ + " " + str(exc)).lower()
    return isinstance(exc, MemoryError) or any(
        token in label
        for token in (
            "outofmemory",
            "out of memory",
            "cuda_error_out_of_memory",
            "std::bad_alloc",
        )
    )


def _r09_diagnostics(
    adapter: Any,
    case: RenderCase,
    config: dict[str, Any],
    output: Path,
    replay: dict[str, Any] | None,
    final_state: dict[str, Any],
) -> dict[str, Any]:
    """Retain optional trace/overhead evidence outside the primary resource window."""
    directory = output / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "publication_eligible": False,
        "outside_primary_measurement": True,
        "outside_primary_resource_window": True,
        "policy": {
            "delivery": case.delivery,
            "diagnostic": case.diagnostic,
            "state_publication": replay is not None and case.temporal_mode == "moving",
            "trace_iterations": 3,
            "trace_warmup": 0,
            "overhead_iterations": min(config["measured_frames"], 100),
            "overhead_warmup": config["warmup_frames"],
            "overhead_repeats": 3,
            "overhead_order": "minimal/direct/common, common/direct/minimal, minimal/direct/common",
        },
        "native_timer": {
            "status": "unqualified",
            "value_s": None,
            "reason": "native_gpu_timer_unqualified: timestamp queries and camera/window scope are not validated",
            "activation_requested": False,
            "reported_enabled": adapter.metadata.get("native_profiler_enabled"),
        },
        "missing_reasons": {},
        "artifacts": [],
    }

    def prepare(index: int) -> None:
        adapter.apply_state(frame_at(replay, index % replay["frame_count"]))

    options = {
        "num_envs": case.num_envs,
        "cameras_per_env": case.cameras_per_env,
        "width": case.width,
        "height": case.height,
        "delivery": case.delivery,
        "diagnostic": case.diagnostic,
        "prepare": prepare if result["policy"]["state_publication"] else None,
    }
    try:
        from scripts.benchmark.rendering.diagnostics import (
            framework_overhead,
            trace_delivery,
        )

        operations = (
            (
                "trace",
                lambda: trace_delivery(
                    adapter,
                    output=directory / "delivery_trace",
                    warmup=0,
                    iterations=3,
                    **options,
                ),
            ),
            (
                "framework_overhead",
                lambda: framework_overhead(
                    adapter,
                    output=directory / "framework_overhead.json",
                    warmup=config["warmup_frames"],
                    iterations=result["policy"]["overhead_iterations"],
                    repeats=3,
                    **options,
                ),
            ),
        )
        for name, operation in operations:
            try:
                payload = operation()
                if name == "trace":
                    result[name] = {
                        "status": payload.get("status", "completed"),
                        "declared_observation_transfers": payload.get(
                            "declared_observation_transfers"
                        ),
                        "observed_transfers": payload.get("observed_transfers"),
                        "profiler": payload.get("profiler"),
                    }
                    reasons = payload.get("profiler", {}).get("reasons", [])
                    if reasons:
                        result["missing_reasons"]["trace_profiler"] = reasons
                else:
                    result[name] = {
                        "status": "completed",
                        "sample_windows": len(payload.get("windows", [])),
                        **{
                            key: payload.get(key)
                            for key in (
                                "paired_windows",
                                "equivalent_policy",
                                "overhead_ratio",
                                "overhead_fraction",
                                "uncertainty",
                                "common_loop_overhead_ratio",
                                "comparison_reasons",
                            )
                        },
                    }
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                result[name] = {"status": "missing", "reason": reason}
                result["missing_reasons"][name] = reason
    except Exception as exc:
        result["missing_reasons"][
            "diagnostics_unavailable"
        ] = f"{type(exc).__name__}: {exc}"
    finally:
        if result["policy"]["state_publication"]:
            try:
                adapter.apply_state(final_state)
                adapter.synchronize()
            except Exception as exc:
                result["missing_reasons"][
                    "state_restore"
                ] = f"{type(exc).__name__}: {exc}"
    if result["missing_reasons"]:
        result["status"] = "partial"
    # Include partial files emitted before an unavailable profiler/capture raised.
    result["artifacts"] = [
        {"path": path.relative_to(output).as_posix(), "sha256": _sha256(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    write_json(directory / "summary.json", result)
    return result


def run_worker(
    config_path: Path, output: Path, backend: str, gpu_id: int, repeat: int = 0
) -> dict[str, Any]:
    """Execute one frozen formal case and always preserve its result record.

    Args:
        config_path: Frozen worker configuration produced by the launcher.
        output: Run directory receiving result, timing, resource, and frame files.
        backend: ``embodichain`` or ``isaaclab``.
        gpu_id: Explicit physical GPU index used by adapter and NVML sampler.
        repeat: Independent process repeat identity from the launcher.

    Returns:
        Completed or failed run record. Unsupported deformation has failed
        execution status and explicit unsupported quality/reason fields.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "definition_version": "2.0",
        "backend": backend,
        "repeat": repeat,
        "status": "failed",
        "quality_status": "not_qualified",
        "execution_attempted": False,
        "metrics": {},
        "metric_reasons": {},
        "hardware_id": None,
        "selected_gpu_id": gpu_id,
        "evidence": [],
        "quality": {
            "status": "not_qualified",
            "reason": "offline_evaluator_proof_not_attached",
        },
    }
    adapter = sampler = None
    case = scene = replay = None
    phase = "configuration"
    manifest: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        if backend not in {"embodichain", "isaaclab"}:
            raise ValueError("Unknown rendering backend")
        if (
            type(gpu_id) is not int
            or gpu_id < 0
            or type(repeat) is not int
            or repeat < 0
        ):
            raise ValueError("GPU index and repeat must be nonnegative integers")
        config = read_json_object(Path(config_path))
        result["config"] = config.get("case", {})
        result["config_sha256"] = stable_hash(result["config"])
        result.update(
            {
                key: config.get(key)
                for key in (
                    "scene_sha256",
                    "view_sha256",
                    "profile_sha256",
                    "approved_scene_receipt",
                )
            }
        )
        if isinstance(config.get("case"), dict):
            result["experiment_id"] = config["case"].get("experiment_id", "unknown")
        case_values = dict(config["case"])
        case_values["modalities"] = tuple(case_values["modalities"])
        case = RenderCase(**case_values)
        result.update(
            case_id=case.case_id,
            experiment_id=case.experiment_id,
            config=case.to_dict(),
            config_sha256=stable_hash(case.to_dict()),
        )
        write_json(output / "worker_config.json", config)
        result["evidence"].append("worker_config.json")
        phase = "source_validation"
        scene, replay = _preflight(config, case)
        sources = [
            path
            for directory in ("core", "rendering")
            for path in (ROOT / "scripts/benchmark" / directory).rglob("*.py")
        ]
        result["software"] = software_snapshot(ROOT, sources)
        result["software"]["packages"] = {
            name: package_version(name)
            for name in (
                "torch",
                "numpy",
                "dexsim-engine",
                "isaacsim",
                "isaaclab",
                "nvidia-ml-py",
                "pynvml",
            )
        }
        result["host"] = host_snapshot()
        if config.get("isaaclab_root"):
            result["software"]["isaaclab_root"] = config["isaaclab_root"]
            result["software"]["isaaclab_commit"] = git_revision(
                Path(config["isaaclab_root"])
            )
        initial = frame_at(replay, 0) if replay is not None else _initial_state(scene)
        moving_ids = _moving_ids(scene, replay)
        manifest = {
            "schema_version": 1,
            "backend": backend,
            "case_id": case.case_id,
            "scene_sha256": config["scene_sha256"],
            "view_sha256": config["view_sha256"],
            "profile_sha256": config["profile_sha256"],
            "modalities": list(case.modalities),
            "num_envs": case.num_envs,
            "cameras_per_env": case.cameras_per_env,
            "representative_env_ids": sorted(
                {0, case.num_envs // 2, case.num_envs - 1}
            ),
            "capture_batches": 0,
            "captured_exposures": 0,
            "saved_exposures": 0,
            "batch_order": "environment_then_camera",
            "frames": [],
            "quality_status": "not_qualified",
            "physics_steps": 0,
            "fps": replay["fps"] if replay is not None else None,
        }
        if replay is not None:
            result["replay"] = {
                "sha256": stable_hash(replay),
                "kind": replay["kind"],
                "frame_count": replay["frame_count"],
                "fps": replay["fps"],
                "source": replay["source"],
                "lineage": replay["lineage"],
                "physical_validation": replay.get("physical_validation"),
            }
        sampler = ResourceSampler(gpu_id=gpu_id)
        with sampler:
            phase = "construction"
            result["execution_attempted"] = True
            build_started = time.perf_counter()
            adapter = _create_adapter(backend, config, case, gpu_id)
            result["setup_s"] = time.perf_counter() - build_started
            result["renderer"] = deepcopy(adapter.metadata)
            if adapter.metadata.get("physics_enabled") is not False:
                raise ValueError(
                    "Backend must explicitly declare physics_enabled=False"
                )
            torch = getattr(adapter, "torch", None)
            result["software"]["torch_runtime"] = getattr(torch, "__version__", None)
            result["software"]["cuda_runtime"] = getattr(
                getattr(torch, "version", None), "cuda", None
            )
            if case.deformation and not adapter.metadata.get(
                "supports_deformation", False
            ):
                raise _UnsupportedInput("frozen_deformation_publication_unavailable")
            sampler.mark_phase("freshness_probe")
            phase = "freshness_probe"
            result["validation"] = _freshness_probe(adapter, case, initial, output)
            write_json(output / "freshness_probe.json", result["validation"])
            result["evidence"].extend(["freshness_probe.json", "freshness_probe.npz"])
            if not result["validation"]["passed"]:
                raise _FreshnessFailure(
                    "Changed and restored views did not establish fresh completed delivery"
                )
            adapter.apply_state(initial)
            sampler.mark_phase("warmup")
            phase = "capture"
            if case.experiment_id in {"R-01", "R-02"}:
                records, warmups = [], []
                if case.experiment_id == "R-01":
                    captures = [
                        (dict(initial, frame_id=index), [view])
                        for index, view in enumerate(fixed_views(scene))
                    ]
                else:
                    captures = [(frame_at(replay, index), None) for index in range(100)]
                for capture_index, (state, views) in enumerate(captures):
                    if views is None:
                        adapter.apply_state(state)
                        views = [state["cameras"][name] for name in _camera_names(case)]
                    # R-01 is static: initial object state is already published.
                    # Its view aliases replace selected camera names, so only
                    # set_cameras owns subsequent camera changes.
                    adapter.set_cameras(views)
                    if case.experiment_id == "R-01" or capture_index == 0:
                        sampler.mark_phase("warmup")
                        for warmup_index in range(config["warmup_frames"]):
                            warmups.append(
                                _capture(
                                    adapter, case, warmup_index, device_only=True
                                ).to_metadata()
                            )
                    sampler.mark_phase("quality_capture")
                    packet = _capture(adapter, case, state["frame_id"])
                    records.append(packet.to_metadata())
                    _save_batch(
                        output,
                        manifest,
                        packet,
                        case,
                        scene,
                        config,
                        backend,
                        state,
                        views,
                        moving_ids,
                    )
                    del packet
                result["quality_capture"] = {
                    "capture_batches": len(records),
                    "captured_exposures": len(records)
                    * case.num_envs
                    * case.cameras_per_env,
                    "saved_exposures": manifest["saved_exposures"],
                    "warmup_batches": len(warmups),
                    "throughput_claim": False,
                    "raw": records,
                    "warmup_raw": warmups,
                }
                write_json(output / "capture.json", result["quality_capture"])
                result["metrics"].update(
                    boundary="untimed_completed_capture",
                    completion="blocking_host_delivery_and_device_sync",
                )
                for name in (
                    "batch_steps_per_s",
                    "camera_frames_per_s",
                    "gpixel_per_s",
                    "latency_p50_s",
                    "latency_p95_s",
                    "state_publish_p50_s",
                    "device_ready_p50_s",
                    "host_ready_p50_s",
                    "native_render_s",
                    "torch_peak_allocated_bytes",
                ):
                    result["metrics"][name] = None
                    result["metric_reasons"][
                        name
                    ] = "quality_capture_not_throughput_experiment"
            else:
                last_state = initial
                allocator_reset_reason = None

                def prepare(index: int) -> None:
                    nonlocal last_state
                    last_state = frame_at(replay, index % replay["frame_count"])
                    adapter.apply_state(last_state)

                def measurement_start() -> None:
                    nonlocal allocator_reset_reason
                    adapter.synchronize()
                    _, allocator_reset_reason = _allocator(adapter, reset=True)
                    sampler.mark_phase("measurement")

                measured = measure_rendering(
                    adapter,
                    warmup=config["warmup_frames"],
                    iterations=config["measured_frames"],
                    num_envs=case.num_envs,
                    cameras_per_env=case.cameras_per_env,
                    width=case.width,
                    height=case.height,
                    delivery=case.delivery,
                    prepare=(
                        prepare
                        if replay is not None and case.temporal_mode == "moving"
                        else None
                    ),
                    diagnostic=case.diagnostic,
                    on_measurement_start=measurement_start,
                )
                peak, reason = _allocator(adapter, reset=False)
                result["metrics"] = measured["metrics"]
                result["metric_reasons"] = measured["metric_reasons"]
                result["metrics"]["torch_peak_allocated_bytes"] = (
                    peak if allocator_reset_reason is None else None
                )
                if reason or allocator_reset_reason:
                    result["metric_reasons"]["torch_peak_allocated_bytes"] = (
                        allocator_reset_reason or reason
                    )
                result["counts"] = measured["counts"]
                result["warmup_counts"] = measured["warmup_counts"]
                sampler.mark_phase("measurement_complete")
                # Store JSON and samples after leaving the resource window below.
        result["resources"] = sampler.summary()
        result["hardware_id"] = result["resources"]["gpu"].get("uuid")
        result["metrics"].update(result["resources"]["metrics"])
        result["metric_reasons"].update(result["resources"].get("metric_reasons", {}))
        if case.experiment_id == "R-09" and case.diagnostic:
            result["metrics"]["native_render_s"] = None
            result["metric_reasons"][
                "native_render_s"
            ] = "native_gpu_timer_unqualified: timestamp queries and camera/window scope are not validated"
        write_json(output / "resources.json", result["resources"])
        result["evidence"].extend(["resources.json", "capture.json"])
        phase = "sample_persistence"
        if case.experiment_id not in {"R-01", "R-02"}:
            write_json(output / "capture.json", measured)
            packet = _capture(adapter, case, last_state["frame_id"])
            views = [last_state["cameras"][name] for name in _camera_names(case)]
            _save_batch(
                output,
                manifest,
                packet,
                case,
                scene,
                config,
                backend,
                last_state,
                views,
                moving_ids,
            )
            result["sample_capture"] = {
                **packet.to_metadata(),
                "outside_measured_window": True,
                "replay_frame_id": last_state["frame_id"],
            }
            del packet
        if hasattr(adapter, "refresh_metadata"):
            adapter.refresh_metadata()
        result["renderer"] = deepcopy(adapter.metadata)
        if case.experiment_id == "R-09" and case.diagnostic:
            # Preserve the primary result even if optional native trace support
            # terminates the process instead of raising a Python exception.
            result["status"] = "completed"
            result["frames_manifest"] = {
                "path": "frames.json",
                "sha256": _sha256(output / "frames.json"),
            }
            result["diagnostics"] = {"status": "pending", "publication_eligible": False}
            write_json(output / "result.json", result)
            try:
                result["diagnostics"] = _r09_diagnostics(
                    adapter, case, config, output, replay, last_state
                )
                result["evidence"].extend(
                    [item["path"] for item in result["diagnostics"]["artifacts"]]
                    + ["diagnostics/summary.json"]
                )
            except Exception as exc:
                result["diagnostics"] = {
                    "status": "missing",
                    "publication_eligible": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        if case.experiment_id == "R-02":
            result["video"] = _videos(output, manifest, replay["fps"])
            write_json(output / "video.json", result["video"])
            result["evidence"].append("video.json")
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["failure_reason_code"] = (
            "unsupported"
            if isinstance(exc, _UnsupportedInput)
            else (
                "oom"
                if _is_oom(exc)
                else (
                    "freshness_probe_failed"
                    if isinstance(exc, _FreshnessFailure)
                    else {
                        "configuration": "invalid_config",
                        "source_validation": "source_validation_failed",
                        "construction": "construction_failed",
                        "sample_persistence": "artifact_persistence_failed",
                    }.get(phase, "capture_failed")
                )
            )
        )
        if isinstance(exc, _UnsupportedInput):
            result["quality_status"] = "unsupported"
            result["quality"] = {"status": "unsupported", "reason": str(exc)}
        result["failure_phase"] = phase
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
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
        if sampler is not None and "resources" not in result:
            result["resources"] = sampler.summary()
            result["hardware_id"] = result["resources"]["gpu"].get("uuid")
            result["metrics"].update(result["resources"]["metrics"])
            result["metric_reasons"].update(
                result["resources"].get("metric_reasons", {})
            )
            write_json(output / "resources.json", result["resources"])
            result["evidence"].append("resources.json")
        if manifest is not None:
            write_json(output / "frames.json", manifest)
            result["frames_manifest"] = {
                "path": "frames.json",
                "sha256": _sha256(output / "frames.json"),
            }
            result["quality"]["frames_manifest"] = "frames.json"
            result["evidence"].append("frames.json")
        _checkpoint_renderer_validation(output, result)
        # Native applications may terminate the interpreter inside close(),
        # including os._exit(0). Persist the execution outcome before entering
        # native teardown; a clean process exit is observed by the launcher.
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
                    result["status"] = "failed"
                    result["failure_reason_code"] = "cleanup_failed"
            result["worker_wall_s"] = time.perf_counter() - started
            write_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the isolated worker CLI.

    Args:
        argv: Optional argument list for tests; otherwise use process arguments.

    Returns:
        Zero for completed execution, one for preserved failed/unsupported work.
    """
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("embodichain", "isaaclab", "E", "I"), required=True
    )
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args(argv)
    backend = {"E": "embodichain", "I": "isaaclab"}.get(args.backend, args.backend)
    result = run_worker(args.config, args.output, backend, args.gpu_id, args.repeat)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
