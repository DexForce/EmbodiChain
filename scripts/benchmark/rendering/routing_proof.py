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

"""Untimed indexed-camera routing evidence for an already frozen renderer.

The source worker supplies its original adapter and completed capture callback.
Only public indexed sensor pose APIs are used; renderer submission methods are
unchanged. This module imports no simulator, CUDA library or array package until
an explicit probe is invoked. Routing evidence is neither an FPS measurement nor
qualification of indirect-light transport or untested environment permutations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.core.measurement import Measurement

__all__ = [
    "publish_indexed_camera",
    "publication_callback_wall",
    "run_routing_probe",
]

_REQUIRED_BINDINGS = (
    "run_id",
    "case_id",
    "backend",
    "config_sha256",
    "scene_sha256",
    "view_sha256",
    "profile_sha256",
    "hardware_id",
    "source_commit",
    "source_files",
)
_DTYPES = {"rgb": "uint8", "depth": "float32", "seg": "int32", "normals": "float32"}


def publish_indexed_camera(
    adapter: Any, backend: str, env_id: int, camera_index: int, spec: dict[str, Any]
) -> dict[str, Any]:
    """Change one camera through the frozen adapter's original sensor API.

    Args:
        adapter: Already constructed original benchmark adapter.
        backend: ``embodichain`` or ``isaaclab``.
        env_id: Global environment index.
        camera_index: Logical camera index within one environment.
        spec: Canonical arena-local look-at view. Shared camera specs stay intact.

    Returns:
        Actual indexing/publication contract, suitable for the diagnostic record.
        Call again with the original view to restore; an unchanged shared-spec
        setter can otherwise skip restoration of a per-environment override.
    """
    if type(env_id) is not int or not 0 <= env_id < adapter.num_envs:
        raise ValueError("Environment index is outside the original camera batch")
    if backend not in {"embodichain", "isaaclab"}:
        raise ValueError("Unsupported frozen renderer for indexed routing proof")
    views = (
        adapter.scene["cameras"] if backend == "embodichain" else adapter.camera_specs
    )
    if type(camera_index) is not int or not 0 <= camera_index < len(views):
        raise ValueError("Camera index is outside the original view layout")
    if any(
        not isinstance(spec.get(name), (list, tuple))
        or len(spec[name]) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in spec[name])
        for name in ("eye", "target", "up")
    ):
        raise ValueError("Indexed camera view needs finite three-vector axes")
    torch = adapter.torch
    if backend == "embodichain":
        batch = adapter.cameras[camera_index]
        part = next(
            (
                (start, stop, camera)
                for start, stop, camera in batch.parts
                if start <= env_id < stop
            ),
            None,
        )
        if part is None:
            raise ValueError("Global environment index is absent from native chunks")
        start, stop, camera = part
        adapter._end_frame()
        # Existing frozen publication helpers preserve native flush completion
        # and counters. No generic SimulationManager or renderer method changes.
        adapter._pending_native_flush = True
        vectors = [
            torch.tensor([spec[key]], dtype=torch.float32)
            for key in ("eye", "target", "up")
        ]
        camera.look_at(*vectors, env_ids=[env_id - start])
        adapter._begin_frame()
        return {
            "api": "Camera.look_at(env_ids)",
            "global_env_id": env_id,
            "camera_index": camera_index,
            "chunk_start": start,
            "chunk_stop": stop,
            "sensor_index": env_id - start,
            "group_id": camera.group_id,
            "publication": "original adapter sync_render_state plus native render-update flush",
        }
    if backend != "isaaclab":
        raise ValueError("Unsupported frozen renderer for indexed routing proof")
    from scripts.benchmark.rendering.scenes import camera_world_pose

    camera_radius = math.sqrt(sum(float(v) ** 2 for v in spec["eye"]))
    if (
        adapter.num_envs > 1
        and camera_radius
        + adapter._scene_radius
        + adapter.scene["camera_model"]["far_m"]
        >= adapter._replica_spacing
    ):
        raise ValueError("Probe would violate frozen primary-ray replica bounds")
    eye, orientation = camera_world_pose(spec)
    offset = adapter._env_offsets[env_id]
    position = [float(eye[axis]) + float(offset[axis]) for axis in range(3)]
    sensor_id = env_id * len(views) + camera_index
    stream = adapter.wp.stream_from_torch(torch.cuda.current_stream(adapter.device))
    with adapter.wp.ScopedStream(stream):
        adapter.camera.set_world_poses(
            positions=torch.tensor(
                [position], dtype=torch.float32, device=adapter.device
            ),
            orientations=torch.tensor(
                [orientation], dtype=torch.float32, device=adapter.device
            ),
            env_ids=[sensor_id],
            convention="opengl",
        )
    adapter.sim.render_context.reset_transform_cadence()
    return {
        "api": "Camera.set_world_poses(env_ids)",
        "global_env_id": env_id,
        "camera_index": camera_index,
        "sensor_index": sensor_id,
        "sensor_index_semantics": "environment_then_camera; env_ids selects sensor rows",
        "publication": "original Warp/Torch stream and transform-cadence reset; no physics step",
    }


def _artifact(root: Path, path: Path, kind: str) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "kind": kind,
    }


def _difference(first: Any, second: Any, modality: str) -> dict[str, Any]:
    import numpy as np

    if modality == "seg":
        return {
            "value_change": float(np.mean(first != second)),
            "validity_change": 0.0,
            "value_unit": "different_pixel_fraction",
            "finite_population": int(first.size),
        }
    a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    valid_a, valid_b = np.isfinite(a), np.isfinite(b)
    common = valid_a & valid_b
    value = float(np.abs(a[common] - b[common]).mean()) if np.any(common) else 0.0
    if modality == "rgb":
        value /= 255
    return {
        "value_change": value,
        "validity_change": float(np.mean(valid_a != valid_b)),
        "value_unit": "m" if modality == "depth" else "1",
        "finite_population": int(common.sum()),
    }


def _outside(delta: Mapping[str, Any], threshold: Mapping[str, float]) -> bool:
    return any(
        delta[key] > threshold[key] for key in ("value_change", "validity_change")
    )


def _assess(
    baselines: list[dict[str, Any]],
    moved: list[dict[str, Any]],
    restored: list[dict[str, Any]],
    *,
    target: int,
    num_envs: int,
    camera_count: int,
    modalities: tuple[str, ...],
    noise_multiplier: float,
) -> dict[str, Any]:
    exposure_rows = []
    unexpected = []
    target_responsive = []
    target_restored = []
    for exposure in range(num_envs * camera_count):
        details = {}
        for modality in modalities:
            controls = [
                _difference(
                    baselines[i][modality][exposure],
                    baselines[j][modality][exposure],
                    modality,
                )
                for i in range(len(baselines))
                for j in range(i + 1, len(baselines))
            ]
            floor = 1 / 255 if modality == "rgb" else 1e-5
            threshold = {
                key: max(
                    floor if key == "value_change" else 1e-5,
                    noise_multiplier * max(item[key] for item in controls),
                )
                for key in ("value_change", "validity_change")
            }
            baseline = baselines[-1][modality][exposure]
            movement = [
                _difference(baseline, item[modality][exposure], modality)
                for item in moved
            ]
            recovery = [
                _difference(baseline, item[modality][exposure], modality)
                for item in restored
            ]
            returned = _difference(
                moved[-1][modality][exposure],
                restored[-1][modality][exposure],
                modality,
            )
            if exposure != target:
                changed = any(_outside(item, threshold) for item in movement + recovery)
                status = "failed" if changed else "passed"
                reason = (
                    "unaffected_exposure_changed_beyond_stationary_envelope"
                    if changed
                    else None
                )
                if changed:
                    unexpected.append({"batch_index": exposure, "modality": modality})
            else:
                responsive = _outside(movement[-1], threshold) and _outside(
                    returned, threshold
                )
                recovered = not _outside(recovery[-1], threshold)
                target_responsive.append(responsive)
                target_restored.append(recovered)
                status = "passed" if responsive and recovered else "inconclusive"
                reason = (
                    None
                    if status == "passed"
                    else (
                        "selected_modality_has_no_discriminating_response"
                        if not responsive
                        else "restored_target_outside_stationary_envelope"
                    )
                )
            details[modality] = {
                "status": status,
                "reason": reason,
                "stationary_pairs": controls,
                "threshold": threshold,
                "moved_from_baseline": movement,
                "restored_from_baseline": recovery,
                "moved_to_restored": returned,
            }
        exposure_rows.append(
            {
                "batch_index": exposure,
                "env_id": exposure // camera_count,
                "camera_index": exposure % camera_count,
                "role": "selected" if exposure == target else "unaffected",
                "modalities": details,
            }
        )
    status = (
        "failed"
        if unexpected
        else (
            "passed"
            if all(target_responsive) and all(target_restored)
            else "inconclusive"
        )
    )
    return {
        "status": status,
        "selected_batch_index": target,
        "unexpected_responses": unexpected,
        "selected_all_modalities_responsive": all(target_responsive),
        "selected_all_modalities_restored": all(target_restored),
        "exposures": exposure_rows,
    }


def run_routing_probe(
    adapter: Any,
    *,
    backend: str,
    num_envs: int,
    camera_specs: list[dict[str, Any]],
    modalities: tuple[str, ...],
    width: int,
    height: int,
    output: Path,
    binding: dict[str, Any],
    capture: Callable[[int], Any] | None = None,
    setter: Callable[[int, int, dict[str, Any]], Any] | None = None,
    stationary_captures: int = 3,
    settle_captures: int = 2,
    sample_edge: int = 64,
    persist_full: bool = False,
    eye_shift_m: float = 0.35,
    noise_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Probe representative input routes while checking every unaffected output.

    Run only outside all performance/resource and history-sensitive acquisition
    windows. ``camera_specs`` must describe the actual pre-probe state; every
    targeted override is restored through the indexed setter, including errors.
    Every baseline, transition/settling and restoration capture is retained.

    Args:
        adapter: Original frozen renderer instance, already at the chosen state.
        backend: Source backend name.
        num_envs: Full source environment count, never reduced for this proof.
        camera_specs: Actual active canonical look-at specs in output order.
        modalities: Exact source output modalities.
        width: Full output width.
        height: Full output height.
        output: Fresh directory for compact/full evidence and atomic results.
        binding: Verified source/run/config/profile/GPU binding from the driver.
        capture: Optional original completed-host CapturePacket callback.
        setter: Optional indexed pose callback for original-runtime integration.
        stationary_captures: Number of per-trial unperturbed noise controls.
        settle_captures: Additional captures after the first moved/restored frame.
        sample_edge: Maximum sampled image axis; all exposure rows are retained.
        persist_full: Also persist every full requested modality array.
        eye_shift_m: Prespecified arena-local x displacement, keeping target/up.
        noise_multiplier: Prespecified multiplier on the maximum stationary pair.

    Returns:
        Completed diagnostic with passed/failed/inconclusive routing status,
        or an explicit execution failure retaining samples and restoration errors.
        No FPS or cross-backend quality eligibility is produced.
    """
    missing = [key for key in _REQUIRED_BINDINGS if not binding.get(key)]
    if missing or binding.get("backend") != backend:
        raise ValueError(f"Incomplete or different frozen source binding: {missing}")
    for name, value, minimum in (
        ("num_envs", num_envs, 1),
        ("width", width, 1),
        ("height", height, 1),
        ("stationary_captures", stationary_captures, 2),
        ("settle_captures", settle_captures, 0),
        ("sample_edge", sample_edge, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (
        not camera_specs
        or len(set(modalities)) != len(modalities)
        or not modalities
        or set(modalities) - _DTYPES.keys()
    ):
        raise ValueError("Need nonempty fixed camera and modality populations")
    if (
        not math.isfinite(eye_shift_m)
        or eye_shift_m == 0
        or not math.isfinite(noise_multiplier)
        or noise_multiplier < 1
    ):
        raise ValueError("Invalid declared camera perturbation/noise policy")
    import numpy as np

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    camera_count = len(camera_specs)
    camera_specs = deepcopy(camera_specs)
    if capture is None:
        from scripts.benchmark.rendering.capture import capture_packet

        def capture(index: int) -> Any:
            return capture_packet(
                adapter,
                frame_index=index,
                exposure_count=num_envs * camera_count,
                delivery="host_ready",
            )

    if setter is None:

        def setter(env: int, camera: int, spec: dict[str, Any]) -> Any:
            return publish_indexed_camera(adapter, backend, env, camera, spec)

    anchors = sorted({0, num_envs // 2, num_envs - 1})
    policy = {
        "stationary_captures": stationary_captures,
        "moved_captures": 1 + settle_captures,
        "restored_captures": 1 + settle_captures,
        "sample_edge": sample_edge,
        "persist_full": persist_full,
        "eye_shift_m": eye_shift_m,
        "noise_multiplier": noise_multiplier,
        "unaffected_rule": "every recorded moved/restored sample in every requested modality stays within its own stationary envelope",
        "selected_rule": "each requested modality responds in the final moved sample and returns within the stationary envelope; insufficient contrast/settling is inconclusive",
        "noise_scope": "diagnostic control envelope, not a statistical confidence bound or image-quality threshold",
    }
    result = {
        "schema_version": 1,
        "kind": "indexed_camera_routing_proof",
        "status": "running",
        "routing_status": "inconclusive",
        "binding": deepcopy(binding),
        "binding_sha256": stable_hash(binding),
        "policy": policy,
        "policy_sha256": stable_hash(policy),
        "num_envs": num_envs,
        "cameras_per_env": camera_count,
        "modalities": list(modalities),
        "tested_env_ids": anchors,
        "expected_trials": len(anchors) * camera_count,
        "outside_measured_window": True,
        "outside_resource_window": True,
        "throughput_claim": False,
        "publication_eligible": False,
        "cross_environment_scope": (
            "representative_input_routes_all_unaffected_outputs"
            if num_envs > 1
            else "not_applicable_single_environment"
        ),
        "limitations": [
            "No indirect-light transport qualification.",
            "Permutations entirely among untested input environments remain untested.",
            "Selected routing response is assessed at the final held pose; native source timestamps and first-capture observation age are not established.",
            "Static identical pixels are valid; absent diagnostic contrast is inconclusive.",
            "Camera reconstruction/noise coupling can produce unexpected changes; failure does not by itself diagnose a broadcast implementation.",
        ],
        "trials": [],
        "artifacts": [],
        "capture_batches": 0,
        "captured_exposures": 0,
    }
    write_json(output / "routing_proof.json", result)
    stride_y, stride_x = max(1, math.ceil(height / sample_edge)), max(
        1, math.ceil(width / sample_edge)
    )

    def take(directory: Path, phase: str) -> tuple[dict[str, Any], dict[str, Any]]:
        ordinal = result["capture_batches"]
        packet = capture(ordinal)
        arrays = packet.arrays
        packet_metadata = packet.to_metadata()
        if set(arrays) != set(modalities):
            raise ValueError("Completed capture changed the requested modalities")
        sampled = {}
        shapes = {}
        for name, array in arrays.items():
            expected = (num_envs * camera_count, height, width)
            allowed = (
                (expected + (3,),)
                if name in ("rgb", "normals")
                else (expected, expected + (1,))
            )
            if (
                not isinstance(array, np.ndarray)
                or array.shape not in allowed
                or str(array.dtype) != _DTYPES[name]
            ):
                raise ValueError(f"Noncanonical completed host {name} shape/dtype")
            sampled[name] = array[:, ::stride_y, ::stride_x].copy()
            shapes[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        result["capture_batches"] += 1
        result["captured_exposures"] += num_envs * camera_count
        record = {
            "ordinal": ordinal,
            "phase": phase,
            "packet": packet_metadata,
            "full_shapes": shapes,
            "sample_stride_yx": [stride_y, stride_x],
            "sampled_shapes": {
                name: list(array.shape) for name, array in sampled.items()
            },
        }
        if persist_full:
            full_path = directory / f"{phase}_full.npz"
            np.savez_compressed(full_path, **arrays)
            record["full_arrays"] = _artifact(output, full_path, "full_routing_capture")
            result["artifacts"].append(record["full_arrays"])
        sample_path = directory / f"{phase}_sampled.npz"
        np.savez_compressed(sample_path, **sampled)
        record["sampled_arrays"] = _artifact(
            output, sample_path, "sampled_all_exposure_routing_capture"
        )
        result["artifacts"].append(record["sampled_arrays"])
        del packet
        return sampled, record

    stop = False
    for env in anchors:
        for camera_index, original in enumerate(camera_specs):
            if stop:
                break
            trial_index = len(result["trials"])
            directory = (
                output
                / f"trial_{trial_index:02d}_env_{env:04d}_camera_{camera_index:02d}"
            )
            directory.mkdir()
            changed = deepcopy(original)
            changed["eye"][0] += eye_shift_m
            trial = {
                "env_id": env,
                "camera_index": camera_index,
                "camera_name": original["name"],
                "original_spec": deepcopy(original),
                "perturbed_spec": changed,
                "observations": [],
                "errors": [],
                "restore_attempted": False,
                "restore_succeeded": False,
            }
            baselines, moved, restored = [], [], []
            mutation_attempted = False
            try:
                for index in range(stationary_captures):
                    arrays, record = take(directory, f"baseline_{index:02d}")
                    baselines.append(arrays)
                    trial["observations"].append(record)
                mutation_attempted = True
                trial["indexed_setter"] = setter(env, camera_index, changed)
                for index in range(1 + settle_captures):
                    arrays, record = take(directory, f"moved_{index:02d}")
                    moved.append(arrays)
                    trial["observations"].append(record)
            except Exception as error:
                trial["errors"].append(f"probe: {type(error).__name__}: {error}")
            finally:
                if mutation_attempted:
                    trial["restore_attempted"] = True
                    try:
                        trial["restore_setter"] = setter(env, camera_index, original)
                        trial["restore_succeeded"] = True
                    except Exception as error:
                        trial["errors"].append(
                            f"restore: {type(error).__name__}: {error}"
                        )
            if trial["restore_succeeded"]:
                try:
                    for index in range(1 + settle_captures):
                        arrays, record = take(directory, f"restored_{index:02d}")
                        restored.append(arrays)
                        trial["observations"].append(record)
                except Exception as error:
                    trial["errors"].append(
                        f"restored_capture: {type(error).__name__}: {error}"
                    )
            if trial["errors"]:
                trial["status"] = "error"
                stop = True
            else:
                trial.update(
                    _assess(
                        baselines,
                        moved,
                        restored,
                        target=env * camera_count + camera_index,
                        num_envs=num_envs,
                        camera_count=camera_count,
                        modalities=modalities,
                        noise_multiplier=noise_multiplier,
                    )
                )
            write_json(directory / "trial.json", trial)
            result["artifacts"].append(
                _artifact(output, directory / "trial.json", "indexed_routing_trial")
            )
            result["trials"].append(trial)
            # Release every trial's pixel buffers; subsequent trials retain only
            # checksummed files and numeric per-exposure decisions in memory.
            del baselines, moved, restored
            write_json(output / "routing_proof.json", result)
    states = [trial["status"] for trial in result["trials"]]
    result["status"] = "failed" if "error" in states else "completed"
    result["routing_status"] = (
        "failed"
        if "failed" in states
        else (
            "passed"
            if len(states) == result["expected_trials"]
            and all(state == "passed" for state in states)
            else "inconclusive"
        )
    )
    result["complete"] = (
        len(states) == result["expected_trials"] and "error" not in states
    )
    write_json(output / "routing_proof.json", result)
    return result


def publication_callback_wall(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive per-record publication host wall intervals before aggregation.

    Args:
        records: Measured raw capture records; callers must exclude warmup.

    Returns:
        Per-row intervals, median/nearest-rank P95 and missing/invalid reasons.
        The interval includes waits within prepare and a small capture envelope;
        it is neither CPU busy time nor a synchronized GPU phase measurement.
    """
    rows, missing, invalid = [], [], []
    for index, record in enumerate(records):
        stamps = record.get("timestamps_s", {})
        if not isinstance(stamps, Mapping):
            invalid.append(index)
            continue
        start, end = stamps.get("state_publish_started_s"), stamps.get(
            "state_publish_callback_returned_s"
        )
        if start is None or end is None:
            missing.append(index)
            continue
        if (
            any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (start, end)
            )
            or end < start
        ):
            invalid.append(index)
            continue
        rows.append(
            {
                "record_index": index,
                "frame_index": record.get("frame_index"),
                "elapsed_s": end - start,
            }
        )
    values = tuple(row["elapsed_s"] for row in rows)
    summary = Measurement(0, values, sum(values)) if values else None
    return {
        "metric": "state_publish_callback_wall_s",
        "clock": "capture_clock_monotonic_seconds",
        "scope": "host prepare interval including waits and small pre-dispatch envelope; not CPU busy/native GPU time",
        "raw": rows,
        "valid_records": len(rows),
        "expected_records": len(records),
        "missing_record_indices": missing,
        "invalid_record_indices": invalid,
        "p50_s": summary.p50_s if summary else None,
        "p95_s": summary.p95_s if summary else None,
        "total_s": sum(values) if values else None,
        "status": (
            "complete"
            if values and not missing and not invalid
            else "partial" if values else "missing"
        ),
        "aggregation": "per-record differences first, then per-process percentiles; never subtract aggregate percentiles",
    }
