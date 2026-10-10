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

"""Fresh untimed history acquisition using the original frozen renderer code.

Launch this file by absolute path, not as a package module: source pinning must
precede every benchmark/simulator import. Added D2H and IO are evidence capture,
not a throughput measurement, and never cause an additional render submission.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import sys
import traceback
from typing import Any

if __package__:
    from . import frozen_source as frozen
    from .history_contract import build_history_schedule, frame_history
else:
    import frozen_source as frozen
    from history_contract import build_history_schedule, frame_history

__all__ = ["capture_history", "main", "run_worker"]


def _persist(
    adapter: Any,
    packet: Any,
    original_worker: Any,
    request: dict[str, Any],
    output: Path,
    manifest: dict[str, Any],
    case: Any,
    scene: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any],
    exposure_index: int,
    moving_ids: list[int],
) -> dict[str, Any]:
    """Persist the already completed packet, using the source frame serializer."""
    extra = {
        "render_submissions": 0,
        "d2h_group_calls": 0,
        "d2h_calls": 0,
        "d2h_bytes": 0,
        "scope": "additional untimed quality persistence; outside original delivery boundary",
    }
    arrays = packet.arrays
    if case.delivery == "device_ready":
        arrays = adapter.to_host(packet.device_arrays)
        extra.update(
            d2h_group_calls=1,
            d2h_calls=len(arrays),
            d2h_bytes=sum(int(array.nbytes) for array in arrays.values()),
        )
    host_packet = replace(packet, arrays=arrays)
    start = len(manifest["frames"])
    views = [state["cameras"][name] for name in original_worker._camera_names(case)]
    original_worker._save_batch(
        output,
        manifest,
        host_packet,
        case,
        scene,
        config,
        request["original"]["backend"],
        state,
        views,
        moving_ids,
    )
    alias = frame_history(request["schedule"], exposure_index)
    for entry in manifest["frames"][start:]:
        path = output / entry["path"]
        record = frozen.read_json(path)
        record["metadata"]["history"] = alias
        record["metadata"]["history_acquisition"] = {
            "original_run_id": request["original"]["run_id"],
            "source_commit": request["source"]["commit"],
            "additional_quality_delivery": extra,
            "performance_claim": False,
        }
        frozen.write_json(path, record)
        entry["sha256"] = frozen.sha256(path)
        entry["history"] = alias
    frozen.write_json(output / "frames.json", manifest)
    return extra


def capture_history(
    adapter: Any,
    original_worker: Any,
    original_capture: Any,
    request: dict[str, Any],
    config: dict[str, Any],
    scene: dict[str, Any],
    replay: dict[str, Any],
    case: Any,
    output: Path,
    sampler: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay the original prelude/loop while tapping existing completed packets.

    Args:
        adapter: Adapter constructed by the original frozen factory.
        original_worker: Imported original formal_worker module.
        original_capture: Imported original capture module.
        request: Hash-bound follow-up request with the original exposure schedule.
        config: Original worker config with only input paths rebased.
        scene: Original scene manifest.
        replay: Original immutable state replay.
        case: Original RenderCase instance.
        output: Fresh acquisition directory for frames/events.
        sampler: Original ResourceSampler used for matching phase hooks.

    Returns:
        Frame manifest, complete exposure events, and original freshness result.
        No derived FPS/latency statistics from this instrumented loop are exported.
    """
    original_packet = original_capture.capture_packet
    worker_packet = original_worker.capture_packet
    events = {"freshness": [], "warmup": [], "measured": []}
    phase = "freshness"
    last_state = original_worker.frame_at(replay, 0)
    moving_ids = original_worker._moving_ids(scene, replay)
    manifest = {
        "schema_version": 1,
        "backend": request["original"]["backend"],
        "case_id": case.case_id,
        **{
            key: config[key]
            for key in ("scene_sha256", "view_sha256", "profile_sha256")
        },
        "modalities": list(case.modalities),
        "num_envs": case.num_envs,
        "cameras_per_env": case.cameras_per_env,
        "representative_env_ids": sorted({0, case.num_envs // 2, case.num_envs - 1}),
        "capture_batches": 0,
        "captured_exposures": 0,
        "saved_exposures": 0,
        "batch_order": "environment_then_camera",
        "frames": [],
        "quality_status": "not_qualified",
        "physics_steps": 0,
        "fps": replay["fps"],
        "history_schedule_sha256": request["schedule_sha256"],
        "performance_claim": False,
    }

    def record_packet(*args: Any, **kwargs: Any) -> Any:
        # Delegation preserves all original delivery/synchronization behavior.
        packet = original_packet(*args, **kwargs)
        packet = replace(
            packet, adapter_evidence=original_capture._adapter_capture_evidence(adapter)
        )
        index = len(events[phase])
        entry = {"phase": phase, "index": index, "packet": packet.to_metadata()}
        if phase == "measured" or (
            phase == "warmup" and index == config["warmup_frames"] - 1
        ):
            exposure = index if phase == "measured" else -1
            entry["additional_quality_delivery"] = _persist(
                adapter,
                packet,
                original_worker,
                request,
                output,
                manifest,
                case,
                scene,
                config,
                last_state,
                exposure,
                moving_ids,
            )
        events[phase].append(entry)
        return packet

    def prepare(index: int) -> None:
        nonlocal last_state
        last_state = original_worker.frame_at(replay, index % replay["frame_count"])
        adapter.apply_state(last_state)

    def measurement_start() -> None:
        nonlocal phase
        adapter.synchronize()
        original_worker._allocator(adapter, reset=True)
        sampler.mark_phase("measurement")
        phase = "measured"

    original_capture.capture_packet = record_packet
    original_worker.capture_packet = record_packet
    try:
        sampler.mark_phase("freshness_probe")
        freshness = original_worker._freshness_probe(adapter, case, last_state, output)
        frozen.write_json(output / "freshness_probe.json", freshness)
        if not freshness.get("passed"):
            raise RuntimeError(
                "Original freshness prefix failed in the fresh history worker"
            )
        # Match the original run_worker call after its four-capture probe.
        adapter.apply_state(last_state)
        sampler.mark_phase("warmup")
        phase = "warmup"
        # The original loop performs W warmups, resets measured indices to zero,
        # then performs K captures. Its timing metrics are deliberately discarded.
        original_capture.measure_rendering(
            adapter,
            warmup=config["warmup_frames"],
            iterations=config["measured_frames"],
            num_envs=case.num_envs,
            cameras_per_env=case.cameras_per_env,
            width=case.width,
            height=case.height,
            delivery=case.delivery,
            prepare=prepare,
            diagnostic=case.diagnostic,
            on_measurement_start=measurement_start,
        )
        original_worker._allocator(adapter, reset=False)
        sampler.mark_phase("measurement_complete")
        return manifest, events, freshness
    finally:
        original_capture.capture_packet = original_packet
        original_worker.capture_packet = worker_packet
        frozen.write_json(output / "capture_events.json", events)


def run_worker(request_path: Path, output: Path, gpu_id: int) -> dict[str, Any]:
    """Acquire history in a new process and checkpoint before native close.

    Args:
        request_path: Producer request at the portable proof root.
        output: Worker capture subdirectory inside that proof root.
        gpu_id: Original selected physical GPU index.

    Returns:
        Execution result with no throughput claim; history.json is saved before
        close because SimulationApp.close may terminate the interpreter.
    """
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    root = request_path.parent
    if output == root or not output.is_relative_to(root):
        raise ValueError("Capture output must be a new subdirectory of the proof root")
    output.mkdir(parents=True, exist_ok=True)
    request = frozen.read_json(request_path)
    original, source = request["original"], request["source"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "backend": original["backend"],
        "original_run_id": original["run_id"],
        "quality_status": "not_qualified",
        "performance_claim": False,
        "metrics": {},
        "evidence": [],
        "pid": os.getpid(),
        "hardware_id": None,
    }
    history = {
        "schema_version": 1,
        "kind": "timed_render_history/v1",
        "status": "failed",
        "performance_claim": False,
        **{
            key: request[key]
            for key in (
                "original",
                "schedule",
                "schedule_sha256",
                "history_quality_protocol",
                "history_quality_requirements",
            )
        },
        "producer_artifacts": [
            *request["producer_artifacts"],
            frozen.artifact(root, request_path, "history_request", role="request"),
        ],
        "acquisition": {
            "fresh_process": True,
            "source_commit": source["commit"],
            "pid": os.getpid(),
        },
    }
    adapter = sampler = worker = None
    try:
        roles = {
            record["role"]: frozen._inside(root, record["path"])
            for record in original["artifacts"]
        }
        for record in original["artifacts"]:
            if frozen.sha256(root / record["path"]) != record["sha256"]:
                raise ValueError("Frozen original input changed before acquisition")
        config = frozen.read_json(roles["worker_config"])
        schedule = build_history_schedule(
            config,
            frozen.read_json(roles["replay"]),
            frozen.read_json(roles["capture"]),
        )
        if (
            request.get("kind") != "timed_render_history_request/v1"
            or request.get("schedule") != schedule
            or request.get("schedule_sha256") != frozen.stable_hash(schedule)
        ):
            raise ValueError(
                "History request differs from the copied original schedule"
            )
        if original.get("source_commit") != source.get("commit") or original.get(
            "source_files"
        ) != source.get("source_files"):
            raise ValueError("History request source identity mismatch")
        for evidence in request["producer_artifacts"]:
            if (
                frozen.sha256(frozen._inside(root, evidence["path"]))
                != evidence["sha256"]
            ):
                raise ValueError("Copied history producer source changed")
        frozen.verify_source_worktree(
            {"row": frozen.read_json(roles["result"]), "config": config},
            Path(source["worktree"]),
            source["commit"],
        )
        history["acquisition"]["runtime_versions"] = frozen.runtime_versions(original)
        frozen.pin_source_imports(source)
        from scripts.benchmark.rendering import formal_worker as worker
        from scripts.benchmark.rendering import capture

        config["scene_path"] = str(roles["scene"])
        config["replay_path"] = str(roles["replay"])
        case_values = dict(config["case"])
        case_values["modalities"] = tuple(case_values["modalities"])
        case = worker.RenderCase(**case_values)
        scene, replay = worker._preflight(config, case)
        if replay is None:
            raise ValueError("History acquisition requires the original moving replay")
        frozen.write_json(output / "worker_config.json", config)
        sampler = worker.ResourceSampler(gpu_id=gpu_id)
        with sampler:
            identity = sampler.summary()["gpu"]
            if (
                identity.get("uuid") != original["hardware_id"]
                or identity.get("driver") != original["gpu_driver"]
            ):
                raise RuntimeError(
                    "Selected GPU UUID/driver differs from the timed original"
                )
            adapter = worker._create_adapter(original["backend"], config, case, gpu_id)
            if adapter.metadata.get("physics_enabled") is not False:
                raise ValueError("Original adapter must explicitly disable physics")
            history["acquisition"]["runtime_versions"] = frozen.runtime_versions(
                original, getattr(adapter, "torch", None)
            )
            frozen.verify_imported_sources(source)
            _, events, _ = capture_history(
                adapter,
                worker,
                capture,
                request,
                config,
                scene,
                replay,
                case,
                output,
                sampler,
            )
            transfers = [
                event["additional_quality_delivery"]
                for phase in events.values()
                for event in phase
                if "additional_quality_delivery" in event
            ]
            history["acquisition"]["persistence"] = {
                "extra_render_submissions": sum(
                    item["render_submissions"] for item in transfers
                ),
                "additional_d2h_group_calls": sum(
                    item["d2h_group_calls"] for item in transfers
                ),
                "additional_d2h_calls": sum(item["d2h_calls"] for item in transfers),
                "additional_d2h_bytes": sum(item["d2h_bytes"] for item in transfers),
                "mode": "inline after original packet completion; no extra exposures",
                "wall_clock_spacing": "changed by explicit quality readback and file IO; no timing or FPS claim",
            }
        if hasattr(adapter, "refresh_metadata"):
            adapter.refresh_metadata()
        renderer = deepcopy(adapter.metadata)
        original_result = frozen.read_json(roles["result"])
        if frozen.renderer_control_signature(
            renderer, original["backend"]
        ) != frozen.renderer_control_signature(
            original_result["renderer"], original["backend"]
        ):
            raise RuntimeError("Fresh renderer controls differ from original run")
        frozen.write_json(output / "renderer.json", renderer)
        history["acquisition"]["imported_sources"] = frozen.verify_imported_sources(
            source
        )
        history["acquisition"]["isaaclab_commit"] = source.get("isaaclab_commit")
        for key, filename, kind in (
            ("renderer", "renderer.json", "history_renderer"),
            ("freshness", "freshness_probe.json", "history_freshness"),
            ("capture_events", "capture_events.json", "history_capture_events"),
            ("frames_manifest", "frames.json", "history_frame_manifest"),
        ):
            history[key] = frozen.artifact(root, output / filename, kind)
            result["evidence"].append(filename)
        result["status"] = history["status"] = "completed"
    except Exception as exc:
        result.update(
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            failure_reason_code="history_acquisition_failed",
        )
        history["failure"] = {
            "reason": result["error"],
            "traceback": result["traceback"],
        }
    finally:
        if sampler is not None:
            resources = sampler.summary()
            frozen.write_json(output / "resources.json", resources)
            history["acquisition"].update(
                hardware_id=resources["gpu"].get("uuid"),
                gpu_driver=resources["gpu"].get("driver"),
            )
            result["hardware_id"] = resources["gpu"].get("uuid")
            result["evidence"].append("resources.json")
        if worker is not None:
            worker._checkpoint_renderer_validation(output, result)
            if result["status"] != "completed":
                history["status"] = "failed"
        result["cleanup_status"] = (
            "pending_native_close" if adapter is not None else "not_required"
        )
        frozen.write_json(root / "history.json", history)
        frozen.write_json(output / "result.json", result)
        if adapter is not None:
            try:
                adapter.close()
                result["cleanup_status"] = "completed"
            except Exception as exc:
                result.update(
                    status="failed",
                    cleanup_status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    failure_reason_code="cleanup_failed",
                )
                history.update(status="failed", failure={"reason": result["error"]})
            frozen.write_json(root / "history.json", history)
            frozen.write_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the absolute-script worker interface without preloading benchmark modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args(argv)
    return int(
        run_worker(args.request, args.output, args.gpu_id)["status"] != "completed"
    )


if __name__ == "__main__":
    raise SystemExit(main())
