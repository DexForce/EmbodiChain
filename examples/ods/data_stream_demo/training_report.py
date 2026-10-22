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

"""Render fixed-budget visual training measurements, retaining failed runs.

Run: ``python -m examples.ods.data_stream_demo.training_report OUTPUT_ROOT``.
Legacy delivery-only summaries are deliberately excluded from this report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

__all__ = ["write_report"]

_ARMS = ("offline8", "offline32", "offline128", "ods_overlap", "ods_serial")
_LABELS = {
    "offline8": "Offline-8",
    "offline32": "Offline-32",
    "offline128": "Offline-128",
    "ods_overlap": "ODS overlap",
    "ods_serial": "ODS serial replay",
}
_COLORS = {
    "offline8": "#4c78a8",
    "offline32": "#f58518",
    "offline128": "#b279a2",
    "ods_overlap": "#159b80",
    "ods_serial": "#c65656",
}
_MARKERS = {
    "offline8": "o",
    "offline32": "s",
    "offline128": "^",
    "ods_overlap": "D",
    "ods_serial": "X",
}
_VISIONS = ("single", "dual")
_METRICS = (
    "total_seconds",
    "training_complete_seconds",
    "post_training_drain_seconds",
    "first_update_seconds",
    "published_sources",
    "trained_sources",
    "effective_sources",
    "initial_xy_occupied_cells",
    "anchor_xyz_occupied_voxels",
    "unique_anchors",
    "unique_target_frames",
    "target_duplicate_fraction",
    "max_source_trained_chunks",
    "dataset_bytes",
    "checkpoint_bytes",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        else None
    )


def _arm(config: dict[str, Any]) -> str:
    arm = config.get("arm")
    if arm == "offline":
        return f"offline{config.get('offline_episodes')}"
    return str(arm)


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(row)
    return records


def _load(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("summary.json")) if root.is_dir() else [root]
    runs = []
    for path in paths:
        summary = _json(path)
        metrics = summary.get("training_metrics", summary.get("metrics", {}))
        config = summary.get("config", {})
        if (
            not isinstance(metrics, dict)
            or metrics.get("schema") != "ods-training-metrics/v1"
        ):
            # A child can fail before constructing its metric recorder. Keep
            # these training failures; do not mix legacy delivery summaries.
            if config.get("arm") not in {"offline", "ods_overlap", "ods_serial"}:
                continue
            metrics = {}
        arm = _arm(config)
        engineering_offline = (
            config.get("arm") == "offline"
            and type(config.get("offline_episodes")) is int
            and config["offline_episodes"] > 0
            and config.get("phase") in {"single", "smoke", "calibration"}
        )
        if (
            arm not in _ARMS
            and not engineering_offline
            or config.get("vision") not in _VISIONS
        ):
            raise ValueError(f"Unknown visual training arm/vision in {path}")
        status = summary.get("status")
        if status not in {"complete", "failed", "unsupported"}:
            raise ValueError(f"Unknown training status in {path}")
        runs.append(
            {
                "path": str(path.resolve()),
                "summary": summary,
                "config": config,
                "metrics": metrics,
                "arm": arm,
                "events": _events(path.parent / "training-events.jsonl"),
            }
        )
    if not runs:
        raise ValueError(f"No fixed-budget visual training summaries found in {root}")
    cells = Counter(
        (run["arm"], run["config"]["vision"], str(run["config"].get("seed")))
        for run in runs
        if not _qualification(run)
    )
    for run in runs:
        run["duplicate_cell"] = (
            cells[(run["arm"], run["config"]["vision"], str(run["config"].get("seed")))]
            > 1
        )
    return sorted(
        runs,
        key=lambda run: (
            run["config"]["vision"],
            run["arm"],
            str(run["config"].get("seed")),
            run["path"],
        ),
    )


def _qualification(run: dict[str, Any]) -> list[str]:
    summary, metrics, config = run["summary"], run["metrics"], run["config"]
    reasons = []
    if config.get("phase") == "pilot" and config.get("prefetch_policy") != "budget":
        reasons.append(
            "pilot primary comparisons require explicit budget prefetch; eager is a separate diagnostic"
        )
    if run.get("duplicate_cell"):
        reasons.append("multiple qualified attempts for the same arm/vision/seed")
    if summary["status"] != "complete":
        reasons.append(f"status={summary['status']}")
    if (
        summary.get("qualification_passed", metrics.get("qualification_passed"))
        is not True
    ):
        reasons.append("no explicit qualification pass")
    updates = config.get("updates")
    batch_size = config.get("batch_size", metrics.get("batch_size"))
    future = config.get("future", metrics.get("future"))
    if any(
        type(value) is not int or value < 1 for value in (updates, batch_size, future)
    ):
        reasons.append("missing positive training budget")
    elif (
        metrics.get("completed_updates") != updates
        or metrics.get("trained_chunks") != updates * batch_size
        or metrics.get("valid_target_occurrences") != updates * batch_size * future
    ):
        reasons.append("incomplete or mismatched training work")
    total = _number(metrics.get("total_seconds"))
    train = _number(metrics.get("training_complete_seconds"))
    if total is None or train is None or not (total >= train > 0):
        reasons.append("invalid completion timing")
    sources = metrics.get("episodes", [])
    if not isinstance(sources, list) or sum(
        row.get("trained_chunks", 0) for row in sources
    ) != metrics.get("trained_chunks"):
        reasons.append("source totals disagree with completed work")
    if metrics.get("training_cap_violations") != 0:
        reasons.append("training cap audit absent or failed")
    actual = (
        config.get("worker") is True
        or config.get("phase") in {"single", "smoke", "calibration", "report", "pilot"}
        or "device_supervision" in summary
        or "supervisor_total_seconds" in summary
        or any(
            key in metrics
            for key in (
                "initial_model_sha256",
                "normalization_sha256",
                "gpu_step_milliseconds",
                "engine_stats",
                "timeline_contract",
            )
        )
    )
    if actual and summary["status"] == "complete":
        _, evidence_errors = _fairness(run)
        reasons.extend(f"fairness evidence: {error}" for error in evidence_errors)
        audit = summary.get("post_audit")
        if (
            summary.get("post_audit_status") != "complete"
            or not isinstance(audit, dict)
            or audit.get("passed") is not True
        ):
            reasons.append("actual run lacks a completed passed full corpus audit")
        expected = summary.get("requested_evaluation_scenes")
        evaluation = summary.get("evaluation")
        if (
            (type(expected) is not int or expected < 0)
            and config.get("phase") not in {"report", "pilot"}
            and isinstance(evaluation, dict)
        ):
            # Older smoke records predate the persisted request. Their actual
            # completed evaluation is diagnostic evidence, never formal pairing.
            expected = evaluation.get("scenes")
        if type(expected) is not int or expected < 0:
            reasons.append("missing actual requested evaluation scene count")
        elif expected == 0:
            if summary.get("evaluation_status") != "not_requested":
                reasons.append(
                    "zero-scene evaluation lacks explicit not_requested status"
                )
        elif not (
            summary.get("evaluation_status") == "complete"
            and isinstance(evaluation, dict)
            and evaluation.get("passed") is True
            and evaluation.get("closed") is True
            and evaluation.get("scenes") == expected
            and len(evaluation.get("records", [])) == expected
        ):
            reasons.append("requested evaluation is pending, failed or incomplete")
    if actual and summary["status"] == "complete":
        reasons.extend(_new_formal_contract_errors(run))
    return reasons


def _qualified(run: dict[str, Any]) -> bool:
    return not _qualification(run)


def _group(runs: list[dict[str, Any]], arm: str, vision: str) -> list[dict[str, Any]]:
    return [
        run
        for run in runs
        if run["arm"] == arm and run["config"]["vision"] == vision and _qualified(run)
    ]


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        vision: {
            arm: {
                name: _stats(
                    [
                        value
                        for run in _group(runs, arm, vision)
                        if (value := _number(run["metrics"].get(name))) is not None
                    ]
                )
                for name in _METRICS
            }
            for arm in _ARMS
        }
        for vision in _VISIONS
    }


def _allocation(value: Any) -> dict[str, Any]:
    """Normalize assigned roles, preserving exact devices and CPU core IDs."""
    if not isinstance(value, dict):
        raise ValueError("missing assigned primary GPU pair")
    uuids = [value.get(role, {}).get("uuid") for role in ("simulation", "training")]
    cpus = value.get("cpu_affinity")
    if (
        any(not isinstance(uuid, str) or not uuid.startswith("GPU-") for uuid in uuids)
        or uuids[0] == uuids[1]
        or not isinstance(cpus, list)
        or len(cpus) != 8
        or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
        or len(set(cpus)) != len(cpus)
    ):
        raise ValueError(
            "expected two distinct role UUIDs and eight distinct CPU cores"
        )
    return {
        "simulation_uuid": uuids[0],
        "training_uuid": uuids[1],
        "cpu_affinity": sorted(cpus),
        "cpu_cores": len(cpus),
    }


def _launch_path(run: dict[str, Any]) -> Path:
    directory = Path(run["path"]).parent
    for root in (directory, *directory.parents):
        relative = directory.relative_to(root)
        candidate = (
            root
            / "_supervision"
            / ("single" if relative == Path(".") else relative)
            / "training"
            / "launch.json"
        )
        if candidate.is_file():
            return candidate
    raise ValueError("missing supervisor training launch.json")


def _fairness(run: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Check actual run evidence before allowing any comparative time ratio.

    Legacy CPU metric fixtures may still render and retain core qualification,
    but missing GPU/hash/supervisor evidence never produces a speedup claim.
    """
    metrics, config, summary = run["metrics"], run["config"], run["summary"]
    evidence: dict[str, Any] = {}
    if type(summary.get("requested_evaluation_scenes")) is int:
        evidence["requested_evaluation_scenes"] = summary["requested_evaluation_scenes"]
    reasons = []
    if "prefetch_policy" in config:
        if config["prefetch_policy"] not in {"budget", "eager"}:
            reasons.append("unknown prefetch policy")
        evidence["prefetch_policy"] = config["prefetch_policy"]
    for key in ("initial_model_sha256", "normalization_sha256"):
        value = metrics.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            reasons.append(f"missing or malformed {key}")
        else:
            evidence[key] = value
    budget = {
        "updates": config.get("updates"),
        "batch_size": config.get("batch_size"),
        "history": config.get("history_length", config.get("history")),
        "future": config.get("prediction_horizon", config.get("future")),
    }
    if any(type(value) is not int or value <= 0 for value in budget.values()):
        reasons.append("missing positive declared U/B/H/F budget")
    elif (
        metrics.get("completed_updates") != budget["updates"]
        or metrics.get("batch_size") != budget["batch_size"]
        or metrics.get("history") != budget["history"]
        or metrics.get("future") != budget["future"]
        or metrics.get("valid_target_occurrences")
        != budget["updates"] * budget["batch_size"] * budget["future"]
    ):
        reasons.append("declared U/B/H/F differs from completed training")
    else:
        evidence["training_budget"] = budget
    try:
        initial = _json(Path(run["path"]).with_name("initial-contract.json"))
        if any(
            initial.get(contract_key) != evidence.get(metric_key)
            for contract_key, metric_key in (
                ("model_sha256", "initial_model_sha256"),
                ("normalization_sha256", "normalization_sha256"),
            )
        ):
            raise ValueError(
                "initial contract disagrees with model/normalization hashes"
            )
        rows = initial.get("initial_episodes", [])
        prefix = [
            {key: row.get(key) for key in ("ordinal", "length", "numeric_sha256")}
            for row in rows
        ]
        if (
            type(config.get("buffer_size")) is not int
            or len(prefix) != config["buffer_size"]
            or [row["ordinal"] for row in prefix] != list(range(len(prefix)))
            or any(
                type(row["length"]) is not int
                or row["length"] <= 0
                or not isinstance(row["numeric_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", row["numeric_sha256"]) is None
                for row in prefix
            )
        ):
            raise ValueError("missing or malformed common initial C numeric prefix")
        cameras = initial.get("camera_order")
        if (
            not isinstance(cameras, list)
            or len(cameras) != (1 if config["vision"] == "single" else 2)
            or len(set(cameras)) != len(cameras)
        ):
            raise ValueError("missing or malformed initial camera order")
        expected = {
            "updates": metrics.get("completed_updates"),
            "chunks": metrics.get("trained_chunks"),
            "targets": metrics.get("valid_target_occurrences"),
        }
        if initial.get("training_budget") != expected:
            raise ValueError("initial contract and completed work disagree")
        evidence.update(initial_numeric_prefix=prefix, camera_order=cameras)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        reasons.append(f"initial contract evidence: {error}")
    try:
        supervision = summary.get("device_supervision", {})
        assigned = _allocation(supervision.get("primary_pair"))
        launch_path = _launch_path(run)
        launch = _json(launch_path)
        if (
            _allocation(launch.get("pair")) != assigned
            or sorted(launch.get("cpu_affinity", [])) != assigned["cpu_affinity"]
        ):
            raise ValueError("summary and launch GPU/CPU assignments disagree")
        arguments = launch.get("command", [])
        expected_flags = {
            "--updates": budget["updates"],
            "--batch-size": budget["batch_size"],
            "--history-length": budget["history"],
            "--prediction-horizon": budget["future"],
            "--sim-device": "cuda:0",
            "--train-device": "cuda:1",
            "--phase": config.get("phase"),
            "--seed": config.get("seed"),
            "--vision": config.get("vision"),
            "--arm": config.get("arm"),
        }
        if "prefetch_policy" in config:
            expected_flags["--prefetch-policy"] = config["prefetch_policy"]
        for flag, value in expected_flags.items():
            if (
                not isinstance(arguments, list)
                or arguments.count(flag) != 1
                or arguments.index(flag) + 1 >= len(arguments)
                or arguments[arguments.index(flag) + 1] != str(value)
            ):
                raise ValueError(f"launch command disagrees with declared {flag}")
        environment = launch.get("environment", {})
        if (
            environment.get("CUDA_VISIBLE_DEVICES")
            != assigned["simulation_uuid"] + "," + assigned["training_uuid"]
            or environment.get("ODS_GPU_UUID") != assigned["simulation_uuid"]
            or any(
                environment.get(key) != "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            )
        ):
            raise ValueError("launch UUID mask or CPU thread budget disagrees")
        observed_cpu = (metrics.get("resources") or {}).get("cpu_utilization") or {}
        if observed_cpu and (
            observed_cpu.get("assigned_cpu_count") != assigned["cpu_cores"]
            or sorted(observed_cpu.get("assigned_cpu_affinity", []))
            != assigned["cpu_affinity"]
        ):
            raise ValueError("observed CPU affinity differs from supervisor assignment")
        guard = supervision.get("guard", {})
        if (
            type(guard.get("samples")) is not int
            or guard["samples"] <= 0
            or guard.get("unexpected_contexts") != []
            or not {assigned["simulation_uuid"], assigned["training_uuid"]}.issubset(
                guard.get("seen_primary_uuids", [])
            )
        ):
            raise ValueError("missing or failed owned-device placement audit")
        process = _json(Path(run["path"]).with_name("process.json"))
        if (
            process.get("stage") != "training"
            or process.get("returncode") != 0
            or any(
                process.get(key) is not False
                for key in ("timeout", "cancelled", "leaked_process_group")
            )
            or process.get("device_guard") != guard
            or launch.get("timed_pipeline") is not True
        ):
            raise ValueError("missing or inconsistent normal-exit process audit")
        total = _number(metrics.get("total_seconds"))
        if total is None or any(
            (number := _number(value)) is None
            or not math.isclose(number, total, rel_tol=1e-9, abs_tol=1e-6)
            for value in (
                summary.get("supervisor_total_seconds"),
                process.get("elapsed_seconds"),
            )
        ):
            raise ValueError("pipeline total differs from supervisor process timing")
        origin = metrics.get("origin_monotonic_ns")
        if (
            type(origin) is not int
            or origin <= 0
            or origin != launch.get("process_launch_monotonic_ns")
            or origin != process.get("process_launch_monotonic_ns")
            or str(origin) != environment.get("ODS_PROCESS_LAUNCH_MONOTONIC_NS")
        ):
            raise ValueError("pipeline origin differs from supervisor launch clock")
        auxiliary = supervision.get("aux_graphics_scope")
        auxiliary_keys = ("uuid", "scope", "max_per_pid_bytes", "idle_required")
        aux = (
            {key: auxiliary.get(key) for key in auxiliary_keys}
            if isinstance(auxiliary, dict)
            else None
        )
        launch_auxiliary = launch.get("aux_graphics_scope")
        launch_aux = (
            {key: launch_auxiliary.get(key) for key in auxiliary_keys}
            if isinstance(launch_auxiliary, dict)
            else None
        )
        if aux != launch_aux:
            raise ValueError("summary and launch auxiliary graphics scopes disagree")
        evidence.update(
            allocation=assigned,
            auxiliary_graphics=aux,
            launch_path=str(launch_path),
            normal_exit_audited=True,
        )
    except (OSError, ValueError, TypeError, AttributeError) as error:
        reasons.append(f"resource/timing evidence: {error}")
    return evidence, reasons


def _new_formal_contract_errors(run: dict[str, Any]) -> list[str]:
    """Require chronological worker services and full video QA in new formal runs.

    Historical smoke/calibration/single engineering inputs retain their original
    sparse-audit scope. Superseded formal inputs lack the new contract and may
    be inspected as diagnostics, but cannot enter the refreshed formal claims.
    """
    if run["config"].get("phase") not in {"report", "pilot"}:
        return []
    metrics, summary = run["metrics"], run["summary"]
    errors = []
    if metrics.get("timeline_contract") != "sdk-worker-service/v1":
        errors.append("new formal run lacks sdk-worker-service/v1 timeline contract")
    total = _number(metrics.get("total_seconds"))
    intervals = metrics.get("intervals", [])
    recording = [row for row in intervals if row.get("name") == "sdk_recording"]
    decoding = [row for row in intervals if row.get("name") == "sdk_decode"]
    for row in recording + decoding:
        start, end = _number(row.get("start_seconds")), _number(row.get("end_seconds"))
        if (
            start is None
            or end is None
            or total is None
            or not 0 <= start < end <= total + 1e-6
            or type(row.get("worker_pid")) is not int
            or row["worker_pid"] <= 0
        ):
            errors.append("invalid chronological SDK worker service interval")
            break
    phases = Counter(row.get("sdk_phase") for row in recording)
    sources = metrics.get("episodes", [])
    published = metrics.get("published_sources")
    if (
        type(published) is not int
        or published < 1
        or not isinstance(sources, list)
        or type(metrics.get("completed_updates")) is not int
        or type(metrics.get("batch_size")) is not int
    ):
        errors.append(
            "new formal run lacks valid source/update counts for worker evidence"
        )
        return errors
    if (
        phases
        != Counter(
            {"sdk_create": 1, "sdk_episode_write_encode": published, "sdk_finalize": 1}
        )
        or len(recording) != (published or 0) + 2
    ):
        errors.append(
            "SDK recording service lacks create, every accepted write, or finalize"
        )
    writes = [
        row for row in recording if row.get("sdk_phase") == "sdk_episode_write_encode"
    ]

    def identity(value: Any) -> str:
        return (
            value
            if isinstance(value, str)
            else (
                f"{value[0]}:{value[1]}"
                if isinstance(value, (list, tuple)) and len(value) == 2
                else "invalid"
            )
        )

    expected_sources = {row.get("source_id"): row.get("ordinal") for row in sources}
    actual_sources = {
        identity(row.get("source_id")): row.get("episode_ordinal") for row in writes
    }
    if actual_sources != expected_sources or len(writes) != len(actual_sources):
        errors.append(
            "SDK write spans do not cover the exact immutable accepted sources"
        )
    if any(row.get("status", "complete") != "complete" for row in recording):
        errors.append("SDK recording worker reported failure")
    if run["config"].get("arm") == "offline":
        item_counts: Counter[int] = Counter()
        for row in decoding:
            update, count = row.get("batch_update"), row.get("item_count")
            if type(update) is not int or type(count) is not int or count <= 0:
                errors.append("invalid SDK temporal fetch update/item attribution")
                break
            item_counts[update] += count
        expected_counts = Counter(
            {
                i: metrics.get("batch_size")
                for i in range(1, metrics.get("completed_updates", 0) + 1)
            }
        )
        if not decoding or item_counts != expected_counts:
            errors.append(
                "SDK temporal fetch spans do not cover every delivered training batch"
            )
    audit = summary.get("post_audit") or {}
    receipt = audit.get("full_video_integrity")
    sidecar = Path(run["path"]).with_name("video-integrity.json")
    try:
        saved = _json(sidecar) if sidecar.is_file() else None
        if receipt is not None and saved is not None and receipt != saved:
            errors.append("inline and sidecar full-video receipts disagree")
        receipt = receipt if isinstance(receipt, dict) else saved
        if not isinstance(receipt, dict):
            raise ValueError("missing revisioned full-video-integrity receipt")
        cameras = _json(Path(run["path"]).with_name("initial-contract.json"))[
            "camera_order"
        ]
        expected_frames = sum(row["frame_count"] for row in sources)
        expected_cameras = {"observation.images." + camera for camera in cameras}
        expected_decoded = {camera: expected_frames for camera in expected_cameras}
        if not (
            receipt.get("revision") == "full-video-integrity/v1"
            and receipt.get("passed") is True
            and receipt.get("all_saved_video_frames_decoded") is True
            and receipt.get("expected_episodes") == published
            and receipt.get("expected_frames_per_camera") == expected_frames
            and receipt.get("decoded_frames_per_camera") == expected_decoded
            and receipt.get("total_decoded_camera_frames")
            == expected_frames * len(cameras)
        ):
            raise ValueError(
                "full-video receipt counts/status do not cover the saved corpus"
            )
        verifier = receipt.get("verifier_sha256")
        metadata = receipt.get("metadata_sha256")
        valid_sha = (
            lambda value: isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        )
        if (
            not valid_sha(verifier)
            or not isinstance(metadata, dict)
            or not metadata
            or not all(valid_sha(value) for value in metadata.values())
        ):
            raise ValueError("full-video verifier/metadata fingerprints are missing")
        files = receipt.get("files")
        file_frames: Counter[str] = Counter()
        episode_coverage: dict[str, set[int]] = {
            camera: set() for camera in expected_cameras
        }
        if not isinstance(files, list) or not files:
            raise ValueError("full-video receipt has no per-file checks")
        for row in files:
            if not (
                row.get("camera") in expected_cameras
                and row.get("passed") is True
                and row.get("probe_exitcode") == 0
                and row.get("decode_exitcode") == 0
                and type(row.get("expected_frames")) is int
                and row["expected_frames"] > 0
                and row.get("decoded_frames") == row["expected_frames"]
                and row.get("decoded_pixel_format") == "yuv420p"
                and valid_sha(row.get("compressed_sha256"))
            ):
                raise ValueError(
                    "failed/incomplete per-file full-video decode evidence"
                )
            file_frames[row["camera"]] += row["decoded_frames"]
            episode_coverage[row["camera"]].update(row.get("episodes", []))
        if dict(file_frames) != expected_decoded or any(
            coverage != set(range(published)) for coverage in episode_coverage.values()
        ):
            raise ValueError("per-file full-video evidence omits saved sources/frames")
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(f"new formal full-video integrity: {error}")
    return errors


def _paired(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    pilot = all(run["config"].get("phase") == "pilot" for run in runs)
    baselines = (
        ("offline32", "ods_serial")
        if pilot
        else ("offline8", "offline32", "offline128", "ods_serial")
    )
    for vision in _VISIONS:
        overlap = _group(runs, "ods_overlap", vision)
        for arm in baselines:
            baseline = _group(runs, arm, vision)
            seeds = sorted(
                {str(run["config"].get("seed")) for run in overlap + baseline}
            )
            for seed in seeds:
                left = [
                    run for run in baseline if str(run["config"].get("seed")) == seed
                ]
                right = [
                    run for run in overlap if str(run["config"].get("seed")) == seed
                ]
                reasons: list[str] = []
                evidence = {}
                if (
                    pilot
                    and len(
                        {
                            str(run["config"].get("seed"))
                            for run in runs
                            if run["config"]["vision"] == vision
                        }
                    )
                    != 1
                ):
                    reasons.append("pilot scope requires exactly one seed per vision")
                if len(left) != 1 or len(right) != 1:
                    reasons.append(
                        "requires exactly one qualified run per arm and seed"
                    )
                else:
                    for role, run in (("baseline", left[0]), ("overlap", right[0])):
                        facts, errors = _fairness(run)
                        evidence[role] = facts
                        reasons.extend(f"{role}: {error}" for error in errors)
                        if run["config"].get("phase") not in {"report", "pilot"}:
                            reasons.append(
                                f"{role}: comparative speedup requires phase=report or pilot, observed {run['config'].get('phase')!r}"
                            )
                        audit = run["summary"].get("post_audit")
                        if (
                            run["summary"].get("post_audit_status") != "complete"
                            or not isinstance(audit, dict)
                            or audit.get("passed") is not True
                        ):
                            reasons.append(
                                f"{role}: missing successful post-timing corpus audit"
                            )
                    if left[0]["config"].get("phase") != right[0]["config"].get(
                        "phase"
                    ):
                        reasons.append("pilot and formal phases cannot be paired")
                    if evidence["baseline"].get("prefetch_policy") != evidence[
                        "overlap"
                    ].get("prefetch_policy"):
                        reasons.append("paired prefetch_policy differs")
                    for key in (
                        "training_budget",
                        "initial_model_sha256",
                        "normalization_sha256",
                        "initial_numeric_prefix",
                        "camera_order",
                        "allocation",
                        "auxiliary_graphics",
                        "requested_evaluation_scenes",
                    ):
                        if evidence["baseline"].get(key) != evidence["overlap"].get(
                            key
                        ):
                            reasons.append(f"paired {key} differs")
                row = {
                    "vision": vision,
                    "baseline": arm,
                    "seed": seed,
                    "excluded_reason": "; ".join(reasons) if reasons else None,
                    "excluded_reasons": reasons,
                    "fairness_evidence": evidence,
                }
                if pilot:
                    row["scope"] = (
                        "single-seed pilot; not a three-seed formal comparison"
                    )
                if not reasons:
                    offline = float(left[0]["metrics"]["total_seconds"])
                    ods = float(right[0]["metrics"]["total_seconds"])
                    row.update(
                        baseline_seconds=offline,
                        overlap_seconds=ods,
                        baseline_over_overlap=offline / ods,
                        overlap_time_saving_fraction=1 - ods / offline,
                        baseline_trained_sources=left[0]["metrics"]["trained_sources"],
                        overlap_trained_sources=right[0]["metrics"]["trained_sources"],
                        baseline_initial_xy_cells=left[0]["metrics"][
                            "initial_xy_occupied_cells"
                        ],
                        overlap_initial_xy_cells=right[0]["metrics"][
                            "initial_xy_occupied_cells"
                        ],
                    )
                rows.append(row)
    return rows


def _save(figure: Any, output: Path, name: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"{name}.{suffix}", dpi=160, bbox_inches="tight")


def _style(axis: Any) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2)
    axis.set_axisbelow(True)


def _tradeoff(runs: list[dict[str, Any]], plt: Any, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for row, vision in enumerate(_VISIONS):
        for col, (metric, label) in enumerate(
            (
                ("initial_xy_occupied_cells", "Initial XY cells (of 64)"),
                ("effective_sources", "Effective trained sources"),
            )
        ):
            axis = axes[row, col]
            for arm in _ARMS:
                group = _group(runs, arm, vision)
                if group:
                    axis.scatter(
                        [run["metrics"]["total_seconds"] for run in group],
                        [run["metrics"][metric] for run in group],
                        color=_COLORS[arm],
                        marker=_MARKERS[arm],
                        s=55,
                        alpha=0.8,
                        label=_LABELS[arm],
                    )
            axis.set(
                title=f"{vision}: time–coverage tradeoff",
                xlabel="End-to-end time (s)",
                ylabel=label,
            )
            axis.set_ylim(bottom=0)
            if col == 0:
                axis.set_ylim(top=66)
            _style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, output, "01_time_diversity")
    plt.close(fig)


def _curves(runs: list[dict[str, Any]], plt: Any, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for row, vision in enumerate(_VISIONS):
        for col, (x_key, label) in enumerate(
            (
                ("time_seconds", "Elapsed time (s)"),
                ("update", "Completed optimizer updates"),
            )
        ):
            axis = axes[row, col]
            for arm in _ARMS:
                for index, run in enumerate(_group(runs, arm, vision)):
                    curve = run["metrics"].get("coverage_curve", [])
                    if curve:
                        axis.step(
                            [0] + [point[x_key] for point in curve],
                            [0]
                            + [point["initial_xy_occupied_cells"] for point in curve],
                            where="post",
                            color=_COLORS[arm],
                            alpha=0.55,
                            label=_LABELS[arm] if index == 0 else None,
                        )
            axis.set(
                title=f"{vision}: coverage after completed training",
                xlabel=label,
                ylabel="Initial XY cells (of 64)",
                ylim=(0, 66),
            )
            _style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, output, "02_coverage_growth")
    plt.close(fig)


def _stage(name: str) -> str:
    lowered = name.lower()
    if lowered == "sdk_recording":
        return "SDK recording service"
    if lowered == "sdk_decode":
        return "SDK temporal fetch"
    if "generation_drain_wait" in lowered:
        return "finalize / cleanup"
    if lowered.endswith("_wait") or "quota_wait" in lowered:
        return "waiting"
    if any(
        word in lowered
        for word in ("generation", "refill", "initial_fill", "acquisition")
    ):
        return "generation"
    if any(word in lowered for word in ("finaliz", "drain", "close", "shutdown")):
        return "finalize / cleanup"
    if any(
        word in lowered
        for word in ("record", "decode", "copy", "sampl", "read", "batch", "transfer")
    ):
        return "data preparation"
    if any(word in lowered for word in ("train", "optimizer", "backward", "forward")):
        return "training"
    return "other"


def _timeline(runs: list[dict[str, Any]], plt: Any, output: Path) -> None:
    stages = (
        "generation",
        "SDK recording service",
        "SDK temporal fetch",
        "data preparation",
        "training",
        "waiting",
        "finalize / cleanup",
        "other",
    )
    colors = dict(
        zip(
            stages,
            (
                "#4c78a8",
                "#f2b134",
                "#b279a2",
                "#b3a9cf",
                "#159b80",
                "#b7a36a",
                "#c65656",
                "#999999",
            ),
        )
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), squeeze=False)
    for row, vision in enumerate(_VISIONS):
        axis = axes[row, 0]
        eligible = [
            run for run in runs if run["config"]["vision"] == vision and _qualified(run)
        ]
        seeds = sorted({str(run["config"].get("seed")) for run in eligible})
        seed = seeds[0] if seeds else None
        selected = [run for run in eligible if str(run["config"].get("seed")) == seed]
        labels, locations = [], []
        y = 0
        for arm in _ARMS:
            candidates = [run for run in selected if run["arm"] == arm]
            if len(candidates) != 1:
                continue
            run = candidates[0]
            intervals = run["metrics"].get("intervals", [])
            for stage in stages:
                spans = [
                    (item["start_seconds"], item["end_seconds"] - item["start_seconds"])
                    for item in intervals
                    if _stage(str(item.get("name", ""))) == stage
                ]
                if spans:
                    axis.broken_barh(
                        spans, (y, 0.7), facecolors=colors[stage], alpha=0.8
                    )
                labels.append(f"{_LABELS[arm]} · {stage}")
                locations.append(y + 0.35)
                y += 1
            end = run["metrics"]["training_complete_seconds"]
            total = run["metrics"]["total_seconds"]
            axis.plot(
                [end, end],
                [y - len(stages), y - 0.3],
                color="#222222",
                linewidth=0.7,
                linestyle=":",
            )
            axis.plot(
                [total, total],
                [y - len(stages), y - 0.3],
                color="#222222",
                linewidth=0.7,
            )
            y += 0.6
        axis.set_yticks(locations, labels, fontsize=6)
        axis.invert_yaxis()
        axis.set(
            title=f"{vision}: actual stages, seed {seed}; dotted=last update, solid=normal exit",
            xlabel="Elapsed time (s)",
        )
        _style(axis)
    fig.tight_layout()
    _save(fig, output, "03_pipeline_timeline")
    plt.close(fig)


def _reuse(runs: list[dict[str, Any]], plt: Any, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), squeeze=False)
    for row, vision in enumerate(_VISIONS):
        axis = axes[row, 0]
        for arm in _ARMS:
            counts = [
                int(episode["trained_chunks"])
                for run in _group(runs, arm, vision)
                for episode in run["metrics"].get("episodes", [])
            ]
            if counts:
                histogram = Counter(counts)
                x = sorted(histogram)
                axis.step(
                    x,
                    [histogram[count] for count in x],
                    where="mid",
                    color=_COLORS[arm],
                    label=_LABELS[arm],
                )
                axis.scatter(
                    x,
                    [histogram[count] for count in x],
                    color=_COLORS[arm],
                    s=14,
                    alpha=0.65,
                )
        caps = sorted(
            {
                run["metrics"].get("reuse_cap")
                for run in runs
                if run["config"]["vision"] == vision
                and run["arm"] in {"ods_overlap", "ods_serial"}
                and _qualified(run)
                and run["metrics"].get("reuse_cap") is not None
            }
        )
        for cap in caps:
            axis.axvline(
                cap, color="#555555", linestyle="--", linewidth=1, label=f"ODS K={cap}"
            )
        axis.set_xscale("symlog", linthresh=1)
        axis.set(
            title=f"{vision}: all published sources, including retired and never trained",
            xlabel="Lifetime completed training chunks per source (symlog)",
            ylabel="Sources across qualified seeds",
        )
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        _style(axis)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8, ncol=3)
    fig.tight_layout()
    _save(fig, output, "04_lifetime_reuse")
    plt.close(fig)


def _formatted(stats: dict[str, Any]) -> str:
    if not stats["n"]:
        return "—"
    return f"{stats['median']:.3g} [{stats['min']:.3g}, {stats['max']:.3g}]"


def _resource_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        "cpu_assigned_percent": [],
        "simulation_gpu_board_percent": [],
        "training_gpu_board_percent": [],
        "sampled_pss_gib": [],
        "sampled_owned_gpu_gib": [],
        "dataset_mib": [],
        "checkpoint_mib": [],
    }
    for run in runs:
        resources = run["metrics"].get("resources") or {}
        value = _number(
            (resources.get("cpu_utilization") or {}).get(
                "mean_assigned_capacity_percent"
            )
        )
        if value is not None:
            values["cpu_assigned_percent"].append(value)
        for key, container, field, scale in (
            ("sampled_pss_gib", resources, "peak_pss_bytes", 1024**3),
            ("sampled_owned_gpu_gib", resources, "peak_vram_bytes", 1024**3),
            ("dataset_mib", run["metrics"], "dataset_bytes", 1024**2),
            ("checkpoint_mib", run["metrics"], "checkpoint_bytes", 1024**2),
        ):
            value = _number(container.get(field))
            if value is not None:
                values[key].append(value / scale)
        pair = run["summary"].get("device_supervision", {}).get("primary_pair", {})
        boards = resources.get("primary_gpu_utilization") or {}
        for role in ("simulation", "training"):
            uuid = pair.get(role, {}).get("uuid")
            value = _number(boards.get(uuid, {}).get("mean_sampled_gpu_percent"))
            if value is not None:
                values[role + "_gpu_board_percent"].append(value)
    return {key: _stats(observed) for key, observed in values.items()}


def write_report(root: Path, output: Path | None = None) -> dict[str, Any]:
    """Write four measured figures and qualified/unqualified JSON/Markdown.

    Args:
        root: Directory recursively containing per-run ``summary.json`` files.
        output: Report directory. Defaults to ``root/training-report``.

    Returns:
        Finite JSON report, including every observed run and ratio exclusion.
    """
    root = Path(root)
    output = Path(output) if output is not None else root / "training-report"
    if (
        (root / "main").is_dir()
        and (root / "suite-summary.json").is_file()
        and _json(root / "suite-summary.json").get("phase") == "pilot"
    ):
        root = root / "main"
    output.mkdir(parents=True, exist_ok=True)
    runs = _load(root)
    aggregates = _aggregate(runs)
    pairs = _paired(runs)
    pilot = all(run["config"].get("phase") == "pilot" for run in runs)
    display_arms = ("offline32", "ods_overlap", "ods_serial") if pilot else _ARMS
    report = {
        "schema": "ods-training-report/v1",
        "input": str(root.resolve()),
        "qualified_runs": sum(_qualified(run) for run in runs),
        "observed_runs": len(runs),
        "qualified_paired_comparisons": sum(
            row["excluded_reason"] is None for row in pairs
        ),
        "qualification_scope": "Per-run completed-work qualification; formal paired ratios additionally require matching model/stat/initial-prefix/resource evidence and successful corpus audit.",
        "aggregates": aggregates,
        "resource_aggregates": {
            vision: {arm: _resource_stats(_group(runs, arm, vision)) for arm in _ARMS}
            for vision in _VISIONS
        },
        "paired_comparisons": pairs,
        "runs": [
            {
                **run["summary"],
                "path": run["path"],
                "arm": run["arm"],
                "qualification_reasons": _qualification(run),
            }
            for run in runs
        ],
    }
    if pilot:
        report["experiment_scope"] = (
            "single-seed systems pilot; eager diagnostics excluded; no policy-learning claim"
        )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for plot in (_tradeoff, _curves, _timeline, _reuse):
        plot(runs, plt, output)
    lines = [
        "# ODS visual training: fixed-work time and diversity",
        "",
        f"Qualified runs: **{report['qualified_runs']}/{len(runs)}**; qualified {'pilot' if pilot else 'formal'} pairs: **{report['qualified_paired_comparisons']}**. Failed, unsupported and unqualified runs remain in report.json and are excluded from aggregate values and paired time ratios.",
        "",
        "Each result completes the declared optimizer-update, batch and valid-target budget. Data amount and final coverage may differ. Ratios compare total pipeline time, including acquisition, recording, decode, actual training, checkpoint and normal teardown; they do not establish equal policy quality or a universal ODS speedup.",
        "",
    ]
    if pilot:
        lines += [
            "Single-seed systems pilot. Eager controls remain outside main data and primary ratios. Zero-scene evaluation is systems-only and provides no policy-quality evidence.",
            "",
        ]
    for vision in _VISIONS:
        lines += [
            f"## {vision}",
            "",
            "Values are median [minimum, maximum] across qualified seeds.",
            "",
            "| Path | n | Total s | Trained sources | Effective sources | XY cells / 64 | Unique targets |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm in display_arms:
            group = aggregates[vision][arm]
            lines.append(
                f"| {_LABELS[arm]} | {group['total_seconds']['n']} | {_formatted(group['total_seconds'])} | {_formatted(group['trained_sources'])} | {_formatted(group['effective_sources'])} | {_formatted(group['initial_xy_occupied_cells'])} | {_formatted(group['unique_target_frames'])} |"
            )
        lines += [
            "",
            "| Baseline / ODS overlap | Paired seeds | Total-time ratio median [min, max] |",
            "| --- | ---: | ---: |",
        ]
        for arm in (
            ("offline32", "ods_serial")
            if pilot
            else ("offline8", "offline32", "offline128", "ods_serial")
        ):
            values = [
                row["baseline_over_overlap"]
                for row in pairs
                if row["vision"] == vision
                and row["baseline"] == arm
                and row["excluded_reason"] is None
            ]
            lines.append(
                f"| {_LABELS[arm]} | {len(values)} | {_formatted(_stats(values))} |"
            )
        lines += [
            "",
            "| Path | CPU mean / assigned capacity % | Simulation GPU board mean % | Training GPU board mean % |",
            "| --- | ---: | ---: | ---: |",
        ]
        for arm in display_arms:
            resource = report["resource_aggregates"][vision][arm]
            lines.append(
                f"| {_LABELS[arm]} | {_formatted(resource['cpu_assigned_percent'])} | {_formatted(resource['simulation_gpu_board_percent'])} | {_formatted(resource['training_gpu_board_percent'])} |"
            )
        lines += [
            "",
            "CPU values use continuing-PID sampled interval deltas normalized by the assigned cores; new or disappeared processes may be undercounted. GPU values are whole-board NVML samples and include foreign work; they do not attribute kernel utilization to this run. Missing observations remain unknown (—), not zero. See raw resources for sample counts and process-coverage gaps.",
            "",
            "| Path | Sampled PSS max GiB | Sampled owned GPU max GiB | Dataset MiB | Checkpoints MiB |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for arm in display_arms:
            resource = report["resource_aggregates"][vision][arm]
            lines.append(
                f"| {_LABELS[arm]} | {_formatted(resource['sampled_pss_gib'])} | {_formatted(resource['sampled_owned_gpu_gib'])} | {_formatted(resource['dataset_mib'])} | {_formatted(resource['checkpoint_mib'])} |"
            )
        lines += [
            "",
            "Memory maxima are sampled observations, not guaranteed allocator/process peaks. Disk sizes are logical finalized dataset and initial/final checkpoint bytes; they exclude temporary disk peaks, logs and post-timing audit/evaluation artifacts.",
            "",
        ]
    lines += [
        "## Figures and interpretation",
        "",
        "![End-to-end time and actual trained-state coverage](01_time_diversity.png)",
        "",
        "![Coverage after completed optimizer updates](02_coverage_growth.png)",
        "",
        "![Measured pipeline stages](03_pipeline_timeline.png)",
        "",
        "![Lifetime source reuse, including retired sources](04_lifetime_reuse.png)",
        "",
        "Coverage uses completed training only. Initial XY bins are fixed 1 cm cells over the task's 8 cm × 8 cm reset area; anchor-state voxels are 2 cm. Source identity growth alone is not physical diversity. Overlapping prediction windows share target frames and are deduplicated by immutable frame identity. Published but untrained data does not increase these metrics.",
        "",
        "AV1 decoding is lossy. Compare source quality audits and held-out evaluation alongside timing. Differences between online raw pixels and decoded pixels prevent a claim of identical optimization trajectories. Serial trace replay isolates overlap only when its source/anchor manifest audit passes. A faster small offline run or broader large offline corpus is a valid result.",
        "",
        "SDK recording spans cover real create/write/encode/bookkeeping/finalize service; SDK temporal fetch spans include worker initialization, dataset reads, decode, conversion and validation. Neither lane claims pure encoder/decoder CPU work or GPU kernel activity. Owner packing/copy and waiting remain separate. Intervals may overlap and are not summed into total time. Timeline rows use the first observed qualified seed per vision, without selecting the fastest run. PNG, SVG and PDF versions share measured inputs.",
        "",
        "Formal ratios require matching initial-model/normalization SHA256, initial-C numeric prefix, camera order, declared/completed training budget, ordered primary GPU UUIDs and the same eight CPU cores. Retained supervisor launch masks/thread limits, placement guards, process exit and clocks must agree. New formal inputs additionally require chronological sdk-worker-service/v1 intervals and a revisioned full-video-integrity/v1 receipt for every saved frame; the original numeric/H2 audit remains required. Full-video YUV sequential integrity is distinct from lossless RGB fidelity or every SDK temporal seek. Missing evidence blocks claims; smoke/calibration or legacy CPU rendering fixtures are not formal comparisons. Post-timing corpus audit and evaluation durations are excluded from pipeline time.",
        "",
        "## Excluded paired claims",
        "",
    ]
    excluded_pairs = [row for row in pairs if row["excluded_reason"] is not None]
    if excluded_pairs:
        for row in excluded_pairs:
            lines.append(
                f"- {row['vision']} / {row['baseline']} / seed {row['seed']}: {row['excluded_reason']}."
            )
    else:
        lines.append("Every observed paired comparison passed its evidence gates.")
    lines += [
        "",
        "## Retained unqualified runs",
        "",
    ]
    excluded = [run for run in runs if not _qualified(run)]
    if excluded:
        for run in excluded:
            lines.append(f"- `{run['path']}`: {', '.join(_qualification(run))}.")
    else:
        lines.append("No unqualified runs were observed in this input directory.")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = write_report(args.root, args.output)
    print(
        json.dumps(
            {
                "qualified_runs": result["qualified_runs"],
                "observed_runs": result["observed_runs"],
            }
        )
    )


if __name__ == "__main__":
    _main()
