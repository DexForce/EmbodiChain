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

"""Post-run independent audit/packaging; default inspect never archives inputs.

Only invoke --mode package --execute after the root agent's explicit trigger.
No simulator, learner or GPU work is performed. Checkpoint verification uses CPU.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

__all__ = ["main"]

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".txt",
    ".csv",
    ".xml",
    ".yaml",
    ".yml",
    ".md",
    ".stderr",
    ".stdout",
}
DIRECT_NAMES = {
    "summary.json",
    "config.json",
    "environment.json",
    "payload_schema.json",
    "initial-contract.json",
    "process.json",
    "audit-process.json",
    "evaluation-process.json",
    "training-trace.json",
    "post-audit.json",
    "evaluation.json",
    "roles.json",
}
METRIC_KEYS = (
    "completed_updates",
    "trained_chunks",
    "valid_target_occurrences",
    "history_frame_occurrences",
    "trained_sources",
    "effective_sources",
    "initial_xy_occupied_cells",
    "initial_xy_coverage_fraction",
    "anchor_xyz_occupied_voxels",
    "unique_anchors",
    "unique_history_frames",
    "unique_target_frames",
    "target_duplicate_fraction",
)
DEFAULT_DIAGNOSTICS = {
    "dev": "/tmp/ods-visual-training-dev-20261021",
    "smoke-v1": "/tmp/ods-visual-training-smoke-20261021-v1",
    "smoke-v2": "/tmp/ods-visual-training-smoke-20261021-v2",
    "wait-probe-v1": "/tmp/ods-visual-training-wait-utilization-20261021-v1",
    "calibration-v1": "/tmp/ods-visual-training-calibration-20261021-v1",
    "calibration-v2": "/tmp/ods-visual-training-calibration-20261021-v2",
    "calibration-v3": "/tmp/ods-visual-training-calibration-20261021-v3",
    "scheduling-validation": "/tmp/ods-visual-training-scheduling-validation-20261021-v1",
    "final-validation": "/tmp/ods-visual-training-final-validation-20261021-v1",
    "full-video-qualification": "/tmp/ods-video-integrity-validation",
    "formal-admission-v1": "/tmp/ods-visual-training-formal-20261021-v1",
    "formal-superseded-v2": "/tmp/ods-visual-training-formal-20261021-v2",
}
HASH_CACHE: dict[tuple[int, int, int, int], str] = {}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    before = path.stat()
    key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if key not in HASH_CACHE:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(chunk)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise RuntimeError(f"Input changed during hashing: {path}")
        HASH_CACHE[key] = h.hexdigest()
    return HASH_CACHE[key]


def equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8)
    return actual == expected


def source_id(value: Any) -> str:
    return value if isinstance(value, str) else f"{int(value[0])}:{int(value[1])}"


def initial_bin(xy: list[float]) -> tuple[int, int] | None:
    if not all(
        low - 1e-7 <= x <= high + 1e-7
        for x, low, high in zip(xy, (-0.46, -0.12), (-0.38, -0.04))
    ):
        return None
    bins = []
    for x, low in zip(xy, (-0.46, -0.12)):
        v = (x - low) / 0.01
        if abs(v - round(v)) <= 1e-5:
            v = round(v)
        bins.append(max(0, min(7, math.floor(v))))
    return tuple(bins)


def reconstruct_metrics(
    directory: Path, events_override: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Recompute coverage/counts from publications and completed-update chunks."""
    summary = read(directory / "summary.json")
    recorded = summary.get("training_metrics", {})
    if not recorded:
        return {
            "passed": summary["status"] != "complete",
            "scope": "No completed-training metrics",
            "status": summary["status"],
        }
    events = events_override
    if events is None:
        events = [
            json.loads(line)
            for line in (directory / "training-events.jsonl").read_text().splitlines()
            if line.strip()
        ]
    H, F, B = (int(recorded[k]) for k in ("history", "future", "batch_size"))
    pubs, rows, errors, updates, curve = {}, {}, [], [], []
    xy_weights: collections.Counter[tuple[int, int]] = collections.Counter()
    voxels: set[tuple[int, int, int]] = set()

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    def coverage() -> dict[str, Any]:
        used = [row for row in rows.values() if row["chunks"]]
        chunks = sum(row["chunks"] for row in used)
        target_unique = sum(len(row["targets"]) for row in used)
        return {
            "completed_updates": len(updates),
            "trained_chunks": chunks,
            "valid_target_occurrences": chunks * F,
            "history_frame_occurrences": chunks * H,
            "trained_sources": len(used),
            "effective_sources": (
                math.exp(
                    -math.fsum(
                        (row["chunks"] / chunks) * math.log(row["chunks"] / chunks)
                        for row in used
                    )
                )
                if used
                else 0.0
            ),
            "initial_xy_occupied_cells": len(xy_weights),
            "initial_xy_coverage_fraction": len(xy_weights) / 64,
            "anchor_xyz_occupied_voxels": len(voxels),
            "unique_anchors": sum(len(row["anchors"]) for row in used),
            "unique_history_frames": sum(len(row["history"]) for row in used),
            "unique_target_frames": target_unique,
            "target_duplicate_fraction": (
                1 - target_unique / (chunks * F) if chunks else 0.0
            ),
        }

    for event in events:
        kind = event.get("kind")
        if kind == "episode_published":
            sid = event["source_id"]
            check(sid not in pubs, f"duplicate publication {sid}")
            pubs[sid] = event
            rows[sid] = {
                "chunks": 0,
                "anchors": collections.Counter(),
                "history": collections.Counter(),
                "targets": collections.Counter(),
                "first_ns": None,
                "retired_ns": None,
            }
        elif kind == "episode_retired":
            sid = event["source_id"]
            check(sid in rows, f"retired unknown source {sid}")
            rows[sid]["retired_ns"] = event["monotonic_ns"]
        elif kind == "update_completed":
            check(event["update"] == len(updates) + 1, "nonconsecutive updates")
            chunks = event["chunks"]
            check(len(chunks) == B, f"batch size at update {event['update']}")
            check(
                event["requested_seconds"]
                <= event["train_started_seconds"]
                <= event["time_seconds"],
                "noncausal update timing",
            )
            check(
                bool(event.get("losses"))
                and all(math.isfinite(v) for v in event["losses"].values()),
                "missing/nonfinite actual losses",
            )
            for chunk in chunks:
                sid, anchor = chunk["source_id"], int(chunk["anchor"])
                pub, row = pubs[sid], rows[sid]
                check(
                    chunk["ordinal"] == pub["ordinal"]
                    and chunk["generation"] == pub["generation"],
                    "source ordinal/generation mismatch",
                )
                check(
                    anchor - H + 1 >= 0 and anchor + F <= pub["frame_count"],
                    "window outside accepted source",
                )
                check(
                    event["monotonic_ns"] >= pub["monotonic_ns"],
                    "training precedes publication",
                )
                row["chunks"] += 1
                row["anchors"][anchor] += 1
                row["history"].update(range(anchor - H + 1, anchor + 1))
                row["targets"].update(range(anchor, anchor + F))
                if row["first_ns"] is None:
                    row["first_ns"] = event["monotonic_ns"]
                cell = initial_bin(pub["initial_xy"])
                if cell is not None:
                    xy_weights[cell] += 1
                xyz = chunk["anchor_cube_xyz"]
                check(
                    len(xyz) == 3 and all(math.isfinite(v) for v in xyz),
                    "invalid cube XYZ",
                )
                voxels.add(tuple(math.floor(v / 0.02) for v in xyz))
            updates.append(event)
            expected = coverage()
            for key, value in expected.items():
                check(
                    equivalent(event["coverage"].get(key), value),
                    f"event coverage {event['update']} {key}",
                )
            curve.append(expected)
    logged_intervals = [
        {
            key: value
            for key, value in row.items()
            if key not in {"kind", "monotonic_ns", "time_seconds"}
        }
        for row in events
        if row.get("kind") == "interval"
    ]
    check(
        logged_intervals == recorded.get("intervals", []),
        "chronological worker/owner interval events differ from summary",
    )
    recomputed = coverage()
    for key, value in recomputed.items():
        check(equivalent(recorded.get(key), value), f"summary {key}")
    check(recorded.get("published_sources") == len(pubs), "publication count")
    check(
        recorded.get("untrained_sources")
        == sum(row["chunks"] == 0 for row in rows.values()),
        "untrained tail exclusion",
    )
    check(
        recorded.get("anchor_xyz_voxels") == [list(v) for v in sorted(voxels)],
        "voxel identities",
    )
    weights = [[xy_weights[(x, y)] for y in range(8)] for x in range(8)]
    check(
        recorded.get("initial_xy_grid", {}).get("trained_chunk_weights_xy") == weights,
        "initial XY weights",
    )
    saved_sources = {row["source_id"]: row for row in recorded.get("episodes", [])}
    check(set(saved_sources) == set(pubs), "source identity set")
    source_results = []
    for sid, state in rows.items():
        pub, saved = pubs[sid], saved_sources[sid]
        check(
            saved["trained_chunks"] == state["chunks"], f"lifetime source count {sid}"
        )
        check(
            saved["initial_xy_bin"]
            == (
                list(initial_bin(pub["initial_xy"]))
                if initial_bin(pub["initial_xy"]) is not None
                else None
            ),
            f"initial grid {sid}",
        )
        for key, counter in (
            ("anchor_counts", state["anchors"]),
            ("history_frame_counts", state["history"]),
            ("target_frame_counts", state["targets"]),
        ):
            check(
                saved[key] == [counter[i] for i in range(pub["frame_count"])],
                f"frame counter {sid} {key}",
            )
        if state["first_ns"] is not None:
            delay = (state["first_ns"] - pub["monotonic_ns"]) / 1e9
            check(
                equivalent(saved["publication_to_first_training_seconds"], delay),
                f"first train delay {sid}",
            )
        else:
            check(
                saved["first_completed_training_seconds"] is None,
                f"unused source trained timestamp {sid}",
            )
        source_results.append(
            {
                "source_id": sid,
                "ordinal": pub["ordinal"],
                "trained_chunks": state["chunks"],
                "retired": state["retired_ns"] is not None,
            }
        )
    ledger = recorded.get("ledger", {}).get("episodes", [])
    check(
        {source_id(row["source_id"]) for row in ledger} == set(pubs),
        "ledger source set",
    )
    cap = recorded.get("reuse_cap")
    for row in ledger:
        sid = source_id(row["source_id"])
        check(
            row.get("reserved") == 0
            and row.get("trained") == rows[sid]["chunks"]
            and row.get("delivered") == rows[sid]["chunks"],
            f"reservation/delivery/training ledger {sid}",
        )
        check(cap is None or row["delivered"] <= cap, f"delivery cap {sid}")
    if (directory / "training-trace.json").exists():
        trace = read(directory / "training-trace.json")
        check(len(trace["updates"]) == len(updates), "trace update length")
        for actual, expected in zip(trace["updates"], updates):
            pairs = [[c["ordinal"], c["anchor"]] for c in expected["chunks"]]
            check(
                actual["update"] == expected["update"] and actual["pairs"] == pairs,
                "ordered trace vs completed chunks",
            )
        for entry in trace["episodes"]:
            sid = source_id(entry["source_id"])
            pub = pubs[sid]
            check(
                (
                    entry["ordinal"],
                    entry["generation"],
                    entry["length"],
                    entry["numeric_sha256"],
                )
                == (
                    pub["ordinal"],
                    pub["generation"],
                    pub["frame_count"],
                    pub["metadata"]["numeric_sha256"],
                ),
                "trace accepted trajectory identity",
            )
    return {
        "passed": not errors,
        "errors": errors,
        "recomputed": recomputed,
        "sources": source_results,
        "registered_sources": len(pubs),
        "untrained_tail_sources": sum(row["chunks"] == 0 for row in rows.values()),
        "scope": "Independent event reconstruction; causal bounds and immutable IDs audited. Full tensor/continuity/codec validation remains the recorded SDK/production qualification.",
    }


def checkpoint_model_hash(path: Path) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    def state_hash(values: dict[str, Any]) -> str:
        h = hashlib.sha256()
        for key, value in sorted(values.items()):
            h.update(key.encode())
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().contiguous()
                h.update(str((tuple(value.shape), value.dtype)).encode())
                h.update(memoryview(value.numpy()))
            else:
                h.update(json.dumps(value, sort_keys=True).encode())
        return h.hexdigest()

    steps = {
        int(row["step"])
        for row in saved["optimizer"].get("state", {}).values()
        if "step" in row
    }
    result = {
        "step": saved["step"],
        "model_sha256": state_hash(saved["model"]),
        "normalization_sha256": state_hash(saved["normalization"]),
        "optimizer_steps": sorted(steps),
        "file_sha256": digest(path),
    }
    del saved
    gc.collect()
    return result


def full_video_evidence(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Independently bind the full sequential-decode receipt to retained bytes."""
    errors = []
    metrics = summary["training_metrics"]
    receipt = summary.get("post_audit", {}).get("full_video_integrity")
    sidecar = directory / "video-integrity.json"
    saved = read(sidecar) if sidecar.is_file() else None
    if receipt is not None and saved is not None and receipt != saved:
        errors.append("inline/sidecar full video evidence differs")
    receipt = receipt if isinstance(receipt, dict) else saved
    if not isinstance(receipt, dict):
        return {"passed": False, "errors": ["missing revisioned full-video receipt"]}
    cameras = read(directory / "initial-contract.json")["camera_order"]
    frames = sum(row["frame_count"] for row in metrics["episodes"])
    expected = {"observation.images." + camera: frames for camera in cameras}
    if not (
        receipt.get("revision") == "full-video-integrity/v1"
        and receipt.get("passed") is True
        and receipt.get("all_saved_video_frames_decoded") is True
        and receipt.get("expected_episodes") == metrics["published_sources"]
        and receipt.get("expected_frames_per_camera") == frames
        and receipt.get("decoded_frames_per_camera") == expected
        and receipt.get("total_decoded_camera_frames") == frames * len(cameras)
    ):
        errors.append("full-frame decoded corpus counts/status differ")
    dataset = (directory / "lerobot_dataset").resolve()
    file_frames = collections.Counter()
    try:
        for row in receipt.get("files", []):
            camera = row["camera"]
            if not (
                row.get("passed") is True
                and row.get("probe_exitcode") == 0
                and row.get("decode_exitcode") == 0
                and row.get("decoded_pixel_format") == "yuv420p"
                and row.get("decoded_frames") == row.get("expected_frames")
            ):
                errors.append("unsuccessful per-file sequential YUV integrity")
            file_frames[camera] += row["decoded_frames"]
            video = (dataset / row["file"]).resolve()
            if (
                not video.is_relative_to(dataset)
                or digest(video) != row["compressed_sha256"]
            ):
                errors.append("saved compressed video no longer matches verified bytes")
        if dict(file_frames) != expected:
            errors.append("per-file decoded totals omit saved camera frames")
        metadata_hashes = receipt.get("metadata_sha256")
        if not isinstance(metadata_hashes, dict) or not metadata_hashes:
            errors.append("missing full-video metadata fingerprints")
            metadata_hashes = {}
        verifier = receipt.get("verifier_sha256")
        if not isinstance(verifier, str) or len(verifier) != 64:
            errors.append("missing full-video verifier fingerprint")
        for relative, sha in metadata_hashes.items():
            path = (dataset / relative).resolve()
            if not path.is_relative_to(dataset) or digest(path) != sha:
                errors.append("episode/video metadata changed after full-frame audit")
    except (KeyError, OSError, TypeError) as error:
        errors.append(f"full-video retained-artifact verification: {error!r}")
    return {
        "passed": not errors,
        "errors": errors,
        "revision": receipt.get("revision"),
        "expected_frames_per_camera": frames,
        "decoded_frames_per_camera": receipt.get("decoded_frames_per_camera"),
        "verifier_sha256": receipt.get("verifier_sha256"),
        "scope": "Full sequential software YUV decode plus exact compressed/metadata hashes; not lossless RGB fidelity or every SDK temporal seek.",
    }


def failed_run_evidence(
    directory: Path,
    summary: dict[str, Any],
    independent_metrics: dict[str, Any] | None,
    verify_checkpoints: bool,
) -> dict[str, Any]:
    """Audit retained training artifacts without overturning a failed outcome."""
    metrics = summary.get("training_metrics", {})
    errors: list[str] = []
    result: dict[str, Any] = {
        "scope": "Retained evidence from a failed terminal attempt; never promotes report qualification",
        "supervisor_status": summary["status"],
        "qualification_preserved": summary.get("qualification_passed") is False,
        "quality_error": summary.get("quality_error"),
        "error": summary.get("error"),
        "post_audit_status": summary.get("post_audit_status"),
        "evaluation_status": summary.get("evaluation_status"),
        "requested_evaluation_scenes": summary.get("requested_evaluation_scenes"),
        "evaluation_required_for_qualification": True,
    }
    if not metrics:
        result.update(passed=True, errors=[], training_artifacts="not produced")
        return result
    if not independent_metrics or not independent_metrics.get("passed"):
        errors.append("failed attempt has inconsistent completed-work event evidence")
    updates = metrics.get("completed_updates")
    if type(updates) is not int or not 0 <= updates <= 128:
        errors.append("failed attempt has invalid completed update count")
    elif (
        metrics.get("trained_chunks") != updates * 32
        or metrics.get("valid_target_occurrences") != updates * 32 * 32
    ):
        errors.append("failed attempt completed-work budget mismatch")
    process_path = directory / "process.json"
    process = read(process_path) if process_path.is_file() else {}
    training_complete = process.get("returncode") == 0 and updates == 128
    result["training_process_completed_full_budget"] = training_complete
    if process and not equivalent(
        process.get("elapsed_seconds"), metrics.get("total_seconds")
    ):
        errors.append("failed attempt supervisor total disagreement")
    receipt_path = directory / "post-audit.json"
    receipt = summary.get("post_audit")
    if not isinstance(receipt, dict) and receipt_path.is_file():
        receipt = read(receipt_path)
        result["post_audit_receipt_unadopted_by_supervisor"] = True
    if isinstance(receipt, dict):
        result["persisted_post_audit"] = {
            "passed": receipt.get("passed"),
            "recorded_episodes": receipt.get("recording", {}).get("episodes"),
            "decoded_episodes": receipt.get("decode", {}).get("episodes"),
            "all_numeric_sha256_verified": receipt.get("recording", {}).get(
                "all_numeric_sha256_verified"
            ),
            "checkpoint_verified": receipt.get("training", {}).get(
                "checkpoint_verified"
            ),
            "qualification_scope": "Receipt alone cannot certify audit process-group cleanup or an unrun evaluation",
        }
    # Failed post-audit supervision may leave evaluation pending. That is a
    # terminal failed attempt, not a running suite and not a qualified policy.
    result["evaluation_evidence"] = {
        "available": (directory / "evaluation.json").is_file(),
        "supervisor_status": summary.get("evaluation_status"),
        "pending_after_failure_is_not_a_pass": True,
    }
    if verify_checkpoints:
        artifacts_exist = all(
            (directory / name).is_file() for name in ("initial.pt", "final.pt")
        )
        if artifacts_exist:
            try:
                initial = checkpoint_model_hash(directory / "initial.pt")
                final = checkpoint_model_hash(directory / "final.pt")
                result["checkpoint_cpu_audit"] = {"initial": initial, "final": final}
                if not (
                    initial["model_sha256"] == metrics["initial_model_sha256"]
                    and final["model_sha256"] == metrics["final_model_sha256"]
                    and final["normalization_sha256"] == metrics["normalization_sha256"]
                    and final["step"] == updates
                    and final["optimizer_steps"] == ([updates] if updates else [])
                    and (
                        updates == 0 or final["model_sha256"] != initial["model_sha256"]
                    )
                ):
                    errors.append(
                        "failed attempt CPU checkpoint/completed-work mismatch"
                    )
            except Exception as error:
                errors.append(f"failed attempt CPU checkpoint verification: {error!r}")
        elif training_complete:
            errors.append("full-budget failed attempt lacks initial/final checkpoints")
        else:
            result["checkpoint_cpu_audit"] = "not produced before failure"
        has_video_receipt = (directory / "video-integrity.json").is_file() or (
            isinstance(receipt, dict)
            and isinstance(receipt.get("full_video_integrity"), dict)
        )
        if has_video_receipt:
            evidence_view = dict(summary)
            evidence_view["post_audit"] = receipt if isinstance(receipt, dict) else {}
            result["full_video_integrity"] = full_video_evidence(
                directory, evidence_view
            )
            if not result["full_video_integrity"]["passed"]:
                errors.extend(result["full_video_integrity"]["errors"])
    result.update(passed=not errors, errors=errors)
    return result


def audit_formal(
    root: Path, protocol: dict[str, Any], verify_checkpoints: bool
) -> dict[str, Any]:
    from examples.ods.data_stream_demo.training_report import (
        _load,
        _qualification,
        _paired,
    )

    suite = read(root / "suite-summary.json")
    if (
        suite.get("status") not in {"complete", "failed", "unsupported"}
        or len(suite.get("results", [])) != 30
    ):
        raise RuntimeError(
            "Formal suite is not terminal with all 30 outcomes; never package partial work"
        )
    loaded = _load(root)
    expected = {
        (arm, vision, seed)
        for arm in protocol["arms"]
        for vision in protocol["visions"]
        for seed in protocol["seeds"]
    }
    actual = {
        (run["arm"], run["config"]["vision"], run["config"]["seed"]) for run in loaded
    }
    if len(loaded) != 30 or actual != expected:
        raise RuntimeError("Formal 30-cell matrix has duplicates/missing outcomes")
    if suite.get("source_sha256") != protocol["source_sha256"]:
        raise RuntimeError("Frozen source hashes differ from suite")
    results, hard_errors = [], []
    for run in loaded:
        directory, summary, config, metrics = (
            Path(run["path"]).parent,
            run["summary"],
            run["config"],
            run["metrics"],
        )
        row = {
            "path": str(directory),
            "arm": run["arm"],
            "vision": config["vision"],
            "seed": config["seed"],
            "status": summary["status"],
            "report_qualification_reasons": _qualification(run),
            "failure_details": (
                {
                    key: summary.get(key)
                    for key in (
                        "error",
                        "quality_error",
                        "post_audit_status",
                        "evaluation_status",
                        "requested_evaluation_scenes",
                    )
                }
                if summary["status"] != "complete"
                else None
            ),
        }
        if metrics and (directory / "training-events.jsonl").exists():
            try:
                row["independent_metrics"] = reconstruct_metrics(directory)
            except Exception as error:
                row["independent_metrics"] = {"passed": False, "error": repr(error)}
        if summary["status"] == "complete":
            errors = row["report_qualification_reasons"].copy()
            if not row.get("independent_metrics", {}).get("passed"):
                errors.append("independent event reconstruction failed")
            for key, value in (
                ("completed_updates", 128),
                ("trained_chunks", 4096),
                ("valid_target_occurrences", 131072),
            ):
                if metrics.get(key) != value:
                    errors.append(f"formal completed budget {key}")
            audit, evaluation = summary.get("post_audit", {}), summary.get(
                "evaluation", {}
            )
            generated = metrics.get("published_sources")
            if not (
                audit.get("passed") is True
                and audit.get("recording", {}).get("episodes") == generated
                and audit.get("decode", {}).get("episodes") == generated
                and audit.get("recording", {}).get("all_numeric_sha256_verified")
                is True
                and audit.get("training", {}).get("checkpoint_verified") is True
            ):
                errors.append("full corpus SDK/numeric/checkpoint audit missing")
            if not (
                summary.get("requested_evaluation_scenes") == 20
                and evaluation.get("scenes") == 20
                and len(evaluation.get("records", [])) == 20
                and evaluation.get("passed") is True
                and evaluation.get("closed") is True
                and evaluation.get("seed") == protocol["evaluation_seed"]
                and evaluation.get("policy_step") == 128
                and evaluation.get("expert_initialized") is False
            ):
                errors.append(
                    "20-state independent evaluation incomplete or wrong fixed policy/scenes"
                )
            row["evaluation_scene_digest"] = evaluation.get("scene_digest")
            if verify_checkpoints:
                row["full_video_integrity"] = full_video_evidence(directory, summary)
                if not row["full_video_integrity"]["passed"]:
                    errors.extend(row["full_video_integrity"]["errors"])
            process = read(directory / "process.json")
            if not equivalent(process["elapsed_seconds"], metrics["total_seconds"]):
                errors.append("supervisor total disagreement")
            if verify_checkpoints:
                initial = checkpoint_model_hash(directory / "initial.pt")
                final = checkpoint_model_hash(directory / "final.pt")
                row["checkpoint_cpu_audit"] = {"initial": initial, "final": final}
                if not (
                    initial["model_sha256"] == metrics["initial_model_sha256"]
                    and final["model_sha256"] == metrics["final_model_sha256"]
                    and final["normalization_sha256"] == metrics["normalization_sha256"]
                    and final["step"] == 128
                    and final["optimizer_steps"] == [128]
                    and final["model_sha256"] != initial["model_sha256"]
                    and evaluation.get("checkpoint_sha256") == final["file_sha256"]
                ):
                    errors.append(
                        "CPU checkpoint/model/optimizer/evaluation-target verification failed"
                    )
            row["independent_errors"] = errors
            hard_errors.extend(f"{directory}: {error}" for error in errors)
        elif summary["status"] == "failed":
            row["failed_attempt_evidence"] = failed_run_evidence(
                directory, summary, row.get("independent_metrics"), verify_checkpoints
            )
            errors = row["failed_attempt_evidence"]["errors"].copy()
            if not row["report_qualification_reasons"]:
                errors.append(
                    "failed attempt unexpectedly promoted by report qualification"
                )
            row["independent_errors"] = errors
            hard_errors.extend(f"{directory}: {error}" for error in errors)
        results.append(row)
        print(
            f"Audit {run['arm']}/{config['vision']}/{config['seed']}: {summary['status']}",
            flush=True,
        )
    scene_digests = {
        row.get("evaluation_scene_digest")
        for row in results
        if row["status"] == "complete"
    }
    if scene_digests and (None in scene_digests or len(scene_digests) != 1):
        hard_errors.append(
            "Completed runs did not evaluate the identical fixed 20-scene set"
        )
    serial = []
    for vision in protocol["visions"]:
        for seed in protocol["seeds"]:
            directories = [
                root / arm / vision / f"seed{seed}"
                for arm in ("ods_overlap", "ods_serial")
            ]
            summaries = [read(d / "summary.json") for d in directories]
            record = {
                "vision": vision,
                "seed": seed,
                "statuses": [s["status"] for s in summaries],
            }
            if all(s["status"] == "complete" for s in summaries):
                traces = [read(d / "training-trace.json") for d in directories]
                numeric = [
                    [
                        {
                            k: e[k]
                            for k in (
                                "ordinal",
                                "generation",
                                "length",
                                "numeric_sha256",
                            )
                        }
                        for e in t["episodes"]
                    ]
                    for t in traces
                ]
                record.update(
                    accepted_numeric_trajectories_equal=numeric[0] == numeric[1],
                    ordered_training_manifest_equal=traces[0]["updates"]
                    == traces[1]["updates"],
                    recipe_equal=traces[0]["recipe"] == traces[1]["recipe"],
                    initial_model_equal=traces[0]["initial_model_sha256"]
                    == traces[1]["initial_model_sha256"],
                    normalization_equal=traces[0]["normalization_sha256"]
                    == traces[1]["normalization_sha256"],
                    final_model_sha256_equal=summaries[0]["training_metrics"][
                        "final_model_sha256"
                    ]
                    == summaries[1]["training_metrics"]["final_model_sha256"],
                )
                for suffix, key in (
                    (".npy", "retained_raw_reference_hashes_equal"),
                    (".mp4", "encoded_video_file_hashes_equal"),
                ):
                    hashes = [
                        {
                            str(p.relative_to(d / "lerobot_dataset")): digest(p)
                            for p in (d / "lerobot_dataset").rglob("*" + suffix)
                        }
                        for d in directories
                    ]
                    record[key] = (
                        (hashes[0] == hashes[1]) if hashes[0] and hashes[1] else None
                    )
                mandatory = (
                    "accepted_numeric_trajectories_equal",
                    "ordered_training_manifest_equal",
                    "recipe_equal",
                    "initial_model_equal",
                    "normalization_equal",
                )
                if not all(record[k] for k in mandatory):
                    hard_errors.append(
                        f"serial mandatory replay contract: {vision}/{seed}"
                    )
                record["needs_final_model_analysis"] = not record[
                    "final_model_sha256_equal"
                ]
                record["visual_scope"] = (
                    "Only retained raw reference arrays compared; no full raw-RGB identity claim. Encoded container byte differences are diagnostic, not a data failure."
                )
            serial.append(record)
    return {
        "schema": "ods-independent-formal-audit/v1",
        "passed": not hard_errors,
        "hard_errors": hard_errors,
        "runs": results,
        "serial_replay": serial,
        "report_paired_checks": _paired(loaded),
        "qualified_main": sum(
            not r["report_qualification_reasons"] and r["arm"] != "ods_serial"
            for r in results
        ),
        "qualified_serial": sum(
            not r["report_qualification_reasons"] and r["arm"] == "ods_serial"
            for r in results
        ),
        "all_attempts": 30,
        "attempt_status_counts": dict(
            collections.Counter(r["status"] for r in results)
        ),
        "qualification_scope": "A passed independent evidence audit does not promote failed attempts; only report qualification controls comparisons",
        "recorded_sdk_audit_scope": "Original numeric/H2/ref QA plus the separately revisioned full-video-integrity/v1 sequential YUV receipt. These are distinct checks; no lossless RGB or every-SDK-seek claim.",
        "retained_video_bytes_verification": (
            "compressed video and metadata SHA256 checked"
            if verify_checkpoints
            else "not requested in log-only re-audit"
        ),
        "checkpoint_verification": (
            "CPU only" if verify_checkpoints else "not requested"
        ),
    }


def selected_files(formal: Path) -> list[Path]:
    selected = []
    for p in formal.rglob("*"):
        if not p.is_file() or "lerobot_dataset" in p.parts:
            continue
        relative = p.relative_to(formal)
        if (
            (
                p.suffix == ".json"
                and p.name.startswith(("post-audit", "full-video", "video-integrity"))
            )
            or p.name in DIRECT_NAMES
            or p.name
            in {
                "gpu-preflight.json",
                "native-library-hashes.json",
            }
        ):
            selected.append(p)
        elif "_supervision" in relative.parts and p.name == "launch.json":
            selected.append(p)
    return sorted(set(selected))


def roots_for(args: argparse.Namespace) -> dict[str, Path]:
    roots = {"formal": args.formal.resolve()}
    for label, value in DEFAULT_DIAGNOSTICS.items():
        candidate = Path(value)
        if candidate.resolve() != args.formal.resolve():
            roots[label] = candidate
    for item in args.extra_root:
        label, path = item.split("=", 1)
        if (
            not label
            or label in {".", ".."}
            or "/" in label
            or "\\" in label
            or label in roots
        ):
            raise ValueError("Extra root needs a unique simple LABEL=PATH")
        roots[label] = Path(path)
    return roots


def raw_files(roots: dict[str, Path]) -> list[tuple[str, Path]]:
    return [
        (label, p)
        for label, root in roots.items()
        if root.is_dir()
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and "lerobot_dataset" not in p.relative_to(root).parts
        and p.suffix in TEXT_SUFFIXES
    ]


def external_inventory(roots: dict[str, Path]) -> list[dict[str, Any]]:
    inventory = []
    for label, root in roots.items():
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and (
                "lerobot_dataset" in p.relative_to(root).parts or p.suffix == ".pt"
            ):
                inventory.append(
                    {
                        "root_label": label,
                        "path": str(p.resolve()),
                        "relative_path": str(p.relative_to(root)),
                        "bytes": p.stat().st_size,
                        "sha256": digest(p),
                        "included_in_bundle": False,
                    }
                )
        print(f"Inventoried external corpus/checkpoints: {label}", flush=True)
    return inventory


def manifest(bundle: Path) -> None:
    rows = [
        {
            "path": str(p.relative_to(bundle)),
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }
        for p in sorted(bundle.rglob("*"))
        if p.is_file() and p != bundle / "manifest.json"
    ]
    write(
        bundle / "manifest.json", {"format": "sha256-file-manifest-v1", "files": rows}
    )


def package(
    args: argparse.Namespace, audit: dict[str, Any], roots: dict[str, Path]
) -> dict[str, Any]:
    if not args.execute:
        raise RuntimeError(
            "Packaging requires --execute after the root agent's explicit trigger"
        )
    if not audit["passed"]:
        raise RuntimeError(
            "Independent audit failed; exact source data retained, no archive promoted"
        )
    target = args.bundle.resolve()
    if target.exists():
        raise FileExistsError(f"Never overwrite an existing result bundle: {target}")
    selected, logs = selected_files(args.formal), raw_files(roots)
    estimate = sum(p.stat().st_size for p in selected) + sum(
        p.stat().st_size for _, p in logs
    )
    disk = shutil.disk_usage(target.parent)
    write(
        args.audit_output.with_name("disk-preflight.json"),
        {
            "free_bytes": disk.free,
            "selected_plain_bytes": sum(p.stat().st_size for p in selected),
            "raw_uncompressed_bytes": sum(p.stat().st_size for _, p in logs),
            "conservative_packaging_bytes": 2 * estimate + 1024**3,
        },
    )
    if disk.free < 2 * estimate + 1024**3:
        raise RuntimeError("Insufficient disk headroom for staging and compression")
    stage = target.with_name(target.name + f".staging-{os.getpid()}")
    stage.mkdir(parents=True)
    input_index = []
    for p in selected:
        out = stage / "main" / p.relative_to(args.formal)
        out.parent.mkdir(parents=True, exist_ok=True)
        sha = digest(p)
        shutil.copyfile(p, out)
        if digest(out) != sha:
            raise RuntimeError(f"Copy changed source bytes: {p}")
        input_index.append(
            {
                "source": str(p),
                "bundle_path": str(out.relative_to(stage)),
                "bytes": p.stat().st_size,
                "sha256": sha,
            }
        )
    overview = read(args.formal / "suite-summary.json")
    overview.pop("report", None)
    overview.pop("calibration", None)
    overview["original_suite_summary"] = {
        "path": str(args.formal / "suite-summary.json"),
        "sha256": digest(args.formal / "suite-summary.json"),
        "retained_in_raw_archive": True,
    }
    write(stage / "main" / "suite-overview.json", overview)
    external = external_inventory(roots)
    write(
        stage / "external-artifacts.json",
        {
            "scope": "Original datasets/checkpoints remain at absolute local paths; no GB binary corpus is committed",
            "files": external,
        },
    )
    raw_index = []
    with (stage / "raw-logs.tar.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as archive:
                for label, p in logs:
                    member = str(Path(label) / p.relative_to(roots[label]))
                    sha = digest(p)
                    archive.add(p, arcname=member, recursive=False)
                    raw_index.append(
                        {
                            "source": str(p),
                            "member": member,
                            "bytes": p.stat().st_size,
                            "sha256": sha,
                        }
                    )
    with tarfile.open(stage / "raw-logs.tar.gz", "r:gz") as archived:
        members = {row["member"]: row for row in raw_index}
        if len(archived.getnames()) != len(members) or set(archived.getnames()) != set(
            members
        ):
            raise RuntimeError("Raw archive member set differs from exact input index")
        for member in archived:
            h = hashlib.sha256()
            stream = archived.extractfile(member)
            if stream is None:
                raise RuntimeError("Raw archive unexpectedly contains a nonfile")
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(chunk)
            if h.hexdigest() != members[member.name]["sha256"]:
                raise RuntimeError(f"Raw archive bytes changed: {member.name}")
    write(
        stage / "raw-log-files.json",
        {
            "missing_roots": [
                label for label, root in roots.items() if not root.is_dir()
            ],
            "files": raw_index,
        },
    )
    write(stage / "report-input-files.json", {"files": input_index})
    write(stage / "independent-audit.json", audit)
    shutil.copyfile(args.protocol, stage / "protocol.json")
    shutil.copyfile(Path(__file__), stage / "independent_audit.py")
    source_report = args.formal / "training-report"
    if source_report.is_dir():
        for p in source_report.iterdir():
            if p.is_file() and p.name != "report.json":
                out = stage / "report" / p.name
                out.parent.mkdir(exist_ok=True)
                shutil.copyfile(p, out)
    from examples.ods.data_stream_demo.training_report import (
        _load,
        _qualification,
        _paired,
        write_report,
    )

    replay = _load(stage / "main")
    if (
        sum(not _qualification(r) for r in replay)
        != audit["qualified_main"] + audit["qualified_serial"]
    ):
        raise RuntimeError(
            "Archived hierarchy no longer passes original fairness gates"
        )
    # Re-render into temporary scratch; retain original final report bytes.
    rerender = args.audit_output.parent / "archive-report-rerender"
    replay_report = write_report(stage / "main", rerender)
    original_report = source_report / "report.json"
    if original_report.exists():
        original = read(original_report)
        for key in (
            "aggregates",
            "resource_aggregates",
            "qualified_runs",
            "observed_runs",
            "qualified_paired_comparisons",
        ):
            if original.get(key) != replay_report.get(key):
                raise RuntimeError(
                    f"Re-rendered report differs from original measured {key}"
                )
    write(
        stage / "reproduction-check.json",
        {
            "passed": True,
            "all_runs": len(replay),
            "qualified_runs": replay_report["qualified_runs"],
            "qualified_paired_comparisons": replay_report[
                "qualified_paired_comparisons"
            ],
            "fairness_inputs_preserved": True,
            "raw_archive_sha256_verified_files": len(raw_index),
            "rerender_output": str(rerender),
        },
    )
    compact = {
        "schema": "ods-fixed-budget-comparison/v1",
        "formal_root": str(args.formal),
        "protocol_commit": read(args.protocol)["implementation_commit"],
        "attempts": 30,
        "qualified_main": audit["qualified_main"],
        "qualified_serial": audit["qualified_serial"],
        "attempt_status_counts": audit["attempt_status_counts"],
        "aggregates": replay_report["aggregates"],
        "resources": replay_report["resource_aggregates"],
        "paired_comparisons": _paired(replay),
        "serial_replay": audit["serial_replay"],
        "recorded_sdk_audit_scope": "Original all-source H2/numeric/ref QA and separate full-video-integrity/v1 sequential YUV certificate; no lossless RGB or exhaustive SDK seek claim.",
        "runs": [],
    }
    for run in replay:
        m, s, c = run["metrics"], run["summary"], run["config"]
        compact["runs"].append(
            {
                "arm": run["arm"],
                "vision": c["vision"],
                "seed": c["seed"],
                "status": s["status"],
                "qualification_reasons": _qualification(run),
                "failure_details": (
                    {
                        key: s.get(key)
                        for key in (
                            "error",
                            "quality_error",
                            "post_audit_status",
                            "evaluation_status",
                            "requested_evaluation_scenes",
                        )
                    }
                    if s["status"] != "complete"
                    else None
                ),
                "metrics": {
                    k: m.get(k)
                    for k in (
                        *METRIC_KEYS,
                        "total_seconds",
                        "training_complete_seconds",
                        "post_training_drain_seconds",
                        "published_sources",
                        "untrained_sources",
                        "max_source_trained_chunks",
                        "initial_model_sha256",
                        "normalization_sha256",
                        "final_model_sha256",
                    )
                },
                "evaluation": {
                    k: s.get("evaluation", {}).get(k)
                    for k in (
                        "scenes",
                        "success_rate",
                        "lift_rate",
                        "goal_error",
                        "episode_length",
                    )
                },
            }
        )
    write(stage / "comparison.json", compact)
    with (stage / "comparison.csv").open("w", newline="") as stream:
        fields = [
            "arm",
            "vision",
            "seed",
            "status",
            "total_seconds",
            "published_sources",
            "trained_sources",
            "effective_sources",
            "initial_xy_occupied_cells",
            "unique_target_frames",
            "completed_updates",
            "valid_target_occurrences",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in compact["runs"]:
            writer.writerow(
                {key: row.get(key, row["metrics"].get(key)) for key in fields}
            )
    (stage / ".gitattributes").write_text(
        "report/*.svg -text -whitespace\n*.csv -text -whitespace\n*.tar.gz -text\n"
    )
    (stage / "PACKAGING.md").write_text(
        "# Evidence packaging\n\nMain report inputs preserve original measured bytes under `main/`, including initial contracts, process records and supervisor training launches required by the fairness gates. Re-render with `python -m examples.ods.data_stream_demo.training_report MAIN_PATH --output SCRATCH_PATH`. All 30 outcomes remain present.\n\nOriginal text/JSON/event/resource/context logs and earlier failed/development/smoke/wait/calibration attempts are stored in `raw-logs.tar.gz`; `raw-log-files.json` records every member's original SHA256. Extract to a separate scratch directory for independent event reconstruction. For log-only re-audit without the external GB checkpoints, use this script with --mode audit --skip-checkpoints --formal SCRATCH/formal --protocol protocol.json; the new audit explicitly records checkpoint verification as not requested. No success or slow run is selected away.\n\nDatasets and checkpoints remain at the exact absolute paths in `external-artifacts.json`, with file hashes. They are not included as GB binaries. Copies or loss of those local paths affect artifact availability, not the preserved measurements. `comparison.json` is compact; full per-run data remain in `main/summary.json` files. The redundant generated giant report.json is reproducible rather than duplicated here. The full suite-summary.json, which embeds that same report again, is retained once in the compressed raw archive; main/suite-overview.json is an explicitly derived compact overview.\n\n`independent-audit.json` reconstructs completed-work source/frame/state diversity and checks replay manifests. Checkpoint verification uses CPU. Final-model equality and encoded-video-file parity are separately labeled diagnostics; no full raw-RGB identity is asserted. Post-timing SDK audit and 20-state evaluation do not contribute to pipeline time. The original numeric/H2 SDK audit covers each source plus fixed RGB references. New formal qualification also requires the separately revisioned full-video-integrity/v1 receipt: every saved AV1 frame is sequentially software-decoded to YUV420p with source/frame counts and compressed/metadata hashes checked. This is not lossless RGB fidelity or exhaustive SDK seek certification. Both receipt types are retained distinctly and both validations are outside pipeline time.\n\nHuman interpretation and final experiment conclusions are owned by the study README, not generated by this packaging tool.\n"
    )
    manifest(stage)
    stage.rename(target)
    return {
        "bundle": str(target),
        "files": sum(p.is_file() for p in target.rglob("*")),
        "bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file()),
        "raw_files": len(raw_index),
        "external_files": len(external),
        "disk_free_bytes_after": shutil.disk_usage(target).free,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formal", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument(
        "--repo", type=Path, default=Path("/root/sources/EmbodiChain-ods-stream-design")
    )
    p.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "/root/sources/EmbodiChain-ods-stream-design/examples/ods/data_stream_demo/results/2026-10-21-visual-training"
        ),
    )
    p.add_argument(
        "--mode", choices=("inspect", "audit", "package", "manifest"), default="inspect"
    )
    p.add_argument(
        "--audit-output",
        type=Path,
        default=Path("/tmp/ods-visual-training-independent-audit.json"),
    )
    p.add_argument("--extra-root", action="append", default=[])
    p.add_argument("--execute", action="store_true")
    p.add_argument(
        "--skip-checkpoints",
        "--skip-external-artifacts",
        action="store_true",
        help="Log-only re-audit after archive extraction; forbidden when packaging",
    )
    args = p.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    args.formal = args.formal.resolve()
    if args.mode == "manifest":
        manifest(args.bundle)
        return
    protocol = read(args.protocol)
    roots = roots_for(args)
    if args.mode == "inspect":
        print(
            json.dumps(
                {
                    "formal": str(args.formal),
                    "suite_status": (
                        read(args.formal / "suite-summary.json").get("status")
                        if (args.formal / "suite-summary.json").exists()
                        else "not terminal"
                    ),
                    "direct_report_input_bytes": sum(
                        p.stat().st_size for p in selected_files(args.formal)
                    ),
                    "raw_text_bytes": sum(
                        p.stat().st_size for _, p in raw_files(roots)
                    ),
                    "disk_free_bytes": shutil.disk_usage(args.formal).free,
                    "roots": {k: str(v) for k, v in roots.items()},
                    "packaging_performed": False,
                },
                indent=2,
            )
        )
        return
    if args.mode == "package" and args.skip_checkpoints:
        raise ValueError("Packaging must verify the original CPU checkpoints")
    changed_sources = [
        name
        for name, sha in protocol["source_sha256"].items()
        if not (args.repo / name).is_file() or digest(args.repo / name) != sha
    ]
    if changed_sources:
        raise RuntimeError(
            f"Audit/report code differs from frozen source hashes: {changed_sources}"
        )
    audit = audit_formal(
        args.formal, protocol, verify_checkpoints=not args.skip_checkpoints
    )
    write(args.audit_output, audit)
    if args.mode == "package":
        print(json.dumps(package(args, audit, roots), indent=2))
    else:
        print(
            json.dumps(
                {"audit": str(args.audit_output), "passed": audit["passed"]}, indent=2
            )
        )
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
