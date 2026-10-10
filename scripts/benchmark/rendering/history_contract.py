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

"""Portable standard-library contract for exact timed-render exposure histories.

Raw replay identifiers/timestamps are never replaced by the derived scoring
timeline. A last-warmup guard plus every measured output is mandatory; the old
post-measurement hundred-frame replay cannot satisfy this contract.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

if __package__:
    from .frozen_source import (
        artifact,
        read_json,
        renderer_control_signature,
        sha256,
        stable_hash,
    )
else:  # Absolute worker scripts pin original packages, but use this new contract.
    from frozen_source import (
        artifact,
        read_json,
        renderer_control_signature,
        sha256,
        stable_hash,
    )

__all__ = ["build_history_schedule", "frame_history", "validate_history_proof"]

_BINDINGS = (
    "experiment_id",
    "run_id",
    "case_id",
    "backend",
    "repeat",
    "config_sha256",
    "scene_sha256",
    "view_sha256",
    "profile_sha256",
    "hardware_id",
)


def _inside(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("History evidence escapes its portable proof root")
    return path


def _logical(record: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("adapter_evidence", {})
    if (
        evidence.get("status") != "recorded"
        or type(evidence.get("render_submission_id")) is not int
    ):
        raise ValueError(
            "Original capture lacks recorded application submission/state evidence"
        )
    return evidence["logical_state"]


def build_history_schedule(
    config: dict[str, Any], replay: dict[str, Any], capture: dict[str, Any]
) -> dict[str, Any]:
    """Derive the full source-bound schedule, checking every original packet.

    Args:
        config: Original frozen worker configuration, including W and K.
        replay: Original immutable replay with raw IDs and timestamps.
        capture: Original capture.json, including warmup and measured records.

    Returns:
        Hashable schedule with an explicit entry guard, repeats and wrap seams.
    """
    case = config["case"]
    if (
        case.get("temporal_mode") != "moving"
        or case.get("experiment_id") in {"R-01", "R-02"}
        or case.get("deformation")
    ):
        raise ValueError("Timed history applies only to rigid moving performance cases")
    warmup, measured = config["warmup_frames"], config["measured_frames"]
    if type(warmup) is not int or warmup < 1:
        raise ValueError(
            "Zero-warmup history is unsupported: no genuine last-warmup guard"
        )
    if type(measured) is not int or measured < 1:
        raise ValueError("Measured history must contain at least one exposure")
    if (
        capture.get("counts", {}).get("batch_steps") != measured
        or capture.get("warmup_counts", {}).get("batch_steps") != warmup
    ):
        raise ValueError("Original capture counts disagree with its declared W/K")
    length, fps = replay["frame_count"], replay["fps"]
    if (
        type(length) is not int
        or length < 1
        or len(replay["frames"]) != length
        or fps != 30
    ):
        raise ValueError("History requires the complete frozen 30 Hz replay")
    names = (
        ["external_front"]
        if case["cameras_per_env"] == 1
        else ["external_front", "external_side", "wrist"]
    )
    rows = {}
    for phase, count, key in (
        ("warmup", warmup, "warmup_raw"),
        ("measured", measured, "raw"),
    ):
        recorded = capture.get(key, [])
        if len(recorded) != count:
            raise ValueError("Original capture population differs from frozen " + phase)
        rows[phase] = []
        for index, packet in enumerate(recorded):
            state = replay["frames"][index % length]
            logical = _logical(packet)
            if (
                packet.get("frame_index") != index
                or packet.get("delivery") != case["delivery"]
                or packet.get("render_calls") != 1
                or packet.get("state_publish_calls") != 1
                or packet.get("exposure_count")
                != case["num_envs"] * case["cameras_per_env"]
                or logical.get("frame_id") != state["frame_id"]
                or logical.get("time_s") != state["time_s"]
                or logical.get("camera_names") != names
                or logical.get("num_envs") != case["num_envs"]
                or logical.get("modalities") != case["modalities"]
            ):
                raise ValueError(
                    f"Original {phase} packet {index} differs from its frozen state/delivery"
                )
            rows[phase].append(
                {
                    "index": index,
                    "cycle": index // length,
                    "frame_id": state["frame_id"],
                    "time_s": state["time_s"],
                    "render_submission_id": packet["adapter_evidence"][
                        "render_submission_id"
                    ],
                    "state_sha256": stable_hash(state),
                }
            )
    all_packets = rows["warmup"] + rows["measured"]
    if any(
        right["render_submission_id"] != left["render_submission_id"] + 1
        for left, right in zip(all_packets, all_packets[1:])
    ):
        raise ValueError(
            "Original submission history is not consecutive across the warmup seam"
        )
    return {
        "kind": "exact_renderer_exposure_schedule/v1",
        "warmup_frames": warmup,
        "measured_frames": measured,
        "replay_frame_count": length,
        "fps": fps,
        "delivery": case["delivery"],
        "prelude": "original_constructor+original_four_capture_freshness",
        "freshness_capture_batches": 4,
        "warmup": rows["warmup"],
        "measured": rows["measured"],
        "guard": deepcopy(rows["warmup"][-1]),
        "guard_source": "last_original_warmup_output_without_extra_render",
        "entry_seam": {
            "from_frame_id": rows["warmup"][-1]["frame_id"],
            "to_frame_id": rows["measured"][0]["frame_id"],
            "measured_index": 0,
        },
        "wrap_measured_indices": [
            index for index in range(1, measured) if index % length == 0
        ],
    }


def frame_history(schedule: dict[str, Any], exposure_index: int) -> dict[str, Any]:
    """Create separate scoring aliases while preserving raw replay metadata.

    Args:
        schedule: Verified original exposure schedule.
        exposure_index: -1 for the last-warmup guard, otherwise measured index.

    Returns:
        Canonical metadata.history fields for one saved exposure batch.
    """
    if (
        type(exposure_index) is not int
        or not -1 <= exposure_index < schedule["measured_frames"]
    ):
        raise ValueError("History exposure index is outside the required population")
    return {
        "phase": "guard" if exposure_index == -1 else "measured",
        "exposure_index": exposure_index,
        "measured_index": None if exposure_index == -1 else exposure_index,
        "cycle": (
            None
            if exposure_index == -1
            else exposure_index // schedule["replay_frame_count"]
        ),
        "schedule_sha256": stable_hash(schedule),
        "scoring_index": exposure_index + 1,
        "scoring_time_s": (exposure_index + 1) / schedule["fps"],
    }


def _proof(
    root: Path, path: Path, expected_original: dict[str, Any] | None
) -> dict[str, Any]:
    manifest = read_json(path)
    proof_root = path.parent
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "timed_render_history/v1"
    ):
        raise ValueError("Not an exact timed-render history manifest")
    if (
        manifest.get("status") != "completed"
        or manifest.get("performance_claim") is not False
    ):
        raise ValueError(
            "History acquisition is incomplete or incorrectly claims performance"
        )
    artifacts = {
        path.relative_to(root).as_posix(): artifact(
            root, path, "timed_history_manifest"
        )
    }

    def verify(record: dict[str, Any]) -> Path:
        evidence = _inside(proof_root, record["path"])
        if sha256(evidence) != record["sha256"]:
            raise ValueError("History artifact checksum mismatch: " + record["path"])
        artifacts[evidence.relative_to(root).as_posix()] = artifact(
            root, evidence, record.get("kind", "timed_history_evidence")
        )
        return evidence

    original = manifest["original"]
    roles = {}
    for record in original["artifacts"]:
        role = record["role"]
        if role in roles:
            raise ValueError("Duplicate original evidence role")
        roles[role] = verify(record)
    required = {
        "result",
        "worker_config",
        "capture",
        "raw",
        "scene",
        "replay",
        "definition",
        "freshness",
    }
    if not required <= roles.keys():
        raise ValueError("History is missing original evidence artifacts")
    result, config, capture = (
        read_json(roles[name]) for name in ("result", "worker_config", "capture")
    )
    replay = read_json(roles["replay"])
    for field in _BINDINGS:
        if original.get(field) is None or original.get(field) != result.get(field):
            raise ValueError("Original history binding mismatch: " + field)
        if expected_original is not None and original[field] != expected_original.get(
            field
        ):
            raise ValueError("History proof belongs to another original run: " + field)
    if (
        result.get("status") != "completed"
        or read_json(roles["freshness"]).get("passed") is not True
    ):
        raise ValueError("Original run/freshness prefix did not complete")
    rows = [
        json.loads(line)
        for line in roles["raw"].read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("run_id") == original["run_id"]]
    ignored = {
        "record_id",
        "quality_status",
        "quality_evaluation_id",
        "quality_evidence",
        "quality",
    }
    clean = lambda row: {key: value for key, value in row.items() if key not in ignored}
    if len(rows) != 1 or clean(rows[0]) != clean(result):
        raise ValueError("Copied original result is not bound to its raw ledger")
    if expected_original is not None:
        report_fields = ignored | {"quality_reasons", "qualified_for_comparison"}
        for key, value in result.items():
            if (
                key not in report_fields
                and not key.startswith("_")
                and expected_original.get(key) != value
            ):
                raise ValueError(
                    "History changed an authoritative original result field: " + key
                )
        authoritative = expected_original.get("_history_artifact_hashes", {})
        original_run = _inside(
            root, expected_original.get("run_dir", expected_original["run_id"])
        )
        known_paths = {
            "result": original_run / "result.json",
            "worker_config": original_run / "worker_config.json",
            "capture": original_run / "capture.json",
            "raw": root / "raw.jsonl",
        }
        for role, actual in known_paths.items():
            # A standalone acquisition has its own raw.jsonl. Explicit trusted
            # hashes refer to the original, not that new acquisition ledger.
            digest = authoritative.get(role)
            if digest is None and actual.is_file():
                digest = sha256(actual)
            if not digest or sha256(roles[role]) != digest:
                raise ValueError(
                    "History original artifact lacks authoritative byte binding: "
                    + role
                )
    if (
        stable_hash(config["case"]) != original["config_sha256"]
        or config["case"] != result["config"]
    ):
        raise ValueError("Original workload changed")
    if (
        stable_hash(config["profile"]) != original["profile_sha256"]
        or stable_hash(read_json(roles["scene"])) != original["scene_sha256"]
    ):
        raise ValueError("Original profile or scene changed")
    for key, value in capture.get("metrics", {}).items():
        if result.get("metrics", {}).get(key) != value:
            raise ValueError("Original raw capture metrics differ from result: " + key)
    if any(capture.get(key) != result.get(key) for key in ("counts", "warmup_counts")):
        raise ValueError("Original capture counters differ from immutable result")
    software = result["software"]
    if original.get("runtime_versions") != software or original.get(
        "gpu_driver"
    ) != result.get("resources", {}).get("gpu", {}).get("driver"):
        raise ValueError("Original runtime/driver binding differs from its raw result")
    if original.get("source_commit") != software.get(
        "benchmark_commit"
    ) or original.get("source_files") != software.get("benchmark_source_sha256"):
        raise ValueError("History does not bind the original source commit/files")
    for name, digest in original["source_files"].items():
        if "source:" + name not in roles or sha256(roles["source:" + name]) != digest:
            raise ValueError("Original source snapshot mismatch: " + name)
    schedule = build_history_schedule(config, replay, capture)
    schedule_sha = stable_hash(schedule)
    if (
        manifest.get("schedule") != schedule
        or manifest.get("schedule_sha256") != schedule_sha
    ):
        raise ValueError(
            "History schedule differs from original warmup/measured population"
        )
    from .quality_protocol import protocol_for_modalities, protocol_requirements

    policy = protocol_for_modalities(
        tuple(config["case"]["modalities"]), timed_history=True, temporal_dt_s=1 / 30
    )
    requirements = protocol_requirements(policy["id"], "sequence")
    if (
        manifest.get("history_quality_protocol") != policy
        or manifest.get("history_quality_requirements") != requirements
    ):
        raise ValueError(
            "History scoring policy differs from the immutable history protocol"
        )
    acquisition = manifest["acquisition"]
    acquisition_result = read_json(verify(manifest["acquisition_result"]))
    if (
        acquisition_result.get("status") != "completed"
        or acquisition_result.get("performance_claim") is not False
        or acquisition_result.get("original_run_id") != original["run_id"]
        or acquisition_result.get("metrics") != {}
    ):
        raise ValueError(
            "History worker result did not complete as an untimed acquisition"
        )
    if (
        acquisition.get("fresh_process") is not True
        or acquisition.get("source_commit") != original["source_commit"]
    ):
        raise ValueError("History lacks a fresh worker bound to the original source")
    if acquisition.get("hardware_id") != original["hardware_id"] or acquisition.get(
        "gpu_driver"
    ) != original.get("gpu_driver"):
        raise ValueError("History acquisition hardware/driver differs from original")
    runtime = acquisition.get("runtime_versions", {})
    for key in ("packages", "python", "torch_runtime", "cuda_runtime"):
        if key in software and runtime.get(key) != software[key]:
            raise ValueError("History runtime differs from original: " + key)
    imported = acquisition.get("imported_sources", {})
    for name in (
        "scripts.benchmark.rendering.formal_worker",
        "scripts.benchmark.rendering.capture",
        "scripts.benchmark.core.measurement",
        "scripts.benchmark.rendering.backends.scene_" + original["backend"],
    ):
        record = imported.get(name, {})
        if (
            original["source_files"].get(record.get("path")) != record.get("sha256")
            or not record
        ):
            raise ValueError(
                "History did not verify original imported core/adapter sources"
            )
    if original["backend"] == "isaaclab" and acquisition.get(
        "isaaclab_commit"
    ) != software.get("isaaclab_commit"):
        raise ValueError("History external Isaac source differs from original")
    renderer = read_json(verify(manifest["renderer"]))
    if renderer_control_signature(
        renderer, original["backend"]
    ) != renderer_control_signature(result["renderer"], original["backend"]):
        raise ValueError("Fresh actual shader/light controls differ from the timed run")
    if read_json(verify(manifest["freshness"])).get("passed") is not True:
        raise ValueError("Replayed original freshness prefix did not pass")
    events = read_json(verify(manifest["capture_events"]))
    fresh, warm, measured = (
        events.get(name, []) for name in ("freshness", "warmup", "measured")
    )
    if (
        len(fresh) != 4
        or len(warm) != schedule["warmup_frames"]
        or len(measured) != schedule["measured_frames"]
    ):
        raise ValueError(
            "Acquired render-submission population does not match the original prefix/history"
        )
    expected_ids = [
        schedule["warmup"][0]["render_submission_id"] - 4 + i for i in range(4)
    ]
    expected_ids += [
        item["render_submission_id"]
        for item in schedule["warmup"] + schedule["measured"]
    ]
    if [
        event["packet"]["adapter_evidence"]["render_submission_id"]
        for event in fresh + warm + measured
    ] != expected_ids:
        raise ValueError(
            "Acquisition inserted, omitted or changed application render submissions"
        )
    for phase, actual, expected in (
        ("warmup", warm, schedule["warmup"]),
        ("measured", measured, schedule["measured"]),
    ):
        for index, (event, state) in enumerate(zip(actual, expected)):
            packet = event["packet"]
            logical = _logical(packet)
            original_packet = capture["warmup_raw" if phase == "warmup" else "raw"][
                index
            ]
            counters = (
                "render_calls",
                "state_publish_calls",
                "exposure_count",
                "synchronize_calls",
                "d2h_group_calls",
                "d2h_calls",
                "d2h_bytes",
                "host_copy_calls",
                "host_copy_bytes",
            )
            if (
                event.get("phase") != phase
                or event.get("index") != index
                or packet.get("frame_index") != index
                or packet.get("delivery") != schedule["delivery"]
                or packet.get("render_calls") != 1
                or logical.get("frame_id") != state["frame_id"]
                or logical.get("time_s") != state["time_s"]
                or any(packet.get(key) != original_packet.get(key) for key in counters)
                or logical != _logical(original_packet)
            ):
                raise ValueError("Acquired packet/state order differs from original")
    frame_manifest_path = verify(manifest["frames_manifest"])
    frame_manifest = read_json(frame_manifest_path)
    case = config["case"]
    envs = sorted({0, case["num_envs"] // 2, case["num_envs"] - 1})
    names = (
        ["external_front"]
        if case["cameras_per_env"] == 1
        else ["external_front", "external_side", "wrist"]
    )
    expected_count = (schedule["measured_frames"] + 1) * len(envs) * len(names)
    if (
        frame_manifest.get("representative_env_ids") != envs
        or frame_manifest.get("capture_batches") != schedule["measured_frames"] + 1
        or len(frame_manifest.get("frames", [])) != expected_count
    ):
        raise ValueError(
            "History frame population lacks the guard or measured exposures"
        )
    frames, seen = [], set()
    for entry in frame_manifest["frames"]:
        metadata_path = _inside(frame_manifest_path.parent, entry["path"])
        if sha256(metadata_path) != entry["sha256"]:
            raise ValueError("History frame metadata checksum mismatch")
        record = read_json(metadata_path)
        metadata = record["metadata"]
        history = metadata["history"]
        exposure = history["exposure_index"]
        if history != frame_history(schedule, exposure):
            raise ValueError("History scoring timeline aliases were altered")
        state_entry = (
            schedule["guard"] if exposure == -1 else schedule["measured"][exposure]
        )
        state = replay["frames"][state_entry["index"] % schedule["replay_frame_count"]]
        identity = (metadata["env_id"], metadata["camera_name"], exposure)
        if identity in seen or identity[0] not in envs or identity[1] not in names:
            raise ValueError("Duplicate or unexpected history exposure identity")
        seen.add(identity)
        for key, value in (
            ("frame_id", state["frame_id"]),
            ("time_s", state["time_s"]),
            ("world_from_objects", state["objects"]),
            ("backend", original["backend"]),
            ("scene_sha256", original["scene_sha256"]),
            ("workload_view_sha256", original["view_sha256"]),
            ("profile_sha256", original["profile_sha256"]),
        ):
            if metadata.get(key) != value:
                raise ValueError("History frame changed original state/binding: " + key)
        if metadata.get("view_sha256") != stable_hash(metadata["camera"]):
            raise ValueError("History frame camera checksum mismatch")
        arrays_path = _inside(metadata_path.parent, record["arrays"]["path"])
        if (
            arrays_path != _inside(frame_manifest_path.parent, entry["arrays_path"])
            or sha256(arrays_path) != record["arrays"]["sha256"]
            or entry["arrays_sha256"] != record["arrays"]["sha256"]
        ):
            raise ValueError("History frame arrays checksum/path mismatch")
        for evidence, kind in (
            (metadata_path, "timed_history_frame"),
            (arrays_path, "timed_history_arrays"),
        ):
            artifacts[evidence.relative_to(root).as_posix()] = artifact(
                root, evidence, kind
            )
        frames.append(
            {
                "metadata_path": metadata_path,
                "arrays_path": arrays_path,
                "metadata": metadata,
            }
        )
    expected_population = {
        (env, name, index)
        for env in envs
        for name in names
        for index in range(-1, schedule["measured_frames"])
    }
    if seen != expected_population:
        raise ValueError("History does not cover every guard/measured sample")
    producer = {}
    for evidence in manifest.get("producer_artifacts", []):
        role = evidence.get("role")
        if role in producer:
            raise ValueError("Duplicate history producer evidence role")
        producer[role] = verify(evidence)
    required_producer = {
        "request",
        "producer_source:history_worker.py",
        "producer_source:history_contract.py",
        "producer_source:frozen_source.py",
    }
    if not required_producer <= producer.keys():
        raise ValueError("History producer source/request provenance is missing")
    request = read_json(producer["request"])
    for key in (
        "original",
        "schedule",
        "schedule_sha256",
        "history_quality_protocol",
        "history_quality_requirements",
    ):
        if request.get(key) != manifest.get(key):
            raise ValueError("Producer request differs from completed history: " + key)
    return {
        "valid": True,
        "reasons": [],
        "root": root,
        "proof_root": proof_root,
        "manifest_path": path,
        "manifest": manifest,
        "original": original,
        "schedule": schedule,
        "schedule_sha256": schedule_sha,
        "frames_manifest_path": frame_manifest_path,
        "frames": frames,
        "renderer": renderer,
        "artifacts": list(artifacts.values()),
        "representative_env_ids": envs,
        "camera_names": names,
        "measured_frames": schedule["measured_frames"],
        "sequence_frames": schedule["measured_frames"] + 1,
    }


def validate_history_proof(
    root: Path,
    manifest_path: Path | str,
    expected_original: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a portable history proof using hashes and frozen original semantics.

    Args:
        root: Allowed evidence root, including an imported proof directory.
        manifest_path: Absolute path or path relative to root of history.json.
        expected_original: Optional original run whose bindings must all match.

    Returns:
        ``valid``, explicit reasons, parsed frames with absolute paths, verified
        fresh renderer metadata and an artifact list relative to root. No NumPy,
        Torch, simulator import or status-bit promotion is performed.
    """
    root = Path(root).resolve()
    try:
        path = _inside(root, manifest_path)
        return _proof(root, path, expected_original)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
    ) as exc:
        return {
            "valid": False,
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "root": root,
            "frames": [],
            "artifacts": [],
        }
