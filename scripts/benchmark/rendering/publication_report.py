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

"""Rebuild publication figures from frozen pure-rendering v2 artifacts.

No simulator, array library or GPU runtime is imported. Run with site packages
disabled: ``python -S -m scripts.benchmark.rendering.publication_report ROOT``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from scripts.benchmark.core.artifacts import (
    ArtifactStore,
    read_json,
    stable_hash,
    write_json,
)
from scripts.benchmark.core.contracts import MetricDefinition
from scripts.benchmark.core.records import validate_run_result
from scripts.benchmark.reporting.aggregation import aggregate_runs
from scripts.benchmark.reporting.comparison import compare_metric
from scripts.benchmark.reporting.figures import write_metric_csv
from scripts.benchmark.reporting.tables import render_markdown_table

from ._publication_figures import write_chart
from .quality_protocol import (
    protocol_by_id,
    protocol_for_modalities,
    protocol_requirements,
)
from .reference_precision import precision_quality_reasons

__all__ = ["rebuild_report"]

_METRICS = {
    "batch_steps_per_s": ("1/s", "completed batch captures"),
    "camera_frames_per_s": ("1/s", "camera exposures"),
    "gpixel_per_s": ("Gpixel/s", "camera pixels, without a modality multiplier"),
    "latency_p50_s": ("s", "per-run capture latency median"),
    "latency_p95_s": ("s", "per-run capture latency 95th percentile"),
    "state_publish_p50_s": ("s", "per-run state publication latency median"),
    "device_ready_p50_s": ("s", "per-run device-ready latency median"),
    "host_ready_p50_s": ("s", "per-run host-ready latency median"),
    "gpu_process_peak_bytes": ("byte", "sampled process GPU memory peak"),
    "gpu_process_steady_bytes": ("byte", "sampled steady process GPU memory"),
    "gpu_device_peak_bytes": ("byte", "sampled device-wide GPU memory peak"),
    "cpu_peak_rss_bytes": ("byte", "process resident memory peak"),
    "torch_peak_allocated_bytes": ("byte", "Torch allocator peak, not total VRAM"),
    "d2h_bytes_per_capture": ("byte/capture", "device-to-host transfers"),
    "host_copy_bytes_per_capture": ("byte/capture", "CPU memory copies"),
    "native_render_s": ("s", "native renderer timer, when available"),
    "assembly_operations_per_capture": (
        "operation/capture",
        "adapter tensor assembly operations",
    ),
    "assembly_output_bytes_per_capture": (
        "byte/capture",
        "materialized assembly outputs, not native transfer totals",
    ),
    **{
        f"{stage}_p{percentile}_s": (
            "s",
            f"per-run {stage} application completion P{percentile}",
        )
        for stage in (
            "render_submit_to_ready",
            "state_publish_to_ready",
            "state_publish_return_to_ready",
        )
        for percentile in (50, 95)
    },
}
_CONFIG_FIELDS = (
    "scene_id",
    "num_envs",
    "cameras_per_env",
    "width",
    "height",
    "modalities",
    "profile",
    "temporal_mode",
    "delivery",
)
_BINDINGS = ("config_sha256", "scene_sha256", "view_sha256", "profile_sha256")
_INVARIANTS = (
    *_BINDINGS,
    "hardware_id",
    "metrics.boundary",
    "metrics.completion",
    *(f"config.{key}" for key in _CONFIG_FIELDS),
    "_quality.reference_id",
    "_quality.protocol_sha256",
)


def _json(value: object) -> str:
    """Encode deterministic table and grouping values."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _number(value: object) -> bool:
    """Exclude booleans and non-finite values from measured populations."""
    return type(value) in (float, int) and math.isfinite(value)


def _sha256(path: Path) -> str:
    """Hash potentially large reference evidence with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(row: dict[str, Any], path: str) -> object:
    """Read a frozen nested invariant without importing a domain config."""
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _object(path: Path) -> dict[str, Any]:
    """Load an optional standard object artifact."""
    if not path.exists():
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    if (
        type(value.get("schema_version", 1)) is not int
        or value.get("schema_version", 1) != 1
    ):
        raise ValueError(f"Unsupported {path.name} schema_version")
    return value


def _load_rows(root: Path, definition: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the frozen plan denominator even if some run records are absent."""
    payload = read_json(root / "runs.json")
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise ValueError("runs.json must contain a list of objects")
    by_id: dict[str, dict[str, Any]] = {}
    for row in payload:
        validate_run_result(row)
        if not row.get("run_id") or row["run_id"] in by_id:
            raise ValueError("runs.json requires unique nonempty run_id values")
        by_id[row["run_id"]] = dict(row)
    manifest = _object(root / "manifest.json")
    planned = manifest.get("runs", [])
    if not isinstance(planned, list) or any(
        not isinstance(row, dict) for row in planned
    ):
        raise ValueError("manifest.json runs must be a list of objects")
    planned_ids = set()
    for plan in planned:
        run_id = plan.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("manifest.json planned runs require run_id")
        if run_id in planned_ids:
            raise ValueError("manifest.json planned runs require unique run_id values")
        planned_ids.add(run_id)
        if run_id in by_id and any(
            plan.get(key) != by_id[run_id].get(key)
            for key in ("backend", "case_id", "repeat")
        ):
            raise ValueError(f"Frozen plan identity differs for run {run_id}")
        if run_id not in by_id:
            by_id[run_id] = {
                **plan,
                "experiment_id": plan.get(
                    "experiment_id", definition.get("experiment_id", "unknown")
                ),
                "status": "not_run",
                "failure_reason_code": "not_run",
                "failure_reason": "Planned run has no frozen result record",
                "quality_status": "missing",
            }
    rows = []
    for row in by_id.values():
        validate_run_result(row)
        for key in ("backend", "case_id"):
            if not row.get(key):
                raise ValueError(f"Run {row['run_id']} is missing {key}")
        if type(row.get("repeat")) is not int:
            raise ValueError(f"Run {row['run_id']} is missing repeat")
        row.setdefault("experiment_id", definition.get("experiment_id", "unknown"))
        row.setdefault("config", {})
        if not isinstance(row["config"], dict):
            raise ValueError("run config must be an object")
        row.setdefault("metrics", {})
        if "metric_reasons" in row and not isinstance(row["metric_reasons"], dict):
            raise ValueError("metric_reasons must be an object")
        row.setdefault(
            "failure_reason_code", "not_run" if row["status"] == "not_run" else None
        )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["experiment_id"],
            row["case_id"],
            row["backend"],
            row["repeat"],
            row["run_id"],
        ),
    )


def _quality_policy_reasons(record: dict[str, Any]) -> list[str]:
    """Validate the complete immutable policy rather than a claimed shorter list."""
    try:
        policy = protocol_by_id(record.get("protocol_id", ""))
        kind = record.get("evaluation_kind")
        expected = protocol_requirements(policy["id"], kind)
    except (ValueError, TypeError):
        return ["quality_protocol"]
    reasons = []
    qualification = record.get("qualification", {})
    if (
        not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or qualification.get("failed_checks")
        or qualification.get("missing_checks")
    ):
        reasons.append("quality_qualification")
    for key, value in expected.items():
        if record.get(key) != value:
            reasons.append(f"quality_contract:{key}")
    if expected["unassessed_modalities"]:
        reasons.append("quality_unassessed_modalities")
    checks = record.get("checks", {})
    if not isinstance(checks, dict) or any(
        not isinstance(checks.get(key), dict) or checks[key].get("status") != "passed"
        for key in expected["required_checks"]
    ):
        reasons.append("quality_required_checks")
    if isinstance(checks, dict) and any(
        not isinstance(check, dict) or check.get("status") != "passed"
        for check in checks.values()
    ):
        reasons.append("quality_checks")
    metrics = record.get("metrics", {})
    thresholds = policy[
        "frame_thresholds" if kind == "frame" else "temporal_thresholds"
    ]
    if not isinstance(metrics, dict) or set(metrics) != set(thresholds):
        reasons.append("quality_metric_schema")
        metrics = metrics if isinstance(metrics, dict) else {}
    for name, (comparison, threshold, unit) in thresholds.items():
        metric = metrics.get(name, {})
        if not isinstance(metric, dict):
            reasons.append("quality_metrics")
            continue
        if (metric.get("comparison"), metric.get("threshold"), metric.get("unit")) != (
            comparison,
            threshold,
            unit,
        ):
            reasons.append(f"quality_metric_definition:{name}")
        if name not in expected["required_metrics"]:
            if not (
                metric.get("required") is False
                and metric.get("status") == "not_requested"
                and metric.get("value") is None
                and metric.get("reason") == "modality_not_requested"
            ):
                reasons.append(f"quality_optional_metric:{name}")
            continue
        value = metric.get("value")
        if (
            metric.get("required", True) is not True
            or metric.get("status") != "passed"
            or not _number(value)
            or not _number(metric.get("denominator"))
            or metric["denominator"] <= 0
            or (value < threshold if comparison == "ge" else value > threshold)
        ):
            reasons.append("quality_metrics")
    if kind == "sequence":
        if "sequences" in record:
            children = record["sequences"]
            coverage = record.get("coverage", {})
            expected_count = (
                coverage.get("sequence_groups") if isinstance(coverage, dict) else None
            )
            child_kind = "sequence"
        else:
            children = record.get("frames", [])
            history = record.get("timed_history")
            expected_count = policy["sequence_frames"]
            if policy.get("history_contract"):
                expected_count = (
                    history.get("sequence_frames")
                    if isinstance(history, dict)
                    else None
                )
            child_kind = "frame"
        if (
            not isinstance(children, list)
            or type(expected_count) is not int
            or expected_count < 1
            or len(children) != expected_count
        ):
            reasons.append("quality_sequence_frames")
        elif any(
            not isinstance(child, dict)
            or child.get("evaluation_kind") != child_kind
            or child.get("quality_status") != "qualified"
            or child.get("protocol_sha256") != record.get("protocol_sha256")
            or _quality_policy_reasons(child)
            for child in children
        ):
            reasons.append("quality_sequence_frame_contract")
        if policy.get("history_contract") and child_kind == "frame":
            pairs = record.get("pairs", [])
            if (
                type(expected_count) is not int
                or not isinstance(pairs, list)
                or len(pairs) != expected_count - 1
                or any(
                    not isinstance(pair, dict)
                    or pair.get("previous_frame") != index
                    or pair.get("current_frame") != index + 1
                    or pair.get("status") != "passed"
                    for index, pair in enumerate(pairs)
                )
            ):
                reasons.append("quality_history_pairs")
            for name in expected["required_metrics"]:
                denominator = (
                    expected_count
                    if name == "temporal_silhouette_iou"
                    else expected_count - 1 if type(expected_count) is int else None
                )
                if metrics.get(name, {}).get("denominator") != denominator:
                    reasons.append("quality_history_metric_population")
    return reasons


def _timed_history_reasons(
    root: Path, row: dict[str, Any], record: dict[str, Any], checksums: dict[Path, str]
) -> list[str]:
    """Independently bind a claimed schedule and aliases to the original run."""

    def checked(artifact: dict[str, Any]) -> Path:
        path = Path(artifact["path"])
        resolved = (root / path).resolve()
        if (
            path.is_absolute()
            or not resolved.is_relative_to(root.resolve())
            or not resolved.is_file()
        ):
            raise ValueError("History artifact is missing or outside the experiment")
        if resolved not in checksums:
            checksums[resolved] = _sha256(resolved)
        if checksums[resolved] != artifact["sha256"]:
            raise ValueError("History artifact checksum mismatch")
        return resolved

    def same_numbers(first: Any, second: Any) -> bool:
        if isinstance(first, list) and isinstance(second, list):
            return len(first) == len(second) and all(
                same_numbers(a, b) for a, b in zip(first, second)
            )
        if isinstance(first, dict) and isinstance(second, dict):
            return first.keys() == second.keys() and all(
                same_numbers(first[key], second[key]) for key in first
            )
        if _number(first) and _number(second):
            return math.isclose(first, second, rel_tol=0, abs_tol=1e-5)
        return first == second

    try:
        descriptor = record.get("timed_history")
        if not isinstance(descriptor, dict):
            return ["quality_timed_history"]
        from .history_contract import validate_history_proof

        path = checked(descriptor)
        proof = validate_history_proof(root, path, expected_original=row)
        if proof.get("valid") is not True:
            return [
                "quality_timed_history",
                *[
                    "quality_history:" + str(reason)
                    for reason in proof.get("reasons", [])
                ],
            ]
        policy = protocol_by_id(record.get("protocol_id", ""))
        if (
            policy.get("history_contract") != "timed_render_history/v1"
            or record.get("evaluation_kind") != "sequence"
        ):
            raise ValueError("Timed history requires its frozen sequence policy")
        count = proof["schedule"]["measured_frames"]
        schedule_hash = proof["manifest"]["schedule_sha256"]
        if (
            descriptor.get("measured_frames") != count
            or descriptor.get("sequence_frames") != count + 1
            or descriptor.get("schedule_sha256") != schedule_hash
        ):
            raise ValueError(
                "History population/schedule differs from the acquired original"
            )
        evidence = {
            (item.get("path"), item.get("sha256"))
            for item in record.get("artifacts", [])
        }
        required = [*proof["artifacts"], descriptor]
        if any((item["path"], item["sha256"]) not in evidence for item in required):
            raise ValueError("Quality record omits source history evidence")
        groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for frame in proof["frames"]:
            meta = frame["metadata"]
            groups[(meta["env_id"], meta["camera_name"])].append(frame)
        children = record.get("sequences", [])
        identities = [
            (child.get("env_id"), child.get("camera_name")) for child in children
        ]
        coverage = record.get("coverage", {})
        if (
            len(identities) != len(groups)
            or set(identities) != set(groups)
            or coverage.get("expected_samples") != len(proof["frames"])
            or coverage.get("sequence_frames") != count + 1
        ):
            raise ValueError(
                "History quality population does not cover the acquired streams"
            )
        for child, identity in zip(children, identities):
            if child.get("timed_history") != descriptor:
                raise ValueError("Sequence history binding differs from its parent")
            timeline_artifact = child["exposure_timeline"]
            timeline = read_json(checked(timeline_artifact))
            if (timeline_artifact["path"], timeline_artifact["sha256"]) not in evidence:
                raise ValueError(
                    "Exposure timeline is not retained as quality evidence"
                )
            if (
                timeline.get("kind") != "scoring_exposure_timeline/v1"
                or timeline.get("history") != descriptor
            ):
                raise ValueError("Exposure timeline has a different history binding")
            if (timeline.get("env_id"), timeline.get("camera_name")) != identity:
                raise ValueError("Exposure timeline relabels its environment/camera")
            aliases = timeline["aliases"]
            source = sorted(
                groups[identity],
                key=lambda item: item["metadata"]["history"]["scoring_index"],
            )
            if len(aliases) != count + 1:
                raise ValueError("Exposure timeline is incomplete")
            for index, (alias, frame) in enumerate(zip(aliases, source)):
                meta = frame["metadata"]
                if (
                    alias.get("history") != meta["history"]
                    or alias.get("replay_frame_id") != meta["frame_id"]
                    or alias.get("replay_time_s") != meta["time_s"]
                ):
                    raise ValueError(
                        "Scoring alias changed the immutable raw replay identity"
                    )
                if (
                    checked(alias["candidate_metadata"]) != frame["metadata_path"]
                    or checked(alias["candidate_arrays"]) != frame["arrays_path"]
                ):
                    raise ValueError("Scoring alias points to a different candidate")
                sources = alias["reference_sources"]
                ref_path = checked(sources["reference_metadata"])
                arrays = checked(sources["reference_arrays"])
                ref = read_json(ref_path)
                refmeta = ref["metadata"]
                if (ref_path.parent / ref["arrays"]["path"]).resolve() != arrays or ref[
                    "arrays"
                ]["sha256"] != sources["reference_arrays"]["sha256"]:
                    raise ValueError(
                        "Reference alias arrays are not bound to their metadata"
                    )
                if (
                    refmeta.get("scene_sha256") != meta["scene_sha256"]
                    or not same_numbers(
                        refmeta.get("world_from_instances"),
                        meta.get("world_from_instances"),
                    )
                    or any(
                        not same_numbers(refmeta["camera"][key], meta["camera"][key])
                        for key in ("intrinsics", "world_from_camera")
                    )
                ):
                    raise ValueError(
                        "Reference alias does not represent the captured replay geometry"
                    )
                for artifact in (
                    alias["candidate_metadata"],
                    alias["candidate_arrays"],
                    *sources.values(),
                ):
                    if (artifact["path"], artifact["sha256"]) not in evidence:
                        raise ValueError(
                            "Scoring alias source is not retained in quality evidence"
                        )
                if index:
                    pair = child["pairs"][index - 1]
                    previous = source[index - 1]["metadata"]
                    transition = (
                        "warmup_entry"
                        if index == 1
                        else (
                            "replay_wrap"
                            if meta["history"]["cycle"] != previous["history"]["cycle"]
                            else "continuous"
                        )
                    )
                    if (
                        pair.get("previous_replay_frame_id") != previous["frame_id"]
                        or pair.get("current_replay_frame_id") != meta["frame_id"]
                        or pair.get("transition") != transition
                    ):
                        raise ValueError(
                            "Temporal pairs omit or relabel an entry/wrap transition"
                        )
        return []
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        AttributeError,
    ) as error:
        return ["quality_timed_history", "quality_history:" + str(error)]


def _moving_performance(row: dict[str, Any]) -> bool:
    """Exclude only an unambiguously declared ordinary R-02 sequence."""
    experiment = row.get("experiment_id")
    configured = row["config"].get("experiment_id", experiment)
    return row["config"].get("temporal_mode") == "moving" and not (
        experiment == configured == "R-02"
    )


def _quality_evidence(
    root: Path, row: dict[str, Any], record: dict[str, Any], checksums: dict[Path, str]
) -> list[str]:
    """Reject unbound, missing, damaged or unqualified quality certificates."""
    reasons = []
    if row["config"].get("experiment_id", row.get("experiment_id")) != row.get(
        "experiment_id"
    ):
        reasons.append("quality_experiment_binding")
    if row.get("quality_status") not in {"qualified", "passed"}:
        reasons.append("run_quality_status")
    if record.get("quality_status") not in {"qualified", "passed"}:
        reasons.append("quality_evidence_status")
    qualification = record.get("qualification", {})
    if (
        not isinstance(qualification, dict)
        or qualification.get("passed") is not True
        or qualification.get("failed_checks")
        or qualification.get("missing_checks")
    ):
        reasons.append("quality_qualification")
    checks = record.get("checks", {})
    if (
        not isinstance(checks, dict)
        or not checks
        or any(
            not isinstance(check, dict) or check.get("status") != "passed"
            for check in checks.values()
        )
    ):
        reasons.append("quality_checks")
    reasons.extend(_quality_policy_reasons(record))
    reasons.extend(precision_quality_reasons(root, record))
    try:
        history_policy = protocol_by_id(record.get("protocol_id", "")).get(
            "history_contract"
        )
    except (TypeError, ValueError):
        history_policy = None
    if history_policy and not _moving_performance(row):
        reasons.append("quality_history_scope")
    requested = record.get("requested_modalities")
    workload_modalities = row["config"].get("modalities")
    if (
        not isinstance(requested, list)
        or not isinstance(workload_modalities, list)
        or any(
            not isinstance(value, str) for value in [*requested, *workload_modalities]
        )
        or set(requested) != set(workload_modalities)
    ):
        reasons.append("quality_requested_modalities")
    if (
        row["config"].get("temporal_mode") == "moving"
        and record.get("evaluation_kind") != "sequence"
    ):
        reasons.append("quality_temporal_coverage")
    if _moving_performance(row):
        reasons.extend(_timed_history_reasons(root, row, record, checksums))
    if "coverage" in record:
        coverage = record["coverage"]
        if (
            not isinstance(coverage, dict)
            or type(coverage.get("expected_samples")) is not int
            or coverage["expected_samples"] < 1
            or coverage.get("evaluated_samples") != coverage["expected_samples"]
        ):
            reasons.append("quality_sample_coverage")
    for key in _BINDINGS:
        if not row.get(key) or record.get(key) != row[key]:
            reasons.append(f"quality_binding:{key}")
    for key in ("reference_id", "protocol_sha256"):
        if not record.get(key):
            reasons.append(f"quality_missing:{key}")
    artifacts = record.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        reasons.append("quality_artifacts_missing")
        artifacts = []
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not artifact.get("sha256")
        ):
            reasons.append("quality_artifact_identity")
            continue
        path = Path(artifact["path"])
        resolved = (root / path).resolve()
        if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
            reasons.append("quality_artifact_outside_root")
        elif not resolved.is_file():
            reasons.append("quality_artifact_missing")
        else:
            if resolved not in checksums:
                checksums[resolved] = _sha256(resolved)
            if checksums[resolved] != artifact["sha256"]:
                reasons.append("quality_artifact_checksum")
    return sorted(set(reasons))


def _attach_quality(
    root: Path, rows: list[dict[str, Any]], definition: dict[str, Any]
) -> None:
    """Attach evidence separately from execution and the claimed quality state."""
    records = _object(root / "quality.json").get("runs", [])
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError("quality.json runs must be a list of objects")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or run_id in indexed:
            raise ValueError("quality.json requires unique run_id values")
        if not isinstance(record.get("metrics", {}), dict):
            raise ValueError("quality metrics must be an object")
        indexed[run_id] = record
    checksums: dict[Path, str] = {}
    for row in rows:
        record = indexed.get(row["run_id"], {})
        row["_quality"] = record
        row["quality_reasons"] = _quality_evidence(root, row, record, checksums)
        frozen_quality = definition.get("quality_protocol", {})
        assignments = frozen_quality.get("case_protocols", {})
        assignment = assignments.get(row["case_id"])
        if "case_protocols" in frozen_quality and assignment is None:
            row["quality_reasons"].append("frozen_quality_assignment_missing")
        if (
            not assignment
            and frozen_quality.get("id")
            and record.get("protocol_id") != frozen_quality["id"]
        ):
            row["quality_reasons"].append("frozen_quality_assignment:protocol_id")
        if assignment:
            original = record.get("original_quality_contract")
            extension = False
            if (
                _moving_performance(row)
                and isinstance(record.get("timed_history"), dict)
                and "quality_timed_history" not in row["quality_reasons"]
                and isinstance(original, dict)
            ):
                try:
                    known_original = protocol_requirements(
                        original["protocol_id"], original["evaluation_kind"]
                    )
                    proposed = protocol_for_modalities(
                        tuple(known_original["requested_modalities"]),
                        temporal_dt_s=1 / 30,
                        timed_history=True,
                    )
                    extension = (
                        original == assignment
                        and all(
                            original.get(key) == value
                            for key, value in known_original.items()
                        )
                        and not (
                            set(original) - set(known_original) - {"temporal_quality"}
                        )
                        and original.get(
                            "temporal_quality", "assessed_by_declared_policy"
                        )
                        in {"unassessed", "assessed_by_declared_policy"}
                        and record.get("protocol_id") == proposed["id"]
                        and record.get("protocol_sha256") == stable_hash(proposed)
                        and (
                            original["evaluation_kind"] == "frame"
                            or protocol_by_id(original["protocol_id"])["sequence_dt_s"]
                            == proposed["sequence_dt_s"]
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    extension = False
            for key in ("protocol_id", "protocol_sha256", "evaluation_kind"):
                if not extension and record.get(key) != assignment.get(key):
                    row["quality_reasons"].append(f"frozen_quality_assignment:{key}")
            policy = frozen_quality.get("policies", {}).get(
                assignment.get("protocol_id")
            )
            if policy is not None and stable_hash(policy) != assignment.get(
                "protocol_sha256"
            ):
                row["quality_reasons"].append("frozen_quality_policy_checksum")
        row["qualified_for_comparison"] = not row["quality_reasons"]


def _frozen_runtime_context(
    root: Path, expected: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Persist campaign context so default and CLI rebuilds retain its gate."""
    path = root / "frozen-runtime.json"
    index_path = root / "artifact_index.json"
    bindings = [
        item
        for item in (
            _object(index_path).get("artifacts", []) if index_path.exists() else []
        )
        if item.get("path") == path.name
    ]
    if bindings and (
        len(bindings) != 1
        or not path.is_file()
        or _sha256(path) != bindings[0]["sha256"]
    ):
        raise ValueError("Registered frozen runtime binding is missing or changed")
    if path.exists():
        record = _object(path)
        saved = record["runtimes"]
        if record.get("schema_version") != 1 or record.get("sha256") != stable_hash(
            saved
        ):
            raise ValueError("Frozen runtime binding checksum mismatch")
        if expected is not None and saved != expected:
            raise ValueError("Frozen runtime binding conflicts with the campaign")
        if not bindings:
            ArtifactStore(root).register_artifact(path, kind="frozen_runtime_binding")
        return saved
    if expected is not None:
        write_json(
            path,
            {
                "schema_version": 1,
                "runtimes": expected,
                "sha256": stable_hash(expected),
            },
        )
        ArtifactStore(root).register_artifact(path, kind="frozen_runtime_binding")
    return expected


def _runtime_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Return recorded runtime scope without host-specific interpreter paths."""
    software = row.get("software") or {}
    return {
        key: software.get(key)
        for key in (
            "python",
            "packages",
            "cuda_runtime",
            "torch_runtime",
            "isaaclab_commit",
        )
    }


def _check_frozen_runtime(
    rows: list[dict[str, Any]], expected: dict[str, Any] | None
) -> None:
    """Keep dependency drift outside frozen-baseline comparisons."""
    for row in rows:
        row["runtime_cohort_id"] = stable_hash(_runtime_identity(row))
        row["frozen_runtime_status"] = "not_requested"
        if expected is None or row["status"] != "completed":
            continue
        baseline = expected.get(row["backend"])
        recorded = row.get("software") or {}
        packages = recorded.get("packages") or {}
        if not baseline or not recorded.get("python") or not packages:
            status = "missing"
        else:
            status = (
                "matched"
                if recorded["python"] == baseline["python"]
                and all(
                    name in packages and packages[name] == version
                    for name, version in baseline["packages"].items()
                )
                else "mismatch"
            )
        row["frozen_runtime_status"] = status
        if status != "matched":
            row["quality_reasons"].append(f"frozen_runtime_{status}")
            row["qualified_for_comparison"] = False


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse core aggregation within separate experiment and hardware strata."""
    metrics = tuple(
        MetricDefinition(key, unit, population, "2.0")
        for key, (unit, population) in _METRICS.items()
    )
    partitions: dict[tuple[str, object, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        partitions[
            (
                row["experiment_id"],
                row.get("hardware_id"),
                _json(row["config"]),
                stable_hash(_runtime_identity(row)),
            )
        ].append(row)
    groups = []
    for (experiment_id, hardware_id, _, runtime_id), members in partitions.items():
        for group in aggregate_runs(members, metrics)["groups"]:
            selected = [
                row
                for row in members
                if row["backend"] == group["backend"]
                and row["case_id"] == group["case_id"]
                and row.get("config_sha256") == group["config_sha256"]
            ]
            # Core may associate an early failure with one known case hash.
            if len(selected) != group["scheduled"]:
                selected += [
                    row
                    for row in members
                    if row["backend"] == group["backend"]
                    and row["case_id"] == group["case_id"]
                    and row.get("config_sha256") is None
                    and row["status"] != "completed"
                    and row not in selected
                ]
            configs = {_json(row["config"]) for row in selected if row["config"]}
            group.update(
                experiment_id=experiment_id,
                hardware_id=hardware_id,
                runtime_cohort_id=runtime_id,
                runtime=_runtime_identity(selected[0]) if selected else {},
                frozen_runtime_status_counts=dict(
                    Counter(
                        row.get("frozen_runtime_status", "not_requested")
                        for row in selected
                    )
                ),
                config=json.loads(next(iter(configs))) if len(configs) == 1 else {},
                config_consistent=len(configs) == 1,
                source_runs=[row["run_id"] for row in selected],
                status_counts=dict(Counter(row["status"] for row in selected)),
                failure_reason_counts=dict(
                    Counter(
                        str(row.get("failure_reason_code") or "unspecified")
                        for row in selected
                        if row["status"] != "completed"
                    )
                ),
                qualified_runs=sum(
                    row["qualified_for_comparison"] and row["status"] == "completed"
                    for row in selected
                ),
            )
            groups.append(group)
    return groups


def _comparisons(
    rows: list[dict[str, Any]], definition: dict[str, Any]
) -> list[dict[str, Any]]:
    """Use paired repeats only, with quality proof and frozen semantics gates."""
    comparisons = []
    invariant_paths = tuple(
        dict.fromkeys((*_INVARIANTS, *definition.get("comparison_invariants", [])))
    )
    cases: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = dict(row)
        value["quality_status"] = (
            "qualified" if row["qualified_for_comparison"] else "not_qualified"
        )
        cases[(row["experiment_id"], row["case_id"])].append(value)
    for (experiment_id, case_id), members in cases.items():
        complete = [row for row in members if row["status"] == "completed"]
        for metric in (
            "batch_steps_per_s",
            "camera_frames_per_s",
            "gpixel_per_s",
            "latency_p50_s",
            "latency_p95_s",
        ):
            result = compare_metric(
                members,
                metric=metric,
                baseline="embodichain",
                candidate="isaaclab",
                invariant_paths=invariant_paths,
            )
            reasons = list(result.reasons)
            for backend in ("embodichain", "isaaclab"):
                if (
                    len(
                        {
                            stable_hash(_runtime_identity(row))
                            for row in complete
                            if row["backend"] == backend
                        }
                    )
                    > 1
                ):
                    reasons.append(f"mixed_runtime_cohorts:{backend}")
            reasons += [
                path
                for path in invariant_paths
                if any(_field(row, path) in (None, "", []) for row in complete)
            ]
            reasons += [reason for row in complete for reason in row["quality_reasons"]]
            if result.pairs < 2:
                reasons.append("need_at_least_two_independent_paired_repeats")
            if result.pairs != len(complete) // 2:
                reasons.append("missing_paired_metric_values")
            paired = []
            for repeat in sorted({row["repeat"] for row in complete}):
                pair = {
                    row["backend"]: row for row in complete if row["repeat"] == repeat
                }
                if set(pair) != {"embodichain", "isaaclab"}:
                    continue
                left = pair["embodichain"]["metrics"].get(metric)
                right = pair["isaaclab"]["metrics"].get(metric)
                if _number(left) and _number(right) and left > 0 and right > 0:
                    paired.append(
                        {
                            "repeat": repeat,
                            "ratio": left / right,
                            "embodichain_value": left,
                            "isaaclab_value": right,
                            "source_runs": [
                                pair[name]["run_id"]
                                for name in ("embodichain", "isaaclab")
                            ],
                        }
                    )
                elif _number(left) and _number(right):
                    reasons.append("nonpositive_comparison_metric")
            reasons = sorted(set(reasons))
            ratio = (
                statistics.median(pair["ratio"] for pair in paired)
                if paired and not reasons
                else None
            )
            mad = (
                statistics.median(abs(pair["ratio"] - ratio) for pair in paired)
                if ratio is not None
                else None
            )
            if reasons:
                for pair in paired:
                    pair["ratio"] = None
            comparisons.append(
                {
                    **asdict(result),
                    "experiment_id": experiment_id,
                    "case_id": case_id,
                    "eligible": not reasons,
                    "reasons": reasons,
                    "ratio": ratio,
                    "delta": result.delta if not reasons else None,
                    "ratio_definition": "embodichain / isaaclab; throughput above 1 favors embodichain; latency below 1 favors embodichain",
                    "uncertainty": {
                        "method": "median_absolute_deviation_of_paired_ratios",
                        "value": mad,
                    },
                    "paired_runs": paired,
                    "scheduled": len(members),
                    "completed": len(complete),
                    "source_runs": [row["run_id"] for row in members],
                    "config_sha256": sorted(
                        {
                            row["config_sha256"]
                            for row in members
                            if row.get("config_sha256")
                        }
                    ),
                }
            )
    return comparisons


def _missing_reason(row: dict[str, Any], metric: str) -> str | None:
    """Preserve domain missing reasons and execution failures without zeros."""
    if row["status"] != "completed":
        return str(row.get("failure_reason_code") or row["status"])
    if _number(row["metrics"].get(metric)):
        return None
    reasons = row.get("metric_reasons") or row.get("missing_metrics") or {}
    return str(reasons.get(metric) or "missing")


def _plot_records(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten independent-run aggregates without dropping missing groups."""
    records = []
    for group in groups:
        for metric, value in group["metrics"].items():
            records.append(
                {
                    **{
                        key: group.get(key)
                        for key in (
                            "experiment_id",
                            "case_id",
                            "backend",
                            "hardware_id",
                            "runtime_cohort_id",
                            "frozen_runtime_status_counts",
                            "runtime",
                            "config_sha256",
                            "scheduled",
                            "completed",
                            "qualified_runs",
                        )
                    },
                    **{key: group["config"].get(key) for key in _CONFIG_FIELDS},
                    "metric": metric,
                    **value,
                    "source_runs": value["source_runs"],
                    "planned_runs": group["source_runs"],
                    "mad": (value["uncertainty"] or {}).get("value"),
                    "uncertainty_method": "median_absolute_deviation_of_independent_process_runs",
                    "status_counts": group["status_counts"],
                    "failure_reason_counts": group["failure_reason_counts"],
                }
            )
    return records


def _write_csv(
    path: Path, records: list[dict[str, Any]], columns: tuple[str, ...]
) -> None:
    """Keep nested source/provenance values machine-readable in CSV cells."""
    write_metric_csv(
        path,
        [
            {
                key: _json(value) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            for row in records
        ],
        columns=columns,
    )


def _scope(group: dict[str, Any], *, omit: tuple[str, ...] = ()) -> dict[str, Any]:
    """Define a chart stratum without crossing hardware or rendering semantics."""
    cfg = group["config"]
    return {
        "hardware_id": group.get("hardware_id"),
        "runtime_scope": (
            "frozen_baseline"
            if group.get("frozen_runtime_status_counts")
            == {"matched": group.get("scheduled")}
            else group.get("runtime_cohort_id", "unknown")
        ),
        **{
            key: cfg.get(key)
            for key in _CONFIG_FIELDS
            if key not in {"num_envs", "cameras_per_env", *omit}
        },
    }


def _capacity(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report observed endpoints, never infer a hard maximum from a partial sweep."""
    curves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if group["experiment_id"] in {"R-04", "R-05"}:
            curves[
                _json(
                    {
                        **_scope(group),
                        "backend": group["backend"],
                        "runtime_cohort_id": group.get("runtime_cohort_id"),
                        "experiment_id": group["experiment_id"],
                        "cameras_per_env": group["config"].get("cameras_per_env"),
                    }
                )
            ].append(group)
    records = []
    for scope, members in curves.items():
        successful = [
            g["config"]["num_envs"]
            for g in members
            if g["completed"] and _number(g["config"].get("num_envs"))
        ]
        oom = [
            g["config"]["num_envs"]
            for g in members
            if g["failure_reason_counts"].get("oom")
            and _number(g["config"].get("num_envs"))
        ]
        records.append(
            {
                **json.loads(scope),
                "max_observed_completed_num_envs": max(successful, default=None),
                "min_observed_oom_num_envs": min(oom, default=None),
                "scheduled": sum(g["scheduled"] for g in members),
                "completed": sum(g["completed"] for g in members),
                "source_runs": [run_id for g in members for run_id in g["source_runs"]],
                "config_sha256": [g["config_sha256"] for g in members],
                "interpretation": "observed endpoints only; missing, failed or untested larger N does not establish capacity",
            }
        )
    return records


def _charts(
    root: Path,
    groups: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Make scope-specific curves and diagnostics with both vector formats."""
    figures = []
    figure_data = []

    def emit(
        name: str,
        title: str,
        scope: dict[str, Any],
        series: dict[str, list[dict[str, Any]]],
        x_label: str,
        y_label: str,
        **kwargs: Any,
    ) -> None:
        stem = f"{name}-{hashlib.sha256(_json(scope).encode()).hexdigest()[:12]}"
        paths = write_chart(
            root / "figures" / stem,
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=[
                {"label": key, "points": value} for key, value in sorted(series.items())
            ],
            provenance={
                "schema_version": 1,
                "definition_version": "2.0",
                "scope": scope,
                "sources": sources,
                "series": series,
            },
            **kwargs,
        )
        figures.append(
            {
                "kind": name,
                "scope": scope,
                "title": title,
                "svg": paths[0].relative_to(root).as_posix(),
                "pdf": paths[1].relative_to(root).as_posix(),
            }
        )
        for label, points in sorted(series.items()):
            for item in points:
                figure_data.append(
                    {
                        "figure": stem,
                        "kind": name,
                        "scope": scope,
                        "series": label,
                        **item,
                    }
                )

    def point(
        group: dict[str, Any], metric: str, x: float, scale: float = 1.0
    ) -> dict[str, Any] | None:
        value = group["metrics"][metric]
        if value["median"] is None:
            return None
        median = value["median"] * scale
        mad = (value["uncertainty"] or {}).get("value", 0) * scale
        return {
            "x": x,
            "y": median,
            "low": max(0.0, median - mad),
            "high": median + mad,
            "run_ids": value["source_runs"],
            "config_sha256": group["config_sha256"],
            "valid_runs": value["valid_runs"],
            "scheduled": group["scheduled"],
            "metric": metric,
            "runtime_cohort_id": group.get("runtime_cohort_id"),
        }

    specs = (
        (
            "scaling",
            "Camera exposure scaling",
            ("camera_frames_per_s",),
            "camera frames/s",
            1.0,
        ),
        (
            "batch-scaling",
            "Batch capture scaling",
            ("batch_steps_per_s",),
            "batch steps/s",
            1.0,
        ),
        (
            "pixel-scaling",
            "Pixel throughput scaling",
            ("gpixel_per_s",),
            "Gpixel/s",
            1.0,
        ),
        (
            "latency",
            "Capture latency by environment count",
            ("latency_p50_s", "latency_p95_s"),
            "latency (ms)",
            1000.0,
        ),
        (
            "memory",
            "Measured memory by environment count",
            (
                "gpu_process_peak_bytes",
                "gpu_process_steady_bytes",
                "gpu_device_peak_bytes",
                "cpu_peak_rss_bytes",
            ),
            "memory (GiB; 2^30 bytes)",
            1.0 / 2**30,
        ),
        (
            "observation-latency",
            "Batch render submission to application-ready observations",
            ("render_submit_to_ready_p50_s", "render_submit_to_ready_p95_s"),
            "application-ready latency (ms); native source age unavailable",
            1000.0,
        ),
    )
    metric_labels = {
        "latency_p50_s": "P50",
        "latency_p95_s": "P95",
        "gpu_process_peak_bytes": "GPU process peak",
        "gpu_process_steady_bytes": "GPU process steady",
        "gpu_device_peak_bytes": "GPU device peak",
        "cpu_peak_rss_bytes": "CPU RSS",
        "render_submit_to_ready_p50_s": "submission to ready P50",
        "render_submit_to_ready_p95_s": "submission to ready P95",
    }
    scaling: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if (
            group["experiment_id"] in {"R-04", "R-05"}
            and _number(group["config"].get("num_envs"))
            and group["config"]["num_envs"] > 0
        ):
            scaling[_json(_scope(group))].append(group)
    for kind, title, metrics, label, scale in specs:
        for frozen, members in scaling.items():
            series: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for group in members:
                for metric in metrics:
                    item = point(group, metric, group["config"]["num_envs"], scale)
                    if item is not None:
                        suffix = (
                            f" / {metric_labels[metric]}" if len(metrics) > 1 else ""
                        )
                        backend = "EC" if group["backend"] == "embodichain" else "Isaac"
                        series[
                            f"{backend} {group['experiment_id']} ({group['config'].get('cameras_per_env')} cam) / runtime {group['runtime_cohort_id'][:8]}{suffix}"
                        ].append(item)
            emit(
                kind,
                title,
                json.loads(frozen),
                series,
                "environments N (log2 spacing)",
                label,
                log_x=True,
            )
        if not scaling:
            emit(kind, title, {}, {}, "environments N", label)

    for frozen, members in scaling.items():
        states = {"not_run": 0, "unsupported": 1, "failed": 2, "oom": 3, "completed": 4}
        series = defaultdict(list)
        for group in members:
            for row in rows:
                if row["run_id"] in group["source_runs"]:
                    state = (
                        "completed"
                        if row["status"] == "completed"
                        else str(row.get("failure_reason_code") or row["status"])
                    )
                    series[
                        f"{row['backend']} {row['experiment_id']} / runtime {group['runtime_cohort_id'][:8]} / {state}"
                    ].append(
                        {
                            "x": group["config"]["num_envs"],
                            "y": states.get(state, 2),
                            "run_ids": [row["run_id"]],
                            "config_sha256": row.get("config_sha256"),
                            "status": row["status"],
                            "failure_reason_code": row.get("failure_reason_code"),
                        }
                    )
        emit(
            "capacity",
            "Observed completion and OOM endpoints",
            json.loads(frozen),
            series,
            "environments N (log2 spacing)",
            "execution outcome",
            connect=False,
            log_x=True,
            y_ticks={value: key for key, value in states.items()},
        )
    if not scaling:
        emit(
            "capacity",
            "Observed completion and OOM endpoints",
            {},
            {},
            "environments N",
            "execution outcome",
            connect=False,
        )

    delivery: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        if group["experiment_id"] == "R-09":
            delivery[
                _json(
                    {
                        **_scope(group, omit=("delivery",)),
                        "num_envs": group["config"].get("num_envs"),
                        "cameras_per_env": group["config"].get("cameras_per_env"),
                    }
                )
            ].append(group)
    for frozen, members in delivery.items():
        series = defaultdict(list)
        for group in members:
            for index, metric in enumerate(
                ("state_publish_p50_s", "device_ready_p50_s", "host_ready_p50_s")
            ):
                item = point(group, metric, float(index), 1000)
                if item is not None:
                    series[
                        f"{group['backend']} / runtime {group['runtime_cohort_id'][:8]} / {group['config'].get('delivery')}"
                    ].append(item)
        emit(
            "delivery",
            "Reported delivery-stage medians (not additive)",
            json.loads(frozen),
            series,
            "instrumented stage",
            "reported stage latency (ms)",
            connect=False,
            x_ticks={0: "publish", 1: "device ready", 2: "host ready"},
        )
    if not delivery:
        emit(
            "delivery",
            "Reported delivery-stage medians (not additive)",
            {},
            {},
            "instrumented stage",
            "reported stage latency (ms)",
            connect=False,
        )

    quality: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for group in groups:
        throughput = group["metrics"]["camera_frames_per_s"]
        if throughput["median"] is None:
            continue
        observations: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = (
            defaultdict(list)
        )
        for row in rows:
            if row["run_id"] not in throughput["source_runs"]:
                continue
            for name, metric in row["_quality"].get("metrics", {}).items():
                if isinstance(metric, dict) and _number(metric.get("value")):
                    observations[name].append((row, metric))
        for name, observations_for_metric in observations.items():
            references = {
                observation[0]["_quality"].get("reference_id")
                for observation in observations_for_metric
            }
            if len(references) != 1 or not next(iter(references)):
                continue
            scope = {
                **_scope(group, omit=("profile",)),
                "num_envs": group["config"].get("num_envs"),
                "cameras_per_env": group["config"].get("cameras_per_env"),
                "reference_id": next(iter(references)),
                "quality_metric": name,
                "quality_unit": observations_for_metric[0][1].get("unit"),
            }
            x = statistics.median(
                observation[1]["value"] for observation in observations_for_metric
            )
            item = point(group, "camera_frames_per_s", x)
            y_values = [
                observation[0]["metrics"]["camera_frames_per_s"]
                for observation in observations_for_metric
            ]
            y = statistics.median(y_values)
            y_mad = statistics.median(abs(value - y) for value in y_values)
            x_mad = statistics.median(
                abs(observation[1]["value"] - x)
                for observation in observations_for_metric
            )
            item.update(
                y=y,
                low=max(0.0, y - y_mad),
                high=y + y_mad,
                x_low=x - x_mad,
                x_high=x + x_mad,
                valid_runs=len(y_values),
            )
            item["quality_source_runs"] = [
                observation[0]["run_id"] for observation in observations_for_metric
            ]
            item["quality_values"] = [
                observation[1]["value"] for observation in observations_for_metric
            ]
            item["run_ids"] = item["quality_source_runs"]
            quality[_json(scope)][
                f"{group['backend']} / runtime {group['runtime_cohort_id'][:8]} / {group['config'].get('profile')}"
            ].append(item)
    for frozen, series in quality.items():
        scope = json.loads(frozen)
        emit(
            "quality-performance",
            "Measured quality and throughput (qualification remains separate)",
            scope,
            series,
            f"{scope['quality_metric']} ({scope.get('quality_unit') or 'recorded unit'})",
            "camera frames/s",
            connect=False,
        )
    if not quality:
        emit(
            "quality-performance",
            "Measured quality and throughput",
            {},
            {},
            "quality metric unavailable",
            "camera frames/s",
            connect=False,
        )
    _write_csv(
        root / "figure_data.csv",
        figure_data,
        (
            "figure",
            "kind",
            "scope",
            "series",
            "metric",
            "x",
            "x_low",
            "x_high",
            "y",
            "low",
            "high",
            "valid_runs",
            "scheduled",
            "run_ids",
            "config_sha256",
            "quality_source_runs",
            "quality_values",
            "status",
            "failure_reason_code",
        ),
    )
    return figures


def _selection_coverage(
    root: Path, rows: list[dict[str, Any]]
) -> dict[str, int] | None:
    """Keep full-matrix and selected-attempt denominators in every rebuild path."""
    if not (root / "selection.json").exists():
        return None
    selection = _object(root / "selection.json")
    trials = selection.get("trials")
    if not isinstance(trials, list) or any(
        not isinstance(row, dict)
        or not row.get("run_id")
        or type(row.get("selected")) is not bool
        or type(row.get("runnable")) is not bool
        for row in trials
    ):
        raise ValueError(
            "selection.json requires planned trial identities and boolean selection states"
        )
    if len({row["run_id"] for row in trials}) != len(trials):
        raise ValueError("selection.json trial identities must be unique")
    selected = {row["run_id"] for row in trials if row["selected"]}
    return {
        "full_planned_trials": len(trials),
        "selected_trials": len(selected),
        "runnable_trials": sum(row["runnable"] for row in trials),
        "excluded_trials": len(trials) - len(selected),
        "started_processes": sum(row["status"] != "not_run" for row in rows),
        "completed_selected_trials": sum(
            row["status"] == "completed" and row["run_id"] in selected for row in rows
        ),
        "selected_not_run": sum(
            row["status"] == "not_run" and row["run_id"] in selected for row in rows
        ),
    }


def rebuild_report(
    root: Path, *, frozen_runtimes: dict[str, Any] | None = None
) -> Path:
    """Rebuild a v2 rendering report entirely from frozen standard artifacts.

    Args:
        root: Experiment directory containing ``runs.json`` and optional frozen
            definition, manifest and proof-linked quality records.
        frozen_runtimes: Optional backend inventories from a campaign. Missing
            or mismatching runtimes cannot support its frozen-baseline ratios.

    Returns:
        The generated ``report.md`` path. CSVs, SVG/PDF figures and aggregate
        ``summary.json``/``metrics.json`` are written alongside it.

    Raises:
        ValueError: Artifact schemas or run identities are invalid.

    .. attention::
        Performance completion never certifies quality. A cross-backend ratio
        requires checksummed matching quality evidence and at least two matched
        independent process repeats. Missing values and all scheduled failures
        remain in the report denominator.
    """
    root = Path(root).resolve()
    definition = _object(root / "definition.json")
    if definition.get("definition_version", "2.0") != "2.0":
        raise ValueError("publication report requires definition_version='2.0'")
    rows = _load_rows(root, definition)
    _attach_quality(root, rows, definition)
    _check_frozen_runtime(rows, _frozen_runtime_context(root, frozen_runtimes))
    groups = _aggregate(rows)
    comparisons = _comparisons(rows, definition)
    plot_records = _plot_records(groups)
    capacity = _capacity(groups)
    sources = [
        {"path": name, "sha256": _sha256(root / name)}
        for name in (
            "frozen-runtime.json",
            "definition.json",
            "runs.json",
            "manifest.json",
            "quality.json",
            "assets_manifest.json",
            "selection.json",
        )
        if (root / name).is_file()
    ]
    figures = _charts(root, groups, rows, sources)
    summary = {
        "schema_version": 1,
        "definition_version": "2.0",
        "experiment_id": definition.get("experiment_id", "pure-rendering"),
        "scheduled_runs": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "groups": groups,
        "comparisons": comparisons,
        "capacity": capacity,
        "comparison_status": (
            "eligible"
            if comparisons and all(item["eligible"] for item in comparisons)
            else "not_qualified"
        ),
        "quality": [
            {
                "run_id": row["run_id"],
                "recorded_status": row.get("quality_status", "missing"),
                "qualified_for_comparison": row["qualified_for_comparison"],
                "reasons": row["quality_reasons"],
            }
            for row in rows
        ],
        "source_artifacts": sources,
        "figures": figures,
        "uncertainty": "median +/- median absolute deviation across independent process repeats; no frame pseudo-replication or confidence-interval claim",
    }
    coverage = _selection_coverage(root, rows)
    if coverage is not None:
        summary["selection_coverage"] = coverage
    write_json(root / "summary.json", summary)
    write_json(root / "metrics.json", summary)
    trial_records = [
        {
            **{
                key: row.get(key)
                for key in (
                    "experiment_id",
                    "case_id",
                    "run_id",
                    "backend",
                    "repeat",
                    "status",
                    "quality_status",
                    "hardware_id",
                    "runtime_cohort_id",
                    "frozen_runtime_status",
                    *_BINDINGS,
                    "failure_reason_code",
                )
            },
            **{key: row["config"].get(key) for key in _CONFIG_FIELDS},
            **{key: row["metrics"].get(key) for key in _METRICS},
            "failure_reason": row.get("failure_reason") or row.get("error"),
            "metric_reasons": {
                key: _missing_reason(row, key)
                for key in _METRICS
                if _missing_reason(row, key)
            },
            "qualified_for_comparison": row["qualified_for_comparison"],
            "quality_reasons": row["quality_reasons"],
            "included_in_aggregate": row["status"] == "completed",
        }
        for row in rows
    ]
    _write_csv(
        root / "metrics.csv",
        trial_records,
        (
            "experiment_id",
            "case_id",
            "run_id",
            "backend",
            "repeat",
            "status",
            "failure_reason_code",
            "failure_reason",
            "quality_status",
            "qualified_for_comparison",
            "quality_reasons",
            "hardware_id",
            "runtime_cohort_id",
            "frozen_runtime_status",
            *_BINDINGS,
            *_CONFIG_FIELDS,
            *_METRICS,
            "metric_reasons",
            "included_in_aggregate",
        ),
    )
    _write_csv(
        root / "plot_data.csv",
        plot_records,
        (
            "experiment_id",
            "case_id",
            "backend",
            "hardware_id",
            "runtime_cohort_id",
            "frozen_runtime_status_counts",
            "runtime",
            "config_sha256",
            *_CONFIG_FIELDS,
            "metric",
            "unit",
            "population",
            "definition_version",
            "aggregation",
            "median",
            "mad",
            "uncertainty_method",
            "scheduled",
            "completed",
            "qualified_runs",
            "valid_runs",
            "missing_runs",
            "denominator",
            "missing_reasons",
            "source_runs",
            "planned_runs",
            "status_counts",
            "failure_reason_counts",
        ),
    )
    _write_csv(
        root / "comparisons.csv",
        comparisons,
        (
            "experiment_id",
            "case_id",
            "metric",
            "eligible",
            "reasons",
            "baseline",
            "candidate",
            "ratio",
            "ratio_definition",
            "uncertainty",
            "pairs",
            "scheduled",
            "completed",
            "source_runs",
            "config_sha256",
            "paired_runs",
        ),
    )
    _write_csv(
        root / "capacity.csv",
        capacity,
        (
            "experiment_id",
            "backend",
            "hardware_id",
            "runtime_scope",
            "runtime_cohort_id",
            *_CONFIG_FIELDS,
            "max_observed_completed_num_envs",
            "min_observed_oom_num_envs",
            "scheduled",
            "completed",
            "source_runs",
            "config_sha256",
            "interpretation",
        ),
    )
    _write_markdown(root, summary, rows, plot_records)
    return root / "report.md"


def _write_markdown(
    root: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    plot_records: list[dict[str, Any]],
) -> None:
    """Write a report whose populations, exclusions and sources are reviewable."""
    lines = [
        "# Pure-rendering publication report",
        "",
        "Definition version 2.0; artifact schema version 1. Rebuilt offline from frozen files.",
        "",
        f"Scheduled trials: **{summary['scheduled_runs']}**. Status counts: `{_json(summary['status_counts'])}`.",
        "",
        "Execution status and quality qualification are independent. Unqualified timings are descriptive measurements; they do not establish equal-quality speedups. Hardware, scene, views, profile, modalities, temporal mode, delivery and completion boundaries must match before a comparison is eligible.",
        "",
        "Each point is a median of independent process repeats; whiskers show median absolute deviation (MAD), not a confidence interval. Per-frame samples are never counted as independent repeats. Per-run P95 values are aggregated as medians of run P95s. Missing metrics remain null; failed and not-run trials remain in scheduled denominators.",
        "",
        "Throughput counts batch steps, camera exposures and image pixels separately. GPU process memory, device-wide memory, CPU RSS and Torch allocation are distinct measurements. Native renderer time remains unavailable when the backend does not supply a native timer. Delivery-stage medians are not summed or subtracted to infer unmeasured stage costs.",
        "",
        "## Trial coverage and failures",
        "",
        render_markdown_table(
            [
                {
                    "Experiment": row["experiment_id"],
                    "Case": row["case_id"],
                    "Backend": row["backend"],
                    "Repeat": row["repeat"],
                    "Status": row["status"],
                    "Reason code": row.get("failure_reason_code"),
                    "Quality": row.get("quality_status", "missing"),
                    "Hardware": row.get("hardware_id") or "missing",
                    "Run ID": row["run_id"],
                    "Config SHA-256": row.get("config_sha256"),
                }
                for row in rows
            ],
            columns=(
                "Experiment",
                "Case",
                "Backend",
                "Repeat",
                "Status",
                "Reason code",
                "Quality",
                "Hardware",
                "Run ID",
                "Config SHA-256",
            ),
        ),
        "",
        "## Measurements",
        "",
        "Complete trial values and missing reasons: [metrics.csv](metrics.csv). All metrics, populations and plotting sources: [plot_data.csv](plot_data.csv).",
        "",
        render_markdown_table(
            [
                {
                    "Experiment": item["experiment_id"],
                    "Case": item["case_id"],
                    "Backend": item["backend"],
                    "Metric": item["metric"],
                    "Median": item["median"],
                    "MAD": item["mad"],
                    "Unit": item["unit"],
                    "Valid / completed / scheduled": f"{item['valid_runs']} / {item['completed']} / {item['scheduled']}",
                    "Missing reasons": _json(item["missing_reasons"]),
                }
                for item in plot_records
                if item["metric"]
                in {
                    "camera_frames_per_s",
                    "batch_steps_per_s",
                    "latency_p50_s",
                    "gpu_process_peak_bytes",
                }
            ],
            columns=(
                "Experiment",
                "Case",
                "Backend",
                "Metric",
                "Median",
                "MAD",
                "Unit",
                "Valid / completed / scheduled",
                "Missing reasons",
            ),
        ),
        "",
        "## Paired comparison eligibility",
        "",
        "Ratios use EmbodiChain / Isaac Lab. A throughput ratio above 1 favors EmbodiChain; a latency ratio below 1 favors EmbodiChain. Only matched process repeats with proof-linked qualified quality enter an eligible ratio. All paired values and source IDs are retained in [comparisons.csv](comparisons.csv).",
        "",
        render_markdown_table(
            [
                {
                    "Experiment": item["experiment_id"],
                    "Case": item["case_id"],
                    "Metric": item["metric"],
                    "Eligible": item["eligible"],
                    "Paired repeats": item["pairs"],
                    "Ratio": item["ratio"],
                    "Ratio MAD": item["uncertainty"]["value"],
                    "Reasons": ", ".join(item["reasons"]),
                }
                for item in summary["comparisons"]
            ],
            columns=(
                "Experiment",
                "Case",
                "Metric",
                "Eligible",
                "Paired repeats",
                "Ratio",
                "Ratio MAD",
                "Reasons",
            ),
        ),
        "",
        "## Capacity and OOM",
        "",
        "These are observed endpoints, not a certified hardware capacity. An OOM is retained as a failed trial with reason code `oom`; unsupported operations and unstarted cells remain distinct. Untested larger N and a partial sweep do not establish a maximum. See [capacity.csv](capacity.csv).",
        "",
        render_markdown_table(
            [
                {
                    "Experiment": item["experiment_id"],
                    "Backend": item["backend"],
                    "Hardware": item["hardware_id"] or "missing",
                    "Scope": _json(
                        {
                            key: item.get(key)
                            for key in _CONFIG_FIELDS
                            if key != "num_envs"
                        }
                    ),
                    "Max observed completed N": item["max_observed_completed_num_envs"],
                    "Min observed OOM N": item["min_observed_oom_num_envs"],
                    "Completed / scheduled": f"{item['completed']} / {item['scheduled']}",
                }
                for item in summary["capacity"]
            ],
            columns=(
                "Experiment",
                "Backend",
                "Hardware",
                "Scope",
                "Max observed completed N",
                "Min observed OOM N",
                "Completed / scheduled",
            ),
        ),
        "",
        "## Figures",
        "",
        "R-04 is the one-camera scaling track; R-05 is the three-camera scaling track. Curves remain separated by hardware and frozen workload semantics. SVG metadata and PDF document metadata contain the plotted run IDs, configuration hashes and source artifact hashes. Quality/performance points are observations, with qualification assessed separately.",
        "",
    ]
    for figure in summary["figures"]:
        lines += [
            f"### {figure['title']}",
            "",
            f"Scope: `{_json(figure['scope'])}`",
            "",
            f"![{figure['title']}]({figure['svg']})",
            "",
            f"[SVG]({figure['svg']}) · [PDF]({figure['pdf']})",
            "",
        ]
    if "selection_coverage" in summary:
        lines += [
            "## Selection coverage",
            "",
            render_markdown_table([summary["selection_coverage"]]),
            "",
            "Excluded and preflight-blocked trials remain not_run. A smoke or filtered execution does not complete the full matrix. Every selection reason and repeat policy is frozen in [selection.json](selection.json).",
            "",
        ]
    lines += [
        "## Frozen sources",
        "",
        render_markdown_table(summary["source_artifacts"], columns=("path", "sha256")),
        "",
        "Quality qualification failures and per-run reasons are retained in [summary.json](summary.json). Rebuilding the report never modifies runs, definition, manifest, raw observations or quality evidence.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(rebuild_report(parser.parse_args().root))
