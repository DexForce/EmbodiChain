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

"""Offline, paired R-05/C3 versus R-04/C1 throughput and memory increments.

Run with ``python -S -m scripts.benchmark.rendering.camera_scaling_report``.
These within-backend camera-count comparisons are descriptive scaling evidence,
not equal-quality speedups. Source experiments are never rewritten.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
from typing import Any, Sequence

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.reporting.tables import render_markdown_table
from ._publication_figures import write_chart
from .formal import view_fingerprint
from .publication_report import (
    _attach_quality,
    _field,
    _load_rows,
    _number,
    _object,
    _sha256,
)
from .scenes import load_scene

__all__ = ["main", "rebuild_camera_scaling_report"]

_METRICS = {
    "camera_frames_per_s": ("camera frames/s", ("ratio", "delta")),
    "batch_steps_per_s": ("batch steps/s", ("ratio", "delta")),
    "gpu_process_steady_bytes": ("byte", ("delta",)),
    "gpu_process_peak_bytes": ("byte", ("delta",)),
}
_CONFIG_FIELDS = (
    "scene_id",
    "num_envs",
    "width",
    "height",
    "modalities",
    "profile",
    "temporal_mode",
    "delivery",
)
_INVARIANTS = (
    "backend",
    "hardware_id",
    "scene_sha256",
    "profile_sha256",
    *(f"config.{name}" for name in _CONFIG_FIELDS),
    "metrics.boundary",
    "metrics.completion",
    "counts.batch_steps",
    "warmup_counts.batch_steps",
)
_CHANGED_CONFIG_FIELDS = {"experiment_id", "case_id", "cameras_per_env"}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _close(left: Any, right: Any) -> bool:
    """Compare observed camera floats without importing an array library."""
    if _number(left) and _number(right):
        return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-5)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _close(left[key], right[key]) for key in left
        )
    return left == right


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Frozen evidence path must stay inside its experiment root")
    return path


def _source(root: Path, role: str, path: Path) -> dict[str, Any]:
    return {
        "experiment": role,
        "root": str(root),
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
    }


def _matrix(value: Any, rows: int, columns: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == rows
        and all(
            isinstance(row, list)
            and len(row) == columns
            and all(_number(x) for x in row)
            for row in value
        )
    )


def _quaternion(value: list[float]) -> list[float] | None:
    if len(value) != 4 or not all(_number(x) for x in value):
        return None
    norm = math.sqrt(sum(x * x for x in value))
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        return None
    sign = next((1 if x > 0 else -1 for x in value if abs(x) > 1e-7), 1)
    return [sign * x / norm for x in value]


def _observed_front(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the common camera from native local or Isaac batched metadata."""
    renderer = row.get("renderer", {})
    if not isinstance(renderer, dict):
        return None
    names = renderer.get("camera_names", [])
    cameras = row["config"].get("cameras_per_env")
    n = row["config"].get("num_envs")
    if type(cameras) is not int or cameras not in (1, 3):
        return None
    expected_names = (
        ["external_front"]
        if cameras == 1
        else ["external_front", "external_side", "wrist"]
    )
    if names != expected_names or type(n) is not int or n < 1:
        return None
    index = names.index("external_front")
    intrinsics = renderer.get("intrinsics", [])
    poses = renderer.get("camera_poses", [])
    if not isinstance(intrinsics, list) or not isinstance(poses, list):
        return None
    if len(poses) == cameras and len(intrinsics) == cameras:
        if not _matrix(poses[index], 4, 4) or not _matrix(intrinsics[index], 3, 3):
            return None
        return {
            "scope": "local_view_repeated_in_arenas",
            "num_envs": n,
            "poses": [poses[index]],
            "intrinsics": [intrinsics[index]],
        }
    positions = renderer.get("camera_positions", [])
    quaternions = renderer.get("camera_quaternions_xyzw", [])
    if not isinstance(positions, list) or not isinstance(quaternions, list):
        return None
    if len(positions) == len(quaternions) == len(intrinsics) == n * cameras:
        indices = [env * cameras + index for env in range(n)]
        selected_positions = [positions[i] for i in indices]
        selected_quaternions = [quaternions[i] for i in indices]
        if not _matrix(selected_positions, n, 3) or not _matrix(
            selected_quaternions, n, 4
        ):
            return None
        normalized = [_quaternion(value) for value in selected_quaternions]
        if any(value is None for value in normalized) or any(
            not _matrix(intrinsics[i], 3, 3) for i in indices
        ):
            return None
        return {
            "scope": "world_views_all_environments",
            "num_envs": n,
            "positions": selected_positions,
            "quaternions": normalized,
            "intrinsics": [intrinsics[i] for i in indices],
        }
    return None


def _load_input(
    root: Path, role: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    definition = _object(root / "definition.json")
    if (
        definition.get("definition_version") != "2.0"
        or definition.get("experiment_id") != role
    ):
        raise ValueError(f"Expected a frozen formal v2 {role} experiment root")
    rows = _load_rows(root, definition)
    sources = {}

    def record(path: Path) -> None:
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in sources:
                sources[relative] = _source(root, role, path)

    for name in (
        "definition.json",
        "runs.json",
        "manifest.json",
        "quality.json",
        "selection.json",
        "assets_manifest.json",
        "approval.json",
    ):
        record(root / name)
    approval = _object(root / "approval.json")
    scenes: dict[str, Any] = {}
    for row in rows:
        reasons = []
        row["source_experiment"] = role
        row["source_root"] = str(root)
        row["source_id"] = f"{role}:{row['run_id']}"
        row["front_sha256"] = None
        row["_observed_front"] = None
        config_path = _inside(root, f"cases/{row['case_id']}/worker-config.json")
        worker = _object(config_path)
        record(config_path)
        if not row["config"] and row["status"] != "completed" and worker:
            row["config"] = deepcopy(worker.get("case", {}))
            row["config_origin"] = "frozen_worker_config_for_uncompleted_trial"
        config = row["config"]
        if row.get("experiment_id") != role or config.get("experiment_id") != role:
            reasons.append("experiment_identity")
        expected_cameras = 1 if role == "R-04" else 3
        if config.get("cameras_per_env") != expected_cameras:
            reasons.append("camera_count")
        if config.get("temporal_mode") != "static":
            reasons.append("static_scaling_workload_required")
        if row.get("config_sha256") != stable_hash(config):
            reasons.append("config_checksum")
        if not worker:
            reasons.append("frozen_worker_config_missing")
        elif worker.get("case") != config:
            reasons.append("frozen_worker_case_mismatch")
        elif not worker.get("profile") or stable_hash(worker["profile"]) != row.get(
            "profile_sha256"
        ):
            reasons.append("frozen_profile_checksum")
        if worker:
            for path, value in (
                ("counts.batch_steps", worker.get("measured_frames")),
                ("warmup_counts.batch_steps", worker.get("warmup_frames")),
            ):
                if row["status"] == "completed" and _field(row, path) != value:
                    reasons.append(f"frozen_window:{path}")
        scene_id = config.get("scene_id")
        try:
            if not isinstance(scene_id, str) or not scene_id:
                raise ValueError("missing scene_id")
            scene_path = _inside(root, f"scenes/{scene_id}/scene.json")
            if scene_id not in scenes:
                scenes[scene_id] = load_scene(scene_path)
                record(scene_path)
                for obj in scenes[scene_id]["objects"]:
                    if obj["geometry"]["kind"] == "mesh":
                        record(
                            _inside(
                                root, f"scenes/{scene_id}/{obj['geometry']['path']}"
                            )
                        )
            scene = scenes[scene_id]
            if stable_hash(scene) != row.get("scene_sha256"):
                reasons.append("scene_checksum")
            if (
                approval.get("status") != "approved"
                or approval.get("scope") != "scene_layout_and_camera_scheme"
                or approval.get("scenes", {}).get(scene_id) != row.get("scene_sha256")
            ):
                reasons.append("scene_approval_binding")
            expected_view = view_fingerprint(SimpleNamespace(**config), scene)
            if expected_view != row.get("view_sha256"):
                reasons.append("view_checksum")
            front = [
                view for view in scene["cameras"] if view["name"] == "external_front"
            ]
            if len(front) != 1:
                raise ValueError("external_front must be unique")
            row["front_sha256"] = stable_hash(
                {
                    "camera_model": scene["camera_model"],
                    "width": config["width"],
                    "height": config["height"],
                    "view": front[0],
                }
            )
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            reasons.append(f"scene_or_view_evidence:{type(exc).__name__}:{exc}")
        if row["status"] == "completed":
            row["_observed_front"] = _observed_front(row)
            if row["_observed_front"] is None:
                reasons.append("observed_external_front_missing")
            effective_profile = _field(row, "renderer.effective_profile")
            if effective_profile is not None and stable_hash(
                effective_profile
            ) != row.get("profile_sha256"):
                reasons.append("observed_profile_checksum")
            batch, frames = row["metrics"].get("batch_steps_per_s"), row["metrics"].get(
                "camera_frames_per_s"
            )
            n = config.get("num_envs")
            if (
                _number(batch)
                and _number(frames)
                and type(n) is int
                and not math.isclose(
                    frames, batch * n * expected_cameras, rel_tol=1e-8, abs_tol=1e-10
                )
            ):
                reasons.append("camera_frame_count_formula")
        row["matching_reasons"] = sorted(set(reasons))
    _attach_quality(root, rows, definition)
    for row in rows:
        for artifact in row["_quality"].get("artifacts", []):
            if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                try:
                    record(_inside(root, artifact["path"]))
                except ValueError:
                    pass  # The shared proof gate retains the outside-root reason.
    return rows, list(sources.values())


def _scope(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": row.get("backend"),
        "hardware_id": row.get("hardware_id"),
        "scene_sha256": row.get("scene_sha256"),
        "profile_sha256": row.get("profile_sha256"),
        "front_sha256": row.get("front_sha256"),
        **{name: row["config"].get(name) for name in _CONFIG_FIELDS},
        "other_config": {
            key: value
            for key, value in row["config"].items()
            if key not in _CHANGED_CONFIG_FIELDS and key not in _CONFIG_FIELDS
        },
        "boundary": row["metrics"].get("boundary"),
        "completion": row["metrics"].get("completion"),
        "measured_frames": _field(row, "counts.batch_steps"),
        "warmup_frames": _field(row, "warmup_counts.batch_steps"),
    }


def _trial(
    left: list[dict[str, Any]], right: list[dict[str, Any]], anchor: tuple[Any, ...]
) -> dict[str, Any]:
    reasons = []
    for role, rows in (("R-04", left), ("R-05", right)):
        if not rows:
            reasons.append(f"missing:{role}")
        elif len(rows) != 1:
            reasons.append(f"ambiguous_repeat:{role}")
        for row in rows:
            reasons.extend(f"{role}:{reason}" for reason in row["matching_reasons"])
            if row["status"] != "completed":
                reasons.append(
                    f"{role}:status:{row['status']}:{row.get('failure_reason_code') or 'unspecified'}"
                )
    a, b = (left[0] if len(left) == 1 else None), (
        right[0] if len(right) == 1 else None
    )
    if a is not None and b is not None:
        for path in _INVARIANTS:
            av, bv = _field(a, path), _field(b, path)
            if av in (None, "", []):
                reasons.append(f"missing:R-04:{path}")
            if bv in (None, "", []):
                reasons.append(f"missing:R-05:{path}")
            if av != bv:
                reasons.append(f"mismatch:{path}")
        common = [
            {
                key: value
                for key, value in row["config"].items()
                if key not in _CHANGED_CONFIG_FIELDS
            }
            for row in (a, b)
        ]
        if common[0] != common[1]:
            reasons.append("mismatch:other_frozen_config")
        if not a["front_sha256"] or a["front_sha256"] != b["front_sha256"]:
            reasons.append("mismatch:canonical_external_front")
        if (
            a["_observed_front"] is None
            or b["_observed_front"] is None
            or not _close(a["_observed_front"], b["_observed_front"])
        ):
            reasons.append("mismatch:observed_external_front")
    reasons = sorted(set(reasons))
    quality_reasons = []
    for role, row in (("R-04", a), ("R-05", b)):
        if row is None:
            quality_reasons.append(f"{role}:missing_unique_quality_trial")
            continue
        quality_reasons.extend(f"{role}:{reason}" for reason in row["quality_reasons"])
        coverage = row["_quality"].get("coverage", {})
        expected_names = (
            {"external_front"}
            if role == "R-04"
            else {"external_front", "external_side", "wrist"}
        )
        camera_names = (
            coverage.get("camera_names", []) if isinstance(coverage, dict) else []
        )
        if (
            not isinstance(camera_names, list)
            or len(camera_names) != len(expected_names)
            or any(not isinstance(name, str) for name in camera_names)
            or set(camera_names) != expected_names
        ):
            quality_reasons.append(f"{role}:quality_camera_coverage")
    if (
        a is not None
        and b is not None
        and _field(a, "_quality.protocol_sha256")
        != _field(b, "_quality.protocol_sha256")
    ):
        quality_reasons.append("quality_protocol_mismatch")
    metrics = {}
    for metric, (_, operations) in _METRICS.items():
        av = a["metrics"].get(metric) if a else None
        bv = b["metrics"].get(metric) if b else None
        invalid = list(reasons)
        for role, row, value in (("R-04", a, av), ("R-05", b, bv)):
            if (
                not _number(value)
                or value < 0
                or ("ratio" in operations and value == 0)
            ):
                reason = (
                    _field(row, f"metric_reasons.{metric}") if row else None
                ) or "missing_or_invalid_metric"
                invalid.append(f"{role}:{reason}")
        metrics[metric] = {
            "r04_value": av,
            "r05_value": bv,
            "ratio": bv / av if not invalid and "ratio" in operations else None,
            "delta": bv - av if not invalid else None,
            "reasons": sorted(set(invalid)),
        }
    members = left + right
    return {
        "trial_id": stable_hash([row["source_id"] for row in members])[:16],
        "backend": anchor[0],
        "num_envs": anchor[1],
        "repeat": anchor[2],
        "scope": _scope((left or right)[0]),
        "matched": not reasons,
        "reasons": reasons,
        "metrics": metrics,
        "r04_run_ids": [row["run_id"] for row in left],
        "r05_run_ids": [row["run_id"] for row in right],
        "source_ids": [row["source_id"] for row in members],
        "config_sha256": [row.get("config_sha256") for row in members],
        "status_counts": dict(Counter(row["status"] for row in members)),
        "both_quality_proofs_verified": not quality_reasons,
        "equivalence_qualified": not reasons and not quality_reasons,
        "quality_reasons": sorted(set(quality_reasons)),
        "claim": "descriptive_within_backend_camera_count_scaling",
    }


def _aggregate(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for trial in trials:
        groups[_json(trial["scope"])].append(trial)
    results = []
    for scope, members in sorted(groups.items()):
        measurements = {}
        for metric, (unit, operations) in _METRICS.items():
            for operation in operations:
                valid = [
                    trial
                    for trial in members
                    if _number(trial["metrics"][metric][operation])
                ]
                values = [trial["metrics"][metric][operation] for trial in valid]
                median = statistics.median(values) if values else None
                mad = (
                    statistics.median(abs(value - median) for value in values)
                    if len(values) >= 2
                    else None
                )
                measurements[f"{metric}:{operation}"] = {
                    "metric": metric,
                    "operation": operation,
                    "unit": "C3/C1" if operation == "ratio" else unit,
                    "median": median,
                    "valid_pairs": len(valid),
                    "planned_pair_slots": len(members),
                    "missing_pairs": len(members) - len(valid),
                    "uncertainty": {
                        "method": "MAD_of_independent_paired_process_repeats",
                        "value": mad,
                        "reason": "fewer_than_two_pairs" if mad is None else None,
                    },
                    "source_ids": [
                        source for trial in valid for source in trial["source_ids"]
                    ],
                    "config_sha256": sorted(
                        {
                            value
                            for trial in valid
                            for value in trial["config_sha256"]
                            if value
                        }
                    ),
                    "paired_repeats": [trial["repeat"] for trial in valid],
                    "missing_reasons": dict(
                        Counter(
                            reason
                            for trial in members
                            if trial not in valid
                            for reason in trial["metrics"][metric]["reasons"]
                        )
                    ),
                    "equivalence_qualified": bool(valid)
                    and len(valid) >= 2
                    and all(trial["equivalence_qualified"] for trial in valid),
                }
        results.append(
            {
                "scope": json.loads(scope),
                "planned_pair_slots": len(members),
                "source_trial_count": sum(
                    len(trial["source_ids"]) for trial in members
                ),
                "matched_pairs": sum(trial["matched"] for trial in members),
                "both_quality_proofs_pairs": sum(
                    trial["both_quality_proofs_verified"] for trial in members
                ),
                "source_ids": [
                    source for trial in members for source in trial["source_ids"]
                ],
                "config_sha256": sorted(
                    {
                        value
                        for trial in members
                        for value in trial["config_sha256"]
                        if value
                    }
                ),
                "metrics": measurements,
            }
        )
    return results


def _csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                key: _json(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def _figures(
    output: Path, groups: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scopes = defaultdict(list)
    for group in groups:
        scope = {
            key: value for key, value in group["scope"].items() if key != "num_envs"
        }
        if type(group["scope"]["num_envs"]) is int and group["scope"]["num_envs"] > 0:
            scopes[_json(scope)].append(group)
    figures, figure_data = [], []
    for encoded_scope, members in sorted(scopes.items()):
        if not any(
            value["median"] is not None
            for group in members
            for value in group["metrics"].values()
        ):
            continue
        scope = json.loads(encoded_scope)
        for metric, (_, operations) in _METRICS.items():
            for operation in operations:
                scale = 1 / 2**20 if metric.endswith("_bytes") else 1.0
                unit = (
                    "MiB (C3 - C1)"
                    if metric.endswith("_bytes")
                    else (
                        "C3 / C1"
                        if operation == "ratio"
                        else "C3 - C1; " + _METRICS[metric][0]
                    )
                )
                name = f"{metric}-{operation}-{stable_hash(scope)[:12]}"
                series, segment = [], []
                for group in sorted(
                    members, key=lambda item: item["scope"]["num_envs"]
                ):
                    value = group["metrics"][f"{metric}:{operation}"]
                    if value["median"] is None:
                        if segment:
                            series.append(
                                {"label": "descriptive scaling", "points": segment}
                            )
                            segment = []
                        continue
                    point = {
                        "x": group["scope"]["num_envs"],
                        "y": value["median"] * scale,
                        "source_ids": value["source_ids"],
                        "run_ids": value["source_ids"],
                        "config_sha256": value["config_sha256"],
                        "valid_pairs": value["valid_pairs"],
                        "paired_repeats": value["paired_repeats"],
                        "equivalence_qualified": value["equivalence_qualified"],
                    }
                    mad = value["uncertainty"]["value"]
                    if mad is not None:
                        point.update(
                            low=(value["median"] - mad) * scale,
                            high=(value["median"] + mad) * scale,
                        )
                    segment.append(point)
                    figure_data.append(
                        {
                            "figure": name,
                            "metric": metric,
                            "operation": operation,
                            "scope": scope,
                            **point,
                        }
                    )
                if segment:
                    series.append({"label": "descriptive scaling", "points": segment})
                paths = write_chart(
                    output / "figures" / name,
                    title=f"{scope['backend']} {str(scope['hardware_id'])[:20]}: {metric} C3-vs-C1 ({operation})",
                    x_label="Environments N; fixed output size and delivery",
                    y_label=unit,
                    series=series,
                    provenance={
                        "scope": scope,
                        "sources": sources,
                        "series": series,
                        "source_trials": [
                            {
                                "num_envs": group["scope"]["num_envs"],
                                "source_ids": group["source_ids"],
                                "config_sha256": group["config_sha256"],
                                "planned_pair_slots": group["planned_pair_slots"],
                            }
                            for group in members
                        ],
                        "claim": "descriptive_within_backend_scaling_not_equal_quality_speedup",
                    },
                    log_x=True,
                )
                figures.append(
                    {
                        "metric": metric,
                        "operation": operation,
                        "scope": scope,
                        "svg": paths[0].relative_to(output).as_posix(),
                        "pdf": paths[1].relative_to(output).as_posix(),
                    }
                )
    if not figures:
        paths = write_chart(
            output / "figures/no-matched-pairs",
            title="No compatible completed R-04/R-05 repeat pairs",
            x_label="Environments N",
            y_label="C3 / C1",
            series=[],
            provenance={
                "sources": sources,
                "source_trials": [
                    {
                        key: group[key]
                        for key in (
                            "scope",
                            "source_ids",
                            "config_sha256",
                            "planned_pair_slots",
                        )
                    }
                    for group in groups
                ],
                "claim": "no_scaling_values_available",
            },
        )
        figures.append(
            {
                "metric": "unavailable",
                "operation": "none",
                "scope": {"backend": "all retained trials"},
                "svg": paths[0].relative_to(output).as_posix(),
                "pdf": paths[1].relative_to(output).as_posix(),
            }
        )
    return figures, figure_data


def rebuild_camera_scaling_report(r04_root: Path, r05_root: Path, output: Path) -> Path:
    """Compare frozen one-camera and three-camera experiments without runtimes.

    Args:
        r04_root: Formal v2 R-04 experiment directory.
        r05_root: Formal v2 R-05 experiment directory.
        output: Separate destination for Markdown, CSV, SVG, PDF and JSON.

    Returns:
        Report Markdown path. Failed, absent, duplicate and mismatched trials
        remain in the JSON/CSV denominators, with no ratio substituted for them.
    """
    roots = {"R-04": Path(r04_root).resolve(), "R-05": Path(r05_root).resolve()}
    output = Path(output).resolve()
    if output in roots.values() or roots["R-04"] == roots["R-05"]:
        raise ValueError(
            "Use two distinct experiment roots and a separate report output"
        )
    rows, sources = [], []
    for role, root in roots.items():
        loaded, evidence = _load_input(root, role)
        rows.extend(loaded)
        sources.extend(evidence)
    slots = defaultdict(lambda: {"R-04": [], "R-05": []})
    for row in rows:
        anchor = (row["backend"], row["config"].get("num_envs"), row["repeat"])
        slots[anchor][row["source_experiment"]].append(row)
    trials = [
        _trial(slot["R-04"], slot["R-05"], anchor)
        for anchor, slot in sorted(slots.items(), key=lambda item: _json(item[0]))
    ]
    groups = _aggregate(trials)
    output.mkdir(parents=True, exist_ok=True)
    figures, figure_data = _figures(output, groups, sources)
    source_rows = [
        {
            key: row.get(key)
            for key in (
                "source_id",
                "source_experiment",
                "source_root",
                "run_id",
                "case_id",
                "backend",
                "repeat",
                "status",
                "failure_reason_code",
                "hardware_id",
                "config_sha256",
                "scene_sha256",
                "view_sha256",
                "profile_sha256",
                "front_sha256",
                "config",
                "metrics",
                "metric_reasons",
                "matching_reasons",
                "quality_reasons",
            )
        }
        for row in rows
    ]
    summary = {
        "schema_version": 1,
        "report_version": "1.0",
        "comparison": "R-05 C3 relative to R-04 C1",
        "claim": "descriptive_within_backend_camera_count_scaling",
        "equal_quality_speedup_claim": False,
        "ratio_definition": "R05/C3 value divided by R04/C1 value within the same independent repeat",
        "delta_definition": "R05/C3 value minus R04/C1 value within the same independent repeat",
        "source_trial_count": len(rows),
        "planned_pair_slots": len(trials),
        "matched_pairs": sum(trial["matched"] for trial in trials),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "sources": sources,
        "source_rows": source_rows,
        "trials": trials,
        "groups": groups,
        "figures": figures,
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "inputs.json",
        {
            "roots": {key: str(value) for key, value in roots.items()},
            "sources": sources,
        },
    )
    _csv(
        output / "source_trials.csv",
        source_rows,
        tuple(source_rows[0]) if source_rows else ("source_id", "status"),
    )
    _csv(
        output / "paired_trials.csv",
        trials,
        (
            "trial_id",
            "backend",
            "num_envs",
            "repeat",
            "matched",
            "reasons",
            "r04_run_ids",
            "r05_run_ids",
            "config_sha256",
            "status_counts",
            "both_quality_proofs_verified",
            "equivalence_qualified",
            "quality_reasons",
        ),
    )
    paired_metrics = [
        {
            "trial_id": trial["trial_id"],
            "backend": trial["backend"],
            "num_envs": trial["num_envs"],
            "repeat": trial["repeat"],
            "metric": metric,
            "source_ids": trial["source_ids"],
            "config_sha256": trial["config_sha256"],
            **value,
        }
        for trial in trials
        for metric, value in trial["metrics"].items()
    ]
    _csv(
        output / "paired_metrics.csv",
        paired_metrics,
        (
            "trial_id",
            "backend",
            "num_envs",
            "repeat",
            "metric",
            "r04_value",
            "r05_value",
            "ratio",
            "delta",
            "reasons",
            "source_ids",
            "config_sha256",
        ),
    )
    plot_data = [
        {"scope": group["scope"], **value}
        for group in groups
        for value in group["metrics"].values()
    ]
    _csv(
        output / "plot_data.csv",
        plot_data,
        (
            "scope",
            "metric",
            "operation",
            "unit",
            "median",
            "valid_pairs",
            "planned_pair_slots",
            "missing_pairs",
            "uncertainty",
            "source_ids",
            "config_sha256",
            "paired_repeats",
            "missing_reasons",
            "equivalence_qualified",
        ),
    )
    _csv(
        output / "figure_data.csv",
        figure_data,
        (
            "figure",
            "metric",
            "operation",
            "scope",
            "x",
            "y",
            "low",
            "high",
            "source_ids",
            "config_sha256",
            "valid_pairs",
            "paired_repeats",
            "equivalence_qualified",
        ),
    )
    lines = [
        "# R-05 relative to R-04: camera-count scaling",
        "",
        "C3/C1 ratios and C3−C1 deltas describe adding external-side and wrist cameras to the common external-front workload within one backend. These are descriptive, unqualified scaling measurements unless both workloads have verified quality evidence; they are never labelled equal-quality speedups.",
        "",
        f"Source trials: {len(rows)}; planned repeat slots: {len(trials)}; matched completed pairs: {summary['matched_pairs']}. Failed, missing, excluded and nonmatching trials remain in the tables and CSVs.",
        "",
        "Pairing requires matching backend, GPU UUID, scene/profile, N, resolution, modalities, delivery/completion boundary and warmup/measured frame counts. Frozen scene/view/profile hashes and observed external-front poses/intrinsics are verified. Duplicate repeat slots are rejected rather than arbitrarily paired.",
        "",
        "Each point is the median of independently paired process repeats. Whiskers are median ± MAD, not confidence intervals; a single pair has no uncertainty estimate. Missing points break lines. VRAM is sampled process steady/peak memory, separate from device-wide and Torch allocator counters.",
        "",
        "## Repeat pairing",
        "",
        render_markdown_table(
            [
                {
                    key: trial[key]
                    for key in (
                        "backend",
                        "num_envs",
                        "repeat",
                        "matched",
                        "reasons",
                        "both_quality_proofs_verified",
                    )
                }
                for trial in trials
            ],
            columns=(
                "backend",
                "num_envs",
                "repeat",
                "matched",
                "reasons",
                "both_quality_proofs_verified",
            ),
        ),
        "",
        "## Curves",
        "",
    ]
    for figure in figures:
        lines.extend(
            [
                f"### {figure['scope']['backend']} — {figure['metric']} {figure['operation']}",
                "",
                f"![Descriptive camera-count scaling]({figure['svg']})",
                "",
                f"[PDF]({figure['pdf']})",
                "",
            ]
        )
    lines.extend(
        [
            "Source run IDs/config hashes are embedded in every SVG/PDF and in `paired_metrics.csv`, `plot_data.csv`, and `figure_data.csv`. `source_trials.csv` preserves each original trial; `inputs.json` binds frozen input files. Different profile versions, missing GPU identities, or incompatible front-camera evidence block ratios rather than becoming a leaderboard.",
            "",
        ]
    )
    path = output / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    """Rebuild the report from two explicit frozen experiment roots.

    Args:
        argv: Optional command arguments, without the module name.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r04", type=Path, required=True)
    parser.add_argument("--r05", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(rebuild_camera_scaling_report(args.r04, args.r05, args.output))


if __name__ == "__main__":
    main()
