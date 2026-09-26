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

"""Aggregate independent runs without mixing workloads or failure populations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any

from scripts.benchmark.core.contracts import MetricDefinition

__all__ = ["MetricDefinition", "aggregate_attempts", "aggregate_runs"]


def _uncertainty(values: Sequence[float]) -> dict[str, object] | None:
    """Return a robust median absolute deviation summary."""
    if not values:
        return None
    median = statistics.median(values)
    deviation = statistics.median(abs(value - median) for value in values)
    return {
        "method": "median_absolute_deviation",
        "value": deviation,
    }


def _missing_reasons(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    """Count explicit domain reasons for missing completed metrics."""
    reasons: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        reasons_map = row.get("metric_reasons") or row.get("missing_metrics")
        reason = reasons_map.get(key) if isinstance(reasons_map, Mapping) else None
        reasons[str(reason or "missing")] += 1
    return dict(sorted(reasons.items()))


def aggregate_runs(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[MetricDefinition]
) -> dict[str, Any]:
    """Group by backend/workload and take medians only over valid completed runs.

    Failures remain in scheduled counts; missing and non-finite metrics are
    represented by a null median and an explicit missing-run count.
    """
    known_hashes = defaultdict(set)
    for row in rows:
        if row.get("config_sha256") is not None:
            known_hashes[row.get("case_id")].add(row["config_sha256"])
    groups = defaultdict(list)
    for row in rows:
        case_id = row.get("case_id")
        fingerprint = row.get("config_sha256")
        candidates = known_hashes[case_id]
        # Associate early failures with a case only when its recorded
        # workload is unambiguous. Completed runs never inherit evidence.
        if (
            fingerprint is None
            and row["status"] != "completed"
            and len(candidates) == 1
        ):
            fingerprint = next(iter(candidates))
        groups[(row["backend"], case_id, fingerprint)].append(row)
    result = {
        "schema_version": 1,
        "completed_runs": sum(r["status"] == "completed" for r in rows),
        "failed_runs": sum(r["status"] not in ("completed", "not_run") for r in rows),
        "not_run": sum(r["status"] == "not_run" for r in rows),
        "groups": [],
    }
    for (backend, case_id, fingerprint), members in groups.items():
        completed = [r for r in members if r["status"] == "completed"]
        values = {}
        for definition in metrics:
            valid = []
            source_runs = []
            for row in completed:
                value = row.get("metrics", {}).get(definition.key)
                if type(value) in (float, int) and math.isfinite(value):
                    valid.append(value)
                    source_runs.append(str(row.get("run_id", "")))
            metric_value = {
                "median": statistics.median(valid) if valid else None,
                "valid_runs": len(valid),
                "missing_runs": len(completed) - len(valid),
                "unit": definition.unit,
                "population": definition.population,
                "aggregation": "median_of_runs",
            }
            # Preserve the v1 camera/report shape for legacy rows that lack
            # run identities. Standard protocol rows always carry run_id and
            # receive denominator, uncertainty and source-run accounting.
            detailed = (
                definition.definition_version != "1"
                or not completed
                or any("run_id" in row for row in completed)
            )
            if detailed:
                metric_value.update(
                    denominator=len(completed),
                    denominator_definition=definition.denominator,
                    definition_version=definition.definition_version,
                    uncertainty=_uncertainty(valid),
                    source_runs=source_runs,
                    missing_reasons=(
                        _missing_reasons(
                            [
                                row
                                for row in completed
                                if not (
                                    type(row.get("metrics", {}).get(definition.key))
                                    in (float, int)
                                    and math.isfinite(
                                        row.get("metrics", {}).get(definition.key)
                                    )
                                )
                            ],
                            definition.key,
                        )
                        if completed
                        else {"missing": 0}
                    ),
                )
            values[definition.key] = metric_value
        result["groups"].append(
            {
                "backend": backend,
                "case_id": case_id,
                "config_sha256": fingerprint,
                "scheduled": len(members),
                "completed": len(completed),
                "metrics": values,
            }
        )
    return result


def aggregate_attempts(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[MetricDefinition]
) -> dict[str, Any]:
    """Aggregate bounded attempts using the same failure-preserving contract.

    Attempt rows use ``attempt_id`` when available; the resulting groups retain
    the original attempt and run denominators instead of converting failures to
    zeros. The shared grouping semantics are identical to run aggregation.
    """
    normalized = []
    for row in rows:
        value = dict(row)
        if "run_id" not in value and "attempt_id" in value:
            value["run_id"] = value["attempt_id"]
        normalized.append(value)
    return aggregate_runs(normalized, metrics)
