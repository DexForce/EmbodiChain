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
from dataclasses import dataclass
import math
import statistics
from typing import Any

__all__ = ["MetricDefinition", "aggregate_runs"]


@dataclass(frozen=True)
class MetricDefinition:
    """An immutable metric definition: raw key, SI unit and counting population."""

    key: str
    unit: str
    population: str


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
            for row in completed:
                value = row.get("metrics", {}).get(definition.key)
                if type(value) in (float, int) and math.isfinite(value):
                    valid.append(value)
            values[definition.key] = {
                "median": statistics.median(valid) if valid else None,
                "valid_runs": len(valid),
                "missing_runs": len(completed) - len(valid),
                "unit": definition.unit,
                "population": definition.population,
                "aggregation": "median_of_runs",
            }
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
