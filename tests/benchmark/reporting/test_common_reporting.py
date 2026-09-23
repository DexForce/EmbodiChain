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

"""Report aggregation keeps workloads, failures, units and quality distinct."""

from __future__ import annotations

from copy import deepcopy

import pytest


def test_aggregation_never_mixes_configs_or_failed_measurements() -> None:
    from scripts.benchmark.reporting.aggregation import MetricDefinition, aggregate_runs

    rows = [
        {
            "backend": "a",
            "status": "completed",
            "config_sha256": "x",
            "metrics": {"seconds": 2.0},
        },
        {
            "backend": "a",
            "status": "completed",
            "config_sha256": "x",
            "metrics": {"seconds": 4.0},
        },
        {
            "backend": "a",
            "status": "failed",
            "config_sha256": "x",
            "metrics": {"seconds": 100.0},
        },
        {
            "backend": "a",
            "status": "completed",
            "config_sha256": "y",
            "metrics": {"seconds": 20.0},
        },
        {"backend": "b", "status": "not_run", "config_sha256": "x"},
    ]
    result = aggregate_runs(rows, (MetricDefinition("seconds", "s", "operation"),))
    assert (result["completed_runs"], result["failed_runs"], result["not_run"]) == (
        3,
        1,
        1,
    )
    a_x = next(
        g for g in result["groups"] if g["backend"] == "a" and g["config_sha256"] == "x"
    )
    assert a_x["metrics"]["seconds"] == {
        "median": 3.0,
        "valid_runs": 2,
        "missing_runs": 0,
        "unit": "s",
        "population": "operation",
        "aggregation": "median_of_runs",
    }
    a_y = next(g for g in result["groups"] if g["config_sha256"] == "y")
    assert a_y["metrics"]["seconds"]["median"] == 20.0


def test_missing_metric_is_not_zero() -> None:
    from scripts.benchmark.reporting.aggregation import MetricDefinition, aggregate_runs

    result = aggregate_runs(
        [{"backend": "a", "status": "completed", "metrics": {}}],
        (MetricDefinition("bytes", "byte", "process"),),
    )
    metric = result["groups"][0]["metrics"]["bytes"]
    assert metric["median"] is None
    assert (metric["valid_runs"], metric["missing_runs"]) == (0, 1)


@pytest.mark.parametrize("case_id", [None, "case1"])
def test_early_failures_keep_the_known_case_denominator(case_id: str | None) -> None:
    from scripts.benchmark.reporting.aggregation import MetricDefinition, aggregate_runs

    rows = [
        {
            "backend": "a",
            "status": "completed",
            "config_sha256": "x",
            "metrics": {"seconds": 2.0},
        },
        {"backend": "a", "status": "failed", "error": "early crash"},
    ]
    if case_id is not None:
        for row in rows:
            row["case_id"] = case_id
    groups = aggregate_runs(rows, (MetricDefinition("seconds", "s", "operation"),))[
        "groups"
    ]
    assert len(groups) == 1
    assert (groups[0]["completed"], groups[0]["scheduled"]) == (1, 2)
    assert groups[0]["metrics"]["seconds"]["median"] == 2.0


def test_unknown_failure_is_not_assigned_to_an_ambiguous_workload() -> None:
    from scripts.benchmark.reporting.aggregation import MetricDefinition, aggregate_runs

    rows = [
        {
            "backend": "a",
            "status": "completed",
            "config_sha256": "x",
            "metrics": {"seconds": 2.0},
        },
        {
            "backend": "a",
            "status": "completed",
            "config_sha256": "y",
            "metrics": {"seconds": 20.0},
        },
        {"backend": "a", "status": "failed"},
    ]
    groups = aggregate_runs(rows, (MetricDefinition("seconds", "s", "operation"),))[
        "groups"
    ]
    assert len(groups) == 3
    assert all(g["scheduled"] == 1 for g in groups)
    assert next(g for g in groups if g["config_sha256"] is None)["completed"] == 0


@pytest.mark.parametrize(
    "path", ["config_sha256", "hardware_id", "metrics.boundary", "metrics.completion"]
)
def test_comparison_rejects_mismatched_invariants(path: str) -> None:
    from scripts.benchmark.reporting.comparison import comparison_reasons

    left = {
        "backend": "a",
        "status": "completed",
        "quality_status": "qualified",
        "config_sha256": "cfg",
        "hardware_id": "gpu",
        "metrics": {"boundary": "operation", "completion": "host_return"},
    }
    right = deepcopy(left)
    right["backend"] = "b"
    target = right
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = "different"
    assert path in comparison_reasons([left, right], invariant_paths=(path,))


def test_quality_is_independent_of_execution_completion() -> None:
    from scripts.benchmark.reporting.comparison import comparison_reasons

    rows = [
        {"backend": b, "status": "completed", "quality_status": "not_qualified"}
        for b in ("a", "b")
    ]
    assert "quality_status" in comparison_reasons(rows, invariant_paths=())
