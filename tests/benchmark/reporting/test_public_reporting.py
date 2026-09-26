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

"""Public reporting helpers rebuild metrics and figures offline."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_standard_artifact_store_is_idempotent_and_complete(tmp_path: Path) -> None:
    from scripts.benchmark.core.artifacts import (
        ArtifactStore,
        create_experiment_directory,
        read_json,
    )

    root = create_experiment_directory(
        tmp_path,
        experiment_id="fixture",
        config={"resolution": [64, 64]},
        assets_manifest={"assets": [{"id": "table", "sha256": "abc"}]},
    )
    expected = {
        "effective_config.yaml",
        "assets_manifest.json",
        "raw.jsonl",
        "metrics.json",
        "quality.json",
        "artifact_index.json",
    }
    assert expected.issubset({path.name for path in root.iterdir()})

    store = ArtifactStore(root)
    record = {"record_id": "attempt-0", "status": "failed", "reason": "oom"}
    assert store.append_raw(record) is True
    assert store.append_raw(record) is False
    with pytest.raises(ValueError, match="different payload"):
        store.append_raw({"record_id": "attempt-0", "status": "completed"})
    artifact = store.register_artifact(root / "raw.jsonl", kind="raw_observation")
    assert artifact["kind"] == "raw_observation"
    assert read_json(root / "artifact_index.json")["artifacts"]


def test_aggregation_reports_uncertainty_denominators_and_missing_reasons() -> None:
    from scripts.benchmark.reporting.aggregation import MetricDefinition, aggregate_runs

    rows = [
        {
            "run_id": "r0",
            "backend": "a",
            "case_id": "case",
            "status": "completed",
            "metrics": {"value": 2.0},
        },
        {
            "run_id": "r1",
            "backend": "a",
            "case_id": "case",
            "status": "completed",
            "metrics": {},
            "metric_reasons": {"value": "unsupported"},
        },
        {
            "run_id": "r2",
            "backend": "a",
            "case_id": "case",
            "status": "failed",
            "error": "timeout",
        },
    ]
    result = aggregate_runs(
        rows,
        (MetricDefinition("value", "s", "operation", definition_version="2"),),
    )
    metric = result["groups"][0]["metrics"]["value"]
    assert metric["median"] == 2.0
    assert metric["denominator"] == 2
    assert metric["source_runs"] == ["r0"]
    assert metric["missing_reasons"] == {"unsupported": 1}
    assert metric["uncertainty"] == {
        "method": "median_absolute_deviation",
        "value": 0.0,
    }


def test_zero_denominator_is_explicit_and_comparison_can_compute_speedup() -> None:
    from scripts.benchmark.reporting.aggregation import MetricDefinition, aggregate_runs
    from scripts.benchmark.reporting.comparison import compare_metric

    empty = aggregate_runs(
        [{"backend": "a", "status": "failed", "case_id": "case"}],
        (MetricDefinition("value", "s", "operation"),),
    )
    metric = empty["groups"][0]["metrics"]["value"]
    assert metric["median"] is None
    assert metric["denominator"] == 0
    assert metric["missing_reasons"] == {"missing": 0}

    rows = [
        {
            "run_id": "a0",
            "backend": "a",
            "case_id": "case",
            "status": "completed",
            "quality_status": "qualified",
            "config_sha256": "cfg",
            "hardware_id": "gpu",
            "metrics": {"value": 2.0},
        },
        {
            "run_id": "b0",
            "backend": "b",
            "case_id": "case",
            "status": "completed",
            "quality_status": "qualified",
            "config_sha256": "cfg",
            "hardware_id": "gpu",
            "metrics": {"value": 1.0},
        },
    ]
    comparison = compare_metric(
        rows,
        metric="value",
        baseline="a",
        candidate="b",
        invariant_paths=("config_sha256", "hardware_id"),
    )
    assert comparison.eligible is True
    assert comparison.ratio == 2.0
    assert comparison.pairs == 1


def test_report_table_and_svg_figure_are_dependency_free(tmp_path: Path) -> None:
    from scripts.benchmark.reporting.figures import write_svg_bar_chart
    from scripts.benchmark.reporting.tables import render_markdown_table

    table = render_markdown_table(
        [{"backend": "a", "rate": 1.5, "note": "a|b"}],
        columns=("backend", "rate", "note"),
    )
    assert "a\\|b" in table
    figure = write_svg_bar_chart(
        tmp_path / "rate.svg",
        title="Rate",
        values={"a": 1.5, "b": 1.0},
    )
    assert figure.read_text(encoding="utf-8").startswith("<svg ")


def test_legacy_conversion_preserves_rows_and_adds_common_identity() -> None:
    from scripts.benchmark.reporting.compat import convert_legacy_rows

    rows = convert_legacy_rows(
        [
            {"algorithm": "planner", "task": "pick", "cost_time_ms": 2.0},
            {"algorithm": "planner", "task": "pick", "status": "failed"},
        ],
        experiment_id="legacy-suite",
    )
    assert [row["status"] for row in rows] == ["completed", "failed"]
    assert rows[0]["metrics"]["cost_time_ms"] == 2.0
    assert rows[0]["legacy_source"]["algorithm"] == "planner"
