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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

pytest.importorskip("newton")

from scripts.benchmark.scene.feasibility_collision import (
    _agreement,
    _markdown,
    _qualification_summary,
)

pytestmark = pytest.mark.no_sim


def test_incomplete_graph_qualification_is_unranked_and_errors_visible(
    tmp_path: Path,
) -> None:
    def complete(fixture: str, algorithm: str) -> dict:
        return {
            "fixture": fixture,
            "algorithm": algorithm,
            "scope": "full_corpus_qualification",
            "success_rate": 1.0,
            "matched": 2,
            "total": 2,
            "false_negative": 0,
            "false_positive": 0,
            "ambiguous": 0,
            "outside_detection_band": 0,
        }

    qualifications = [
        complete("primitive", "newton_runtime_cpu"),
        complete("robot", "newton_runtime_cpu"),
        complete("primitive", "newton_cuda_graph"),
        {
            "fixture": "robot",
            "scope": "cuda_graph_qualification_error",
            "reason": "contact buffer saturated",
        },
    ]
    summary = {r["algorithm"]: r for r in _qualification_summary([], qualifications)}
    assert summary["newton_runtime_cpu"]["success_rate"] == 1.0
    graph = summary["newton_cuda_graph"]
    assert graph["success_rate"] is None
    assert graph["coverage"] == "1/2" and graph["status"] == "incomplete"
    output = tmp_path / "report.md"
    _markdown(output, [], {}, qualifications)
    text = output.read_text()
    assert "contact buffer saturated" in text and "| incomplete |" in text


def test_nonfinite_or_boundary_gaps_cannot_count_as_agreement() -> None:
    metrics = _agreement(np.array([False, True, True]), np.array([np.nan, np.inf, 0.0]))
    assert metrics["matched"] == 1 and metrics["total"] == 3
    assert metrics["invalid_outputs"] == 1 and metrics["ambiguous"] == 1


def test_duplicate_qualification_cannot_silently_replace_a_failed_capture() -> None:
    failed = {
        "fixture": "capture",
        "algorithm": "cpu",
        "scope": "full_corpus_qualification",
        "matched": 0,
        "total": 1,
    }
    with pytest.raises(ValueError, match="Duplicate fixture qualification"):
        _qualification_summary([], [failed, {**failed, "matched": 1}])


@pytest.mark.parametrize("analytic_failure", [False, True])
def test_benchmark_reports_empty_captures_and_historical_disagreements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, analytic_failure: bool
) -> None:
    from scripts.benchmark.scene import feasibility_collision as benchmark
    from embodichain.lab.task_program.integrations import feasibility as workflow

    monkeypatch.setattr(benchmark, "_fixtures", lambda: [])
    monkeypatch.setattr(benchmark.wp, "is_cuda_available", lambda: False)
    # A valid workflow may reject its scene before constructing a query model.
    monkeypatch.setattr(
        workflow, "run_feasibility", lambda *args: SimpleNamespace(metrics={})
    )
    source = tmp_path / "search.empty.json"
    source.write_text(
        json.dumps({"task_program": {"deployment": "task.yaml"}, "search": {}})
    )
    a, b = np.eye(4), np.eye(4)
    b[0, 3] = 0.2 + 1e-7
    data = {
        "bodies": [
            {
                "collision_id": "a",
                "frame_uid": "a",
                "kind": "sphere",
                "dimensions": [0.1],
                "robot": True,
            },
            {
                "collision_id": "b",
                "frame_uid": "b",
                "kind": "sphere",
                "dimensions": [0.1],
            },
        ],
        "ignored_pairs": [],
        "samples": [{"frames": {"a": a.tolist(), "b": b.tolist()}, "status": "passed"}],
    }
    if analytic_failure:
        data["samples"][0]["expected"] = True
    first = tmp_path / "run-a/fixture-state.json"
    first.parent.mkdir()
    first.write_text(json.dumps(data))
    second = tmp_path / "run-b/fixture-state.json"
    second.parent.mkdir()
    data["samples"][0]["status"] = "failed"
    second.write_text(json.dumps(data))
    empty = tmp_path / "fixture-empty.json"
    empty.write_text(json.dumps({"bodies": None, "samples": []}))
    output = tmp_path / "report"
    args = SimpleNamespace(
        threads=1,
        fixture=[first, second, empty],
        collect_config=[source],
        output=output,
        cuda_device="cuda:0",
        warmup=1,
        repeats=1,
        batches=[1],
        detection_gap=0.03,
    )
    if analytic_failure:
        with pytest.raises(AssertionError, match="Independent analytic oracle failed"):
            benchmark.run_all_benchmarks(args)
        return
    payload = benchmark.run_all_benchmarks(args)
    comparisons = [
        row
        for row in payload["qualification"]
        if row["scope"] == "captured_outcome_comparison"
    ]
    assert len(comparisons) == 2
    assert len({row["fixture"] for row in comparisons}) == 2
    assert sorted(row["matched"] for row in comparisons) == [0, 1]
    assert all(row["total"] == 1 for row in comparisons)
    assert any(row["changes"] == {"passed->failed": 1} for row in comparisons)
    errors = {row["scope"] for row in payload["qualification"]}
    assert {"workload_capture_error", "fixture_load_error"} <= errors
    summary = _qualification_summary(payload["results"], payload["qualification"])
    assert summary and all(row["status"] == "incomplete" for row in summary)
    profile = json.loads((output / "native-input-0/profile.json").read_text())
    assert profile["capture_status"] == "unavailable"
    assert not list((output / "native-input-0").glob("fixture-*.json"))
    assert (output / "results.json").is_file()
    report = (output / "report.md").read_text()
    assert "Captured outcomes versus current checker" in report
    assert "passed->failed" in report and "workload_capture_error" in report
