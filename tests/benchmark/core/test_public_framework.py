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

"""Public framework contracts cover definitions, budgets and raw evidence."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


def test_experiment_definition_and_records_are_json_ready() -> None:
    from scripts.benchmark.core.contracts import (
        AggregateMetric,
        Budget,
        ExperimentDefinition,
        RawObservation,
        RunRecord,
    )

    definition = ExperimentDefinition(
        experiment_id="r04-camera",
        definition_version="1.0",
        parameter_matrix={"resolution": (64, 128), "num_envs": (1,)},
        budget=Budget(max_runs=4, max_attempts=8, wall_time_s=20.0),
        quality_protocol={"image": "nonempty", "status": "not_qualified"},
        comparison_invariants=("config_sha256", "hardware_id"),
    )
    payload = definition.to_dict()
    assert payload["budget"] == {
        "max_runs": 4,
        "max_attempts": 8,
        "wall_time_s": 20.0,
    }
    assert payload["parameter_matrix"]["resolution"] == [64, 128]

    record = RunRecord(
        experiment_id="r04-camera",
        run_id="run-0",
        case_id="case-0",
        backend="embodichain",
        repeat=0,
        attempt=0,
        status="completed",
        quality_status="qualified",
        task_status="not_evaluated",
        data_status="not_evaluated",
        hardware_id="gpu-0",
        config_sha256="config",
        metrics={"frames_per_s": 10.0},
        evidence=("run-0/result.json",),
    )
    observation = RawObservation(
        record_id="obs-0",
        run_id="run-0",
        case_id="case-0",
        stage="capture",
        index=0,
        elapsed_s=0.1,
        counters={"frames": 1},
        evidence=("run-0/frame.png",),
    )
    aggregate = AggregateMetric(
        key="frames_per_s",
        definition_version="1.0",
        unit="1/s",
        population="camera_exposure",
        aggregation="median_of_runs",
        denominator=1,
        valid_count=1,
        missing_count=0,
        value=10.0,
        uncertainty={"method": "mad", "value": 0.0},
        source_runs=("run-0",),
    )
    assert record.to_dict()["status"] == "completed"
    assert observation.to_dict()["counters"] == {"frames": 1}
    assert aggregate.to_dict()["source_runs"] == ["run-0"]


def test_parameter_matrix_and_plan_are_deterministic_and_budgeted(
    tmp_path: Path,
) -> None:
    from scripts.benchmark.core.contracts import Budget, ExperimentDefinition
    from scripts.benchmark.core.planning import (
        build_run_plan,
        expand_parameter_matrix,
    )

    cases = expand_parameter_matrix({"resolution": (64, 128), "num_envs": (1, 2)})
    assert [case.case_id for case in cases] == [
        "case-0000",
        "case-0001",
        "case-0002",
        "case-0003",
    ]
    assert cases[0].parameters == {"num_envs": 1, "resolution": 64}

    definition = ExperimentDefinition(
        experiment_id="matrix",
        parameter_matrix={"resolution": (64, 128)},
        budget=Budget(max_runs=4),
    )
    plan = build_run_plan(
        definition,
        backends=("a", "b"),
        repeats=1,
        output_root=tmp_path,
        command_factory=lambda backend, case, repeat, run_dir: (
            sys.executable,
            "-S",
            "-c",
            "pass",
        ),
        timeout_s=12.0,
    )
    assert len(plan.runs) == 4
    assert plan.runs[0].case_id == "case-0000"
    assert plan.runs[-1].output.parent == tmp_path
    assert plan.runs[0].timeout_s == 12.0

    with pytest.raises(ValueError, match="max_runs"):
        build_run_plan(
            ExperimentDefinition(
                experiment_id="too-small",
                parameter_matrix={"x": (1, 2)},
                budget=Budget(max_runs=1),
            ),
            backends=("a", "b"),
            repeats=1,
            output_root=tmp_path,
            command_factory=lambda *_: (sys.executable, "-S", "-c", "pass"),
        )


def test_budget_ledger_preserves_attempt_and_run_limits() -> None:
    from scripts.benchmark.core.contracts import Budget
    from scripts.benchmark.core.planning import BudgetExceeded, BudgetLedger

    ledger = BudgetLedger(Budget(max_runs=1, max_attempts=2))
    ledger.start_run()
    ledger.start_attempt()
    ledger.finish_attempt("failed")
    ledger.start_attempt()
    ledger.finish_attempt("completed")
    with pytest.raises(BudgetExceeded, match="attempt"):
        ledger.start_attempt()
    with pytest.raises(BudgetExceeded, match="run"):
        ledger.start_run()
    assert ledger.to_dict()["attempts_started"] == 2


def test_run_experiment_marks_budget_exhaustion_as_not_run(tmp_path: Path) -> None:
    from scripts.benchmark.core.contracts import Budget
    from scripts.benchmark.core.execution import run_experiment
    from scripts.benchmark.core.records import RunSpec

    specs = tuple(
        RunSpec(
            "fixture",
            index,
            (
                sys.executable,
                "-S",
                "-c",
                "from pathlib import Path; Path(__import__('sys').argv[1]).write_text('{\"status\":\"completed\"}')",
                str(tmp_path / f"r{index}" / "result.json"),
            ),
            tmp_path / f"r{index}",
        )
        for index in range(2)
    )
    rows = run_experiment(
        tmp_path,
        specs,
        experiment_id="budgeted",
        budget=Budget(max_runs=1, max_attempts=1),
    )
    assert [row["status"] for row in rows] == ["completed", "not_run"]
    assert "budget exhausted" in rows[1]["error"].lower()
    assert len((tmp_path / "raw.jsonl").read_text().splitlines()) == 2


def test_stage_attempt_and_resource_measurements_keep_failure_reasons() -> None:
    from scripts.benchmark.core.measurement import (
        measure_attempts,
        measure_stages,
        sample_resource,
    )

    def unsupported() -> None:
        raise RuntimeError("no backend")

    stages = measure_stages((("load", lambda: 1), ("unsupported", unsupported)))
    assert [event.status for event in stages.events] == ["completed", "failed"]
    assert stages.events[1].failure_reason == "RuntimeError: no backend"

    attempts = measure_attempts(
        lambda index: index,
        attempts=3,
        validate=lambda value: value != 1,
    )
    assert [outcome.status for outcome in attempts.outcomes] == [
        "completed",
        "failed",
        "completed",
    ]
    assert attempts.completed == 2
    assert attempts.failed == 1
    assert sample_resource("gpu_memory", lambda: 128, unit="byte").to_dict() == {
        "kind": "gpu_memory",
        "value": 128.0,
        "unit": "byte",
        "method": "callback",
    }
