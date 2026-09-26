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

"""G-03 fixture tests fix stage accounting and receipt idempotency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_fixture_keeps_attempt_denominators_and_confirmed_yield(tmp_path: Path) -> None:
    from scripts.benchmark.expert_generation.contracts import GenerationCase
    from scripts.benchmark.expert_generation.runner import run_generation

    case = GenerationCase(
        experiment_id="g03-fixture",
        case_id="case-0",
        source_kind="atomic_action",
        source_id="fixture_action",
        source_revision="fixture:v1",
        scene_case_id="scene-0",
        initial_state_id="initial-0",
        seed=7,
    )

    def execute(current: GenerationCase, attempt_id: int):
        from scripts.benchmark.expert_generation.fixtures import fixture_executor

        return fixture_executor(current, attempt_id)

    result = run_generation(
        (case,), attempts_per_case=4, execute=execute, output_root=tmp_path
    )
    assert result.summary["attempted"] == 4
    assert result.summary["execution_completed"] == 4
    assert result.summary["validation_passed"] == 3
    assert result.summary["persisted"] == 2
    assert result.summary["accepted"] == 2
    assert result.summary["accepted_yield"] == 0.5
    assert [row.data_status for row in result.attempts] == [
        "accepted",
        "rejected",
        "not_persisted",
        "accepted",
    ]
    raw = (result.root / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw) == 4
    assert json.loads((result.root / "metrics.json").read_text())["accepted"] == 2
    assert (result.root / "quality.json").is_file()
    assert (result.root / "report.md").is_file()
    metrics = json.loads((result.root / "metrics.json").read_text())
    assert metrics["attempt_aggregates"]["groups"]
    artifact_index = json.loads((result.root / "artifact_index.json").read_text())
    assert {item["kind"] for item in artifact_index["artifacts"]} >= {
        "raw_observations",
        "aggregate_metrics",
        "quality_records",
        "report",
    }


def test_duplicate_receipt_does_not_increase_accepted_count(tmp_path: Path) -> None:
    from scripts.benchmark.expert_generation.contracts import (
        GenerationCase,
        GenerationAttempt,
        PersistenceReceipt,
    )
    from scripts.benchmark.expert_generation.runner import run_generation

    case = GenerationCase(
        experiment_id="g03-duplicate",
        case_id="case-0",
        source_kind="handwritten",
        source_id="fixture",
        source_revision="fixture:v1",
        scene_case_id="scene-0",
        initial_state_id="initial-0",
        seed=1,
    )
    receipt = PersistenceReceipt(
        episode_id="episode-0",
        commit_id="commit-0",
        submission_id=0,
        storage_id="storage-0",
        confirmed=True,
    )

    def execute(current: GenerationCase, attempt_id: int) -> GenerationAttempt:
        return GenerationAttempt(
            case=current,
            attempt_id=attempt_id,
            execution_status="completed",
            validation_status="passed",
            task_status="passed",
            receipt=receipt,
            elapsed_s=0.1,
        )

    result = run_generation(
        (case,), attempts_per_case=2, execute=execute, output_root=tmp_path
    )
    assert result.summary["accepted"] == 1
    assert result.summary["duplicates"] == 1
    assert result.attempts[1].data_status == "duplicate"


def test_budget_exhaustion_retains_not_run_attempts(tmp_path: Path) -> None:
    from scripts.benchmark.core.contracts import Budget
    from scripts.benchmark.expert_generation.contracts import GenerationCase
    from scripts.benchmark.expert_generation.runner import run_generation

    case = GenerationCase(
        experiment_id="g03-budget",
        case_id="case-0",
        source_kind="motion_generator",
        source_id="fixture",
        source_revision="fixture:v1",
        scene_case_id="scene-0",
        initial_state_id="initial-0",
        seed=2,
    )
    result = run_generation(
        (case,),
        attempts_per_case=3,
        execute=lambda current, attempt: None,
        budget=Budget(max_runs=1, max_attempts=2),
        output_root=tmp_path,
    )
    assert [attempt.execution_status for attempt in result.attempts] == [
        "failed",
        "failed",
        "not_run",
    ]
    assert result.summary["attempted"] == 3
    assert result.summary["not_run"] == 1


def test_invalid_receipt_identity_is_rejected() -> None:
    from scripts.benchmark.expert_generation.contracts import (
        GenerationCase,
        GenerationAttempt,
        PersistenceReceipt,
    )

    case = GenerationCase(
        experiment_id="g03-invalid",
        case_id="case-0",
        source_kind="task_program",
        source_id="fixture",
        source_revision="fixture:v1",
        scene_case_id="scene-0",
        initial_state_id="initial-0",
        seed=3,
    )
    with pytest.raises(ValueError, match="episode_id"):
        GenerationAttempt(
            case=case,
            attempt_id=0,
            execution_status="completed",
            validation_status="passed",
            task_status="passed",
            receipt=PersistenceReceipt(
                episode_id="",
                commit_id="commit",
                submission_id=0,
                storage_id="storage",
                confirmed=True,
            ),
        )


def test_expert_generation_cli_fixture_is_offline(tmp_path: Path) -> None:
    from scripts.benchmark.expert_generation.run_benchmark import main

    main(["--fixture", "--attempts", "2", "--output", str(tmp_path)])
    run_dirs = sorted(path for path in tmp_path.iterdir() if path.is_dir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "report.md").is_file()
