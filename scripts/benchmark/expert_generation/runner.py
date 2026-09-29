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

"""Budgeted source-neutral G-03 attempt runner.

The executor callback is the integration boundary for #670. It may call the
production CandidateCoordinator/GenerationSession/PhysicalExecutor stack, but
the benchmark runner owns no simulator, candidate or persistence state machine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import (
    ArtifactStore,
    create_experiment_directory,
    write_json,
)
from scripts.benchmark.core.contracts import Budget, MetricDefinition
from scripts.benchmark.core.planning import BudgetExceeded, BudgetLedger
from scripts.benchmark.reporting.aggregation import aggregate_attempts

from .contracts import GenerationAttempt, GenerationCase
from .report import write_generation_report

__all__ = ["GenerationRun", "run_generation"]

AttemptExecutor = Callable[[GenerationCase, int], GenerationAttempt | None]


@dataclass(frozen=True)
class GenerationRun:
    """Persisted G-03 attempt results and summary."""

    root: Path
    attempts: tuple[GenerationAttempt, ...]
    summary: Mapping[str, Any]
    budget: Mapping[str, Any]


def _not_run(case: GenerationCase, attempt_id: int, reason: str) -> GenerationAttempt:
    """Create an explicit row for a budget-exhausted attempt."""
    return GenerationAttempt(
        case=case,
        attempt_id=attempt_id,
        execution_status="not_run",
        validation_status="not_run",
        task_status="not_evaluated",
        failure_reason=reason,
    )


def _failed(case: GenerationCase, attempt_id: int, reason: str) -> GenerationAttempt:
    """Convert an executor crash or malformed result into retained evidence."""
    return GenerationAttempt(
        case=case,
        attempt_id=attempt_id,
        execution_status="failed",
        validation_status="not_run",
        task_status="not_evaluated",
        failure_reason=reason,
    )


def _classify_attempt(
    attempt: GenerationAttempt,
    seen_commits: dict[str, str],
) -> GenerationAttempt:
    """Apply confirmed receipt and duplicate semantics exactly once."""
    receipt = attempt.receipt
    if not attempt.execution_completed:
        return attempt.with_data_status("not_persisted")
    if not attempt.validation_passed or not attempt.task_passed:
        return attempt.with_data_status("rejected")
    if receipt is None or not receipt.confirmed:
        return attempt.with_data_status("not_persisted")
    previous = seen_commits.get(receipt.commit_id)
    if previous is not None:
        return attempt.with_data_status("duplicate")
    seen_commits[receipt.commit_id] = receipt.storage_id
    return attempt.with_data_status("accepted")


def _summary(attempts: Sequence[GenerationAttempt]) -> dict[str, Any]:
    """Summarize independent attempt populations without replacing missing data."""
    attempted = len(attempts)
    completed = sum(item.execution_completed for item in attempts)
    validation = sum(item.validation_passed for item in attempts)
    task = sum(item.task_passed for item in attempts)
    persisted = sum(item.data_status in {"accepted", "duplicate"} for item in attempts)
    accepted = sum(item.data_status == "accepted" for item in attempts)
    duplicates = sum(item.data_status == "duplicate" for item in attempts)
    not_run = sum(item.execution_status == "not_run" for item in attempts)
    elapsed = sum(
        item.elapsed_s for item in attempts if item.execution_status == "completed"
    )
    return {
        "schema_version": 1,
        "attempted": attempted,
        "execution_completed": completed,
        "validation_passed": validation,
        "task_passed": task,
        "persisted": persisted,
        "accepted": accepted,
        "duplicates": duplicates,
        "not_run": not_run,
        "elapsed_s": elapsed,
        "accepted_yield": accepted / attempted if attempted else None,
        "accepted_episodes_per_hour": accepted / elapsed * 3600 if elapsed else None,
    }


def run_generation(
    cases: Sequence[GenerationCase],
    *,
    attempts_per_case: int,
    execute: AttemptExecutor,
    output_root: Path,
    budget: Budget | None = None,
    experiment_id: str | None = None,
) -> GenerationRun:
    """Run bounded expert-generation attempts and persist standard artifacts.

    ``execute`` is responsible for source adaptation, physical execution,
    measured validation and the production EpisodeSink receipt. The runner only
    classifies those outcomes and ensures a confirmed receipt is required for
    accepted data. Every planned attempt receives a retained row, including
    budget-exhausted ``not_run`` rows.
    """
    if not cases:
        raise ValueError("at least one generation case is required")
    if type(attempts_per_case) is not int or attempts_per_case < 1:
        raise ValueError("attempts_per_case must be a positive integer")
    experiment = experiment_id or cases[0].experiment_id
    if any(case.experiment_id != experiment for case in cases):
        raise ValueError("all cases must belong to the same experiment")
    planned_attempts = len(cases) * attempts_per_case
    resolved_budget = budget or Budget(
        max_runs=len(cases), max_attempts=planned_attempts
    )
    definition = {
        "schema_version": 1,
        "experiment_id": experiment,
        "domain": "expert_generation",
        "cases": [case.to_dict() for case in cases],
        "attempts_per_case": attempts_per_case,
        "budget": resolved_budget.to_dict(),
        "acceptance": {
            "execution_status": "completed",
            "validation_status": "passed",
            "task_status": "passed",
            "persistence": "confirmed_receipt",
        },
    }
    root = create_experiment_directory(
        Path(output_root),
        experiment_id=experiment,
        config={"cases": [case.to_dict() for case in cases]},
        definition=definition,
        assets_manifest={"assets": []},
    )
    store = ArtifactStore(root)
    ledger = BudgetLedger(resolved_budget)
    seen_commits: dict[str, str] = {}
    rows: list[GenerationAttempt] = []
    for case in cases:
        try:
            ledger.start_run()
        except BudgetExceeded as exc:
            for attempt_id in range(attempts_per_case):
                rows.append(_not_run(case, attempt_id, f"budget_exhausted:{exc}"))
            continue
        for attempt_id in range(attempts_per_case):
            try:
                ledger.start_attempt()
            except BudgetExceeded as exc:
                attempt = _not_run(case, attempt_id, f"budget_exhausted:{exc}")
            else:
                try:
                    attempt = execute(case, attempt_id)
                    if attempt is None:
                        attempt = _failed(case, attempt_id, "executor_returned_none")
                    if attempt.case != case or attempt.attempt_id != attempt_id:
                        raise ValueError(
                            "executor returned mismatched attempt identity"
                        )
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - retain domain failures as evidence.
                    attempt = _failed(
                        case,
                        attempt_id,
                        f"{type(exc).__name__}: {exc}",
                    )
                attempt = _classify_attempt(attempt, seen_commits)
                ledger.finish_attempt(attempt.execution_status)
            rows.append(attempt)
            store.append_raw(attempt.to_dict())
    attempts = tuple(rows)
    summary = _summary(attempts)
    summary["budget"] = ledger.to_dict()
    metrics = dict(summary)
    metrics["cases"] = len(cases)
    metrics["attempts_per_case"] = attempts_per_case
    normalized_rows = [
        {
            "run_id": item.to_dict()["record_id"],
            "attempt_id": item.attempt_id,
            "backend": item.case.source_kind,
            "case_id": item.case.case_id,
            "status": item.execution_status,
            "metrics": {
                "elapsed_s": item.elapsed_s,
                "accepted": float(item.data_status == "accepted"),
            },
            "metric_reasons": {
                "elapsed_s": (
                    "not_run" if item.execution_status == "not_run" else "missing"
                ),
            },
        }
        for item in attempts
    ]
    metrics["attempt_aggregates"] = aggregate_attempts(
        normalized_rows,
        (
            MetricDefinition(
                "elapsed_s", "s", "attempt", denominator="attempts_started"
            ),
            MetricDefinition(
                "accepted", "ratio", "attempt", denominator="attempts_started"
            ),
        ),
    )
    write_json(root / "metrics.json", metrics)
    write_json(
        root / "quality.json",
        {
            "schema_version": 1,
            "acceptance_rule": definition["acceptance"],
            "attempts": [
                {
                    "record_id": f"{item.case.case_id}/attempt-{item.attempt_id}",
                    "case_id": item.case.case_id,
                    "execution_status": item.execution_status,
                    "validation_status": item.validation_status,
                    "task_status": item.task_status,
                    "data_status": item.data_status,
                    "failure_reason": item.failure_reason,
                }
                for item in attempts
            ],
        },
    )
    write_json(root / "budget.json", ledger.to_dict())
    write_generation_report(root, attempts, summary)
    for name, kind in (
        ("definition.json", "experiment_definition"),
        ("config.json", "effective_config"),
        ("effective_config.yaml", "effective_config_yaml"),
        ("assets_manifest.json", "assets_manifest"),
        ("raw.jsonl", "raw_observations"),
        ("metrics.json", "aggregate_metrics"),
        ("quality.json", "quality_records"),
        ("budget.json", "budget_ledger"),
        ("report.md", "report"),
    ):
        store.register_artifact(root / name, kind=kind)
    return GenerationRun(root, attempts, summary, ledger.to_dict())
