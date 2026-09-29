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

"""Offline G-03 attempt, stage and confirmed-yield report."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from scripts.benchmark.core.artifacts import read_json
from scripts.benchmark.reporting.tables import render_markdown_table

from .contracts import GenerationAttempt

__all__ = ["rebuild_generation_report", "write_generation_report"]


def _rows(attempts: Sequence[GenerationAttempt]) -> list[dict[str, object]]:
    """Build one report row per attempt."""
    return [
        {
            "case": attempt.case.case_id,
            "attempt": attempt.attempt_id,
            "source": attempt.case.source_kind,
            "execution": attempt.execution_status,
            "validation": attempt.validation_status,
            "task": attempt.task_status,
            "persistence": attempt.data_status,
            "elapsed_s": attempt.elapsed_s,
            "reason": attempt.failure_reason,
        }
        for attempt in attempts
    ]


def write_generation_report(
    root: Path,
    attempts: Sequence[GenerationAttempt],
    summary: dict[str, Any],
) -> Path:
    """Write one report with attempt populations and confirmed yield."""
    output = Path(root) / "report.md"
    grouped: defaultdict[str, list[GenerationAttempt]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.case.case_id].append(attempt)
    population_rows = []
    for case_id, case_attempts in grouped.items():
        population_rows.append(
            {
                "case": case_id,
                "attempted": len(case_attempts),
                "completed": sum(item.execution_completed for item in case_attempts),
                "validated": sum(item.validation_passed for item in case_attempts),
                "task_passed": sum(item.task_passed for item in case_attempts),
                "persisted": sum(
                    item.data_status in {"accepted", "duplicate"}
                    for item in case_attempts
                ),
                "accepted": sum(
                    item.data_status == "accepted" for item in case_attempts
                ),
                "accepted_yield": (
                    sum(item.data_status == "accepted" for item in case_attempts)
                    / len(case_attempts)
                    if case_attempts
                    else None
                ),
            }
        )
    failure_counts = Counter(
        attempt.failure_reason or attempt.data_status
        for attempt in attempts
        if attempt.failure_reason or attempt.data_status not in {"accepted"}
    )
    lines = [
        "# G-03 Expert Generation Fixture",
        "",
        "Accepted data requires completed execution, measured validation, task success, and a confirmed persistence receipt.",
        "",
        "## Attempt population",
        "",
        render_markdown_table(
            population_rows,
            columns=(
                "case",
                "attempted",
                "completed",
                "validated",
                "task_passed",
                "persisted",
                "accepted",
                "accepted_yield",
            ),
        ),
        "",
        "## Stage outcomes",
        "",
        render_markdown_table(
            _rows(attempts),
            columns=(
                "case",
                "attempt",
                "source",
                "execution",
                "validation",
                "task",
                "persistence",
                "elapsed_s",
                "reason",
            ),
        ),
        "",
        "## Confirmed yield",
        "",
        render_markdown_table(
            [
                {
                    "attempted": summary["attempted"],
                    "accepted": summary["accepted"],
                    "persisted": summary["persisted"],
                    "duplicates": summary["duplicates"],
                    "not_run": summary["not_run"],
                    "accepted_yield": summary["accepted_yield"],
                    "accepted_episodes_per_hour": summary["accepted_episodes_per_hour"],
                }
            ],
            columns=(
                "attempted",
                "accepted",
                "persisted",
                "duplicates",
                "not_run",
                "accepted_yield",
                "accepted_episodes_per_hour",
            ),
        ),
    ]
    if failure_counts:
        lines.extend(["", "## Failure accounting", ""])
        lines.append(
            render_markdown_table(
                [
                    {"reason": reason, "count": count}
                    for reason, count in sorted(failure_counts.items())
                ],
                columns=("reason", "count"),
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def rebuild_generation_report(root: Path) -> Path:
    """Rebuild a fixture report from standard raw/metrics artifacts."""
    raw_path = Path(root) / "raw.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    # The report-only path intentionally remains lightweight: the stored
    # summary and raw rows are already the canonical offline inputs. Rebuilding
    # writes the same summary/metrics references and preserves original rows.
    summary = read_json(Path(root) / "metrics.json")
    raw_rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = Counter(
        row.get("failure_reason") or row.get("data_status", "missing")
        for row in raw_rows
        if row.get("failure_reason") or row.get("data_status") != "accepted"
    )
    population = []
    by_case: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        by_case[str(row.get("case_id", "unknown"))].append(row)
    for case_id, rows in by_case.items():
        population.append(
            {
                "case": case_id,
                "attempted": len(rows),
                "completed": sum(
                    row.get("execution_status") == "completed" for row in rows
                ),
                "validated": sum(
                    row.get("validation_status") == "passed" for row in rows
                ),
                "persisted": sum(
                    row.get("data_status") in {"accepted", "duplicate"} for row in rows
                ),
                "accepted": sum(row.get("data_status") == "accepted" for row in rows),
            }
        )
    output = Path(root) / "report.md"
    lines = [
        "# G-03 Expert Generation Fixture",
        "",
        "Offline report rebuilt from raw.jsonl and metrics.json.",
        "",
        "## Confirmed yield",
        "",
        render_markdown_table(
            population,
            columns=(
                "case",
                "attempted",
                "completed",
                "validated",
                "persisted",
                "accepted",
            ),
        ),
        "",
        render_markdown_table(
            [summary],
            columns=(
                "attempted",
                "accepted",
                "persisted",
                "duplicates",
                "not_run",
                "accepted_yield",
                "accepted_episodes_per_hour",
            ),
        ),
        "",
        "## Stage outcomes",
        "",
        render_markdown_table(
            [
                {
                    "case": row.get("case_id"),
                    "attempt": row.get("attempt_id"),
                    "execution": row.get("execution_status"),
                    "validation": row.get("validation_status"),
                    "task": row.get("task_status"),
                    "persistence": row.get("data_status"),
                    "reason": row.get("failure_reason"),
                }
                for row in raw_rows
            ],
            columns=(
                "case",
                "attempt",
                "execution",
                "validation",
                "task",
                "persistence",
                "reason",
            ),
        ),
        "",
        f"Raw attempt records: {len(raw_rows)}",
    ]
    if failures:
        lines.extend(["", "## Failure accounting", ""])
        lines.append(
            render_markdown_table(
                [
                    {"reason": key, "count": value}
                    for key, value in sorted(failures.items())
                ],
                columns=("reason", "count"),
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
