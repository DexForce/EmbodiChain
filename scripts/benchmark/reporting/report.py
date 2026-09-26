# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Generic offline technical-report assembly over stored run records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.benchmark.core.artifacts import read_json, write_json
from .aggregation import MetricDefinition, aggregate_runs
from .tables import render_markdown_table

__all__ = ["rebuild_summary", "write_technical_report"]


def rebuild_summary(
    root: Path, *, metrics: Sequence[MetricDefinition]
) -> dict[str, object]:
    """Read ``runs.json`` and rebuild a simulator-independent summary."""
    rows = read_json(Path(root) / "runs.json")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("runs.json must contain a list of objects")
    summary = aggregate_runs(rows, metrics)
    from scripts.benchmark.core.artifacts import ArtifactStore

    ArtifactStore(root).write_metrics(summary)
    write_json(Path(root) / "summary.json", summary)
    return summary


def write_technical_report(
    path: Path,
    *,
    title: str,
    summary: Mapping[str, object],
    metric_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]] = (),
    notes: Sequence[str] = (),
) -> Path:
    """Write a standard report with measured, validation and grouped sections."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Measured metrics", ""]
    columns = tuple(metric_rows[0]) if metric_rows else ("backend", "status")
    lines.append(render_markdown_table(metric_rows, columns=columns))
    lines.extend(["", "## Validation and failures", ""])
    validation_columns = (
        tuple(validation_rows[0])
        if validation_rows
        else ("backend", "status", "reason")
    )
    lines.append(render_markdown_table(validation_rows, columns=validation_columns))
    lines.extend(["", "## Aggregated groups", ""])
    groups = summary.get("groups", [])
    if not isinstance(groups, Sequence):
        raise ValueError("summary groups must be a sequence")
    group_rows = [
        {
            "backend": group.get("backend"),
            "case_id": group.get("case_id"),
            "completed": group.get("completed"),
            "scheduled": group.get("scheduled"),
        }
        for group in groups
        if isinstance(group, Mapping)
    ]
    lines.append(
        render_markdown_table(
            group_rows,
            columns=("backend", "case_id", "completed", "scheduled"),
        )
    )
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
