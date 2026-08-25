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

"""Render the deterministic declarative Expert Program rollout report."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_REPORT_PATH",
    "REPOSITORY_ROOT",
    "SourceSnapshot",
    "TaskSizeMetric",
    "build_task_size_metrics",
    "main",
    "render_report",
]


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / "docs/design/expert_program_rollout_report.md"


@dataclass(frozen=True)
class SourceSnapshot:
    """One source file included in a task migration size snapshot.

    Args:
        path: Repository-relative source path.
        lines: Raw LF-byte count.
        bytes: Raw on-disk byte count.
    """

    path: str
    lines: int
    bytes: int


@dataclass(frozen=True)
class TaskSizeMetric:
    """Baseline and current source size for one migrated task.

    Args:
        task: Stable task label.
        baseline_lines: Recorded pre-migration LF-byte count.
        baseline_bytes: Recorded pre-migration byte count.
        sources: Explicit current source snapshots.
    """

    task: str
    baseline_lines: int
    baseline_bytes: int
    sources: tuple[SourceSnapshot, ...]

    @property
    def current_lines(self) -> int:
        """Return the current LF-delimited line count across all source files."""
        return sum(source.lines for source in self.sources)

    @property
    def current_bytes(self) -> int:
        """Return the current raw byte count across all source files."""
        return sum(source.bytes for source in self.sources)


@dataclass(frozen=True)
class _TaskSizeSpec:
    """Stable baseline snapshot and explicit current source paths."""

    task: str
    baseline_lines: int
    baseline_bytes: int
    baseline_blob: str
    source_paths: tuple[str, ...]


_TASK_SIZE_SPECS = (
    _TaskSizeSpec(
        task="Cube",
        baseline_lines=598,
        baseline_bytes=23_912,
        baseline_blob="1965563b060d1fc889f03ad13d47655c2edcd99b",
        source_paths=(
            "embodichain_tasks/embodichain_tasks/expert_program/"
            "repeated_pick_place.py",
            "embodichain_tasks/configs/expert_program/repeated_pick_place.yaml",
        ),
    ),
    _TaskSizeSpec(
        task="Drawer",
        baseline_lines=245,
        baseline_bytes=8_833,
        baseline_blob="3b4cbdc09537098b4f109d46efb8785b88f31ce1",
        source_paths=(
            "embodichain_tasks/embodichain_tasks/expert_program/open_drawer.py",
            "embodichain_tasks/configs/expert_program/open_drawer.yaml",
        ),
    ),
)


_FRAMEWORK_CAPABILITIES = (
    (
        "Pick + Place(at)",
        "framework-tested",
        "per-embodiment integration",
        "Typed goals, compilation, execution, and terminal effects are covered.",
    ),
    (
        "Physical attach/release evidence",
        "framework-tested",
        "per-embodiment integration",
        "Effects require live constraint and object-to-endpoint pose evidence.",
    ),
    (
        "Slide",
        "framework-tested",
        "per-embodiment integration",
        "Typed handle geometry, grasping, and axis-constrained motion are covered.",
    ),
    (
        "Articulation joint validator",
        "framework-tested",
        "per-task integration",
        "Measured joint-state application acceptance is covered.",
    ),
    (
        "Schema-v2 sequential",
        "framework-tested",
        "per-task integration",
        "Ordered call execution and failure propagation are covered.",
    ),
    (
        "HandOver",
        "framework-tested",
        "integration-required",
        "No landed task integration is claimed by this report.",
    ),
    (
        "Place relation (on/inside)",
        "framework-tested",
        "integration-required",
        "Embodiment frames and relation validators must be supplied.",
    ),
    (
        "Registered call",
        "framework-tested",
        "integration-required",
        "Production registration must declare and validate its concrete contract.",
    ),
    (
        "Schema-v2 parallel",
        "framework-tested",
        "integration-required",
        "Fail-closed by default; production use requires an authoritative validator.",
    ),
)


_LANDED_INTEGRATIONS = (
    (
        "UR5",
        "Cube Pick + Place",
        "Pick + Place(at)",
        "pose relation; no task-local constraint observer",
        "schema-v2 sequential",
        "checked in",
        "blocked: install grasp evidence before a physical gate",
    ),
    (
        "UR5",
        "Open Drawer",
        "Registered call -> Slide",
        "articulation joint validator",
        "schema-v2 sequential",
        "checked in",
        "seed 0 regression passed; broader multi-seed gate remains",
    ),
)


def _count_source(repository_root: Path, relative_path: str) -> SourceSnapshot:
    """Count raw LF bytes and total bytes for one explicit repository file."""
    data = (repository_root / relative_path).read_bytes()
    return SourceSnapshot(
        path=relative_path,
        lines=data.count(b"\n"),
        bytes=len(data),
    )


def build_task_size_metrics(
    repository_root: str | Path = REPOSITORY_ROOT,
) -> tuple[TaskSizeMetric, ...]:
    """Build deterministic migration metrics from the four declared source files.

    Args:
        repository_root: EmbodiChain checkout root containing the declared files.

    Returns:
        Metrics in the stable order defined by the report specification.
    """
    root = Path(repository_root)
    return tuple(
        TaskSizeMetric(
            task=spec.task,
            baseline_lines=spec.baseline_lines,
            baseline_bytes=spec.baseline_bytes,
            sources=tuple(
                _count_source(root, relative_path)
                for relative_path in spec.source_paths
            ),
        )
        for spec in _TASK_SIZE_SPECS
    )


def _render_table(headers: tuple[str, ...], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a Markdown table with stable column and row ordering."""
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _format_delta(current: int, baseline: int) -> str:
    """Format an absolute and baseline-relative size delta."""
    delta = current - baseline
    percentage = delta / baseline * 100.0
    return f"{delta:+d} ({percentage:+.1f}%)"


def render_report(metrics: Sequence[TaskSizeMetric]) -> str:
    """Render the static rollout snapshot as deterministic Markdown.

    Args:
        metrics: Task size metrics, normally from :func:`build_task_size_metrics`.

    Returns:
        Complete Markdown document ending with exactly one newline.
    """
    if not metrics:
        raise ValueError("metrics must contain at least one task snapshot.")

    metric_rows = []
    for metric in metrics:
        source_paths = "<br>".join(f"`{source.path}`" for source in metric.sources)
        metric_rows.append(
            (
                metric.task,
                str(metric.baseline_lines),
                str(metric.current_lines),
                _format_delta(metric.current_lines, metric.baseline_lines),
                str(metric.baseline_bytes),
                str(metric.current_bytes),
                _format_delta(metric.current_bytes, metric.baseline_bytes),
                source_paths,
            )
        )

    total_baseline_lines = sum(metric.baseline_lines for metric in metrics)
    total_current_lines = sum(metric.current_lines for metric in metrics)
    total_baseline_bytes = sum(metric.baseline_bytes for metric in metrics)
    total_current_bytes = sum(metric.current_bytes for metric in metrics)
    metric_rows.append(
        (
            "Total",
            str(total_baseline_lines),
            str(total_current_lines),
            _format_delta(total_current_lines, total_baseline_lines),
            str(total_baseline_bytes),
            str(total_current_bytes),
            _format_delta(total_current_bytes, total_baseline_bytes),
            "the four files above",
        )
    )

    lines = [
        "# Declarative Expert Program Rollout Report",
        "",
        (
            "This is a deterministic, static Phase 8 snapshot of checked-in "
            "framework and integration code. It does not run simulation, report "
            "physical acceptance, or certify production readiness for an embodiment."
        ),
        "",
        "## Framework Contract Matrix",
        "",
        (
            "`framework-tested` describes the reusable framework contract only. A "
            "task appears in the matrix below only when its integration/production "
            "code is checked in; that code status does not imply physical acceptance."
        ),
        "",
    ]
    lines.extend(
        _render_table(
            ("Capability", "Framework status", "Integration gate", "Scope"),
            _FRAMEWORK_CAPABILITIES,
        )
    )
    lines.extend(
        [
            "",
            (
                "Parallel execution remains fail-closed by default. Resource "
                "declarations alone do not authorize production concurrency; the "
                "selected embodiment must provide an authoritative validator."
            ),
            "",
            "## Checked-in Integration Matrix",
            "",
            (
                "Only the two checked-in vertical slices below are classified as "
                "integration/production code. Physical acceptance is tracked "
                "separately."
            ),
            "",
        ]
    )
    lines.extend(
        _render_table(
            (
                "Embodiment",
                "Task",
                "Skill contract",
                "Terminal effect",
                "Program schema",
                "Code status",
                "Physical acceptance",
            ),
            _LANDED_INTEGRATIONS,
        )
    )
    lines.extend(
        [
            "",
            (
                "HandOver, Place relations (`on`/`inside`), and schema-v2 parallel "
                "are framework-tested but integration-required. They are "
                "intentionally not listed as checked-in integrations."
            ),
            "",
            (
                "Both checked-in environment classes have zero task-local motion or "
                "demo-generation overrides; "
                "`test_task_classes_do_not_override_motion_or_demo_generation` "
                "keeps that structural metric at zero."
            ),
            "",
            "## Migration Size Snapshot",
            "",
            (
                "The baseline is a fixed, manually recorded pre-migration snapshot: "
                "Cube is 598 lines / 23912 bytes and Drawer is 245 lines / 8833 bytes. "
                "The tool does not inspect Git history. Current values are recomputed "
                "only from the four explicit files in the table."
            ),
            "",
            (
                "Baseline identity: Cube uses legacy Git blob "
                f"`{_TASK_SIZE_SPECS[0].baseline_blob}` and Drawer uses legacy Git "
                f"blob `{_TASK_SIZE_SPECS[1].baseline_blob}`. Current Python paths "
                "point to the consolidated canonical integrations; blob IDs remain "
                "stable across stack rebases."
            ),
            "",
            (
                "Current totals include only the canonical environment "
                "implementations and their declarative programs; removed legacy "
                "modules are not counted."
            ),
            "",
            (
                "Counting rule: `lines` is the number of raw LF (`0x0A`) bytes; "
                "`bytes` is the raw on-disk byte length. Counts are summed per task "
                "without normalizing encoding or line endings."
            ),
            "",
        ]
    )
    lines.extend(
        _render_table(
            (
                "Task",
                "Baseline lines",
                "Current lines",
                "Line delta",
                "Baseline bytes",
                "Current bytes",
                "Byte delta",
                "Current source files",
            ),
            metric_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Demo Success Measurement",
            "",
            (
                "`scripts/benchmark/expert_program/demo_success.py` executes each "
                "fixed seed exactly once, always discards the episode buffer, and "
                "counts executor exceptions as failed rows. It writes raw JSON plus "
                "a three-table Markdown report. Its CLI supports offline raw-JSON "
                "re-aggregation and an explicit `--run-simulation` mode that "
                "constructs one standard Gym environment from Gym and Expert "
                "Program configurations."
            ),
            "",
            (
                "The supported-simulation Open Drawer seed-0 regression is checked "
                "in and passes locally; no multi-seed success-rate or release gate "
                "is checked in yet. Repeated Cube needs an environment-qualified "
                "grasp-evidence provider before a physical rate is meaningful."
            ),
            "",
            "## Drift Check",
            "",
            (
                "Regenerate the checked-in report after an intentional source or "
                "capability snapshot change:"
            ),
            "",
            "```bash",
            "python scripts/tools/expert_program_rollout_report.py",
            "```",
            "",
            "CI and local validation can reject stale output without rewriting it:",
            "",
            "```bash",
            "python scripts/tools/expert_program_rollout_report.py --check",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate or check the declarative Expert Program rollout report."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the output file differs from the deterministic render.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown output path (defaults to the checked-in design report).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the rollout report or check its checked-in representation.

    Args:
        argv: Optional command-line argument sequence for tests and embedding.

    Returns:
        Zero on success, or one when ``--check`` detects missing or stale output.
    """
    args = _build_parser().parse_args(argv)
    rendered = render_report(build_task_size_metrics())
    output = args.output

    if args.check:
        try:
            existing = output.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"rollout report is missing: {output}")
            return 1
        if existing != rendered:
            print(f"rollout report is stale: {output}")
            return 1
        print(f"rollout report is up to date: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"wrote rollout report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
