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

"""Lightweight hierarchical profiler shared by simulation and environments.

The profiler records sections below an explicit root section. A root entered
while another root is active is transparent, allowing lower-level simulation
instrumentation to compose with an environment's profiling hierarchy without
adding duplicate path components.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch

from embodichain.utils import configclass, logger

__all__ = ["ProfilerCfg", "Profiler"]

_MODULE_COLOR_CODES = (
    39,
    82,
    214,
    207,
    45,
    220,
    141,
    203,
    75,
    121,
    177,
    208,
    50,
    110,
    190,
    213,
)


@configclass
class ProfilerCfg:
    """Configuration for hierarchical wall-time profiling."""

    enable_time: bool = True
    """Enable per-section wall-time statistics (mean/min/max/std)."""

    sync_cuda: bool = False
    """Synchronize CUDA at section boundaries for accurate GPU wall time."""

    warmup_steps: int = 5
    """Number of top-level root sections to discard before recording."""

    nvtx: bool = False
    """Push NVTX ranges for sections so they appear in Nsight Systems."""

    output_path: str | None = None
    """Optional JSON path written by :meth:`Profiler.report`."""

    color_output: bool = True
    """Color terminal report rows by logical module.

    This only affects the logged table; JSON report data remains unchanged.
    """


@dataclass
class _SectionStats:
    """Running statistics for one named section."""

    n: int = 0
    total_s: float = 0.0
    sq_s: float = 0.0
    min_s: float = math.inf
    max_s: float = 0.0


@dataclass
class _ReportRow:
    """One formatted row in the profiler report tree."""

    path: str
    depth: int
    display_name: str
    calls: int
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    total_s: float
    pct_parent: float
    module: str | None


class Profiler:
    """Hierarchical wall-time profiler.

    Args:
        cfg: Profiler configuration. ``None`` disables profiling entirely.
        device: Device used for optional CUDA synchronization and NVTX ranges.

    .. note::
        One profiler tracks one synchronous call stack. Use it on the
        simulation thread that owns the associated simulation manager.
    """

    def __init__(self, cfg: Optional[ProfilerCfg], device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self._is_cuda = device.type == "cuda"
        self._stats: Dict[str, _SectionStats] = {}
        self._stack: list[str] = []
        self._root_active = False
        self._warmup = cfg.warmup_steps if cfg is not None else 0
        self._on = cfg is not None and cfg.enable_time
        self._do_time = self._on
        self._do_nvtx = self._on and cfg.nvtx and self._is_cuda

    @property
    def enabled(self) -> bool:
        """Whether profiling is active."""

        return self._on

    @contextmanager
    def section(self, name: str, *, is_root: bool = False) -> Iterator[None]:
        """Record wall time for a named section.

        Args:
            name: Leaf section name. The full name is derived from the active
                section stack.
            is_root: Whether this section starts a top-level profiling sample.
                A nested root is transparent and its children remain attached
                to the active outer hierarchy.
        """

        if not self._on:
            yield
            return
        if is_root and self._root_active:
            yield
            return
        if not is_root and not self._root_active:
            yield
            return

        opens_root = is_root
        if opens_root:
            self._root_active = True

        try:
            if self._do_time and self.cfg.sync_cuda and self._is_cuda:
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()
            if self._do_nvtx:
                torch.cuda.nvtx.range_push(name)
            self._stack.append(name)
            try:
                yield
            finally:
                if self._do_time and self.cfg.sync_cuda and self._is_cuda:
                    torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - start_time
                full_name = ".".join(self._stack)
                self._stack.pop()
                if self._do_nvtx:
                    torch.cuda.nvtx.range_pop()

                if self._warmup > 0:
                    if opens_root:
                        self._warmup -= 1
                else:
                    self._record(full_name, elapsed)
        finally:
            if opens_root:
                self._root_active = False

    def report(self) -> Dict[str, object]:
        """Log a profiling report table and optionally dump JSON.

        Returns:
            Report data, or an empty dictionary when profiling is disabled.
        """

        if not self._on:
            return {}
        data = self._build_report_data()
        self._log_table(data)
        if self.cfg.output_path:
            try:
                with open(self.cfg.output_path, "w") as output_file:
                    json.dump(data, output_file, indent=2)
                logger.log_info(f"[Profiler] report dumped to {self.cfg.output_path}")
            except OSError as error:
                logger.log_warning(f"[Profiler] failed to dump JSON: {error}")
        return data

    def _record(self, name: str, elapsed: float) -> None:
        stats = self._stats.get(name)
        if stats is None:
            stats = _SectionStats()
            self._stats[name] = stats
        stats.n += 1
        stats.total_s += elapsed
        stats.sq_s += elapsed * elapsed
        if elapsed < stats.min_s:
            stats.min_s = elapsed
        if elapsed > stats.max_s:
            stats.max_s = elapsed

    def _build_report_data(self) -> Dict[str, object]:
        sections: Dict[str, Dict[str, float]] = {}
        for name in sorted(self._stats):
            stats = self._stats[name]
            mean = stats.total_s / stats.n if stats.n else 0.0
            variance = max(0.0, stats.sq_s / stats.n - mean * mean) if stats.n else 0.0
            sections[name] = {
                "calls": stats.n,
                "mean_ms": mean * 1e3,
                "min_ms": (stats.min_s if stats.n else 0.0) * 1e3,
                "max_ms": (stats.max_s if stats.n else 0.0) * 1e3,
                "std_ms": math.sqrt(variance) * 1e3,
                "total_s": stats.total_s,
            }
        rows = self._build_tree_rows(sections)
        return {
            "sections": sections,
            "table": {
                "columns": [
                    "section",
                    "calls",
                    "mean_ms",
                    "min_ms",
                    "max_ms",
                    "std_ms",
                    "total_s",
                    "pct_parent",
                ],
                "rows": [
                    {
                        "path": row.path,
                        "depth": row.depth,
                        "section": row.display_name,
                        "calls": row.calls,
                        "mean_ms": row.mean_ms,
                        "min_ms": row.min_ms,
                        "max_ms": row.max_ms,
                        "std_ms": row.std_ms,
                        "total_s": row.total_s,
                        "pct_parent": row.pct_parent,
                    }
                    for row in rows
                ],
            },
        }

    def _to_row(
        self,
        section_path: str,
        stats: Dict[str, float],
        parent_total: float,
        depth: int,
        module: str | None,
    ) -> _ReportRow:
        pct = 100.0 * stats["total_s"] / parent_total if parent_total > 0 else 0.0
        leaf = section_path if depth == 0 else section_path.split(".")[-1]
        return _ReportRow(
            path=section_path,
            depth=depth,
            display_name=f"{'  ' * depth}{leaf}",
            calls=int(stats["calls"]),
            mean_ms=stats["mean_ms"],
            min_ms=stats["min_ms"],
            max_ms=stats["max_ms"],
            std_ms=stats["std_ms"],
            total_s=stats["total_s"],
            pct_parent=pct,
            module=module,
        )

    def _root_paths(self, sections: Dict[str, Dict[str, float]]) -> list[str]:
        preferred = [name for name in ("step", "reset") if name in sections]
        remaining = sorted(
            name
            for name in sections
            if "." not in name and name not in {"step", "reset"}
        )
        return preferred + remaining

    def _module_name(self, section_path: str, depth: int) -> str | None:
        if depth == 0:
            return None
        parts = section_path.split(".")
        return parts[1] if len(parts) > 1 else parts[0]

    def _build_tree_rows(
        self, sections: Dict[str, Dict[str, float]]
    ) -> list[_ReportRow]:
        rows: list[_ReportRow] = []
        for root in self._root_paths(sections):
            self._append_tree(rows, root, sections, depth=0)
        return rows

    def _append_tree(
        self,
        rows: list[_ReportRow],
        root_path: str,
        sections: Dict[str, Dict[str, float]],
        parent_total: float | None = None,
        depth: int = 0,
    ) -> None:
        stats = sections.get(root_path)
        if stats is None:
            return
        current_parent_total = (
            stats["total_s"] if parent_total is None else parent_total
        )
        module = self._module_name(root_path, depth)
        rows.append(self._to_row(root_path, stats, current_parent_total, depth, module))
        prefix = root_path + "."
        children = [
            path
            for path in sections
            if path.startswith(prefix) and "." not in path[len(prefix) :]
        ]
        children.sort(key=lambda path: -sections[path]["total_s"])
        child_sum = 0.0
        self_total = stats["total_s"]
        for child in children:
            child_sum += sections[child]["total_s"]
            self._append_tree(
                rows,
                child,
                sections,
                parent_total=self_total,
                depth=depth + 1,
            )
        other = self_total - child_sum
        if children and other > 1e-6:
            calls = stats["calls"]
            other_stats = {
                "calls": calls,
                "mean_ms": (other * 1e3 / calls) if calls else 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
                "total_s": other,
            }
            rows.append(
                self._to_row("(other)", other_stats, self_total, depth + 1, module)
            )

    def _format_row(
        self,
        row: _ReportRow,
        section_width: int,
        color_code: int | None = None,
    ) -> str:
        line = (
            f"  {row.display_name:<{section_width}} "
            f"{row.calls:>8d} "
            f"{row.mean_ms:>10.3f} "
            f"{row.min_ms:>10.3f} "
            f"{row.max_ms:>10.3f} "
            f"{row.std_ms:>10.3f} "
            f"{row.total_s:>10.3f} "
            f"{row.pct_parent:>11.2f}%"
        )
        if color_code is None:
            return line
        return f"\033[38;5;{color_code}m{line}\033[0m"

    def _module_color_map(self, rows: list[_ReportRow]) -> dict[str, int]:
        if not self.cfg.color_output:
            return {}
        modules = sorted({row.module for row in rows if row.module is not None})
        return {
            module: _MODULE_COLOR_CODES[index % len(_MODULE_COLOR_CODES)]
            for index, module in enumerate(modules)
        }

    def _format_module_legend(self, module_colors: dict[str, int]) -> str:
        labels = [
            f"\033[38;5;{color_code}m{module}\033[0m"
            for module, color_code in module_colors.items()
        ]
        return "  module colors: " + " | ".join(labels)

    def _log_table(self, data: Dict[str, object]) -> None:
        sections = data["sections"]  # type: ignore[assignment]
        if not sections:
            logger.log_info(
                "[Profiler] no samples recorded "
                "(still in warmup, or no root section was called)."
            )
            return

        rows = self._build_tree_rows(sections)
        module_colors = self._module_color_map(rows)
        section_width = max(38, min(96, max(len(row.display_name) for row in rows) + 2))
        header = (
            f"  {'section':<{section_width}} {'calls':>8} {'mean(ms)':>10} "
            f"{'min(ms)':>10} {'max(ms)':>10} {'std(ms)':>10} "
            f"{'total(s)':>10} {'% of parent':>12}"
        )
        separator = "-" * len(header)
        lines = [
            "=" * 18 + " EmbodiChain Profiling Report " + "=" * 18,
            f"  device={self.device} | sync_cuda={self.cfg.sync_cuda} | "
            f"warmup={self.cfg.warmup_steps} | nvtx={self.cfg.nvtx}",
            self._format_module_legend(module_colors) if module_colors else "",
            "  % of parent = this row's total time / its direct parent's total time.",
            "  Root rows are 100%; (other) is time in the parent outside named children.",
            "",
            header,
            separator,
        ]

        for root in self._root_paths(sections):
            root_rows: list[_ReportRow] = []
            self._append_tree(root_rows, root, sections, depth=0)
            for row in root_rows:
                lines.append(
                    self._format_row(row, section_width, module_colors.get(row.module))
                )
            lines.append("")

        lines.append("=" * len(header))
        logger.log_info("\n".join(lines))
