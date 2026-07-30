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

"""Lightweight, toggleable time profiler for env reset/step.

Wrap any phase of :meth:`BaseEnv.step` / :meth:`BaseEnv.reset` with::

    with self._profiler.section("sim_update"):
        ...

The profiler builds a hierarchical section name from the active call stack
(Isaac Lab ``Timer`` semantics): a parent section's wall time includes its
children. When disabled (``cfg is None`` or ``enable_time`` is False) the
context manager is effectively a no-op, so the instrumentation can stay in
place at zero overhead.

.. note::
    GPU-memory profiling was temporarily removed (entry parameter, recording,
    and report output). Only wall-time statistics are collected for now.

See :class:`EnvProfilerCfg` for the available toggles.
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

__all__ = ["EnvProfilerCfg", "EnvProfiler"]

# Top-level section names that delimit a profiled step / reset loop. Sections
# recorded outside one of these roots (e.g. the init-time ``get_obs`` call) are
# silently skipped so they do not pollute the report.
_ROOT_NAMES = frozenset({"step", "reset"})


@configclass
class EnvProfilerCfg:
    """Configuration for env reset/step time profiling."""

    enable_time: bool = True
    """Enable per-section wall-time statistics (mean/min/max/std)."""

    sync_cuda: bool = False
    """Call ``torch.cuda.synchronize()`` at section boundaries for accurate GPU
    wall time. Disabled by default to keep profiling low-overhead; enable when
    absolute (not just relative) GPU timings are needed."""

    warmup_steps: int = 5
    """Number of top-level (step/reset) sections to discard before recording,
    so JIT/cuDNN autotune setup does not skew the averages."""

    nvtx: bool = False
    """Also push NVTX ranges around each section so they show up named in an
    Nsight Systems timeline. Near-zero cost when not running under nsys."""

    output_path: str | None = None
    """If set, dump the final report as JSON to this path on :meth:`report`."""


@dataclass
class _SectionStats:
    """Running statistics for one named section."""

    n: int = 0
    total_s: float = 0.0
    sq_s: float = 0.0
    min_s: float = math.inf
    max_s: float = 0.0


class EnvProfiler:
    """Time profiler for env reset/step.

    Args:
        cfg: Profiler configuration. ``None`` disables profiling entirely.
        device: The device the env runs on (used for CUDA sync / NVTX).
    """

    def __init__(self, cfg: Optional[EnvProfilerCfg], device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self._is_cuda = device.type == "cuda"
        self._stats: Dict[str, _SectionStats] = {}
        self._stack: list[str] = []
        self._warmup = cfg.warmup_steps if cfg is not None else 0
        self._on = cfg is not None and cfg.enable_time
        self._do_time = self._on
        self._do_nvtx = self._on and cfg.nvtx and self._is_cuda

    # -- public API ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether profiling is active."""
        return self._on

    @contextmanager
    def section(self, name: str, *, is_root: bool = False) -> Iterator[None]:
        """Record wall time for a named section.

        Args:
            name: Leaf section name (e.g. ``"sim_update"``). The full
                hierarchical name is derived from the active section stack.
            is_root: True for top-level ``step`` / ``reset`` sections. A root
                section entered while another root is already active (e.g.
                ``reset()`` called during step's auto-reset) is a no-op so its
                children attribute to the outer root instead of opening a
                duplicate reset root.
        """
        if not self._on:
            yield
            return
        # Root called while already inside a root: don't open a duplicate.
        if is_root and self._stack:
            yield
            return

        if self._do_time and self.cfg.sync_cuda and self._is_cuda:
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        if self._do_nvtx:
            torch.cuda.nvtx.range_push(name)
        self._stack.append(name)
        try:
            yield
        finally:
            if self._do_time and self.cfg.sync_cuda and self._is_cuda:
                torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - t0
            full_name = ".".join(self._stack)
            self._stack.pop()
            if self._do_nvtx:
                torch.cuda.nvtx.range_pop()

            root = full_name.split(".", 1)[0]
            if root not in _ROOT_NAMES:
                return  # outside a profiled step/reset loop
            if self._warmup > 0:
                if is_root:
                    self._warmup -= 1
                return  # discard samples during warmup
            self._record(full_name, dt)

    def report(self) -> Dict[str, object]:
        """Log a profiling report table and optionally dump JSON.

        Returns:
            The report data dict (empty if profiling is disabled).
        """
        if not self._on:
            return {}
        data = self._build_report_data()
        self._log_table(data)
        if self.cfg.output_path:
            try:
                with open(self.cfg.output_path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.log_info(f"[Profiler] report dumped to {self.cfg.output_path}")
            except OSError as e:
                logger.log_warning(f"[Profiler] failed to dump JSON: {e}")
        return data

    # -- internals ----------------------------------------------------------

    def _record(self, name: str, dt: float) -> None:
        s = self._stats.get(name)
        if s is None:
            s = _SectionStats()
            self._stats[name] = s
        s.n += 1
        s.total_s += dt
        s.sq_s += dt * dt
        if dt < s.min_s:
            s.min_s = dt
        if dt > s.max_s:
            s.max_s = dt

    def _build_report_data(self) -> Dict[str, object]:
        sections: Dict[str, Dict[str, float]] = {}
        for name, s in self._stats.items():
            mean = s.total_s / s.n if s.n else 0.0
            var = max(0.0, s.sq_s / s.n - mean * mean) if s.n else 0.0
            sections[name] = {
                "calls": s.n,
                "mean_ms": mean * 1e3,
                "min_ms": (s.min_s if s.n else 0.0) * 1e3,
                "max_ms": (s.max_s if s.n else 0.0) * 1e3,
                "std_ms": math.sqrt(var) * 1e3,
                "total_s": s.total_s,
            }
        return {"sections": sections}

    def _emit_row(
        self,
        lines: list[str],
        leaf: str,
        s: Dict[str, float],
        parent_total: float,
        depth: int,
    ) -> None:
        pct = (100.0 * s["total_s"] / parent_total) if parent_total > 0 else 0.0
        pad = "  " * depth
        lines.append(
            f"  {pad}{leaf:<26} {s['calls']:>6} {s['mean_ms']:>10.3f} "
            f"{s['min_ms']:>8.3f} {s['max_ms']:>8.3f} {s['std_ms']:>7.3f} "
            f"{s['total_s']:>9.3f} {pct:>7.1f}%"
        )

    def _print_tree(
        self,
        lines: list[str],
        root_path: str,
        sections: Dict[str, Dict[str, float]],
        depth: int = 0,
    ) -> None:
        s = sections.get(root_path)
        if s is None:
            return
        parent_total = s["total_s"]
        self._emit_row(lines, root_path, s, parent_total, depth)
        prefix = root_path + "."
        children = [
            p for p in sections if p.startswith(prefix) and "." not in p[len(prefix) :]
        ]
        children.sort(key=lambda p: -sections[p]["total_s"])
        child_sum = 0.0
        for c in children:
            child_sum += sections[c]["total_s"]
            self._print_tree(lines, c, sections, depth + 1)
        other = parent_total - child_sum
        if children and other > 1e-6:
            calls = s["calls"]
            other_s = {
                "calls": calls,
                "mean_ms": (other * 1e3 / calls) if calls else 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "std_ms": 0.0,
                "total_s": other,
            }
            self._emit_row(lines, "(other)", other_s, parent_total, depth + 1)

    def _log_table(self, data: Dict[str, object]) -> None:
        sections = data["sections"]  # type: ignore[assignment]
        if not sections:
            logger.log_info(
                "[Profiler] no samples recorded "
                "(still in warmup, or step/reset never called)."
            )
            return
        header = (
            f"  {'section':<26} {'calls':>6} {'mean(ms)':>10} {'min':>8} "
            f"{'max':>8} {'std':>7} {'total(s)':>9} {'%par':>8}"
        )
        sep = "-" * len(header)
        lines = [
            "=" * 18 + " EmbodiChain Env Profiling Report " + "=" * 18,
            f"  device={self.device} | sync_cuda={self.cfg.sync_cuda} | "
            f"warmup={self.cfg.warmup_steps} | nvtx={self.cfg.nvtx}",
            "",
            header,
            sep,
        ]
        for root in ("step", "reset"):
            if root in sections:
                self._print_tree(lines, root, sections, depth=0)
                lines.append("")
        other_roots = sorted(
            p for p in sections if "." not in p and p not in ("step", "reset")
        )
        for root in other_roots:
            self._print_tree(lines, root, sections, depth=0)
            lines.append("")
        lines.append("=" * 70)
        logger.log_info("\n".join(lines))
