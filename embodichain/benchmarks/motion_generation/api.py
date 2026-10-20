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

"""Programmatic evaluation of installed suites and external planner adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from .config import PlannerSpecCfg, SuiteCfg, load_suite

if TYPE_CHECKING:
    from .planners.base import PlannerAdapter
    from .runner import BenchmarkRunResult
    from .video import VideoRecordCfg

__all__ = ["run_benchmark"]


def run_benchmark(
    suite: str | Path | SuiteCfg = "motion_panda_free_space_smoke",
    *,
    planners: Sequence[PlannerSpecCfg] | None = None,
    adapters: Mapping[str, type[PlannerAdapter]] | None = None,
    device: str = "auto",
    headless: bool = True,
    output_root: str | Path = "outputs/benchmarks",
    video: VideoRecordCfg | None = None,
) -> BenchmarkRunResult:
    """Evaluate planners against a suite without modifying caller configuration.

    Args:
        suite: Installed suite name, YAML path, or assembled configuration.
            Paths inside a suite file resolve relative to that file; paths in
            an assembled configuration resolve relative to the working directory.
        planners: Algorithms to evaluate, replacing the suite's planner list.
            Every explicitly supplied planner is enabled. When omitted, use
            the enabled planners declared by the suite.
        adapters: PlannerAdapter subclasses indexed by adapter name. Entries
            override registered adapters for this call only. The runner owns
            construction, preparation, warmup, evaluation and cleanup.
        device: Simulation device: ``auto``, ``cpu``, or a Torch device string.
        headless: Whether to run without a viewer.
        output_root: Directory under which a unique run directory is created.
        video: Optional Atomic Skills replay recording configuration.

    Returns:
        Raw trial records, aggregate metrics and paths to persisted artifacts.

    Raises:
        ValueError: The suite is invalid or no planners are selected.
        TypeError: A suite, planner or adapter has the wrong type.

    .. note::
        Runs create and destroy their own simulator. Execute calls serially
        within one process. Backend and model dependencies must be installed
        separately. Atomic Skills adapters also expose a MotionGenerator.
    """
    if isinstance(suite, (str, Path)):
        resolved = load_suite(suite)
    elif isinstance(suite, SuiteCfg):
        resolved = deepcopy(suite)
    else:
        raise TypeError("suite must be a suite name, YAML path, or SuiteCfg.")

    if planners is not None:
        selected = deepcopy(list(planners))
        if any(not isinstance(spec, PlannerSpecCfg) for spec in selected):
            raise TypeError("planners must contain PlannerSpecCfg instances.")
        for spec in selected:
            spec.enabled = True
        resolved.planners = selected
    resolved.validate_benchmark()
    selected = [spec for spec in resolved.planners if spec.enabled]
    if not selected:
        raise ValueError("No algorithms were selected for the benchmark.")

    from .runner import BenchmarkRunner
    from .registry import planner_adapter_names

    known_adapters = set(planner_adapter_names()) | set(adapters or {})
    for spec in selected:
        if spec.adapter not in known_adapters:
            raise ValueError(
                f"Unknown planner adapter {spec.adapter!r}; "
                f"available adapters: {sorted(known_adapters)}."
            )

    return BenchmarkRunner(
        resolved,
        selected,
        adapters=adapters,
        device=device,
        headless=headless,
        output_root=output_root,
        video=video,
    ).run()
