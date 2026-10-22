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
    case_manifest: str | Path | None = None,
) -> BenchmarkRunResult:
    """Evaluate planners against a suite without modifying caller configuration.

    Args:
        suite: Installed suite name, YAML path, or assembled configuration.
            Embodiment component paths in YAML resolve relative to that file;
            component paths in an assembled configuration use the working
            directory. Adapter-specific paths follow the adapter's rules;
            relative NMG model paths use the working directory.
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
        case_manifest: Hashed case_manifest.json to replay instead of generating
            new inputs. Cases must match the suite's tracks, version and robot.
            Retain the host-written resolved_embodiment identity; replay rejects
            changed robot/embodiment configurations before creating simulation.
            This supports fixed quality populations and declared timing subsets.

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
        resolved = suite
    else:
        raise TypeError("suite must be a suite name, YAML path, or SuiteCfg.")

    selected = (
        list(planners)
        if planners is not None
        else [spec for spec in resolved.planners if spec.enabled]
    )
    if any(not isinstance(spec, PlannerSpecCfg) for spec in selected):
        raise TypeError("planners must contain PlannerSpecCfg instances.")
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

    runner = BenchmarkRunner(
        resolved,
        selected,
        adapters=adapters,
        device=device,
        headless=headless,
        output_root=output_root,
        video=video,
        case_manifest=case_manifest,
    )
    if planners is not None:
        runner.suite.planners = runner.planner_specs
        for spec in runner.planner_specs:
            spec.enabled = True
    runner.suite.validate_benchmark()
    return runner.run()
