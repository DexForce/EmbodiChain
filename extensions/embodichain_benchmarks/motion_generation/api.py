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
from contextlib import nullcontext
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .config import PlannerSpecCfg, SuiteCfg, load_suite

if TYPE_CHECKING:
    from .planners.base import PlannerAdapter
    from .runner import BenchmarkRunResult
    from .video import VideoRecordCfg

__all__ = ["prepare_case_manifest", "run_benchmark"]


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
    workers: int = 1,
    worker_threads: int = 1,
    quality_batch_size: int = 1,
    quality_devices: Sequence[str] | None = None,
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
        workers: Isolated processes for quality cases. More than one uses spawn,
            freezes all inputs before candidate execution, and merges raw trials.
            Planner RNG remains backend-owned: input pairing is exact but
            stochastic solver outputs need not match serial execution.
            Performance replay requires one worker. Concurrent timing is retained
            in raw records but excluded from report timing and memory aggregates.
        worker_threads: Torch and BLAS thread limit in each parallel worker.
            Serial execution retains the caller's thread settings.
        quality_batch_size: Execution slots for frozen B=1 Free-space NMG-only
            quality replay. Original cases and standalone validation grids are
            retained. Shared batch timings are not per-case planner latency.
            Atomic, traditional baselines and performance replay require one.
        quality_devices: Optional unique explicit ``cuda:N`` devices, indexed
            within ``CUDA_VISIBLE_DEVICES``. Quality shards use these devices
            round-robin; ``workers`` remains the total process count. One worker
            uses the first device. Parallel input preparation retains ``device``.
            Performance replay does not accept this option.

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
    if type(quality_batch_size) is not int or quality_batch_size < 1:
        raise ValueError("quality_batch_size must be a positive integer.")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer.")
    if type(worker_threads) is not int or worker_threads < 1:
        raise ValueError("worker_threads must be a positive integer.")
    if quality_devices is not None:
        import torch

        from .parallel import _resolve_quality_devices

        quality_devices = _resolve_quality_devices(quality_devices)
        if case_manifest is not None and any(
            row.get("case_parameters", {}).get("measurement_purpose")
            == "performance_replay"
            for row in json.loads(Path(case_manifest).read_text()).get("cases", [])
        ):
            raise ValueError("Performance replay does not accept quality_devices.")
    if workers > 1:
        from .parallel import run_parallel_benchmark

        return run_parallel_benchmark(
            suite,
            planners=planners,
            adapters=adapters,
            device=device,
            headless=headless,
            output_root=output_root,
            video=video,
            case_manifest=case_manifest,
            workers=workers,
            worker_threads=worker_threads,
            quality_batch_size=quality_batch_size,
            quality_devices=quality_devices,
        )
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

    from .registry import planner_adapter_names
    from .runner import BenchmarkRunner

    known_adapters = set(planner_adapter_names()) | set(adapters or {})
    for spec in selected:
        if spec.adapter not in known_adapters:
            raise ValueError(
                f"Unknown planner adapter {spec.adapter!r}; "
                f"available adapters: {sorted(known_adapters)}."
            )

    # Native planners may resolve CUDA from the process's current device.
    # Restore the host device after serial quality execution, including failures.
    device_scope = (
        torch.cuda.device(quality_devices[0])
        if quality_devices is not None
        else nullcontext()
    )
    with device_scope:
        runner = BenchmarkRunner(
            resolved,
            selected,
            adapters=adapters,
            device=quality_devices[0] if quality_devices is not None else device,
            headless=headless,
            output_root=output_root,
            video=video,
            case_manifest=case_manifest,
            quality_batch_size=quality_batch_size,
            **(
                {"execution": {"quality_devices": quality_devices}}
                if quality_devices
                else {}
            ),
        )
        if planners is not None:
            runner.suite.planners = runner.planner_specs
            for spec in runner.planner_specs:
                spec.enabled = True
        runner.suite.validate_benchmark()
        return runner.run()


def prepare_case_manifest(
    suite: str | Path | SuiteCfg,
    output: str | Path,
    *,
    device: str = "auto",
    headless: bool = True,
) -> Path:
    """Freeze suite inputs independently of every candidate planner.

    Args:
        suite: Installed suite name, YAML path or assembled configuration.
        output: New case manifest path; existing files are never replaced.
        device: Simulation device used for case generation and reachability.
        headless: Whether to construct simulation without a viewer.

    Returns:
        The complete hashed corpus, ready for serial or parallel replay.
    """
    if isinstance(suite, (str, Path)):
        resolved = load_suite(suite)
    elif isinstance(suite, SuiteCfg):
        resolved = suite
    else:
        raise TypeError("suite must be a suite name, YAML path, or SuiteCfg.")
    resolved.validate_benchmark()
    from .runner import BenchmarkRunner

    return BenchmarkRunner(
        resolved, [], device=device, headless=headless
    ).prepare_case_manifest(output)
