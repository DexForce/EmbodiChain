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

"""Measure Expert Program demo success without retries.

The command line can either aggregate an existing raw artifact or construct one
real Gym environment from explicit Gym and Expert Program configurations. Live
runs execute every fixed seed once, discard every episode buffer, then reuse the
same raw JSON and three-table report pipeline as injected programmatic runs.

Run offline:
``python -m scripts.benchmark.expert_program.demo_success --raw-json RAW``

Run live:
``python -m scripts.benchmark.expert_program.demo_success --run-simulation
--gym_config GYM --expert-program PROGRAM --case-id CASE --seeds 0 1
--raw-json RAW``
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Any

import psutil
import torch
import gymnasium

from embodichain.lab.gym.envs.demo import (
    DEMO_SCHEMA_VERSION,
    DemoEpisodeResult,
    execute_demo_episode,
)
from embodichain.lab.gym.envs.expert_program import load_expert_program
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)

__all__ = [
    "DEMO_SUCCESS_SCHEMA_VERSION",
    "DemoSuccessAggregates",
    "DemoSuccessArtifacts",
    "DemoSuccessCase",
    "DemoSuccessRow",
    "DemoSuccessTrial",
    "MemorySnapshot",
    "aggregate_demo_success_trials",
    "capture_memory",
    "collect_demo_success_trials",
    "load_raw_trials",
    "main",
    "run_all_benchmarks",
    "run_demo_success_benchmark",
    "run_gym_demo_success_benchmark",
    "write_markdown_report",
    "write_raw_trials",
]

DEMO_SUCCESS_SCHEMA_VERSION = 1
_BENCHMARK_ID = "expert_program_demo_success"

_TIME_COLUMNS = (
    "case",
    "episodes",
    "attempted_rows",
    "cost_time_ms",
    "mean_episode_ms",
    "cpu_delta_mb",
    "gpu_delta_mb",
    "peak_gpu_mb",
)
_METRIC_COLUMNS = (
    "case",
    "attempted",
    "successes",
    "success_rate",
    "terminal_reasons",
    "segment_failures",
    "segment_failure_breakdown",
    "call_failures",
    "call_failure_breakdown",
    "length_mean",
    "length_min",
    "length_max",
)
_LEADERBOARD_COLUMNS = (
    "rank",
    "case",
    "attempted",
    "successes",
    "overall_success_rate",
    "length_mean",
    "mean_episode_ms",
)

EpisodeExecutor = Callable[..., DemoEpisodeResult]
EnvironmentProvider = Callable[["DemoSuccessCase"], Any]
MemorySampler = Callable[..., "MemorySnapshot"]
GymEnvironmentFactory = Callable[[argparse.Namespace, str | Path], Any]
EnvironmentCloser = Callable[[Any], None]


def _validate_nonempty_string(value: object, *, field_name: str) -> str:
    """Return one exact non-empty string without outer whitespace."""
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty without outer whitespace.")
    return value


def _snapshot_string_tuple(
    values: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Validate and snapshot one list-or-tuple of stable string labels."""
    if type(values) not in (list, tuple):
        raise TypeError(f"{field_name} must be a list or tuple.")
    snapshot = tuple(values)
    for index, value in enumerate(snapshot):
        _validate_nonempty_string(
            value,
            field_name=f"{field_name}[{index}]",
        )
    return snapshot


@dataclass(frozen=True, slots=True)
class DemoSuccessCase:
    """One named demo benchmark case and its fixed evaluation seeds.

    Args:
        case_id: Stable identity shown in raw artifacts and reports.
        seeds: Unique seeds, each executed exactly once in the given order.
    """

    case_id: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.case_id, field_name="case_id")
        if type(self.seeds) not in (list, tuple):
            raise TypeError("seeds must be a list or tuple.")
        owned_seeds = tuple(self.seeds)
        if not owned_seeds:
            raise ValueError("seeds must contain at least one fixed evaluation seed.")
        if any(type(seed) is not int for seed in owned_seeds):
            raise TypeError("Every evaluation seed must be an integer.")
        if len(set(owned_seeds)) != len(owned_seeds):
            raise ValueError("Evaluation seeds must be unique within a case.")
        object.__setattr__(self, "seeds", owned_seeds)


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Current process and PyTorch GPU memory in megabytes.

    Args:
        cpu_rss_mb: Current process resident memory.
        gpu_allocated_mb: Current PyTorch-allocated GPU memory.
        gpu_peak_allocated_mb: Peak PyTorch GPU allocation since the last reset.
    """

    cpu_rss_mb: float
    gpu_allocated_mb: float
    gpu_peak_allocated_mb: float


@dataclass(frozen=True, slots=True)
class DemoSuccessRow:
    """Normalized result for one vector-environment row.

    Args:
        env_index: Zero-based row index in the vector environment.
        success: Whether this row completed the episode successfully.
        terminal_reason: Stable terminal-reason label.
        length: Recorded row length in environment steps.
        segment_failure_reasons: Segment-name-qualified failure keys.
        call_failure_keys: Segment/call/status-qualified runtime failure keys.
    """

    env_index: int
    success: bool
    terminal_reason: str
    length: int
    segment_failure_reasons: tuple[str, ...] = ()
    call_failure_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.env_index) is not int:
            raise TypeError("env_index must be an integer.")
        if self.env_index < 0:
            raise ValueError("env_index must be non-negative.")
        if type(self.success) is not bool:
            raise TypeError("success must be a boolean.")
        _validate_nonempty_string(
            self.terminal_reason,
            field_name="terminal_reason",
        )
        if type(self.length) is not int:
            raise TypeError("length must be an integer.")
        if self.length < 0:
            raise ValueError("length must be non-negative.")
        object.__setattr__(
            self,
            "segment_failure_reasons",
            _snapshot_string_tuple(
                self.segment_failure_reasons,
                field_name="segment_failure_reasons",
            ),
        )
        object.__setattr__(
            self,
            "call_failure_keys",
            _snapshot_string_tuple(
                self.call_failure_keys,
                field_name="call_failure_keys",
            ),
        )


@dataclass(frozen=True, slots=True)
class DemoSuccessTrial:
    """Raw result for one no-retry seed execution.

    Args:
        case_id: Stable benchmark case identity.
        seed: Fixed seed executed exactly once.
        cost_time_ms: Executor wall-clock duration in milliseconds.
        cpu_delta_mb: Process RSS delta across execution.
        gpu_delta_mb: PyTorch GPU allocation delta across execution.
        peak_gpu_mb: Peak PyTorch GPU allocation during execution.
        rows: Normalized per-environment outcomes.
        episode_result: Owned JSON-compatible executor metadata.
    """

    case_id: str
    seed: int
    cost_time_ms: float
    cpu_delta_mb: float
    gpu_delta_mb: float
    peak_gpu_mb: float
    rows: tuple[DemoSuccessRow, ...]
    episode_result: dict[str, object]

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.case_id, field_name="case_id")
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer.")
        numeric_fields = {
            "cost_time_ms": self.cost_time_ms,
            "cpu_delta_mb": self.cpu_delta_mb,
            "gpu_delta_mb": self.gpu_delta_mb,
            "peak_gpu_mb": self.peak_gpu_mb,
        }
        normalized_numeric: dict[str, float] = {}
        for field_name, value in numeric_fields.items():
            if type(value) not in (int, float):
                raise TypeError(f"{field_name} must be a real number.")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{field_name} must be finite.")
            normalized_numeric[field_name] = normalized
        if (
            normalized_numeric["cost_time_ms"] < 0.0
            or normalized_numeric["peak_gpu_mb"] < 0.0
        ):
            raise ValueError("Elapsed time and peak GPU memory cannot be negative.")
        if type(self.rows) not in (list, tuple):
            raise TypeError("rows must be a list or tuple.")
        owned_rows = tuple(self.rows)
        if not owned_rows:
            raise ValueError("A demo success trial must contain at least one row.")
        if not all(type(row) is DemoSuccessRow for row in owned_rows):
            raise TypeError("rows must contain exactly DemoSuccessRow values.")
        env_indices = tuple(row.env_index for row in owned_rows)
        if env_indices != tuple(range(len(owned_rows))):
            raise ValueError(
                "rows must have unique contiguous env_index values starting at zero."
            )
        if type(self.episode_result) is not dict:
            raise TypeError("episode_result must be a dictionary.")
        owned_result = deepcopy(self.episode_result)
        json.dumps(owned_result, allow_nan=False)
        for field_name, value in normalized_numeric.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "rows", owned_rows)
        object.__setattr__(self, "episode_result", owned_result)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible raw trial mapping.

        Returns:
            An independently owned raw trial mapping.
        """
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "cost_time_ms": self.cost_time_ms,
            "cpu_delta_mb": self.cpu_delta_mb,
            "gpu_delta_mb": self.gpu_delta_mb,
            "peak_gpu_mb": self.peak_gpu_mb,
            "rows": [asdict(row) for row in self.rows],
            "episode_result": deepcopy(self.episode_result),
        }


@dataclass(frozen=True, slots=True)
class DemoSuccessAggregates:
    """The three stable row sets rendered into the Markdown report.

    Args:
        time_and_memory: Per-case timing and memory summaries.
        success_and_metrics: Per-case success and diagnostic summaries.
        leaderboard: All cases ranked by success rate.
    """

    time_and_memory: tuple[dict[str, object], ...]
    success_and_metrics: tuple[dict[str, object], ...]
    leaderboard: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class DemoSuccessArtifacts:
    """Paths and in-memory results produced by one benchmark run.

    Args:
        raw_json_path: Written lossless raw artifact.
        report_path: Written three-table Markdown report.
        trials: In-memory no-retry trials.
        aggregates: In-memory report rows.
    """

    raw_json_path: Path
    report_path: Path
    trials: tuple[DemoSuccessTrial, ...]
    aggregates: DemoSuccessAggregates


def capture_memory(*, reset_gpu_peak: bool = False) -> MemorySnapshot:
    """Capture CPU RSS and PyTorch GPU allocation.

    Args:
        reset_gpu_peak: Reset the PyTorch peak-memory counter before sampling.

    Returns:
        Current CPU, GPU, and peak GPU memory in megabytes.
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available and reset_gpu_peak:
        torch.cuda.reset_peak_memory_stats()
    cpu_rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    gpu_allocated_mb = (
        torch.cuda.memory_allocated() / 1024**2 if cuda_available else 0.0
    )
    gpu_peak_allocated_mb = (
        torch.cuda.max_memory_allocated() / 1024**2 if cuda_available else 0.0
    )
    return MemorySnapshot(
        cpu_rss_mb=cpu_rss_mb,
        gpu_allocated_mb=gpu_allocated_mb,
        gpu_peak_allocated_mb=gpu_peak_allocated_mb,
    )


def _vector_or_default(
    values: tuple[Any, ...],
    *,
    row_count: int,
    default: Any,
    field_name: str,
) -> tuple[Any, ...]:
    """Return a validated per-row tuple or broadcast its scalar fallback."""
    if not values:
        return tuple(default for _ in range(row_count))
    if len(values) != row_count:
        raise ValueError(
            f"DemoEpisodeResult.{field_name} has {len(values)} rows; "
            f"expected {row_count}."
        )
    return values


def _normalize_episode_rows(result: DemoEpisodeResult) -> tuple[DemoSuccessRow, ...]:
    """Project a batched episode result into independent benchmark rows."""
    row_count = len(result.success)
    if row_count == 0:
        raise ValueError("DemoEpisodeResult.success must contain at least one row.")
    lengths = _vector_or_default(
        result.lengths,
        row_count=row_count,
        default=result.length,
        field_name="lengths",
    )
    terminal_reasons = _vector_or_default(
        result.terminal_reasons,
        row_count=row_count,
        default=result.terminal_reason,
        field_name="terminal_reasons",
    )
    failures: list[list[str]] = [[] for _ in range(row_count)]
    call_failures: list[list[str]] = [[] for _ in range(row_count)]
    for segment in result.segments:
        active = _vector_or_default(
            segment.active,
            row_count=row_count,
            default=True,
            field_name="segments.active",
        )
        successes = _vector_or_default(
            segment.successes,
            row_count=row_count,
            default=segment.success,
            field_name="segments.successes",
        )
        reasons = _vector_or_default(
            segment.failure_reasons,
            row_count=row_count,
            default=segment.failure_reason,
            field_name="segments.failure_reasons",
        )
        for env_index in range(row_count):
            if not active[env_index]:
                continue
            reason = reasons[env_index]
            if reason is not None:
                failures[env_index].append(f"{segment.name}:{reason}")
            elif not successes[env_index]:
                failures[env_index].append(f"{segment.name}:segment_failed")
        runtime = segment.metadata.get("runtime")
        if isinstance(runtime, Mapping):
            _append_runtime_call_failures(
                runtime,
                segment_name=segment.name,
                row_failures=call_failures,
            )

    return tuple(
        DemoSuccessRow(
            env_index=env_index,
            success=bool(result.success[env_index]),
            terminal_reason=str(terminal_reasons[env_index]),
            length=int(lengths[env_index]),
            segment_failure_reasons=tuple(failures[env_index]),
            call_failure_keys=tuple(call_failures[env_index]),
        )
        for env_index in range(row_count)
    )


def _append_runtime_call_failures(
    runtime: Mapping[str, object],
    *,
    segment_name: str,
    row_failures: list[list[str]],
    branch_id: str | None = None,
) -> None:
    """Attribute canonical runtime call failures to their environment rows."""
    env_ids = runtime.get("env_ids")
    calls = runtime.get("calls")
    if isinstance(env_ids, list) and isinstance(calls, list):
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            semantic_id = call.get("semantic_id")
            status = call.get("status")
            masks = call.get("masks")
            failed = masks.get("failed") if isinstance(masks, Mapping) else None
            if (
                not isinstance(semantic_id, str)
                or not isinstance(status, str)
                or not isinstance(failed, list)
                or len(failed) != len(env_ids)
            ):
                continue
            identity = (
                f"{segment_name}:{semantic_id}:{status}"
                if branch_id is None
                else f"{segment_name}:{branch_id}:{semantic_id}:{status}"
            )
            for env_id, is_failed in zip(env_ids, failed):
                if (
                    type(env_id) is int
                    and type(is_failed) is bool
                    and is_failed
                    and 0 <= env_id < len(row_failures)
                ):
                    row_failures[env_id].append(identity)

    branches = runtime.get("branches")
    if isinstance(branches, Mapping):
        branch_ids = sorted(key for key in branches if isinstance(key, str))
        for child_branch_id in branch_ids:
            branch_runtime = branches[child_branch_id]
            if isinstance(branch_runtime, Mapping):
                _append_runtime_call_failures(
                    branch_runtime,
                    segment_name=segment_name,
                    row_failures=row_failures,
                    branch_id=child_branch_id,
                )


def _executor_error_trial_rows(env: Any, reason: str) -> tuple[DemoSuccessRow, ...]:
    """Return zero-length failed rows for one executor exception."""
    configured_rows = getattr(env, "num_envs", 1)
    row_count = (
        configured_rows if type(configured_rows) is int and configured_rows > 0 else 1
    )
    return tuple(
        DemoSuccessRow(
            env_index=env_index,
            success=False,
            terminal_reason=reason,
            length=0,
        )
        for env_index in range(row_count)
    )


def _executor_error_metadata(
    *,
    episode_index: int,
    reason: str,
    error: Exception,
    row_count: int,
) -> dict[str, object]:
    """Return raw episode-shaped metadata that preserves one executor error."""
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "episode_index": episode_index,
        "length": 0,
        "completed": False,
        "success": [False] * row_count,
        "terminated": [False] * row_count,
        "truncated": [False] * row_count,
        "terminal_reason": reason,
        "segments": [],
        "lengths": [0] * row_count,
        "completed_by_env": [False] * row_count,
        "terminal_reasons": [reason] * row_count,
        "executor_error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def collect_demo_success_trials(
    cases: Sequence[DemoSuccessCase],
    env_provider: EnvironmentProvider,
    *,
    episode_executor: EpisodeExecutor = execute_demo_episode,
    clock: Callable[[], float] = time.perf_counter,
    memory_sampler: MemorySampler = capture_memory,
) -> tuple[DemoSuccessTrial, ...]:
    """Execute every fixed seed once and discard every resulting episode buffer.

    The caller owns environment construction and teardown. The harness performs
    one non-committing seeded reset, one executor call, and one mandatory
    non-committing discard reset for each seed. Executor exceptions become
    failed trials only after that discard succeeds.

    Args:
        cases: Named cases with fixed, unique seed sequences.
        env_provider: Required environment injection. It is called once per case.
        episode_executor: Demo executor, injectable for pure unit tests.
        clock: High-resolution monotonic timer.
        memory_sampler: CPU/GPU memory sampler.

    Returns:
        Raw per-seed trials in case and seed order.

    Raises:
        ValueError: If cases are empty, case IDs are duplicated, or an episode
            result is malformed.
        TypeError: If ``cases`` contains non-``DemoSuccessCase`` values.
    """
    try:
        case_values = tuple(cases)
    except TypeError as error:
        raise TypeError(
            "cases must be an iterable of DemoSuccessCase values."
        ) from error
    if not case_values:
        raise ValueError("cases must contain at least one benchmark case.")
    if not all(type(case) is DemoSuccessCase for case in case_values):
        raise TypeError("cases must contain exactly DemoSuccessCase values.")
    case_ids = [case.case_id for case in case_values]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Demo success benchmark case IDs must be unique.")

    trials: list[DemoSuccessTrial] = []
    episode_index = 0
    for case in case_values:
        env = env_provider(case)
        for seed in case.seeds:
            env.reset(seed=seed, options={"save_data": False})
            executor_error: Exception | None = None
            body_error: BaseException | None = None
            try:
                before = memory_sampler(reset_gpu_peak=True)
                start = clock()
                result: DemoEpisodeResult | None = None
                try:
                    result = episode_executor(env, episode_index=episode_index)
                except Exception as error:
                    executor_error = error
                elapsed_ms = (clock() - start) * 1000.0
                after = memory_sampler(reset_gpu_peak=False)
            except BaseException as error:
                body_error = error
                if executor_error is not None:
                    body_error.add_note(
                        "Episode executor also failed before benchmark measurement "
                        f"completed: {type(executor_error).__name__}: "
                        f"{executor_error}"
                    )
                raise
            finally:
                try:
                    env.reset(options={"save_data": False})
                except BaseException as discard_error:
                    discard_note = (
                        "Episode discard also failed: "
                        f"{type(discard_error).__name__}: {discard_error}"
                    )
                    if body_error is not None:
                        body_error.add_note(discard_note)
                    elif executor_error is not None:
                        executor_error.add_note(discard_note)
                        raise executor_error
                    else:
                        raise

            if executor_error is None:
                if result is None:
                    raise RuntimeError("The demo episode executor returned no result.")
                rows = _normalize_episode_rows(result)
                episode_result = result.to_metadata()
            else:
                reason = f"executor_error:{type(executor_error).__name__}"
                rows = _executor_error_trial_rows(env, reason)
                episode_result = _executor_error_metadata(
                    episode_index=episode_index,
                    reason=reason,
                    error=executor_error,
                    row_count=len(rows),
                )
            trials.append(
                DemoSuccessTrial(
                    case_id=case.case_id,
                    seed=seed,
                    cost_time_ms=elapsed_ms,
                    cpu_delta_mb=after.cpu_rss_mb - before.cpu_rss_mb,
                    gpu_delta_mb=after.gpu_allocated_mb - before.gpu_allocated_mb,
                    peak_gpu_mb=after.gpu_peak_allocated_mb,
                    rows=rows,
                    episode_result=episode_result,
                )
            )
            episode_index += 1
    return tuple(trials)


def _counter_json(counter: Counter[str]) -> str:
    """Render a deterministic compact JSON counter for one Markdown cell."""
    ordered = dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _validate_unique_trials(
    trials: Sequence[DemoSuccessTrial],
) -> tuple[DemoSuccessTrial, ...]:
    """Snapshot non-empty exact trials and reject duplicate identities."""
    try:
        trial_values = tuple(trials)
    except TypeError as error:
        raise TypeError(
            "trials must be an iterable of DemoSuccessTrial values."
        ) from error
    if not trial_values:
        raise ValueError("trials must contain at least one demo success trial.")
    if not all(type(trial) is DemoSuccessTrial for trial in trial_values):
        raise TypeError("trials must contain exactly DemoSuccessTrial values.")
    seen: set[tuple[str, int]] = set()
    for trial in trial_values:
        identity = (trial.case_id, trial.seed)
        if identity in seen:
            raise ValueError(
                "Duplicate demo success trial for "
                f"case_id={trial.case_id!r}, seed={trial.seed}."
            )
        seen.add(identity)
    return trial_values


def aggregate_demo_success_trials(
    trials: Sequence[DemoSuccessTrial],
) -> DemoSuccessAggregates:
    """Aggregate raw trials by case and rank every represented case.

    Args:
        trials: Unique case-and-seed trials.

    Returns:
        Stable rows for the three report tables.

    Raises:
        ValueError: If trials are empty or a case-and-seed identity occurs more
            than once.
        TypeError: If ``trials`` contains non-``DemoSuccessTrial`` values.
    """
    trial_values = _validate_unique_trials(trials)
    grouped: dict[str, list[DemoSuccessTrial]] = defaultdict(list)
    for trial in trial_values:
        grouped[trial.case_id].append(trial)

    time_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for case_id in sorted(grouped):
        case_trials = grouped[case_id]
        rows = [row for trial in case_trials for row in trial.rows]
        attempted = len(rows)
        successes = sum(row.success for row in rows)
        lengths = [row.length for row in rows]
        terminal_reasons = Counter(row.terminal_reason for row in rows)
        segment_reasons = Counter(
            reason for row in rows for reason in row.segment_failure_reasons
        )
        call_failure_keys = Counter(
            key for row in rows for key in row.call_failure_keys
        )
        time_rows.append(
            {
                "case": case_id,
                "episodes": len(case_trials),
                "attempted_rows": attempted,
                "cost_time_ms": sum(trial.cost_time_ms for trial in case_trials),
                "mean_episode_ms": mean(trial.cost_time_ms for trial in case_trials),
                "cpu_delta_mb": mean(trial.cpu_delta_mb for trial in case_trials),
                "gpu_delta_mb": mean(trial.gpu_delta_mb for trial in case_trials),
                "peak_gpu_mb": max(trial.peak_gpu_mb for trial in case_trials),
            }
        )
        metric_rows.append(
            {
                "case": case_id,
                "attempted": attempted,
                "successes": successes,
                "success_rate": successes / attempted,
                "terminal_reasons": _counter_json(terminal_reasons),
                "segment_failures": sum(segment_reasons.values()),
                "segment_failure_breakdown": _counter_json(segment_reasons),
                "call_failures": sum(call_failure_keys.values()),
                "call_failure_breakdown": _counter_json(call_failure_keys),
                "length_mean": mean(lengths),
                "length_min": min(lengths),
                "length_max": max(lengths),
            }
        )

    time_by_case = {str(row["case"]): row for row in time_rows}
    ranked_metrics = sorted(
        metric_rows,
        key=lambda row: (-float(row["success_rate"]), str(row["case"])),
    )
    leaderboard = tuple(
        {
            "rank": rank,
            "case": row["case"],
            "attempted": row["attempted"],
            "successes": row["successes"],
            "overall_success_rate": row["success_rate"],
            "length_mean": row["length_mean"],
            "mean_episode_ms": time_by_case[str(row["case"])]["mean_episode_ms"],
        }
        for rank, row in enumerate(ranked_metrics, start=1)
    )
    return DemoSuccessAggregates(
        time_and_memory=tuple(time_rows),
        success_and_metrics=tuple(metric_rows),
        leaderboard=leaderboard,
    )


def write_raw_trials(path: str | Path, trials: Sequence[DemoSuccessTrial]) -> Path:
    """Write lossless per-seed and per-row results to one raw JSON artifact.

    Args:
        path: Destination JSON path.
        trials: Unique case-and-seed trials.

    Returns:
        Written artifact path.

    Raises:
        ValueError: If trials are empty or a case-and-seed identity occurs more
            than once.
        TypeError: If ``trials`` contains non-``DemoSuccessTrial`` values.
    """
    trial_values = _validate_unique_trials(trials)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": DEMO_SUCCESS_SCHEMA_VERSION,
        "benchmark": _BENCHMARK_ID,
        "trials": [trial.to_dict() for trial in trial_values],
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Validate one raw JSON mapping boundary."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


def _load_row(value: object, field_name: str) -> DemoSuccessRow:
    """Decode one normalized row from a raw JSON trial."""
    data = _require_mapping(value, field_name)
    failures = data.get("segment_failure_reasons")
    if not isinstance(failures, list) or not all(
        isinstance(reason, str) for reason in failures
    ):
        raise ValueError(f"{field_name}.segment_failure_reasons must be a string list.")
    call_failures = data.get("call_failure_keys")
    if not isinstance(call_failures, list) or not all(
        isinstance(key, str) for key in call_failures
    ):
        raise ValueError(f"{field_name}.call_failure_keys must be a string list.")
    env_index = data.get("env_index")
    success = data.get("success")
    terminal_reason = data.get("terminal_reason")
    length = data.get("length")
    if type(env_index) is not int or env_index < 0:
        raise ValueError(f"{field_name}.env_index must be a non-negative integer.")
    if type(success) is not bool:
        raise ValueError(f"{field_name}.success must be a boolean.")
    if not isinstance(terminal_reason, str):
        raise ValueError(f"{field_name}.terminal_reason must be a string.")
    if type(length) is not int or length < 0:
        raise ValueError(f"{field_name}.length must be a non-negative integer.")
    return DemoSuccessRow(
        env_index=env_index,
        success=success,
        terminal_reason=terminal_reason,
        length=length,
        segment_failure_reasons=tuple(failures),
        call_failure_keys=tuple(call_failures),
    )


def _required_number(data: Mapping[str, object], key: str, field_name: str) -> float:
    """Read one finite raw numeric field without accepting booleans."""
    value = data.get(key)
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{field_name}.{key} must be a finite number.")
    return float(value)


def _load_trial(value: object, index: int) -> DemoSuccessTrial:
    """Decode one validated trial from a raw JSON artifact."""
    field_name = f"trials[{index}]"
    data = _require_mapping(value, field_name)
    case_id = data.get("case_id")
    seed = data.get("seed")
    rows = data.get("rows")
    episode_result = data.get("episode_result")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"{field_name}.case_id must be a non-empty string.")
    if type(seed) is not int:
        raise ValueError(f"{field_name}.seed must be an integer.")
    if not isinstance(rows, list):
        raise ValueError(f"{field_name}.rows must be a list.")
    episode_mapping = _require_mapping(episode_result, f"{field_name}.episode_result")
    return DemoSuccessTrial(
        case_id=case_id,
        seed=seed,
        cost_time_ms=_required_number(data, "cost_time_ms", field_name),
        cpu_delta_mb=_required_number(data, "cpu_delta_mb", field_name),
        gpu_delta_mb=_required_number(data, "gpu_delta_mb", field_name),
        peak_gpu_mb=_required_number(data, "peak_gpu_mb", field_name),
        rows=tuple(
            _load_row(row, f"{field_name}.rows[{i}]") for i, row in enumerate(rows)
        ),
        episode_result=dict(episode_mapping),
    )


def load_raw_trials(path: str | Path) -> tuple[DemoSuccessTrial, ...]:
    """Load a raw artifact for deterministic offline re-aggregation.

    Args:
        path: Existing raw JSON artifact.

    Returns:
        Validated trials in artifact order.

    Raises:
        ValueError: If the artifact schema is invalid or contains no valid trial.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = _require_mapping(payload, "raw benchmark")
    if data.get("schema_version") != DEMO_SUCCESS_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported demo success raw schema version: "
            f"{data.get('schema_version')!r}."
        )
    if data.get("benchmark") != _BENCHMARK_ID:
        raise ValueError("Raw JSON is not an Expert Program demo success artifact.")
    raw_trials = data.get("trials")
    if not isinstance(raw_trials, list):
        raise ValueError("raw benchmark.trials must be a list.")
    trials = tuple(_load_trial(trial, index) for index, trial in enumerate(raw_trials))
    return _validate_unique_trials(trials)


def _format_value(column: str, value: object) -> str:
    """Format one Markdown value deterministically."""
    if isinstance(value, float):
        if column.endswith("rate"):
            return f"{value:.2%}"
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_table(
    rows: Sequence[Mapping[str, object]], columns: tuple[str, ...]
) -> list[str]:
    """Render one Markdown table with a stable schema."""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(_format_value(column, row.get(column)) for column in columns)
        + " |"
        for row in rows
    )
    return lines


def write_markdown_report(path: str | Path, aggregates: DemoSuccessAggregates) -> Path:
    """Write exactly one report containing exactly the required three tables.

    Args:
        path: Destination Markdown path.
        aggregates: Rows for timing, success metrics, and leaderboard tables.

    Returns:
        Written report path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Expert Program Demo Success Benchmark",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "Each fixed seed is executed once, no failed episode is retried, and all "
        "episode buffers are discarded without being committed.",
        "",
        "## Time & Memory",
        "",
    ]
    lines.extend(_format_table(aggregates.time_and_memory, _TIME_COLUMNS))
    lines.extend(["", "## Success & Other Metrics", ""])
    lines.extend(_format_table(aggregates.success_and_metrics, _METRIC_COLUMNS))
    lines.extend(["", "## Leaderboard", ""])
    lines.extend(_format_table(aggregates.leaderboard, _LEADERBOARD_COLUMNS))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def run_demo_success_benchmark(
    cases: Sequence[DemoSuccessCase],
    env_provider: EnvironmentProvider,
    *,
    raw_json_path: str | Path,
    report_path: str | Path,
    episode_executor: EpisodeExecutor = execute_demo_episode,
    clock: Callable[[], float] = time.perf_counter,
    memory_sampler: MemorySampler = capture_memory,
) -> DemoSuccessArtifacts:
    """Collect no-retry trials and write one raw JSON plus one Markdown report.

    Args:
        cases: Named cases with fixed, unique seed sequences.
        env_provider: Environment injection called once per case.
        raw_json_path: Destination for lossless trials.
        report_path: Destination for the three-table report.
        episode_executor: Demo executor, injectable for tests.
        clock: High-resolution monotonic timer.
        memory_sampler: CPU/GPU memory sampler.

    Returns:
        Written paths, raw trials, and aggregate rows.

    Raises:
        ValueError: If output paths collide or trial identities are invalid.
    """
    if Path(raw_json_path).resolve() == Path(report_path).resolve():
        raise ValueError("raw_json_path and report_path must be different files.")
    trials = collect_demo_success_trials(
        cases,
        env_provider,
        episode_executor=episode_executor,
        clock=clock,
        memory_sampler=memory_sampler,
    )
    aggregates = aggregate_demo_success_trials(trials)
    raw_path = write_raw_trials(raw_json_path, trials)
    markdown_path = write_markdown_report(report_path, aggregates)
    return DemoSuccessArtifacts(
        raw_json_path=raw_path,
        report_path=markdown_path,
        trials=trials,
        aggregates=aggregates,
    )


def _create_gym_demo_success_environment(
    launcher_args: argparse.Namespace,
    expert_program_path: str | Path,
) -> Any:
    """Create one configured Gym environment through the standard launcher APIs."""
    gym_config_path = getattr(launcher_args, "gym_config", "")
    if not gym_config_path:
        raise ValueError("launcher_args.gym_config must select a Gym config file.")
    if getattr(launcher_args, "action_config", None) is not None:
        raise ValueError(
            "--action_config is not supported by the Expert Program benchmark."
        )

    discover_task_packages()
    execute_init_hooks()
    env_cfg, gym_config, action_config = build_env_cfg_from_args(launcher_args)
    if action_config:
        raise RuntimeError(
            "The Expert Program benchmark environment builder produced an "
            "unexpected action configuration."
        )
    env_cfg.expert_program = load_expert_program(expert_program_path)
    return gymnasium.make(id=gym_config["id"], cfg=env_cfg)


def _flush_simulation_cleanup_queue() -> None:
    """Flush deferred simulation cleanup after live benchmark work."""
    from embodichain.lab.sim.sim_manager import SimulationManager

    SimulationManager.flush_cleanup_queue()


def _close_gym_demo_success_environment(env: Any) -> None:
    """Close one benchmark environment without terminating the host process."""
    target = getattr(env, "unwrapped", env)
    close = getattr(target, "close", None)
    if not callable(close):
        raise TypeError("Benchmark environment must expose close().")

    close_error: BaseException | None = None
    try:
        close(exit_process=False)
    except BaseException as error:
        close_error = error

    try:
        _flush_simulation_cleanup_queue()
    except BaseException as error:
        if close_error is None:
            raise
        close_error.add_note(
            "Simulation cleanup also failed: " f"{type(error).__name__}: {error}"
        )
    if close_error is not None:
        raise close_error


def run_gym_demo_success_benchmark(
    case: DemoSuccessCase,
    *,
    launcher_args: argparse.Namespace,
    expert_program_path: str | Path,
    raw_json_path: str | Path,
    report_path: str | Path,
    episode_executor: EpisodeExecutor = execute_demo_episode,
    clock: Callable[[], float] = time.perf_counter,
    memory_sampler: MemorySampler = capture_memory,
    environment_factory: GymEnvironmentFactory | None = None,
    environment_closer: EnvironmentCloser | None = None,
) -> DemoSuccessArtifacts:
    """Run one configured real-environment benchmark case and close it safely.

    One environment is constructed for the case and reused across its fixed
    seeds. The shared harness performs exactly one execution per seed between
    non-committing seeded and discard resets. Closing the environment is an
    additional abort barrier and never commits an episode.

    Args:
        case: Named case and unique fixed evaluation seeds.
        launcher_args: Standard environment-launcher arguments containing the
            Gym configuration path and simulation overrides.
        expert_program_path: Explicit Expert Program JSON/YAML configuration.
        raw_json_path: Destination for lossless per-seed results.
        report_path: Destination for the three-table Markdown report.
        episode_executor: Demo executor, injectable for pure tests.
        clock: High-resolution monotonic timer.
        memory_sampler: CPU/GPU memory sampler.
        environment_factory: Optional environment construction override.
        environment_closer: Optional deterministic close override.

    Returns:
        Written artifacts and the in-memory no-retry results.

    Raises:
        ValueError: If launcher inputs, output paths, or trials are invalid.
        RuntimeError: If environment construction, execution, or cleanup fails.
    """
    factory = environment_factory or _create_gym_demo_success_environment
    closer = environment_closer or _close_gym_demo_success_environment
    try:
        env = factory(launcher_args, expert_program_path)
    except BaseException as factory_error:
        try:
            _flush_simulation_cleanup_queue()
        except BaseException as cleanup_error:
            factory_error.add_note(
                "Benchmark environment construction cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )
        raise
    body_error: BaseException | None = None
    try:
        return run_all_benchmarks(
            (case,),
            lambda requested_case: env,
            raw_json_path=raw_json_path,
            report_path=report_path,
            episode_executor=episode_executor,
            clock=clock,
            memory_sampler=memory_sampler,
        )
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            closer(env)
        except BaseException as cleanup_error:
            if body_error is None:
                raise
            body_error.add_note(
                "Benchmark environment cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


def run_all_benchmarks(
    cases: Sequence[DemoSuccessCase],
    env_provider: EnvironmentProvider,
    *,
    raw_json_path: str | Path,
    report_path: str | Path,
    episode_executor: EpisodeExecutor = execute_demo_episode,
    clock: Callable[[], float] = time.perf_counter,
    memory_sampler: MemorySampler = capture_memory,
) -> DemoSuccessArtifacts:
    """Run the injected demo benchmark and print its two artifact paths.

    Args:
        cases: Named cases with fixed, unique seed sequences.
        env_provider: Environment injection called once per case.
        raw_json_path: Destination for lossless trials.
        report_path: Destination for the three-table report.
        episode_executor: Demo executor, injectable for tests.
        clock: High-resolution monotonic timer.
        memory_sampler: CPU/GPU memory sampler.

    Returns:
        Written paths, raw trials, and aggregate rows.
    """
    print("=" * 60)
    print("Expert Program Demo Success Benchmark")
    print("=" * 60)
    artifacts = run_demo_success_benchmark(
        cases,
        env_provider,
        raw_json_path=raw_json_path,
        report_path=report_path,
        episode_executor=episode_executor,
        clock=clock,
        memory_sampler=memory_sampler,
    )
    print(f"Raw JSON saved: {artifacts.raw_json_path}")
    print(f"Markdown report saved: {artifacts.report_path}")
    print("=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)
    return artifacts


def _build_parser() -> argparse.ArgumentParser:
    """Build the offline-aggregation and live-simulation command parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed-seed Expert Program benchmark or aggregate an "
            "existing raw JSON artifact."
        )
    )
    add_env_launcher_args_to_parser(parser, require_gym_config=False)
    parser.set_defaults(
        num_envs=None,
        renderer=None,
        viser_image_fps=None,
    )
    parser.add_argument(
        "--run-simulation",
        action="store_true",
        help="Create a Gym environment and collect raw fixed-seed trials.",
    )
    parser.add_argument(
        "--expert-program",
        type=Path,
        default=None,
        help="Expert Program JSON/YAML file used by --run-simulation.",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Stable benchmark case identity used by --run-simulation.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Unique fixed seeds, each executed exactly once in the given order.",
    )
    parser.add_argument(
        "--raw-json",
        type=Path,
        required=True,
        help=(
            "Raw JSON destination for --run-simulation, or an existing raw "
            "artifact in offline aggregation mode."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output Markdown path (default: RAW with a .md suffix).",
    )
    return parser


def _provided_option_strings(argv: Sequence[str]) -> frozenset[str]:
    """Return normalized long option names explicitly present in ``argv``."""
    return frozenset(token.split("=", 1)[0] for token in argv if token.startswith("--"))


def _validate_cli_mode(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    provided_options: frozenset[str],
) -> None:
    """Reject incomplete or mixed live/offline command-line inputs."""
    live_values = {
        "--gym_config": args.gym_config,
        "--expert-program": args.expert_program,
        "--case-id": args.case_id,
        "--seeds": args.seeds,
    }
    if args.run_simulation:
        missing = [name for name, value in live_values.items() if not value]
        if missing:
            parser.error("--run-simulation requires " + ", ".join(missing) + ".")
        if args.preview:
            parser.error("--preview is not supported by --run-simulation.")
        if args.action_config is not None:
            parser.error("--action_config is not supported by --run-simulation.")
        return

    offline_options = frozenset({"--raw-json", "--report"})
    mixed_options = sorted(provided_options - offline_options)
    if mixed_options:
        parser.error(
            "Offline aggregation accepts only --raw-json and --report; "
            "live environment options require --run-simulation: "
            + ", ".join(mixed_options)
            + "."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run live fixed-seed trials or aggregate existing raw benchmark data.

    Args:
        argv: Optional command-line arguments for embedding and tests.

    Returns:
        Zero after the report is written.
    """
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    _validate_cli_mode(
        parser,
        args,
        provided_options=_provided_option_strings(raw_argv),
    )
    report_path = args.report or args.raw_json.with_suffix(".md")
    if args.raw_json.resolve() == report_path.resolve():
        raise ValueError(
            "The Markdown report must not overwrite the raw JSON artifact."
        )
    if args.run_simulation:
        case = DemoSuccessCase(
            case_id=args.case_id,
            seeds=tuple(args.seeds),
        )
        run_gym_demo_success_benchmark(
            case,
            launcher_args=args,
            expert_program_path=args.expert_program,
            raw_json_path=args.raw_json,
            report_path=report_path,
        )
        return 0

    trials = load_raw_trials(args.raw_json)
    write_markdown_report(report_path, aggregate_demo_success_trials(trials))
    print(f"Markdown report saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
