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

from __future__ import annotations

from collections import defaultdict
from math import isnan
from statistics import mean, pstdev
from typing import Any


def _iter_valid_threshold_points(
    eval_history: list[dict[str, float]],
    metric_key: str,
):
    """Yield `(step, metric)` pairs with valid numeric values."""
    for item in eval_history:
        metric_value = item.get(metric_key)
        step_value = item.get("global_step")
        if metric_value is None or step_value is None:
            continue
        if not isinstance(metric_value, (int, float)) or not isinstance(
            step_value, (int, float)
        ):
            continue
        if isnan(metric_value):
            continue
        yield int(step_value), float(metric_value)


def compute_final_metric_stable(
    eval_history: list[dict[str, float]],
    metric_key: str,
    window_size: int = 3,
) -> float | None:
    """Return the mean of the last `window_size` valid metric values."""
    valid_values = [
        metric_value
        for _, metric_value in _iter_valid_threshold_points(eval_history, metric_key)
    ]
    if not valid_values:
        return None
    effective_window = max(1, window_size)
    return mean(valid_values[-effective_window:])


def compute_steps_to_threshold_first_hit(
    eval_history: list[dict[str, float]],
    metric_key: str,
    threshold: float,
) -> int | None:
    """Return the first step where `metric_key` reaches `threshold`."""
    for step_value, metric_value in _iter_valid_threshold_points(
        eval_history, metric_key
    ):
        if metric_value >= threshold:
            return step_value
    return None


def compute_steps_to_threshold_sustained(
    eval_history: list[dict[str, float]],
    metric_key: str,
    threshold: float,
    sustain_count: int = 3,
) -> int | None:
    """Return the first step where the threshold is met for `sustain_count` evals."""
    if sustain_count <= 1:
        return compute_steps_to_threshold_first_hit(eval_history, metric_key, threshold)

    consecutive_hits = 0
    first_step_in_window: int | None = None
    for step_value, metric_value in _iter_valid_threshold_points(eval_history, metric_key):
        if metric_value >= threshold:
            consecutive_hits += 1
            if first_step_in_window is None:
                first_step_in_window = step_value
            if consecutive_hits >= sustain_count:
                return first_step_in_window
        else:
            consecutive_hits = 0
            first_step_in_window = None
    return None


def aggregate_runs(run_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate run results by task and algorithm."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in run_results:
        grouped[(result["task"], result["algorithm"])].append(result)

    summaries: list[dict[str, Any]] = []
    for (task, algorithm), runs in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "task": task,
            "algorithm": algorithm,
            "num_runs": len(runs),
        }
        scalar_keys = {
            "final_reward",
            "final_success_rate",
            "final_success_rate_stable",
            "final_episode_length",
            "training_fps",
            "environment_fps",
            "peak_gpu_memory_mb",
        }
        for key in scalar_keys:
            values = [
                float(run[key])
                for run in runs
                if isinstance(run.get(key), (int, float)) and not isnan(run[key])
            ]
            if values:
                summary[f"{key}_mean"] = mean(values)
                summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        step_keys = {
            "steps_to_success_threshold",
            "steps_to_success_threshold_first_hit",
        }
        for step_key in step_keys:
            steps = [int(run[step_key]) for run in runs if isinstance(run.get(step_key), int)]
            if steps:
                summary[f"{step_key}_mean"] = mean(steps)
                summary[f"{step_key}_std"] = pstdev(steps) if len(steps) > 1 else 0.0
        summaries.append(summary)

    return summaries


def _valid_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isnan(float(value)):
        return float(value)
    return None


def build_leaderboard(
    aggregate_results: list[dict[str, Any]],
    run_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build leaderboard entries from aggregated benchmark summaries."""
    grouped_summary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in aggregate_results:
        grouped_summary[item["algorithm"]].append(item)

    grouped_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in run_results or []:
        grouped_runs[item["algorithm"]].append(item)

    leaderboard: list[dict[str, Any]] = []
    for algorithm, items in grouped_summary.items():
        stable_success_values = [
            float(item["final_success_rate_stable_mean"])
            for item in items
            if isinstance(item.get("final_success_rate_stable_mean"), (int, float))
            and not isnan(item["final_success_rate_stable_mean"])
        ]
        success_values = [
            float(item["final_success_rate_mean"])
            for item in items
            if isinstance(item.get("final_success_rate_mean"), (int, float))
            and not isnan(item["final_success_rate_mean"])
        ]
        reward_values = [
            float(item["final_reward_mean"])
            for item in items
            if isinstance(item.get("final_reward_mean"), (int, float))
            and not isnan(item["final_reward_mean"])
        ]
        score = mean(stable_success_values) if stable_success_values else float("nan")
        steps_values = [
            float(item["steps_to_success_threshold_mean"])
            for item in items
            if isinstance(item.get("steps_to_success_threshold_mean"), (int, float))
            and not isnan(item["steps_to_success_threshold_mean"])
        ]
        run_success_values = [
            float(run["final_success_rate"])
            for run in grouped_runs.get(algorithm, [])
            if _valid_float(run.get("final_success_rate")) is not None
        ]
        task_scores = {
            item["task"]: float(item["final_success_rate_stable_mean"])
            for item in items
            if _valid_float(item.get("final_success_rate_stable_mean")) is not None
        }
        raw_task_scores = {
            item["task"]: float(item["final_success_rate_mean"])
            for item in items
            if _valid_float(item.get("final_success_rate_mean")) is not None
        }
        leaderboard.append(
            {
                "algorithm": algorithm,
                "score": score,
                "steps_to_success_threshold": mean(steps_values)
                if steps_values
                else float("nan"),
                "success_rate_std": pstdev(run_success_values)
                if len(run_success_values) > 1
                else 0.0,
                "avg_success_rate": mean(success_values)
                if success_values
                else float("nan"),
                "avg_success_rate_stable": score,
                "avg_final_reward": mean(reward_values)
                if reward_values
                else float("nan"),
                "tasks_covered": len(items),
                "tasks": task_scores,
                "tasks_raw": raw_task_scores,
            }
        )

    leaderboard.sort(
        key=lambda item: (
            -(item["score"])
            if isinstance(item["score"], float) and not isnan(item["score"])
            else float("inf"),
            item["algorithm"],
        )
    )
    for index, item in enumerate(leaderboard, start=1):
        item["rank"] = index
    return leaderboard


__all__ = [
    "aggregate_runs",
    "build_leaderboard",
    "compute_final_metric_stable",
    "compute_steps_to_threshold_first_hit",
    "compute_steps_to_threshold_sustained",
]
