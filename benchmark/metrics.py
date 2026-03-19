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


__all__ = [
    "aggregate_runs",
    "compute_final_metric_stable",
    "compute_steps_to_threshold_first_hit",
    "compute_steps_to_threshold_sustained",
]
