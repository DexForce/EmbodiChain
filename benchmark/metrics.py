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


def compute_steps_to_threshold(
    eval_history: list[dict[str, float]],
    metric_key: str,
    threshold: float,
) -> int | None:
    """Return the first step where `metric_key` reaches `threshold`."""
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
        if metric_value >= threshold:
            return int(step_value)
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
        step_key = "steps_to_success_threshold"
        steps = [
            int(run[step_key])
            for run in runs
            if isinstance(run.get(step_key), int)
        ]
        if steps:
            summary[f"{step_key}_mean"] = mean(steps)
            summary[f"{step_key}_std"] = pstdev(steps) if len(steps) > 1 else 0.0
        summaries.append(summary)

    return summaries


__all__ = ["aggregate_runs", "compute_steps_to_threshold"]
