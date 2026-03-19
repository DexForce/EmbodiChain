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


__all__ = ["build_leaderboard"]
