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

from benchmark.leaderboard import build_leaderboard


def test_build_leaderboard_ranks_higher_success_first():
    aggregate_results = [
        {
            "algorithm": "ppo",
            "task": "cart_pole",
            "final_success_rate_mean": 0.8,
            "final_reward_mean": 10.0,
            "steps_to_success_threshold_mean": 100.0,
        },
        {
            "algorithm": "ppo",
            "task": "push_cube",
            "final_success_rate_mean": 0.6,
            "final_reward_mean": 5.0,
            "steps_to_success_threshold_mean": 200.0,
        },
        {
            "algorithm": "grpo",
            "task": "cart_pole",
            "final_success_rate_mean": 0.7,
            "final_reward_mean": 8.0,
            "steps_to_success_threshold_mean": 150.0,
        },
        {
            "algorithm": "grpo",
            "task": "push_cube",
            "final_success_rate_mean": 0.5,
            "final_reward_mean": 4.0,
            "steps_to_success_threshold_mean": 250.0,
        },
    ]
    run_results = [
        {"algorithm": "ppo", "final_success_rate": 0.8},
        {"algorithm": "ppo", "final_success_rate": 0.6},
        {"algorithm": "grpo", "final_success_rate": 0.7},
        {"algorithm": "grpo", "final_success_rate": 0.5},
    ]

    leaderboard = build_leaderboard(aggregate_results, run_results=run_results)

    assert leaderboard[0]["algorithm"] == "ppo"
    assert leaderboard[0]["rank"] == 1
    assert "steps_to_success_threshold" in leaderboard[0]
    assert "success_rate_std" in leaderboard[0]
    assert "tasks" in leaderboard[0]
    assert leaderboard[1]["algorithm"] == "grpo"
