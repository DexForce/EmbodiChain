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

from benchmark.metrics import aggregate_runs, compute_steps_to_threshold


def test_compute_steps_to_threshold_returns_first_matching_step():
    eval_history = [
        {"global_step": 128.0, "eval/success_rate": 0.2},
        {"global_step": 256.0, "eval/success_rate": 0.75},
        {"global_step": 384.0, "eval/success_rate": 0.81},
    ]

    assert compute_steps_to_threshold(eval_history, "eval/success_rate", 0.8) == 384


def test_aggregate_runs_groups_by_task_and_algorithm():
    run_results = [
        {
            "task": "cart_pole",
            "algorithm": "ppo",
            "seed": 0,
            "final_reward": 1.0,
            "final_success_rate": 0.9,
            "final_episode_length": 50.0,
            "training_fps": 100.0,
            "environment_fps": 500.0,
            "peak_gpu_memory_mb": 0.0,
            "steps_to_success_threshold": 1000,
        },
        {
            "task": "cart_pole",
            "algorithm": "ppo",
            "seed": 1,
            "final_reward": 3.0,
            "final_success_rate": 0.7,
            "final_episode_length": 40.0,
            "training_fps": 200.0,
            "environment_fps": 700.0,
            "peak_gpu_memory_mb": 0.0,
            "steps_to_success_threshold": 2000,
        },
    ]

    summaries = aggregate_runs(run_results)

    assert len(summaries) == 1
    assert summaries[0]["task"] == "cart_pole"
    assert summaries[0]["algorithm"] == "ppo"
    assert summaries[0]["final_reward_mean"] == 2.0
    assert summaries[0]["steps_to_success_threshold_mean"] == 1500
