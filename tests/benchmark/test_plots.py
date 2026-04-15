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

from benchmark.rl.plots import build_plot_artifacts


def test_build_plot_artifacts_writes_svg_files(tmp_path):
    run_results = [
        {
            "task": "cart_pole",
            "algorithm": "ppo",
            "eval_history": [
                {
                    "global_step": 100.0,
                    "eval/success_rate": 0.2,
                    "eval/avg_reward": 1.0,
                },
                {
                    "global_step": 200.0,
                    "eval/success_rate": 0.8,
                    "eval/avg_reward": 2.0,
                },
            ],
        },
        {
            "task": "cart_pole",
            "algorithm": "grpo",
            "eval_history": [
                {
                    "global_step": 100.0,
                    "eval/success_rate": 0.1,
                    "eval/avg_reward": 0.5,
                },
                {
                    "global_step": 200.0,
                    "eval/success_rate": 0.6,
                    "eval/avg_reward": 1.5,
                },
            ],
        },
    ]
    leaderboard = [
        {"algorithm": "ppo", "score": 0.8},
        {"algorithm": "grpo", "score": 0.6},
    ]

    artifacts = build_plot_artifacts(run_results, leaderboard, tmp_path)

    assert "cart_pole_success_rate" in artifacts
    assert "leaderboard_score" in artifacts
    for path in artifacts.values():
        assert path.endswith(".svg")
