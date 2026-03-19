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

from benchmark.reporting import generate_markdown_report


def test_generate_markdown_report_writes_expected_sections(tmp_path):
    run_results = [
        {
            "task": "cart_pole",
            "algorithm": "ppo",
            "seed": 0,
            "final_reward": 1.5,
            "final_success_rate": 0.8,
            "steps_to_success_threshold": 256,
            "checkpoint_path": "outputs/checkpoint.pt",
        }
    ]
    aggregate_results = [
        {
            "task": "cart_pole",
            "algorithm": "ppo",
            "num_runs": 1,
            "final_reward_mean": 1.5,
            "final_success_rate_mean": 0.8,
            "training_fps_mean": 100.0,
            "environment_fps_mean": 500.0,
        }
    ]
    leaderboard = [
        {
            "rank": 1,
            "algorithm": "ppo",
            "score": 0.8,
            "steps_to_success_threshold": 256.0,
            "success_rate_std": 0.1,
            "avg_success_rate": 0.8,
            "avg_final_reward": 1.5,
            "tasks_covered": 1,
        }
    ]
    plot_artifacts = {"cart_pole_success_rate": str(tmp_path / "plot.svg")}
    (tmp_path / "plot.svg").write_text("<svg></svg>", encoding="utf-8")

    output_path = tmp_path / "benchmark_report.md"
    generate_markdown_report(
        run_results,
        aggregate_results,
        leaderboard,
        plot_artifacts,
        {"device": "cpu", "iterations": 10},
        output_path,
    )

    report = output_path.read_text(encoding="utf-8")
    assert "RL Benchmark Report" in report
    assert "Benchmark Overview" in report
    assert "Leaderboard" in report
    assert "Plots" in report
    assert "Stability Analysis" in report
    assert "System Performance" in report
    assert "Aggregate Results" in report
    assert "Per-Run Results" in report
    assert "cart_pole" in report
