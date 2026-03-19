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

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import deep_update, load_algorithm_spec, load_suite_spec, load_task_spec
from .leaderboard import build_leaderboard
from .metrics import (
    aggregate_runs,
    compute_final_metric_stable,
    compute_steps_to_threshold_first_hit,
    compute_steps_to_threshold_sustained,
)
from .plots import build_plot_artifacts
from .reporting import generate_leaderboard_markdown, generate_markdown_report
from .runtime import dump_json, evaluate_checkpoint, train_with_config


class BenchmarkRunner:
    """Coordinate benchmark training, evaluation, aggregation, and reporting."""

    def __init__(
        self,
        tasks: list[str] | None = None,
        algorithms: list[str] | None = None,
        seeds: list[int] | None = None,
        suite: str = "default",
        output_root: str | Path = "benchmark/reports",
        overrides: dict[str, Any] | None = None,
    ) -> None:
        suite_spec = load_suite_spec(suite)
        self.tasks = tasks or list(suite_spec["tasks"])
        self.algorithms = algorithms or list(suite_spec["algorithms"])
        self.seeds = seeds or list(suite_spec["seeds"])
        self.protocol = deepcopy(suite_spec.get("protocol", {}))
        if overrides:
            self.protocol = deep_update(self.protocol, overrides)
        self.output_root = Path(output_root)

    def build_run_config(
        self,
        task_name: str,
        algorithm_name: str,
        seed: int,
    ) -> dict[str, Any]:
        task_spec = load_task_spec(task_name)
        algorithm_spec = load_algorithm_spec(algorithm_name)

        cfg = deep_update(task_spec["base_config"], algorithm_spec["config"])
        cfg["trainer"]["exp_name"] = f"{task_name}_{algorithm_name}_seed{seed}"
        cfg["trainer"]["seed"] = seed
        cfg["trainer"]["enable_eval"] = True
        cfg["trainer"]["eval_freq"] = int(self.protocol["evaluation_interval"])
        cfg["trainer"]["num_eval_episodes"] = int(self.protocol["evaluation_episodes"])
        cfg["trainer"]["iterations"] = int(self.protocol["iterations"])
        cfg["trainer"]["buffer_size"] = int(self.protocol["buffer_size"])
        cfg["trainer"]["num_envs"] = int(self.protocol["num_envs"])
        cfg["trainer"]["num_eval_envs"] = int(self.protocol["num_eval_envs"])
        cfg["trainer"]["device"] = str(self.protocol["device"])
        cfg["trainer"]["headless"] = bool(self.protocol["headless"])
        cfg["trainer"]["save_freq"] = int(self.protocol["save_interval"])
        cfg["trainer"]["use_wandb"] = False
        return cfg

    def _iter_jobs(self) -> list[tuple[str, str, int]]:
        jobs = []
        for task_name in self.tasks:
            for algorithm_name in self.algorithms:
                for seed in self.seeds:
                    jobs.append((task_name, algorithm_name, seed))
        return jobs

    def run_training(self) -> list[dict[str, Any]]:
        """Run benchmark training and store per-run training artifacts."""
        training_runs: list[dict[str, Any]] = []
        for task_name, algorithm_name, seed in self._iter_jobs():
            task_spec = load_task_spec(task_name)
            run_dir = (
                self.output_root / "runs" / task_name / algorithm_name / f"seed_{seed}"
            )
            run_config = self.build_run_config(task_name, algorithm_name, seed)
            dump_json(run_config, run_dir / "run_config.json")
            train_summary = train_with_config(run_config, run_dir)
            training_record = {
                "task": task_name,
                "env_id": task_spec["env_id"],
                "algorithm": algorithm_name,
                "seed": seed,
                "train_steps": int(train_summary["global_step"]),
                "training_fps": train_summary["training_fps"],
                "peak_gpu_memory_mb": train_summary["peak_gpu_memory_mb"],
                "checkpoint_path": train_summary["checkpoint_path"],
                "output_dir": train_summary["output_dir"],
                "eval_history": train_summary.get("eval_history", []),
                "train_history": train_summary.get("train_history", []),
            }
            dump_json(training_record, run_dir / "train_result.json")
            training_runs.append(training_record)
        return training_runs

    def run_evaluation(
        self, training_runs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Evaluate trained checkpoints and write final per-run benchmark results."""
        results: list[dict[str, Any]] = []
        for training_record in training_runs:
            task_name = training_record["task"]
            algorithm_name = training_record["algorithm"]
            seed = training_record["seed"]
            task_spec = load_task_spec(task_name)
            run_dir = Path(training_record["output_dir"])
            run_config = self.build_run_config(task_name, algorithm_name, seed)
            dump_json(run_config, run_dir / "run_config.json")
            eval_summary = evaluate_checkpoint(
                cfg_json=run_config,
                checkpoint_path=training_record["checkpoint_path"],
                num_episodes=int(self.protocol["evaluation_episodes"]),
                num_envs=int(self.protocol["num_eval_envs"]),
            )
            result = {
                "task": task_name,
                "env_id": task_spec["env_id"],
                "algorithm": algorithm_name,
                "seed": seed,
                "train_steps": training_record["train_steps"],
                "final_reward": eval_summary["avg_reward"],
                "final_success_rate": eval_summary["success_rate"],
                "final_episode_length": eval_summary["avg_episode_length"],
                "training_fps": training_record["training_fps"],
                "environment_fps": eval_summary["environment_fps"],
                "peak_gpu_memory_mb": training_record["peak_gpu_memory_mb"],
                "checkpoint_path": training_record["checkpoint_path"],
                "output_dir": training_record["output_dir"],
                "eval_history": training_record.get("eval_history", []),
                "train_history": training_record.get("train_history", []),
            }
            threshold = task_spec.get("success_threshold", 0.8)
            sustain_count = int(self.protocol.get("threshold_sustain_count", 3))
            stable_eval_window = int(self.protocol.get("final_eval_window", 3))
            result["final_success_rate_stable"] = compute_final_metric_stable(
                training_record.get("eval_history", []),
                metric_key="eval/success_rate",
                window_size=stable_eval_window,
            )
            result["steps_to_success_threshold_first_hit"] = (
                compute_steps_to_threshold_first_hit(
                    training_record.get("eval_history", []),
                    metric_key="eval/success_rate",
                    threshold=float(threshold),
                )
            )
            result["steps_to_success_threshold"] = compute_steps_to_threshold_sustained(
                training_record.get("eval_history", []),
                metric_key="eval/success_rate",
                threshold=float(threshold),
                sustain_count=sustain_count,
            )
            result["final_metrics"] = eval_summary["metrics"]
            dump_json(result, run_dir / "result.json")
            results.append(result)
        return results

    def aggregate_results(
        self, run_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Aggregate multiple seeds into task-algorithm summaries."""
        return aggregate_runs(run_results)

    def update_leaderboard(
        self,
        aggregate_result: list[dict[str, Any]],
        run_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build and persist leaderboard artifacts."""
        leaderboard = build_leaderboard(aggregate_result, run_results=run_results)
        leaderboard_dir = self.output_root / "leaderboard"
        dump_json({"leaderboard": leaderboard}, leaderboard_dir / "leaderboard.json")
        generate_leaderboard_markdown(
            leaderboard=leaderboard,
            output_path=leaderboard_dir / "leaderboard.md",
        )
        return leaderboard

    def generate_report(
        self,
        run_results: list[dict[str, Any]],
        aggregate_result: list[dict[str, Any]],
        leaderboard: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Create a markdown benchmark report and result json files."""
        leaderboard = leaderboard or self.update_leaderboard(
            aggregate_result, run_results
        )
        plot_artifacts = build_plot_artifacts(
            run_results=run_results,
            leaderboard=leaderboard,
            output_dir=self.output_root / "plots",
        )
        dump_json({"runs": run_results}, self.output_root / "benchmark_runs.json")
        dump_json(
            {"aggregate": aggregate_result},
            self.output_root / "benchmark_summary.json",
        )
        return generate_markdown_report(
            run_results=run_results,
            aggregate_results=aggregate_result,
            leaderboard=leaderboard,
            plot_artifacts=plot_artifacts,
            protocol=self.protocol,
            output_path=self.output_root / "benchmark_report.md",
        )


__all__ = ["BenchmarkRunner"]
