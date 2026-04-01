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

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import deep_update, load_algorithm_spec, load_suite_spec, load_task_spec
from .metrics import (
    aggregate_runs,
    build_leaderboard,
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
        self.suite = suite
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

    def _run_dir(self, task_name: str, algorithm_name: str, seed: int) -> Path:
        return self.output_root / "runs" / task_name / algorithm_name / f"seed_{seed}"

    @staticmethod
    def _job_key(task_name: str, algorithm_name: str, seed: int) -> tuple[str, str, int]:
        return (task_name, algorithm_name, int(seed))

    @staticmethod
    def _load_json_artifact(path: str | Path) -> dict[str, Any] | None:
        artifact_path = Path(path)
        if not artifact_path.exists():
            return None
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected JSON object at {artifact_path}, got {type(data)!r}."
            )
        return data

    @staticmethod
    def _record_matches_job(
        record: dict[str, Any],
        task_name: str,
        algorithm_name: str,
        seed: int,
    ) -> bool:
        return (
            record.get("task") == task_name
            and record.get("algorithm") == algorithm_name
            and int(record.get("seed", -1)) == int(seed)
        )

    @staticmethod
    def _protocol_from_run_config(run_config: dict[str, Any]) -> dict[str, Any]:
        trainer = run_config.get("trainer", {})
        return {
            "device": trainer.get("device"),
            "headless": trainer.get("headless"),
            "iterations": trainer.get("iterations"),
            "buffer_size": trainer.get("buffer_size"),
            "num_envs": trainer.get("num_envs"),
            "num_eval_envs": trainer.get("num_eval_envs"),
            "evaluation_interval": trainer.get("eval_freq"),
            "evaluation_episodes": trainer.get("num_eval_episodes"),
        }

    def _expected_protocol_for_job(
        self,
        task_name: str,
        algorithm_name: str,
        seed: int,
    ) -> dict[str, Any]:
        return self._protocol_from_run_config(
            self.build_run_config(task_name, algorithm_name, seed)
        )

    def _artifact_is_compatible(
        self,
        artifact: dict[str, Any],
        task_name: str,
        algorithm_name: str,
        seed: int,
        run_dir: Path,
    ) -> bool:
        artifact_protocol = artifact.get("protocol")
        if isinstance(artifact_protocol, dict):
            return artifact_protocol == self.protocol
        run_config = self._load_json_artifact(run_dir / "run_config.json")
        if run_config is None:
            return False
        return self._protocol_from_run_config(run_config) == self._expected_protocol_for_job(
            task_name, algorithm_name, seed
        )

    def _load_existing_training_record(
        self,
        task_name: str,
        algorithm_name: str,
        seed: int,
    ) -> dict[str, Any] | None:
        run_dir = self._run_dir(task_name, algorithm_name, seed)
        record = self._load_json_artifact(run_dir / "train_result.json")
        if record is None:
            return None
        if not self._record_matches_job(record, task_name, algorithm_name, seed):
            return None
        if not self._artifact_is_compatible(
            record, task_name, algorithm_name, seed, run_dir
        ):
            return None
        checkpoint_path = record.get("checkpoint_path")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return None
        return record

    def collect_existing_run_results(self) -> list[dict[str, Any]]:
        """Load compatible existing result artifacts for the requested jobs."""
        results: list[dict[str, Any]] = []
        for task_name, algorithm_name, seed in self._iter_jobs():
            run_dir = self._run_dir(task_name, algorithm_name, seed)
            record = self._load_json_artifact(run_dir / "result.json")
            if record is None:
                continue
            if not self._record_matches_job(record, task_name, algorithm_name, seed):
                continue
            if not self._artifact_is_compatible(
                record, task_name, algorithm_name, seed, run_dir
            ):
                continue
            results.append(record)
        return results

    def merge_run_results(
        self,
        *result_sets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge multiple run result lists, preferring later duplicates."""
        merged: dict[tuple[str, str, int], dict[str, Any]] = {}
        for result_set in result_sets:
            for record in result_set:
                key = self._job_key(
                    str(record["task"]),
                    str(record["algorithm"]),
                    int(record["seed"]),
                )
                merged[key] = record
        return [
            merged[key]
            for key in sorted(merged.keys(), key=lambda item: (item[0], item[1], item[2]))
        ]

    def run_training(self, skip_existing: bool = False) -> list[dict[str, Any]]:
        """Run benchmark training and store per-run training artifacts."""
        training_runs: list[dict[str, Any]] = []
        existing_result_keys = set()
        if skip_existing:
            existing_result_keys = {
                self._job_key(item["task"], item["algorithm"], item["seed"])
                for item in self.collect_existing_run_results()
            }
        for task_name, algorithm_name, seed in self._iter_jobs():
            run_dir = self._run_dir(task_name, algorithm_name, seed)
            if skip_existing and self._job_key(
                task_name, algorithm_name, seed
            ) in existing_result_keys:
                continue
            if skip_existing:
                existing_training = self._load_existing_training_record(
                    task_name, algorithm_name, seed
                )
                if existing_training is not None:
                    training_runs.append(existing_training)
                    continue

            task_spec = load_task_spec(task_name)
            run_config = self.build_run_config(task_name, algorithm_name, seed)
            dump_json(run_config, run_dir / "run_config.json")
            train_summary = train_with_config(run_config, run_dir)
            training_record = {
                "task": task_name,
                "env_id": task_spec["env_id"],
                "algorithm": algorithm_name,
                "seed": seed,
                "suite": self.suite,
                "protocol": deepcopy(self.protocol),
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
                "suite": self.suite,
                "protocol": deepcopy(self.protocol),
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
        dump_json(
            {
                "suite": self.suite,
                "tasks": self.tasks,
                "algorithms": self.algorithms,
                "seeds": self.seeds,
                "protocol": self.protocol,
            },
            self.output_root / "benchmark_protocol.json",
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
