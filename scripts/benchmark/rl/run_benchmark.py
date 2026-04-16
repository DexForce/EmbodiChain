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

import argparse

from .runner import BenchmarkRunner


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for full benchmark execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--suite", type=str, default="default")
    parser.add_argument(
        "--output-root", type=str, default="scripts/benchmark/rl/reports"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rebuild-report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Train, evaluate, aggregate, and report benchmark results."""
    args = parse_args()
    overrides = {
        key: value
        for key, value in {
            "device": args.device,
            "iterations": args.iterations,
            "buffer_size": args.buffer_size,
            "evaluation_interval": args.evaluation_interval,
            "evaluation_episodes": args.evaluation_episodes,
            "num_envs": args.num_envs,
            "num_eval_envs": args.num_eval_envs,
            "headless": args.headless if args.headless else None,
        }.items()
        if value is not None
    }
    runner = BenchmarkRunner(
        tasks=args.tasks,
        algorithms=args.algorithms,
        seeds=args.seeds,
        suite=args.suite,
        output_root=args.output_root,
        overrides=overrides,
    )

    if args.rebuild_report_only:
        run_results = runner.collect_existing_run_results()
        if not run_results:
            raise SystemExit(
                "No compatible existing benchmark results were found for the requested jobs."
            )
    else:
        existing_results = (
            runner.collect_existing_run_results() if args.skip_existing else []
        )
        training_runs = runner.run_training(skip_existing=args.skip_existing)
        new_results = runner.run_evaluation(training_runs)
        run_results = runner.merge_run_results(existing_results, new_results)

    aggregate_result = runner.aggregate_results(run_results)
    leaderboard = runner.update_leaderboard(aggregate_result, run_results)
    report_path = runner.generate_report(run_results, aggregate_result, leaderboard)
    print(f"Benchmark report written to: {report_path}")


if __name__ == "__main__":
    main()
