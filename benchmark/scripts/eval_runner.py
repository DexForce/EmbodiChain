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
from pathlib import Path

from benchmark.runner import BenchmarkRunner
from benchmark.runtime import dump_json, evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse CLI args for evaluating a benchmark checkpoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--algorithm", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--episodes", default=20, type=int)
    parser.add_argument("--suite", default="default", type=str)
    parser.add_argument("--output-root", default="benchmark/reports", type=str)
    return parser.parse_args()


def main() -> None:
    """Evaluate a single checkpoint using the benchmark protocol."""
    args = parse_args()
    runner = BenchmarkRunner(
        tasks=[args.task],
        algorithms=[args.algorithm],
        seeds=[args.seed],
        suite=args.suite,
        output_root=args.output_root,
    )
    cfg_json = runner.build_run_config(args.task, args.algorithm, args.seed)
    result = evaluate_checkpoint(
        cfg_json=cfg_json,
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        num_envs=runner.protocol["num_eval_envs"],
    )
    output_path = Path(args.output_root) / "single_eval_result.json"
    dump_json(result, output_path)
    print(f"Evaluation result written to: {output_path}")


if __name__ == "__main__":
    main()
