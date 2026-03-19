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
from benchmark.runtime import dump_json


def parse_args() -> argparse.Namespace:
    """Parse CLI args for running benchmark training only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--algorithm", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--suite", default="default", type=str)
    parser.add_argument("--output-root", default="benchmark/reports", type=str)
    return parser.parse_args()


def main() -> None:
    """Run a single benchmark training job and save the training result JSON."""
    args = parse_args()
    runner = BenchmarkRunner(
        tasks=[args.task],
        algorithms=[args.algorithm],
        seeds=[args.seed],
        suite=args.suite,
        output_root=args.output_root,
    )
    results = runner.run_training()
    output_path = Path(args.output_root) / "single_train_result.json"
    dump_json({"training_runs": results}, output_path)
    print(f"Training result written to: {output_path}")


if __name__ == "__main__":
    main()
