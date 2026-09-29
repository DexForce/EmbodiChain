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

"""Run the simulator-free G-03 protocol fixture.

Run: ``python -m scripts.benchmark expert-generation --fixture``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scripts.benchmark.core.contracts import Budget

from .contracts import GenerationCase
from .fixtures import fixture_executor
from .report import rebuild_generation_report
from .runner import run_generation

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Run or rebuild the deterministic G-03 fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", action="store_true", help="run the offline fixture"
    )
    parser.add_argument(
        "--report-only", type=Path, help="rebuild a saved fixture report"
    )
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/expert-generation")
    )
    args = parser.parse_args(argv)
    if args.report_only is not None:
        print(rebuild_generation_report(args.report_only.resolve()))
        return
    if not args.fixture:
        parser.error("the first implementation only supports --fixture")
    if args.attempts < 1:
        parser.error("--attempts must be positive")
    case = GenerationCase(
        experiment_id="g03-fixture",
        case_id="fixture-case-0",
        source_kind="atomic_action",
        source_id="fixture_action",
        source_revision="fixture:v1",
        scene_case_id="fixture-scene-0",
        initial_state_id="fixture-initial-0",
        seed=0,
    )
    result = run_generation(
        (case,),
        attempts_per_case=args.attempts,
        execute=fixture_executor,
        budget=Budget(max_runs=1, max_attempts=args.attempts),
        output_root=args.output,
    )
    print(result.root / "report.md")
