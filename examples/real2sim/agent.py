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
"""Run semantic review, numerical residual correction and native physics replay.

python -m examples.real2sim.agent SOURCE REGISTERED OUTPUT --instruction '...'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodichain.gen_sim.real2sim.execution import run_prepared_real2sim

__all__ = ["main"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("registered", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Correction budget after initial replay; stop at the first successful execution",
    )
    parser.add_argument("--visual-urdf", type=Path)
    parser.add_argument(
        "--place-base",
        action="store_true",
        help="Recompute mounting clearance over the source arm motion; keep scene objects fixed",
    )
    parser.add_argument(
        "--continue-from",
        type=Path,
        help="Review a completed prior run first; execute only newly corrected trajectories",
    )
    parser.add_argument(
        "--initial-trajectory",
        type=Path,
        help="Start from an existing qpos NPZ; default is the original recording",
    )
    args = parser.parse_args()
    result = run_prepared_real2sim(
        args.source,
        args.registered,
        args.output,
        instruction=args.instruction,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_revisions=args.max_revisions,
        visual_urdf=args.visual_urdf,
        place_base=args.place_base,
        continue_from=args.continue_from,
        initial_trajectory=args.initial_trajectory,
    )
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
