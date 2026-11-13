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
"""Refine a prepared recording against an existing registered scene.

Run ``python -m examples.real2sim.refine SOURCE REGISTERED OUTPUT
--instruction '...' --clearance 0.004``. Generation services are not needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from embodichain.gen_sim.real2sim.execution import refine_prepared_recording

__all__ = ["main"]


def main() -> None:
    """Place the base, refine source joints and package the physical replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("registered", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--clearance", type=float, default=0.004)
    parser.add_argument("--grasp-seeds", type=Path)
    parser.add_argument(
        "--grasp-observations",
        type=Path,
        help="JSON events with side, grasp and observed world object_deltas",
    )
    args = parser.parse_args()
    result = refine_prepared_recording(
        args.source,
        args.registered,
        args.output,
        instruction=args.instruction,
        clearance=args.clearance,
        grasp_seeds=args.grasp_seeds,
        grasp_observations=args.grasp_observations,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
