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

"""Analyze the sampled joint-space workspace of DexForce W1."""

from __future__ import annotations

import argparse

from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
    shutdown_sim,
)
from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
    WorkspaceAnalyzer,
)


def main() -> None:
    """Run joint-space workspace analysis."""
    parser = add_demo_args(
        argparse.ArgumentParser(description="Analyze the W1 joint workspace.")
    )
    parser.add_argument(
        "--num_samples",
        "--num-samples",
        type=int,
        default=30000,
        help="Number of joint configurations to sample.",
    )
    args = parser.parse_args()
    setup_print_options()

    sim = create_default_sim(args, add_default_light=False)
    try:
        sim.set_manual_update(False)
        robot = sim.add_robot(
            cfg=DexforceW1Cfg.from_dict(
                {
                    "uid": "dexforce_w1",
                    "version": "v021",
                    "arm_kind": "industrial",
                }
            )
        )
        maybe_open_window(sim, args)
        print("DexforceW1 robot added to the simulation.")

        results = WorkspaceAnalyzer(robot=robot, sim_manager=sim).analyze(
            num_samples=args.num_samples,
            visualize=not args.headless,
        )
        print("\nJoint Space Results:")
        print(f"  Valid points: {results['num_valid']} / {results['num_samples']}")
        print(f"  Analysis time: {results['analysis_time']:.2f}s")
        print(f"  Metrics: {results['metrics']}")
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
