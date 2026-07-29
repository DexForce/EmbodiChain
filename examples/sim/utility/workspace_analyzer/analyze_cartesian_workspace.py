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

"""Analyze the sampled Cartesian workspace of DexForce W1."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    maybe_wait_for_user,
    setup_print_options,
    shutdown_sim,
)
from embodichain.lab.sim.utility.workspace_analyzer.configs.visualization_config import (
    VisualizationConfig,
)
from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)


def main() -> None:
    """Run Cartesian workspace analysis."""
    parser = add_demo_args(
        argparse.ArgumentParser(description="Analyze the W1 Cartesian workspace.")
    )
    parser.add_argument(
        "--num_samples",
        "--num-samples",
        type=int,
        default=50000,
        help="Number of Cartesian poses to sample.",
    )
    parser.set_defaults(device="cuda")
    args = parser.parse_args()
    setup_print_options()

    sim = create_default_sim(
        args,
        width=1080,
        height=1080,
        add_default_light=False,
    )
    try:
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

        left_qpos = torch.tensor(
            [0, -np.pi / 4, 0.0, -np.pi / 2, -np.pi / 4, 0.0, 0.0],
            dtype=torch.float32,
            device=robot.device,
        )
        robot.set_qpos(left_qpos, joint_ids=robot.get_joint_ids("left_arm"))
        robot.set_qpos(-left_qpos, joint_ids=robot.get_joint_ids("right_arm"))
        left_arm_pose = robot.compute_fk(
            qpos=left_qpos,
            name="left_arm",
            to_matrix=True,
        )
        sim.draw_marker(
            cfg=MarkerCfg(
                name="left_arm_pose_axis",
                marker_type="axis",
                axis_xpos=left_arm_pose,
                axis_size=0.005,
                axis_len=0.15,
                arena_index=0,
            )
        )

        results = WorkspaceAnalyzer(
            robot=robot,
            config=WorkspaceAnalyzerConfig(
                mode=AnalysisMode.CARTESIAN_SPACE,
                visualization=VisualizationConfig(
                    show_unreachable_points=False,
                    point_size=8.0,
                ),
                control_part_name="left_arm",
            ),
            sim_manager=sim,
        ).analyze(
            num_samples=args.num_samples,
            visualize=not args.headless,
        )
        print("\nCartesian Space Results:")
        print(
            f"  Reachable points: {results['num_reachable']} / {results['num_samples']}"
        )
        print(f"  Analysis time: {results['analysis_time']:.2f}s")
        print(f"  Metrics: {results['metrics']}")
        maybe_wait_for_user(args, "Press Enter to exit...")
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
