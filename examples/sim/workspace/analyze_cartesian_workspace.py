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

"""Analyze the DexForce W1 left-arm workspace by sampling Cartesian positions."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.workspace import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.workspace.configs import (
    SamplingConfig,
    SamplingStrategy,
    VisualizationConfig,
)
from embodichain.lab.visualization import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)


def _run_until_interrupted(sim: SimulationManager) -> None:
    """Keep the native or Viser visualization alive until Ctrl+C.

    Args:
        sim: Simulation manager owning the visualization.
    """
    print("Workspace visualization is ready. Press Ctrl+C to exit.")
    try:
        while True:
            sim.update(step=1)
    except KeyboardInterrupt:
        pass


def main() -> None:
    """Run the Cartesian-space workspace analysis example."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_viser_args_to_parser(parser)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50_000,
        help="Number of Cartesian positions to sample (default: 50000).",
    )
    parser.add_argument(
        "--sim-device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Simulation and workspace-computation device.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Exit after analysis instead of keeping the visualization open.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be greater than zero")

    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    config = SimulationManagerCfg(
        headless=False,
        sim_device=args.sim_device,
        width=1080,
        height=1080,
        visualization=visualization_cfg_from_args(args),
    )
    sim = SimulationManager(config)
    try:
        cfg = DexforceW1Cfg.from_dict(
            {
                "uid": "dexforce_w1",
                "version": "v021",
                "arm_kind": "industrial",
                "with_default_eef": False,
            }
        )
        robot = sim.add_robot(cfg=cfg)
        print("DexforceW1 robot added to the simulation.")

        left_qpos = torch.tensor(
            [0, -np.pi / 4, 0.0, -np.pi / 2, -np.pi / 4, 0.0, 0.0],
            dtype=torch.float32,
            device=robot.device,
        )
        robot.set_qpos(
            qpos=left_qpos,
            joint_ids=robot.get_joint_ids("left_arm"),
        )
        robot.set_qpos(
            qpos=-left_qpos,
            joint_ids=robot.get_joint_ids("right_arm"),
        )

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

        analyzer = WorkspaceAnalyzer(
            robot=robot,
            config=WorkspaceAnalyzerConfig(
                mode=AnalysisMode.CARTESIAN_SPACE,
                sampling=SamplingConfig(strategy=SamplingStrategy.RANDOM),
                visualization=VisualizationConfig(
                    show_unreachable_points=True,
                    point_size=8.0,
                ),
                control_part_name="left_arm",
            ),
            sim_manager=sim,
        )
        results = analyzer.analyze(num_samples=args.num_samples, visualize=True)
        print("\nCartesian Space Results:")
        print(
            f"  Reachable points: {results['num_reachable']} / "
            f"{results['num_samples']}"
        )
        print(f"  Analysis time: {results['analysis_time']:.2f}s")
        print(f"  Metrics: {results['metrics']}")

        sim.capture_visualization(force=True)
        if not args.no_wait:
            _run_until_interrupted(sim)
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
