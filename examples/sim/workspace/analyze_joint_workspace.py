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

"""Analyze the DexForce W1 left-arm workspace by sampling joint positions."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.workspace import (
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.workspace.configs import (
    SamplingConfig,
    SamplingStrategy,
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
    """Run the joint-space workspace analysis example."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_viser_args_to_parser(parser)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30_000,
        help="Number of joint configurations to sample (default: 30000).",
    )
    parser.add_argument(
        "--sim-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Simulation and workspace-computation device (default: cpu).",
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
        visualization=visualization_cfg_from_args(args),
    )
    sim_manager = SimulationManager(config)
    try:
        cfg = DexforceW1Cfg.from_dict(
            {
                "uid": "dexforce_w1",
                "version": "v021",
                # Workspace analysis only needs the arms. Excluding the grippers
                # avoids loading unrelated joints and assets.
                "with_default_eef": False,
            }
        )
        robot = sim_manager.add_robot(cfg=cfg)
        print("DexforceW1 robot added to the simulation.")

        analyzer = WorkspaceAnalyzer(
            robot=robot,
            config=WorkspaceAnalyzerConfig(
                control_part_name="left_arm",
                sampling=SamplingConfig(strategy=SamplingStrategy.RANDOM),
            ),
            sim_manager=sim_manager,
        )
        results = analyzer.analyze(num_samples=args.num_samples, visualize=True)

        print("\nJoint Space Results:")
        print(f"  Valid points: {results['num_valid']} / {results['num_samples']}")
        print(f"  Analysis time: {results['analysis_time']:.2f}s")
        print(f"  Metrics: {results['metrics']}")

        sim_manager.capture_visualization(force=True)
        if not args.no_wait:
            _run_until_interrupted(sim_manager)
    finally:
        sim_manager.destroy()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
