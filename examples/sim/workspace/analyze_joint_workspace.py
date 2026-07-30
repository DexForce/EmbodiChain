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

import numpy as np
import torch
from IPython import embed

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.workspace.analyzer import (
    WorkspaceAnalyzer,
)
from embodichain.lab.visualization import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_viser_args_to_parser(parser)
    args = parser.parse_args()

    # Example usage
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    config = SimulationManagerCfg(
        headless=False,
        sim_device="cpu",
        visualization=visualization_cfg_from_args(args),
    )
    sim_manager = SimulationManager(config)
    sim_manager.set_manual_update(False)

    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "industrial"}
    )
    robot = sim_manager.add_robot(cfg=cfg)
    print("DexforceW1 robot added to the simulation.")

    print("Example: Joint Space Analysis")

    wa_joint = WorkspaceAnalyzer(robot=robot, sim_manager=sim_manager)
    results_joint = wa_joint.analyze(num_samples=30000, visualize=True)

    print(f"\nJoint Space Results:")
    print(
        f"  Valid points: {results_joint['num_valid']} / {results_joint['num_samples']}"
    )
    print(f"  Analysis time: {results_joint['analysis_time']:.2f}s")
    print(f"  Metrics: {results_joint['metrics']}")

    sim_manager.capture_visualization(force=True)
    embed(header="End of Joint Space Analysis Example")
