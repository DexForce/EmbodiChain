# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
import numpy as np
from IPython import embed

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
    AnalysisMode,
)
from embodichain.lab.sim.utility.workspace_analyzer.configs.visualization_config import (
    VisualizationConfig,
)


if __name__ == "__main__":
    # Example usage
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    config = SimulationManagerCfg(headless=False, sim_device="cpu")
    sim = SimulationManager(config)
    sim.build_multiple_arenas(1)
    sim.set_manual_update(False)

    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "anthropomorphic"}
    )
    robot = sim.add_robot(cfg=cfg)
    print("DexforceW1 robot added to the simulation.")

    # Set left arm joint positions (mirrored)
    robot.set_qpos(
        qpos=[0, -np.pi / 4, 0.0, -np.pi / 2, -np.pi / 4, 0.0, 0.0],
        joint_ids=robot.get_joint_ids("left_arm"),
    )
    # Set right arm joint positions (mirrored)
    robot.set_qpos(
        qpos=[0, np.pi / 4, 0.0, np.pi / 2, np.pi / 4, 0.0, 0.0],
        joint_ids=robot.get_joint_ids("right_arm"),
    )

    cartesian_config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.PLANE_SAMPLING,
        plane_normal=torch.tensor([0.0, 0.0, 1.0]),
        plane_point=torch.tensor([0.0, 0.0, 1.2]),
        # plane_bounds=torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]),
        visualization=VisualizationConfig(show_unreachable_points=True),
    )
    wa_cartesian = WorkspaceAnalyzer(
        robot=robot, config=cartesian_config, sim_manager=sim
    )
    results_cartesian = wa_cartesian.analyze(num_samples=1000, visualize=True)
    print(f"\nCartesian Space Results:")
    print(
        f"  Reachable points: {results_cartesian['num_reachable']} / {results_cartesian['num_samples']}"
    )
    print(f"  Analysis time: {results_cartesian['analysis_time']:.2f}s")
    print(f"  Metrics: {results_cartesian['metrics']}")

    embed(header="Workspace Analyzer Test Environment")
