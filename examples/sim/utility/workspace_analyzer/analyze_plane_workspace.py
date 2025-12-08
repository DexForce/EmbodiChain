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
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.utility.workspace_analyzer.configs.visualization_config import (
    VisualizationConfig,
)


if __name__ == "__main__":
    # Example usage
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    config = SimulationManagerCfg(
        headless=False, sim_device="cpu", width=1080, height=1080
    )
    sim = SimulationManager(config)
    sim.build_multiple_arenas(1)
    sim.set_manual_update(False)

    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "industrial"}
    )
    robot = sim.add_robot(cfg=cfg)
    print("DexforceW1 robot added to the simulation.")

    # Set left arm joint positions (mirrored)
    left_qpos = torch.tensor([0, -np.pi / 4, 0.0, -np.pi / 2, -np.pi / 4, 0.0, 0.0])
    right_qpos = -left_qpos
    robot.set_qpos(
        qpos=left_qpos,
        joint_ids=robot.get_joint_ids("left_arm"),
    )
    # Set right arm joint positions (mirrored)
    robot.set_qpos(
        qpos=right_qpos,
        joint_ids=robot.get_joint_ids("right_arm"),
    )

    left_arm_pose = robot.compute_fk(
        qpos=left_qpos,
        name="left_arm",
        to_matrix=True,
    )

    sim.draw_marker(
        cfg=MarkerCfg(
            name=f"left_arm_pose_axis",
            marker_type="axis",
            axis_xpos=left_arm_pose,
            axis_size=0.005,
            axis_len=0.15,
            arena_index=0,
        )
    )

    cartesian_config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.PLANE_SAMPLING,
        plane_normal=torch.tensor([0.0, 0.0, 1.0]),
        plane_point=torch.tensor([0.0, 0.0, 1.2]),
        # plane_bounds=torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]),
        reference_pose=left_arm_pose[0]
        .cpu()
        .numpy(),  # Use computed left arm pose as reference
        visualization=VisualizationConfig(
            show_unreachable_points=False, vis_type="axis"
        ),
        control_part_name="left_arm",
    )
    wa_cartesian = WorkspaceAnalyzer(
        robot=robot, config=cartesian_config, sim_manager=sim
    )
    results_cartesian = wa_cartesian.analyze(num_samples=1500, visualize=True)
    print(f"\nCartesian Space Results:")
    print(
        f"  Reachable points: {results_cartesian['num_reachable']} / {results_cartesian['num_samples']}"
    )
    print(f"  Analysis time: {results_cartesian['analysis_time']:.2f}s")
    print(f"  Metrics: {results_cartesian['metrics']}")

    embed(header="Workspace Analyzer Test Environment")
