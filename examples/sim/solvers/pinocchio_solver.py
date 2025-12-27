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
import os
import time
import numpy as np
import torch
from IPython import embed

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg


def main():
    # Set print options for better readability
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # Initialize simulation
    sim_device = "cpu"
    config = SimulationManagerCfg(headless=False, sim_device=sim_device)
    sim = SimulationManager(config)
    sim.set_manual_update(False)

    # Load robot URDF file
    urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
    assert os.path.isfile(urdf)

    # Robot configuration dictionary
    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "left_arm": [f"LEFT_J{i+1}" for i in range(7)],
            "right_arm": [f"RIGHT_J{i+1}" for i in range(7)],
        },
        "solver_cfg": {
            "left_arm": {
                "class_type": "PinocchioSolver",
                "end_link_name": "left_ee",
                "root_link_name": "left_arm_base",
            },
            "right_arm": {
                "class_type": "PinocchioSolver",
                "end_link_name": "right_ee",
                "root_link_name": "right_arm_base",
            },
        },
    }

    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))
    arm_name = "left_arm"
    # Set initial joint positions for left arm
    qpos_seed = torch.tensor(
        [[0.0, 0.1, 0.0, -np.pi / 4, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    qpos_fk = torch.tensor(
        [[0.0, 0.0, 0.0, -np.pi / 4, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    fk_xpos = robot.compute_fk(qpos=qpos_fk, name=arm_name, to_matrix=True)
    link_pose = robot._entities[0].get_link_pose("left_arm_base")
    link_pose_tensor = torch.from_numpy(link_pose).to(
        fk_xpos.device, dtype=fk_xpos.dtype
    )

    # Solve IK for the left arm
    res, ik_qpos = robot.compute_ik(pose=fk_xpos, name=arm_name, joint_seed=qpos_seed)

    # Measure IK computation time and visualize result
    a = time.time()
    if ik_qpos.dim() == 3:
        ik_xpos = robot.compute_fk(qpos=ik_qpos[0][0], name=arm_name, to_matrix=True)
    else:
        ik_xpos = robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)
    b = time.time()
    print(f"IK computation time: {b-a:.6f} seconds")

    fk_xpos = link_pose_tensor @ fk_xpos
    ik_xpos = link_pose_tensor @ ik_xpos

    # Visualize the result in simulation
    sim.draw_marker(
        cfg=MarkerCfg(
            name="fk_xpos",
            marker_type="axis",
            axis_xpos=np.array(fk_xpos.tolist()),
            axis_size=0.002,
            axis_len=0.005,
        )
    )

    sim.draw_marker(
        cfg=MarkerCfg(
            name="ik_xpos",
            marker_type="axis",
            axis_xpos=np.array(ik_xpos.tolist()),
            axis_size=0.002,
            axis_len=0.005,
        )
    )

    # Move robot to IK result joint positions
    if ik_qpos.dim() == 3:
        robot.set_qpos(qpos=ik_qpos[0][0], joint_ids=robot.get_joint_ids(arm_name))
    else:
        robot.set_qpos(qpos=ik_qpos, joint_ids=robot.get_joint_ids(arm_name))

    embed(header="Test PinocchioSolver example. Press Ctrl-D to exit.")


if __name__ == "__main__":
    main()
