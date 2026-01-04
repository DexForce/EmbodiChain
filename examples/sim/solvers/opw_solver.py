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
import torch
import numpy as np
from IPython import embed

from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import CobotMagicCfg
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

    # Robot configuration dictionary
    cfg_dict = {
        "uid": "CobotMagic",
        "init_pos": [0.0, 0.0, 0.7775],
        "init_qpos": [
            -0.3,
            0.3,
            1.0,
            1.0,
            -1.2,
            -1.2,
            0.0,
            0.0,
            0.6,
            0.6,
            0.0,
            0.0,
            0.05,
            0.05,
            0.05,
            0.05,
        ],
        "solver_cfg": {
            "left_arm": {
                "class_type": "OPWSolver",
                "end_link_name": "left_link6",
                "root_link_name": "left_arm_base",
                "tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
            },
            "right_arm": {
                "class_type": "OPWSolver",
                "end_link_name": "right_link6",
                "root_link_name": "right_arm_base",
                "tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
            },
        },
    }

    # Add robot to simulation
    robot: Robot = sim.add_robot(cfg=CobotMagicCfg.from_dict(cfg_dict))

    # Left arm control
    arm_name = "left_arm"
    # Set initial joint positions for left arm
    qpos_seed = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    qpos_fk = torch.tensor(
        [[0.0, np.pi / 4, -np.pi / 4, 0.0, np.pi / 4, 0.0]], dtype=torch.float32
    )
    fk_xpos = robot.compute_fk(qpos=qpos_fk, name=arm_name, to_matrix=True)

    link_pose = robot._entities[0].get_link_pose("left_base_link")
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
    print(f"Left arm IK computation time: {b-a:.6f} seconds")

    # Visualize the result in simulation
    sim.draw_marker(
        cfg=MarkerCfg(
            name="fk_xpos_left",
            marker_type="axis",
            axis_xpos=np.array(fk_xpos.tolist()),
            axis_size=0.002,
            axis_len=0.005,
        )
    )

    sim.draw_marker(
        cfg=MarkerCfg(
            name="ik_xpos_left",
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

    # Right arm control
    arm_name_r = "right_arm"
    # Set initial joint positions for right arm
    qpos_seed_r = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    qpos_fk_r = torch.tensor(
        [[0.0, np.pi / 4, -np.pi / 4, 0.0, np.pi / 4, 0.0]], dtype=torch.float32
    )
    fk_xpos_r = robot.compute_fk(qpos=qpos_fk_r, name=arm_name_r, to_matrix=True)

    link_pose_r = robot._entities[0].get_link_pose("right_base_link")
    link_pose_tensor_r = torch.from_numpy(link_pose_r).to(
        fk_xpos_r.device, dtype=fk_xpos_r.dtype
    )

    # Solve IK for the right arm
    res_r, ik_qpos_r = robot.compute_ik(
        pose=fk_xpos_r, name=arm_name_r, joint_seed=qpos_seed_r
    )

    # Measure IK computation time and visualize result
    a_r = time.time()
    if ik_qpos_r.dim() == 3:
        ik_xpos_r = robot.compute_fk(
            qpos=ik_qpos_r[0][0], name=arm_name_r, to_matrix=True
        )
    else:
        ik_xpos_r = robot.compute_fk(qpos=ik_qpos_r, name=arm_name_r, to_matrix=True)
    b_r = time.time()
    print(f"Right arm IK computation time: {b_r-a_r:.6f} seconds")

    # Visualize the result in simulation
    sim.draw_marker(
        cfg=MarkerCfg(
            name="fk_xpos_right",
            marker_type="axis",
            axis_xpos=np.array(fk_xpos_r.tolist()),
            axis_size=0.002,
            axis_len=0.005,
        )
    )

    sim.draw_marker(
        cfg=MarkerCfg(
            name="ik_xpos_right",
            marker_type="axis",
            axis_xpos=np.array(ik_xpos_r.tolist()),
            axis_size=0.002,
            axis_len=0.005,
        )
    )

    # Move robot to IK result joint positions
    if ik_qpos_r.dim() == 3:
        robot.set_qpos(qpos=ik_qpos_r[0][0], joint_ids=robot.get_joint_ids(arm_name_r))
    else:
        robot.set_qpos(qpos=ik_qpos_r, joint_ids=robot.get_joint_ids(arm_name_r))

    embed(header="Test OPWSolver example. Press Ctrl-D to exit.")


if __name__ == "__main__":
    main()
