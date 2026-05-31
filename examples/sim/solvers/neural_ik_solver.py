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
import math
import os
import time

import numpy as np
import torch
from IPython import embed

from embodichain.data import get_data_path
from embodichain.data.assets.solver_assets import download_neural_ik_checkpoint
from embodichain.lab.sim.cfg import MarkerCfg, RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg


def main():
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # Set up simulation with specified device (CPU or CUDA)
    sim_device = "cpu"
    config = SimulationManagerCfg(headless=False, sim_device=sim_device)
    sim = SimulationManager(config)
    sim.set_manual_update(False)

    # Load robot URDF file
    urdf = get_data_path("Franka/Panda/PandaWithHand.urdf")
    assert os.path.isfile(urdf)

    checkpoint_path = download_neural_ik_checkpoint()

    # TCP offset (rotation of -pi/4 around Z, translation along Z)
    c = math.cos(-math.pi / 4)
    s = math.sin(-math.pi / 4)
    tcp = [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ]

    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "main_arm": [
                "Joint1",
                "Joint2",
                "Joint3",
                "Joint4",
                "Joint5",
                "Joint6",
                "Joint7",
            ],
        },
        "solver_cfg": {
            "main_arm": {
                "class_type": "NeuralIKSolver",
                "end_link_name": "ee_link",
                "root_link_name": "base_link",
                "tcp": tcp,
                "checkpoint_path": checkpoint_path,
                "num_arm_joints": 7,
                "max_steps": 30,
                "action_scale": 0.2,
                "hidden_dims": [256, 256],
                "pos_eps": 0.1,
                "rot_eps": 0.5,
            },
        },
    }

    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))
    arm_name = "main_arm"

    # Set initial joint positions
    qpos = torch.tensor(
        [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
        dtype=torch.float32,
        device=sim_device,
    ).unsqueeze(0)
    robot.set_qpos(qpos=qpos, joint_ids=robot.get_joint_ids(arm_name))
    time.sleep(3.0)

    # Compute FK to get current EE pose (with TCP applied)
    fk_xpos = robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)
    print(f"fk_xpos: {fk_xpos}")
    start_pose = fk_xpos.clone()[0]
    end_pose = fk_xpos.clone()[0]
    end_pose[:3, 3] += torch.tensor([0.3, 0.4, -0.2], device=sim_device)

    num_steps = 50

    # Interpolate poses between start and end
    interpolated_poses = [
        torch.lerp(start_pose, end_pose, t) for t in np.linspace(0, 1, num_steps)
    ]

    ik_qpos = qpos

    # Solve IK for the end pose and measure time
    start_time = time.time()
    res, ik_qpos = robot.compute_ik(pose=end_pose, joint_seed=ik_qpos, name=arm_name)
    end_time = time.time()
    print(f"End-pose IK: success={res}, time={end_time - start_time:.4f}s")

    if ik_qpos.dim() == 3:
        ik_xpos = robot.compute_fk(qpos=ik_qpos[0][0], name=arm_name, to_matrix=True)
    else:
        ik_xpos = robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)

    sim.draw_marker(
        cfg=MarkerCfg(
            name="fk_xpos",
            marker_type="axis",
            axis_xpos=np.array(end_pose.tolist()),
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

    for i, pose in enumerate(interpolated_poses):
        print(f"Step {i}: Moving to pose:\n{pose}")
        start_time = time.time()
        res, ik_qpos = robot.compute_ik(pose=pose, joint_seed=ik_qpos, name=arm_name)
        end_time = time.time()
        compute_time = end_time - start_time
        print(f"Step {i}: IK computation time: {compute_time:.6f} seconds")

        print(f"IK result: {res}, ik_qpos: {ik_qpos}")
        if not res:
            print(f"Step {i}: IK failed for pose:\n{pose}")
            continue

        # Set robot joint positions
        if ik_qpos.dim() == 3:
            robot.set_qpos(qpos=ik_qpos[0][0], joint_ids=robot.get_joint_ids(arm_name))
        else:
            robot.set_qpos(qpos=ik_qpos, joint_ids=robot.get_joint_ids(arm_name))

        # Visualize current pose
        ik_xpos = robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)

        sim.draw_marker(
            cfg=MarkerCfg(
                name=f"ik_xpos_step_{i}",
                marker_type="axis",
                axis_xpos=np.array(ik_xpos.tolist()),
                axis_size=0.002,
                axis_len=0.005,
            )
        )

        time.sleep(0.02)

    embed(header="NeuralIKSolver example. Press Ctrl+D to exit.")


if __name__ == "__main__":
    main()
