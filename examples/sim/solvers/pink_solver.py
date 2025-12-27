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
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # Set up simulation with specified device (CPU or CUDA)
    sim_device = "cpu"
    config = SimulationManagerCfg(headless=False, sim_device=sim_device)
    sim = SimulationManager(config)
    sim.build_multiple_arenas(1)
    sim.set_manual_update(False)

    # Load robot URDF file
    urdf = get_data_path("Rokae/SR5/SR5.urdf")

    assert os.path.isfile(urdf)

    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "main_arm": [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ],
        },
        "solver_cfg": {
            "main_arm": {
                "class_type": "PinkSolver",
                "end_link_name": "ee_link",
                "root_link_name": "base_link",
            },
        },
    }

    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

    # Define a sample target pose as a 1x4x4 homogeneous matrix
    rad = torch.deg2rad(torch.tensor(45.0))

    arm_name = "main_arm"
    fk_qpos = torch.full((1, 6), rad, dtype=torch.float32, device="cpu")

    # Set initial joint positions
    qpos = torch.from_numpy(np.array([0.0, 0.0, np.pi / 2, 0.0, np.pi / 2, 0.0])).to(
        fk_qpos.device
    )
    qpos = qpos.unsqueeze(0)
    robot.set_qpos(qpos=qpos, joint_ids=robot.get_joint_ids("main_arm"))
    import time

    time.sleep(3.0)
    fk_xpos = robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)
    print(f"fk_xpos: {fk_xpos}")
    start_pose = fk_xpos.clone()[0]  # Start pose
    end_pose = fk_xpos.clone()[0]  # End pose

    end_pose[:3, 3] = end_pose[:3, 3][:3] + torch.tensor(
        [0.0, 0.4, 0.0], device=fk_xpos.device
    )

    num_steps = 100

    # Interpolate poses between start and end
    interpolated_poses = [
        torch.lerp(start_pose, end_pose, t) for t in np.linspace(0, 1, num_steps)
    ]

    ik_qpos = qpos

    qpos = ik_qpos
    res, ik_qpos = robot.compute_ik(pose=end_pose, joint_seed=qpos, name=arm_name)
    import time

    a = time.time()
    if ik_qpos.dim() == 3:
        ik_xpos = robot.compute_fk(qpos=ik_qpos[0][0], name=arm_name, to_matrix=True)
    else:
        ik_xpos = robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)
    b = time.time()
    print(f"ik time: {b-a}")

    ik_xpos = ik_xpos

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
        ik_xpos = ik_xpos

        sim.draw_marker(
            cfg=MarkerCfg(
                name=f"ik_xpos_step_{i}",
                marker_type="axis",
                axis_xpos=np.array(ik_xpos.tolist()),
                axis_size=0.002,
                axis_len=0.005,
            )
        )

        # Add delay to simulate motion
        time.sleep(0.005)

    embed(header="Test PinkSolver example. Press Ctrl-D to exit.")


if __name__ == "__main__":
    main()
