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
from embodichain.lab.sim.cfg import MarkerCfg, RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg


def main():
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    sim_device = "cpu"
    config = SimulationManagerCfg(headless=False, sim_device=sim_device)
    sim = SimulationManager(config)
    sim.set_manual_update(False)

    # Franka FR3 URDF from EmbodiChain data
    urdf = get_data_path("Franka/FR3/fr3.urdf")
    assert os.path.isfile(urdf)

    checkpoint_path = os.path.expanduser(
        "~/文档/Research/analytic_policy_gradients/checkpoints/"
        "FrankaReach-v0__rl__1__1779942193/best.pt"
    )
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    c = math.cos(-math.pi / 4)
    s = math.sin(-math.pi / 4)
    tcp = [
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ]
    joint_names = [
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7",
    ]


    cfg_dict = {
        "fpath": urdf,
        "control_parts": {
            "main_arm": joint_names,
        },
        "solver_cfg": {
            "main_arm": {
                "class_type": "NeuralIKSolver",
                "end_link_name": "fr3_link8",
                "root_link_name": "fr3_link0",
                "tcp": tcp,
                "checkpoint_path": checkpoint_path,
                "num_arm_joints": 7,
                "max_steps": 30,
                "action_scale": 0.2,
                "hidden_dims": [256, 256],
            },
        },
    }
    import ipdb; ipdb.set_trace()
    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))
    ipdb.set_trace()
    arm_name = "main_arm"

    # Set home position and also set it as the joint target for PD control
    qpos = torch.tensor(
        [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854],
        dtype=torch.float32,
        device=sim_device,
    ).unsqueeze(0)
    robot.set_qpos(qpos=qpos, joint_ids=robot.get_joint_ids(arm_name))
    robot.set_qpos_target(qpos=qpos, joint_ids=robot.get_joint_ids(arm_name))

    # Wait for physics to settle
    time.sleep(2.0)

    # Compute FK to get current EE pose (with TCP applied)
    fk_xpos = robot.compute_fk(qpos=qpos, name=arm_name, to_matrix=True)
    print(f"Current EE pose (TCP):\n{fk_xpos}")

    # Define target offset from current pose
    start_pose = fk_xpos.clone()[0]
    end_pose = fk_xpos.clone()[0]
    end_pose[:3, 3] += torch.tensor([0.1, 0.2, -0.1], device=sim_device)

    # Interpolate between start and end
    num_steps = 50
    interpolated_poses = [
        torch.lerp(start_pose, end_pose, t) for t in np.linspace(0, 1, num_steps)
    ]

    ik_qpos = qpos

    # Solve IK for the end pose and measure time
    t0 = time.time()
    res, ik_qpos = robot.compute_ik(
        pose=end_pose, joint_seed=ik_qpos, name=arm_name
    )
    t1 = time.time()
    print(f"End-pose IK: success={res}, time={t1 - t0:.4f}s")

    if ik_qpos.dim() == 3:
        ik_xpos = robot.compute_fk(
            qpos=ik_qpos[0][0], name=arm_name, to_matrix=True
        )
    else:
        ik_xpos = robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)
    pos_err = (ik_xpos[:, :3, 3] - end_pose[:3, 3].unsqueeze(0)).norm().item()
    print(f"Position error at end pose: {pos_err:.6f} m")

    # Draw markers for target and IK result
    sim.draw_marker(
        cfg=MarkerCfg(
            name="target",
            marker_type="axis",
            axis_xpos=np.array(end_pose.tolist()),
            axis_size=0.002,
            axis_len=0.005,
        )
    )
    sim.draw_marker(
        cfg=MarkerCfg(
            name="ik_result",
            marker_type="axis",
            axis_xpos=np.array(ik_xpos.tolist()),
            axis_size=0.002,
            axis_len=0.005,
        )
    )

    # Step through interpolated poses
    for i, pose in enumerate(interpolated_poses):
        t_start = time.time()
        res, ik_qpos = robot.compute_ik(
            pose=pose, joint_seed=ik_qpos, name=arm_name
        )
        t_end = time.time()

        if not res:
            print(f"Step {i}: IK failed (pos_err shown in solver)")
            continue

        if ik_qpos.dim() == 3:
            q = ik_qpos[0][0]
        else:
            q = ik_qpos
        robot.set_qpos(qpos=q, joint_ids=robot.get_joint_ids(arm_name))
        robot.set_qpos_target(qpos=q, joint_ids=robot.get_joint_ids(arm_name))

        print(f"Step {i}: time={t_end - t_start:.4f}s")
        time.sleep(0.01)

    embed(header="NeuralIKSolver example. Press Ctrl+D to exit.")


if __name__ == "__main__":
    main()
