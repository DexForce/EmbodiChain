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

import time
import torch
import numpy as np
from copy import deepcopy
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import CobotMagicCfg
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.planners.motion_generator import MotionGenerator


def move_robot_along_trajectory(
    robot: Robot, arm_name: str, qpos_list: list[torch.Tensor], delay: float = 0.1
):
    """
    Set the robot joint positions sequentially along the given joint trajectory.
    Args:
        robot: Robot instance.
        arm_name: Name of the robot arm.
        qpos_list: List of joint positions (torch.Tensor).
        delay: Time delay between each step (seconds).
    """
    for q in qpos_list:
        robot.set_qpos(qpos=q.unsqueeze(0), joint_ids=robot.get_joint_ids(arm_name))
        time.sleep(delay)


def create_demo_trajectory(
    robot: Robot, arm_name: str
) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    """
    Generate a three-point trajectory (start, middle, end) for demonstration.
    Args:
        robot: Robot instance.
        arm_name: Name of the robot arm.
    Returns:
        qpos_list: List of joint positions (torch.Tensor).
        xpos_list: List of end-effector poses (numpy arrays).
    """
    qpos_fk = torch.tensor(
        [[0.0, np.pi / 4, -np.pi / 4, 0.0, np.pi / 4, 0.0]], dtype=torch.float32
    )
    xpos_begin = robot.compute_fk(name=arm_name, qpos=qpos_fk, to_matrix=True)
    xpos_mid = deepcopy(xpos_begin)
    xpos_mid[0, 2, 3] -= 0.1  # Move down by 0.1m in Z direction
    xpos_final = deepcopy(xpos_mid)
    xpos_final[0, 0, 3] += 0.2  # Move forward by 0.2m in X direction

    qpos_begin = robot.compute_ik(pose=xpos_begin, name=arm_name)[1][0]
    qpos_mid = robot.compute_ik(pose=xpos_mid, name=arm_name)[1][0]
    qpos_final = robot.compute_ik(pose=xpos_final, name=arm_name)[1][0]
    return [qpos_begin, qpos_mid, qpos_final], [
        xpos_begin[0],
        xpos_mid[0],
        xpos_final[0],
    ]


def main():
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    # Initialize simulation
    sim = SimulationManager(SimulationManagerCfg(headless=False, sim_device="cpu"))
    sim.build_multiple_arenas(1)
    sim.set_manual_update(False)

    # Robot configuration
    cfg_dict = {"uid": "CobotMagic"}
    robot: Robot = sim.add_robot(cfg=CobotMagicCfg.from_dict(cfg_dict))
    arm_name = "left_arm"

    # # Generate trajectory points
    qpos_list, xpos_list = create_demo_trajectory(robot, arm_name)

    # Initialize motion generator
    motion_generator = MotionGenerator(
        robot=robot,
        uid=arm_name,
        planner_type="toppra",
        default_velocity=0.2,
        default_acceleration=0.5,
    )

    # Joint space trajectory
    out_qpos_list, _ = motion_generator.create_discrete_trajectory(
        qpos_list=[q.numpy() for q in qpos_list],
        is_linear=False,
        sample_method=TrajectorySampleMethod.QUANTITY,
        sample_num=20,
    )
    move_robot_along_trajectory(robot, arm_name, out_qpos_list)

    # Cartesian space trajectory
    out_qpos_list, _ = motion_generator.create_discrete_trajectory(
        xpos_list=[x.numpy() for x in xpos_list],
        is_linear=True,
        sample_method=TrajectorySampleMethod.QUANTITY,
        sample_num=20,
    )
    move_robot_along_trajectory(robot, arm_name, out_qpos_list)


if __name__ == "__main__":
    main()
