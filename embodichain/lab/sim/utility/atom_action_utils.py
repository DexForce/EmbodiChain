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

import numpy as np
import torch
import ast
from typing import List

from embodichain.utils.logger import log_error, log_warning
from embodichain.lab.gym.utils.misc import mul_linear_expand
from embodichain.lab.sim.planners.motion_generator import MotionGenerator


def draw_axis(env, pose):
    """Draw an axis marker in the simulation for debugging/visualization.

    Args:
        env: The simulation environment.
        pose: The pose (4x4 matrix) where to draw the axis.
    """
    from embodichain.lab.sim.cfg import MarkerCfg

    marker_cfg = MarkerCfg(
        name="test",
        marker_type="axis",
        axis_xpos=pose,
        axis_size=0.01,
        axis_len=0.2,
        arena_index=-1,  # All arenas
    )
    env.sim.draw_marker(cfg=marker_cfg)
    env.sim.update()


def get_arm_states(env, robot_name):
    """Get the current state of the specified robot arm.

    Args:
        env: The simulation environment.
        robot_name: Name of the robot arm (should contain "left" or "right").

    Returns:
        Tuple of (is_left, select_arm, current_qpos, current_pose, current_gripper_state):
            - is_left: bool, whether this is the left arm
            - select_arm: str, arm identifier ("left_arm" or "right_arm")
            - current_qpos: Current joint positions
            - current_pose: Current end-effector pose (4x4 matrix)
            - current_gripper_state: Current gripper state
    """
    left_arm_current_qpos, right_arm_current_qpos = env.get_current_qpos_agent()
    left_arm_current_pose, right_arm_current_pose = env.get_current_xpos_agent()
    left_arm_current_gripper_state, right_arm_current_gripper_state = (
        env.get_current_gripper_state_agent()
    )

    side = "right" if "right" in robot_name else "left"
    is_left = True if side == "left" else False
    select_arm = "left_arm" if is_left else "right_arm"

    arms = {
        "left": (
            left_arm_current_qpos,
            left_arm_current_pose,
            left_arm_current_gripper_state,
        ),
        "right": (
            right_arm_current_qpos,
            right_arm_current_pose,
            right_arm_current_gripper_state,
        ),
    }
    (
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = arms[side]

    return (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    )


def find_nearest_valid_pose(env, select_arm, pose, xpos_resolution=0.1):
    """Find the nearest valid pose using reachability validation.

    Args:
        env: The simulation environment.
        select_arm: Arm identifier ("left_arm" or "right_arm").
        pose: Target pose (4x4 matrix).
        xpos_resolution: Resolution for reachability checking.

    Returns:
        torch.Tensor: The nearest valid pose (4x4 matrix).
    """
    # use the validator to choose the nearest valid pose
    # delete the cache every time
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    ret, _ = env.robot.compute_xpos_reachability(
        select_arm,
        pose,
        xpos_resolution=xpos_resolution,
        qpos_resolution=np.radians(60),
        cache_mode="disk",
        use_cached=False,
        visualize=False,
    )
    ret = np.stack(ret, axis=0)
    # find the nearest valid pose
    xyz = pose[:3, 3]
    ts = np.stack([M[:3, 3] for M in ret], axis=0)  # shape (N,3)
    dists = np.linalg.norm(ts - xyz[None, :], axis=1)
    best_idx = np.argmin(dists)
    nearest_valid_pose = ret[best_idx]
    return torch.from_numpy(nearest_valid_pose)


def get_qpos(env, is_left, select_arm, pose, qpos_seed, force_valid=False, name=""):
    """Solve inverse kinematics to get joint positions for a given pose.

    Args:
        env: The simulation environment.
        is_left: bool, whether this is the left arm.
        select_arm: Arm identifier ("left_arm" or "right_arm").
        pose: Target end-effector pose (4x4 matrix).
        qpos_seed: Seed joint positions for IK solving.
        force_valid: If True, use nearest valid pose if IK fails.
        name: Name for logging purposes.

    Returns:
        Joint positions (qpos) corresponding to the target pose.
    """
    if force_valid:
        try:
            ret, qpos = env.get_arm_ik(pose, is_left=is_left, qpos_seed=qpos_seed)
            if not ret:
                log_error(f"Generate {name} qpos failed.\n")
        except Exception as e:
            log_warning(
                f"Original {name} pose invalid, using nearest valid pose. ({e})\n"
            )
            pose = find_nearest_valid_pose(env, select_arm, pose)

            ret, qpos = env.get_arm_ik(pose, is_left=is_left, qpos_seed=qpos_seed)
    else:
        ret, qpos = env.get_arm_ik(pose, is_left=is_left, qpos_seed=qpos_seed)
        if not ret:
            log_error(f"Generate {name} qpos failed.\n")

    return qpos


def plan_trajectory(
    env,
    select_arm,
    qpos_list,
    sample_num,
    select_arm_current_gripper_state,
    select_qpos_traj,
    ee_state_list_select,
):
    """Plan a trajectory between joint positions and append to trajectory lists.

    Args:
        env: The simulation environment.
        select_arm: Arm identifier ("left_arm" or "right_arm").
        qpos_list: List of joint positions to plan between.
        sample_num: Number of samples for trajectory interpolation.
        select_arm_current_gripper_state: Current gripper state.
        select_qpos_traj: List to append planned joint positions to (modified in-place).
        ee_state_list_select: List to append gripper states to (modified in-place).
    """
    motion_generator = MotionGenerator(
        robot=env.robot,
        uid=select_arm,
        **getattr(env, "planning_config", {}),
    )
    traj_list, _ = motion_generator.create_discrete_trajectory(
        qpos_list=qpos_list,
        sample_num=sample_num,
        qpos_seed=qpos_list[0],
        is_use_current_qpos=False,
        **getattr(env, "planning_config", {}),
    )

    select_qpos_traj.extend(traj_list)
    ee_state_list_select.extend([select_arm_current_gripper_state] * len(traj_list))


def plan_gripper_trajectory(
    env,
    is_left,
    sample_num,
    execute_open,
    select_arm_current_qpos,
    select_qpos_traj,
    ee_state_list_select,
):
    """Plan a gripper trajectory (opening or closing) and append to trajectory lists.

    Args:
        env: The simulation environment.
        is_left: bool, whether this is the left arm.
        sample_num: Number of samples for gripper motion.
        execute_open: bool, True for opening, False for closing.
        select_arm_current_qpos: Current joint positions.
        select_qpos_traj: List to append joint positions to (modified in-place).
        ee_state_list_select: List to append gripper states to (modified in-place).
    """
    open_state = env.open_state
    close_state = env.close_state

    if execute_open:
        ee_state_expand_select = np.array([close_state, open_state])
        env.set_current_gripper_state_agent(open_state, is_left=is_left)
    else:
        ee_state_expand_select = np.array([open_state, close_state])
        env.set_current_gripper_state_agent(close_state, is_left=is_left)

    ee_state_expand_select = mul_linear_expand(ee_state_expand_select, [sample_num])

    select_qpos_traj.extend([select_arm_current_qpos] * sample_num)
    ee_state_list_select.extend(ee_state_expand_select)


def finalize_actions(select_qpos_traj, ee_state_list_select):
    """Format trajectory data into action format.

    Args:
        select_qpos_traj: List of joint positions.
        ee_state_list_select: List of gripper states.

    Returns:
        np.ndarray: Formatted actions array with joint positions and gripper states.
    """
    # mimic eef state
    actions = np.concatenate(
        [
            np.array(select_qpos_traj),
            np.array(ee_state_list_select),
            np.array(ee_state_list_select),
        ],
        axis=-1,
    )
    return actions


def extract_drive_calls(code_str: str) -> List[str]:
    """Extract all drive() function calls from a code string.

    Args:
        code_str: Python code string to parse.

    Returns:
        List of code blocks containing drive() calls.
    """
    tree = ast.parse(code_str)
    lines = code_str.splitlines()

    drive_blocks = []

    for node in tree.body:
        # Match: drive(...)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "drive"
        ):
            # AST line numbers are 1-based
            start = node.lineno - 1
            end = node.end_lineno
            block = "\n".join(lines[start:end])
            drive_blocks.append(block)

    return drive_blocks
