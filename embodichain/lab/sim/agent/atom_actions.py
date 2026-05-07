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
from embodichain.utils.logger import log_info, log_warning, log_error
from copy import deepcopy
from embodichain.lab.gym.utils.misc import (
    mul_linear_expand,
    get_rotation_replaced_pose,
)
from embodichain.utils.math import get_offset_pose
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from embodichain.utils.utility import encode_image
from functools import partial

# Import utility functions for atom actions
from embodichain.lab.sim.agent.atom_action_utils import (
    draw_axis,
    get_arm_states,
    find_nearest_valid_pose,
    get_qpos,
    plan_trajectory,
    plan_gripper_trajectory,
    finalize_actions,
    extract_drive_calls,
    apply_offset_to_pose,
    resolve_action,
    get_error_probability,
)
from embodichain.lab.sim.agent.monitor_functions import *
from embodichain.agents.hierarchy.error_agent import (
    inject_object_error,
    inject_action_error,
    action_error_types,
    object_error_types,
)


"""
--------------------------------------------Atom action functions----------------------------------------------------
--------------------------------------------Atom action functions----------------------------------------------------
--------------------------------------------Atom action functions----------------------------------------------------
"""
def move_to_target_pose(
    robot_name: str,
    target_pose=None,
    sample_num: int = 20,
    env=None,
    force_valid=False,
    **kwargs
):
    select_qpos_traj = []
    ee_state_list_select = []

    is_left, select_arm, select_arm_current_qpos, select_arm_current_pose, \
        select_arm_current_gripper_state = get_arm_states(env, robot_name)

    target_pose, move_target_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        target_pose,
        select_arm_current_qpos,
        force_valid=force_valid,
        name=kwargs.get("name", "move to target pose"),
    )
    qpos_list_move = [select_arm_current_qpos, move_target_qpos]

    env.set_current_qpos_agent(move_target_qpos, is_left=is_left)
    env.set_current_xpos_agent(target_pose, is_left=is_left)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_move,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select
    )

    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(f"Total generated trajectory number for move to target: {len(actions)}.", color="green")

    return actions

def grasp(
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float = 0.05,
    env=None,
    force_valid=False,
    **kwargs,
):
    # Get target object
    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name in obj_uids:
        target_obj = env.sim.get_rigid_object(obj_name)
    else:
        log_error(f"No matched object {obj_uids}.")
    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)

    # Open the gripper if currently closed
    actions = None
    select_arm_current_gripper_state = (
        env.left_arm_current_gripper_state
        if "left" in robot_name
        else env.right_arm_current_gripper_state
    )
    if select_arm_current_gripper_state <= env.open_state - 0.01:
        actions = open_gripper(robot_name, env, **kwargs)

    # Retract the end-effector to avoid collision
    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)
    select_arm_base_pose = (
        env.left_arm_base_pose if is_left else env.right_arm_base_pose
    )
    base_to_eef_xy_dis = torch.norm(
        select_arm_base_pose[:2, 3] - select_arm_current_pose[:2, 3]
    )
    base_to_obj_xy_dis = torch.norm(
        select_arm_base_pose[:2, 3] - target_obj_pose[:2, 3]
    )
    dis_eps = kwargs.get("dis_eps", 0.05)
    select_arm_init_pose = (
        env.left_arm_init_xpos if is_left else env.right_arm_init_xpos
    )
    if base_to_eef_xy_dis > base_to_obj_xy_dis and not torch.allclose(
        select_arm_current_pose, select_arm_init_pose, rtol=1e-5, atol=1e-8
    ):
        delta = float(base_to_eef_xy_dis - (base_to_obj_xy_dis - dis_eps))
        back_actions = move_by_relative_offset(
            robot_name=robot_name,
            dx=0.0,
            dy=0.0,
            dz=-delta,
            env=env,
            force_valid=force_valid,
            mode="intrinsic",
            sample_num=15,
            **kwargs,
        )
        actions = (
            np.concatenate([actions, back_actions], axis=0)
            if actions is not None
            else back_actions
        )

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    # ---------------------------------------- Pose ----------------------------------------
    # Rotate the arm base to face the object for better grasping
    delta_xy = target_obj_pose[:2, 3] - select_arm_base_pose[:2, 3]
    dx, dy = delta_xy[0], delta_xy[1]
    aim_horizontal_angle = np.arctan2(dy, dx)
    select_arm_aim_qpos = deepcopy(select_arm_current_qpos)
    select_arm_aim_qpos[0] = aim_horizontal_angle

    # Get best grasp pose from affordance data
    grasp_pose_object = env.obj_info.get(obj_name)["grasp_pose_obj"]
    if (
        grasp_pose_object[0, 2] > 0.5
    ):  # whether towards x direction TODO: make it robust
        # Align the object pose's z-axis with the arm's aiming direction
        target_obj_pose = torch.tensor(
            get_rotation_replaced_pose(
                np.array(target_obj_pose),
                float(select_arm_aim_qpos[0]),
                "z",
                "intrinsic",
            )
        )
    best_pickpose = target_obj_pose @ grasp_pose_object
    grasp_pose = deepcopy(best_pickpose)
    grasp_pose_pre1 = deepcopy(grasp_pose)
    grasp_pose_pre1 = get_offset_pose(grasp_pose_pre1, -pre_grasp_dis, "z", "intrinsic")

    # Solve IK for pre-grasp and grasp poses
    grasp_pose_pre1, grasp_qpos_pre1 = get_qpos(
        env,
        is_left,
        select_arm,
        grasp_pose_pre1,
        select_arm_aim_qpos,
        force_valid=force_valid,
        name="grasp pre1",
    )
    grasp_pose, grasp_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        grasp_pose,
        grasp_qpos_pre1,
        force_valid=force_valid,
        name="grasp",
    )

    # Update env state to final grasp pose
    env.set_current_qpos_agent(grasp_qpos, is_left=is_left)
    env.set_current_xpos_agent(grasp_pose, is_left=is_left)

    # ------------------------------------ Traj 0: init → aim ------------------------------------
    qpos_list_init_to_aim = [select_arm_current_qpos, select_arm_aim_qpos]
    # base_sample_num = 10
    # base_angle = 0.08
    # sample_num = max(int(delta_angle / base_angle * base_sample_num), 2)

    sample_num = 10

    plan_trajectory(
        env,
        select_arm,
        qpos_list_init_to_aim,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ------------------------------------ Traj 1: aim → pre-grasp ------------------------------------
    qpos_list_aim_to_pre1 = [select_arm_aim_qpos, grasp_qpos_pre1]
    sample_num = kwargs.get("sample_num", 30)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_aim_to_pre1,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ------------------------------------ Traj 2: pre-grasp → grasp ------------------------------------
    qpos_list_pre1_to_grasp = [grasp_qpos_pre1, grasp_qpos]
    sample_num = kwargs.get("sample_num", 20)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_pre1_to_grasp,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    traj_actions = finalize_actions(select_qpos_traj, ee_state_list_select)
    actions = (
        traj_actions
        if actions is None
        else np.concatenate([actions, traj_actions], axis=0)
    )

    # ------------------------------------ Close gripper ------------------------------------
    close_gripper_actions = close_gripper(robot_name, env, **kwargs)
    actions = np.concatenate([actions, close_gripper_actions], axis=0)

    log_info(
        f"Total generated trajectory number for grasp: {len(actions)}.", color="green"
    )

    return actions


def place_on_table(
    robot_name: str,
    obj_name: str,
    x: float = None,
    y: float = None,
    pre_place_dis: float = 0.08,
    env=None,
    force_valid=False,
    **kwargs,
):

    init_obj_height = env.obj_info.get(obj_name).get("height")
    height = init_obj_height + kwargs.get("eps", 0.03)

    traj_actions = move_to_absolute_position(
        robot_name, x=x, y=y, z=height, env=env, force_valid=force_valid, **kwargs
    )
    open_actions = open_gripper(robot_name, env, **kwargs)

    actions = np.concatenate([traj_actions, open_actions], axis=0)

    log_info(
        f"Total generated trajectory number for place on table: {len(actions)}.",
        color="green",
    )

    return actions


def move_relative_to_object(
    robot_name: str,
    obj_name: str,
    x_offset: float = 0,
    y_offset: float = 0,
    z_offset: float = 0,
    env=None,
    force_valid=False,
    **kwargs,
):

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    # ---------------------------------------- Pose ----------------------------------------
    # Resolve target object
    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name in obj_uids:
        target_obj = env.sim.get_rigid_object(obj_name)
    else:
        log_error("No matched object.")

    # Get object base pose (4x4 matrix)
    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)

    # Construct target pose (preserve orientation)
    move_target_pose = deepcopy(select_arm_current_pose)
    move_target_pose[:3, 3] = target_obj_pose[:3, 3]
    move_target_pose[0, 3] += x_offset
    move_target_pose[1, 3] += y_offset
    move_target_pose[2, 3] += z_offset

    # Solve IK for target pose
    move_target_pose, move_target_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        move_target_pose,
        select_arm_current_qpos,
        force_valid=force_valid,
        name="move relative to object",
    )

    # Update env states
    env.set_current_qpos_agent(move_target_qpos, is_left=is_left)
    env.set_current_xpos_agent(move_target_pose, is_left=is_left)

    # ------------------------------------ Traj 1: init → target ------------------------------------
    qpos_list_init_to_target = [select_arm_current_qpos, move_target_qpos]
    sample_num = kwargs.get("sample_num", 30)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_init_to_target,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for move relative to object: {len(actions)}.",
        color="green",
    )

    return actions


def move_to_absolute_position(
    robot_name: str,
    x: float = None,
    y: float = None,
    z: float = None,
    env=None,
    force_valid=False,
    **kwargs,
):

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    # ---------------------------------------- Pose ----------------------------------------
    # Start from current pose, then selectively update xyz
    move_pose = deepcopy(select_arm_current_pose)

    current_xyz = move_pose[:3, 3].clone()

    target_xyz = current_xyz.clone()
    if x is not None:
        target_xyz[0] = x
    if y is not None:
        target_xyz[1] = y
    if z is not None:
        target_xyz[2] = z

    move_pose[:3, 3] = target_xyz

    # Try IK on target pose
    move_pose, move_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        move_pose,
        select_arm_current_qpos,
        force_valid=force_valid,
        name="move to absolute position",
    )

    # Update env states
    env.set_current_qpos_agent(move_qpos, is_left=is_left)
    env.set_current_xpos_agent(move_pose, is_left=is_left)

    # ------------------------------------ Traj: init → target ------------------------------------
    qpos_list_init_to_move = [select_arm_current_qpos, move_qpos]
    sample_num = kwargs.get("sample_num", 30)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_init_to_move,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for move to absolute position: {len(actions)}.",
        color="green",
    )

    return actions


def move_by_relative_offset(
    robot_name: str,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    mode: str = "extrinsic",
    env=None,
    force_valid=False,
    **kwargs,
):

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    # ---------------------------------------- Pose ----------------------------------------
    move_pose = deepcopy(select_arm_current_pose)

    # Apply relative offsets (dx, dy, dz always floats)
    move_pose = get_offset_pose(move_pose, dx, "x", mode)
    move_pose = get_offset_pose(move_pose, dy, "y", mode)
    move_pose = get_offset_pose(move_pose, dz, "z", mode)

    # Solve IK
    move_pose, move_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        move_pose,
        select_arm_current_qpos,
        force_valid=force_valid,
        name="move by relative offset",
    )

    # Update environment states
    env.set_current_qpos_agent(move_qpos, is_left=is_left)
    env.set_current_xpos_agent(move_pose, is_left=is_left)

    # ------------------------------------ Traj: init → target ------------------------------------
    qpos_list_init_to_move = [select_arm_current_qpos, move_qpos]
    sample_num = kwargs.get("sample_num", 20)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_init_to_move,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for move by relative offset: {len(actions)}.",
        color="green",
    )

    return actions


def back_to_initial_pose(robot_name: str, env=None, **kwargs):

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    # Get arm states
    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    # Retrieve the initial joint configuration of this arm
    target_qpos = env.left_arm_init_qpos if is_left else env.right_arm_init_qpos
    target_qpos = torch.as_tensor(target_qpos, dtype=select_arm_current_qpos.dtype)

    # ---------------------------------------- Pose ----------------------------------------
    # Pre-back pose: move along tool z by a small offset (use intrinsic frame)
    pre_back_pose = deepcopy(select_arm_current_pose)
    pre_back_pose = get_offset_pose(pre_back_pose, -0.08, "z", "intrinsic")

    # IK for pre-back
    pre_back_pose, pre_back_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        pre_back_pose,
        select_arm_current_qpos,
        force_valid=kwargs.get("force_valid", False),
        name="pre back pose",
    )

    # Update env states (move to target pose)
    target_pose = env.get_arm_fk(qpos=target_qpos, is_left=is_left)
    env.set_current_qpos_agent(target_qpos, is_left=is_left)
    env.set_current_xpos_agent(target_pose, is_left=is_left)

    # ------------------------------------ Traj: init → pre back_pose ------------------------------------
    qpos_list_init_to_preback = [select_arm_current_qpos, pre_back_qpos]
    sample_num = 20

    plan_trajectory(
        env,
        select_arm,
        qpos_list_init_to_preback,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ------------------------------------ Traj: init → initial_pose ------------------------------------
    qpos_list_preback_to_target = [pre_back_qpos, target_qpos]
    sample_num = kwargs.get("sample_num", 30)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_preback_to_target,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for back to initial pose: {len(actions)}.",
        color="green",
    )

    return actions


def rotate_eef(robot_name: str, degree: float = 0, env=None, **kwargs):

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    # ---------------------------------------- Pose ----------------------------------------
    # Compute new joint positions
    rotated_qpos = deepcopy(select_arm_current_qpos)
    rotated_qpos[5] += np.deg2rad(degree)

    # Optional: limit checking (commented out by default)
    # joint5_limit = env.get_joint_limits(select_arm)[5]
    # if rotated_qpos[5] < joint5_limit[0] or rotated_qpos[5] > joint5_limit[1]:
    #     log_warning("Rotated qpos exceeds joint limits.\n")

    # Compute FK for new pose
    rotated_pose = env.get_arm_fk(
        qpos=rotated_qpos,
        is_left=is_left,
    )

    # Update environment state
    env.set_current_qpos_agent(rotated_qpos, is_left=is_left)
    env.set_current_xpos_agent(rotated_pose, is_left=is_left)

    # ------------------------------------ Traj 1: init → rotated ------------------------------------
    qpos_list_init_to_rotated = [select_arm_current_qpos, rotated_qpos]
    sample_num = kwargs.get("sample_num", 20)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_init_to_rotated,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for rotate eef: {len(actions)}.",
        color="green",
    )

    return actions


def orient_eef(
    robot_name: str,
    direction: str = "front",  # 'front' or 'down'
    env=None,
    force_valid=False,
    **kwargs,
):

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    # Get arm state
    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    # ---------------------------------------- Pose ----------------------------------------
    # Generate replacement rotation matrix
    replaced_rotation_matrix = np.eye(4)
    if direction == "front":
        rotation_matrix = R.from_euler("xyz", [180, -90, 0], degrees=True).as_matrix()
        replaced_rotation_matrix[:3, :3] = (
            rotation_matrix @ replaced_rotation_matrix[:3, :3]
        )
    elif direction == "down":
        rotation_matrix = R.from_euler("x", 180, degrees=True).as_matrix()
        replaced_rotation_matrix[:3, :3] = (
            rotation_matrix @ replaced_rotation_matrix[:3, :3]
        )
    else:
        log_error("Rotation direction must be 'front' or 'down'.")

    rotation_replaced_pose = deepcopy(select_arm_current_pose)
    rot_torch = torch.as_tensor(
        replaced_rotation_matrix[:3, :3],
        dtype=rotation_replaced_pose.dtype,
        device=rotation_replaced_pose.device,
    )
    rotation_replaced_pose[:3, :3] = rot_torch

    # Solve IK for the new pose
    rotation_replaced_pose, replace_target_qpos = get_qpos(
        env,
        is_left,
        select_arm,
        rotation_replaced_pose,
        select_arm_current_qpos,
        force_valid=force_valid,
        name="replaced-rotation",
    )

    # ---------------------------------------- Update env ----------------------------------------
    env.set_current_qpos_agent(replace_target_qpos, is_left=is_left)
    env.set_current_xpos_agent(rotation_replaced_pose, is_left=is_left)

    # ------------------------------------ Traj: init → target ------------------------------------
    qpos_list_init_to_rotated = [select_arm_current_qpos, replace_target_qpos]
    sample_num = kwargs.get("sample_num", 20)

    plan_trajectory(
        env,
        select_arm,
        qpos_list_init_to_rotated,
        sample_num,
        select_arm_current_gripper_state,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for orient eef: {len(actions)}.",
        color="green",
    )

    return actions


def close_gripper(robot_name: str, env=None, **kwargs):

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    # ---------------------------------------- Traj ----------------------------------------
    sample_num = kwargs.get("sample_num", 15)
    execute_open = False  # False → closing motion

    plan_gripper_trajectory(
        env,
        is_left,
        sample_num,
        execute_open,
        select_arm_current_qpos,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for close gripper: {len(actions)}.",
        color="green",
    )

    return actions


def open_gripper(robot_name: str, env=None, **kwargs):

    # ---------------------------------------- Prepare ----------------------------------------
    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    if select_arm_current_gripper_state >= (env.open_state - kwargs.get('open_threshold', 0.01)):
        actions = finalize_actions(
            [select_arm_current_qpos],
            [select_arm_current_gripper_state],
        )
        log_info(
            "Skip open gripper because current gripper state already satisfies the skip condition.",
            color="green",
        )
        return actions

    # ---------------------------------------- Traj ----------------------------------------
    sample_num = kwargs.get("sample_num", 15)
    execute_open = True  # True → opening motion

    plan_gripper_trajectory(
        env,
        is_left,
        sample_num,
        execute_open,
        select_arm_current_qpos,
        select_qpos_traj,
        ee_state_list_select,
    )

    # ---------------------------------------- Final ----------------------------------------
    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for open gripper: {len(actions)}.",
        color="green",
    )

    return actions


OBJECT_ERROR_TRIGGER_PROBABILITY = 0.1
ACTION_ERROR_TRIGGER_PROBABILITY = 0


def drive(
    left_arm_action=None,
    right_arm_action=None,
    error_functions=None,
    monitor_sequences=None,
    recovery_sequences=None,
    env=None,
    **kwargs,
):

    action_error_functions = []
    object_error_functions = []

    left_arm_action = resolve_action(left_arm_action, env, kwargs)
    right_arm_action = resolve_action(right_arm_action, env, kwargs)

    if error_functions is not None:
        for error_function in error_functions:
            error_type = error_function.keywords.get("error_type", None)
            if error_type in object_error_types:
                object_error_functions.append(error_function)
            else:
                action_error_functions.append(error_function)

    for error_function in action_error_functions:
        probability = get_error_probability(
            error_function, ACTION_ERROR_TRIGGER_PROBABILITY
        )
        if np.random.rand() > probability:
            continue

        left_arm_action, right_arm_action = error_function(
            left_arm_action=left_arm_action,
            right_arm_action=right_arm_action,
            env=env,
            **kwargs,
        )

    original_left_arm_action_provided = left_arm_action is not None
    original_right_arm_action_provided = right_arm_action is not None

    if left_arm_action is not None and right_arm_action is not None:
        len_left = len(left_arm_action)
        len_right = len(right_arm_action)

        if len_left < len_right:
            diff = len_right - len_left
            padding = np.repeat(left_arm_action[-1:], diff, axis=0)
            left_arm_action = np.concatenate([left_arm_action, padding], axis=0)
        elif len_right < len_left:
            diff = len_left - len_right
            padding = np.repeat(right_arm_action[-1:], diff, axis=0)
            right_arm_action = np.concatenate([right_arm_action, padding], axis=0)

        left_arm_index = env.left_arm_joints + env.left_eef_joints
        right_arm_index = env.right_arm_joints + env.right_eef_joints
        actions = np.zeros((len(right_arm_action), len(env.init_qpos)))
        actions[:, left_arm_index] = left_arm_action
        actions[:, right_arm_index] = right_arm_action

    elif left_arm_action is None and right_arm_action is not None:
        left_arm_index = env.left_arm_joints + env.left_eef_joints
        right_arm_index = env.right_arm_joints + env.right_eef_joints
        left_arm_action = finalize_actions(
            env.left_arm_current_qpos, env.left_arm_current_gripper_state
        )
        left_arm_action = np.repeat(
            left_arm_action[None, :], len(right_arm_action), axis=0
        )

        actions = np.zeros(
            (len(right_arm_action), len(env.robot.get_qpos().squeeze(0))),
            dtype=np.float32,
        )
        actions[:, left_arm_index] = left_arm_action
        actions[:, right_arm_index] = right_arm_action

    elif right_arm_action is None and left_arm_action is not None:
        left_arm_index = env.left_arm_joints + env.left_eef_joints
        right_arm_index = env.right_arm_joints + env.right_eef_joints
        right_arm_action = finalize_actions(
            env.right_arm_current_qpos, env.right_arm_current_gripper_state
        )
        right_arm_action = np.repeat(
            right_arm_action[None, :], len(left_arm_action), axis=0
        )

        actions = np.zeros(
            (len(left_arm_action), len(env.robot.get_qpos().squeeze(0))),
            dtype=np.float32,
        )
        actions[:, left_arm_index] = left_arm_action
        actions[:, right_arm_index] = right_arm_action

    else:
        log_error("At least one arm action should be provided.")

    actions = torch.from_numpy(actions).to(dtype=torch.float32).unsqueeze(1)
    actions = list(actions.unbind(dim=0))

    for i in tqdm(range(len(actions))):
        action = actions[i]

        env.step(action)
        env.update_obj_info()

        for error_function in object_error_functions:
            probability = get_error_probability(
                error_function, OBJECT_ERROR_TRIGGER_PROBABILITY
            )
            if np.random.rand() > probability:
                continue

            error_function(actions=None, env=env, **kwargs)
            error_function_name = getattr(
                error_function.func, "__name__", error_function.__class__.__name__
            )
            print(f"\033[91mError function {error_function_name} triggered at step {i}.\033[0m")

        if monitor_sequences is not None:
            for monitor_idx, monitor_sequence in enumerate(monitor_sequences):
                for function in monitor_sequence:
                    result = function()
                    if result == True:
                        function_name = getattr(
                            function.func, "__name__", function.__class__.__name__
                        )
                        log_warning(
                            f"Monitor function {function_name} triggered at step {i}."
                        )

                        if recovery_sequences is not None and monitor_idx < len(
                            recovery_sequences
                        ):
                            recovery_sequence = recovery_sequences[monitor_idx]
                            if recovery_sequence is not None:
                                remaining_left_arm_action = None
                                remaining_right_arm_action = None
                                if (
                                    original_left_arm_action_provided
                                    and left_arm_action is not None
                                    and i + 1 < len(left_arm_action)
                                ):
                                    remaining_left_arm_action = left_arm_action[i + 1 :]
                                if (
                                    original_right_arm_action_provided
                                    and right_arm_action is not None
                                    and i + 1 < len(right_arm_action)
                                ):
                                    remaining_right_arm_action = right_arm_action[i + 1 :]

                                for recovery_step in recovery_sequence:

                                    # update info
                                    # TODO: do not set gripper state, even if when regrasping, it will first close
                                    action = env.robot.get_qpos().squeeze(0)
                                    env.left_arm_current_qpos = action[
                                        env.left_arm_joints
                                    ]
                                    env.left_arm_current_xpos = env.robot.compute_fk(
                                        qpos=env.left_arm_current_qpos,
                                        name="left_arm",
                                        to_matrix=True,
                                    ).squeeze(0)
                                    # env.left_arm_current_gripper_state = action[
                                    #     env.left_eef_joints
                                    # ][0].unsqueeze(0)

                                    env.right_arm_current_qpos = action[
                                        env.right_arm_joints
                                    ]
                                    env.right_arm_current_xpos = env.robot.compute_fk(
                                        qpos=env.right_arm_current_qpos,
                                        name="right_arm",
                                        to_matrix=True,
                                    ).squeeze(0)
                                    # env.right_arm_current_gripper_state = action[
                                    #     env.right_eef_joints
                                    # ][0].unsqueeze(0)

                                    recovery_step_keywords = getattr(
                                        recovery_step, "keywords", {}
                                    )
                                    recovery_step_kwargs = dict(kwargs)
                                    if (
                                        recovery_step_keywords.get(
                                            "left_arm_action", "__missing__"
                                        )
                                        is None
                                        and remaining_left_arm_action is not None
                                    ):
                                        recovery_step_kwargs["left_arm_action"] = (
                                            remaining_left_arm_action
                                        )
                                        remaining_left_arm_action = None
                                    if (
                                        recovery_step_keywords.get(
                                            "right_arm_action", "__missing__"
                                        )
                                        is None
                                        and remaining_right_arm_action is not None
                                    ):
                                        recovery_step_kwargs["right_arm_action"] = (
                                            remaining_right_arm_action
                                        )
                                        remaining_right_arm_action = None

                                    recovery_step(env=env, **recovery_step_kwargs)

                        return actions

    if monitor_sequences is not None:
        log_info("No monitor sequences triggered during execution.")
    return actions
