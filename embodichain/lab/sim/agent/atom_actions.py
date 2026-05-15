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

from __future__ import annotations

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
    sync_agent_state_from_robot,
)
from embodichain.lab.sim.agent.atomic_action_adapter import (
    build_public_upright_place_pose_candidates,
    public_atomic_actions_enabled,
    register_pending_public_place_validation,
    try_public_gripper_action,
    try_public_grasp_action,
    try_public_move_action,
    try_public_place_action,
    validate_pending_public_grasp_after_action,
    validate_pending_public_place_after_action,
    validate_public_place_preconditions,
)
from embodichain.lab.sim.agent.error_functions import (
    fallen_object,
    inject_forced_recovery_error,
    inject_interactive_error,
    interactive_error_requested,
    misplaced_object,
    restore_interactive_error_input,
    setup_interactive_error_input,
)
from embodichain.lab.sim.agent.monitor_functions import *

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
    **kwargs,
):
    public_actions = try_public_move_action(
        env=env,
        robot_name=robot_name,
        target_pose=target_pose,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name=kwargs.get("name", "move to target pose"),
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for move to target: {len(public_actions)}.",
            color="green",
        )
        return public_actions

    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

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
        ee_state_list_select,
    )

    actions = finalize_actions(select_qpos_traj, ee_state_list_select)

    log_info(
        f"Total generated trajectory number for move to target: {len(actions)}.",
        color="green",
    )

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
    current_gripper_state = torch.as_tensor(
        select_arm_current_gripper_state,
        dtype=env.open_state.dtype,
        device=env.open_state.device,
    )
    if torch.all(current_gripper_state <= env.open_state - 0.01).item():
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

    use_public_semantic_grasp = bool(
        kwargs.get("use_public_grasp_semantics", False)
        or kwargs.get("public_grasp_strategy") is not None
    )
    if (
        public_atomic_actions_enabled(kwargs)
        and not use_public_semantic_grasp
        and bool(
            kwargs.get("use_public_grasp_action", False)
            or kwargs.get("require_public_grasp_action", False)
        )
    ):
        (
            is_left,
            select_arm,
            select_arm_current_qpos,
            select_arm_current_pose,
            select_arm_current_gripper_state,
        ) = get_arm_states(env, robot_name)
        delta_xy = target_obj_pose[:2, 3] - select_arm_base_pose[:2, 3]
        aim_horizontal_angle = torch.atan2(delta_xy[1], delta_xy[0]).item()
        select_arm_aim_qpos = deepcopy(select_arm_current_qpos)
        select_arm_aim_qpos[0] = aim_horizontal_angle

        if not torch.allclose(
            torch.as_tensor(select_arm_current_qpos),
            torch.as_tensor(select_arm_aim_qpos),
            rtol=1e-5,
            atol=1e-8,
        ):
            aim_qpos_traj = []
            aim_gripper_traj = []
            plan_trajectory(
                env,
                select_arm,
                [select_arm_current_qpos, select_arm_aim_qpos],
                10,
                select_arm_current_gripper_state,
                aim_qpos_traj,
                aim_gripper_traj,
            )
            aim_actions = finalize_actions(aim_qpos_traj, aim_gripper_traj)
            actions = (
                aim_actions
                if actions is None
                else np.concatenate([actions, aim_actions], axis=0)
            )
            aim_pose = env.get_arm_fk(qpos=select_arm_aim_qpos, is_left=is_left)
            env.set_current_qpos_agent(select_arm_aim_qpos, is_left=is_left)
            env.set_current_xpos_agent(aim_pose, is_left=is_left)

    public_grasp_actions = try_public_grasp_action(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        pre_grasp_dis=pre_grasp_dis,
        kwargs=kwargs,
    )
    if public_grasp_actions is not None:
        actions = (
            public_grasp_actions
            if actions is None
            else np.concatenate([actions, public_grasp_actions], axis=0)
        )
        log_info(
            f"Total generated trajectory number for grasp: {len(actions)}.",
            color="green",
        )
        return actions

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
    aim_horizontal_angle = torch.atan2(dy, dx).item()
    select_arm_aim_qpos = deepcopy(select_arm_current_qpos)
    select_arm_aim_qpos[0] = aim_horizontal_angle

    # Get best grasp pose from affordance data
    grasp_pose_object = torch.as_tensor(
        env.obj_info.get(obj_name)["grasp_pose_obj"],
        dtype=target_obj_pose.dtype,
        device=target_obj_pose.device,
    )
    if (
        not kwargs.get("public_grasp_preserve_object_rotation", False)
        and grasp_pose_object[0, 2] > 0.5
    ):  # whether towards x direction TODO: make it robust
        # Align the object pose's z-axis with the arm's aiming direction
        target_obj_pose = torch.as_tensor(
            get_rotation_replaced_pose(
                target_obj_pose.detach().cpu().numpy(),
                float(select_arm_aim_qpos[0]),
                "z",
                "intrinsic",
            ),
            dtype=target_obj_pose.dtype,
            device=target_obj_pose.device,
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


def _plan_place_pose_fallback(
    *,
    env,
    robot_name: str,
    obj_name: str,
    is_left: bool,
    select_arm: str,
    current_qpos,
    current_gripper_state,
    base_place_pose,
    sample_num: int,
):
    """Plan a conservative place trajectory when exact table placement IK fails."""

    def _clone_pose_with_rotation(pose, rotation_matrix=None, dz: float = 0.0):
        candidate = deepcopy(pose)
        if rotation_matrix is not None:
            candidate[:3, :3] = torch.as_tensor(
                rotation_matrix,
                dtype=candidate.dtype,
                device=candidate.device,
            )
        candidate[2, 3] = candidate[2, 3] + dz
        return candidate

    rotation_candidates = [
        ("current_rotation", None),
        ("down_rotation", R.from_euler("x", 180, degrees=True).as_matrix()),
        (
            "front_rotation",
            R.from_euler("xyz", [180, -90, 0], degrees=True).as_matrix(),
        ),
    ]
    z_offsets = [0.03, 0.06, 0.1, 0.15]

    candidates = []
    for rotation_name, rotation_matrix in rotation_candidates:
        for dz in z_offsets:
            candidates.append(
                (
                    f"{rotation_name}, dz={dz:.2f}",
                    _clone_pose_with_rotation(
                        base_place_pose,
                        rotation_matrix=rotation_matrix,
                        dz=dz,
                    ),
                )
            )

    for label, candidate_pose in candidates:
        try:
            ret, qpos = env.get_arm_ik(
                candidate_pose,
                is_left=is_left,
                qpos_seed=current_qpos,
            )
        except Exception as exc:
            log_warning(
                f"place_on_table fallback candidate failed for "
                f"{robot_name}/{obj_name}: {label}. ({exc})"
            )
            continue
        if not ret:
            continue

        select_qpos_traj = []
        ee_state_list_select = []
        env.set_current_qpos_agent(qpos, is_left=is_left)
        env.set_current_xpos_agent(candidate_pose, is_left=is_left)
        plan_trajectory(
            env,
            select_arm,
            [current_qpos, qpos],
            sample_num,
            current_gripper_state,
            select_qpos_traj,
            ee_state_list_select,
        )
        xyz = (
            torch.as_tensor(candidate_pose[:3, 3])
            .detach()
            .cpu()
            .numpy()
            .round(4)
            .tolist()
        )
        log_warning(
            f"Using place_on_table fallback pose for {robot_name}/{obj_name}: "
            f"{label}, xyz={xyz}."
        )
        return finalize_actions(select_qpos_traj, ee_state_list_select)

    return None


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

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)
    place_pose = deepcopy(select_arm_current_pose)
    if x is not None:
        place_pose[0, 3] = x
    if y is not None:
        place_pose[1, 3] = y
    place_pose[2, 3] = height

    validate_public_place_preconditions(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        target_x=x,
        target_y=y,
        target_height=height,
        kwargs=kwargs,
    )

    obj_info = getattr(env, "obj_info", {}).get(obj_name, {})
    upright_base_height = kwargs.get(
        "public_place_upright_object_height",
        obj_info.get("place_object_height", init_obj_height),
    )
    upright_place_height = upright_base_height + kwargs.get(
        "public_place_upright_eps", 0.0
    )
    upright_place_candidates = build_public_upright_place_pose_candidates(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        target_pose=place_pose,
        x=x,
        y=y,
        object_height=upright_place_height,
        kwargs=kwargs,
    )
    place_candidates = upright_place_candidates or [("current_eef_pose", place_pose)]
    place_try_kwargs = kwargs
    if len(place_candidates) > 1 and kwargs.get("require_public_non_grasp_actions"):
        place_try_kwargs = dict(kwargs)
        place_try_kwargs["require_public_non_grasp_actions"] = False

    for place_label, candidate_place_pose in place_candidates:
        if place_label != "current_eef_pose":
            xyz = (
                candidate_place_pose[:3, 3]
                .detach()
                .cpu()
                .numpy()
                .round(4)
                .tolist()
            )
            log_info(
                f"Trying public upright place pose for {robot_name}/{obj_name}: "
                f"{place_label}, eef_xyz={xyz}.",
                color="cyan",
            )
        public_actions = try_public_place_action(
            env=env,
            robot_name=robot_name,
            target_pose=candidate_place_pose,
            pre_place_dis=pre_place_dis,
            kwargs=place_try_kwargs,
        )
        if public_actions is not None:
            register_pending_public_place_validation(
                env=env,
                kwargs=kwargs,
                robot_name=robot_name,
                obj_name=obj_name,
                target_x=x,
                target_y=y,
                target_height=upright_place_height,
                label=place_label,
            )
            log_info(
                f"Total generated trajectory number for place on table: "
                f"{len(public_actions)}.",
                color="green",
            )
            return public_actions

    if kwargs.get("require_public_non_grasp_actions"):
        raise RuntimeError(
            f"Public place_on_table failed for all candidates of "
            f"{robot_name}/{obj_name}."
        )

    legacy_kwargs = dict(kwargs)
    legacy_kwargs["use_public_atomic_actions"] = False
    try:
        traj_actions = move_to_absolute_position(
            robot_name,
            x=x,
            y=y,
            z=height,
            env=env,
            force_valid=force_valid,
            **legacy_kwargs,
        )
    except RuntimeError as exc:
        if force_valid:
            raise
        log_warning(
            f"Exact place_on_table IK failed for {robot_name}/{obj_name}; "
            f"retrying with nearest valid pose. ({exc})"
        )
        try:
            traj_actions = move_to_absolute_position(
                robot_name,
                x=x,
                y=y,
                z=height,
                env=env,
                force_valid=True,
                **legacy_kwargs,
            )
        except RuntimeError as force_valid_exc:
            log_warning(
                f"Nearest-valid place_on_table fallback failed for "
                f"{robot_name}/{obj_name}; trying local placement pose candidates. "
                f"({force_valid_exc})"
            )
            traj_actions = _plan_place_pose_fallback(
                env=env,
                robot_name=robot_name,
                obj_name=obj_name,
                is_left=is_left,
                select_arm=select_arm,
                current_qpos=select_arm_current_qpos,
                current_gripper_state=select_arm_current_gripper_state,
                base_place_pose=place_pose,
                sample_num=legacy_kwargs.get("sample_num", 30),
            )
            if traj_actions is None:
                raise force_valid_exc from exc
    open_actions = open_gripper(robot_name, env, **legacy_kwargs)

    actions = np.concatenate([traj_actions, open_actions], axis=0)
    register_pending_public_place_validation(
        env=env,
        kwargs=kwargs,
        robot_name=robot_name,
        obj_name=obj_name,
        target_x=x,
        target_y=y,
        target_height=height,
        label="legacy_place_on_table",
    )

    log_info(
        f"Total generated trajectory number for place on table: {len(actions)}.",
        color="green",
    )

    return actions


def _plan_move_relative_orientation_fallback(
    *,
    env,
    robot_name: str,
    is_left: bool,
    select_arm: str,
    current_qpos,
    current_gripper_state,
    base_target_pose,
    sample_num: int,
):
    """Try alternate EEF orientations for move_relative_to_object target pose."""

    rotation_candidates = [
        ("down_rotation", R.from_euler("x", 180, degrees=True).as_matrix()),
        (
            "front_rotation",
            R.from_euler("xyz", [180, -90, 0], degrees=True).as_matrix(),
        ),
        (
            "back_rotation",
            R.from_euler("xyz", [180, 90, 0], degrees=True).as_matrix(),
        ),
        ("current_rotation", None),
    ]
    z_offsets = [0.0, 0.03, -0.03, 0.06]

    for rotation_name, rotation_matrix in rotation_candidates:
        for dz in z_offsets:
            candidate_pose = deepcopy(base_target_pose)
            if rotation_matrix is not None:
                candidate_pose[:3, :3] = torch.as_tensor(
                    rotation_matrix,
                    dtype=candidate_pose.dtype,
                    device=candidate_pose.device,
                )
            candidate_pose[2, 3] = candidate_pose[2, 3] + dz
            try:
                ret, qpos = env.get_arm_ik(
                    candidate_pose,
                    is_left=is_left,
                    qpos_seed=current_qpos,
                )
            except Exception as exc:
                log_warning(
                    f"move_relative_to_object orientation fallback failed for "
                    f"{robot_name}: {rotation_name}, dz={dz:.2f}. ({exc})"
                )
                continue
            if not ret:
                continue

            select_qpos_traj = []
            ee_state_list_select = []
            env.set_current_qpos_agent(qpos, is_left=is_left)
            env.set_current_xpos_agent(candidate_pose, is_left=is_left)
            plan_trajectory(
                env,
                select_arm,
                [current_qpos, qpos],
                sample_num,
                current_gripper_state,
                select_qpos_traj,
                ee_state_list_select,
            )
            xyz = (
                torch.as_tensor(candidate_pose[:3, 3])
                .detach()
                .cpu()
                .numpy()
                .round(4)
                .tolist()
            )
            log_warning(
                f"Using move_relative_to_object orientation fallback for "
                f"{robot_name}: {rotation_name}, dz={dz:.2f}, xyz={xyz}."
            )
            return finalize_actions(select_qpos_traj, ee_state_list_select)

    return None


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
    sample_num = kwargs.get("sample_num", 30)

    public_actions = try_public_move_action(
        env=env,
        robot_name=robot_name,
        target_pose=move_target_pose,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="move relative to object",
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for move relative to object: {len(public_actions)}.",
            color="green",
        )
        return public_actions

    # Solve IK for target pose
    try:
        move_target_pose, move_target_qpos = get_qpos(
            env,
            is_left,
            select_arm,
            move_target_pose,
            select_arm_current_qpos,
            force_valid=force_valid,
            name="move relative to object",
        )
    except RuntimeError as exc:
        if not kwargs.get("allow_move_relative_orientation_fallback", False):
            raise
        fallback_actions = _plan_move_relative_orientation_fallback(
            env=env,
            robot_name=robot_name,
            is_left=is_left,
            select_arm=select_arm,
            current_qpos=select_arm_current_qpos,
            current_gripper_state=select_arm_current_gripper_state,
            base_target_pose=move_target_pose,
            sample_num=sample_num,
        )
        if fallback_actions is None:
            raise
        log_info(
            f"Total generated trajectory number for move relative to object: {len(fallback_actions)}.",
            color="green",
        )
        return fallback_actions

    # Update env states
    env.set_current_qpos_agent(move_target_qpos, is_left=is_left)
    env.set_current_xpos_agent(move_target_pose, is_left=is_left)

    # ------------------------------------ Traj 1: init → target ------------------------------------
    qpos_list_init_to_target = [select_arm_current_qpos, move_target_qpos]

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
    sample_num = kwargs.get("sample_num", 30)

    public_actions = try_public_move_action(
        env=env,
        robot_name=robot_name,
        target_pose=move_pose,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="move to absolute position",
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for move to absolute position: {len(public_actions)}.",
            color="green",
        )
        return public_actions

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
    sample_num = kwargs.get("sample_num", 20)

    public_actions = try_public_move_action(
        env=env,
        robot_name=robot_name,
        target_pose=move_pose,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="move by relative offset",
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for move by relative offset: {len(public_actions)}.",
            color="green",
        )
        return public_actions

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

    target_pose = env.get_arm_fk(qpos=target_qpos, is_left=is_left)
    sample_num = kwargs.get("sample_num", 30)
    public_actions = try_public_move_action(
        env=env,
        robot_name=robot_name,
        target_pose=target_pose,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="back to initial pose",
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for back to initial pose: {len(public_actions)}.",
            color="green",
        )
        return public_actions

    # IK for pre-back. This waypoint is only a clearance motion; if it is not
    # reachable, keep the return-to-initial action valid by moving directly back.
    skip_pre_back = False
    try:
        pre_back_pose, pre_back_qpos = get_qpos(
            env,
            is_left,
            select_arm,
            pre_back_pose,
            select_arm_current_qpos,
            force_valid=kwargs.get("force_valid", False),
            name="pre back pose",
        )
    except RuntimeError as exc:
        log_warning(
            f"Pre back pose IK failed for {robot_name}; moving directly to initial pose. ({exc})"
        )
        pre_back_pose = select_arm_current_pose
        pre_back_qpos = select_arm_current_qpos
        skip_pre_back = True

    # Update env states (move to target pose)
    env.set_current_qpos_agent(target_qpos, is_left=is_left)
    env.set_current_xpos_agent(target_pose, is_left=is_left)

    # ------------------------------------ Traj: init → pre back_pose ------------------------------------
    if not skip_pre_back:
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

    sample_num = kwargs.get("sample_num", 20)
    public_actions = try_public_move_action(
        env=env,
        robot_name=robot_name,
        target_pose=rotated_pose,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="rotate eef",
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for rotate eef: {len(public_actions)}.",
            color="green",
        )
        return public_actions

    # Update environment state
    env.set_current_qpos_agent(rotated_qpos, is_left=is_left)
    env.set_current_xpos_agent(rotated_pose, is_left=is_left)

    # ------------------------------------ Traj 1: init → rotated ------------------------------------
    qpos_list_init_to_rotated = [select_arm_current_qpos, rotated_qpos]

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

    sample_num = kwargs.get("sample_num", 20)
    public_actions = try_public_move_action(
        env=env,
        robot_name=robot_name,
        target_pose=rotation_replaced_pose,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="orient eef",
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for orient eef: {len(public_actions)}.",
            color="green",
        )
        return public_actions

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

    public_actions = try_public_gripper_action(
        env=env,
        robot_name=robot_name,
        target_state=env.close_state,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="close_gripper",
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for close gripper: {len(public_actions)}.",
            color="green",
        )
        return public_actions

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

    current_gripper_state = torch.as_tensor(
        select_arm_current_gripper_state,
        dtype=env.open_state.dtype,
        device=env.open_state.device,
    )
    if (
        not kwargs.get("require_public_non_grasp_actions", False)
        and torch.all(
            current_gripper_state
            >= (env.open_state - kwargs.get("open_threshold", 0.01))
        ).item()
    ):
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

    public_actions = try_public_gripper_action(
        env=env,
        robot_name=robot_name,
        target_state=env.open_state,
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="open_gripper",
    )
    if public_actions is not None:
        log_info(
            f"Total generated trajectory number for open gripper: {len(public_actions)}.",
            color="green",
        )
        return public_actions

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


def _normalize_vector(vector, *, eps: float = 1e-6):
    vector = torch.as_tensor(vector, dtype=torch.float32)
    norm = torch.linalg.norm(vector)
    if norm <= eps:
        return None
    return vector / norm


def _object_world_bounds(env, obj_name: str, obj_pose: torch.Tensor):
    target_obj = env.sim.get_rigid_object(obj_name)
    try:
        vertices = target_obj.get_vertices(env_ids=[0], scale=True)[0]
    except Exception:
        return None

    vertices = torch.as_tensor(
        vertices,
        dtype=obj_pose.dtype,
        device=obj_pose.device,
    )
    if vertices.numel() == 0:
        return None

    world_vertices = vertices @ obj_pose[:3, :3].transpose(0, 1)
    world_vertices = world_vertices + obj_pose[:3, 3]
    mins = world_vertices.min(dim=0).values
    maxs = world_vertices.max(dim=0).values
    extents = maxs - mins
    return {
        "center": (mins + maxs) * 0.5,
        "mins": mins,
        "maxs": maxs,
        "extents": extents,
        "vertices": world_vertices,
    }


def _rotation_from_approach_and_roll(
    approach_axis,
    roll_axis_seed,
    roll_rad: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    z_axis = _normalize_vector(approach_axis)
    if z_axis is None:
        return None
    z_axis = z_axis.to(dtype=dtype, device=device)

    x_axis = torch.as_tensor(roll_axis_seed, dtype=dtype, device=device)
    x_axis = x_axis - torch.dot(x_axis, z_axis) * z_axis
    if torch.linalg.norm(x_axis) <= 1e-6:
        x_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        x_axis = x_axis - torch.dot(x_axis, z_axis) * z_axis
    if torch.linalg.norm(x_axis) <= 1e-6:
        x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device)
        x_axis = x_axis - torch.dot(x_axis, z_axis) * z_axis
    x_axis = x_axis / torch.linalg.norm(x_axis).clamp_min(1e-6)
    y_axis = torch.cross(z_axis, x_axis, dim=0)
    y_axis = y_axis / torch.linalg.norm(y_axis).clamp_min(1e-6)

    if abs(roll_rad) > 1e-8:
        cos_value = float(np.cos(roll_rad))
        sin_value = float(np.sin(roll_rad))
        x_axis, y_axis = (
            cos_value * x_axis + sin_value * y_axis,
            -sin_value * x_axis + cos_value * y_axis,
        )

    return torch.stack([x_axis, y_axis, z_axis], dim=1)


def _upright_object_rotation_from_pose(obj_pose: torch.Tensor) -> torch.Tensor:
    world_z = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=obj_pose.dtype,
        device=obj_pose.device,
    )
    x_axis = obj_pose[:3, 0].detach().clone()
    x_axis[2] = 0.0
    if torch.linalg.norm(x_axis) <= 1e-6:
        x_axis = obj_pose[:3, 1].detach().clone()
        x_axis[2] = 0.0
    if torch.linalg.norm(x_axis) <= 1e-6:
        x_axis = torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=obj_pose.dtype,
            device=obj_pose.device,
        )
    x_axis = x_axis / torch.linalg.norm(x_axis).clamp_min(1e-6)
    y_axis = torch.cross(world_z, x_axis, dim=0)
    y_axis = y_axis / torch.linalg.norm(y_axis).clamp_min(1e-6)
    return torch.stack([x_axis, y_axis, world_z], dim=1)


def _aim_arm_base_at_xy(
    *,
    env,
    robot_name: str,
    target_x: float,
    target_y: float,
    kwargs: dict,
):
    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)
    arm_base_pose = env.left_arm_base_pose if is_left else env.right_arm_base_pose
    delta_xy = torch.tensor(
        [
            float(target_x) - float(arm_base_pose[0, 3]),
            float(target_y) - float(arm_base_pose[1, 3]),
        ],
        dtype=torch.float32,
    )
    if torch.linalg.norm(delta_xy) <= 1e-6:
        return None

    aim_qpos = deepcopy(select_arm_current_qpos)
    aim_qpos[0] = torch.atan2(delta_xy[1], delta_xy[0]).item()
    if torch.allclose(
        torch.as_tensor(select_arm_current_qpos),
        torch.as_tensor(aim_qpos),
        rtol=1e-5,
        atol=1e-8,
    ):
        return None

    qpos_traj = []
    gripper_traj = []
    plan_trajectory(
        env,
        select_arm,
        [select_arm_current_qpos, aim_qpos],
        int(kwargs.get("upright_object_aim_sample_num", 10)),
        select_arm_current_gripper_state,
        qpos_traj,
        gripper_traj,
    )
    aim_actions = finalize_actions(qpos_traj, gripper_traj)
    aim_pose = env.get_arm_fk(qpos=aim_qpos, is_left=is_left)
    env.set_current_qpos_agent(aim_qpos, is_left=is_left)
    env.set_current_xpos_agent(aim_pose, is_left=is_left)
    return aim_actions


def _fallen_object_grasp_candidates(
    *,
    env,
    robot_name: str,
    obj_name: str,
    obj_pose: torch.Tensor,
    bounds: dict | None,
    is_left: bool,
    kwargs: dict,
):
    device = obj_pose.device
    dtype = obj_pose.dtype
    base_pose = env.left_arm_base_pose if is_left else env.right_arm_base_pose
    center = bounds["center"] if bounds is not None else obj_pose[:3, 3]
    extents = bounds["extents"] if bounds is not None else torch.tensor(
        [0.08, 0.08, 0.08],
        dtype=dtype,
        device=device,
    )

    arm_to_obj = center[:2] - base_pose[:2, 3].to(device=device, dtype=dtype)
    arm_to_obj_3d = torch.tensor(
        [arm_to_obj[0], arm_to_obj[1], 0.0],
        dtype=dtype,
        device=device,
    )
    local_z_world = obj_pose[:3, 2].detach().clone()
    local_z_world[2] = 0.0
    long_axis = _normalize_vector(local_z_world)
    if long_axis is not None:
        long_axis = long_axis.to(dtype=dtype, device=device)
    arm_axis = _normalize_vector(arm_to_obj_3d)
    if arm_axis is not None:
        arm_axis = arm_axis.to(dtype=dtype, device=device)

    approach_axes = [
        (
            "top_down",
            torch.tensor([0.0, 0.0, -1.0], dtype=dtype, device=device),
        )
    ]
    if arm_axis is not None:
        approach_axes.append(("arm_to_object", arm_axis))
    if long_axis is not None:
        lateral = torch.tensor(
            [-long_axis[1], long_axis[0], 0.0],
            dtype=dtype,
            device=device,
        )
        for sign in (1.0, -1.0):
            approach_axes.append((f"object_lateral_{sign:+.0f}", lateral * sign))
    if arm_axis is not None:
        approach_axes.append(("object_to_arm", -arm_axis))

    roll_seeds = []
    if long_axis is not None:
        roll_seeds.append(("object_long_axis", long_axis))
    roll_seeds.append(
        ("world_z", torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device))
    )

    roll_offsets = kwargs.get(
        "upright_object_roll_offsets",
        [0.0, np.pi / 2.0, -np.pi / 2.0, np.pi],
    )
    if isinstance(roll_offsets, str):
        roll_offsets = [float(item) for item in roll_offsets.split(",") if item]

    vertices = bounds.get("vertices") if bounds is not None else None
    table_clearance = float(kwargs.get("upright_object_table_clearance", 0.035))
    side_depth = float(
        kwargs.get(
            "upright_object_side_grasp_depth",
            max(0.0, float(torch.clamp(extents[:2].max() * 0.15, min=0.0).item())),
        )
    )
    grasp_z_offset = float(kwargs.get("upright_object_grasp_z_offset", 0.015))

    lever_points = []
    if long_axis is not None:
        if vertices is not None and vertices.numel() > 0:
            projections = (vertices - center) @ long_axis
            half_axis_extent = float(torch.max(torch.abs(projections)).item())
        else:
            half_axis_extent = float(torch.max(extents[:2]).item()) * 0.5
        axis_fractions = kwargs.get(
            "upright_object_long_axis_grasp_fractions",
            [0.45, -0.45, 0.30, -0.30],
        )
        if isinstance(axis_fractions, str):
            axis_fractions = [
                float(item) for item in axis_fractions.split(",") if item
            ]
        for fraction in axis_fractions:
            if abs(float(fraction)) <= 1e-8:
                continue
            point = center + long_axis * (half_axis_extent * float(fraction))
            lever_points.append((f"lever_end_{float(fraction):+.2f}", point))

    center_points = lever_points
    if vertices is not None and vertices.numel() > 0:
        top_fraction = float(kwargs.get("upright_object_high_surface_fraction", 0.30))
        top_count = max(1, int(round(vertices.shape[0] * top_fraction)))
        top_indices = torch.topk(vertices[:, 2], k=top_count).indices
        center_points.append(("high_surface", vertices[top_indices].mean(dim=0)))
    center_points.append(("center", center))

    grasp_z_values = []
    if bounds is not None:
        min_z = float(bounds["mins"][2].item())
        max_z = float(bounds["maxs"][2].item())
        z_ratios = kwargs.get(
            "upright_object_grasp_z_ratios",
            [0.50, 0.65, 0.35, 0.80],
        )
        if isinstance(z_ratios, str):
            z_ratios = [float(item) for item in z_ratios.split(",") if item]
        for ratio in z_ratios:
            z_value = min_z + (max_z - min_z) * float(ratio) + grasp_z_offset
            z_value = max(z_value, min_z + table_clearance)
            grasp_z_values.append((f"z_ratio={float(ratio):.2f}", z_value))
    else:
        grasp_z_values.append(("center_z", float(center[2].item()) + grasp_z_offset))

    candidates = []
    grasp_pose_object = getattr(env, "obj_info", {}).get(obj_name, {}).get(
        "grasp_pose_obj"
    )
    if grasp_pose_object is not None:
        grasp_relation = torch.as_tensor(
            grasp_pose_object,
            dtype=dtype,
            device=device,
        )
        candidates.append(("object_grasp_relation", obj_pose @ grasp_relation))

    seen = set()
    for center_label, grasp_center in center_points:
        grasp_center = torch.as_tensor(
            grasp_center,
            dtype=dtype,
            device=device,
        )
        if bounds is not None:
            grasp_center = torch.minimum(
                torch.maximum(grasp_center, bounds["mins"]),
                bounds["maxs"],
            )
        for approach_label, approach_axis in approach_axes:
            approach = _normalize_vector(approach_axis)
            if approach is None:
                continue
            approach = approach.to(dtype=dtype, device=device)
            for roll_label, roll_seed in roll_seeds:
                for roll_rad in roll_offsets:
                    rotation = _rotation_from_approach_and_roll(
                        approach,
                        roll_seed,
                        float(roll_rad),
                        dtype=dtype,
                        device=device,
                    )
                    if rotation is None:
                        continue
                    for z_label, grasp_z in grasp_z_values:
                        pose = torch.eye(4, dtype=dtype, device=device)
                        pose[:3, :3] = rotation
                        pose[:3, 3] = grasp_center - approach * side_depth
                        pose[2, 3] = float(grasp_z)
                        key = tuple(
                            torch.round(pose[:3, 3].detach().cpu() * 1000)
                            .int()
                            .tolist()
                            + torch.round(rotation[:, 2].detach().cpu() * 1000)
                            .int()
                            .tolist()
                            + [int(round(float(roll_rad) * 1000))]
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            (
                                f"{center_label}/{approach_label}/{roll_label}/"
                                f"roll={float(roll_rad):.2f}/{z_label}",
                                pose,
                            )
                        )

    return candidates


def _plan_fallen_object_top_down_grasp_place(
    *,
    env,
    robot_name: str,
    obj_name: str,
    obj_pose: torch.Tensor,
    bounds: dict | None,
    x: float | None,
    y: float | None,
    pre_grasp_dis: float,
    pre_place_dis: float,
    force_valid: bool,
    kwargs: dict,
):
    """Recover a fallen object with a vertical grasp, lift, upright release, and retreat."""

    if bounds is None:
        raise RuntimeError("top_down_grasp_place requires object mesh bounds.")

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        _,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    device = obj_pose.device
    dtype = obj_pose.dtype

    def _format_xyz(pose):
        return (
            torch.as_tensor(pose[:3, 3])
            .detach()
            .cpu()
            .numpy()
            .round(4)
            .tolist()
        )

    def _add_seed(seed_list, label: str, qpos) -> None:
        if qpos is None:
            seed_list.append((label, None))
            return
        qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32).detach().clone()
        for _, existing in seed_list:
            if existing is not None and torch.allclose(
                qpos_tensor,
                torch.as_tensor(existing, dtype=torch.float32),
                rtol=1e-4,
                atol=1e-5,
            ):
                return
        seed_list.append((label, qpos_tensor))

    def _solve_pose_ik(pose, seed_options, label: str):
        failures = []
        for seed_label, qpos_seed in seed_options:
            try:
                ret, qpos = env.get_arm_ik(
                    pose,
                    is_left=is_left,
                    qpos_seed=qpos_seed,
                )
            except Exception as exc:
                failures.append(f"{seed_label}: {type(exc).__name__}: {exc}")
                continue
            if ret:
                return qpos, seed_label, []
            failures.append(f"{seed_label}: no_ik")
        return None, None, [
            f"{label}: IK failed at xyz={_format_xyz(pose)} "
            f"with seeds=[{'; '.join(failures)}]"
        ]

    seed_candidates = []
    _add_seed(seed_candidates, "current", select_arm_current_qpos)
    init_qpos = env.left_arm_init_qpos if is_left else env.right_arm_init_qpos
    if init_qpos is not None:
        init_qpos = torch.as_tensor(init_qpos, dtype=torch.float32).detach().clone()
        _add_seed(seed_candidates, "init", init_qpos)
        aimed_init_qpos = init_qpos.detach().clone()
        aimed_init_qpos[0] = torch.as_tensor(select_arm_current_qpos)[0]
        _add_seed(seed_candidates, "aimed_init", aimed_init_qpos)
    _add_seed(seed_candidates, "none", None)

    center = bounds["center"].to(dtype=dtype, device=device)
    min_z = float(bounds["mins"][2].item())
    max_z = float(bounds["maxs"][2].item())
    vertices = bounds.get("vertices")

    local_z_world = obj_pose[:3, 2].detach().clone()
    local_z_world[2] = 0.0
    long_axis = _normalize_vector(local_z_world)
    if long_axis is not None:
        long_axis = long_axis.to(dtype=dtype, device=device)

    if long_axis is not None and vertices is not None and vertices.numel() > 0:
        projections = (vertices - center) @ long_axis
        half_axis_extent = float(torch.max(torch.abs(projections)).item())
    else:
        half_axis_extent = float(torch.max(bounds["extents"][:2]).item()) * 0.5

    target_x = float(center[0]) if x is None else float(x)
    target_y = float(center[1]) if y is None else float(y)
    target_z = float(
        kwargs.get(
            "upright_object_target_z",
            getattr(env, "obj_info", {}).get(obj_name, {}).get("height", obj_pose[2, 3]),
        )
    )

    initial_pose = getattr(env, "obj_info", {}).get(obj_name, {}).get("initial_pose")
    if initial_pose is not None:
        initial_pose = torch.as_tensor(initial_pose, dtype=dtype, device=device)
        base_upright_rotation = initial_pose[:3, :3]
    else:
        base_upright_rotation = _upright_object_rotation_from_pose(obj_pose)

    axis_fractions = kwargs.get(
        "upright_object_top_down_axis_fractions",
        [0.0, 0.18, -0.18, 0.32, -0.32],
    )
    if isinstance(axis_fractions, str):
        axis_fractions = [float(item) for item in axis_fractions.split(",") if item]

    z_ratios = kwargs.get(
        "upright_object_top_down_z_ratios",
        [0.55, 0.65, 0.45, 0.75, 0.35],
    )
    if isinstance(z_ratios, str):
        z_ratios = [float(item) for item in z_ratios.split(",") if item]

    pre_grasp_distances = kwargs.get(
        "upright_object_top_down_pre_grasp_distances",
        [pre_grasp_dis, 0.06, 0.03, 0.0],
    )
    if isinstance(pre_grasp_distances, str):
        pre_grasp_distances = [
            float(item) for item in pre_grasp_distances.split(",") if item
        ]

    yaw_offsets_deg = kwargs.get(
        "upright_object_release_yaw_offsets_deg",
        [0.0, 90.0, -90.0, 180.0],
    )
    if isinstance(yaw_offsets_deg, str):
        yaw_offsets_deg = [float(item) for item in yaw_offsets_deg.split(",") if item]

    release_z_offsets = kwargs.get(
        "upright_object_release_z_offsets",
        [0.0, 0.06, 0.12],
    )
    if isinstance(release_z_offsets, str):
        release_z_offsets = [float(item) for item in release_z_offsets.split(",") if item]

    release_pre_offsets = kwargs.get(
        "upright_object_release_pre_offsets",
        [0.0, pre_place_dis],
    )
    if isinstance(release_pre_offsets, str):
        release_pre_offsets = [
            float(item) for item in release_pre_offsets.split(",") if item
        ]

    lateral_axis = None
    if long_axis is not None:
        lateral_axis = torch.tensor(
            [-long_axis[1], long_axis[0], 0.0],
            dtype=dtype,
            device=device,
        )
        if torch.linalg.norm(lateral_axis) > 1e-6:
            lateral_axis = lateral_axis / torch.linalg.norm(lateral_axis)
        else:
            lateral_axis = None

    roll_seeds = []
    if long_axis is not None:
        roll_seeds.append(("x_long_axis", long_axis))
    if lateral_axis is not None:
        roll_seeds.append(("x_lateral_axis", lateral_axis))
    roll_seeds.append(
        ("x_world", torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device))
    )

    arm_base_pose = env.left_arm_base_pose if is_left else env.right_arm_base_pose
    arm_axis = torch.tensor(
        [
            float(center[0]) - float(arm_base_pose[0, 3]),
            float(center[1]) - float(arm_base_pose[1, 3]),
            0.0,
        ],
        dtype=dtype,
        device=device,
    )
    arm_axis = _normalize_vector(arm_axis)
    if arm_axis is not None:
        arm_axis = arm_axis.to(dtype=dtype, device=device)

    rotation_candidates = []
    top_down = torch.tensor([0.0, 0.0, -1.0], dtype=dtype, device=device)
    approach_candidates = [("top_down", top_down)]
    tilt_ratios = kwargs.get(
        "upright_object_top_down_tilt_ratios",
        [0.25, 0.5, 0.75, 1.0],
    )
    if isinstance(tilt_ratios, str):
        tilt_ratios = [float(item) for item in tilt_ratios.split(",") if item]

    tilt_axes = []
    if arm_axis is not None:
        tilt_axes.extend(
            [
                ("arm_to_object", arm_axis),
                ("object_to_arm", -arm_axis),
            ]
        )
    if lateral_axis is not None:
        tilt_axes.extend(
            [
                ("lateral_pos", lateral_axis),
                ("lateral_neg", -lateral_axis),
            ]
        )

    for tilt_label, tilt_axis in tilt_axes:
        for tilt_ratio in tilt_ratios:
            approach = _normalize_vector(
                top_down + tilt_axis * float(tilt_ratio)
            )
            if approach is not None:
                approach_candidates.append(
                    (
                        f"{tilt_label}/tilt={float(tilt_ratio):.2f}",
                        approach.to(dtype=dtype, device=device),
                    )
                )

    for approach_label, approach_axis in approach_candidates:
        tool_z_modes = [
            ("tool_z_along_approach", approach_axis, -1.0),
            ("tool_z_against_approach", -approach_axis, 1.0),
        ]
        for tool_z_label, tool_z_axis, pre_offset_sign in tool_z_modes:
            for seed_label, seed in roll_seeds:
                for roll_rad in (0.0, np.pi):
                    rotation = _rotation_from_approach_and_roll(
                        tool_z_axis,
                        seed,
                        float(roll_rad),
                        dtype=dtype,
                        device=device,
                    )
                    if rotation is not None:
                        rotation_candidates.append(
                            (
                                f"{approach_label}/{tool_z_label}/"
                                f"{seed_label}/roll={float(roll_rad):.2f}",
                                rotation,
                                pre_offset_sign,
                            )
                        )

    sample_num = int(
        kwargs.get("upright_object_top_down_sample_num", kwargs.get("sample_num", 30))
    )
    lift_height = float(kwargs.get("upright_object_top_down_lift_height", 0.18))
    table_clearance = float(kwargs.get("upright_object_table_clearance", 0.035))
    grasp_z_offset = float(kwargs.get("upright_object_top_down_z_offset", 0.0))
    errors = []

    for fraction in axis_fractions:
        if long_axis is None:
            grasp_center = center.detach().clone()
        else:
            grasp_center = center + long_axis * half_axis_extent * float(fraction)
        grasp_center = torch.minimum(torch.maximum(grasp_center, bounds["mins"]), bounds["maxs"])
        for z_ratio in z_ratios:
            grasp_z = min_z + (max_z - min_z) * float(z_ratio) + grasp_z_offset
            grasp_z = max(grasp_z, min_z + table_clearance)
            for rotation_label, rotation, pre_offset_sign in rotation_candidates:
                grasp_pose = torch.eye(4, dtype=dtype, device=device)
                grasp_pose[:3, :3] = rotation
                grasp_pose[:3, 3] = grasp_center
                grasp_pose[2, 3] = float(grasp_z)
                candidate_label = (
                    f"fraction={float(fraction):+.2f}/"
                    f"z_ratio={float(z_ratio):.2f}/{rotation_label}"
                )

                for pre_grasp_distance in pre_grasp_distances:
                    grasp_pose_pre = get_offset_pose(
                        deepcopy(grasp_pose),
                        float(pre_offset_sign) * float(pre_grasp_distance),
                        "z",
                        "intrinsic",
                    )
                    pre_label = (
                        f"{candidate_label}/pre={float(pre_grasp_distance):.2f}"
                    )
                    if force_valid:
                        try:
                            _, grasp_qpos_pre = get_qpos(
                                env,
                                is_left,
                                select_arm,
                                grasp_pose_pre,
                                select_arm_current_qpos,
                                force_valid=force_valid,
                                name=(
                                    "upright_object top_down pre-grasp "
                                    f"{pre_label}"
                                ),
                            )
                        except RuntimeError as exc:
                            errors.append(f"{pre_label}: {exc}")
                            continue
                    else:
                        grasp_qpos_pre, pre_seed_label, solve_errors = _solve_pose_ik(
                            grasp_pose_pre,
                            seed_candidates,
                            f"pre-grasp {pre_label}",
                        )
                        if grasp_qpos_pre is None:
                            errors.extend(solve_errors)
                            continue

                    grasp_qpos, _, solve_errors = _solve_pose_ik(
                        grasp_pose,
                        [("pre", grasp_qpos_pre)],
                        f"grasp {pre_label}",
                    )
                    if grasp_qpos is None:
                        errors.extend(solve_errors)
                        continue

                    grasp_relation = torch.linalg.inv(obj_pose) @ grasp_pose
                    lift_pose = deepcopy(grasp_pose)
                    lift_pose[2, 3] = lift_pose[2, 3] + lift_height
                    lift_qpos, _, solve_errors = _solve_pose_ik(
                        lift_pose,
                        [("grasp", grasp_qpos)],
                        f"lift {pre_label}",
                    )
                    if lift_qpos is None:
                        errors.extend(solve_errors)
                        continue

                    release_pose = None
                    release_qpos = None
                    release_label = None
                    release_errors = []
                    for yaw_deg in yaw_offsets_deg:
                        yaw_rot = R.from_euler(
                            "z", float(yaw_deg), degrees=True
                        ).as_matrix()
                        yaw_rot = torch.as_tensor(
                            yaw_rot,
                            dtype=dtype,
                            device=device,
                        )
                        for z_offset in release_z_offsets:
                            upright_obj_pose = torch.eye(
                                4,
                                dtype=dtype,
                                device=device,
                            )
                            upright_obj_pose[:3, :3] = yaw_rot @ base_upright_rotation
                            upright_obj_pose[:3, 3] = torch.tensor(
                                [target_x, target_y, target_z + float(z_offset)],
                                dtype=dtype,
                                device=device,
                            )
                            base_release_pose = upright_obj_pose @ grasp_relation
                            for pre_offset in release_pre_offsets:
                                candidate_release_pose = get_offset_pose(
                                    deepcopy(base_release_pose),
                                    float(pre_offset),
                                    "z",
                                    "intrinsic",
                                )
                                release_candidate_label = (
                                    f"{pre_label}/yaw={float(yaw_deg):.0f}/"
                                    f"z={float(z_offset):.2f}/"
                                    f"pre={float(pre_offset):.2f}"
                                )
                                candidate_release_qpos, _, solve_errors = (
                                    _solve_pose_ik(
                                        candidate_release_pose,
                                        [("lift", lift_qpos)],
                                        f"release {release_candidate_label}",
                                    )
                                )
                                if candidate_release_qpos is None:
                                    release_errors.extend(solve_errors)
                                    continue
                                release_pose = candidate_release_pose
                                release_qpos = candidate_release_qpos
                                release_label = release_candidate_label
                                break
                            if release_qpos is not None:
                                break
                        if release_qpos is not None:
                            break
                    if release_qpos is None:
                        errors.extend(release_errors[-3:])
                        continue

                    approach_qpos_traj = []
                    approach_gripper_traj = []
                    plan_trajectory(
                        env,
                        select_arm,
                        [select_arm_current_qpos, grasp_qpos_pre, grasp_qpos],
                        sample_num,
                        select_arm_current_gripper_state,
                        approach_qpos_traj,
                        approach_gripper_traj,
                    )
                    approach_actions = finalize_actions(
                        approach_qpos_traj,
                        approach_gripper_traj,
                    )
                    env.set_current_qpos_agent(grasp_qpos, is_left=is_left)
                    env.set_current_xpos_agent(grasp_pose, is_left=is_left)
                    close_actions = close_gripper(robot_name, env=env, **kwargs)
                    close_state = (
                        env.left_arm_current_gripper_state
                        if is_left
                        else env.right_arm_current_gripper_state
                    )

                    lift_release_qpos_traj = []
                    lift_release_gripper_traj = []
                    plan_trajectory(
                        env,
                        select_arm,
                        [grasp_qpos, lift_qpos, release_qpos],
                        sample_num,
                        close_state,
                        lift_release_qpos_traj,
                        lift_release_gripper_traj,
                    )
                    lift_release_actions = finalize_actions(
                        lift_release_qpos_traj,
                        lift_release_gripper_traj,
                    )
                    env.set_current_qpos_agent(release_qpos, is_left=is_left)
                    env.set_current_xpos_agent(release_pose, is_left=is_left)
                    open_actions = open_gripper(robot_name, env=env, **kwargs)
                    open_state = (
                        env.left_arm_current_gripper_state
                        if is_left
                        else env.right_arm_current_gripper_state
                    )

                    retreat_qpos = env.left_arm_init_qpos if is_left else env.right_arm_init_qpos
                    retreat_qpos = torch.as_tensor(
                        retreat_qpos,
                        dtype=select_arm_current_qpos.dtype,
                        device=select_arm_current_qpos.device,
                    )
                    retreat_pose = env.get_arm_fk(
                        qpos=retreat_qpos, is_left=is_left
                    ).to(dtype=dtype, device=device)
                    retreat_qpos_traj = []
                    retreat_gripper_traj = []
                    plan_trajectory(
                        env,
                        select_arm,
                        [release_qpos, retreat_qpos],
                        int(
                            kwargs.get(
                                "upright_object_top_down_retreat_sample_num",
                                max(8, sample_num // 2),
                            )
                        ),
                        open_state,
                        retreat_qpos_traj,
                        retreat_gripper_traj,
                    )
                    retreat_actions = finalize_actions(
                        retreat_qpos_traj,
                        retreat_gripper_traj,
                    )
                    env.set_current_qpos_agent(retreat_qpos, is_left=is_left)
                    env.set_current_xpos_agent(retreat_pose, is_left=is_left)

                    actions = np.concatenate(
                        [
                            approach_actions,
                            close_actions,
                            lift_release_actions,
                            open_actions,
                            retreat_actions,
                        ],
                        axis=0,
                    )
                    log_info(
                        f"Selected upright_object top_down_grasp_place candidate "
                        f"for {robot_name}/{obj_name}: {release_label}, "
                        f"actions={len(actions)}.",
                        color="green",
                    )
                    return actions

    raise RuntimeError(
        f"upright_object top_down_grasp_place failed for {robot_name}/{obj_name}. "
        f"Recent failures: {' | '.join(errors[-3:])}"
    )


def _plan_fallen_object_top_clamp(
    *,
    env,
    robot_name: str,
    obj_name: str,
    obj_pose: torch.Tensor,
    bounds: dict | None,
    x: float | None,
    y: float | None,
    pre_grasp_dis: float,
    pre_place_dis: float,
    force_valid: bool,
    kwargs: dict,
):
    """Plan a top-down clamp, lift, and upright release for a fallen object."""

    if bounds is None:
        raise RuntimeError("top_clamp requires object mesh bounds.")

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    device = obj_pose.device
    dtype = obj_pose.dtype
    center = bounds["center"].to(dtype=dtype, device=device)
    vertices = bounds.get("vertices")
    local_z_world = obj_pose[:3, 2].detach().clone()
    local_z_world[2] = 0.0
    long_axis = _normalize_vector(local_z_world)
    if long_axis is not None:
        long_axis = long_axis.to(dtype=dtype, device=device)

    if long_axis is not None and vertices is not None and vertices.numel() > 0:
        projections = (vertices - center) @ long_axis
        half_axis_extent = float(torch.max(torch.abs(projections)).item())
    else:
        half_axis_extent = float(torch.max(bounds["extents"][:2]).item()) * 0.5

    target_x = float(center[0]) if x is None else float(x)
    target_y = float(center[1]) if y is None else float(y)
    target_z = float(
        kwargs.get(
            "upright_object_target_z",
            getattr(env, "obj_info", {}).get(obj_name, {}).get("height", obj_pose[2, 3]),
        )
    )
    min_z = float(bounds["mins"][2].item())
    max_z = float(bounds["maxs"][2].item())

    initial_pose = getattr(env, "obj_info", {}).get(obj_name, {}).get("initial_pose")
    if initial_pose is not None:
        initial_pose = torch.as_tensor(
            initial_pose,
            dtype=dtype,
            device=device,
        )
        base_upright_rotation = initial_pose[:3, :3]
    else:
        base_upright_rotation = _upright_object_rotation_from_pose(obj_pose)

    clamp_offsets = kwargs.get(
        "upright_object_top_clamp_axis_fractions",
        [0.0, 0.18, -0.18, 0.32, -0.32],
    )
    if isinstance(clamp_offsets, str):
        clamp_offsets = [float(item) for item in clamp_offsets.split(",") if item]

    z_ratios = kwargs.get(
        "upright_object_top_clamp_z_ratios",
        [0.58, 0.48, 0.68, 0.38],
    )
    if isinstance(z_ratios, str):
        z_ratios = [float(item) for item in z_ratios.split(",") if item]

    roll_offsets = kwargs.get(
        "upright_object_top_clamp_roll_offsets",
        [0.0, np.pi / 2.0, -np.pi / 2.0, np.pi],
    )
    if isinstance(roll_offsets, str):
        roll_offsets = [float(item) for item in roll_offsets.split(",") if item]

    yaw_offsets_deg = kwargs.get(
        "upright_object_release_yaw_offsets_deg",
        [0.0, 90.0, -90.0, 180.0],
    )
    if isinstance(yaw_offsets_deg, str):
        yaw_offsets_deg = [float(item) for item in yaw_offsets_deg.split(",") if item]

    release_z_offsets = kwargs.get(
        "upright_object_release_z_offsets",
        [0.0, 0.06, 0.12],
    )
    if isinstance(release_z_offsets, str):
        release_z_offsets = [float(item) for item in release_z_offsets.split(",") if item]

    release_pre_offsets = kwargs.get(
        "upright_object_release_pre_offsets",
        [0.0, pre_place_dis],
    )
    if isinstance(release_pre_offsets, str):
        release_pre_offsets = [
            float(item) for item in release_pre_offsets.split(",") if item
        ]

    roll_seeds = []
    if long_axis is not None:
        roll_seeds.append(("long_axis", long_axis))
        lateral = torch.tensor(
            [-long_axis[1], long_axis[0], 0.0],
            dtype=dtype,
            device=device,
        )
        if torch.linalg.norm(lateral) > 1e-6:
            roll_seeds.append(("lateral_axis", lateral))
    roll_seeds.append(
        ("world_x", torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device))
    )

    rotation_candidates = [
        (
            "current_eef",
            select_arm_current_pose[:3, :3].to(dtype=dtype, device=device),
        )
    ]
    for roll_seed_label, roll_seed in roll_seeds:
        for roll_rad in roll_offsets:
            rotation = _rotation_from_approach_and_roll(
                torch.tensor([0.0, 0.0, -1.0], dtype=dtype, device=device),
                roll_seed,
                float(roll_rad),
                dtype=dtype,
                device=device,
            )
            if rotation is not None:
                rotation_candidates.append(
                    (
                        f"top_down/{roll_seed_label}/roll={float(roll_rad):.2f}",
                        rotation,
                    )
                )

    sample_num = int(
        kwargs.get("upright_object_top_clamp_sample_num", kwargs.get("sample_num", 30))
    )
    lift_height = float(kwargs.get("upright_object_top_clamp_lift_height", 0.22))
    table_clearance = float(kwargs.get("upright_object_table_clearance", 0.035))
    grasp_z_offset = float(kwargs.get("upright_object_top_clamp_z_offset", 0.0))
    errors = []

    for fraction in clamp_offsets:
        if long_axis is None:
            grasp_center = center.detach().clone()
        else:
            grasp_center = center + long_axis * half_axis_extent * float(fraction)
        grasp_center = torch.minimum(torch.maximum(grasp_center, bounds["mins"]), bounds["maxs"])
        for z_ratio in z_ratios:
            grasp_z = min_z + (max_z - min_z) * float(z_ratio) + grasp_z_offset
            grasp_z = max(grasp_z, min_z + table_clearance)
            for rotation_label, rotation in rotation_candidates:

                grasp_pose = torch.eye(4, dtype=dtype, device=device)
                grasp_pose[:3, :3] = rotation
                grasp_pose[:3, 3] = grasp_center
                grasp_pose[2, 3] = float(grasp_z)
                candidate_label = (
                    f"fraction={float(fraction):+.2f}/"
                    f"z_ratio={float(z_ratio):.2f}/rotation={rotation_label}"
                )
                grasp_pose_pre = get_offset_pose(
                    deepcopy(grasp_pose),
                    -float(pre_grasp_dis),
                    "z",
                    "intrinsic",
                )

                try:
                    _, grasp_qpos_pre = get_qpos(
                        env,
                        is_left,
                        select_arm,
                        grasp_pose_pre,
                        select_arm_current_qpos,
                        force_valid=force_valid,
                        name=f"upright_object top_clamp pre-grasp {candidate_label}",
                    )
                    grasp_pose, grasp_qpos = get_qpos(
                        env,
                        is_left,
                        select_arm,
                        grasp_pose,
                        grasp_qpos_pre,
                        force_valid=force_valid,
                        name=f"upright_object top_clamp grasp {candidate_label}",
                    )

                    grasp_relation = torch.linalg.inv(obj_pose) @ grasp_pose
                    lift_pose = deepcopy(grasp_pose)
                    lift_pose[2, 3] = lift_pose[2, 3] + lift_height
                    _, lift_qpos = get_qpos(
                        env,
                        is_left,
                        select_arm,
                        lift_pose,
                        grasp_qpos,
                        force_valid=force_valid,
                        name=f"upright_object top_clamp lift {candidate_label}",
                    )

                    release_pose = None
                    release_qpos = None
                    release_label = None
                    release_errors = []
                    for yaw_deg in yaw_offsets_deg:
                        yaw_rot = R.from_euler(
                            "z", float(yaw_deg), degrees=True
                        ).as_matrix()
                        yaw_rot = torch.as_tensor(
                            yaw_rot,
                            dtype=dtype,
                            device=device,
                        )
                        for z_offset in release_z_offsets:
                            upright_obj_pose = torch.eye(
                                4,
                                dtype=dtype,
                                device=device,
                            )
                            upright_obj_pose[:3, :3] = yaw_rot @ base_upright_rotation
                            upright_obj_pose[:3, 3] = torch.tensor(
                                [target_x, target_y, target_z + float(z_offset)],
                                dtype=dtype,
                                device=device,
                            )
                            base_release_pose = upright_obj_pose @ grasp_relation
                            for pre_offset in release_pre_offsets:
                                release_candidate_pose = get_offset_pose(
                                    deepcopy(base_release_pose),
                                    float(pre_offset),
                                    "z",
                                    "intrinsic",
                                )
                                release_candidate_label = (
                                    f"{candidate_label}/yaw={float(yaw_deg):.0f}/"
                                    f"z={float(z_offset):.2f}/pre={float(pre_offset):.2f}"
                                )
                                try:
                                    (
                                        release_candidate_pose,
                                        release_candidate_qpos,
                                    ) = get_qpos(
                                        env,
                                        is_left,
                                        select_arm,
                                        release_candidate_pose,
                                        lift_qpos,
                                        force_valid=force_valid,
                                        name=(
                                            "upright_object top_clamp release "
                                            f"{release_candidate_label}"
                                        ),
                                    )
                                except RuntimeError as release_exc:
                                    release_errors.append(
                                        f"{release_candidate_label}: {release_exc}"
                                    )
                                    continue
                                release_pose = release_candidate_pose
                                release_qpos = release_candidate_qpos
                                release_label = release_candidate_label
                                break
                            if release_qpos is not None:
                                break
                        if release_qpos is not None:
                            break
                    if release_qpos is None:
                        errors.extend(release_errors[-3:])
                        continue
                except RuntimeError as exc:
                    errors.append(f"{candidate_label}: {exc}")
                    continue

                approach_qpos_traj = []
                approach_gripper_traj = []
                plan_trajectory(
                    env,
                    select_arm,
                    [select_arm_current_qpos, grasp_qpos_pre, grasp_qpos],
                    sample_num,
                    select_arm_current_gripper_state,
                    approach_qpos_traj,
                    approach_gripper_traj,
                )
                approach_actions = finalize_actions(
                    approach_qpos_traj,
                    approach_gripper_traj,
                )
                env.set_current_qpos_agent(grasp_qpos, is_left=is_left)
                env.set_current_xpos_agent(grasp_pose, is_left=is_left)
                close_actions = close_gripper(robot_name, env=env, **kwargs)
                close_state = (
                    env.left_arm_current_gripper_state
                    if is_left
                    else env.right_arm_current_gripper_state
                )

                lift_release_qpos_traj = []
                lift_release_gripper_traj = []
                plan_trajectory(
                    env,
                    select_arm,
                    [grasp_qpos, lift_qpos, release_qpos],
                    sample_num,
                    close_state,
                    lift_release_qpos_traj,
                    lift_release_gripper_traj,
                )
                lift_release_actions = finalize_actions(
                    lift_release_qpos_traj,
                    lift_release_gripper_traj,
                )
                env.set_current_qpos_agent(release_qpos, is_left=is_left)
                env.set_current_xpos_agent(release_pose, is_left=is_left)
                open_actions = open_gripper(robot_name, env=env, **kwargs)

                actions = np.concatenate(
                    [approach_actions, close_actions, lift_release_actions, open_actions],
                    axis=0,
                )
                log_info(
                    f"Selected upright_object top_clamp candidate for "
                    f"{robot_name}/{obj_name}: {release_label}, actions={len(actions)}.",
                    color="green",
                )
                return actions

    raise RuntimeError(
        f"upright_object top_clamp failed for {robot_name}/{obj_name}. "
        f"Recent failures: {' | '.join(errors[-3:])}"
    )


def _plan_fallen_object_lever_sweep(
    *,
    env,
    robot_name: str,
    obj_name: str,
    obj_pose: torch.Tensor,
    bounds: dict | None,
    x: float | None,
    y: float | None,
    force_valid: bool,
    kwargs: dict,
):
    """Plan a contact-rich endpoint sweep for a fallen object.

    This primitive does not assume the object can already be lifted as a stable
    grasp. It uses the gripper as a paddle/hook at one end of the fallen object,
    then lifts and sweeps the end toward the target upright center.
    """

    if bounds is None:
        raise RuntimeError("lever_sweep requires object mesh bounds.")

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    device = obj_pose.device
    dtype = obj_pose.dtype
    center = bounds["center"].to(dtype=dtype, device=device)
    vertices = bounds.get("vertices")
    if vertices is None or vertices.numel() == 0:
        raise RuntimeError("lever_sweep requires object vertices.")

    local_z_world = obj_pose[:3, 2].detach().clone()
    local_z_world[2] = 0.0
    long_axis = _normalize_vector(local_z_world)
    if long_axis is None:
        raise RuntimeError("lever_sweep cannot infer fallen object long axis.")
    long_axis = long_axis.to(dtype=dtype, device=device)

    projections = (vertices - center) @ long_axis
    half_axis_extent = float(torch.max(torch.abs(projections)).item())
    if half_axis_extent <= 1e-4:
        raise RuntimeError("lever_sweep object long-axis extent is too small.")

    base_pose = env.left_arm_base_pose if is_left else env.right_arm_base_pose
    base_xy = base_pose[:2, 3].to(dtype=dtype, device=device)
    endpoint_scale = float(kwargs.get("upright_object_lever_endpoint_scale", 0.98))
    endpoints = [
        center + long_axis * half_axis_extent * endpoint_scale,
        center - long_axis * half_axis_extent * endpoint_scale,
    ]
    endpoints.sort(
        key=lambda point: float(torch.linalg.norm(point[:2] - base_xy).item())
    )

    target_x = float(center[0]) if x is None else float(x)
    target_y = float(center[1]) if y is None else float(y)
    target_z = float(
        kwargs.get(
            "upright_object_target_z",
            getattr(env, "obj_info", {}).get(obj_name, {}).get("height", obj_pose[2, 3]),
        )
    )

    min_z = float(bounds["mins"][2].item())
    max_z = float(bounds["maxs"][2].item())
    pre_z = max_z + float(kwargs.get("upright_object_lever_pre_height", 0.14))
    contact_height_fraction = float(
        kwargs.get("upright_object_lever_contact_height_fraction", 0.45)
    )
    contact_z = min_z + (max_z - min_z) * contact_height_fraction
    contact_z = max(
        contact_z + float(kwargs.get("upright_object_lever_contact_z_offset", 0.0)),
        min_z + float(kwargs.get("upright_object_lever_table_clearance", 0.025)),
    )
    lift_z = target_z + float(kwargs.get("upright_object_lever_lift_height", 0.22))
    sweep_z = target_z + float(kwargs.get("upright_object_lever_sweep_height", 0.18))
    settle_z = target_z + float(kwargs.get("upright_object_lever_settle_height", 0.14))
    sample_num = int(
        kwargs.get("upright_object_lever_sample_num", kwargs.get("sample_num", 30))
    )
    sweep_endpoint_fraction = float(
        kwargs.get("upright_object_lever_sweep_endpoint_fraction", 0.20)
    )
    settle_endpoint_fraction = float(
        kwargs.get("upright_object_lever_settle_endpoint_fraction", 0.0)
    )

    rotation = select_arm_current_pose[:3, :3].to(dtype=dtype, device=device)
    errors = []
    for endpoint_idx, endpoint in enumerate(endpoints):
        endpoint = endpoint.to(dtype=dtype, device=device)
        center_xy = torch.tensor([target_x, target_y], dtype=dtype, device=device)
        outward_xy = endpoint[:2] - center_xy
        outward_norm = torch.linalg.norm(outward_xy)
        if float(outward_norm.item()) > 1e-6:
            outward_xy = outward_xy / outward_norm
        else:
            outward_xy = torch.zeros_like(outward_xy)
        contact_xy = endpoint[:2] + outward_xy * float(
            kwargs.get("upright_object_lever_contact_xy_outset", 0.035)
        )
        lift_xy = endpoint[:2] * 0.80 + center_xy * 0.20
        sweep_xy = (
            endpoint[:2] * sweep_endpoint_fraction
            + center_xy * (1.0 - sweep_endpoint_fraction)
        )
        settle_xy = (
            endpoint[:2] * settle_endpoint_fraction
            + center_xy * (1.0 - settle_endpoint_fraction)
        )

        waypoints = [
            (
                "pre",
                torch.tensor(
                    [contact_xy[0], contact_xy[1], pre_z],
                    dtype=dtype,
                    device=device,
                ),
            ),
            (
                "contact",
                torch.tensor(
                    [contact_xy[0], contact_xy[1], contact_z],
                    dtype=dtype,
                    device=device,
                ),
            ),
            ("lift", torch.tensor([lift_xy[0], lift_xy[1], lift_z], dtype=dtype, device=device)),
            (
                "sweep",
                torch.tensor(
                    [sweep_xy[0], sweep_xy[1], sweep_z],
                    dtype=dtype,
                    device=device,
                ),
            ),
            (
                "settle",
                torch.tensor(
                    [settle_xy[0], settle_xy[1], settle_z],
                    dtype=dtype,
                    device=device,
                ),
            ),
        ]

        qposes = []
        poses = []
        qpos_seed = select_arm_current_qpos
        try:
            for waypoint_label, xyz in waypoints:
                pose = torch.eye(4, dtype=dtype, device=device)
                pose[:3, :3] = rotation
                pose[:3, 3] = xyz
                pose, qpos_seed = get_qpos(
                    env,
                    is_left,
                    select_arm,
                    pose,
                    qpos_seed,
                    force_valid=force_valid,
                    name=(
                        "upright_object lever_sweep "
                        f"endpoint={endpoint_idx}/{waypoint_label}"
                    ),
                )
                poses.append(pose)
                qposes.append(qpos_seed)
        except RuntimeError as exc:
            errors.append(f"endpoint={endpoint_idx}: {exc}")
            continue

        approach_qpos = qposes[:2]
        lever_qpos = qposes[1:]
        approach_qpos_traj = []
        approach_gripper_traj = []
        plan_trajectory(
            env,
            select_arm,
            [select_arm_current_qpos, *approach_qpos],
            sample_num,
            select_arm_current_gripper_state,
            approach_qpos_traj,
            approach_gripper_traj,
        )
        approach_actions = finalize_actions(approach_qpos_traj, approach_gripper_traj)

        env.set_current_qpos_agent(qposes[1], is_left=is_left)
        env.set_current_xpos_agent(poses[1], is_left=is_left)
        close_actions = close_gripper(robot_name, env=env, **kwargs)
        close_state = (
            env.left_arm_current_gripper_state
            if is_left
            else env.right_arm_current_gripper_state
        )

        lever_qpos_traj = []
        lever_gripper_traj = []
        plan_trajectory(
            env,
            select_arm,
            lever_qpos,
            sample_num,
            close_state,
            lever_qpos_traj,
            lever_gripper_traj,
        )
        lever_actions = finalize_actions(lever_qpos_traj, lever_gripper_traj)
        env.set_current_qpos_agent(qposes[-1], is_left=is_left)
        env.set_current_xpos_agent(poses[-1], is_left=is_left)
        open_actions = open_gripper(robot_name, env=env, **kwargs)
        open_state = (
            env.left_arm_current_gripper_state
            if is_left
            else env.right_arm_current_gripper_state
        )

        retreat_pose = deepcopy(poses[-1])
        retreat_pose[:3, 3] = torch.tensor(
            [
                target_x,
                target_y,
                target_z
                + float(kwargs.get("upright_object_lever_post_open_retreat_height", 0.24)),
            ],
            dtype=dtype,
            device=device,
        )
        retreat_label = "cartesian"
        try:
            retreat_pose, retreat_qpos = get_qpos(
                env,
                is_left,
                select_arm,
                retreat_pose,
                qposes[-1],
                force_valid=force_valid,
                name=f"upright_object lever_sweep endpoint={endpoint_idx}/retreat",
            )
        except RuntimeError as exc:
            if kwargs.get("upright_object_lever_require_cartesian_retreat", False):
                errors.append(f"endpoint={endpoint_idx}/retreat: {exc}")
                continue
            log_warning(
                f"upright_object lever_sweep endpoint={endpoint_idx} cartesian "
                f"retreat IK failed; falling back to joint-space retreat. ({exc})"
            )
            retreat_label = "joint_initial"
            retreat_qpos = env.left_arm_init_qpos if is_left else env.right_arm_init_qpos
            retreat_qpos = torch.as_tensor(
                retreat_qpos,
                dtype=select_arm_current_qpos.dtype,
                device=select_arm_current_qpos.device,
            )
            retreat_pose = env.get_arm_fk(qpos=retreat_qpos, is_left=is_left).to(
                dtype=dtype, device=device
            )
        retreat_qpos_traj = []
        retreat_gripper_traj = []
        plan_trajectory(
            env,
            select_arm,
            [qposes[-1], retreat_qpos],
            int(
                kwargs.get(
                    "upright_object_lever_joint_retreat_sample_num",
                    max(8, sample_num // 2),
                )
            ),
            open_state,
            retreat_qpos_traj,
            retreat_gripper_traj,
        )
        retreat_actions = finalize_actions(retreat_qpos_traj, retreat_gripper_traj)
        env.set_current_qpos_agent(retreat_qpos, is_left=is_left)
        env.set_current_xpos_agent(retreat_pose, is_left=is_left)

        actions = np.concatenate(
            [
                approach_actions,
                close_actions,
                lever_actions,
                open_actions,
                retreat_actions,
            ],
            axis=0,
        )
        log_info(
            f"Selected upright_object lever_sweep candidate for "
            f"{robot_name}/{obj_name}: endpoint={endpoint_idx}, "
            f"retreat={retreat_label}, actions={len(actions)}.",
            color="green",
        )
        return actions

    raise RuntimeError(
        f"upright_object lever_sweep failed for {robot_name}/{obj_name}. "
        f"Recent failures: {' | '.join(errors[-3:])}"
    )


def _plan_fallen_object_grasp_place(
    *,
    env,
    robot_name: str,
    obj_name: str,
    x: float | None,
    y: float | None,
    pre_grasp_dis: float,
    pre_place_dis: float,
    force_valid: bool,
    kwargs: dict,
):
    """Physically recover a fallen object by grasping its current pose and placing upright."""

    grasp_kwargs = dict(kwargs)
    grasp_kwargs.setdefault("use_public_grasp_semantics", False)
    grasp_kwargs["public_grasp_preserve_object_rotation"] = True
    grasp_actions = grasp(
        robot_name,
        obj_name,
        pre_grasp_dis=pre_grasp_dis,
        env=env,
        force_valid=force_valid,
        **grasp_kwargs,
    )
    place_actions = place_on_table(
        robot_name,
        obj_name,
        x=x,
        y=y,
        pre_place_dis=pre_place_dis,
        env=env,
        force_valid=force_valid,
        **kwargs,
    )
    actions = np.concatenate([grasp_actions, place_actions], axis=0)
    log_info(
        f"Selected upright_object grasp_place primitive for "
        f"{robot_name}/{obj_name}: actions={len(actions)}.",
        color="green",
    )
    return actions


def _plan_dedicated_upright_object(
    *,
    env,
    robot_name: str,
    obj_name: str,
    obj_pose: torch.Tensor,
    bounds: dict | None,
    x: float | None,
    y: float | None,
    pre_grasp_dis: float,
    pre_place_dis: float,
    force_valid: bool,
    kwargs: dict,
):
    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        _,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    candidates = _fallen_object_grasp_candidates(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        obj_pose=obj_pose,
        bounds=bounds,
        is_left=is_left,
        kwargs=kwargs,
    )
    if not candidates:
        raise RuntimeError(f"upright_object found no grasp candidates for {obj_name}.")

    sample_num = int(
        kwargs.get("upright_object_sample_num", kwargs.get("sample_num", 30))
    )
    lift_height = float(kwargs.get("upright_object_lift_height", 0.18))
    target_x = float(obj_pose[0, 3]) if x is None else float(x)
    target_y = float(obj_pose[1, 3]) if y is None else float(y)
    target_z = float(
        kwargs.get(
            "upright_object_target_z",
            getattr(env, "obj_info", {}).get(obj_name, {}).get("height", obj_pose[2, 3]),
        )
    )
    release_z_offsets = kwargs.get(
        "upright_object_release_z_offsets",
        [0.0, 0.06, 0.12],
    )
    if isinstance(release_z_offsets, str):
        release_z_offsets = [
            float(item) for item in release_z_offsets.split(",") if item
        ]
    release_pre_offsets = kwargs.get(
        "upright_object_release_pre_offsets",
        [0.0, pre_place_dis],
    )
    if isinstance(release_pre_offsets, str):
        release_pre_offsets = [
            float(item) for item in release_pre_offsets.split(",") if item
        ]
    yaw_offsets_deg = kwargs.get(
        "upright_object_release_yaw_offsets_deg",
        [0.0, 90.0, -90.0, 180.0],
    )
    if isinstance(yaw_offsets_deg, str):
        yaw_offsets_deg = [float(item) for item in yaw_offsets_deg.split(",") if item]
    pre_grasp_distances = kwargs.get(
        "upright_object_pre_grasp_distances",
        [pre_grasp_dis, pre_grasp_dis * 0.5, 0.0],
    )
    if isinstance(pre_grasp_distances, str):
        pre_grasp_distances = [
            float(item) for item in pre_grasp_distances.split(",") if item
        ]
    initial_pose = getattr(env, "obj_info", {}).get(obj_name, {}).get("initial_pose")
    if initial_pose is not None:
        initial_pose = torch.as_tensor(
            initial_pose,
            dtype=obj_pose.dtype,
            device=obj_pose.device,
        )
        base_upright_rotation = initial_pose[:3, :3]
    else:
        base_upright_rotation = _upright_object_rotation_from_pose(obj_pose)

    errors = []
    for label, grasp_pose in candidates:
        for pre_grasp_distance in pre_grasp_distances:
            candidate_label = f"{label}/pre_grasp={float(pre_grasp_distance):.2f}"
            grasp_pose_pre = get_offset_pose(
                deepcopy(grasp_pose),
                -float(pre_grasp_distance),
                "z",
                "intrinsic",
            )
            try:
                _, grasp_qpos_pre = get_qpos(
                    env,
                    is_left,
                    select_arm,
                    grasp_pose_pre,
                    select_arm_current_qpos,
                    force_valid=force_valid,
                    name=f"upright_object pre-grasp {candidate_label}",
                )
                grasp_pose, grasp_qpos = get_qpos(
                    env,
                    is_left,
                    select_arm,
                    grasp_pose,
                    grasp_qpos_pre,
                    force_valid=force_valid,
                    name=f"upright_object grasp {candidate_label}",
                )

                grasp_relation = torch.linalg.inv(obj_pose) @ grasp_pose
                lift_pose = deepcopy(grasp_pose)
                lift_pose[2, 3] = lift_pose[2, 3] + lift_height
                _, lift_qpos = get_qpos(
                    env,
                    is_left,
                    select_arm,
                    lift_pose,
                    grasp_qpos,
                    force_valid=force_valid,
                    name=f"upright_object lift {candidate_label}",
                )

                release_pose = None
                release_qpos = None
                release_label = None
                release_errors = []
                for yaw_deg in yaw_offsets_deg:
                    yaw_rot = R.from_euler(
                        "z", float(yaw_deg), degrees=True
                    ).as_matrix()
                    yaw_rot = torch.as_tensor(
                        yaw_rot,
                        dtype=obj_pose.dtype,
                        device=obj_pose.device,
                    )
                    for z_offset in release_z_offsets:
                        upright_obj_pose = torch.eye(
                            4,
                            dtype=obj_pose.dtype,
                            device=obj_pose.device,
                        )
                        upright_obj_pose[:3, :3] = yaw_rot @ base_upright_rotation
                        upright_obj_pose[:3, 3] = torch.tensor(
                            [target_x, target_y, target_z + float(z_offset)],
                            dtype=obj_pose.dtype,
                            device=obj_pose.device,
                        )
                        base_release_pose = upright_obj_pose @ grasp_relation
                        for pre_offset in release_pre_offsets:
                            candidate_release_pose = get_offset_pose(
                                deepcopy(base_release_pose),
                                float(pre_offset),
                                "z",
                                "intrinsic",
                            )
                            release_candidate_label = (
                                f"{candidate_label}/yaw={float(yaw_deg):.0f}/"
                                f"z={float(z_offset):.2f}/"
                                f"pre={float(pre_offset):.2f}"
                            )
                            try:
                                (
                                    candidate_release_pose,
                                    candidate_release_qpos,
                                ) = get_qpos(
                                    env,
                                    is_left,
                                    select_arm,
                                    candidate_release_pose,
                                    lift_qpos,
                                    force_valid=force_valid,
                                    name=(
                                        "upright_object release "
                                        f"{release_candidate_label}"
                                    ),
                                )
                            except RuntimeError as release_exc:
                                release_errors.append(
                                    f"{release_candidate_label}: {release_exc}"
                                )
                                continue
                            release_pose = candidate_release_pose
                            release_qpos = candidate_release_qpos
                            release_label = release_candidate_label
                            break
                        if release_qpos is not None:
                            break
                    if release_qpos is not None:
                        break
                if release_qpos is None:
                    errors.extend(release_errors[-3:])
                    continue
            except RuntimeError as exc:
                errors.append(f"{candidate_label}: {exc}")
                continue
            break
        else:
            continue

        select_qpos_traj = []
        ee_state_list_select = []
        plan_trajectory(
            env,
            select_arm,
            [select_arm_current_qpos, grasp_qpos_pre],
            sample_num,
            select_arm_current_gripper_state,
            select_qpos_traj,
            ee_state_list_select,
        )
        plan_trajectory(
            env,
            select_arm,
            [grasp_qpos_pre, grasp_qpos],
            max(8, sample_num // 2),
            select_arm_current_gripper_state,
            select_qpos_traj,
            ee_state_list_select,
        )
        approach_actions = finalize_actions(select_qpos_traj, ee_state_list_select)
        env.set_current_qpos_agent(grasp_qpos, is_left=is_left)
        env.set_current_xpos_agent(grasp_pose, is_left=is_left)
        close_actions = close_gripper(robot_name, env=env, **kwargs)

        close_state = (
            env.left_arm_current_gripper_state
            if is_left
            else env.right_arm_current_gripper_state
        )
        lift_release_qpos_traj = []
        lift_release_gripper_traj = []
        plan_trajectory(
            env,
            select_arm,
            [grasp_qpos, lift_qpos, release_qpos],
            sample_num,
            close_state,
            lift_release_qpos_traj,
            lift_release_gripper_traj,
        )
        lift_release_actions = finalize_actions(
            lift_release_qpos_traj,
            lift_release_gripper_traj,
        )
        env.set_current_qpos_agent(release_qpos, is_left=is_left)
        env.set_current_xpos_agent(release_pose, is_left=is_left)
        open_actions = open_gripper(robot_name, env=env, **kwargs)

        actions = np.concatenate(
            [approach_actions, close_actions, lift_release_actions, open_actions],
            axis=0,
        )
        log_info(
            f"Selected dedicated upright_object candidate for {robot_name}/{obj_name}: "
            f"{release_label}, actions={len(actions)}.",
            color="green",
        )
        return actions

    raise RuntimeError(
        f"upright_object failed to find a reachable fallen-object strategy for "
        f"{robot_name}/{obj_name}. Tried {len(candidates)} candidates. "
        f"Recent failures: {' | '.join(errors[-3:])}"
    )


def upright_object(
    robot_name: str,
    obj_name: str,
    x: float = None,
    y: float = None,
    pre_grasp_dis: float = 0.08,
    pre_place_dis: float = 0.08,
    env=None,
    force_valid=True,
    **kwargs,
):
    """Recover a fallen object with dedicated side-grasp/upright-release planning."""

    if env is None:
        raise RuntimeError("upright_object requires an environment.")
    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name not in obj_uids:
        log_error(f"No matched object {obj_uids}.")

    env.update_obj_info()
    obj_info = getattr(env, "obj_info", {}).get(obj_name, {})
    target_obj = env.sim.get_rigid_object(obj_name)
    obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    vertical_alignment = float(
        torch.abs(torch.as_tensor(obj_pose[:3, 2][2])).item()
    )
    upright_threshold = float(kwargs.get("upright_object_threshold", 0.65))

    is_left = "right" not in robot_name
    if vertical_alignment >= upright_threshold:
        log_info(
            f"{obj_name} is already upright "
            f"(vertical_alignment={vertical_alignment:.3f}); holding arm state.",
            color="green",
        )
        return _current_arm_hold_action(
            env,
            is_left=is_left,
            n_steps=int(kwargs.get("upright_object_noop_steps", 5)),
        )

    bounds = _object_world_bounds(env, obj_name, obj_pose)
    target_xy_source = bounds["center"] if bounds is not None else obj_pose[:3, 3]
    target_x = float(target_xy_source[0]) if x is None else x
    target_y = float(target_xy_source[1]) if y is None else y
    log_warning(
        f"Uprighting {obj_name} with {robot_name}: "
        f"vertical_alignment={vertical_alignment:.3f}, "
        f"target_xy=({target_x:.3f}, {target_y:.3f})."
    )

    pre_actions = []
    if bool(kwargs.get("upright_object_prepare_start", True)):
        if bool(kwargs.get("upright_object_prepare_open_gripper", True)):
            try:
                pre_actions.append(open_gripper(robot_name, env=env, **kwargs))
            except RuntimeError as exc:
                log_warning(
                    f"upright_object could not open {robot_name} before recovery; "
                    f"continuing from current gripper state. ({exc})"
                )
        try:
            pre_actions.append(back_to_initial_pose(robot_name, env=env, **kwargs))
        except RuntimeError as exc:
            log_warning(
                f"upright_object could not return {robot_name} to initial pose; "
                f"continuing from current arm state. ({exc})"
            )
        if bool(kwargs.get("upright_object_prepare_aim", True)):
            aim_actions = _aim_arm_base_at_xy(
                env=env,
                robot_name=robot_name,
                target_x=target_x,
                target_y=target_y,
                kwargs=kwargs,
            )
            if aim_actions is not None:
                pre_actions.append(aim_actions)

    initial_pose = obj_info.get("initial_pose")
    if initial_pose is not None:
        target_rotation = torch.as_tensor(
            initial_pose,
            dtype=obj_pose.dtype,
            device=obj_pose.device,
        )[:3, :3]
    else:
        target_rotation = _upright_object_rotation_from_pose(obj_pose)
    planning_failure = None
    try:
        primitive = str(kwargs.get("upright_object_physical_primitive", "auto"))
        actions = None
        if primitive in ("top_down_grasp_place", "auto"):
            try:
                actions = _plan_fallen_object_top_down_grasp_place(
                    env=env,
                    robot_name=robot_name,
                    obj_name=obj_name,
                    x=target_x,
                    y=target_y,
                    obj_pose=obj_pose,
                    bounds=bounds,
                    pre_grasp_dis=pre_grasp_dis,
                    pre_place_dis=pre_place_dis,
                    force_valid=force_valid,
                    kwargs=kwargs,
                )
            except RuntimeError as exc:
                if primitive == "top_down_grasp_place":
                    raise
                log_warning(
                    f"upright_object top_down_grasp_place planning failed for "
                    f"{robot_name}/{obj_name}; trying grasp_place. ({exc})"
                )
        if primitive in ("grasp_place", "auto"):
            try:
                if actions is None:
                    actions = _plan_fallen_object_grasp_place(
                        env=env,
                        robot_name=robot_name,
                        obj_name=obj_name,
                        x=target_x,
                        y=target_y,
                        pre_grasp_dis=pre_grasp_dis,
                        pre_place_dis=pre_place_dis,
                        force_valid=force_valid,
                        kwargs=kwargs,
                    )
            except RuntimeError as exc:
                if primitive == "grasp_place":
                    raise
                log_warning(
                    f"upright_object grasp_place planning failed for "
                    f"{robot_name}/{obj_name}; trying lever_sweep. ({exc})"
                )
        if primitive in ("lever_sweep", "auto"):
            try:
                if actions is None:
                    actions = _plan_fallen_object_lever_sweep(
                        env=env,
                        robot_name=robot_name,
                        obj_name=obj_name,
                        x=target_x,
                        y=target_y,
                        obj_pose=obj_pose,
                        bounds=bounds,
                        force_valid=force_valid,
                        kwargs=kwargs,
                    )
            except RuntimeError as exc:
                if primitive == "lever_sweep":
                    raise
                log_warning(
                    f"upright_object lever_sweep planning failed for "
                    f"{robot_name}/{obj_name}; trying top_clamp. ({exc})"
                )
        if actions is None and primitive in ("top_clamp", "auto"):
            try:
                actions = _plan_fallen_object_top_clamp(
                    env=env,
                    robot_name=robot_name,
                    obj_name=obj_name,
                    x=target_x,
                    y=target_y,
                    obj_pose=obj_pose,
                    bounds=bounds,
                    pre_grasp_dis=pre_grasp_dis,
                    pre_place_dis=pre_place_dis,
                    force_valid=force_valid,
                    kwargs=kwargs,
                )
            except RuntimeError as exc:
                if primitive == "top_clamp":
                    raise
                log_warning(
                    f"upright_object top_clamp planning failed for "
                    f"{robot_name}/{obj_name}; trying grasp-release fallback. "
                    f"({exc})"
                )
        if actions is None:
            actions = _plan_dedicated_upright_object(
                env=env,
                robot_name=robot_name,
                obj_name=obj_name,
                x=target_x,
                y=target_y,
                obj_pose=obj_pose,
                bounds=bounds,
                pre_grasp_dis=pre_grasp_dis,
                pre_place_dis=pre_place_dis,
                force_valid=force_valid,
                kwargs=kwargs,
            )
    except RuntimeError as exc:
        if not bool(kwargs.get("upright_object_assisted_correction", False)):
            raise
        planning_failure = str(exc)
        log_warning(
            f"upright_object physical planning failed for {robot_name}/{obj_name}; "
            "using assisted correction after a short hold. "
            f"Failure: {planning_failure}"
        )
        actions = _current_arm_hold_action(
            env,
            is_left=is_left,
            n_steps=int(kwargs.get("upright_object_assisted_correction_hold_steps", 5)),
        )
    if pre_actions:
        actions = np.concatenate([*pre_actions, actions], axis=0)
    _register_pending_upright_object_validation(
        env=env,
        kwargs=kwargs,
        robot_name=robot_name,
        obj_name=obj_name,
        target_x=target_x,
        target_y=target_y,
        target_z=obj_info.get("height", obj_pose[2, 3]),
        target_rotation=target_rotation,
        planning_failure=planning_failure,
    )
    log_info(
        f"Total generated trajectory number for upright object: {len(actions)}.",
        color="green",
    )
    return actions


def _register_pending_upright_object_validation(
    *,
    env,
    kwargs: dict,
    robot_name: str,
    obj_name: str,
    target_x: float,
    target_y: float,
    target_z,
    target_rotation: torch.Tensor,
    planning_failure: str | None = None,
) -> None:
    if not bool(kwargs.get("validate_upright_object_after_action", True)):
        return
    pending = {
        "robot_name": robot_name,
        "obj_name": obj_name,
        "target_x": float(target_x),
        "target_y": float(target_y),
        "target_z": float(torch.as_tensor(target_z).item()),
        "target_rotation": target_rotation.detach().cpu().tolist(),
        "strict_physical": bool(kwargs.get("upright_object_strict_physical", True)),
        "assisted_correction": bool(
            kwargs.get("upright_object_assisted_correction", False)
        ),
        "min_upright_dot": float(
            kwargs.get(
                "upright_object_validation_min_upright_dot",
                kwargs.get("upright_object_threshold", 0.65),
            )
        ),
        "max_xy_error": float(
            kwargs.get("upright_object_validation_max_xy_error", 0.16)
        ),
        "max_height_error": float(
            kwargs.get("upright_object_validation_max_height_error", 0.08)
        ),
        "planning_failure": planning_failure,
    }
    pending_list = getattr(env, "_pending_upright_object_validations", None)
    if not isinstance(pending_list, list):
        pending_list = []
    pending_list.append(pending)
    setattr(env, "_pending_upright_object_validations", pending_list)


def validate_pending_upright_object_after_action(env, kwargs: dict) -> None:
    pending_items = getattr(env, "_pending_upright_object_validations", None)
    if not isinstance(pending_items, list) or not pending_items:
        return
    setattr(env, "_pending_upright_object_validations", [])

    records = getattr(env, "_upright_object_validation_records", None)
    if not isinstance(records, list):
        records = []
    failures = []
    for pending in pending_items:
        record = _evaluate_upright_object_validation(env, pending)
        record["assisted_correction_used"] = False
        if record["success"]:
            log_info(
                f"Upright object validation passed for "
                f"{pending['robot_name']}/{pending['obj_name']}: "
                f"vertical_alignment={record['vertical_alignment']:.3f}.",
                color="green",
            )
            records.append(record)
            continue

        if pending["assisted_correction"]:
            _apply_assisted_upright_object_correction(env, pending)
            corrected_record = _evaluate_upright_object_validation(env, pending)
            corrected_record["physical_success"] = False
            corrected_record["assisted_correction_used"] = True
            if corrected_record["success"]:
                log_warning(
                    f"Assisted upright correction used for "
                    f"{pending['robot_name']}/{pending['obj_name']}. "
                    "This validates recovery strategy routing, not pure physical "
                    "uprighting."
                )
                records.append(corrected_record)
                continue
            record = corrected_record

        records.append(record)
        if pending["strict_physical"]:
            failures.append(
                f"{pending['robot_name']}/{pending['obj_name']}: "
                + "; ".join(record["failure_reasons"])
            )
        else:
            log_warning(
                f"Upright object validation failed for "
                f"{pending['robot_name']}/{pending['obj_name']}: "
                + "; ".join(record["failure_reasons"])
            )

    setattr(env, "_upright_object_validation_records", records)
    setattr(
        env,
        "_upright_object_assisted_correction_used",
        any(record.get("assisted_correction_used", False) for record in records),
    )
    if failures:
        raise RuntimeError(
            "Upright object validation failed: " + " | ".join(failures)
        )


def _evaluate_upright_object_validation(env, pending: dict) -> dict:
    obj_name = pending["obj_name"]
    pose = (
        env.sim.get_rigid_object(obj_name)
        .get_local_pose(to_matrix=True)
        .squeeze(0)
    )
    pose = torch.as_tensor(pose, dtype=torch.float32)
    vertical_alignment = float(torch.abs(pose[:3, 2][2]).item())
    target_xy = torch.tensor(
        [pending["target_x"], pending["target_y"]],
        dtype=pose.dtype,
        device=pose.device,
    )
    xy_error = float(torch.linalg.norm(pose[:2, 3] - target_xy).item())
    height_error = abs(float(pose[2, 3].item()) - float(pending["target_z"]))

    failure_reasons = []
    if vertical_alignment < pending["min_upright_dot"]:
        failure_reasons.append(
            f"vertical_alignment={vertical_alignment:.3f}<"
            f"{pending['min_upright_dot']:.3f}"
        )
    if xy_error > pending["max_xy_error"]:
        failure_reasons.append(
            f"xy_error={xy_error:.3f}>{pending['max_xy_error']:.3f}"
        )
    if height_error > pending["max_height_error"]:
        failure_reasons.append(
            f"height_error={height_error:.3f}>{pending['max_height_error']:.3f}"
        )

    return {
        "robot_name": pending["robot_name"],
        "obj_name": obj_name,
        "physical_success": len(failure_reasons) == 0,
        "success": len(failure_reasons) == 0,
        "failure_reasons": failure_reasons,
        "planning_failure": pending.get("planning_failure"),
        "vertical_alignment": vertical_alignment,
        "xy_error": xy_error,
        "height_error": height_error,
        "position": pose[:3, 3].detach().cpu().tolist(),
    }


def _apply_assisted_upright_object_correction(env, pending: dict) -> None:
    obj_name = pending["obj_name"]
    pose = (
        env.sim.get_rigid_object(obj_name)
        .get_local_pose(to_matrix=True)
        .squeeze(0)
    )
    pose = torch.as_tensor(pose, dtype=torch.float32).clone()
    pose[:3, :3] = torch.as_tensor(
        pending["target_rotation"],
        dtype=pose.dtype,
        device=pose.device,
    )
    pose[:3, 3] = torch.tensor(
        [pending["target_x"], pending["target_y"], pending["target_z"]],
        dtype=pose.dtype,
        device=pose.device,
    )
    env.sim.get_rigid_object(obj_name).set_local_pose(pose.unsqueeze(0))
    env.sim.update(step=100)
    if hasattr(env, "update_obj_info"):
        env.update_obj_info()


def _compiled_action_name(action) -> str | None:
    function = getattr(action, "func", action)
    return getattr(function, "__name__", None)


def _should_sequence_dual_place(left_arm_action, right_arm_action, kwargs) -> bool:
    if not kwargs.get("sequence_dual_place_actions", True):
        return False
    return (
        _compiled_action_name(left_arm_action) == "place_on_table"
        and _compiled_action_name(right_arm_action) == "place_on_table"
    )


def _current_arm_hold_action(env, *, is_left: bool, n_steps: int) -> np.ndarray:
    arm_qpos = env.left_arm_current_qpos if is_left else env.right_arm_current_qpos
    gripper_state = (
        env.left_arm_current_gripper_state
        if is_left
        else env.right_arm_current_gripper_state
    )
    action = finalize_actions(arm_qpos, gripper_state)
    return np.repeat(action[None, :], n_steps, axis=0)


def _compose_full_actions(env, left_arm_action, right_arm_action) -> np.ndarray:
    left_arm_index = env.left_arm_joints + env.left_eef_joints
    right_arm_index = env.right_arm_joints + env.right_eef_joints
    actions = np.zeros(
        (len(left_arm_action), len(env.robot.get_qpos().squeeze(0))),
        dtype=np.float32,
    )
    actions[:, left_arm_index] = left_arm_action
    actions[:, right_arm_index] = right_arm_action
    return actions


def _sequence_dual_place_actions(left_arm_action, right_arm_action, env, kwargs):
    log_info(
        "Sequencing dual place_on_table actions to avoid simultaneous arm/object collision.",
        color="cyan",
    )

    right_place_action = resolve_action(right_arm_action, env, kwargs)
    left_hold_during_right = _current_arm_hold_action(
        env, is_left=True, n_steps=len(right_place_action)
    )
    right_segment = _compose_full_actions(env, left_hold_during_right, right_place_action)

    left_place_action = resolve_action(left_arm_action, env, kwargs)
    right_hold_during_left = _current_arm_hold_action(
        env, is_left=False, n_steps=len(left_place_action)
    )
    left_segment = _compose_full_actions(env, left_place_action, right_hold_during_left)

    return np.concatenate([right_segment, left_segment], axis=0)


def drive(
    left_arm_action=None,
    right_arm_action=None,
    monitor_sequences=None,
    env=None,
    return_result=False,
    interactive_error_injection=False,
    forced_recovery_injection=None,
    **kwargs,
):
    if (
        kwargs.get("sequence_dual_place_actions", True)
        and getattr(
            getattr(left_arm_action, "func", left_arm_action),
            "__name__",
            None,
        )
        == "place_on_table"
        and getattr(
            getattr(right_arm_action, "func", right_arm_action),
            "__name__",
            None,
        )
        == "place_on_table"
    ):
        actions = _sequence_dual_place_actions(
            left_arm_action, right_arm_action, env, kwargs
        )
    else:
        left_arm_action = resolve_action(left_arm_action, env, kwargs)
        right_arm_action = resolve_action(right_arm_action, env, kwargs)

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

    forced_config = (
        forced_recovery_injection
        if isinstance(forced_recovery_injection, dict)
        else None
    )
    force_this_edge = False
    blind_force = bool(forced_config and forced_config.get("blind", False))
    if (
        forced_config is not None
        and forced_config.get("enabled", False)
        and not forced_config.get("_injected", False)
    ):
        if blind_force:
            edge_count = forced_config.get("_seen_edges", 0) + 1
            forced_config["_seen_edges"] = edge_count
            force_this_edge = edge_count == int(forced_config.get("edge_index", 1))
        elif monitor_sequences is not None:
            edge_count = forced_config.get("_seen_monitored_edges", 0) + 1
            forced_config["_seen_monitored_edges"] = edge_count
            force_this_edge = edge_count == int(forced_config.get("edge_index", 1))
        if force_this_edge:
            log_warning(
                f"Forced error injection armed on "
                f"{'blind' if blind_force else 'monitored'} edge #{edge_count}."
            )

    interactive_input = setup_interactive_error_input(interactive_error_injection)
    try:
        for i in tqdm(range(len(actions))):
            action = actions[i]

            env.step(action)

            if interactive_error_requested(interactive_input):
                restore_interactive_error_input(interactive_input)
                interactive_input = None
                inject_interactive_error(env)
                interactive_input = setup_interactive_error_input(
                    interactive_error_injection
                )

            if (
                force_this_edge
                and forced_config is not None
                and not forced_config.get("_injected", False)
                and not forced_config.get("_attempted", False)
            ):
                step_index = int(forced_config.get("step_index", -1))
                should_inject = (
                    i == len(actions) - 1 if step_index < 0 else i >= step_index
                )
                if should_inject:
                    forced_config["_attempted"] = True
                    if blind_force:
                        error_obj = forced_config.get("blind_obj_name", "bottle")
                        relative_error_xyz = forced_config.get("relative_error_xyz")
                        error_type = forced_config.get(
                            "error_type", "misplaced_object"
                        )
                        if error_type == "fallen_object":
                            fallen_object(
                                env,
                                error_obj=error_obj,
                                error_pose=None,
                                relative_error_xyz=relative_error_xyz,
                            )
                        else:
                            misplaced_object(
                                env,
                                error_obj=error_obj,
                                error_pose=None,
                                relative_error_xyz=relative_error_xyz,
                            )
                        injected_monitor = f"blind:{error_obj}"
                        log_warning(
                            f"Injected blind forced error on {error_obj} "
                            f"with error_type={error_type} "
                            f"relative_error_xyz={relative_error_xyz}."
                        )
                    else:
                        injected_monitor = inject_forced_recovery_error(
                            env,
                            monitor_sequences,
                            relative_error_xyz=forced_config.get("relative_error_xyz"),
                            error_type=forced_config.get(
                                "error_type", "misplaced_object"
                            ),
                        )
                    if injected_monitor is not None:
                        forced_config["_injected"] = True
                        forced_config["_injected_at_step"] = i
                        forced_config["_injected_monitor"] = injected_monitor

            if monitor_sequences is not None:
                for monitor_idx, monitor_sequence in enumerate(monitor_sequences):
                    for function in monitor_sequence:
                        result = function()
                        if result == True:
                            env.update_obj_info()
                            function_name = getattr(
                                function.func, "__name__", function.__class__.__name__
                            )
                            log_warning(
                                f"Monitor function {function_name} triggered at step {i}."
                            )

                            if return_result:
                                if forced_config is not None and forced_config.get(
                                    "_injected", False
                                ):
                                    forced_config["_triggered"] = True
                                    forced_config["_triggered_monitor_index"] = (
                                        monitor_idx
                                    )
                                    forced_config["_triggered_monitor_name"] = (
                                        function_name
                                    )
                                    forced_config["_triggered_step"] = i
                                sync_agent_state_from_robot(env)
                                return {
                                    "actions": actions[: i + 1],
                                    "monitor_index": monitor_idx,
                                    "monitor_name": function_name,
                                    "step_index": i,
                                }

                            return actions

            env.update_obj_info()
    finally:
        restore_interactive_error_input(interactive_input)

    public_grasp_validator = globals().get(
        "validate_pending_public_grasp_after_action"
    )
    if public_grasp_validator is not None:
        public_grasp_validator(env, kwargs)
    public_place_validator = globals().get("validate_pending_public_place_after_action")
    if public_place_validator is not None:
        public_place_validator(env, kwargs)
    upright_object_validator = globals().get(
        "validate_pending_upright_object_after_action"
    )
    if upright_object_validator is not None:
        upright_object_validator(env, kwargs)

    if monitor_sequences is not None:
        log_info("No monitor sequences triggered during execution.")
    if return_result:
        return {
            "actions": actions,
            "monitor_index": None,
            "monitor_name": None,
            "step_index": None,
        }
    return actions
