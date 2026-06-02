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

import hashlib
import os
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
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    AntipodalAffordance,
    MoveActionCfg,
    ObjectSemantics,
    PickUpActionCfg,
    PlaceActionCfg,
)
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
    GripperCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GRASP_ANNOTATOR_CACHE_DIR,
)

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
    resolve_arm_side,
    sync_agent_state_from_robot,
)
from embodichain.lab.sim.agent.error_functions import (
    inject_interactive_error,
    interactive_error_requested,
    restore_interactive_error_input,
    setup_interactive_error_input,
)
from embodichain.lab.sim.agent.monitor_functions import *

"""
--------------------------------------------Atom action functions----------------------------------------------------
--------------------------------------------Atom action functions----------------------------------------------------
--------------------------------------------Atom action functions----------------------------------------------------
"""


def _use_public_atomic_actions(kwargs):
    return kwargs.get("use_public_atomic_actions", False) is True


def _require_public_atomic_actions(kwargs):
    return kwargs.get("require_public_atomic_actions", False) is True


def _require_public_grasp_actions(kwargs):
    return (
        _require_public_atomic_actions(kwargs)
        or kwargs.get("require_public_grasp_action", False) is True
    )


def _require_public_non_grasp_actions(kwargs):
    return (
        _require_public_atomic_actions(kwargs)
        or kwargs.get("require_public_non_grasp_actions", False) is True
    )


def _select_arm_parts(env, robot_name):
    is_left = resolve_arm_side(env, robot_name) == "left"
    if hasattr(env, "get_agent_arm_control_part"):
        arm_part = env.get_agent_arm_control_part(is_left)
        hand_part = env.get_agent_eef_control_part(is_left)
    else:
        arm_part = "left_arm" if is_left else "right_arm"
        hand_part = "left_eef" if is_left else "right_eef"
    arm_joints = env.left_arm_joints if is_left else env.right_arm_joints
    eef_joints = env.left_eef_joints if is_left else env.right_eef_joints
    return is_left, arm_part, hand_part, list(arm_joints), list(eef_joints)


def _get_arm_aim_yaw_offset(env, robot_name):
    offset = getattr(env, "arm_aim_yaw_offset", 0.0)
    if isinstance(offset, dict):
        side = resolve_arm_side(env, robot_name)
        offset = offset.get(f"{side}_arm", offset.get(side, 0.0))
    return float(offset)


def _make_motion_generator(env):
    return MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
    )


def _make_atomic_engine(env, cfg):
    return AtomicActionEngine(
        motion_generator=_make_motion_generator(env),
        actions_cfg_list=[cfg],
    )


def _state_to_hand_qpos(state, hand_dof, device):
    if hand_dof <= 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    state = torch.as_tensor(state, dtype=torch.float32, device=device).flatten()
    if state.numel() == 0:
        return torch.zeros(hand_dof, dtype=torch.float32, device=device)
    if state.numel() == hand_dof:
        return state
    if state.numel() == 1:
        return state.repeat(hand_dof)
    if state.numel() > hand_dof:
        return state[:hand_dof]

    repeat_num = int(np.ceil(hand_dof / state.numel()))
    return state.repeat(repeat_num)[:hand_dof]


def _is_current_gripper_open(env, current_state, threshold):
    current_state = torch.as_tensor(
        current_state,
        dtype=env.open_state.dtype,
        device=env.open_state.device,
    ).flatten()
    if current_state.numel() == 0:
        return True

    open_state = _state_to_hand_qpos(
        env.open_state,
        max(current_state.numel(), int(env.open_state.numel())),
        env.open_state.device,
    )[: current_state.numel()]
    return torch.all(torch.abs(current_state - open_state) <= threshold).item()


def _as_2d_action(action, action_name):
    if action is None:
        return None
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2 or len(action) == 0:
        log_error(
            f"{action_name} must have shape (T, D) with T > 0, got {action.shape}."
        )
    return action


def _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids):
    _, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)
    (
        _,
        _,
        select_arm_current_qpos,
        _,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach()
    else:
        trajectory = torch.as_tensor(trajectory)

    if trajectory.dim() == 3:
        trajectory = trajectory[0]
    if trajectory.dim() != 2 or trajectory.shape[0] == 0:
        raise ValueError(
            "Public atomic trajectory must have shape (T, D) or (N, T, D), "
            f"got {trajectory.shape}."
        )

    joint_ids = [int(joint_id) for joint_id in joint_ids]
    if len(joint_ids) != trajectory.shape[-1]:
        raise ValueError(
            f"Public atomic joint_ids length {len(joint_ids)} does not match "
            f"trajectory width {trajectory.shape[-1]}."
        )

    device = trajectory.device
    current_arm_qpos = torch.as_tensor(
        select_arm_current_qpos, dtype=torch.float32, device=device
    ).flatten()
    current_hand_qpos = _state_to_hand_qpos(
        select_arm_current_gripper_state,
        len(eef_joints),
        device,
    )
    agent_action = torch.cat([current_arm_qpos, current_hand_qpos], dim=0)
    agent_action = agent_action.unsqueeze(0).repeat(trajectory.shape[0], 1)

    joint_id_to_col = {joint_id: col for col, joint_id in enumerate(joint_ids)}
    for out_col, joint_id in enumerate(arm_joints + eef_joints):
        if joint_id in joint_id_to_col:
            agent_action[:, out_col] = trajectory[:, joint_id_to_col[joint_id]]

    return agent_action.detach().cpu().numpy().astype(np.float32)


def _sync_agent_state_from_public_action(env, robot_name, action_np):
    if action_np is None or len(action_np) == 0:
        raise ValueError("Public atomic action is empty; cannot sync agent state.")

    is_left, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)
    (
        _,
        _,
        _,
        _,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    final_action = np.asarray(action_np[-1], dtype=np.float32)
    arm_dof = len(arm_joints)
    arm_qpos = torch.as_tensor(
        final_action[:arm_dof],
        dtype=torch.float32,
        device=env.robot.device,
    )
    env.set_current_qpos_agent(arm_qpos, is_left=is_left)
    env.set_current_xpos_agent(
        env.get_arm_fk(qpos=arm_qpos, is_left=is_left),
        is_left=is_left,
    )

    if len(eef_joints) == 0:
        return

    eef_qpos = final_action[arm_dof : arm_dof + len(eef_joints)]
    state_dof = max(int(torch.as_tensor(select_arm_current_gripper_state).numel()), 1)
    if len(eef_qpos) >= state_dof:
        gripper_qpos = eef_qpos[:state_dof]
    else:
        gripper_qpos = np.resize(eef_qpos, state_dof)

    current_gripper_state = torch.as_tensor(select_arm_current_gripper_state)
    env.set_current_gripper_state_agent(
        torch.as_tensor(
            gripper_qpos,
            dtype=current_gripper_state.dtype,
            device=current_gripper_state.device,
        ),
        is_left=is_left,
    )


def _sync_agent_state_from_public_qpos_action(
    env,
    robot_name,
    action_np,
    control_part,
):
    if action_np is None or len(action_np) == 0:
        raise ValueError("Public atomic action is empty; cannot sync agent state.")

    is_left, arm_part, hand_part, arm_joints, eef_joints = _select_arm_parts(
        env, robot_name
    )
    final_action = np.asarray(action_np[-1], dtype=np.float32)
    arm_dof = len(arm_joints)

    if control_part == arm_part:
        arm_qpos = torch.as_tensor(
            final_action[:arm_dof],
            dtype=torch.float32,
            device=env.robot.device,
        )
        env.set_current_qpos_agent(arm_qpos, is_left=is_left)
        env.set_current_xpos_agent(
            env.get_arm_fk(qpos=arm_qpos, is_left=is_left),
            is_left=is_left,
        )
        return

    if control_part == hand_part:
        if len(eef_joints) == 0:
            return
        (
            _,
            _,
            _,
            _,
            select_arm_current_gripper_state,
        ) = get_arm_states(env, robot_name)
        eef_qpos = final_action[arm_dof : arm_dof + len(eef_joints)]
        state_dof = max(
            int(torch.as_tensor(select_arm_current_gripper_state).numel()), 1
        )
        if len(eef_qpos) >= state_dof:
            gripper_qpos = eef_qpos[:state_dof]
        else:
            gripper_qpos = np.resize(eef_qpos, state_dof)

        current_gripper_state = torch.as_tensor(select_arm_current_gripper_state)
        env.set_current_gripper_state_agent(
            torch.as_tensor(
                gripper_qpos,
                dtype=current_gripper_state.dtype,
                device=current_gripper_state.device,
            ),
            is_left=is_left,
        )
        return

    raise ValueError(f"Unsupported public qpos control_part: {control_part}.")


def _try_public_qpos_move_action(
    *,
    env,
    robot_name: str,
    control_part: str,
    target_qpos,
    start_qpos,
    sample_num: int,
    kwargs: dict,
    log_name: str,
):
    def _fail(message, exc=None):
        suffix = f" ({exc})" if exc is not None else ""
        log_warning(f"{message} for {log_name}; fallback to legacy logic.{suffix}")
        if _require_public_non_grasp_actions(kwargs):
            raise RuntimeError(message) from exc
        return None

    if env is None:
        if _require_public_non_grasp_actions(kwargs):
            return _fail("Public qpos MoveAction requires env")
        return None
    if not _use_public_atomic_actions(kwargs):
        if _require_public_non_grasp_actions(kwargs):
            return _fail(
                "Public qpos MoveAction requires use_public_atomic_actions=True"
            )
        return None

    try:
        device = env.robot.device
        cfg = MoveActionCfg(
            name="move",
            control_part=control_part,
            sample_interval=int(sample_num),
        )
        engine = _make_atomic_engine(env, cfg)
        action = engine._actions["move"]
        is_success, trajectory, joint_ids = action.execute(
            target=torch.as_tensor(target_qpos, dtype=torch.float32, device=device),
            start_qpos=torch.as_tensor(start_qpos, dtype=torch.float32, device=device),
        )
        if not is_success:
            return _fail("Public qpos MoveAction failed")

        action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
        _sync_agent_state_from_public_qpos_action(
            env, robot_name, action_np, control_part
        )
        return action_np
    except Exception as e:
        return _fail("Public qpos MoveAction failed", e)


def _semantic_public_grasp_enabled(kwargs):
    return (
        kwargs.get("use_public_grasp_semantics", False) is True
        or kwargs.get("public_grasp_strategy", None) == "semantic"
    )


def _cfg_supported_kwargs(cfg_cls, values):
    supported = set()
    for cls in reversed(cfg_cls.__mro__):
        supported.update(getattr(cls, "__annotations__", {}).keys())
    return {key: value for key, value in values.items() if key in supported}


def _public_grasp_cache_path(mesh_vertices, mesh_triangles):
    vert_bytes = mesh_vertices.to("cpu").numpy().tobytes()
    face_bytes = mesh_triangles.to("cpu").numpy().tobytes()
    md5_hash = hashlib.md5(vert_bytes + face_bytes).hexdigest()
    return os.path.join(GRASP_ANNOTATOR_CACHE_DIR, f"antipodal_cache_{md5_hash}.npy")


def _build_public_grasp_semantics(env, obj_name: str, kwargs: dict):
    """Build ObjectSemantics for tutorial-style AntipodalAffordance grasp."""
    target_obj = env.sim.get_rigid_object(obj_name)
    if target_obj is None:
        raise ValueError(f"No rigid object found for {obj_name}.")

    mesh_vertices = target_obj.get_vertices(env_ids=[0], scale=True)[0]
    mesh_triangles = target_obj.get_triangles(env_ids=[0])[0]
    mesh_vertices = torch.as_tensor(mesh_vertices, dtype=torch.float32)
    mesh_triangles = torch.as_tensor(mesh_triangles, dtype=torch.int64)
    if (
        mesh_vertices.numel() == 0
        or mesh_triangles.numel() == 0
        or mesh_vertices.shape[-1] != 3
        or mesh_triangles.shape[-1] != 3
    ):
        raise ValueError(f"Object {obj_name} has empty or invalid mesh geometry.")

    allow_annotation = bool(kwargs.get("allow_public_grasp_annotation", False))
    force_reannotate = bool(kwargs.get("force_public_grasp_reannotate", False))
    if force_reannotate and not allow_annotation:
        log_warning(
            "Public semantic grasp requested force re-annotation without "
            "allow_public_grasp_annotation=True; fallback to legacy grasp."
        )
        return None

    cache_path = _public_grasp_cache_path(mesh_vertices, mesh_triangles)
    if not os.path.exists(cache_path) and not allow_annotation:
        log_warning(
            "Public semantic grasp cache is missing and annotation is disabled; "
            "fallback to legacy grasp."
        )
        return None

    antipodal_sampler_cfg = AntipodalSamplerCfg(
        **_cfg_supported_kwargs(
            AntipodalSamplerCfg,
            {
                "n_sample": int(kwargs.get("grasp_antipodal_n_sample", 20000)),
                "max_angle": kwargs.get("grasp_antipodal_max_angle", np.pi / 12),
                "max_length": kwargs.get("grasp_max_open_length", 0.088),
                "min_length": kwargs.get("grasp_min_open_length", 0.003),
            },
        )
    )
    generator_cfg = GraspGeneratorCfg(
        **_cfg_supported_kwargs(
            GraspGeneratorCfg,
            {
                "viser_port": int(kwargs.get("public_grasp_viser_port", 11801)),
                "antipodal_sampler_cfg": antipodal_sampler_cfg,
                "max_deviation_angle": kwargs.get(
                    "grasp_max_deviation_angle", np.pi / 6
                ),
            },
        )
    )
    gripper_collision_cfg = GripperCollisionCfg(
        **_cfg_supported_kwargs(
            GripperCollisionCfg,
            {
                "max_open_length": kwargs.get("grasp_max_open_length", 0.088),
                "finger_length": kwargs.get("grasp_finger_length", 0.078),
                "point_sample_dense": kwargs.get("grasp_point_sample_dense", 0.012),
            },
        )
    )

    affordance = AntipodalAffordance(
        object_label=obj_name,
        force_reannotate=force_reannotate,
        custom_config={
            "gripper_collision_cfg": gripper_collision_cfg,
            "generator_cfg": generator_cfg,
        },
    )
    return ObjectSemantics(
        label=obj_name,
        geometry={
            "mesh_vertices": mesh_vertices,
            "mesh_triangles": mesh_triangles,
        },
        affordance=affordance,
        entity=target_obj,
    )


def _try_public_semantic_grasp_action(
    *,
    env,
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float,
    kwargs: dict,
):
    def _fail(message, exc=None):
        suffix = f" ({exc})" if exc is not None else ""
        log_warning(f"{message}; fallback to legacy grasp.{suffix}")
        if _require_public_grasp_actions(kwargs):
            raise RuntimeError(message) from exc
        return None

    if env is None:
        return _fail("Public semantic grasp requires env")
    if not _use_public_atomic_actions(kwargs):
        if _require_public_grasp_actions(kwargs):
            return _fail(
                "Public semantic grasp requires use_public_atomic_actions=True"
            )
        return None
    if not _semantic_public_grasp_enabled(kwargs):
        if kwargs.get("use_public_grasp_action", False) is True:
            return None
        if _require_public_grasp_actions(kwargs):
            return _fail("Public semantic grasp is disabled")
        return None

    try:
        target = _build_public_grasp_semantics(env, obj_name, kwargs)
        if target is None:
            return _fail("Public semantic grasp target is unavailable")

        is_left, arm_part, hand_part, arm_joints, _ = _select_arm_parts(env, robot_name)
        device = env.robot.device
        hand_dof = len(env.left_eef_joints if is_left else env.right_eef_joints)
        approach_direction = torch.as_tensor(
            kwargs.get("public_grasp_approach_direction", [0, 0, -1]),
            dtype=torch.float32,
            device=device,
        )
        cfg = PickUpActionCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_open_qpos=_state_to_hand_qpos(env.open_state, hand_dof, device),
            hand_close_qpos=_state_to_hand_qpos(env.close_state, hand_dof, device),
            pre_grasp_distance=pre_grasp_dis,
            lift_height=kwargs.get("lift_height", 0.1),
            approach_direction=approach_direction,
            sample_interval=int(
                kwargs.get("sample_interval", kwargs.get("sample_num", 80))
            ),
            hand_interp_steps=int(kwargs.get("hand_interp_steps", 5)),
        )
        engine = _make_atomic_engine(env, cfg)
        action = engine._actions[cfg.name]
        start_qpos = (
            env.left_arm_current_qpos if is_left else env.right_arm_current_qpos
        )
        start_qpos = torch.as_tensor(
            start_qpos, dtype=torch.float32, device=device
        ).reshape(1, len(arm_joints))

        is_success, trajectory, joint_ids = action.execute(
            target=target,
            start_qpos=start_qpos,
        )
        if not is_success:
            return _fail("Public semantic grasp action failed")

        action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
        _sync_agent_state_from_public_action(env, robot_name, action_np)
        log_info(
            "Using public semantic grasp action with "
            f"{len(action_np)} trajectory steps.",
            color="green",
        )
        return action_np
    except Exception as e:
        return _fail("Public semantic grasp action failed", e)


def _try_public_move_action(
    env, robot_name, target_pose, public_sample_num, action_name, **kwargs
):
    def _fail(message, exc=None):
        suffix = f" ({exc})" if exc is not None else ""
        log_warning(f"{message}; fallback to legacy logic.{suffix}")
        if _require_public_non_grasp_actions(kwargs):
            raise RuntimeError(message) from exc
        return None

    if env is None:
        if _require_public_non_grasp_actions(kwargs):
            return _fail("Public MoveAction requires env")
        return None
    if not _use_public_atomic_actions(kwargs):
        if _require_public_non_grasp_actions(kwargs):
            return _fail("Public MoveAction requires use_public_atomic_actions=True")
        return None

    try:
        _, arm_part, _, _, _ = _select_arm_parts(env, robot_name)
        (
            _,
            _,
            select_arm_current_qpos,
            _,
            _,
        ) = get_arm_states(env, robot_name)
        target_pose = torch.as_tensor(
            target_pose, dtype=torch.float32, device=env.robot.device
        )
        start_qpos = torch.as_tensor(
            select_arm_current_qpos,
            dtype=torch.float32,
            device=env.robot.device,
        )

        cfg = MoveActionCfg(
            control_part=arm_part,
            sample_interval=int(public_sample_num),
        )
        engine = _make_atomic_engine(env, cfg)
        action = engine._actions[cfg.name]
        is_success, trajectory, joint_ids = action.execute(
            target=target_pose,
            start_qpos=start_qpos,
        )
        if not is_success:
            return _fail(f"Public atomic action failed for {action_name}")

        action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
        _sync_agent_state_from_public_action(env, robot_name, action_np)
        return action_np
    except Exception as e:
        return _fail(f"Public atomic action failed for {action_name}", e)


def _try_public_pickup_action(
    env,
    robot_name,
    obj_name,
    target_obj_pose,
    pre_grasp_dis,
    **kwargs,
):
    def _fail(message, exc=None):
        suffix = f" ({exc})" if exc is not None else ""
        log_warning(f"{message}; fallback to legacy logic.{suffix}")
        if _require_public_grasp_actions(kwargs):
            raise RuntimeError(message) from exc
        return None

    if not _use_public_atomic_actions(kwargs):
        if _require_public_grasp_actions(kwargs):
            return _fail("Public PickUpAction requires use_public_atomic_actions=True")
        return None
    if kwargs.get("use_public_grasp_action", False) is not True:
        if _require_public_grasp_actions(kwargs):
            return _fail("Public PickUpAction is disabled")
        return None

    try:
        is_left, arm_part, hand_part, _, eef_joints = _select_arm_parts(env, robot_name)
        (
            _,
            _,
            select_arm_current_qpos,
            _,
            _,
        ) = get_arm_states(env, robot_name)

        grasp_pose_object = env.obj_info.get(obj_name, {}).get("grasp_pose_obj")
        if grasp_pose_object is None:
            raise ValueError(f"No grasp_pose_obj found for object {obj_name}.")

        device = env.robot.device
        target_obj_pose = torch.as_tensor(
            target_obj_pose, dtype=torch.float32, device=device
        )
        grasp_pose_object = torch.as_tensor(
            grasp_pose_object, dtype=torch.float32, device=device
        )

        select_arm_base_pose = (
            env.left_arm_base_pose if is_left else env.right_arm_base_pose
        )
        select_arm_base_pose = torch.as_tensor(
            select_arm_base_pose, dtype=torch.float32, device=device
        )
        delta_xy = target_obj_pose[:2, 3] - select_arm_base_pose[:2, 3]
        aim_horizontal_angle = float(
            torch.atan2(delta_xy[1], delta_xy[0]).detach().cpu()
        ) + _get_arm_aim_yaw_offset(env, robot_name)
        if bool((grasp_pose_object[0, 2] > 0.5).item()):
            target_obj_pose = torch.as_tensor(
                get_rotation_replaced_pose(
                    target_obj_pose.detach().cpu().numpy(),
                    aim_horizontal_angle,
                    "z",
                    "intrinsic",
                ),
                dtype=torch.float32,
                device=device,
            )
        grasp_pose = target_obj_pose @ grasp_pose_object

        cfg = PickUpActionCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_open_qpos=_state_to_hand_qpos(env.open_state, len(eef_joints), device),
            hand_close_qpos=_state_to_hand_qpos(
                env.close_state, len(eef_joints), device
            ),
            pre_grasp_distance=pre_grasp_dis,
            sample_interval=int(kwargs.get("sample_num", 80)),
        )
        engine = _make_atomic_engine(env, cfg)
        action = engine._actions[cfg.name]
        is_success, trajectory, joint_ids = action.execute(
            target=grasp_pose,
            start_qpos=torch.as_tensor(
                select_arm_current_qpos, dtype=torch.float32, device=device
            ),
        )
        if not is_success:
            return _fail("Public atomic action failed for grasp")

        action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
        _sync_agent_state_from_public_action(env, robot_name, action_np)
        return action_np
    except Exception as e:
        return _fail("Public atomic action failed for grasp", e)


def _try_public_place_action(
    env,
    robot_name,
    target_pose,
    pre_place_dis,
    **kwargs,
):
    def _fail(message, exc=None):
        suffix = f" ({exc})" if exc is not None else ""
        log_warning(f"{message}; fallback to legacy logic.{suffix}")
        if _require_public_atomic_actions(kwargs):
            raise RuntimeError(message) from exc
        return None

    if not _use_public_atomic_actions(kwargs):
        if _require_public_atomic_actions(kwargs):
            return _fail("Public PlaceAction requires use_public_atomic_actions=True")
        return None
    if kwargs.get("use_public_place_action", False) is not True:
        if _require_public_atomic_actions(kwargs):
            return _fail("Public PlaceAction is disabled")
        return None

    try:
        _, arm_part, hand_part, _, eef_joints = _select_arm_parts(env, robot_name)
        (
            _,
            _,
            select_arm_current_qpos,
            _,
            _,
        ) = get_arm_states(env, robot_name)
        device = env.robot.device

        cfg = PlaceActionCfg(
            control_part=arm_part,
            hand_control_part=hand_part,
            hand_open_qpos=_state_to_hand_qpos(env.open_state, len(eef_joints), device),
            hand_close_qpos=_state_to_hand_qpos(
                env.close_state, len(eef_joints), device
            ),
            lift_height=pre_place_dis,
            sample_interval=int(kwargs.get("sample_num", 80)),
        )
        engine = _make_atomic_engine(env, cfg)
        action = engine._actions[cfg.name]
        is_success, trajectory, joint_ids = action.execute(
            target=torch.as_tensor(target_pose, dtype=torch.float32, device=device),
            start_qpos=torch.as_tensor(
                select_arm_current_qpos, dtype=torch.float32, device=device
            ),
        )
        if not is_success:
            return _fail("Public atomic action failed for place on table")

        action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
        _sync_agent_state_from_public_action(env, robot_name, action_np)
        return action_np
    except Exception as e:
        return _fail("Public atomic action failed for place on table", e)


def move_to_target_pose(
    robot_name: str,
    target_pose=None,
    sample_num: int = 20,
    env=None,
    force_valid=False,
    **kwargs,
):
    select_qpos_traj = []
    ee_state_list_select = []

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        select_arm_current_pose,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    public_actions = _try_public_move_action(
        env,
        robot_name,
        target_pose,
        sample_num,
        "move to target",
        **kwargs,
    )
    if public_actions is not None:
        log_info(
            "Total generated trajectory number for move to target: "
            f"{len(public_actions)}.",
            color="green",
        )
        return public_actions

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

    semantic_public_actions = _try_public_semantic_grasp_action(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        pre_grasp_dis=pre_grasp_dis,
        kwargs=kwargs,
    )
    if semantic_public_actions is not None:
        log_info(
            "Total generated trajectory number for public semantic grasp: "
            f"{len(semantic_public_actions)}.",
            color="green",
        )
        return semantic_public_actions

    if not _semantic_public_grasp_enabled(kwargs):
        public_actions = _try_public_pickup_action(
            env,
            robot_name,
            obj_name,
            target_obj_pose,
            pre_grasp_dis,
            **kwargs,
        )
        if public_actions is not None:
            log_info(
                f"Total generated trajectory number for grasp: {len(public_actions)}.",
                color="green",
            )
            return public_actions

    # Open the gripper if currently closed
    actions = None
    select_arm_current_gripper_state = (
        env.left_arm_current_gripper_state
        if "left" in robot_name
        else env.right_arm_current_gripper_state
    )
    if not _is_current_gripper_open(
        env,
        select_arm_current_gripper_state,
        kwargs.get("open_threshold", 0.01),
    ):
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
        legacy_move_kwargs = dict(kwargs)
        legacy_move_kwargs["use_public_atomic_actions"] = False
        legacy_move_kwargs.pop("sample_num", None)
        back_actions = move_by_relative_offset(
            robot_name=robot_name,
            dx=0.0,
            dy=0.0,
            dz=-delta,
            env=env,
            force_valid=force_valid,
            mode="intrinsic",
            sample_num=15,
            **legacy_move_kwargs,
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
    aim_horizontal_angle = float(torch.atan2(dy, dx).detach().cpu())
    aim_horizontal_angle += _get_arm_aim_yaw_offset(env, robot_name)
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

    (
        _,
        _,
        _,
        select_arm_current_pose,
        _,
    ) = get_arm_states(env, robot_name)
    place_pose = deepcopy(select_arm_current_pose)
    if x is not None:
        place_pose[0, 3] = x
    if y is not None:
        place_pose[1, 3] = y
    place_pose[2, 3] = height

    public_actions = _try_public_place_action(
        env,
        robot_name,
        place_pose,
        pre_place_dis,
        **kwargs,
    )
    if public_actions is not None:
        log_info(
            "Total generated trajectory number for place on table: "
            f"{len(public_actions)}.",
            color="green",
        )
        return public_actions

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

    public_actions = _try_public_move_action(
        env,
        robot_name,
        move_target_pose,
        kwargs.get("sample_num", 30),
        "move relative to object",
        **kwargs,
    )
    if public_actions is not None:
        log_info(
            "Total generated trajectory number for move relative to object: "
            f"{len(public_actions)}.",
            color="green",
        )
        return public_actions

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

    public_actions = _try_public_move_action(
        env,
        robot_name,
        move_pose,
        kwargs.get("sample_num", 30),
        "move to absolute position",
        **kwargs,
    )
    if public_actions is not None:
        log_info(
            "Total generated trajectory number for move to absolute position: "
            f"{len(public_actions)}.",
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

    public_actions = _try_public_move_action(
        env,
        robot_name,
        move_pose,
        kwargs.get("sample_num", 20),
        "move by relative offset",
        **kwargs,
    )
    if public_actions is not None:
        log_info(
            "Total generated trajectory number for move by relative offset: "
            f"{len(public_actions)}.",
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
    target_qpos = torch.as_tensor(target_qpos, dtype=torch.float32)

    sample_num = kwargs.get("sample_num", 30)
    if _use_public_atomic_actions(kwargs):
        public_actions = _try_public_qpos_move_action(
            env=env,
            robot_name=robot_name,
            control_part=select_arm,
            target_qpos=target_qpos.to(env.robot.device),
            start_qpos=torch.as_tensor(
                select_arm_current_qpos,
                dtype=torch.float32,
                device=env.robot.device,
            ).reshape(1, -1),
            sample_num=sample_num,
            kwargs=kwargs,
            log_name="back to initial pose",
        )
        if public_actions is not None:
            log_info(
                "Total generated trajectory number for back to initial pose: "
                f"{len(public_actions)}.",
                color="green",
            )
            return public_actions

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
    sample_num = kwargs.get("sample_num", 20)

    if _use_public_atomic_actions(kwargs):
        public_actions = _try_public_qpos_move_action(
            env=env,
            robot_name=robot_name,
            control_part=select_arm,
            target_qpos=rotated_qpos,
            start_qpos=torch.as_tensor(
                select_arm_current_qpos,
                dtype=torch.float32,
                device=env.robot.device,
            ).reshape(1, -1),
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
    if _use_public_atomic_actions(kwargs):
        _, _, hand_part, _, eef_joints = _select_arm_parts(env, robot_name)
        hand_dof = len(eef_joints)
        public_actions = _try_public_qpos_move_action(
            env=env,
            robot_name=robot_name,
            control_part=hand_part,
            target_qpos=_state_to_hand_qpos(
                env.close_state, hand_dof, env.robot.device
            ),
            start_qpos=_state_to_hand_qpos(
                select_arm_current_gripper_state, hand_dof, env.robot.device
            ).reshape(1, hand_dof),
            sample_num=sample_num,
            kwargs=kwargs,
            log_name="close gripper",
        )
        if public_actions is not None:
            log_info(
                f"Total generated trajectory number for close gripper: {len(public_actions)}.",
                color="green",
            )
            return public_actions

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

    if _is_current_gripper_open(
        env,
        select_arm_current_gripper_state,
        kwargs.get("open_threshold", 0.01),
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
    if _use_public_atomic_actions(kwargs):
        _, _, hand_part, _, eef_joints = _select_arm_parts(env, robot_name)
        hand_dof = len(eef_joints)
        public_actions = _try_public_qpos_move_action(
            env=env,
            robot_name=robot_name,
            control_part=hand_part,
            target_qpos=_state_to_hand_qpos(env.open_state, hand_dof, env.robot.device),
            start_qpos=_state_to_hand_qpos(
                select_arm_current_gripper_state, hand_dof, env.robot.device
            ).reshape(1, hand_dof),
            sample_num=sample_num,
            kwargs=kwargs,
            log_name="open gripper",
        )
        if public_actions is not None:
            log_info(
                f"Total generated trajectory number for open gripper: {len(public_actions)}.",
                color="green",
            )
            return public_actions

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


def drive(
    left_arm_action=None,
    right_arm_action=None,
    monitor_sequences=None,
    env=None,
    return_result=False,
    interactive_error_injection=False,
    **kwargs,
):
    left_arm_action = resolve_action(left_arm_action, env, kwargs)
    right_arm_action = resolve_action(right_arm_action, env, kwargs)

    left_arm_action = _as_2d_action(left_arm_action, "left_arm_action")
    right_arm_action = _as_2d_action(right_arm_action, "right_arm_action")
    arm_actions = {"left": left_arm_action, "right": right_arm_action}

    if all(action is None for action in arm_actions.values()):
        log_error("At least one arm action should be provided.")

    action_len = max(
        len(action) for action in arm_actions.values() if action is not None
    )
    for side, action in arm_actions.items():
        if action is not None and len(action) < action_len:
            diff = action_len - len(action)
            padding = np.repeat(action[-1:], diff, axis=0)
            arm_actions[side] = np.concatenate([action, padding], axis=0)

    current_qpos = (
        env.robot.get_qpos().squeeze(0).detach().cpu().numpy().astype(np.float32)
    )
    actions = np.repeat(current_qpos[None, :], action_len, axis=0)

    for side, action in arm_actions.items():
        if action is None:
            continue

        arm_index = list(getattr(env, f"{side}_arm_joints", [])) + list(
            getattr(env, f"{side}_eef_joints", [])
        )
        if not arm_index:
            log_error(
                f"{side}_arm_action was provided, but {side}_arm is not configured "
                f"on robot control parts {getattr(env.robot, 'control_parts', None)}."
            )
        if action.shape[-1] != len(arm_index):
            log_error(
                f"{side}_arm_action width {action.shape[-1]} does not match "
                f"{side}_arm joints plus eef joints ({len(arm_index)})."
            )
        actions[:, arm_index] = action

    actions = torch.from_numpy(actions).to(dtype=torch.float32).unsqueeze(1)
    actions = list(actions.unbind(dim=0))

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
