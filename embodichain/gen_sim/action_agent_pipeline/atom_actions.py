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
from embodichain.lab.gym.utils.misc import get_rotation_replaced_pose
from embodichain.utils.math import get_offset_pose
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
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
from embodichain.gen_sim.action_agent_pipeline.atom_action_utils import (
    get_arm_states,
    resolve_action,
    resolve_arm_side,
    sync_agent_state_from_robot,
)
from embodichain.gen_sim.action_agent_pipeline.error_functions import (
    inject_interactive_error,
    interactive_error_requested,
    restore_interactive_error_input,
    setup_interactive_error_input,
)
from embodichain.gen_sim.action_agent_pipeline.monitor_functions import *

"""
--------------------------------------------Atom action functions----------------------------------------------------
--------------------------------------------Atom action functions----------------------------------------------------
--------------------------------------------Atom action functions----------------------------------------------------
"""


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


def _log_public_atomic_backend(
    *,
    wrapper_name: str,
    action,
    cfg,
    target_kind: str,
    control_part: str,
    steps: int,
):
    log_info(
        "Using public AtomicAction backend: "
        f"wrapper={wrapper_name}, "
        f"action={action.__class__.__name__}, "
        f"cfg={cfg.__class__.__name__}, "
        f"cfg_name={cfg.name}, "
        f"target={target_kind}, "
        f"control_part={control_part}, "
        f"steps={steps}.",
        color="green",
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


def _current_agent_action(env, robot_name):
    _, _, arm_qpos, _, gripper_state = get_arm_states(env, robot_name)
    _, _, _, arm_joints, eef_joints = _select_arm_parts(env, robot_name)
    device = env.robot.device
    action = torch.cat(
        [
            torch.as_tensor(arm_qpos, dtype=torch.float32, device=device).flatten(),
            _state_to_hand_qpos(gripper_state, len(eef_joints), device),
        ],
        dim=0,
    )
    expected_width = len(arm_joints) + len(eef_joints)
    if action.numel() != expected_width:
        raise ValueError(
            f"Current agent action width {action.numel()} does not match "
            f"configured arm+eef joints ({expected_width})."
        )
    return action.unsqueeze(0).detach().cpu().numpy().astype(np.float32)


def _append_hold_steps(action_np, hold_steps: int, log_name: str):
    hold_steps = int(hold_steps)
    if hold_steps <= 0:
        return action_np
    if action_np is None or len(action_np) == 0:
        raise ValueError(f"{log_name} action is empty; cannot append hold steps.")

    hold_actions = np.repeat(action_np[-1:], hold_steps, axis=0)
    action_np = np.concatenate([action_np, hold_actions], axis=0)
    log_info(
        f"Append {hold_steps} hold steps after {log_name}; "
        f"total trajectory length is {len(action_np)}.",
        color="green",
    )
    return action_np


def _public_qpos_move_action(
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
    if env is None:
        raise ValueError("Public qpos MoveAction requires env.")

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
        raise RuntimeError(f"Public qpos MoveAction failed for {log_name}.")

    action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
    _log_public_atomic_backend(
        wrapper_name=log_name,
        action=action,
        cfg=cfg,
        target_kind="qpos",
        control_part=control_part,
        steps=len(action_np),
    )
    _sync_agent_state_from_public_qpos_action(env, robot_name, action_np, control_part)
    return action_np


def _semantic_public_grasp_enabled(kwargs):
    return (
        kwargs.get("use_public_grasp_semantics", True) is True
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


def _rigid_object_mesh_path(obj) -> str | None:
    shape = getattr(getattr(obj, "cfg", None), "shape", None)
    fpath = getattr(shape, "fpath", None)
    return str(fpath) if fpath else None


def _rigid_object_body_scale(obj) -> list[float] | None:
    body_scale = obj.get_body_scale(env_ids=[0])[0]
    return body_scale.detach().to("cpu", dtype=torch.float32).tolist()


def _public_grasp_max_decomposition_hulls(target_obj, kwargs: dict) -> int:
    if "grasp_max_decomposition_hulls" in kwargs:
        return int(kwargs["grasp_max_decomposition_hulls"])

    max_convex_hull_num = getattr(
        getattr(target_obj, "cfg", None), "max_convex_hull_num", None
    )
    if max_convex_hull_num is not None and int(max_convex_hull_num) > 1:
        return int(max_convex_hull_num)
    return 8


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

    allow_annotation = bool(kwargs.get("allow_public_grasp_annotation", True))
    force_reannotate = bool(kwargs.get("force_public_grasp_reannotate", False))
    if force_reannotate and not allow_annotation:
        raise RuntimeError(
            "Public semantic grasp requested force re-annotation without "
            "allow_public_grasp_annotation=True."
        )

    cache_path = _public_grasp_cache_path(mesh_vertices, mesh_triangles)
    if not os.path.exists(cache_path) and not allow_annotation:
        raise RuntimeError(
            "Public semantic grasp cache is missing and annotation is disabled; "
            "set allow_public_grasp_annotation=True or use public grasp_pose_obj mode."
        )

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
    max_decomposition_hulls = _public_grasp_max_decomposition_hulls(target_obj, kwargs)
    gripper_collision_cfg = GripperCollisionCfg(
        **_cfg_supported_kwargs(
            GripperCollisionCfg,
            {
                "max_open_length": kwargs.get("grasp_max_open_length", 0.088),
                "finger_length": kwargs.get("grasp_finger_length", 0.078),
                "point_sample_dense": kwargs.get("grasp_point_sample_dense", 0.012),
                "max_decomposition_hulls": max_decomposition_hulls,
                "env_coacd_source_mesh_path": _rigid_object_mesh_path(target_obj),
                "env_coacd_body_scale": _rigid_object_body_scale(target_obj),
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


def _public_semantic_grasp_action(
    *,
    env,
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float,
    kwargs: dict,
):
    if env is None:
        raise ValueError("Public semantic grasp requires env.")
    if not _semantic_public_grasp_enabled(kwargs):
        raise RuntimeError("Public semantic grasp is disabled.")

    target = _build_public_grasp_semantics(env, obj_name, kwargs)
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
    start_qpos = env.left_arm_current_qpos if is_left else env.right_arm_current_qpos
    start_qpos = torch.as_tensor(
        start_qpos, dtype=torch.float32, device=device
    ).reshape(1, len(arm_joints))

    is_success, trajectory, joint_ids = action.execute(
        target=target,
        start_qpos=start_qpos,
    )
    if not is_success:
        raise RuntimeError("Public semantic grasp action failed.")

    action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
    _log_public_atomic_backend(
        wrapper_name="grasp",
        action=action,
        cfg=cfg,
        target_kind="ObjectSemantics(AntipodalAffordance)",
        control_part=arm_part,
        steps=len(action_np),
    )
    _sync_agent_state_from_public_action(env, robot_name, action_np)
    return action_np


def _public_move_action(
    env, robot_name, target_pose, public_sample_num, action_name, **kwargs
):
    if env is None:
        raise ValueError("Public MoveAction requires env.")

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
        raise RuntimeError(f"Public atomic action failed for {action_name}.")

    action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
    _log_public_atomic_backend(
        wrapper_name=action_name,
        action=action,
        cfg=cfg,
        target_kind="pose",
        control_part=arm_part,
        steps=len(action_np),
    )
    _sync_agent_state_from_public_action(env, robot_name, action_np)
    return action_np


def _public_pickup_action(
    env,
    robot_name,
    obj_name,
    target_obj_pose,
    pre_grasp_dis,
    **kwargs,
):
    if kwargs.get("use_public_grasp_action", False) is not True:
        raise RuntimeError("Public PickUpAction with grasp_pose_obj is disabled.")

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
        hand_close_qpos=_state_to_hand_qpos(env.close_state, len(eef_joints), device),
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
        raise RuntimeError("Public atomic action failed for grasp.")

    action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
    _log_public_atomic_backend(
        wrapper_name="grasp",
        action=action,
        cfg=cfg,
        target_kind="pose(grasp_pose_obj)",
        control_part=arm_part,
        steps=len(action_np),
    )
    _sync_agent_state_from_public_action(env, robot_name, action_np)
    return action_np


def _public_place_action(
    env,
    robot_name,
    target_pose,
    pre_place_dis,
    **kwargs,
):
    if kwargs.get("use_public_place_action", True) is not True:
        raise RuntimeError("Public PlaceAction is disabled.")

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
        hand_close_qpos=_state_to_hand_qpos(env.close_state, len(eef_joints), device),
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
        raise RuntimeError("Public atomic action failed for place on table.")

    action_np = _trajectory_to_agent_action(env, robot_name, trajectory, joint_ids)
    _log_public_atomic_backend(
        wrapper_name="place_on_table",
        action=action,
        cfg=cfg,
        target_kind="pose",
        control_part=arm_part,
        steps=len(action_np),
    )
    _sync_agent_state_from_public_action(env, robot_name, action_np)
    return action_np


def move_to_target_pose(
    robot_name: str,
    target_pose=None,
    sample_num: int = 20,
    env=None,
    **kwargs,
):
    actions = _public_move_action(
        env,
        robot_name,
        target_pose,
        sample_num,
        "move to target",
        **kwargs,
    )
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
    **kwargs,
):
    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name in obj_uids:
        target_obj = env.sim.get_rigid_object(obj_name)
    else:
        log_error(f"No matched object {obj_uids}.")
    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)

    if _semantic_public_grasp_enabled(kwargs):
        actions = _public_semantic_grasp_action(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            pre_grasp_dis=pre_grasp_dis,
            kwargs=kwargs,
        )
        log_info(
            "Total generated trajectory number for public semantic grasp: "
            f"{len(actions)}.",
            color="green",
        )
        return actions

    actions = _public_pickup_action(
        env,
        robot_name,
        obj_name,
        target_obj_pose,
        pre_grasp_dis,
        **kwargs,
    )

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

    actions = _public_place_action(
        env,
        robot_name,
        place_pose,
        pre_place_dis,
        **kwargs,
    )

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
    **kwargs,
):

    (
        _,
        _,
        _,
        select_arm_current_pose,
        _,
    ) = get_arm_states(env, robot_name)

    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name in obj_uids:
        target_obj = env.sim.get_rigid_object(obj_name)
    else:
        log_error("No matched object.")

    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    move_target_pose = deepcopy(select_arm_current_pose)
    move_target_pose[:3, 3] = target_obj_pose[:3, 3]
    move_target_pose[0, 3] += x_offset
    move_target_pose[1, 3] += y_offset
    move_target_pose[2, 3] += z_offset

    actions = _public_move_action(
        env,
        robot_name,
        move_target_pose,
        kwargs.get("sample_num", 30),
        "move relative to object",
        **kwargs,
    )

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
    **kwargs,
):

    (
        _,
        _,
        _,
        select_arm_current_pose,
        _,
    ) = get_arm_states(env, robot_name)

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

    actions = _public_move_action(
        env,
        robot_name,
        move_pose,
        kwargs.get("sample_num", 30),
        "move to absolute position",
        **kwargs,
    )

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
    **kwargs,
):

    (
        _,
        _,
        _,
        select_arm_current_pose,
        _,
    ) = get_arm_states(env, robot_name)

    move_pose = deepcopy(select_arm_current_pose)

    move_pose = get_offset_pose(move_pose, dx, "x", mode)
    move_pose = get_offset_pose(move_pose, dy, "y", mode)
    move_pose = get_offset_pose(move_pose, dz, "z", mode)

    actions = _public_move_action(
        env,
        robot_name,
        move_pose,
        kwargs.get("sample_num", 20),
        "move by relative offset",
        **kwargs,
    )

    log_info(
        f"Total generated trajectory number for move by relative offset: {len(actions)}.",
        color="green",
    )

    return actions


def back_to_initial_pose(robot_name: str, env=None, **kwargs):

    (
        is_left,
        select_arm,
        select_arm_current_qpos,
        _,
        _,
    ) = get_arm_states(env, robot_name)

    target_qpos = env.left_arm_init_qpos if is_left else env.right_arm_init_qpos
    target_qpos = torch.as_tensor(target_qpos, dtype=torch.float32)

    sample_num = kwargs.get("sample_num", 30)
    actions = _public_qpos_move_action(
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

    log_info(
        f"Total generated trajectory number for back to initial pose: {len(actions)}.",
        color="green",
    )

    return actions


def rotate_eef(robot_name: str, degree: float = 0, env=None, **kwargs):

    (
        _,
        select_arm,
        select_arm_current_qpos,
        _,
        _,
    ) = get_arm_states(env, robot_name)

    rotated_qpos = deepcopy(select_arm_current_qpos)
    rotated_qpos[5] += np.deg2rad(degree)
    sample_num = kwargs.get("sample_num", 20)
    actions = _public_qpos_move_action(
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

    log_info(
        f"Total generated trajectory number for rotate eef: {len(actions)}.",
        color="green",
    )

    return actions


def orient_eef(
    robot_name: str,
    direction: str = "front",  # 'front' or 'down'
    env=None,
    **kwargs,
):
    # Get arm state
    (
        _,
        _,
        _,
        select_arm_current_pose,
        _,
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
    actions = _public_move_action(
        env,
        robot_name,
        rotation_replaced_pose,
        sample_num,
        "orient eef",
        **kwargs,
    )

    log_info(
        f"Total generated trajectory number for orient eef: {len(actions)}.",
        color="green",
    )

    return actions


def close_gripper(robot_name: str, env=None, **kwargs):

    (
        _,
        _,
        _,
        _,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    sample_num = kwargs.get("sample_num", 15)
    _, _, hand_part, _, eef_joints = _select_arm_parts(env, robot_name)
    hand_dof = len(eef_joints)
    actions = _public_qpos_move_action(
        env=env,
        robot_name=robot_name,
        control_part=hand_part,
        target_qpos=_state_to_hand_qpos(env.close_state, hand_dof, env.robot.device),
        start_qpos=_state_to_hand_qpos(
            select_arm_current_gripper_state, hand_dof, env.robot.device
        ).reshape(1, hand_dof),
        sample_num=sample_num,
        kwargs=kwargs,
        log_name="close gripper",
    )

    log_info(
        f"Total generated trajectory number for close gripper: {len(actions)}.",
        color="green",
    )

    return actions


def open_gripper(robot_name: str, env=None, **kwargs):

    (
        _,
        _,
        _,
        _,
        select_arm_current_gripper_state,
    ) = get_arm_states(env, robot_name)

    if _is_current_gripper_open(
        env,
        select_arm_current_gripper_state,
        kwargs.get("open_threshold", 0.01),
    ):
        actions = _current_agent_action(env, robot_name)
        log_info(
            "Skip open gripper because current gripper state already satisfies the skip condition.",
            color="green",
        )
        return actions

    sample_num = kwargs.get("sample_num", 15)
    _, _, hand_part, _, eef_joints = _select_arm_parts(env, robot_name)
    hand_dof = len(eef_joints)
    actions = _public_qpos_move_action(
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
    actions = _append_hold_steps(
        actions,
        kwargs.get("settle_steps", kwargs.get("hold_steps", 0)),
        "open gripper",
    )

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
