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

from dataclasses import dataclass
import hashlib
import math
import os
from typing import Any

import numpy as np
import torch

from embodichain.lab.gym.utils.misc import get_rotation_replaced_pose
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    MoveActionCfg,
    ObjectSemantics,
    PickUpActionCfg,
    PlaceActionCfg,
)
from embodichain.lab.sim.atomic_actions.semantic_grasp import (
    format_semantic_candidate_message as _semantic_candidate_message,
    semantic_candidate_record_label as _semantic_candidate_record_label,
)
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalSamplerCfg,
    GraspGenerator,
    GraspGeneratorCfg,
    GripperCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GRASP_ANNOTATOR_CACHE_DIR,
)
from embodichain.utils.logger import log_info, log_warning


def _public_atomic_actions_enabled(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("use_public_atomic_actions", True))


def public_atomic_actions_enabled(kwargs: dict[str, Any]) -> bool:
    return _public_atomic_actions_enabled(kwargs)


def _public_grasp_strict(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("require_public_grasp_action", False))


def _public_grasp_requested(kwargs: dict[str, Any]) -> bool:
    return bool(
        kwargs.get("use_public_grasp_action", False) or _public_grasp_strict(kwargs)
    )


def _public_grasp_semantics_requested(kwargs: dict[str, Any]) -> bool:
    return bool(
        kwargs.get("use_public_grasp_semantics", False)
        or kwargs.get("public_grasp_strategy") is not None
    )


def _public_non_grasp_strict(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("require_public_non_grasp_actions", False))


def _handle_public_non_grasp_unavailable(
    message: str,
    *,
    strict: bool,
    fallback_name: str,
) -> None:
    if strict:
        raise RuntimeError(message)
    log_warning(f"{message} Falling back to legacy agent {fallback_name}.")


def _handle_public_grasp_unavailable(
    message: str,
    *,
    strict: bool,
) -> None:
    if strict:
        raise RuntimeError(message)
    log_warning(f"{message} Falling back to legacy agent grasp.")


def _select_arm(robot_name: str) -> tuple[bool, str, str]:
    is_left = "right" not in robot_name
    arm_part = "left_arm" if is_left else "right_arm"
    hand_part = "left_eef" if is_left else "right_eef"
    return is_left, arm_part, hand_part


def _as_pose_tensor(
    pose,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(pose, torch.Tensor):
        pose_tensor = pose.detach().clone()
    else:
        pose_tensor = torch.as_tensor(pose)
    return pose_tensor.to(device=device or pose_tensor.device, dtype=torch.float32)


def _as_qpos_tensor(
    qpos,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    qpos_tensor = torch.as_tensor(qpos, dtype=torch.float32, device=device).flatten()
    return qpos_tensor.unsqueeze(0)


def _state_to_hand_qpos(
    state,
    hand_dof: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).flatten()
    if state_tensor.numel() == hand_dof:
        return state_tensor
    if state_tensor.numel() == 1:
        return state_tensor.repeat(hand_dof)
    raise ValueError(
        f"Cannot convert gripper state with shape {tuple(state_tensor.shape)} "
        f"to hand_dof={hand_dof}."
    )


def _as_vector_tensor(
    values,
    length: int,
    *,
    name: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    vector = torch.as_tensor(values, dtype=torch.float32, device=device).flatten()
    if vector.numel() != length:
        raise ValueError(
            f"{name} must contain {length} values, got shape {tuple(vector.shape)}."
        )
    return vector


def _normalized_vector_tensor(
    values,
    length: int,
    *,
    name: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    vector = _as_vector_tensor(values, length, name=name, device=device)
    norm = torch.linalg.norm(vector)
    if norm <= 1e-6:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _format_vector(vector: torch.Tensor) -> str:
    values = vector.detach().cpu().tolist()
    return ",".join(f"{value:.4f}" for value in values)


def _create_engine(env, cfg) -> AtomicActionEngine:
    motion_generator = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=env.robot.uid))
    )
    return AtomicActionEngine(
        motion_generator=motion_generator,
        actions_cfg_list=[cfg],
    )


def _current_arm_qpos(env, is_left: bool):
    return env.left_arm_current_qpos if is_left else env.right_arm_current_qpos


def _current_gripper_state(env, is_left: bool):
    return (
        env.left_arm_current_gripper_state
        if is_left
        else env.right_arm_current_gripper_state
    )


def _extract_legacy_action(
    env,
    robot_name: str,
    trajectory: torch.Tensor,
    joint_ids: list[int],
) -> np.ndarray:
    is_left, _, _ = _select_arm(robot_name)
    arm_joints = env.left_arm_joints if is_left else env.right_arm_joints
    hand_joints = env.left_eef_joints if is_left else env.right_eef_joints

    if trajectory.ndim != 3 or trajectory.shape[0] != 1:
        raise ValueError(
            "Agent atomic action adapter currently supports trajectory shape "
            f"(1, T, dof), got {tuple(trajectory.shape)}."
        )

    trajectory_np = trajectory.detach().cpu().numpy()
    joint_id_to_col = {joint_id: idx for idx, joint_id in enumerate(joint_ids)}
    current_hand_qpos = (
        _state_to_hand_qpos(
            _current_gripper_state(env, is_left),
            len(hand_joints),
        )
        .detach()
        .cpu()
        .numpy()
    )

    columns = []
    for joint_id in arm_joints:
        if joint_id not in joint_id_to_col:
            raise ValueError(
                f"Public action trajectory does not include arm joint {joint_id}."
            )
        columns.append(trajectory_np[0, :, joint_id_to_col[joint_id]])

    for hand_idx, joint_id in enumerate(hand_joints):
        if joint_id in joint_id_to_col:
            columns.append(trajectory_np[0, :, joint_id_to_col[joint_id]])
        else:
            columns.append(
                np.full(
                    trajectory_np.shape[1],
                    current_hand_qpos[hand_idx],
                    dtype=np.float32,
                )
            )

    return np.stack(columns, axis=-1).astype(np.float32, copy=False)


def _sync_agent_arm_state(env, robot_name: str, legacy_action: np.ndarray) -> None:
    is_left, _, _ = _select_arm(robot_name)
    arm_joints = env.left_arm_joints if is_left else env.right_arm_joints
    arm_dof = len(arm_joints)

    arm_device = getattr(env.robot, "device", None)
    gripper_device = getattr(getattr(env, "open_state", None), "device", None)
    final_arm_qpos = torch.as_tensor(
        legacy_action[-1, :arm_dof],
        dtype=torch.float32,
        device=arm_device,
    )
    final_gripper = torch.as_tensor(
        [legacy_action[-1, arm_dof]],
        dtype=torch.float32,
        device=gripper_device,
    )
    final_pose = env.get_arm_fk(qpos=final_arm_qpos, is_left=is_left)

    env.set_current_qpos_agent(final_arm_qpos, is_left=is_left)
    env.set_current_xpos_agent(final_pose, is_left=is_left)
    env.set_current_gripper_state_agent(final_gripper, is_left=is_left)


def _extract_legacy_gripper_action(
    env,
    robot_name: str,
    trajectory: torch.Tensor,
    joint_ids: list[int],
) -> np.ndarray:
    is_left, _, _ = _select_arm(robot_name)
    arm_joints = env.left_arm_joints if is_left else env.right_arm_joints
    hand_joints = env.left_eef_joints if is_left else env.right_eef_joints

    if trajectory.ndim != 3 or trajectory.shape[0] != 1:
        raise ValueError(
            "Agent gripper action adapter currently supports trajectory shape "
            f"(1, T, dof), got {tuple(trajectory.shape)}."
        )

    trajectory_np = trajectory.detach().cpu().numpy()
    joint_id_to_col = {joint_id: idx for idx, joint_id in enumerate(joint_ids)}
    n_waypoints = trajectory_np.shape[1]
    current_arm_qpos = (
        torch.as_tensor(_current_arm_qpos(env, is_left))
        .detach()
        .cpu()
        .numpy()
        .reshape(1, -1)
    )
    current_hand_qpos = (
        _state_to_hand_qpos(
            _current_gripper_state(env, is_left),
            len(hand_joints),
        )
        .detach()
        .cpu()
        .numpy()
    )

    arm_action = np.repeat(current_arm_qpos, n_waypoints, axis=0)
    hand_columns = []
    for hand_idx, joint_id in enumerate(hand_joints):
        if joint_id in joint_id_to_col:
            hand_columns.append(trajectory_np[0, :, joint_id_to_col[joint_id]])
        else:
            hand_columns.append(
                np.full(n_waypoints, current_hand_qpos[hand_idx], dtype=np.float32)
            )
    hand_action = np.stack(hand_columns, axis=-1)
    return np.concatenate([arm_action, hand_action], axis=-1).astype(
        np.float32, copy=False
    )


def _sync_agent_gripper_state(env, robot_name: str, legacy_action: np.ndarray) -> None:
    is_left, _, _ = _select_arm(robot_name)
    arm_dof = len(env.left_arm_joints if is_left else env.right_arm_joints)
    gripper_device = getattr(getattr(env, "open_state", None), "device", None)
    final_gripper = torch.as_tensor(
        [legacy_action[-1, arm_dof]],
        dtype=torch.float32,
        device=gripper_device,
    )
    env.set_current_gripper_state_agent(final_gripper, is_left=is_left)


def _object_pose(env, obj_name: str) -> torch.Tensor | None:
    obj_info = getattr(env, "obj_info", None)
    if isinstance(obj_info, dict):
        cached = obj_info.get(obj_name, {}).get("pose")
        if cached is not None:
            return _as_pose_tensor(cached).detach().clone()

    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name not in obj_uids:
        return None
    target_obj = env.sim.get_rigid_object(obj_name)
    return target_obj.get_local_pose(to_matrix=True).squeeze(0).detach().clone()


def _object_geometry_bounds(
    env,
    obj_name: str,
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor] | None:
    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name not in obj_uids:
        return None
    target_obj = env.sim.get_rigid_object(obj_name)
    try:
        vertices = target_obj.get_vertices(env_ids=[0], scale=True)[0]
    except Exception:
        return None
    pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    pose = _as_pose_tensor(pose, device=device)
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=pose.device)
    if vertices.numel() == 0:
        return None

    world_vertices = vertices @ pose[:3, :3].transpose(0, 1)
    world_vertices = world_vertices + pose[:3, 3]
    mins = world_vertices.min(dim=0).values
    maxs = world_vertices.max(dim=0).values
    extents = maxs - mins
    return {
        "center": ((mins + maxs) * 0.5).detach(),
        "mins": mins.detach(),
        "maxs": maxs.detach(),
        "extents": extents.detach(),
        "xy_radius": torch.clamp(extents[:2].max() * 0.5, min=1e-6).detach(),
        "height": torch.clamp(extents[2], min=1e-6).detach(),
    }


@dataclass
class PublicSemanticGraspPlan:
    """Structured public semantic grasp plan selected from candidate ranking."""

    label: str
    direction: torch.Tensor
    grasp_pose: torch.Tensor
    trajectory: torch.Tensor
    joint_ids: list[int]
    lift_height: float


def _rotation_error_rad(
    candidate_pose: torch.Tensor, reference_pose: torch.Tensor
) -> float:
    candidate_rot = candidate_pose[:3, :3]
    reference_rot = reference_pose[:3, :3].to(
        dtype=candidate_rot.dtype,
        device=candidate_rot.device,
    )
    delta_rot = reference_rot.transpose(0, 1) @ candidate_rot
    cos_angle = ((torch.trace(delta_rot) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.acos(cos_angle).item())


def _xy_distance(pose_a: torch.Tensor, pose_b: torch.Tensor) -> float:
    return float(torch.linalg.norm(pose_a[:2, 3] - pose_b[:2, 3]).item())


def _relative_pose(base_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(base_pose) @ target_pose


def _public_grasp_relation_key(robot_name: str, obj_name: str) -> str:
    return f"{robot_name}:{obj_name}"


def _store_public_grasp_relation(
    *,
    env,
    robot_name: str,
    obj_name: str,
    grasp_pose: torch.Tensor,
    source: str,
) -> None:
    obj_pose = _object_pose(env, obj_name)
    if obj_pose is None:
        log_warning(
            f"Cannot store public grasp relation for '{obj_name}': object pose is unavailable."
        )
        return

    grasp_pose = _as_pose_tensor(grasp_pose, device=obj_pose.device)
    relation = torch.linalg.inv(obj_pose) @ grasp_pose
    relations = getattr(env, "_public_grasp_object_eef_relations", None)
    if not isinstance(relations, dict):
        relations = {}
    relations[_public_grasp_relation_key(robot_name, obj_name)] = {
        "relation": relation.detach().clone(),
        "source": source,
    }
    setattr(env, "_public_grasp_object_eef_relations", relations)


def _get_public_grasp_relation(
    *,
    env,
    robot_name: str,
    obj_name: str,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor | None, str | None]:
    relations = getattr(env, "_public_grasp_object_eef_relations", None)
    key = _public_grasp_relation_key(robot_name, obj_name)
    if isinstance(relations, dict) and key in relations:
        entry = relations[key]
        relation = entry.get("relation") if isinstance(entry, dict) else entry
        if relation is not None:
            return _as_pose_tensor(relation), (
                entry.get("source") if isinstance(entry, dict) else "stored"
            )

    if not bool(kwargs.get("public_place_upright_allow_legacy_grasp_pose", True)):
        return None, None

    grasp_pose_obj = (
        getattr(env, "obj_info", {}).get(obj_name, {}).get("grasp_pose_obj")
    )
    if grasp_pose_obj is None:
        return None, None
    return _as_pose_tensor(grasp_pose_obj), "legacy_grasp_pose_obj"


def _relation_error(
    actual_relation: torch.Tensor,
    expected_relation: torch.Tensor,
) -> tuple[float, float]:
    expected_relation = expected_relation.to(
        dtype=actual_relation.dtype,
        device=actual_relation.device,
    )
    pos_error = float(
        torch.linalg.norm(actual_relation[:3, 3] - expected_relation[:3, 3]).item()
    )
    rot_error = _rotation_error_rad(actual_relation, expected_relation)
    return pos_error, rot_error


def validate_public_place_preconditions(
    *,
    env,
    robot_name: str,
    obj_name: str,
    target_x: float | None,
    target_y: float | None,
    target_height,
    kwargs: dict[str, Any],
) -> None:
    if not bool(kwargs.get("validate_place_preconditions", False)):
        return
    if env is None:
        raise RuntimeError("place_on_table precondition failed: env is None.")
    if target_height is None:
        raise RuntimeError(
            f"place_on_table precondition failed for {robot_name}/{obj_name}: "
            "target height is unavailable."
        )
    if not np.isfinite(float(target_height)):
        raise RuntimeError(
            f"place_on_table precondition failed for {robot_name}/{obj_name}: "
            f"target height is not finite ({target_height})."
        )

    obj_pose = _object_pose(env, obj_name)
    if obj_pose is None:
        raise RuntimeError(
            f"place_on_table precondition failed: object '{obj_name}' is unavailable."
        )

    relation, relation_source = _get_public_grasp_relation(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        kwargs=kwargs,
    )
    if relation is None:
        if bool(kwargs.get("public_place_require_grasp_relation", True)):
            raise RuntimeError(
                f"place_on_table precondition failed for {robot_name}/{obj_name}: "
                "no object-to-EEF grasp relation is available."
            )
        return

    eef_pose = _current_eef_pose(env, robot_name)
    obj_pose = obj_pose.to(dtype=eef_pose.dtype, device=eef_pose.device)
    actual_relation = torch.linalg.inv(obj_pose) @ eef_pose
    pos_error, rot_error = _relation_error(actual_relation, relation)
    max_pos_error = float(kwargs.get("public_place_max_grasp_relation_pos_error", 0.16))
    max_rot_error = float(kwargs.get("public_place_max_grasp_relation_rot_error", 1.4))
    refresh_relation = bool(kwargs.get("public_place_refresh_grasp_relation", True))
    if pos_error > max_pos_error or (
        rot_error > max_rot_error and not refresh_relation
    ):
        raise RuntimeError(
            f"place_on_table precondition failed for {robot_name}/{obj_name}: "
            f"object does not appear to be held by the gripper "
            f"(relation_source={relation_source}, pos_error={pos_error:.4f}m/"
            f"{max_pos_error:.4f}m, rot_error={rot_error:.4f}rad/"
            f"{max_rot_error:.4f}rad)."
        )
    if refresh_relation:
        _store_public_grasp_relation(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            grasp_pose=eef_pose,
            source=f"refreshed_before_place_from_{relation_source}",
        )
        if rot_error > max_rot_error:
            log_warning(
                f"Refreshing public place relation for {robot_name}/{obj_name}: "
                f"object is still near the gripper but rotated "
                f"(relation_source={relation_source}, pos_error={pos_error:.4f}m, "
                f"rot_error={rot_error:.4f}rad>{max_rot_error:.4f}rad)."
            )

    min_object_lift = kwargs.get("public_place_precondition_min_object_lift")
    if min_object_lift is not None:
        initial_height = getattr(env, "obj_info", {}).get(obj_name, {}).get("height")
        if initial_height is not None:
            object_lift = float(obj_pose[2, 3].item() - float(initial_height))
            if object_lift < float(min_object_lift):
                raise RuntimeError(
                    f"place_on_table precondition failed for {robot_name}/{obj_name}: "
                    f"object_lift={object_lift:.4f}m < "
                    f"{float(min_object_lift):.4f}m."
                )


def _upright_object_rotation(object_pose: torch.Tensor) -> torch.Tensor:
    device = object_pose.device
    dtype = object_pose.dtype
    world_z = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

    x_axis = object_pose[:3, 0].detach().clone()
    x_axis[2] = 0.0
    if torch.linalg.norm(x_axis) > 1e-6:
        x_axis = x_axis / torch.linalg.norm(x_axis)
        y_axis = torch.cross(world_z, x_axis, dim=0)
        y_axis = y_axis / torch.linalg.norm(y_axis).clamp_min(1e-6)
        return torch.stack([x_axis, y_axis, world_z], dim=1)

    y_axis = object_pose[:3, 1].detach().clone()
    y_axis[2] = 0.0
    if torch.linalg.norm(y_axis) > 1e-6:
        y_axis = y_axis / torch.linalg.norm(y_axis)
        x_axis = torch.cross(y_axis, world_z, dim=0)
        x_axis = x_axis / torch.linalg.norm(x_axis).clamp_min(1e-6)
        return torch.stack([x_axis, y_axis, world_z], dim=1)

    return torch.eye(3, dtype=dtype, device=device)


def _z_rotation(
    angle_rad: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    cos_value = math.cos(angle_rad)
    sin_value = math.sin(angle_rad)
    return torch.tensor(
        [
            [cos_value, -sin_value, 0.0],
            [sin_value, cos_value, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def _public_place_upright_yaw_offsets(kwargs: dict[str, Any]) -> list[float]:
    value = kwargs.get("public_place_upright_yaw_offsets_deg")
    if value is None:
        return [0.0, 90.0, -90.0, 180.0]
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = value
    return [float(item) for item in values]


def build_public_upright_place_pose_candidates(
    *,
    env,
    robot_name: str,
    obj_name: str,
    target_pose,
    x: float | None,
    y: float | None,
    object_height,
    kwargs: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    if not bool(kwargs.get("public_place_upright", True)):
        return []
    if env is None or target_pose is None:
        return []

    obj_pose = _object_pose(env, obj_name)
    if obj_pose is None:
        log_warning(
            f"Public upright place cannot build object target pose: "
            f"object '{obj_name}' is unavailable."
        )
        return []

    relation, relation_source = _get_public_grasp_relation(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        kwargs=kwargs,
    )
    if relation is None:
        log_warning(
            f"Public upright place has no object-to-EEF grasp relation for "
            f"{robot_name}/{obj_name}; using current EEF orientation."
        )
        return []

    target_pose = _as_pose_tensor(target_pose)
    obj_pose = obj_pose.to(dtype=target_pose.dtype, device=target_pose.device)
    relation = relation.to(dtype=target_pose.dtype, device=target_pose.device)

    initial_pose = getattr(env, "obj_info", {}).get(obj_name, {}).get("initial_pose")
    if (
        bool(kwargs.get("public_place_upright_use_initial_pose", True))
        and initial_pose is not None
    ):
        initial_pose = _as_pose_tensor(initial_pose, device=target_pose.device).to(
            dtype=target_pose.dtype
        )
        base_object_rotation = initial_pose[:3, :3]
        rotation_source = "initial_pose"
    else:
        base_object_rotation = _upright_object_rotation(obj_pose)
        rotation_source = "current_pose_projection"
    candidates: list[tuple[str, torch.Tensor]] = []
    for yaw_deg in _public_place_upright_yaw_offsets(kwargs):
        object_target_pose = obj_pose.detach().clone()
        yaw_rot = _z_rotation(
            math.radians(yaw_deg),
            dtype=target_pose.dtype,
            device=target_pose.device,
        )
        object_target_pose[:3, :3] = yaw_rot @ base_object_rotation
        if x is not None:
            object_target_pose[0, 3] = float(x)
        if y is not None:
            object_target_pose[1, 3] = float(y)
        object_target_pose[2, 3] = torch.as_tensor(
            object_height,
            dtype=target_pose.dtype,
            device=target_pose.device,
        )

        place_pose = object_target_pose @ relation
        label = (
            f"upright_yaw{yaw_deg:+.0f}_rot_{rotation_source}_"
            f"from_{relation_source}"
        )
        candidates.append((label, place_pose))
    return candidates


def try_build_public_upright_place_pose(
    **kwargs,
) -> torch.Tensor | None:
    candidates = build_public_upright_place_pose_candidates(**kwargs)
    if not candidates:
        return None
    return candidates[0][1]


def register_pending_public_place_validation(
    *,
    env,
    kwargs: dict[str, Any],
    robot_name: str,
    obj_name: str,
    target_x: float | None,
    target_y: float | None,
    target_height,
    label: str,
) -> None:
    if not bool(kwargs.get("validate_public_place_after_action", False)):
        return
    pending = {
        "robot_name": robot_name,
        "obj_name": obj_name,
        "target_x": None if target_x is None else float(target_x),
        "target_y": None if target_y is None else float(target_y),
        "target_height": None if target_height is None else float(target_height),
        "label": label,
        "max_xy_error": float(kwargs.get("public_place_validation_max_xy_error", 0.16)),
        "min_upright_dot": float(
            kwargs.get("public_place_validation_min_upright_dot", 0.65)
        ),
        "max_height_error": _optional_float(
            kwargs,
            "public_place_validation_max_height_error",
        ),
    }
    pending_list = getattr(env, "_pending_public_place_validations", None)
    if not isinstance(pending_list, list):
        pending_list = []
    pending_list.append(pending)
    setattr(env, "_pending_public_place_validations", pending_list)


def validate_pending_public_place_after_action(env, kwargs: dict[str, Any]) -> None:
    pending_items = getattr(env, "_pending_public_place_validations", None)
    if not isinstance(pending_items, list) or not pending_items:
        return
    setattr(env, "_pending_public_place_validations", [])

    failures = []
    for pending in pending_items:
        obj_name = pending["obj_name"]
        pose = _object_pose(env, obj_name)
        if pose is None:
            failures.append(f"object '{obj_name}' is unavailable")
            continue

        message_parts = [f"label={pending['label']}"]
        target_x = pending.get("target_x")
        target_y = pending.get("target_y")
        if target_x is not None and target_y is not None:
            xy_error = float(
                torch.linalg.norm(
                    pose[:2, 3]
                    - torch.tensor(
                        [target_x, target_y],
                        dtype=pose.dtype,
                        device=pose.device,
                    )
                ).item()
            )
            max_xy_error = float(pending["max_xy_error"])
            message_parts.append(f"xy_error={xy_error:.4f}m")
            if xy_error > max_xy_error:
                failures.append(
                    f"{obj_name} xy_error={xy_error:.4f}m>{max_xy_error:.4f}m"
                )

        upright_dot = float(
            torch.dot(
                pose[:3, 2],
                torch.tensor([0.0, 0.0, 1.0], dtype=pose.dtype, device=pose.device),
            ).item()
        )
        min_upright_dot = float(pending["min_upright_dot"])
        message_parts.append(f"upright_dot={upright_dot:.4f}")
        if upright_dot < min_upright_dot:
            failures.append(
                f"{obj_name} upright_dot={upright_dot:.4f}<{min_upright_dot:.4f}"
            )

        max_height_error = pending.get("max_height_error")
        target_height = pending.get("target_height")
        if max_height_error is not None and target_height is not None:
            height_error = abs(float(pose[2, 3].item()) - float(target_height))
            message_parts.append(f"height_error={height_error:.4f}m")
            if height_error > float(max_height_error):
                failures.append(
                    f"{obj_name} height_error={height_error:.4f}m>"
                    f"{float(max_height_error):.4f}m"
                )

        log_info(
            f"Public place validation for '{obj_name}': "
            + ", ".join(message_parts)
            + "."
        )

    if failures:
        raise RuntimeError(
            "Public place validation failed: " + "; ".join(failures) + "."
        )


def _current_eef_pose(env, robot_name: str) -> torch.Tensor:
    is_left, _, _ = _select_arm(robot_name)
    arm_joints = env.left_arm_joints if is_left else env.right_arm_joints
    qpos = env.robot.get_qpos().squeeze(0)
    arm_qpos = qpos[arm_joints]
    return env.get_arm_fk(qpos=arm_qpos, is_left=is_left).detach().clone()


def _optional_float(kwargs: dict[str, Any], name: str) -> float | None:
    value = kwargs.get(name)
    if value is None:
        return None
    return float(value)


def _record_public_grasp_attempt(
    *,
    env,
    kwargs: dict[str, Any],
    obj_name: str,
    label: str,
    direction: torch.Tensor,
    status: str,
    message: str,
) -> None:
    record = {
        "obj_name": obj_name,
        "label": label,
        "direction": _format_vector(direction),
        "status": status,
        "message": message.replace("\n", " "),
    }

    records = getattr(env, "public_grasp_attempt_log", None)
    if records is None:
        records = []
        setattr(env, "public_grasp_attempt_log", records)
    records.append(record)

    log_dir = kwargs.get("log_dir")
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "public_grasp_attempts.tsv")
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("obj_name\tlabel\tdirection\tstatus\tmessage\n")
        f.write(
            "\t".join(
                [
                    record["obj_name"],
                    record["label"],
                    record["direction"],
                    record["status"],
                    record["message"],
                ]
            )
            + "\n"
        )


def _register_public_grasp_physical_validation(
    *,
    env,
    kwargs: dict[str, Any],
    robot_name: str,
    obj_name: str,
    label: str,
    direction: torch.Tensor,
    lift_height: float,
    legacy_reference_pose: torch.Tensor | None = None,
) -> None:
    if not bool(kwargs.get("validate_public_grasp_after_action", False)):
        return

    pose = _object_pose(env, obj_name)
    if pose is None:
        log_warning(
            f"Cannot register public grasp physical validation: "
            f"object '{obj_name}' is unavailable."
        )
        return

    validate_relative = bool(
        kwargs.get("public_grasp_validate_relative_to_legacy_pose", False)
    )
    reference_relative_pose = None
    if validate_relative:
        if legacy_reference_pose is None:
            log_warning(
                f"Cannot register public grasp relative-pose validation: "
                f"legacy grasp reference is unavailable for '{obj_name}'."
            )
        else:
            reference_relative_pose = _relative_pose(
                legacy_reference_pose.to(dtype=pose.dtype, device=pose.device),
                pose,
            )

    pending = {
        "robot_name": robot_name,
        "obj_name": obj_name,
        "pose_before": pose,
        "label": label,
        "direction": direction.detach().clone(),
        "min_lift": float(kwargs.get("public_grasp_validation_min_object_lift", 0.05)),
        "max_lift": _optional_float(
            kwargs,
            "public_grasp_validation_max_object_lift",
        ),
        "max_xy_displacement": _optional_float(
            kwargs,
            "public_grasp_validation_max_object_xy_displacement",
        ),
        "reference_relative_pose": reference_relative_pose,
        "max_legacy_relative_pos_error": _optional_float(
            kwargs,
            "public_grasp_max_legacy_relative_pos_error",
        ),
        "max_legacy_relative_rot_error": _optional_float(
            kwargs,
            "public_grasp_max_legacy_relative_rot_error",
        ),
        "planned_lift": float(lift_height),
    }
    pending_list = getattr(env, "_pending_public_grasp_physical_validations", None)
    if not isinstance(pending_list, list):
        pending_list = []
    pending_list.append(pending)
    setattr(env, "_pending_public_grasp_physical_validations", pending_list)
    setattr(env, "_pending_public_grasp_physical_validation", None)


def validate_pending_public_grasp_after_action(env, kwargs: dict[str, Any]) -> None:
    pending_items = getattr(env, "_pending_public_grasp_physical_validations", None)
    if not isinstance(pending_items, list):
        pending_items = []
    legacy_pending = getattr(env, "_pending_public_grasp_physical_validation", None)
    if legacy_pending:
        pending_items.append(legacy_pending)
    if not pending_items:
        return
    setattr(env, "_pending_public_grasp_physical_validations", [])
    setattr(env, "_pending_public_grasp_physical_validation", None)

    failures = []
    for pending in pending_items:
        try:
            _validate_one_pending_public_grasp_after_action(env, kwargs, pending)
        except RuntimeError as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError("; ".join(failures))


def _validate_one_pending_public_grasp_after_action(
    env,
    kwargs: dict[str, Any],
    pending: dict[str, Any],
) -> None:
    obj_name = pending["obj_name"]
    pose_after = _object_pose(env, obj_name)
    if pose_after is None:
        raise RuntimeError(
            f"Public grasp physical validation failed: object '{obj_name}' is unavailable."
        )

    pose_before = pending["pose_before"]
    object_lift = float((pose_after[2, 3] - pose_before[2, 3]).item())
    min_lift = float(pending["min_lift"])
    label = pending["label"]
    direction = pending["direction"]
    failures = []
    message_parts = [
        f"object_lift={object_lift:.4f}m",
        f"required>={min_lift:.4f}m",
        f"planned_lift={pending['planned_lift']:.4f}m",
    ]
    if object_lift < min_lift:
        failures.append(f"object_lift<{min_lift:.4f}m")

    max_lift = pending.get("max_lift")
    if max_lift is not None:
        message_parts.append(f"max_lift={float(max_lift):.4f}m")
        if object_lift > float(max_lift):
            failures.append(f"object_lift>{float(max_lift):.4f}m")

    max_xy_displacement = pending.get("max_xy_displacement")
    if max_xy_displacement is not None:
        object_xy_displacement = _xy_distance(pose_after, pose_before)
        message_parts.append(f"object_xy_displacement={object_xy_displacement:.4f}m")
        message_parts.append(f"max_xy={float(max_xy_displacement):.4f}m")
        if object_xy_displacement > float(max_xy_displacement):
            failures.append(f"object_xy_displacement>{float(max_xy_displacement):.4f}m")

    reference_relative_pose = pending.get("reference_relative_pose")
    if reference_relative_pose is not None:
        eef_pose_after = _current_eef_pose(env, pending["robot_name"])
        actual_relative_pose = _relative_pose(eef_pose_after, pose_after)
        relative_pos_error = float(
            torch.linalg.norm(
                actual_relative_pose[:3, 3]
                - reference_relative_pose[:3, 3].to(
                    dtype=actual_relative_pose.dtype,
                    device=actual_relative_pose.device,
                )
            ).item()
        )
        relative_rot_error = _rotation_error_rad(
            actual_relative_pose,
            reference_relative_pose,
        )
        message_parts.append(f"legacy_relative_pos_error={relative_pos_error:.4f}m")
        message_parts.append(f"legacy_relative_rot_error={relative_rot_error:.4f}rad")

        max_relative_pos = pending.get("max_legacy_relative_pos_error")
        if max_relative_pos is not None:
            message_parts.append(f"max_relative_pos={float(max_relative_pos):.4f}m")
            if relative_pos_error > float(max_relative_pos):
                failures.append(
                    f"legacy_relative_pos_error>{float(max_relative_pos):.4f}m"
                )
        max_relative_rot = pending.get("max_legacy_relative_rot_error")
        if max_relative_rot is not None:
            message_parts.append(f"max_relative_rot={float(max_relative_rot):.4f}rad")
            if relative_rot_error > float(max_relative_rot):
                failures.append(
                    f"legacy_relative_rot_error>{float(max_relative_rot):.4f}rad"
                )

    message = ", ".join(message_parts)
    if failures:
        message = f"{message}; failures={';'.join(failures)}"
        _record_public_grasp_attempt(
            env=env,
            kwargs=kwargs,
            obj_name=obj_name,
            label=label,
            direction=direction,
            status="physical_failed",
            message=message,
        )
        raise RuntimeError(f"Public grasp physical validation failed: {message}.")

    _record_public_grasp_attempt(
        env=env,
        kwargs=kwargs,
        obj_name=obj_name,
        label=label,
        direction=direction,
        status="physical_ok",
        message=message,
    )
    log_info(f"Public grasp physical validation passed for '{obj_name}': {message}.")


def _antipodal_cache_path(vertices: torch.Tensor, triangles: torch.Tensor) -> str:
    vert_bytes = vertices.to("cpu").numpy().tobytes()
    face_bytes = triangles.to("cpu").numpy().tobytes()
    md5_hash = hashlib.md5(vert_bytes + face_bytes).hexdigest()
    return os.path.join(GRASP_ANNOTATOR_CACHE_DIR, f"antipodal_cache_{md5_hash}.npy")


def _build_public_grasp_semantics(
    env,
    obj_name: str,
    kwargs: dict[str, Any],
    *,
    strict: bool,
) -> ObjectSemantics | None:
    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name not in obj_uids:
        _handle_public_grasp_unavailable(
            f"Public semantic grasp cannot find rigid object '{obj_name}'.",
            strict=strict,
        )
        return None

    target_obj = env.sim.get_rigid_object(obj_name)
    try:
        vertices = target_obj.get_vertices(env_ids=[0], scale=True)[0]
        triangles = target_obj.get_triangles(env_ids=[0])[0]
    except Exception as exc:
        _handle_public_grasp_unavailable(
            f"Public semantic grasp cannot read mesh for '{obj_name}': {exc}.",
            strict=strict,
        )
        return None

    allow_annotation = bool(kwargs.get("allow_public_grasp_annotation", False))
    force_reannotate = bool(kwargs.get("force_public_grasp_reannotate", False))
    generate_candidates = bool(kwargs.get("generate_public_grasp_candidates", False))
    cache_path = _antipodal_cache_path(vertices, triangles)
    if force_reannotate and not allow_annotation:
        _handle_public_grasp_unavailable(
            "Public semantic grasp force_reannotate=True requires "
            "allow_public_grasp_annotation=True.",
            strict=strict,
        )
        return None
    if (
        not allow_annotation
        and not generate_candidates
        and not os.path.exists(cache_path)
    ):
        _handle_public_grasp_unavailable(
            f"Public grasp annotation cache is missing for '{obj_name}' at "
            f"{cache_path}; non-interactive mode will not open viser.",
            strict=strict,
        )
        return None

    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    geometry_stats = _object_geometry_stats(
        vertices,
        target_obj_pose,
        device=getattr(env.robot, "device", None),
    )
    max_open_length = float(kwargs.get("grasp_max_open_length", 0.088))
    min_open_length = _public_grasp_effective_antipodal_min_open_length(
        geometry_stats,
        kwargs,
    )
    open_check_margin = _public_grasp_effective_open_check_margin(
        geometry_stats,
        kwargs,
    )
    filter_ground_collision = _public_grasp_effective_filter_ground_collision(
        geometry_stats,
        kwargs,
    )
    generator_cfg = GraspGeneratorCfg(
        viser_port=int(kwargs.get("public_grasp_viser_port", 11801)),
        use_largest_connected_component=bool(
            kwargs.get("public_grasp_largest_component", False)
        ),
        max_deviation_angle=float(kwargs.get("grasp_max_deviation_angle", np.pi / 6)),
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=int(kwargs.get("grasp_antipodal_n_sample", 20000)),
            max_angle=float(kwargs.get("grasp_antipodal_max_angle", np.pi / 12)),
            max_length=max_open_length,
            min_length=min_open_length,
        ),
        is_filter_ground_collision=filter_ground_collision,
    )
    gripper_collision_cfg = GripperCollisionCfg(
        max_open_length=max_open_length,
        finger_length=float(kwargs.get("grasp_finger_length", 0.078)),
        y_thickness=float(kwargs.get("grasp_y_thickness", 0.03)),
        x_thickness=float(kwargs.get("grasp_x_thickness", 0.01)),
        root_z_width=float(kwargs.get("grasp_root_z_width", 0.08)),
        point_sample_dense=float(kwargs.get("grasp_point_sample_dense", 0.012)),
        max_decomposition_hulls=int(kwargs.get("grasp_max_decomposition_hulls", 16)),
        open_check_margin=open_check_margin,
        query_batch_size=int(kwargs.get("grasp_collision_query_batch_size", 512)),
    )
    affordance = AntipodalAffordance(
        object_label=obj_name,
        force_reannotate=force_reannotate,
        is_draw_grasp_xpos=bool(kwargs.get("draw_public_grasp_xpos", False)),
        custom_config={
            "generator_cfg": generator_cfg,
            "gripper_collision_cfg": gripper_collision_cfg,
        },
    )
    if generate_candidates:
        generator = GraspGenerator(
            vertices=vertices,
            triangles=triangles,
            cfg=generator_cfg,
            gripper_collision_cfg=gripper_collision_cfg,
        )
        generator._hit_point_pairs = generator._antipodal_sampler.sample(
            vertices,
            triangles,
        )
        affordance.generator = generator
    return ObjectSemantics(
        label=obj_name,
        geometry={
            "mesh_vertices": vertices,
            "mesh_triangles": triangles,
            "geometry_stats": geometry_stats,
        },
        affordance=affordance,
        entity=target_obj,
    )


def _run_public_action(
    env,
    robot_name: str,
    cfg,
    target,
    *,
    log_name: str,
    strict: bool = False,
):
    try:
        engine = _create_engine(env, cfg)
        if not isinstance(target, ObjectSemantics):
            target = _as_pose_tensor(target, device=engine.device)
        is_left, _, _ = _select_arm(robot_name)
        start_qpos = _as_qpos_tensor(
            _current_arm_qpos(env, is_left), device=engine.device
        )

        atom_action = engine._actions[cfg.name]
        is_success, trajectory, joint_ids = atom_action.execute(
            target=target,
            start_qpos=start_qpos,
        )
        if not is_success or trajectory.numel() == 0:
            if strict:
                raise RuntimeError(
                    f"Public atomic action adapter failed for {log_name}."
                )
            log_warning(
                f"Public atomic action adapter failed for {log_name}; "
                "falling back to legacy agent atomic action."
            )
            return None

        legacy_action = _extract_legacy_action(env, robot_name, trajectory, joint_ids)
        _sync_agent_arm_state(env, robot_name, legacy_action)
        return legacy_action
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Public atomic action adapter raised {type(exc).__name__} for "
                f"{log_name}: {exc}."
            ) from exc
        log_warning(
            f"Public atomic action adapter raised {type(exc).__name__} for "
            f"{log_name}: {exc}. Falling back to legacy agent atomic action."
        )
        return None


def try_public_move_action(
    *,
    env,
    robot_name: str,
    target_pose,
    sample_num: int,
    kwargs: dict[str, Any],
    log_name: str,
):
    strict = _public_non_grasp_strict(kwargs)
    if env is None:
        _handle_public_non_grasp_unavailable(
            f"Public move action cannot run for {log_name}: env is None.",
            strict=strict,
            fallback_name="move action",
        )
        return None
    if target_pose is None:
        _handle_public_non_grasp_unavailable(
            f"Public move action cannot run for {log_name}: target_pose is None.",
            strict=strict,
            fallback_name="move action",
        )
        return None
    if not _public_atomic_actions_enabled(kwargs):
        _handle_public_non_grasp_unavailable(
            f"Public move action is disabled for {log_name}.",
            strict=strict,
            fallback_name="move action",
        )
        return None

    _, arm_part, _ = _select_arm(robot_name)
    cfg = MoveActionCfg(
        control_part=arm_part,
        sample_interval=sample_num,
    )
    return _run_public_action(
        env,
        robot_name,
        cfg,
        target_pose,
        log_name=log_name,
        strict=strict,
    )


def try_public_gripper_action(
    *,
    env,
    robot_name: str,
    target_state,
    sample_num: int,
    kwargs: dict[str, Any],
    log_name: str,
):
    strict = _public_non_grasp_strict(kwargs)
    if env is None:
        _handle_public_non_grasp_unavailable(
            f"Public gripper action cannot run for {log_name}: env is None.",
            strict=strict,
            fallback_name="gripper action",
        )
        return None
    if target_state is None:
        _handle_public_non_grasp_unavailable(
            f"Public gripper action cannot run for {log_name}: target_state is None.",
            strict=strict,
            fallback_name="gripper action",
        )
        return None
    if not _public_atomic_actions_enabled(kwargs):
        _handle_public_non_grasp_unavailable(
            f"Public gripper action is disabled for {log_name}.",
            strict=strict,
            fallback_name="gripper action",
        )
        return None
    if not kwargs.get("use_public_gripper_action", True):
        _handle_public_non_grasp_unavailable(
            f"Public gripper action is disabled for {log_name}.",
            strict=strict,
            fallback_name="gripper action",
        )
        return None

    try:
        engine = _create_engine(
            env,
            MoveActionCfg(
                name="move",
                control_part=_select_arm(robot_name)[2],
                sample_interval=sample_num,
            ),
        )
        is_left, _, _ = _select_arm(robot_name)
        hand_joints = env.left_eef_joints if is_left else env.right_eef_joints
        hand_dof = len(hand_joints)
        start_qpos = _state_to_hand_qpos(
            _current_gripper_state(env, is_left),
            hand_dof,
            device=engine.device,
        ).unsqueeze(0)
        target_qpos = _state_to_hand_qpos(
            target_state,
            hand_dof,
            device=engine.device,
        )

        atom_action = engine._actions["move"]
        is_success, trajectory, joint_ids = atom_action.execute(
            target=target_qpos,
            start_qpos=start_qpos,
        )
        if not is_success or trajectory.numel() == 0:
            if strict:
                raise RuntimeError(
                    f"Public atomic action adapter failed for {log_name}."
                )
            log_warning(
                f"Public atomic action adapter failed for {log_name}; "
                "falling back to legacy agent gripper action."
            )
            return None

        legacy_action = _extract_legacy_gripper_action(
            env, robot_name, trajectory, joint_ids
        )
        _sync_agent_gripper_state(env, robot_name, legacy_action)
        return legacy_action
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Public atomic action adapter raised {type(exc).__name__} for "
                f"{log_name}: {exc}."
            ) from exc
        log_warning(
            f"Public atomic action adapter raised {type(exc).__name__} for "
            f"{log_name}: {exc}. Falling back to legacy agent gripper action."
        )
        return None


def _build_legacy_grasp_pose(
    env,
    robot_name: str,
    obj_name: str,
    *,
    preserve_object_rotation: bool = False,
):
    obj_uids = env.sim.get_rigid_object_uid_list()
    if obj_name not in obj_uids:
        return None

    target_obj = env.sim.get_rigid_object(obj_name)
    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    target_obj_pose = _as_pose_tensor(target_obj_pose)
    obj_info = getattr(env, "obj_info", {})
    grasp_pose_object = obj_info.get(obj_name, {}).get("grasp_pose_obj")
    if grasp_pose_object is None:
        return None
    grasp_pose_object = _as_pose_tensor(
        grasp_pose_object, device=target_obj_pose.device
    )

    is_left, _, _ = _select_arm(robot_name)
    select_arm_base_pose = (
        env.left_arm_base_pose if is_left else env.right_arm_base_pose
    )
    delta_xy = target_obj_pose[:2, 3] - select_arm_base_pose[:2, 3]
    aim_horizontal_angle = torch.atan2(delta_xy[1], delta_xy[0]).item()

    if not preserve_object_rotation and grasp_pose_object[0, 2] > 0.5:
        target_obj_pose = torch.as_tensor(
            get_rotation_replaced_pose(
                target_obj_pose.detach().cpu().numpy(),
                float(aim_horizontal_angle),
                "z",
                "intrinsic",
            ),
            dtype=torch.float32,
            device=target_obj_pose.device,
        )

    return target_obj_pose @ grasp_pose_object


def _auto_horizontal_grasp_direction(
    env,
    robot_name: str,
    obj_name: str,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor | None:
    target_obj = env.sim.get_rigid_object(obj_name)
    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    target_obj_pose = _as_pose_tensor(target_obj_pose, device=device)
    is_left, _, _ = _select_arm(robot_name)
    arm_base_pose = env.left_arm_base_pose if is_left else env.right_arm_base_pose
    arm_base_pose = _as_pose_tensor(arm_base_pose, device=target_obj_pose.device)
    direction = torch.zeros(3, dtype=torch.float32, device=target_obj_pose.device)
    direction[:2] = target_obj_pose[:2, 3] - arm_base_pose[:2, 3]
    direction_norm = torch.linalg.norm(direction)
    if direction_norm <= 1e-6:
        return None
    return direction / direction_norm


def _object_geometry_axes(
    env,
    obj_name: str,
    *,
    device: torch.device | str | None = None,
) -> tuple[dict[str, float], list[tuple[str, torch.Tensor]]]:
    target_obj = env.sim.get_rigid_object(obj_name)
    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    target_obj_pose = _as_pose_tensor(target_obj_pose, device=device)
    vertices = target_obj.get_vertices(env_ids=[0], scale=True)[0]
    vertices = torch.as_tensor(
        vertices, dtype=torch.float32, device=target_obj_pose.device
    )
    if vertices.numel() == 0:
        return _empty_object_geometry_stats(), []

    world_vertices = vertices @ target_obj_pose[:3, :3].transpose(0, 1)
    world_vertices = world_vertices + target_obj_pose[:3, 3]
    stats = _object_geometry_stats_from_world_vertices(world_vertices)

    centered_xy = world_vertices[:, :2] - world_vertices[:, :2].mean(dim=0)
    axes: list[tuple[str, torch.Tensor]] = []
    if centered_xy.shape[0] >= 3 and torch.linalg.norm(centered_xy) > 1e-6:
        covariance = centered_xy.transpose(0, 1) @ centered_xy / centered_xy.shape[0]
        _, eigvecs = torch.linalg.eigh(covariance)
        for idx, axis_idx in enumerate((1, 0)):
            axis_xy = eigvecs[:, axis_idx]
            axis = torch.tensor(
                [axis_xy[0], axis_xy[1], 0.0],
                dtype=torch.float32,
                device=target_obj_pose.device,
            )
            if torch.linalg.norm(axis) > 1e-6:
                axes.append((f"object_pca_{idx}", axis))
    return stats, axes


def _empty_object_geometry_stats() -> dict[str, float]:
    return {
        "height": 0.0,
        "xy_extent": 0.0,
        "xy_long_extent": 0.0,
        "xy_short_extent": 0.0,
        "slenderness": 0.0,
        "xy_slenderness": 0.0,
    }


def _object_geometry_stats(
    vertices,
    pose,
    *,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    pose = _as_pose_tensor(pose, device=device)
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=pose.device)
    if vertices.numel() == 0:
        return _empty_object_geometry_stats()
    world_vertices = vertices @ pose[:3, :3].transpose(0, 1)
    world_vertices = world_vertices + pose[:3, 3]
    return _object_geometry_stats_from_world_vertices(world_vertices)


def _object_geometry_stats_from_world_vertices(
    world_vertices: torch.Tensor,
) -> dict[str, float]:
    mins = world_vertices.min(dim=0).values
    maxs = world_vertices.max(dim=0).values
    extents = maxs - mins
    xy_long_extent = float(torch.clamp(extents[:2].max(), min=1e-6).item())
    xy_short_extent = float(torch.clamp(extents[:2].min(), min=1e-6).item())
    centered_xy = world_vertices[:, :2] - world_vertices[:, :2].mean(dim=0)
    if centered_xy.shape[0] >= 3 and torch.linalg.norm(centered_xy) > 1e-6:
        covariance = centered_xy.transpose(0, 1) @ centered_xy / centered_xy.shape[0]
        _, eigvecs = torch.linalg.eigh(covariance)
        projected = centered_xy @ eigvecs
        pca_extents = projected.max(dim=0).values - projected.min(dim=0).values
        pca_long_extent = float(torch.clamp(pca_extents.max(), min=1e-6).item())
        pca_short_extent = float(torch.clamp(pca_extents.min(), min=1e-6).item())
        if pca_long_extent / pca_short_extent > xy_long_extent / xy_short_extent:
            xy_long_extent = pca_long_extent
            xy_short_extent = pca_short_extent
    height = float(torch.clamp(extents[2], min=0.0).item())
    return {
        "height": height,
        "xy_extent": xy_long_extent,
        "xy_long_extent": xy_long_extent,
        "xy_short_extent": xy_short_extent,
        "slenderness": height / xy_long_extent,
        "xy_slenderness": xy_long_extent / xy_short_extent,
    }


def _set_thin_object_public_grasp_defaults(kwargs: dict[str, Any]) -> None:
    _set_if_missing_or_none(kwargs, "public_grasp_thin_object_xy_slenderness", 3.0)
    _set_if_missing_or_none(kwargs, "public_grasp_thin_object_open_check_margin", 0.004)
    _set_if_missing_or_none(kwargs, "public_grasp_thin_object_min_open_length", 0.001)
    _set_if_missing_or_none(kwargs, "public_grasp_thin_object_lift_height", 0.06)
    _set_if_missing_or_none(
        kwargs,
        "public_grasp_thin_object_filter_ground_collision",
        False,
    )


def _is_public_grasp_thin_object(
    geometry_stats: dict[str, float],
    kwargs: dict[str, Any],
) -> bool:
    threshold = float(kwargs.get("public_grasp_thin_object_xy_slenderness", 3.0))
    return float(geometry_stats.get("xy_slenderness", 0.0)) >= threshold


def _public_grasp_effective_antipodal_min_open_length(
    geometry_stats: dict[str, float],
    kwargs: dict[str, Any],
) -> float:
    min_open_length = float(kwargs.get("grasp_min_open_length", 0.003))
    if _is_public_grasp_thin_object(geometry_stats, kwargs):
        thin_min_open_length = float(
            kwargs.get("public_grasp_thin_object_min_open_length", 0.001)
        )
        min_open_length = min(min_open_length, thin_min_open_length)
    return min_open_length


def _public_grasp_effective_open_check_margin(
    geometry_stats: dict[str, float],
    kwargs: dict[str, Any],
) -> float:
    open_check_margin = float(kwargs.get("grasp_open_check_margin", 0.01))
    if _is_public_grasp_thin_object(geometry_stats, kwargs):
        thin_open_check_margin = float(
            kwargs.get("public_grasp_thin_object_open_check_margin", 0.004)
        )
        open_check_margin = min(open_check_margin, thin_open_check_margin)
    return open_check_margin


def _public_grasp_effective_filter_ground_collision(
    geometry_stats: dict[str, float],
    kwargs: dict[str, Any],
) -> bool:
    if _is_public_grasp_thin_object(geometry_stats, kwargs):
        return bool(
            kwargs.get("public_grasp_thin_object_filter_ground_collision", False)
        )
    return bool(kwargs.get("public_grasp_filter_ground_collision", True))


def _with_effective_thin_object_public_grasp_kwargs(
    env,
    obj_name: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    geometry_stats, _ = _object_geometry_axes(
        env,
        obj_name,
        device=getattr(env.robot, "device", None),
    )
    if not _is_public_grasp_thin_object(geometry_stats, kwargs):
        return kwargs
    updated = dict(kwargs)
    _set_thin_object_public_grasp_defaults(updated)
    updated["public_grasp_is_thin_object"] = True
    _set_if_missing_or_none(
        updated,
        "public_grasp_thin_object_prefer_top_down",
        True,
    )
    lift_height = _public_grasp_lift_height(updated)
    thin_lift_height = float(updated.get("public_grasp_thin_object_lift_height", 0.06))
    if lift_height <= 0.0:
        updated["public_grasp_lift_height"] = thin_lift_height
    else:
        updated["public_grasp_lift_height"] = min(lift_height, thin_lift_height)
    return updated


def _public_grasp_effective_candidate_min_open_length(
    geometry_stats: dict[str, float],
    kwargs: dict[str, Any],
    configured_min_open_length: float | None,
) -> float | None:
    if not _is_public_grasp_thin_object(geometry_stats, kwargs):
        return configured_min_open_length
    thin_min_open_length = float(
        kwargs.get("public_grasp_thin_object_min_open_length", 0.001)
    )
    if configured_min_open_length is None:
        return thin_min_open_length
    return min(float(configured_min_open_length), thin_min_open_length)


def _public_grasp_attempt_context(
    geometry_stats: dict[str, float],
    kwargs: dict[str, Any],
    *,
    candidate_min_open_length: float | None,
    candidate_max_open_length: float | None,
) -> str:
    thin_object = _is_public_grasp_thin_object(geometry_stats, kwargs)
    return (
        f"geometry_height={geometry_stats['height']:.5f}, "
        f"xy_long_extent={geometry_stats['xy_long_extent']:.5f}, "
        f"xy_short_extent={geometry_stats['xy_short_extent']:.5f}, "
        f"height_slenderness={geometry_stats['slenderness']:.5f}, "
        f"xy_slenderness={geometry_stats['xy_slenderness']:.5f}, "
        f"thin_object={thin_object}, "
        f"grasp_min_open_length="
        f"{_public_grasp_effective_antipodal_min_open_length(geometry_stats, kwargs):.5f}, "
        f"grasp_max_open_length={float(kwargs.get('grasp_max_open_length', 0.088)):.5f}, "
        f"grasp_open_check_margin="
        f"{_public_grasp_effective_open_check_margin(geometry_stats, kwargs):.5f}, "
        f"lift_height={_public_grasp_lift_height(kwargs):.5f}, "
        f"filter_ground_collision="
        f"{_public_grasp_effective_filter_ground_collision(geometry_stats, kwargs)}, "
        f"candidate_min_open_length="
        f"{_format_optional_metric(candidate_min_open_length)}, "
        f"candidate_max_open_length="
        f"{_format_optional_metric(candidate_max_open_length)}"
    )


def _public_grasp_strategy(kwargs: dict[str, Any]) -> str | None:
    strategy = kwargs.get("public_grasp_strategy")
    if strategy is None:
        return None
    strategy = str(strategy).strip().lower().replace("-", "_")
    if not strategy or strategy in {"none", "default"}:
        return None
    return strategy


def _with_public_grasp_strategy_defaults(kwargs: dict[str, Any]) -> dict[str, Any]:
    strategy = _public_grasp_strategy(kwargs)
    if strategy is None:
        return kwargs

    updated = dict(kwargs)
    if strategy == "top_down":
        _set_if_missing_or_none(updated, "public_grasp_pre_grasp_distance", 0.05)
        _set_strategy_public_grasp_candidate_num(updated, 32)
        return updated

    if strategy in {"bottle_lateral", "lateral_down"}:
        updated["public_grasp_use_candidate_selection"] = True
        updated.setdefault("public_grasp_rank_mode", "planned_order")
        _set_strategy_public_grasp_candidate_num(updated, 32)
        _set_if_missing_or_none(updated, "public_grasp_pre_grasp_distance", 0.05)
        _set_if_missing_or_none(updated, "public_grasp_lift_height", 0.15)
        _set_if_missing_or_none(updated, "public_grasp_lateral_down_z", -0.34)
        _set_if_missing_or_none(
            updated,
            "public_grasp_candidate_min_open_length",
            0.055,
        )
        if _is_zero_vector_like(updated.get("public_grasp_pose_offset_world")):
            updated["public_grasp_pose_offset_world"] = [0.0, 0.0, 0.011]
        if not updated.get("public_grasp_pose_offset_along_approach"):
            updated["public_grasp_pose_offset_along_approach"] = 0.025
        return updated

    if strategy == "legacy_guided":
        updated["public_grasp_try_approach_directions"] = True
        updated["public_grasp_rank_by_legacy_pose"] = True
        updated["public_grasp_use_legacy_orientation"] = True
        _set_thin_object_public_grasp_defaults(updated)
        _set_strategy_public_grasp_candidate_num(updated, 32)
        _set_if_missing_or_none(updated, "public_grasp_pre_grasp_distance", 0.05)
        _set_if_missing_or_none(updated, "public_grasp_lift_height", 0.15)
        return updated

    if strategy in {"auto_try_all", "auto_general"}:
        updated["public_grasp_try_approach_directions"] = True
        updated["public_grasp_use_candidate_selection"] = True
        updated.setdefault(
            "public_grasp_rank_mode",
            "",
        )
        _set_strategy_public_grasp_candidate_num(
            updated,
            64 if strategy == "auto_general" else 32,
        )
        _set_if_missing_or_none(updated, "public_grasp_pre_grasp_distance", 0.05)
        if strategy == "auto_general":
            _set_thin_object_public_grasp_defaults(updated)
            _set_if_missing_or_none(updated, "public_grasp_lift_height", 0.15)
            _set_if_missing_or_none(
                updated, "public_grasp_auto_general_slenderness", 1.2
            )
            _set_if_missing_or_none(
                updated, "public_grasp_auto_general_lateral_down_z", -0.28
            )
            _set_if_missing_or_none(
                updated, "public_grasp_top_down_target_height_ratio", 0.75
            )
            _set_if_missing_or_none(
                updated, "public_grasp_top_down_target_height_weight", 0.35
            )
            _set_if_missing_or_none(
                updated, "public_grasp_validation_min_object_lift", 0.04
            )
            _set_if_missing_or_none(
                updated, "public_grasp_validation_max_object_xy_displacement", 0.12
            )
            if updated.get("validate_public_grasp_after_action") is None:
                updated["validate_public_grasp_after_action"] = True
        return updated

    raise ValueError(
        "Unsupported public_grasp_strategy "
        f"'{kwargs.get('public_grasp_strategy')}'. Expected one of "
        "top_down, bottle_lateral, lateral_down, legacy_guided, auto_try_all, "
        "auto_general."
    )


def _is_zero_vector_like(value) -> bool:
    if value is None:
        return True
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    except Exception:
        return False
    return bool(tensor.numel() == 3 and torch.linalg.norm(tensor).item() <= 1e-8)


def _set_strategy_public_grasp_candidate_num(
    kwargs: dict[str, Any],
    default_minimum: int,
) -> None:
    if kwargs.get("_recovery_public_grasp_candidate_num_override"):
        return
    kwargs["public_grasp_candidate_num"] = max(
        int(kwargs.get("public_grasp_candidate_num") or 0),
        int(default_minimum),
    )


def _set_if_missing_or_none(kwargs: dict[str, Any], name: str, value: Any) -> None:
    if kwargs.get(name) is None:
        kwargs[name] = value


def _append_unique_grasp_direction(
    candidates: list[tuple[str, torch.Tensor]],
    label: str,
    direction,
    *,
    device: torch.device | str | None = None,
) -> None:
    normalized = _normalized_vector_tensor(
        direction,
        3,
        name=f"public grasp approach direction '{label}'",
        device=device,
    )
    for _, existing in candidates:
        if torch.allclose(existing, normalized, rtol=1e-4, atol=1e-4):
            return
    candidates.append((label, normalized))


def _object_local_grasp_axes(
    env,
    obj_name: str,
    *,
    device: torch.device | str | None = None,
) -> list[tuple[str, torch.Tensor]]:
    target_obj = env.sim.get_rigid_object(obj_name)
    target_obj_pose = target_obj.get_local_pose(to_matrix=True).squeeze(0)
    target_obj_pose = _as_pose_tensor(target_obj_pose, device=device)
    object_x = target_obj_pose[:3, 0]
    object_y = target_obj_pose[:3, 1]
    return [
        ("object_x", object_x),
        ("object_neg_x", -object_x),
        ("object_y", object_y),
        ("object_neg_y", -object_y),
    ]


def _append_object_local_grasp_directions(
    candidates: list[tuple[str, torch.Tensor]],
    env,
    obj_name: str,
    *,
    device: torch.device | str | None = None,
) -> None:
    for label, direction in _object_local_grasp_axes(env, obj_name, device=device):
        _append_unique_grasp_direction(
            candidates,
            label,
            direction,
            device=device,
        )


def _append_object_local_lateral_down_grasp_directions(
    candidates: list[tuple[str, torch.Tensor]],
    env,
    obj_name: str,
    kwargs: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> None:
    for label, direction in _object_local_grasp_axes(env, obj_name, device=device):
        _append_lateral_down_grasp_direction(
            candidates,
            f"{label}_down",
            direction,
            kwargs,
            device=device,
        )


def _append_lateral_down_grasp_direction(
    candidates: list[tuple[str, torch.Tensor]],
    label: str,
    direction: torch.Tensor,
    kwargs: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> None:
    lateral_down_z = float(
        kwargs.get("public_grasp_auto_general_lateral_down_z", -0.28)
    )
    lateral_direction = direction.clone()
    lateral_direction[2] = lateral_down_z
    _append_unique_grasp_direction(
        candidates,
        label,
        lateral_direction,
        device=device,
    )


def _append_auto_general_grasp_directions(
    candidates: list[tuple[str, torch.Tensor]],
    env,
    robot_name: str,
    obj_name: str,
    kwargs: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> None:
    auto_direction = _auto_horizontal_grasp_direction(
        env,
        robot_name,
        obj_name,
        device=device,
    )
    stats, geometry_axes = _object_geometry_axes(env, obj_name, device=device)
    slenderness_threshold = float(
        kwargs.get("public_grasp_auto_general_slenderness", 1.2)
    )
    thin_object = _is_public_grasp_thin_object(stats, kwargs)
    prefer_lateral = stats["slenderness"] >= slenderness_threshold or thin_object

    if prefer_lateral and auto_direction is not None:
        _append_lateral_down_grasp_direction(
            candidates,
            "auto_arm_lateral_down",
            auto_direction,
            kwargs,
            device=device,
        )
    if prefer_lateral:
        for label, axis in geometry_axes:
            _append_lateral_down_grasp_direction(
                candidates,
                f"auto_{label}_down",
                axis,
                kwargs,
                device=device,
            )
            _append_lateral_down_grasp_direction(
                candidates,
                f"auto_neg_{label}_down",
                -axis,
                kwargs,
                device=device,
            )

    _append_unique_grasp_direction(
        candidates,
        "auto_top_down",
        [0.0, 0.0, -1.0],
        device=device,
    )

    if not prefer_lateral and auto_direction is not None:
        _append_lateral_down_grasp_direction(
            candidates,
            "auto_arm_lateral_down",
            auto_direction,
            kwargs,
            device=device,
        )
    if not prefer_lateral:
        for label, axis in geometry_axes:
            _append_lateral_down_grasp_direction(
                candidates,
                f"auto_{label}_down",
                axis,
                kwargs,
                device=device,
            )
            _append_lateral_down_grasp_direction(
                candidates,
                f"auto_neg_{label}_down",
                -axis,
                kwargs,
                device=device,
            )

    if auto_direction is not None:
        _append_unique_grasp_direction(
            candidates,
            "auto_arm_to_object",
            auto_direction,
            device=device,
        )
        _append_unique_grasp_direction(
            candidates,
            "auto_object_to_arm",
            -auto_direction,
            device=device,
        )
    if thin_object:
        _append_object_local_lateral_down_grasp_directions(
            candidates,
            env,
            obj_name,
            kwargs,
            device=device,
        )
    _append_object_local_grasp_directions(candidates, env, obj_name, device=device)


def _public_grasp_approach_direction_candidates(
    env,
    robot_name: str,
    obj_name: str,
    kwargs: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> list[tuple[str, torch.Tensor]]:
    configured_directions = kwargs.get("public_grasp_approach_directions")
    candidates: list[tuple[str, torch.Tensor]] = []
    if configured_directions:
        for idx, direction in enumerate(configured_directions):
            _append_unique_grasp_direction(
                candidates,
                f"configured_{idx}",
                direction,
                device=device,
            )
        return candidates

    explicit_direction = kwargs.get("public_grasp_approach_direction")
    if explicit_direction is not None:
        _append_unique_grasp_direction(
            candidates,
            "explicit",
            explicit_direction,
            device=device,
        )
        return candidates

    strategy = _public_grasp_strategy(kwargs)
    auto_direction = _auto_horizontal_grasp_direction(
        env,
        robot_name,
        obj_name,
        device=device,
    )
    if strategy == "top_down":
        _append_unique_grasp_direction(
            candidates,
            "top_down",
            [0.0, 0.0, -1.0],
            device=device,
        )
        return candidates

    if strategy in {"bottle_lateral", "lateral_down"}:
        if auto_direction is None:
            raise ValueError(
                f"Cannot build {strategy} approach direction for '{obj_name}'."
            )
        lateral_down_z = float(kwargs.get("public_grasp_lateral_down_z", -0.34))
        direction = auto_direction.clone()
        direction[2] = lateral_down_z
        _append_unique_grasp_direction(
            candidates,
            strategy,
            direction,
            device=device,
        )
        return candidates

    if strategy == "auto_general":
        _append_auto_general_grasp_directions(
            candidates,
            env,
            robot_name,
            obj_name,
            kwargs,
            device=device,
        )
        return candidates

    if bool(kwargs.get("public_grasp_try_approach_directions", False)):
        _append_unique_grasp_direction(
            candidates,
            "top_down",
            [0.0, 0.0, -1.0],
            device=device,
        )
        if auto_direction is not None:
            _append_unique_grasp_direction(
                candidates,
                "arm_to_object",
                auto_direction,
                device=device,
            )
            _append_unique_grasp_direction(
                candidates,
                "object_to_arm",
                -auto_direction,
                device=device,
            )
            side_left = torch.stack(
                [-auto_direction[1], auto_direction[0], auto_direction[2]]
            )
            _append_unique_grasp_direction(
                candidates,
                "arm_side_left",
                side_left,
                device=device,
            )
            _append_unique_grasp_direction(
                candidates,
                "arm_side_right",
                -side_left,
                device=device,
            )
        stats, _ = _object_geometry_axes(env, obj_name, device=device)
        if _is_public_grasp_thin_object(stats, kwargs):
            _append_object_local_lateral_down_grasp_directions(
                candidates,
                env,
                obj_name,
                kwargs,
                device=device,
            )
        _append_object_local_grasp_directions(
            candidates,
            env,
            obj_name,
            device=device,
        )
        return candidates

    if bool(kwargs.get("public_grasp_auto_approach_direction", False)):
        if auto_direction is not None:
            _append_unique_grasp_direction(
                candidates,
                "arm_to_object",
                auto_direction,
                device=device,
            )
            return candidates

    _append_unique_grasp_direction(
        candidates,
        "top_down",
        [0.0, 0.0, -1.0],
        device=device,
    )
    return candidates


def _public_grasp_approach_direction(
    env,
    robot_name: str,
    obj_name: str,
    kwargs: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return _public_grasp_approach_direction_candidates(
        env,
        robot_name,
        obj_name,
        kwargs,
        device=device,
    )[0][1]


def _build_pickup_cfg(
    *,
    env,
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float,
    kwargs: dict[str, Any],
    approach_direction: torch.Tensor,
) -> PickUpActionCfg:
    is_left, arm_part, hand_part = _select_arm(robot_name)
    hand_joints = env.left_eef_joints if is_left else env.right_eef_joints
    hand_dof = len(hand_joints)
    device = getattr(env.robot, "device", None)
    lift_height = _public_grasp_lift_height(kwargs)
    offset_world, offset_along_approach = _public_grasp_pose_offsets(
        kwargs,
        approach_direction=approach_direction,
    )
    grasp_pose_offset_world = _as_vector_tensor(
        offset_world,
        3,
        name="public_grasp_pose_offset_world",
        device=device,
    )
    public_pre_grasp_distance = kwargs.get("public_grasp_pre_grasp_distance")
    if public_pre_grasp_distance is None:
        public_pre_grasp_distance = pre_grasp_dis
    public_pre_grasp_distance = float(public_pre_grasp_distance)
    rank_options = dict(kwargs)
    for source_key, target_key in (
        ("_public_grasp_approach_directions", "approach_directions"),
        ("_public_grasp_reference_pose", "reference_grasp_pose"),
        ("_public_grasp_geometry_bounds", "geometry_bounds"),
        ("_public_grasp_roll_offsets", "roll_offsets"),
        ("_public_grasp_attempt_context", "attempt_context"),
        (
            "_public_grasp_auto_general_preferred_roll_offset",
            "auto_general_preferred_roll_offset",
        ),
    ):
        if kwargs.get(source_key) is not None:
            rank_options[target_key] = kwargs[source_key]
    return PickUpActionCfg(
        control_part=arm_part,
        hand_control_part=hand_part,
        hand_open_qpos=_state_to_hand_qpos(env.open_state, hand_dof, device=device),
        hand_close_qpos=_state_to_hand_qpos(env.close_state, hand_dof, device=device),
        pre_grasp_distance=public_pre_grasp_distance,
        approach_direction=approach_direction.to(device=device),
        grasp_candidate_num=int(kwargs.get("public_grasp_candidate_num", 8)),
        grasp_pose_offset_world=grasp_pose_offset_world,
        grasp_pose_offset_along_approach=float(offset_along_approach),
        lift_height=lift_height,
        sample_interval=kwargs.get("sample_interval", kwargs.get("sample_num", 75)),
        hand_interp_steps=kwargs.get("hand_interp_steps", 15),
        ranked_grasp_selection=bool(
            kwargs.get("public_grasp_use_candidate_selection", False)
            or kwargs.get("public_grasp_rank_by_legacy_pose", False)
            or kwargs.get("public_grasp_use_legacy_orientation", False)
            or kwargs.get("public_grasp_legacy_pose_max_position_error") is not None
            or kwargs.get("public_grasp_legacy_pose_max_rotation_error") is not None
            or kwargs.get("_public_grasp_approach_directions") is not None
        ),
        grasp_approach_directions=kwargs.get("_public_grasp_approach_directions"),
        grasp_rank_options=rank_options,
    )


def _public_grasp_pose_offsets(
    kwargs: dict[str, Any],
    *,
    approach_direction: torch.Tensor,
) -> tuple[Any, float]:
    offset_world = kwargs.get("public_grasp_pose_offset_world", [0.0, 0.0, 0.0])
    offset_along_approach = float(
        kwargs.get("public_grasp_pose_offset_along_approach", 0.0) or 0.0
    )
    if _public_grasp_strategy(kwargs) != "auto_general":
        return offset_world, offset_along_approach

    direction = approach_direction.detach().to(dtype=torch.float32)
    direction = direction / torch.linalg.norm(direction).clamp_min(1e-6)
    is_lateral = abs(float(direction[2].item())) < 0.75
    if _is_zero_vector_like(offset_world):
        offset_world = [0.0, 0.0, 0.011 if is_lateral else 0.006]
    if offset_along_approach == 0.0:
        offset_along_approach = 0.025 if is_lateral else 0.015
    return offset_world, offset_along_approach


def _public_grasp_lift_height(kwargs: dict[str, Any]) -> float:
    lift_height = kwargs.get("public_grasp_lift_height")
    if lift_height is None:
        lift_height = kwargs.get("lift_height", 0.0)
    if lift_height is None:
        lift_height = 0.0
    return float(lift_height)


def _public_grasp_candidate_selection_enabled(kwargs: dict[str, Any]) -> bool:
    return bool(
        kwargs.get("public_grasp_use_candidate_selection", False)
        or kwargs.get("public_grasp_rank_by_legacy_pose", False)
        or kwargs.get("public_grasp_use_legacy_orientation", False)
        or kwargs.get("public_grasp_legacy_pose_max_position_error") is not None
        or kwargs.get("public_grasp_legacy_pose_max_rotation_error") is not None
    )


def _public_grasp_legacy_reference_required(kwargs: dict[str, Any]) -> bool:
    return bool(
        kwargs.get("public_grasp_rank_by_legacy_pose", False)
        or kwargs.get("public_grasp_use_legacy_orientation", False)
        or kwargs.get("public_grasp_legacy_pose_max_position_error") is not None
        or kwargs.get("public_grasp_legacy_pose_max_rotation_error") is not None
        or kwargs.get("public_grasp_validate_relative_to_legacy_pose", False)
    )


def _is_cobotmagic_env(env) -> bool:
    robot = getattr(env, "robot", None)
    robot_text = " ".join(
        str(value)
        for value in (
            getattr(robot, "uid", ""),
            getattr(robot, "name", ""),
            robot.__class__.__name__ if robot is not None else "",
        )
    )
    return "cobotmagic" in robot_text.lower()


def _public_grasp_roll_offsets(env, kwargs: dict[str, Any]) -> list[float]:
    configured = kwargs.get("public_grasp_roll_offsets")
    if configured is not None:
        if isinstance(configured, str):
            configured_values = [
                value.strip()
                for value in configured.replace(";", ",").split(",")
                if value.strip()
            ]
        else:
            configured_values = list(configured)
        roll_offsets = [float(value) for value in configured_values]
        return roll_offsets or [0.0]

    if not bool(kwargs.get("public_grasp_try_eef_roll_offsets", True)):
        return [0.0]
    if _public_grasp_strategy(kwargs) == "auto_general" and _is_cobotmagic_env(env):
        return [0.0, math.pi / 2.0, -math.pi / 2.0, math.pi]
    return [0.0]


def _format_optional_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.5f}"


def _run_ranked_public_semantic_grasp_action(
    *,
    env,
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float,
    kwargs: dict[str, Any],
    target: ObjectSemantics,
    directions: list[tuple[str, torch.Tensor]],
    strict: bool,
) -> np.ndarray | None:
    try:
        semantic_plan = plan_public_semantic_grasp_action(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            pre_grasp_dis=pre_grasp_dis,
            kwargs=kwargs,
            target=target,
            directions=directions,
        )
    except Exception as exc:
        _handle_public_grasp_unavailable(
            f"Public semantic grasp candidate selection raised "
            f"{type(exc).__name__} for '{obj_name}': {exc}.",
            strict=strict,
        )
        return None

    legacy_action = _extract_legacy_action(
        env,
        robot_name,
        semantic_plan.trajectory,
        semantic_plan.joint_ids,
    )
    _sync_agent_arm_state(env, robot_name, legacy_action)
    return legacy_action


def plan_public_semantic_grasp_action(
    *,
    env,
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float,
    kwargs: dict[str, Any],
    target: ObjectSemantics,
    directions: list[tuple[str, torch.Tensor]] | None = None,
) -> PublicSemanticGraspPlan:
    """Plan one ranked semantic grasp through canonical PickUpAction."""
    kwargs = _with_public_grasp_strategy_defaults(kwargs)
    kwargs = _with_effective_thin_object_public_grasp_kwargs(env, obj_name, kwargs)
    if directions is None:
        directions = _public_grasp_approach_direction_candidates(
            env,
            robot_name,
            obj_name,
            kwargs,
            device=getattr(env.robot, "device", None),
        )

    legacy_reference_pose = (
        _build_legacy_grasp_pose(env, robot_name, obj_name)
        if _public_grasp_legacy_reference_required(kwargs)
        else None
    )
    if (
        _public_grasp_legacy_reference_required(kwargs)
        and legacy_reference_pose is None
    ):
        raise RuntimeError(
            "Public semantic grasp candidate selection requires legacy "
            f"grasp_pose_obj reference for '{obj_name}'."
        )

    rank_kwargs = dict(kwargs)
    rank_kwargs["obj_name"] = obj_name
    rank_kwargs["_public_grasp_approach_directions"] = directions
    rank_kwargs["_public_grasp_reference_pose"] = legacy_reference_pose
    rank_kwargs["_public_grasp_geometry_bounds"] = _object_geometry_bounds(
        env,
        obj_name,
        device=getattr(env.robot, "device", None),
    )
    rank_kwargs["_public_grasp_roll_offsets"] = _public_grasp_roll_offsets(
        env,
        rank_kwargs,
    )
    cfg = _build_pickup_cfg(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        pre_grasp_dis=pre_grasp_dis,
        kwargs=rank_kwargs,
        approach_direction=directions[0][1],
    )
    engine = _create_engine(env, cfg)
    is_success, _trajectory = engine.execute_static(target_list=[target])
    atom_action = engine._actions[cfg.name]
    selected = getattr(atom_action, "last_selected_grasp", None)
    if not is_success or selected is None:
        failure_text = "; ".join(getattr(atom_action, "last_grasp_failures", []))
        if not failure_text:
            failure_text = "no planned candidates"
        raise RuntimeError(
            "Public semantic grasp candidate selection failed for "
            f"'{obj_name}': {failure_text}."
        )

    label = _semantic_candidate_record_label(selected)
    _store_public_grasp_relation(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        grasp_pose=selected.grasp_pose,
        source=f"semantic:{label}",
    )
    _record_public_grasp_attempt(
        env=env,
        kwargs=kwargs,
        obj_name=obj_name,
        label=label,
        direction=selected.direction,
        status="selected",
        message=_semantic_candidate_message(selected),
    )
    _register_public_grasp_physical_validation(
        env=env,
        kwargs=kwargs,
        robot_name=robot_name,
        obj_name=obj_name,
        label=label,
        direction=selected.direction,
        lift_height=_public_grasp_lift_height(kwargs),
        legacy_reference_pose=legacy_reference_pose,
    )
    return PublicSemanticGraspPlan(
        label=label,
        direction=selected.direction.detach().clone(),
        grasp_pose=selected.grasp_pose.detach().clone(),
        trajectory=selected.trajectory.detach().clone(),
        joint_ids=list(selected.joint_ids),
        lift_height=_public_grasp_lift_height(kwargs),
    )


def _run_public_semantic_grasp_action(
    *,
    env,
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float,
    kwargs: dict[str, Any],
    target: ObjectSemantics,
    strict: bool,
) -> np.ndarray | None:
    device = getattr(env.robot, "device", None)
    try:
        directions = _public_grasp_approach_direction_candidates(
            env,
            robot_name,
            obj_name,
            kwargs,
            device=device,
        )
    except Exception as exc:
        _handle_public_grasp_unavailable(
            f"Public semantic grasp cannot build approach directions for "
            f"'{obj_name}': {exc}.",
            strict=strict,
        )
        return None

    if _public_grasp_candidate_selection_enabled(kwargs):
        return _run_ranked_public_semantic_grasp_action(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            pre_grasp_dis=pre_grasp_dis,
            kwargs=kwargs,
            target=target,
            directions=directions,
            strict=strict,
        )

    legacy_reference_pose = (
        _build_legacy_grasp_pose(env, robot_name, obj_name)
        if _public_grasp_legacy_reference_required(kwargs)
        else None
    )
    failures: list[str] = []
    for label, direction in directions:
        log_name = f"semantic_grasp:{obj_name}:{label}"
        log_info(
            f"Trying public semantic grasp direction {label}="
            f"{_format_vector(direction)} for '{obj_name}'.",
            color="cyan",
        )
        cfg = _build_pickup_cfg(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            pre_grasp_dis=pre_grasp_dis,
            kwargs=kwargs,
            approach_direction=direction,
        )
        try:
            action = _run_public_action(
                env,
                robot_name,
                cfg,
                target,
                log_name=log_name,
                strict=True,
            )
        except Exception as exc:
            message = str(exc)
            failures.append(f"{label}={_format_vector(direction)}: {message}")
            _record_public_grasp_attempt(
                env=env,
                kwargs=kwargs,
                obj_name=obj_name,
                label=label,
                direction=direction,
                status="failed",
                message=message,
            )
            log_warning(
                f"Public semantic grasp direction {label} failed for "
                f"'{obj_name}': {message}"
            )
            continue

        if action is not None:
            _record_public_grasp_attempt(
                env=env,
                kwargs=kwargs,
                obj_name=obj_name,
                label=label,
                direction=direction,
                status="selected",
                message="planned",
            )
            _register_public_grasp_physical_validation(
                env=env,
                kwargs=kwargs,
                robot_name=robot_name,
                obj_name=obj_name,
                label=label,
                direction=direction,
                lift_height=float(cfg.lift_height),
                legacy_reference_pose=legacy_reference_pose,
            )
            log_info(
                f"Selected public semantic grasp direction {label}="
                f"{_format_vector(direction)} for '{obj_name}'.",
                color="green",
            )
            return action

    failure_text = "; ".join(failures) if failures else "no directions were attempted"
    message = (
        f"Public semantic grasp failed for '{obj_name}' across "
        f"{len(directions)} approach directions: {failure_text}."
    )
    _handle_public_grasp_unavailable(message, strict=strict)
    return None


def try_public_grasp_action(
    *,
    env,
    robot_name: str,
    obj_name: str,
    pre_grasp_dis: float,
    kwargs: dict[str, Any],
):
    strict = _public_grasp_strict(kwargs)
    if (
        env is None
        or not _public_atomic_actions_enabled(kwargs)
        or not _public_grasp_requested(kwargs)
    ):
        return None
    kwargs = _with_public_grasp_strategy_defaults(kwargs)

    if _public_grasp_semantics_requested(kwargs):
        target = _build_public_grasp_semantics(
            env,
            obj_name,
            kwargs,
            strict=strict,
        )
        if target is None:
            return None
        return _run_public_semantic_grasp_action(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            pre_grasp_dis=pre_grasp_dis,
            kwargs=kwargs,
            target=target,
            strict=strict,
        )
    else:
        target = _build_legacy_grasp_pose(
            env,
            robot_name,
            obj_name,
            preserve_object_rotation=bool(
                kwargs.get("public_grasp_preserve_object_rotation", False)
            ),
        )
        log_name = f"grasp:{obj_name}"
        if target is None:
            _handle_public_grasp_unavailable(
                f"Public grasp cannot build a grasp pose for '{obj_name}' from obj_info.",
                strict=strict,
            )
            return None
    if target is None:
        return None

    device = getattr(env.robot, "device", None)
    approach_direction = _public_grasp_approach_direction(
        env,
        robot_name,
        obj_name,
        kwargs,
        device=device,
    )
    cfg = _build_pickup_cfg(
        env=env,
        robot_name=robot_name,
        obj_name=obj_name,
        pre_grasp_dis=pre_grasp_dis,
        kwargs=kwargs,
        approach_direction=approach_direction,
    )
    action = _run_public_action(
        env,
        robot_name,
        cfg,
        target,
        log_name=log_name,
        strict=strict,
    )
    if action is not None:
        _store_public_grasp_relation(
            env=env,
            robot_name=robot_name,
            obj_name=obj_name,
            grasp_pose=target,
            source="legacy_grasp_pose",
        )
        legacy_reference_pose = (
            _build_legacy_grasp_pose(env, robot_name, obj_name)
            if _public_grasp_legacy_reference_required(kwargs)
            else None
        )
        _register_public_grasp_physical_validation(
            env=env,
            kwargs=kwargs,
            robot_name=robot_name,
            obj_name=obj_name,
            label="legacy_pose",
            direction=approach_direction,
            lift_height=float(cfg.lift_height),
            legacy_reference_pose=legacy_reference_pose,
        )
    return action


def try_public_place_action(
    *,
    env,
    robot_name: str,
    target_pose,
    pre_place_dis: float,
    kwargs: dict[str, Any],
):
    strict = _public_non_grasp_strict(kwargs)
    if env is None:
        _handle_public_non_grasp_unavailable(
            "Public place action cannot run: env is None.",
            strict=strict,
            fallback_name="place action",
        )
        return None
    if target_pose is None:
        _handle_public_non_grasp_unavailable(
            "Public place action cannot run: target_pose is None.",
            strict=strict,
            fallback_name="place action",
        )
        return None
    if not _public_atomic_actions_enabled(kwargs):
        _handle_public_non_grasp_unavailable(
            "Public place action is disabled.",
            strict=strict,
            fallback_name="place action",
        )
        return None
    if not kwargs.get("use_public_place_action", False):
        _handle_public_non_grasp_unavailable(
            "Public place action is disabled.",
            strict=strict,
            fallback_name="place action",
        )
        return None

    is_left, arm_part, hand_part = _select_arm(robot_name)
    hand_joints = env.left_eef_joints if is_left else env.right_eef_joints
    hand_dof = len(hand_joints)
    device = getattr(env.robot, "device", None)
    cfg = PlaceActionCfg(
        control_part=arm_part,
        hand_control_part=hand_part,
        hand_open_qpos=_state_to_hand_qpos(env.open_state, hand_dof, device=device),
        hand_close_qpos=_state_to_hand_qpos(env.close_state, hand_dof, device=device),
        lift_height=pre_place_dis,
        sample_interval=kwargs.get("sample_interval", kwargs.get("sample_num", 45)),
        hand_interp_steps=kwargs.get("hand_interp_steps", 15),
        post_open_wait_steps=kwargs.get("public_place_post_open_wait_steps", 20),
    )
    return _run_public_action(
        env,
        robot_name,
        cfg,
        target_pose,
        log_name="place",
        strict=strict,
    )
