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

"""Resolve object, end-effector, and joint targets plus orientation policies.

Pose math is kept separate from action dispatch so surface and orientation
rules can be tested without running a motion planner.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
)
from embodichain.gen_sim.action_agent_pipeline.domain.orientation_policy import (
    _has_bottle_like_keyword as _policy_has_bottle_like_keyword,
    _is_bottle_like as _policy_is_bottle_like,
    _is_normalized_local_z_label,
    _preview_preserving_upright_rotation as _policy_preview_upright_rotation,
    _rotation_distance_score as _policy_rotation_distance_score,
    _rotation_from_axis_targets as _policy_rotation_from_axis_targets,
    principal_local_axis_order,
    resolve_target_rotation,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_parts import (
    _select_arm_parts,
    _state_to_hand_qpos,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atom_action_utils import (
    get_arm_states,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
    SURFACE_Z_POLICY_FIELDS,
    _surface_support_uid,
    _xyz,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.pose_utils import (
    _ensure_batched_pose_tensor,
    _normalize_vector,
    _object_world_vertices,
)
from embodichain.lab.sim.atomic_actions import (
    HeldObjectState,
    ObjectSemantics,
    WorldState,
)
from embodichain.utils.math import get_offset_pose, pose_inv

_DEFAULT_SURFACE_RELEASE_CLEARANCE = DEFAULT_SURFACE_RELEASE_CLEARANCE

__all__ = [
    "_resolve_coordinated_object_pose_target",
    "_resolve_object_target_pose_like",
    "_semantics_as_held_object_state",
    "_resolve_pose_target",
    "_resolve_held_object_pose_target",
    "_held_object_current_pose",
    "_resolve_object_orientation",
    "_apply_surface_z_policy",
    "_surface_release_clearance",
    "_surface_support_top_z",
    "_target_local_zmin_after_rotation",
    "_held_object_mesh_vertices",
    "_principal_local_axes",
    "_is_bottle_like_held_object",
    "_held_object_local_z_is_upright_semantic",
    "_has_bottle_like_keyword",
    "_semantic_local_z_upright_rotation",
    "_preview_aware_upright_rotation",
    "_rotation_distance_score",
    "_axis_align_target_direction",
    "_axis_align_current_direction",
    "_reference_object_axis_direction",
    "_yaw_aligned_rotation",
    "_signed_yaw_delta",
    "_yaw_rotation_matrix",
    "_rotation_from_axis_targets",
    "_resolve_qpos_target",
    "_resolve_object_pose_target",
    "_resolve_absolute_pose_target",
    "_resolve_relative_pose_target",
    "_resolve_initial_qpos_target",
    "_resolve_gripper_qpos_target",
    "_resolve_joint_delta_qpos_target",
]


def _resolve_coordinated_object_pose_target(
    env,
    spec: AtomicActionSpec,
    semantics: ObjectSemantics,
    state: WorldState | None,
) -> torch.Tensor:
    target_pose_spec = spec.target_object_pose
    current_object_pose = _ensure_batched_pose_tensor(
        semantics.entity.get_local_pose(to_matrix=True),
        env.robot.device,
    )
    target_pose = _resolve_object_target_pose_like(
        env,
        target_pose_spec,
        current_object_pose,
    )
    orientation_state = state or WorldState(last_qpos=env.robot.get_qpos().clone())
    if orientation_state.held_object is None:
        orientation_state = WorldState(
            last_qpos=orientation_state.last_qpos,
            held_object=_semantics_as_held_object_state(
                semantics,
                current_object_pose,
                env.robot.device,
            ),
            coordinated_held_object=orientation_state.coordinated_held_object,
        )
    target_pose[..., :3, :3] = _resolve_object_orientation(
        env,
        target_pose_spec,
        current_object_pose,
        orientation_state,
    )
    target_pose = _apply_surface_z_policy(
        env,
        target_pose_spec,
        target_pose,
        orientation_state,
    )
    return target_pose


def _resolve_object_target_pose_like(
    env,
    target_pose_spec: Mapping[str, Any],
    current_object_pose: torch.Tensor,
) -> torch.Tensor:
    reference = target_pose_spec["reference"]
    target_pose = current_object_pose.clone()
    is_batched = target_pose.ndim == 3
    if reference == "absolute":
        position = target_pose_spec.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("absolute target_object_pose requires position.")
        for index, value in enumerate(position):
            if value is not None:
                if is_batched:
                    target_pose[:, index, 3] = float(value)
                else:
                    target_pose[index, 3] = float(value)
        return target_pose
    if reference == "object":
        obj_name = target_pose_spec.get("obj_name")
        target_obj = env.sim.get_rigid_object(obj_name)
        if target_obj is None:
            raise ValueError(f"No rigid object found for {obj_name}.")
        target_pose = _ensure_batched_pose_tensor(
            target_obj.get_local_pose(to_matrix=True),
            env.robot.device,
        )
        offset = _xyz(target_pose_spec.get("offset", [0.0, 0.0, 0.0]), "offset")
        target_pose[..., :3, 3] += torch.tensor(
            offset,
            dtype=torch.float32,
            device=env.robot.device,
        )
        return target_pose
    if reference == "relative":
        offset = _xyz(target_pose_spec.get("offset", [0.0, 0.0, 0.0]), "offset")
        frame = target_pose_spec.get("frame", "world")
        mode = "extrinsic" if frame == "world" else "intrinsic"

        def _apply_offsets(pose):
            result = pose.clone()
            for offset_value, direction in zip(offset, ("x", "y", "z")):
                result = get_offset_pose(result, offset_value, direction, mode)
            return result

        if is_batched:
            target_pose = torch.stack([_apply_offsets(pose) for pose in target_pose])
        else:
            target_pose = _apply_offsets(target_pose)
        return torch.as_tensor(
            target_pose,
            dtype=torch.float32,
            device=env.robot.device,
        )
    raise ValueError(f"Unsupported target_object_pose reference: {reference}.")


def _semantics_as_held_object_state(
    semantics: ObjectSemantics,
    object_pose: torch.Tensor,
    device,
):
    object_pose = _ensure_batched_pose_tensor(object_pose, device)
    n_envs = object_pose.shape[0]
    identity = (
        torch.eye(4, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .repeat(n_envs, 1, 1)
    )
    return HeldObjectState(
        semantics=semantics,
        object_to_eef=identity,
        grasp_xpos=object_pose,
    )


def _resolve_pose_target(env, spec: AtomicActionSpec):
    reference = spec.target_pose["reference"]
    if reference == "object":
        return _resolve_object_pose_target(env, spec)
    if reference == "absolute":
        return _resolve_absolute_pose_target(env, spec)
    if reference == "relative":
        return _resolve_relative_pose_target(env, spec)
    raise ValueError(f"Unsupported target_pose reference: {reference}.")


def _resolve_held_object_pose_target(
    env,
    spec: AtomicActionSpec,
    state: WorldState,
) -> torch.Tensor:
    target_pose_spec = spec.target_object_pose
    pose_metadata_fields = {
        "orientation_goal",
        "orientation_axis",
        "align_to",
    } | SURFACE_Z_POLICY_FIELDS
    pose_spec = AtomicActionSpec(
        atomic_action_class="MoveEndEffector",
        robot_name=spec.robot_name,
        control="arm",
        target_pose={
            key: deepcopy(value)
            for key, value in target_pose_spec.items()
            if key not in pose_metadata_fields
        },
        cfg={},
    )
    target_pose = _resolve_pose_target(env, pose_spec)
    current_object_pose = _held_object_current_pose(state, env.robot.device)
    num_envs = current_object_pose.shape[0]
    if target_pose.ndim == 2:
        target_pose = target_pose.unsqueeze(0).repeat(num_envs, 1, 1)
    target_pose[..., :3, :3] = _resolve_object_orientation(
        env,
        target_pose_spec,
        current_object_pose,
        state,
    )
    target_pose = _apply_surface_z_policy(env, target_pose_spec, target_pose, state)
    return target_pose


def _held_object_current_pose(state: WorldState, device) -> torch.Tensor:
    held = state.held_object
    if held is None:
        raise ValueError("Held object state is required.")
    entity = held.semantics.entity
    if entity is not None and hasattr(entity, "get_local_pose"):
        pose = entity.get_local_pose(to_matrix=True)
        return _ensure_batched_pose_tensor(pose, device)
    return held.grasp_xpos.to(device=device, dtype=torch.float32)


def _resolve_object_orientation(
    env,
    target_pose_spec: Mapping[str, Any],
    current_object_pose: torch.Tensor,
    state: WorldState,
) -> torch.Tensor:
    orientation_goal = target_pose_spec.get("orientation_goal", "preserve")
    current_rotation = current_object_pose[..., :3, :3].clone()
    if orientation_goal == "preserve":
        held = state.held_object
        if held is None:
            return current_rotation
        pickup_object_pose = torch.matmul(
            held.grasp_xpos.to(device=env.robot.device, dtype=torch.float32),
            pose_inv(
                held.object_to_eef.to(
                    device=env.robot.device,
                    dtype=torch.float32,
                )
            ),
        )
        return pickup_object_pose[..., :3, :3]
    # Non-preserve orientation goals are computed from a single representative env
    # and broadcast to all envs.
    if current_rotation.ndim == 3:
        current_rotation = current_rotation[0]

    mesh_vertices = _held_object_mesh_vertices(state, env.robot.device)
    if orientation_goal in {"upright", "lay_flat"}:
        rotation = resolve_target_rotation(
            orientation_goal=orientation_goal,
            local_bounds=_tensor_local_bounds(mesh_vertices),
            current_rotation=_tensor_matrix3(current_rotation),
            object_label=_held_object_label(state),
        )
        return torch.tensor(
            rotation,
            dtype=current_rotation.dtype,
            device=current_rotation.device,
        )

    local_axes = _principal_local_axes(mesh_vertices)
    if orientation_goal == "axis_align":
        target_direction = _axis_align_target_direction(
            env,
            target_pose_spec,
            env.robot.device,
        )
        current_direction = _axis_align_current_direction(
            current_rotation,
            local_axes,
            env.robot.device,
        )
        if current_direction is None:
            return current_rotation
        return _yaw_aligned_rotation(
            current_rotation, current_direction, target_direction
        )
    raise ValueError(f"Unsupported orientation_goal: {orientation_goal}.")


def _apply_surface_z_policy(
    env,
    target_pose_spec: Mapping[str, Any],
    target_pose: torch.Tensor,
    state: WorldState,
) -> torch.Tensor:
    policy = target_pose_spec.get("z_policy", "preserve")
    if policy == "preserve":
        return target_pose
    if policy not in {"object_on_surface", "surface_release"}:
        raise ValueError(f"Unsupported target_object_pose z_policy: {policy!r}.")
    support_uid = _surface_support_uid(
        target_pose_spec,
        target_name="target_object_pose",
        require=True,
    )
    support_top_z = _surface_support_top_z(env, support_uid, env.robot.device)
    mesh_vertices = _held_object_mesh_vertices(state, env.robot.device)
    target_local_zmin = _target_local_zmin_after_rotation(
        mesh_vertices,
        target_pose[..., :3, :3],
    )
    resolved_pose = target_pose.clone()
    resolved_pose[..., 2, 3] = (
        float(support_top_z)
        + _surface_release_clearance(target_pose_spec)
        - target_local_zmin
    )
    return resolved_pose


def _surface_release_clearance(target_pose_spec: Mapping[str, Any]) -> float:
    clearance = target_pose_spec.get(
        "surface_clearance",
        _DEFAULT_SURFACE_RELEASE_CLEARANCE,
    )
    return float(clearance)


def _surface_support_top_z(env, support_uid: str, device) -> float:
    support_obj = env.sim.get_rigid_object(support_uid)
    if support_obj is None:
        raise ValueError(f"No support object found for {support_uid}.")
    world_vertices = _object_world_vertices(support_obj, device)
    return float(world_vertices[:, 2].max())


def _target_local_zmin_after_rotation(
    mesh_vertices: torch.Tensor,
    target_rotation: torch.Tensor,
) -> torch.Tensor:
    if target_rotation.ndim == 2:
        rotated_vertices = (target_rotation @ mesh_vertices.T).T
        return torch.as_tensor(
            rotated_vertices[:, 2].min(),
            dtype=torch.float32,
            device=target_rotation.device,
        )
    rotated_vertices = torch.matmul(
        target_rotation,
        mesh_vertices.T.unsqueeze(0).expand(target_rotation.shape[0], -1, -1),
    )
    return rotated_vertices[..., 2, :].min(dim=-1).values


def _held_object_mesh_vertices(state: WorldState, device) -> torch.Tensor:
    held = state.held_object
    if held is None:
        raise ValueError("Held object state is required.")
    vertices = held.semantics.geometry.get("mesh_vertices")
    if vertices is None and held.semantics.entity is not None:
        vertices = held.semantics.entity.get_vertices(env_ids=[0], scale=True)[0]
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("Held object mesh_vertices must have shape (N, 3).")
    return vertices


def _held_object_label(state: WorldState) -> str:
    held = state.held_object
    if held is None:
        return ""
    return str(getattr(held.semantics, "label", ""))


def _tensor_local_bounds(
    vertices: torch.Tensor,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # Only six scalar bounds cross the device boundary; the full mesh stays on
    # its Runtime device for the final exact z-min calculation.
    mins = _tensor_vector3(vertices.min(dim=0).values)
    maxs = _tensor_vector3(vertices.max(dim=0).values)
    return mins, maxs


def _tensor_vector3(vector: torch.Tensor) -> tuple[float, float, float]:
    values = vector.detach().cpu().tolist()
    return tuple(float(value) for value in values)


def _tensor_matrix3(
    matrix: torch.Tensor,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    values = matrix.detach().cpu().tolist()
    return tuple(tuple(float(value) for value in row) for row in values)


def _matrix3_tensor(
    matrix,
    reference: torch.Tensor,
) -> torch.Tensor:
    return torch.tensor(matrix, dtype=reference.dtype, device=reference.device)


def _principal_local_axes(vertices: torch.Tensor) -> torch.Tensor:
    order = principal_local_axis_order(_tensor_local_bounds(vertices))
    order_tensor = torch.tensor(order, dtype=torch.long, device=vertices.device)
    axes = torch.eye(3, dtype=torch.float32, device=vertices.device)[:, order_tensor]
    return axes


def _is_bottle_like_held_object(state: WorldState, vertices: torch.Tensor) -> bool:
    return _policy_is_bottle_like(
        _held_object_label(state),
        _tensor_local_bounds(vertices),
    )


def _held_object_local_z_is_upright_semantic(state: WorldState) -> bool:
    return _is_normalized_local_z_label(_held_object_label(state))


def _has_bottle_like_keyword(text: str) -> bool:
    return _policy_has_bottle_like_keyword(text)


def _semantic_local_z_upright_rotation(current_rotation: torch.Tensor) -> torch.Tensor:
    rotation = _policy_preview_upright_rotation(
        primary_axis=(0.0, 0.0, 1.0),
        secondary_axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        current_rotation=_tensor_matrix3(current_rotation),
    )
    return _matrix3_tensor(rotation, current_rotation)


def _preview_aware_upright_rotation(
    *,
    local_axes: torch.Tensor,
    current_rotation: torch.Tensor,
) -> torch.Tensor:
    rotation = _policy_preview_upright_rotation(
        primary_axis=_tensor_vector3(local_axes[:, 0]),
        secondary_axes=tuple(
            _tensor_vector3(local_axes[:, index])
            for index in range(1, local_axes.shape[1])
        ),
        current_rotation=_tensor_matrix3(current_rotation),
    )
    return _matrix3_tensor(rotation, current_rotation)


def _rotation_distance_score(
    rotation: torch.Tensor,
    reference_rotation: torch.Tensor,
) -> float:
    return _policy_rotation_distance_score(
        _tensor_matrix3(rotation),
        _tensor_matrix3(reference_rotation),
    )


def _axis_align_target_direction(
    env,
    target_pose_spec: Mapping[str, Any],
    device,
) -> torch.Tensor:
    orientation_axis = target_pose_spec.get("orientation_axis", "none")
    align_to = target_pose_spec.get("align_to")
    if align_to:
        return _reference_object_axis_direction(env, align_to, orientation_axis, device)
    if orientation_axis == "x":
        return torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    if orientation_axis == "y":
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    raise ValueError(
        "axis_align without align_to requires orientation_axis 'x' or 'y'."
    )


def _axis_align_current_direction(
    current_rotation: torch.Tensor,
    local_axes: torch.Tensor,
    device,
) -> torch.Tensor | None:
    horizontal_epsilon = 1e-4
    long_axis = local_axes[:, 0].to(device=device, dtype=torch.float32)
    long_direction = current_rotation @ long_axis
    long_horizontal = long_direction.clone()
    long_horizontal[2] = 0.0
    if float(torch.linalg.norm(long_horizontal)) >= horizontal_epsilon:
        return long_direction

    candidates: list[tuple[float, torch.Tensor]] = []
    for index in range(local_axes.shape[1]):
        local_axis = local_axes[:, index].to(device=device, dtype=torch.float32)
        direction = current_rotation @ local_axis
        horizontal = direction.clone()
        horizontal[2] = 0.0
        candidates.append((float(torch.linalg.norm(horizontal)), direction))
    score, direction = max(candidates, key=lambda item: item[0])
    if score < horizontal_epsilon:
        return None
    return direction


def _reference_object_axis_direction(
    env,
    align_to: str,
    orientation_axis: str,
    device,
) -> torch.Tensor:
    if orientation_axis not in {"long_axis", "short_axis"}:
        raise ValueError(
            "Reference-object axis alignment requires orientation_axis "
            "'long_axis' or 'short_axis'."
        )
    target_obj = env.sim.get_rigid_object(align_to)
    if target_obj is None:
        raise ValueError(f"No rigid object found for align_to={align_to}.")
    vertices = torch.as_tensor(
        target_obj.get_vertices(env_ids=[0], scale=True)[0],
        dtype=torch.float32,
        device=device,
    )
    extents = vertices.max(dim=0).values - vertices.min(dim=0).values
    axis_index = 0 if extents[0] >= extents[1] else 1
    if orientation_axis == "short_axis":
        axis_index = 1 - axis_index
    pose = _ensure_batched_pose_tensor(
        target_obj.get_local_pose(to_matrix=True), device
    )
    direction = pose[0, :3, axis_index].clone()
    direction[2] = 0.0
    norm = torch.linalg.norm(direction)
    if float(norm) < 1e-6:
        raise ValueError(f"Reference object {align_to!r} has no valid XY axis.")
    return direction / norm


def _yaw_aligned_rotation(
    current_rotation: torch.Tensor,
    current_direction: torch.Tensor,
    target_direction: torch.Tensor,
) -> torch.Tensor:
    device = current_rotation.device
    current_xy = current_direction.to(device=device, dtype=torch.float32).clone()
    target_xy = target_direction.to(device=device, dtype=torch.float32).clone()
    current_xy[2] = 0.0
    target_xy[2] = 0.0
    current_xy = _normalize_vector(current_xy)
    target_xy = _normalize_vector(target_xy)
    same_delta = _signed_yaw_delta(current_xy, target_xy)
    opposite_delta = _signed_yaw_delta(current_xy, -target_xy)
    delta = (
        same_delta
        if torch.abs(same_delta) <= torch.abs(opposite_delta)
        else opposite_delta
    )
    return _yaw_rotation_matrix(delta, device) @ current_rotation


def _signed_yaw_delta(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cross_z = source[0] * target[1] - source[1] * target[0]
    dot = source[0] * target[0] + source[1] * target[1]
    return torch.atan2(cross_z, dot)


def _yaw_rotation_matrix(delta: torch.Tensor, device) -> torch.Tensor:
    c = torch.cos(delta)
    s = torch.sin(delta)
    rotation = torch.eye(3, dtype=torch.float32, device=device)
    rotation[0, 0] = c
    rotation[0, 1] = -s
    rotation[1, 0] = s
    rotation[1, 1] = c
    return rotation


def _rotation_from_axis_targets(
    *,
    local_primary: torch.Tensor,
    world_primary: torch.Tensor,
    local_secondary: torch.Tensor,
    world_secondary: torch.Tensor,
) -> torch.Tensor:
    rotation = _policy_rotation_from_axis_targets(
        local_primary=_tensor_vector3(local_primary),
        world_primary=_tensor_vector3(world_primary),
        local_secondary=_tensor_vector3(local_secondary),
        world_secondary=_tensor_vector3(world_secondary),
    )
    return _matrix3_tensor(rotation, world_primary)


def _resolve_qpos_target(env, spec: AtomicActionSpec):
    source = spec.target_qpos["source"]
    if source == "initial":
        return _resolve_initial_qpos_target(env, spec)
    if source == "gripper_state":
        return _resolve_gripper_qpos_target(env, spec)
    if source == "joint_delta":
        return _resolve_joint_delta_qpos_target(env, spec)
    raise ValueError(f"Unsupported target_qpos source: {source}.")


def _resolve_object_pose_target(env, spec: AtomicActionSpec):
    obj_name = spec.target_pose.get("obj_name")
    target_obj = env.sim.get_rigid_object(obj_name)
    if target_obj is None:
        raise ValueError(f"No rigid object found for {obj_name}.")
    offset = _xyz(spec.target_pose.get("offset", [0.0, 0.0, 0.0]), "offset")
    _, _, _, current_pose, _ = get_arm_states(env, spec.robot_name)
    target_pose = deepcopy(current_pose)
    target_obj_pose = target_obj.get_local_pose(to_matrix=True)
    if target_pose.ndim == 2:
        target_pose[:3, 3] = target_obj_pose[:3, 3]
        target_pose[0, 3] += offset[0]
        target_pose[1, 3] += offset[1]
        target_pose[2, 3] += offset[2]
    else:
        target_pose[:, :3, 3] = target_obj_pose[:, :3, 3]
        target_pose[:, 0, 3] += offset[0]
        target_pose[:, 1, 3] += offset[1]
        target_pose[:, 2, 3] += offset[2]
    return torch.as_tensor(target_pose, dtype=torch.float32, device=env.robot.device)


def _resolve_absolute_pose_target(env, spec: AtomicActionSpec):
    position = spec.target_pose.get("position")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("absolute target_pose requires position with three entries.")
    _, _, _, current_pose, _ = get_arm_states(env, spec.robot_name)
    target_pose = deepcopy(current_pose)
    if target_pose.ndim == 2:
        for index, value in enumerate(position):
            if value is not None:
                target_pose[index, 3] = float(value)
    else:
        for index, value in enumerate(position):
            if value is not None:
                target_pose[:, index, 3] = float(value)
    return torch.as_tensor(target_pose, dtype=torch.float32, device=env.robot.device)


def _resolve_relative_pose_target(env, spec: AtomicActionSpec):
    offset = _xyz(spec.target_pose.get("offset", [0.0, 0.0, 0.0]), "offset")
    frame = spec.target_pose.get("frame", "world")
    if frame not in {"world", "eef"}:
        raise ValueError("relative target_pose frame must be 'world' or 'eef'.")
    mode = "extrinsic" if frame == "world" else "intrinsic"
    _, _, _, current_pose, _ = get_arm_states(env, spec.robot_name)
    current_pose = torch.as_tensor(
        current_pose, dtype=torch.float32, device=env.robot.device
    )

    def _apply_offsets(pose):
        target_pose = pose.clone()
        for offset_value, direction in zip(offset, ("x", "y", "z")):
            target_pose = get_offset_pose(target_pose, offset_value, direction, mode)
        return target_pose

    if current_pose.ndim == 2:
        target_pose = _apply_offsets(current_pose)
    else:
        target_pose = torch.stack([_apply_offsets(pose) for pose in current_pose])
    return target_pose


def _resolve_initial_qpos_target(env, spec: AtomicActionSpec):
    if spec.control != "arm":
        raise ValueError("initial target_qpos requires control='arm'.")
    is_left, _, _, _, _ = _select_arm_parts(env, spec.robot_name)
    target_qpos = env.left_arm_init_qpos if is_left else env.right_arm_init_qpos
    return torch.as_tensor(target_qpos, dtype=torch.float32, device=env.robot.device)


def _resolve_gripper_qpos_target(env, spec: AtomicActionSpec):
    if spec.control != "hand":
        raise ValueError("gripper_state target_qpos requires control='hand'.")
    state = spec.target_qpos.get("state")
    if state == "open":
        source = env.open_state
    elif state == "close":
        source = env.close_state
    else:
        raise ValueError("gripper_state target_qpos state must be 'open' or 'close'.")
    _, _, _, _, eef_joints = _select_arm_parts(env, spec.robot_name)
    return _state_to_hand_qpos(source, len(eef_joints), env.robot.device)


def _resolve_joint_delta_qpos_target(env, spec: AtomicActionSpec):
    if spec.control != "arm":
        raise ValueError("joint_delta target_qpos requires control='arm'.")
    joint_index = int(spec.target_qpos["joint_index"])
    delta_degrees = float(spec.target_qpos.get("delta_degrees", 0.0))
    _, _, current_qpos, _, _ = get_arm_states(env, spec.robot_name)
    target_qpos = torch.as_tensor(
        current_qpos,
        dtype=torch.float32,
        device=env.robot.device,
    ).clone()
    if target_qpos.ndim == 1:
        if joint_index < 0 or joint_index >= target_qpos.numel():
            raise ValueError(f"joint_index {joint_index} is out of range.")
        target_qpos[joint_index] += float(np.deg2rad(delta_degrees))
    else:
        if joint_index < 0 or joint_index >= target_qpos.shape[-1]:
            raise ValueError(f"joint_index {joint_index} is out of range.")
        target_qpos[:, joint_index] += float(np.deg2rad(delta_degrees))
    return target_qpos
