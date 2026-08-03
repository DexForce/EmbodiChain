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

"""Dispatch normalized action specs into typed runtime targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.atomic_action_spec import (
    AtomicActionSpec,
    _COORDINATED_WORLD_Y_ANGLE_CFG_KEY,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.action_parts import (
    _select_arm_parts,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.coordinated_grasp import (
    _default_coordinated_object_to_eef,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.coordinated_payload import (
    _record_coordinated_payload_runtime_state,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.grasp_support import (
    _build_object_semantics,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.object_pose import (
    _apply_surface_z_policy,
    _resolve_coordinated_object_pose_target,
    _resolve_held_object_pose_target,
    _resolve_object_orientation,
    _resolve_object_target_pose_like,
    _resolve_pose_target,
    _resolve_qpos_target,
    _semantics_as_held_object_state,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.pose_utils import (
    _ensure_batched_pose_tensor,
)
from embodichain.lab.sim.atomic_actions import (
    CoordinatedPickmentTarget,
    ObjectSemantics,
    WorldState,
)

_DEFAULT_PICKUP_APPROACH_ALIGNMENT_MAX_ANGLE = float(torch.pi / 36)

__all__ = [
    "_resolve_target",
    "_resolve_pickup_target",
    "_pickup_approach_alignment_angle",
    "_resolve_pickup_downstream_object_targets",
    "_resolve_move_end_effector_target",
    "_resolve_move_joints_target",
    "_resolve_move_held_object_target",
    "_resolve_place_target",
    "_resolve_coordinated_pickment_target",
    "_target_summary",
    "_target_object_pose_summary",
]


def _resolve_target(
    env,
    spec: AtomicActionSpec,
    runtime_kwargs: dict[str, Any],
    *,
    state: WorldState | None,
):
    if spec.atomic_action_class == "PickUp":
        return _resolve_pickup_target(env, spec, runtime_kwargs)
    if spec.atomic_action_class == "MoveEndEffector":
        return _resolve_move_end_effector_target(env, spec)
    if spec.atomic_action_class == "MoveJoints":
        return _resolve_move_joints_target(env, spec)
    if spec.atomic_action_class == "MoveHeldObject":
        return _resolve_move_held_object_target(env, spec, state)
    if spec.atomic_action_class == "Place":
        return _resolve_place_target(env, spec, state)
    if spec.atomic_action_class == "CoordinatedPickment":
        return _resolve_coordinated_pickment_target(env, spec, runtime_kwargs, state)
    raise ValueError(f"Unsupported atomic action class: {spec.atomic_action_class}.")


def _resolve_pickup_target(
    env,
    spec: AtomicActionSpec,
    runtime_kwargs: dict[str, Any],
):
    if not spec.target_object:
        raise ValueError("PickUp requires target_object.")
    return _build_object_semantics(
        env,
        spec.target_object,
        runtime_kwargs,
        max_approach_alignment_angle=_pickup_approach_alignment_angle(spec.cfg),
    )


def _pickup_approach_alignment_angle(cfg: Mapping[str, Any]) -> float | None:
    if cfg.get("rotate_upright") is not None:
        return None
    value = cfg.get(
        "approach_alignment_max_angle",
        _DEFAULT_PICKUP_APPROACH_ALIGNMENT_MAX_ANGLE,
    )
    return None if value is None else float(value)


def _resolve_pickup_downstream_object_targets(
    env,
    spec: AtomicActionSpec,
    semantics: ObjectSemantics,
    runtime_kwargs: Mapping[str, Any],
) -> tuple[torch.Tensor, ...]:
    """Resolve graph-provided held-object targets before selecting a grasp pose."""
    target_specs_by_robot = runtime_kwargs.get(
        "pickup_downstream_object_target_specs", {}
    )
    if not isinstance(target_specs_by_robot, Mapping):
        return ()
    target_specs = target_specs_by_robot.get(spec.robot_name, ())
    if not isinstance(target_specs, Sequence):
        return ()

    object_pose = _ensure_batched_pose_tensor(
        semantics.entity.get_local_pose(to_matrix=True), env.robot.device
    )
    _, arm_part, _, _, _ = _select_arm_parts(env, spec.robot_name)
    state = WorldState(
        last_qpos=env.robot.get_qpos().clone(),
        held_objects={
            arm_part: _semantics_as_held_object_state(
                semantics, object_pose, env.robot.device
            )
        },
    )
    targets: list[torch.Tensor] = []
    for target_spec in target_specs:
        if not isinstance(target_spec, Mapping):
            continue
        target_pose = _resolve_object_target_pose_like(env, target_spec, object_pose)
        target_pose[..., :3, :3] = _resolve_object_orientation(
            env, target_spec, object_pose, state
        )
        targets.append(_apply_surface_z_policy(env, target_spec, target_pose, state))
    return tuple(targets)


def _resolve_move_end_effector_target(env, spec: AtomicActionSpec):
    if not spec.target_pose:
        raise ValueError("MoveEndEffector requires target_pose.")
    return _resolve_pose_target(env, spec)


def _resolve_move_joints_target(env, spec: AtomicActionSpec):
    if not spec.target_qpos:
        raise ValueError("MoveJoints requires target_qpos.")
    return _resolve_qpos_target(env, spec)


def _resolve_move_held_object_target(
    env,
    spec: AtomicActionSpec,
    state: WorldState | None,
):
    if not spec.target_object_pose:
        raise ValueError("MoveHeldObject requires target_object_pose.")
    _, arm_part, _, _, _ = _select_arm_parts(env, spec.robot_name)
    if state is None or state.get_held_object(arm_part) is None:
        raise ValueError("MoveHeldObject requires a held object from a prior PickUp.")
    return _resolve_held_object_pose_target(env, spec, state)


def _resolve_place_target(
    env,
    spec: AtomicActionSpec,
    state: WorldState | None,
) -> torch.Tensor:
    if spec.target_pose:
        return _resolve_pose_target(env, spec)
    if not spec.target_object_pose:
        raise ValueError("Place requires target_pose or target_object_pose.")
    _, arm_part, _, _, _ = _select_arm_parts(env, spec.robot_name)
    held_object = None if state is None else state.get_held_object(arm_part)
    if held_object is None:
        raise ValueError(
            "Place with target_object_pose requires a held object from a prior PickUp."
        )

    object_target_pose = _resolve_held_object_pose_target(env, spec, state)
    object_to_eef = held_object.object_to_eef.to(
        device=env.robot.device,
        dtype=torch.float32,
    )
    if object_to_eef.shape == (4, 4):
        object_to_eef = object_to_eef.unsqueeze(0).repeat(
            object_target_pose.shape[0], 1, 1
        )
    return torch.bmm(object_target_pose, object_to_eef)


def _resolve_coordinated_pickment_target(
    env,
    spec: AtomicActionSpec,
    runtime_kwargs: dict[str, Any],
    state: WorldState | None,
) -> CoordinatedPickmentTarget:
    if not spec.target_object:
        raise ValueError("CoordinatedPickment requires target_object.")
    if not spec.target_object_pose:
        raise ValueError("CoordinatedPickment requires target_object_pose.")
    semantics = _build_object_semantics(env, spec.target_object, runtime_kwargs)
    object_target_pose = _resolve_coordinated_object_pose_target(
        env,
        spec,
        semantics,
        state,
    )
    object_initial_pose = _ensure_batched_pose_tensor(
        semantics.entity.get_local_pose(to_matrix=True),
        env.robot.device,
    )
    num_envs = object_initial_pose.shape[0]
    raw_world_y_angle_limit = spec.cfg.get(_COORDINATED_WORLD_Y_ANGLE_CFG_KEY)
    world_y_angle_limit = (
        None if raw_world_y_angle_limit is None else float(raw_world_y_angle_limit)
    )
    left_object_to_eef, right_object_to_eef = _default_coordinated_object_to_eef(
        semantics,
        env.robot.device,
        object_initial_pose,
        object_label=semantics.label,
        object_target_pose=object_target_pose,
        pre_grasp_distance=float(spec.cfg.get("pre_grasp_distance", 0.10)),
        lift_height=float(spec.cfg.get("lift_height", 0.08)),
        sample_interval=int(spec.cfg.get("sample_interval", 120)),
        hand_interp_steps=int(spec.cfg.get("hand_interp_steps", 10)),
        hold_steps=int(spec.cfg.get("hold_steps", 4)),
        object_motion_keyframes=int(spec.cfg.get("object_motion_keyframes", 6)),
        max_grasp_separation_angle_to_world_y_degrees=world_y_angle_limit,
        payload_uids=tuple(spec.target_object.get("payloads", [])),
        env=env,
    )
    _record_coordinated_payload_runtime_state(
        env,
        spec,
        semantics,
        object_initial_pose,
    )
    if left_object_to_eef.ndim == 2:
        left_object_to_eef = left_object_to_eef.unsqueeze(0).repeat(num_envs, 1, 1)
    if right_object_to_eef.ndim == 2:
        right_object_to_eef = right_object_to_eef.unsqueeze(0).repeat(num_envs, 1, 1)
    return CoordinatedPickmentTarget(
        object_target_pose=object_target_pose,
        semantics=semantics,
        left_object_to_eef=left_object_to_eef,
        right_object_to_eef=right_object_to_eef,
        object_initial_pose=object_initial_pose,
    )


def _target_summary(spec: AtomicActionSpec) -> str:
    if spec.target_object:
        return f"target_object:{spec.target_object.get('obj_name')}"
    if spec.target_pose:
        return f"target_pose:{spec.target_pose.get('reference')}"
    if spec.target_qpos:
        return f"target_qpos:{spec.target_qpos.get('source')}"
    if spec.target_object_pose:
        return _target_object_pose_summary(spec.target_object_pose)
    return "target:none"


def _target_object_pose_summary(target_object_pose: Mapping[str, Any]) -> str:
    reference = target_object_pose.get("reference")
    parts = [f"target_object_pose:{reference}"]
    if reference == "absolute":
        parts.append(f"position={target_object_pose.get('position')}")
    elif reference == "object":
        parts.append(f"obj_name={target_object_pose.get('obj_name')}")
        parts.append(f"offset={target_object_pose.get('offset')}")
    elif reference == "relative":
        parts.append(f"offset={target_object_pose.get('offset')}")
        parts.append(f"frame={target_object_pose.get('frame', 'world')}")
    for key in (
        "orientation_goal",
        "orientation_axis",
        "align_to",
        "z_policy",
        "support",
    ):
        value = target_object_pose.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
