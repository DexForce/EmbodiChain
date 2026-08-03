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

"""Ground Seed v5 bindings from the current state of every environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.motion_policy import (
    resolve_motion_policy,
)

__all__ = [
    "ArmCandidateScore",
    "GroundedSymbolicAction",
    "ground_symbolic_action",
    "score_arm_candidate",
    "select_auto_arm_from_candidates",
]


@dataclass(frozen=True)
class GroundedSymbolicAction:
    """One atomic action spec plus the live values used to construct it."""

    action_spec: dict[str, Any]
    motion_policy: dict[str, Any]
    object_pose: torch.Tensor | None
    reference_pose: torch.Tensor | None
    target_object_pose: torch.Tensor | None


@dataclass(frozen=True)
class ArmCandidateScore:
    """Auditable cost components for one semantic arm candidate."""

    motion_cost: torch.Tensor
    normalized_motion_cost: torch.Tensor
    pickup_crossing_penalty: torch.Tensor
    placement_crossing_penalty: torch.Tensor
    total_cost: torch.Tensor


def score_arm_candidate(
    *,
    arm: str,
    motion_cost: torch.Tensor,
    source_pose: torch.Tensor | None,
    target_pose: torch.Tensor | None,
    workspace_center_y: torch.Tensor,
    workspace_half_width: torch.Tensor,
    crossing_deadband_ratio: float,
    pickup_crossing_weight: float,
    placement_crossing_weight: float,
    motion_cost_scale: float,
) -> ArmCandidateScore:
    """Combine normalized motion length with continuous cross-zone penalties.

    Positive world-y is the semantic robot-view left side. A deadband permits
    small opposite-side reaches, while squared penetration makes deep crossing
    progressively less attractive without overriding IK feasibility.
    """
    if arm not in {"left_arm", "right_arm"}:
        raise ValueError(f"Unsupported semantic arm slot: {arm!r}.")
    if motion_cost.ndim != 1:
        raise ValueError("Arm candidate motion cost must be one-dimensional.")
    if not 0.0 <= crossing_deadband_ratio < 1.0:
        raise ValueError("Arm crossing deadband ratio must be in [0, 1).")
    if pickup_crossing_weight < 0.0 or placement_crossing_weight < 0.0:
        raise ValueError("Arm crossing penalty weights must be non-negative.")
    if motion_cost_scale <= 0.0:
        raise ValueError("Arm candidate motion cost scale must be positive.")

    center = _candidate_batch_values(
        workspace_center_y,
        like=motion_cost,
        name="workspace center",
    )
    half_width = _candidate_batch_values(
        workspace_half_width,
        like=motion_cost,
        name="workspace half-width",
    )
    if bool((half_width <= 0.0).any()):
        raise ValueError("Arm selection workspace half-width must be positive.")

    arm_sign = 1.0 if arm == "left_arm" else -1.0
    deadband = half_width * crossing_deadband_ratio

    def crossing_penalty(
        pose: torch.Tensor | None,
        weight: float,
    ) -> torch.Tensor:
        if pose is None:
            return torch.zeros_like(motion_cost)
        batched_pose = torch.as_tensor(
            pose,
            dtype=motion_cost.dtype,
            device=motion_cost.device,
        )
        if batched_pose.ndim == 1 and batched_pose.shape[0] == 3:
            batched_pose = batched_pose.unsqueeze(0)
        if batched_pose.ndim == 2 and batched_pose.shape == (4, 4):
            batched_pose = batched_pose.unsqueeze(0)
        if batched_pose.ndim == 2 and batched_pose.shape[-1] == 3:
            positions = batched_pose
        elif batched_pose.ndim == 3 and batched_pose.shape[-2:] == (4, 4):
            positions = batched_pose[:, :3, 3]
        else:
            raise ValueError(
                "Arm candidate positions must have shape (3,), (N, 3), "
                "(4, 4), or (N, 4, 4)."
            )
        if positions.shape[0] == 1 and motion_cost.shape[0] != 1:
            positions = positions.expand(motion_cost.shape[0], -1)
        if positions.shape[0] != motion_cost.shape[0]:
            raise ValueError("Arm candidate pose batch does not match motion cost.")
        lateral = positions[:, 1] - center
        wrong_side_depth = torch.clamp(
            -arm_sign * lateral - deadband,
            min=0.0,
        )
        return weight * torch.square(wrong_side_depth / half_width)

    pickup_penalty = crossing_penalty(source_pose, pickup_crossing_weight)
    placement_penalty = crossing_penalty(target_pose, placement_crossing_weight)
    normalized_motion = motion_cost / motion_cost_scale
    return ArmCandidateScore(
        motion_cost=motion_cost,
        normalized_motion_cost=normalized_motion,
        pickup_crossing_penalty=pickup_penalty,
        placement_crossing_penalty=placement_penalty,
        total_cost=normalized_motion + pickup_penalty + placement_penalty,
    )


def _candidate_batch_values(
    values: torch.Tensor,
    *,
    like: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=like.dtype, device=like.device)
    if tensor.ndim == 0:
        tensor = tensor.expand_as(like)
    if tensor.ndim != 1 or tensor.shape != like.shape:
        raise ValueError(f"Arm selection {name} must match the candidate batch.")
    return tensor


def ground_symbolic_action(
    action: Mapping[str, Any],
    semantic_step: Any,
    *,
    env: Any,
    arm: str,
    arrangement_plan: Any = None,
    policy_reference_pose: torch.Tensor | None = None,
) -> GroundedSymbolicAction:
    """Resolve one symbolic action without mutating the Seed graph."""
    robot_profile = str(getattr(env, "agent_robot_profile", ""))
    policy = resolve_motion_policy(
        robot_profile,
        str(action["motion_policy"]),
    )
    binding = action["target_binding"]
    binding_kind = str(binding["kind"])
    action_class = str(action["atomic_action_class"])
    object_pose = _live_pose(env, semantic_step.object_uid)
    reference_pose = _reference_pose(env, semantic_step)
    target_object_pose = None

    spec: dict[str, Any] = {
        "atomic_action_class": action_class,
        "robot_name": arm,
        "control": str(action["control"]),
        "cfg": _atomic_cfg(action_class, policy),
    }
    if binding_kind == "object":
        spec["target_object"] = {
            "obj_name": str(binding["object"]),
            "affordance": str(binding.get("affordance", "antipodal")),
        }
        if action["motion_policy"] == "upright_in_place_pickup":
            axis = str(semantic_step.goal.get("upright_local_axis", "z"))
            spec["cfg"]["obj_upright_direction"] = {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            }[axis]
    elif binding_kind == "semantic_goal":
        target_object_pose = _semantic_target_positions(
            env,
            semantic_step,
            object_pose=object_pose,
            reference_pose=reference_pose,
            policy=policy,
            phase=str(binding.get("phase", "final")),
            arrangement_plan=arrangement_plan,
        )
        target_spec: dict[str, Any] = {
            "reference": "absolute",
            "position_by_env": _positions_json(target_object_pose),
            "orientation_goal": str(
                semantic_step.goal.get("orientation_goal", "preserve")
            ),
            "orientation_axis": str(semantic_step.goal.get("orientation_axis", "none")),
        }
        align_to = semantic_step.goal.get("orientation_reference_object")
        if align_to is not None:
            target_spec["align_to"] = str(align_to)
        phase = str(binding.get("phase", "final"))
        upright_staging = (
            phase == "staging"
            and semantic_step.goal.get("placement_mode") == "upright_in_place"
        )
        support = (
            _surface_support(semantic_step)
            if phase != "staging" or upright_staging
            else None
        )
        if support is not None:
            surface_clearance = float(policy["surface_clearance"])
            if upright_staging:
                surface_clearance += float(policy["staging_lift_height"])
            target_spec.update(
                {
                    "z_policy": "object_on_surface",
                    "support": support,
                    "surface_clearance": surface_clearance,
                }
            )
        spec["target_object_pose"] = target_spec
    elif binding_kind == "current_held_pose":
        spec["target_pose"] = {
            "reference": "relative",
            "offset": [0.0, 0.0, 0.0],
            "frame": "world",
        }
    elif binding_kind == "policy_pose":
        retreat_target = _resolved_retreat_target(
            env,
            arm,
            policy,
            reference_pose=policy_reference_pose,
        )
        if retreat_target is None:
            resolved_heights = [float(policy["retreat_height"])] * int(env.num_envs)
            spec["target_pose"] = {
                "reference": "relative",
                "offset": [0.0, 0.0, float(policy["retreat_height"])],
                "frame": "world",
            }
        else:
            target_positions, target_rotations, resolved_heights = retreat_target
            spec["target_pose"] = {
                "reference": "absolute",
                "position_by_env": target_positions,
                "rotation_matrix_by_env": target_rotations,
            }
        policy["resolved_retreat_height_by_env"] = resolved_heights
    elif binding_kind == "joint_state":
        source = str(binding.get("source", "initial"))
        if source in {"gripper_closed", "gripper_open"}:
            spec["control"] = "hand"
            spec["target_qpos"] = {
                "source": "gripper_state",
                "state": "close" if source == "gripper_closed" else "open",
            }
        else:
            spec["target_qpos"] = {"source": "initial"}
    elif binding_kind == "coordinated_goal":
        target_object_pose = _semantic_target_positions(
            env,
            semantic_step,
            object_pose=object_pose,
            reference_pose=reference_pose,
            policy=policy,
            phase="final",
            arrangement_plan=arrangement_plan,
        )
        spec["control"] = "arm"
        spec["target_object"] = {
            "obj_name": semantic_step.object_uid,
            "affordance": "antipodal",
        }
        spec["target_object_pose"] = {
            "reference": "absolute",
            "position_by_env": _positions_json(target_object_pose),
            "orientation_goal": "preserve",
            "orientation_axis": "none",
        }
    else:
        raise ValueError(f"Unsupported target binding kind: {binding_kind!r}.")
    return GroundedSymbolicAction(
        action_spec=spec,
        motion_policy=policy,
        object_pose=object_pose,
        reference_pose=reference_pose,
        target_object_pose=target_object_pose,
    )


def _resolved_retreat_target(
    env: Any,
    arm: str,
    policy: Mapping[str, Any],
    *,
    reference_pose: torch.Tensor | None,
) -> tuple[list[list[float]], list[list[list[float]]], list[float]] | None:
    """Cap the upward retreat against the robot profile's EEF workspace."""
    pose = reference_pose
    if pose is None:
        if not hasattr(env, "get_current_xpos_agent"):
            return None
        left_pose, right_pose = env.get_current_xpos_agent()
        pose = left_pose if arm == "left_arm" else right_pose
    pose = torch.as_tensor(pose, dtype=torch.float32, device=env.device)
    if pose.ndim == 2:
        pose = pose.unsqueeze(0)
    if pose.ndim != 3 or tuple(pose.shape[-2:]) != (4, 4):
        raise ValueError(
            "Dynamic retreat reference pose must have shape (4, 4) or (N, 4, 4)."
        )
    if pose.shape[0] == 1 and int(env.num_envs) > 1:
        pose = pose.expand(int(env.num_envs), -1, -1)
    if pose.shape[0] != int(env.num_envs):
        raise ValueError("Dynamic retreat pose batch does not match env.num_envs.")

    desired = float(policy["retreat_height"])
    minimum = float(policy["minimum_retreat_height"])
    ceiling = float(policy["maximum_eef_height"])
    available = torch.clamp(ceiling - pose[:, 2, 3], min=0.0)
    heights = torch.minimum(
        available,
        torch.full_like(available, desired),
    )
    # A zero-distance MoveEndEffector is not useful. When the current EEF is
    # already near the ceiling, keep the target at the current pose and let
    # the cleanup failure policy fall through to Home.
    heights = torch.where(heights >= minimum, heights, torch.zeros_like(heights))
    target = pose[:, :3, 3].clone()
    target[:, 2] += heights
    return (
        target.detach().cpu().tolist(),
        pose[:, :3, :3].detach().cpu().tolist(),
        [float(value) for value in heights.detach().cpu().tolist()],
    )


def select_auto_arm_from_candidates(
    left_feasible: torch.Tensor,
    right_feasible: torch.Tensor,
    left_cost: torch.Tensor,
    right_cost: torch.Tensor,
) -> tuple[list[str | None], torch.Tensor]:
    """Choose the feasible lower-cost arm per environment, breaking ties left."""
    tensors = (left_feasible, right_feasible, left_cost, right_cost)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("Auto-arm candidate tensors must be one-dimensional.")
    if len({int(tensor.shape[0]) for tensor in tensors}) != 1:
        raise ValueError("Auto-arm candidate tensors must have matching lengths.")
    assignments: list[str | None] = []
    failed = ~(left_feasible | right_feasible)
    for index in range(len(left_feasible)):
        left_ok = bool(left_feasible[index].item())
        right_ok = bool(right_feasible[index].item())
        if left_ok and (
            not right_ok or float(left_cost[index]) <= float(right_cost[index])
        ):
            assignments.append("left_arm")
        elif right_ok:
            assignments.append("right_arm")
        else:
            assignments.append(None)
    return assignments, failed


def _atomic_cfg(action_class: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    if action_class in {"PickUp", "CoordinatedPickment"}:
        return {
            key: policy[key]
            for key in (
                "pre_grasp_distance",
                "lift_height",
                "sample_interval",
                "rotate_upright",
            )
            if key in policy
        }
    if action_class == "Place":
        return {
            key: policy[key]
            for key in (
                "hand_interp_steps",
                "lift_height",
                "post_hold_steps",
                "sample_interval",
            )
            if key in policy
        }
    return {
        "sample_interval": int(policy.get("sample_interval", 30)),
        **(
            {"post_hold_steps": int(policy["post_hold_steps"])}
            if int(policy.get("post_hold_steps", 0)) > 0
            else {}
        ),
    }


def _live_pose(env: Any, uid: str) -> torch.Tensor:
    entity = env.sim.get_rigid_object(uid)
    if entity is None:
        raise ValueError(f"Unknown live object {uid!r}.")
    return entity.get_local_pose(to_matrix=True).clone()


def _reference_pose(env: Any, semantic_step: Any) -> torch.Tensor | None:
    reference_uid = semantic_step.goal.get("reference_object")
    if not isinstance(reference_uid, str) or not reference_uid:
        return None
    if semantic_step.goal.get("reference_state") == "initial":
        initial = getattr(env, "agent_initial_object_poses", {}).get(reference_uid)
        if initial is None:
            raise ValueError(
                f"Initial pose for semantic reference {reference_uid!r} is unavailable."
            )
        return initial.clone()
    return _live_pose(env, reference_uid)


def _semantic_target_positions(
    env: Any,
    semantic_step: Any,
    *,
    object_pose: torch.Tensor,
    reference_pose: torch.Tensor | None,
    policy: Mapping[str, Any],
    phase: str,
    arrangement_plan: Any,
) -> torch.Tensor:
    goal = semantic_step.goal
    operator = semantic_step.operator
    if operator == "coordinated_pickment":
        target = object_pose[:, :3, 3].clone()
        direction = str(goal.get("direction", "none"))
        distance = float(policy["relation_distance"])
        if "front" in direction:
            target[:, 0] += distance
        elif "back" in direction:
            target[:, 0] -= distance
        if "left" in direction:
            target[:, 1] += distance
        elif "right" in direction:
            target[:, 1] -= distance
        if (
            direction in {"up", "above", "none"}
            or goal.get("terminal_behavior") == "hold"
        ):
            target[:, 2] += float(policy["hover_height"])
        return target
    if operator == "place_in_line":
        if arrangement_plan is None:
            raise ValueError(
                "Arrangement semantic goals require a per-environment runtime plan."
            )
        return arrangement_plan.target_positions(
            semantic_step,
            object_pose=object_pose,
            phase=phase,
            policy=policy,
        )
    if goal.get("placement_mode") == "upright_in_place":
        initial_pose = getattr(env, "agent_initial_object_poses", {}).get(
            semantic_step.object_uid
        )
        return (
            initial_pose[:, :3, 3].clone()
            if initial_pose is not None
            else object_pose[:, :3, 3].clone()
        )

    base = (
        reference_pose[:, :3, 3].clone()
        if reference_pose is not None
        else object_pose[:, :3, 3].clone()
    )
    relation = str(goal.get("relation", "on"))
    distance = float(policy["relation_distance"])
    if relation in {"left", "left_of"}:
        base[:, 1] += distance
    elif relation in {"right", "right_of"}:
        base[:, 1] -= distance
    elif relation in {"front", "in_front_of"}:
        base[:, 0] += distance
    elif relation in {"behind", "back", "back_of"}:
        base[:, 0] -= distance
    elif relation in {"front_left", "front_left_of"}:
        base[:, 0] += distance
        base[:, 1] += distance
    elif relation in {"front_right", "front_right_of"}:
        base[:, 0] += distance
        base[:, 1] -= distance
    elif relation in {"back_left", "back_left_of"}:
        base[:, 0] -= distance
        base[:, 1] += distance
    elif relation in {"back_right", "back_right_of"}:
        base[:, 0] -= distance
        base[:, 1] -= distance
    elif relation == "inside":
        # Keep the object's observed center height while grounding the target
        # to the live container center in XY. Container-specific collision
        # clearance remains a runtime motion-policy concern.
        base[:, 2] = object_pose[:, 2, 3]
    elif relation in {"above", "held_above_initial"}:
        base[:, 2] += float(policy["hover_height"])
    return base


def _surface_support(semantic_step: Any) -> str | None:
    relation = str(semantic_step.goal.get("relation", ""))
    if semantic_step.operator == "coordinated_pickment":
        if semantic_step.goal.get("terminal_behavior") != "place":
            return None
        reference = semantic_step.goal.get("reference_object")
        return str(reference) if relation == "on" and reference else "table"
    if relation in {"on", "on_top", "on_top_of"}:
        reference = semantic_step.goal.get("reference_object")
        return str(reference) if reference else "table"
    if semantic_step.operator == "place_in_line":
        return "table"
    if semantic_step.operator == "place_relative" and relation != "inside":
        return "table"
    return None


def _positions_json(poses: torch.Tensor) -> list[list[float]]:
    positions = poses.detach().cpu().tolist()
    return [[float(value) for value in row] for row in positions]
