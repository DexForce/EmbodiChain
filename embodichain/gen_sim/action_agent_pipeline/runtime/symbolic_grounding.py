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
    "GroundedSymbolicAction",
    "ground_symbolic_action",
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
        support = (
            None
            if binding.get("phase") == "staging"
            else _surface_support(semantic_step)
        )
        if support is not None:
            target_spec.update(
                {
                    "z_policy": "object_on_surface",
                    "support": support,
                    "surface_clearance": float(policy["surface_clearance"]),
                }
            )
        spec["target_object_pose"] = target_spec
    elif binding_kind == "current_held_pose":
        spec["target_object_pose"] = {
            "reference": "relative",
            "offset": [0.0, 0.0, 0.0],
            "frame": "world",
            "orientation_goal": "preserve",
            "orientation_axis": "none",
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
        [[float(value) for value in row] for row in target.detach().cpu().tolist()],
        [
            [[float(value) for value in row] for row in matrix]
            for matrix in pose[:, :3, :3].detach().cpu().tolist()
        ],
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
            for key in ("pre_grasp_distance", "lift_height", "sample_interval")
            if key in policy
        }
    if action_class == "Place":
        return {
            key: policy[key]
            for key in ("lift_height", "post_hold_steps", "sample_interval")
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
