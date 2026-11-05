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

"""Source bounded cleanup routes, using explicit shared suffix authority."""

from __future__ import annotations
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
import random
from typing import Any
import numpy as np
import torch
from ..bindings import JointPositionTarget
from ..plans import PlanningFailure
from .move_end_effector import EndEffectorPoseGoal
from ._released_park import validate_motion, _trajectory_shape_valid

__all__: list[str] = []
_RETREATS = (0.05, 0.0)
_EXTRA_HEIGHTS = (0.05, 0.10)


def _route_goal(
    goal: EndEffectorPoseGoal, retreat: float, extra_height: float
) -> EndEffectorPoseGoal:
    poses = goal.xpos.clone()
    if poses.shape != (1, 2, 4, 4):
        raise ValueError("Released-hand recovery requires its exact two-stage goal.")
    direction = torch.nn.functional.normalize(
        poses[:, 1, :2, 3] - poses[:, 0, :2, 3], dim=-1
    )
    poses[:, 1, :2, 3] = poses[:, 0, :2, 3] + retreat * direction
    poses[:, :, 2, 3] += extra_height
    return EndEffectorPoseGoal(poses)


def _annotate(plan: Any, report: dict[str, Any], *, failed: bool = False) -> Any:
    diagnostics = replace(
        plan.diagnostics,
        metadata={**plan.diagnostics.metadata, "gen_sim_clearance_recovery": report},
        **(
            {"failure": PlanningFailure("cleanup_clearance_failed", retryable=False)}
            if failed
            else {}
        ),
    )
    return replace(
        plan,
        diagnostics=diagnostics,
        **({"plan_success": torch.zeros_like(plan.plan_success)} if failed else {}),
    )


@contextmanager
def _preserve_random_state(device: torch.device) -> Iterator[None]:
    devices = []
    if device.type == "cuda":
        devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def recover(
    request: Any,
    context: Any,
    original: Any,
    *,
    build: Callable[[Any, Any], Any],
    session: Any,
    robot: Any,
    query: Any,
    plan_preview: Callable[[Any, Any], Any] | None = None,
) -> Any:
    """Qualify bounded routes using shared skills; return only the cleanup plan."""
    endpoint = request.binding.endpoint("primary", "motion")
    motion = endpoint.require_target(JointPositionTarget)
    park_bindings = session.park_bindings
    if type(park_bindings) is not tuple or not 1 <= len(park_bindings) <= 2:
        return _annotate(
            original,
            {"accepted": False, "reason": "invalid_released_hand_scope"},
            failed=True,
        )
    park_endpoints = [item.endpoint("primary", "motion") for item in park_bindings]
    park_motions = [item.require_target(JointPositionTarget) for item in park_endpoints]
    if (
        context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or type(request.goal) is not EndEffectorPoseGoal
        or request.goal.xpos.shape != (1, 2, 4, 4)
        or motion not in park_motions
        or len({item.control_part for item in park_motions}) != len(park_motions)
        or context.task.held_object_mask(endpoint.task_state_key).any()
        or any(
            context.task.held_object_mask(item.task_state_key).any()
            for item in park_endpoints
        )
    ):
        return _annotate(
            original,
            {"accepted": False, "reason": "invalid_released_hand_scope"},
            failed=True,
        )
    attempts: list[dict[str, Any]] = []
    park_collision = False
    unavailable = False
    initial_collision = False

    def qualify(plan: Any, name: str) -> bool:
        nonlocal park_collision, unavailable, initial_collision
        entry = {"route": name}
        attempts.append(entry)
        if plan.joint_trajectory is None or not plan.plan_success.all():
            entry.update({"valid": False, "reason": "clearance_planning_failed"})
            return False
        if not _trajectory_shape_valid(plan.joint_trajectory, context):
            entry.update({"valid": False, "reason": "invalid_clearance_trajectory"})
            return False
        entry.update(validate_motion(plan, context, robot, query, motion.control_part))
        entry["clearance_frames"] = plan.joint_trajectory.positions.shape[1]
        if not entry["valid"]:
            unavailable = entry.get("reason") == "collision_check_unavailable"
            collision = entry.get("collision", {})
            initial_collision = (
                collision.get("valid") is False and collision.get("frame") == 0
            )
            entry["failed_stage"] = (
                "initial_state" if initial_collision else "clearance"
            )
            if initial_collision:
                entry["reason"] = "initial_state_collision"
            return False
        end = plan.joint_trajectory.positions[:, -1]
        predicted = replace(
            context,
            robot=replace(context.robot, qpos=end, qvel=torch.zeros_like(end)),
            is_preview=True,
            observed_robot=context.observed_robot or context.robot,
        )
        entry["parks"] = []
        entry["park_frames"] = 0
        # Project every full-robot endpoint in real call order; each segment
        # certifies its own commanded arm and holds the other arm unchanged.
        for index, target in enumerate(park_motions):
            try:
                selected = session.resolve_park(index, predicted)
                park = (
                    session.plan_park(selected, predicted)
                    if plan_preview is None
                    else plan_preview(selected, predicted)
                )
            except (ValueError, RuntimeError) as exc:
                unavailable = True
                entry.update(
                    {"valid": False, "reason": "park_planning_error", "error": str(exc)}
                )
                return False
            acceptance = validate_motion(
                park, predicted, robot, query, target.control_part
            )
            acceptance["invocation_id"] = selected.invocation_id
            entry["parks"].append(acceptance)
            if not acceptance["valid"]:
                entry.update({**acceptance, "failed_stage": "park"})
                unavailable = acceptance.get("reason") == "collision_check_unavailable"
                park_collision = (
                    park_collision
                    or acceptance.get("collision", {}).get("valid") is False
                )
                return False
            entry["park_frames"] += park.joint_trajectory.positions.shape[1]
            end = park.joint_trajectory.positions[:, -1]
            predicted = replace(
                predicted,
                robot=replace(predicted.robot, qpos=end, qvel=torch.zeros_like(end)),
            )
        return entry["valid"]

    with _preserve_random_state(context.robot.qpos.device):
        if qualify(original, "original"):
            return _annotate(
                original,
                {"accepted": True, "selected_route": "original", "attempts": attempts},
            )
        for extra_height in (0.0, *_EXTRA_HEIGHTS):
            if unavailable or initial_collision:
                break
            if extra_height and not park_collision:
                break
            for retreat in _RETREATS:
                name = f"retreat_{retreat:.2f}_extra_height_{extra_height:.2f}"
                candidate = replace(
                    request, goal=_route_goal(request.goal, retreat, extra_height)
                )
                try:
                    plan = build(candidate, context)
                except (ValueError, RuntimeError) as exc:
                    attempts.append(
                        {
                            "route": name,
                            "valid": False,
                            "reason": "clearance_planning_error",
                            "error": str(exc),
                        }
                    )
                    return _annotate(
                        original, {"accepted": False, "attempts": attempts}, failed=True
                    )
                if qualify(plan, name):
                    return _annotate(
                        plan,
                        {
                            "accepted": True,
                            "selected_route": name,
                            "attempts": attempts,
                        },
                    )
                if unavailable or initial_collision:
                    break
    return _annotate(
        original,
        {
            "accepted": False,
            **({"reason": "initial_state_collision"} if initial_collision else {}),
            "attempts": attempts,
        },
        failed=True,
    )
