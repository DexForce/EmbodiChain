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

"""Bounded released-hand routes qualified together with the configured park."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
import random
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    EndEffectorPoseGoal,
    JointPositionTarget,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, PlanningFailure
from embodichain.lab.sim.atomic_actions.runtime_commands import JointPositionPayload
from embodichain.lab.task_program.semantics import (
    EffectAssurance,
    Place,
    RegisteredSemanticCall,
)

from .articulation_collision import check_free_motion
from .motion import _joint_velocity_limits, _velocity_diagnostics, _velocity_validity
from .release_clearance import CLEAR_RELEASED_CALL

__all__: list[str] = []

CLEARANCE_RECOVERY_REVISION = 2
_PARK_CALL = "simulation.park"
_RELEASE_CALLS = frozenset(
    {"simulation.place_relative", "gen_sim.place_upright", "gen_sim.stack_place"}
)
_RETREATS = (0.05, 0.0)
_EXTRA_HEIGHTS = (0.05, 0.10)


@dataclass(frozen=True, slots=True)
class ClearanceRoute:
    """One exact generated cleanup occurrence and its bounded park sequence."""

    clearance_invocation_id: str
    park_invocation_ids: tuple[str, ...]
    park_calls: tuple[RegisteredSemanticCall, ...]
    release_call: RegisteredSemanticCall | Place

    @property
    def release_call_id(self) -> str:
        return self.release_call.semantic_id


@dataclass(frozen=True, slots=True)
class _ParkPreview:
    """Ground one actual following call through canonical shared policy."""

    compiler: Any
    workflow: Any
    call_index: int
    invocation_id: str
    skill_id: str = "move_joints"

    @property
    def binding(self) -> Any:
        return self.workflow.calls[self.call_index].bound.binding.action_binding

    def resolve(self, context: Any) -> Any:
        grounded = self.compiler.ground(self.workflow, self.call_index, context)
        invocation = replace(grounded.invocation, invocation_id=self.invocation_id)
        return self.compiler.integration.engine.resolve(invocation)


def bind_clearance_calls(program: Any) -> tuple[ClearanceRoute, ...]:
    """Bind only adjacent generated releases and at most two sequential parks."""
    segments = tuple(program.iter_segments())
    routes = []

    def invocation_id(segment: Any) -> str:
        return f"{program.program_id}/{segment.segment_id}:{segment.calls[0].segment_call_index}"

    for index, segment in enumerate(segments):
        if len(segment.calls) != 1:
            if any(
                getattr(item.call, "call_id", None) == CLEAR_RELEASED_CALL
                for item in segment.calls
            ):
                raise ValueError(
                    "Released-hand clearance requires one sequential call."
                )
            continue
        call = segment.calls[0].call
        if getattr(call, "call_id", None) != CLEAR_RELEASED_CALL:
            continue
        following_parks = []
        for following in segments[index + 1 :]:
            if (
                len(following.calls) != 1
                or following.parallel_block is not None
                or getattr(following.calls[0].call, "call_id", None) != _PARK_CALL
            ):
                break
            park = following.calls[0].call
            if (
                type(park) is not RegisteredSemanticCall
                or dict(park.arguments)
                or set(park.resources) != {"primary"}
            ):
                raise ValueError(
                    "Released-hand clearance requires explicit pure parks."
                )
            following_parks.append((invocation_id(following), park))
        park_resources = [dict(park.resources) for _, park in following_parks]
        previous = segments[index - 1] if index else None
        release = (
            previous.calls[0].call
            if previous is not None and len(previous.calls) == 1
            else None
        )
        if (
            type(call) is not RegisteredSemanticCall
            or set(call.resources) != {"primary"}
            or segment.parallel_block is not None
            or not 1 <= len(following_parks) <= 2
            or len({park.resources["primary"] for _, park in following_parks})
            != len(following_parks)
            or dict(call.resources) not in park_resources
        ):
            raise ValueError(
                "Released-hand clearance requires at most two distinct sequential "
                "parks including its explicit resource."
            )
        release_object = (
            release.arguments.get("object")
            if type(release) is RegisteredSemanticCall
            else release.object.entity_id if type(release) is Place else None
        )
        if (
            type(release) not in {RegisteredSemanticCall, Place}
            or (
                type(release) is RegisteredSemanticCall
                and release.call_id not in _RELEASE_CALLS
            )
            or release_object != call.arguments.get("object")
            or dict(release.resources) != dict(call.resources)
            or previous.parallel_block is not None
        ):
            raise ValueError(
                "Released-hand clearance requires its preceding same-object release."
            )
        routes.append(
            ClearanceRoute(
                invocation_id(segment),
                tuple(value[0] for value in following_parks),
                tuple(value[1] for value in following_parks),
                release,
            )
        )
    return tuple(routes)


def bind_park_requests(
    compiler: Any,
    routes: tuple[ClearanceRoute, ...],
) -> dict[str, tuple[_ParkPreview, ...]]:
    """Analyze actual park calls using canonical runtime binding policy."""
    requests = {}
    for route in routes:
        release = compiler.integration.link_call(route.release_call)
        release_preset = release.preset
        if (
            release_preset.effect_assurance is not EffectAssurance.VERIFIED
            or route.release_call_id not in release_preset.effect_monitors
        ):
            raise ValueError(
                "Released-hand recovery requires verified preceding release effects."
            )
        workflow = compiler.analyze(
            route.park_calls, workflow_id=f"{route.clearance_invocation_id}/parks"
        )
        requests[route.clearance_invocation_id] = tuple(
            _ParkPreview(compiler, workflow, index, invocation_id)
            for index, invocation_id in enumerate(route.park_invocation_ids)
        )
    return requests


def validate_motion(
    plan: Any, context: Any, robot: Any, simulation: Any, control_part: str
) -> dict[str, Any]:
    """Fail closed on unqualified timing, limits, scene, self or other-arm meshes."""
    if context.batch_size != 1 or context.env_ids.tolist() != [0]:
        return {"valid": False, "reason": "unsupported_collision_batch"}
    trajectory = plan.joint_trajectory
    if trajectory is None or not bool(plan.plan_success.all()):
        return {"valid": False, "reason": "planning_failed"}
    if not _trajectory_shape_valid(trajectory, context):
        return {"valid": False, "reason": "invalid_trajectory"}
    if isinstance(plan, ActionPlan):
        commands = _command_trajectory_check(plan, context, robot, control_part)
        if not commands["valid"]:
            return commands
    positions, dt = trajectory.positions, trajectory.dt
    # Keep zero-time boundaries zero-time: a changed first qpos must not be
    # granted an extra control period that the emitted commands do not own.
    positions = torch.cat((context.robot.qpos[:, None], positions), dim=1)
    dt = torch.cat((dt.new_zeros((1, 1)), dt), dim=1)
    joint_ids = [int(value) for value in robot.get_joint_ids(name=control_part)]
    if (
        not joint_ids
        or len(set(joint_ids)) != len(joint_ids)
        or any(value < 0 or value >= positions.shape[-1] for value in joint_ids)
    ):
        return {"valid": False, "reason": "invalid_control_joint_ids"}
    passive = [value for value in range(positions.shape[-1]) if value not in joint_ids]
    if passive and not torch.equal(
        positions[:, :, passive],
        context.robot.qpos[:, None, passive].expand(-1, positions.shape[1], -1),
    ):
        return {"valid": False, "reason": "uncommanded_joint_changed"}
    limits = _joint_velocity_limits(robot, None).to(positions)[context.env_ids]
    velocity = {
        "valid": bool(_velocity_validity(positions, dt, limits).all()),
        "diagnostics": _velocity_diagnostics(positions, dt, limits),
    }
    bounds = robot.get_qpos_limits(name=control_part, env_ids=context.env_ids).to(
        positions
    )
    if (
        bounds.shape != (1, len(joint_ids), 2)
        or torch.isnan(bounds).any()
        or (bounds[..., 0] > bounds[..., 1]).any()
    ):
        return {"valid": False, "reason": "invalid_joint_limits", "velocity": velocity}
    commanded = positions[:, :, joint_ids]
    invalid = (commanded < bounds[:, None, :, 0] - 1e-5) | (
        commanded > bounds[:, None, :, 1] + 1e-5
    )
    joint_limits = {"valid": not bool(invalid.any())}
    if invalid.any():
        _, frame, joint = invalid.nonzero()[0].tolist()
        joint_limits.update({"frame": frame, "joint": joint_ids[joint]})
    result = {"valid": False, "velocity": velocity, "joint_limits": joint_limits}
    if not velocity["valid"] or not joint_limits["valid"]:
        result["reason"] = "velocity_or_joint_limit"
        return result
    try:
        collision = check_free_motion(robot, simulation, positions[0], control_part)
    except Exception as exc:
        result.update({"reason": "collision_check_unavailable", "error": str(exc)})
        return result
    result.update({"valid": collision.get("valid") is True, "collision": collision})
    return result


def _command_trajectory_check(
    plan: ActionPlan, context: Any, robot: Any, control_part: str
) -> dict[str, Any]:
    """Certify emitted arm payloads and scheduling, not just inspection tensors."""
    trajectory = plan.joint_trajectory
    joint_ids = tuple(int(value) for value in robot.get_joint_ids(name=control_part))
    failure = {"valid": False, "reason": "command_trajectory_mismatch"}
    if (
        trajectory is None
        or plan.commands.frame_count != trajectory.positions.shape[1]
        or not torch.equal(plan.commands.env_ids, context.env_ids)
        or not torch.equal(trajectory.env_ids, context.env_ids)
    ):
        return failure
    limits = _joint_velocity_limits(robot, control_part).to(trajectory.positions)[
        context.env_ids
    ]
    for index, frame in enumerate(plan.commands.frames):
        if (
            len(frame.commands) != 1
            or not torch.equal(frame.env_ids, context.env_ids)
            or not torch.equal(frame.active_mask, plan.plan_success)
            or not torch.equal(
                frame.hold_duration,
                trajectory.dt[:, min(index + 1, trajectory.positions.shape[1] - 1)],
            )
        ):
            return {**failure, "frame": index, "detail": "command_scope_or_timing"}
        command = frame.commands[0]
        if (
            type(command.target) is not JointPositionTarget
            or type(command.payload) is not JointPositionPayload
            or command.target.control_part != control_part
            or command.target.joint_ids != joint_ids
            or not torch.equal(
                command.payload.positions,
                trajectory.positions[:, index, list(joint_ids)],
            )
        ):
            return {**failure, "frame": index, "detail": "joint_position_payload"}
        velocities = command.payload.velocities
        if (velocities is None) != (trajectory.velocities is None) or (
            velocities is not None
            and (
                not torch.equal(
                    velocities, trajectory.velocities[:, index, list(joint_ids)]
                )
                or (velocities.abs() > limits + 1e-5).any()
            )
        ):
            return {**failure, "frame": index, "detail": "joint_velocity_payload"}
    return {"valid": True}


def _trajectory_shape_valid(trajectory: Any, context: Any) -> bool:
    positions, dt = trajectory.positions, trajectory.dt
    return (
        positions.ndim == 3
        and positions.shape[0] == 1
        and positions.shape[1] > 0
        and positions.shape[2] == context.robot.qpos.shape[1]
        and dt.shape == positions.shape[:2]
        and bool(torch.isfinite(positions).all())
        and bool(torch.isfinite(dt).all())
        and not bool((dt < 0).any())
    )


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
    park_request: tuple[Any, ...],
    robot: Any,
    simulation: Any,
) -> Any:
    """Qualify bounded routes using shared skills; return only the cleanup plan."""
    endpoint = request.binding.endpoint("primary", "motion")
    motion = endpoint.require_target(JointPositionTarget)
    if type(park_request) is not tuple or not 1 <= len(park_request) <= 2:
        return _annotate(
            original,
            {"accepted": False, "reason": "invalid_released_hand_scope"},
            failed=True,
        )
    park_endpoints = [
        item.binding.endpoint("primary", "motion") for item in park_request
    ]
    park_motions = [item.require_target(JointPositionTarget) for item in park_endpoints]
    held = context.task.get_held_object(endpoint.task_state_key)
    if (
        context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or type(request.goal) is not EndEffectorPoseGoal
        or request.goal.xpos.shape != (1, 2, 4, 4)
        or motion not in park_motions
        or len({item.control_part for item in park_motions}) != len(park_motions)
        or (held is not None and held.active_mask.any())
        or any(
            (attachment := context.task.get_held_object(item.task_state_key))
            is not None
            and attachment.active_mask.any()
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
        entry.update(
            validate_motion(plan, context, robot, simulation, motion.control_part)
        )
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
            context, robot=replace(context.robot, qpos=end, qvel=torch.zeros_like(end))
        )
        entry["parks"] = []
        entry["park_frames"] = 0
        # Project every full-robot endpoint in real call order; each segment
        # certifies its own commanded arm and holds the other arm unchanged.
        for selected, target in zip(park_request, park_motions):
            try:
                selected = (
                    selected.resolve(predicted)
                    if isinstance(selected, _ParkPreview)
                    else selected
                )
                park = build(selected, predicted)
            except (ValueError, RuntimeError) as exc:
                unavailable = True
                entry.update(
                    {"valid": False, "reason": "park_planning_error", "error": str(exc)}
                )
                return False
            acceptance = validate_motion(
                park, predicted, robot, simulation, target.control_part
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
