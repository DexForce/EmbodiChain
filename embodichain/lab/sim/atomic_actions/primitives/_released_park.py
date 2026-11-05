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

"""Fixed-source released Park qualification and bounded vias."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any
import torch

from embodichain.compute.trajectory.limits import (
    joint_velocity_validity as _velocity_validity,
    joint_velocity_diagnostics as _velocity_diagnostics,
)
from embodichain.lab.sim.motion.limits import (
    authored_joint_velocity_limits as _joint_velocity_limits,
)
from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
from ..bindings import JointPositionTarget
from ..control import PARK_COMMAND
from ..plans import ActionPlan, PlanningFailure
from ..runtime_commands import JointPositionPayload
from .move_joints import JointPositionGoal

__all__: list[str] = []
_UNAVAILABLE = (
    ImportError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    TypeError,
)


def eligible(action, request, context):
    try:
        endpoint = request.binding.endpoint("primary", "motion")
        return bool(
            request.skill_id == "move_joints"
            and type(request.goal) is JointPositionGoal
            and isinstance(request.goal.target, str)
            and request.goal.target == PARK_COMMAND
            and len(request.binding.endpoints) == 1
            and request.motion_policy.strategy == "ik_interp"
            and request.motion_policy.dynamic_collision_mode == "off"
            and context.batch_size == 1
            and context.env_ids.tolist() == [0]
            and not context.task.held_object_mask(endpoint.task_state_key).any()
            and not any(
                endpoint.task_state_key in keys
                and (state.env_mask is None or state.env_mask.any())
                for keys, state in context.task.coordinated_held_objects.items()
            )
        )
    except _UNAVAILABLE:
        return False


def validate_motion(
    plan: Any, context: Any, robot: Any, query: Any, control_part: str
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
        if query is None:
            raise ValueError("Explicit free-motion query is unavailable.")
        report = query.inspect(positions[0], control_part=control_part)
        if (
            type(report) is not MotionCollisionReport
            or report.scope != "discrete_free_motion_meshes"
        ):
            raise ValueError("Park requires complete free-scene qualification.")
        collision = report.metadata()
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


def basic_check(
    action: Any, request: Any, context: Any, plan: Any, *, frames: int | None = None
) -> dict[str, Any]:
    """Check final ready, the full control grid and every emitted payload."""
    try:
        if not bool(plan.plan_success.all()) or plan.joint_trajectory is None:
            return {"valid": False, "reason": "planning_failed"}
        endpoint = request.binding.endpoint("primary", "motion")
        target = endpoint.require_target(JointPositionTarget)
        trajectory = plan.joint_trajectory
        positions, dt = trajectory.positions, trajectory.dt
        ready = endpoint.joint_positions(
            PARK_COMMAND, num_envs=1, device=positions.device, dtype=positions.dtype
        )
        if (
            plan.skill_id != request.skill_id
            or plan.invocation_id != request.invocation_id
            or positions.ndim != 3
            or positions.shape[0] != 1
            or not positions.shape[1]
            or positions.shape[-1] != context.robot.qpos.shape[-1]
            or dt.shape != positions.shape[:2]
            or not torch.isfinite(dt).all()
            or not torch.equal(dt[:, :1], torch.zeros_like(dt[:, :1]))
            or not torch.equal(
                dt[:, 1:], torch.full_like(dt[:, 1:], context.control_dt)
            )
            or (frames is not None and positions.shape[1] != frames)
        ):
            return {"valid": False, "reason": "control_grid_changed"}
        if not torch.allclose(
            positions[:, -1, list(target.joint_ids)], ready, atol=1e-5, rtol=0
        ):
            return {"valid": False, "reason": "final_ready_changed"}
        velocities = trajectory.velocities
        if velocities is not None:
            limits = _joint_velocity_limits(action.robot, None).to(velocities)[
                context.env_ids
            ]
            if (
                velocities.shape != positions.shape
                or not torch.isfinite(velocities).all()
                or (velocities.abs() > limits[:, None] + 1e-5).any()
            ):
                return {"valid": False, "reason": "velocity_payload_rejected"}
        return _command_trajectory_check(
            plan, context, action.robot, target.control_part
        )
    except _UNAVAILABLE as error:
        return {"valid": False, "reason": str(error)}


def _annotate(plan: Any, report: dict[str, Any], *, failed: bool = False) -> Any:
    return replace(
        plan,
        plan_success=(
            torch.zeros_like(plan.plan_success) if failed else plan.plan_success
        ),
        diagnostics=replace(
            plan.diagnostics,
            metadata={**plan.diagnostics.metadata, "gen_sim_park_recovery": report},
            **(
                {"failure": PlanningFailure("park_clearance_failed", retryable=False)}
                if failed
                else {}
            ),
        ),
    )


def _neutral_joint_routes(
    current: torch.Tensor, ready: torch.Tensor, bounds: torch.Tensor
) -> Iterator[tuple[int, torch.Tensor]]:
    """Temporarily free one interior joint, then restore the exact ready target."""
    for index in range(1, min(current.shape[1] - 1, 5)):
        neutral = torch.zeros_like(current[:, index]).clamp(
            min=bounds[:, index, 0], max=bounds[:, index, 1]
        )
        first, second = current.clone(), ready.clone()
        first[:, index] = second[:, index] = neutral
        yield index, torch.stack((first, second, ready), dim=1)


def recover(
    action: Any,
    request: Any,
    context: Any,
    original: Any,
    *,
    acceptance: dict[str, Any],
    build: Callable[[Any, Any], Any],
    query: Any,
) -> Any:
    """Keep bounded, fully checked vias in previews and actual Park planning."""
    if not eligible(action, request, context):
        return original
    basic = basic_check(action, request, context, original)
    if basic["valid"] and acceptance.get("valid") is True:
        return original
    report: dict[str, Any] = {"accepted": False, "attempts": []}
    collision = acceptance.get("collision", {})
    frames = (
        original.joint_trajectory.positions.shape[1]
        if original.joint_trajectory is not None
        else 0
    )
    if (
        not basic["valid"]
        or collision.get("kind") not in {"scene", "self", "other_arm"}
        or type(collision.get("frame")) is not int
        or not 1 < collision["frame"] < frames
        or acceptance.get("joint_limits", {}).get("valid") is not True
        or acceptance.get("velocity", {}).get("valid") is not True
    ):
        return _annotate(
            original,
            {**report, "reason": basic.get("reason", "nonrecoverable_park_failure")},
            failed=True,
        )
    report["original_collision"] = collision
    robot = action.robot
    stop = PlanningRandomState.capture((robot.device, context.robot.qpos.device))
    before = robot.get_qpos().detach().clone()
    try:
        endpoint = request.binding.endpoint("primary", "motion")
        target = endpoint.require_target(JointPositionTarget)
        ids, part = list(target.joint_ids), target.control_part
        current = context.robot.qpos[:, ids].clone()
        ready = endpoint.joint_positions(
            PARK_COMMAND, num_envs=1, device=current.device, dtype=current.dtype
        )

        def qualify(waypoints: torch.Tensor, entry: dict[str, Any]) -> Any:
            selected = replace(request, goal=JointPositionGoal(waypoints))
            candidate = build(selected, context)
            check = basic_check(action, selected, context, candidate, frames=frames)
            if check["valid"]:
                check = validate_motion(candidate, context, robot, query, part)
            entry.update(
                accepted=check["valid"],
                reason=check.get("reason"),
                collision=check.get(
                    "collision",
                    candidate.diagnostics.metadata.get(
                        "gen_sim_clearance_park", {}
                    ).get("collision"),
                ),
            )
            if check["valid"]:
                report["accepted"] = True
                return _annotate(candidate, report)
            return None

        start_pose = (
            robot.compute_fk(qpos=current, name=part, to_matrix=True).detach().clone()
        )
        ready_pose = (
            robot.compute_fk(qpos=ready, name=part, to_matrix=True).detach().clone()
        )
        vias = (("midpoint", 0.05), ("over_ready", 0.10))
        for kind, height in vias:
            pose = (start_pose if kind == "midpoint" else ready_pose).clone()
            if kind == "midpoint":
                pose[:, :3, 3] = (start_pose[:, :3, 3] + ready_pose[:, :3, 3]) / 2
            pose[:, 2, 3] += height
            for seed_name, seed in (("current", current), ("ready", ready)):
                stop.restore()
                entry = {
                    "kind": kind,
                    "height": height,
                    "seed": seed_name,
                    "accepted": False,
                }
                report["attempts"].append(entry)
                try:
                    solved = robot.compute_ik(
                        pose=pose.clone(),
                        joint_seed=seed.clone(),
                        name=part,
                        env_ids=[0],
                    )
                    if solved is None or not bool(solved[0].all()):
                        entry["reason"] = "via_ik_failed"
                        continue
                    via = solved[1].reshape(1, -1).detach().clone()
                    if via.shape != ready.shape or not torch.isfinite(via).all():
                        raise ValueError("Invalid Park IK waypoint.")
                    accepted = qualify(torch.stack((via, ready), dim=1), entry)
                    if accepted is not None:
                        return accepted
                except _UNAVAILABLE as error:
                    entry["reason"] = str(error)
                finally:
                    if not torch.equal(before, robot.get_qpos()):
                        raise RuntimeError("Park planning changed native robot state.")
        if collision["kind"] in {"self", "other_arm"}:
            bounds = robot.get_qpos_limits(name=part, env_ids=context.env_ids).to(
                current
            )
            for index, waypoints in _neutral_joint_routes(current, ready, bounds):
                stop.restore()
                entry = {
                    "kind": "neutral_joint",
                    "joint_index": index,
                    "accepted": False,
                }
                report["attempts"].append(entry)
                try:
                    accepted = qualify(waypoints, entry)
                    if accepted is not None:
                        return accepted
                except _UNAVAILABLE as error:
                    entry["reason"] = str(error)
                finally:
                    if not torch.equal(before, robot.get_qpos()):
                        raise RuntimeError("Park planning changed native robot state.")
        return _annotate(original, report, failed=True)
    finally:
        stop.restore()


def qualify_and_recover_park(action, request, context, plan, build):
    """Retain recursive source qualification on the same observed Park commands."""
    endpoint = request.binding.endpoint("primary", "motion")
    motion = endpoint.require_target(JointPositionTarget)
    query = action.planning_services.free_motion_query
    held = context.task.get_held_object(endpoint.task_state_key)
    acceptance = (
        {"valid": False, "reason": "park_requires_released_hand"}
        if held is not None and held.active_mask.any()
        else validate_motion(plan, context, action.robot, query, motion.control_part)
    )
    original = plan
    plan = recover(
        action,
        request,
        context,
        plan,
        acceptance=acceptance,
        build=lambda selected, current: qualify_and_recover_park(
            action, selected, current, build(selected), build
        ),
        query=query,
    )
    if plan is not original and plan.plan_success.all():
        acceptance = plan.diagnostics.metadata["gen_sim_clearance_park"]
    return replace(
        plan,
        plan_success=(
            plan.plan_success
            if acceptance["valid"]
            else torch.zeros_like(plan.plan_success)
        ),
        diagnostics=replace(
            plan.diagnostics,
            metadata={
                **plan.diagnostics.metadata,
                "gen_sim_clearance_park": acceptance,
            },
            **(
                {
                    "failure": (
                        plan.diagnostics.failure
                        if "gen_sim_park_recovery" in plan.diagnostics.metadata
                        else PlanningFailure("cleanup_park_failed", retryable=False)
                    )
                }
                if not acceptance["valid"]
                else {}
            ),
        ),
    )
