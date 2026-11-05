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

"""Final full-robot command and robot-proxy checks for manipulation planning."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import torch

from embodichain.compute.trajectory.limits import (
    joint_velocity_validity,
    joint_velocity_diagnostics,
)
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.mesh_collision import inspect_robot_collision
from embodichain.lab.sim.motion.search import PlanningRandomState
from ..bindings import JointPositionTarget
from ..runtime_commands import JointPositionPayload

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from ..invocation import ResolvedActionRequest
    from ..state import PlanningContext
    from ..plans import ActionPlan

__all__: list[str] = []


def inspect_robot_motion(
    robot: Robot, positions: torch.Tensor, part: str
) -> dict[str, Any]:
    """Check robot collision proxies while restoring planning random streams.

    Args:
        robot: Robot owning the input geometry and explicit control part.
        positions: Full-robot joint samples, shape ``(T, D)``.
        part: Exact declared motion control part.

    Returns:
        Robot-mesh qualification metadata. Missing inputs remain unavailable;
        the result does not establish scene clearance or physical contact.
    """
    state = PlanningRandomState.capture((robot.device, positions.device))
    try:
        return inspect_robot_collision(robot, positions, control_part=part).metadata()
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        return {"valid": False, "unavailable": str(error)}
    finally:
        state.restore()


def inspect_manipulation_plan(
    robot: Robot,
    request: ResolvedActionRequest,
    context: PlanningContext,
    plan: ActionPlan,
    *,
    _initial_command_qpos: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Check complete B1 manipulation commands, limits, timing and robot proxies.

    Args:
        robot: Exact robot whose full joint state was observed.
        request: Resolved request with primary motion and grasp endpoints.
        context: Original B1/env0 planning state and declared cadence.
        plan: Complete emitted command/trajectory plan.
        _initial_command_qpos: Optional retained-hand command boundary; every
            observed non-hand joint and all emitted hand values must agree.

    Returns:
        Qualification metadata with the first rejection or unavailable reason.
        This function issues no commands and proves no physical effect.
    """

    try:
        trajectory = plan.joint_trajectory
        positions, dt = trajectory.positions, trajectory.dt
        endpoints = tuple(
            request.binding.endpoint("primary", name) for name in ("motion", "grasp")
        )
        targets = tuple(
            endpoint.require_target(JointPositionTarget) for endpoint in endpoints
        )
        active = [index for target in targets for index in target.joint_ids]
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or positions.shape[1] == 0
            or positions.shape[-1] != context.robot.qpos.shape[-1]
            or dt.shape != positions.shape[:2]
            or context.env_ids.tolist() != [0]
            or not torch.isfinite(positions).all()
            or not torch.isfinite(dt).all()
            or (dt < 0).any()
            or len(set(active)) != len(active)
            or not active
            or min(active) < 0
            or max(active) >= positions.shape[-1]
            or endpoints[0].task_state_key != endpoints[1].task_state_key
        ):
            raise ValueError(
                "Pick full-robot trajectory or endpoint scope is unavailable."
            )
        passive = [index for index in range(positions.shape[-1]) if index not in active]
        if passive and not torch.equal(
            positions[:, :, passive],
            context.robot.qpos[:, None, passive].expand(-1, positions.shape[1], -1),
        ):
            raise ValueError("Pick changed an uncommanded joint.")
        if not torch.equal(
            positions[:, 0, list(targets[0].joint_ids)],
            context.robot.qpos[:, list(targets[0].joint_ids)],
        ):
            raise ValueError("Pick changed its zero-time arm start.")
        limits = authored_joint_velocity_limits(robot, None).to(positions)[
            context.env_ids
        ]
        if not joint_velocity_validity(positions, dt, limits).all():
            velocity = joint_velocity_diagnostics(positions, dt, limits)
            for row in velocity:
                row["phase"] = next(
                    (s.name for s in plan.segments if s.start <= row["frame"] < s.stop),
                    None,
                )
            return {
                "valid": False,
                "reason": "final_velocity_rejected",
                "velocity": velocity,
            }
        for target in targets:
            bounds = robot.get_qpos_limits(
                name=target.control_part, env_ids=context.env_ids
            ).to(positions)
            values = positions[:, :, list(target.joint_ids)]
            if (
                bounds.shape != (1, len(target.joint_ids), 2)
                or torch.isnan(bounds).any()
                or (bounds[..., 0] > bounds[..., 1]).any()
            ):
                raise ValueError("Pick endpoint joint limits are unavailable.")
            if (
                (values < bounds[:, None, :, 0]) | (values > bounds[:, None, :, 1])
            ).any():
                return {"valid": False, "reason": "joint_limits_rejected"}
        if (
            plan.commands.frame_count != positions.shape[1]
            or not torch.equal(plan.commands.env_ids, context.env_ids)
            or not torch.equal(trajectory.env_ids, context.env_ids)
        ):
            raise ValueError("Pick command/trajectory scope differs.")
        for index, frame in enumerate(plan.commands.frames):
            if (
                len(frame.commands) != len(targets)
                or not torch.equal(frame.env_ids, context.env_ids)
                or not torch.equal(frame.active_mask, plan.plan_success)
                or not torch.equal(
                    frame.hold_duration, dt[:, min(index + 1, positions.shape[1] - 1)]
                )
            ):
                raise ValueError("Pick emitted a different command schedule.")
            seen = set()
            for command in frame.commands:
                if (
                    type(command.target) is not JointPositionTarget
                    or command.target not in targets
                    or command.target.control_part in seen
                    or type(command.payload) is not JointPositionPayload
                ):
                    raise ValueError("Pick emitted an unexpected command target.")
                seen.add(command.target.control_part)
                ids = list(command.target.joint_ids)
                if not torch.equal(command.payload.positions, positions[:, index, ids]):
                    raise ValueError(
                        "Pick command positions differ from its trajectory."
                    )
                velocities = command.payload.velocities
                if (velocities is None) != (trajectory.velocities is None) or (
                    velocities is not None
                    and (
                        not torch.equal(
                            velocities, trajectory.velocities[:, index, ids]
                        )
                        or not torch.isfinite(velocities).all()
                        or (velocities.abs() > limits[:, ids] + 1e-5).any()
                    )
                ):
                    raise ValueError("Pick command velocities are unqualified.")
        first_hold = plan.commands.frames[0].hold_duration
        boundary_start = context.robot.qpos.to(positions)
        if _initial_command_qpos is not None:
            hand_ids = list(targets[1].joint_ids)
            observed_ids = [
                index for index in range(positions.shape[-1]) if index not in hand_ids
            ]
            if (
                _initial_command_qpos.shape != boundary_start.shape
                or not torch.isfinite(_initial_command_qpos).all()
                or not torch.equal(
                    _initial_command_qpos[:, observed_ids],
                    boundary_start[:, observed_ids],
                )
                or not torch.equal(
                    positions[:, :, hand_ids],
                    _initial_command_qpos[:, None, hand_ids].expand(
                        -1, positions.shape[1], -1
                    ),
                )
            ):
                raise ValueError(
                    "Retained-hand boundary cannot replace robot observations."
                )
            boundary_start = _initial_command_qpos.to(positions)
        boundary_positions = torch.cat(
            (boundary_start[:, None], positions[:, :1]), dim=1
        )
        boundary_dt = torch.stack((torch.zeros_like(first_hold), first_hold), dim=1)
        if not joint_velocity_validity(boundary_positions, boundary_dt, limits).all():
            return {"valid": False, "reason": "initial_velocity_rejected"}
        return inspect_robot_motion(robot, positions[0], targets[0].control_part)
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        return {"valid": False, "unavailable": str(error)}
