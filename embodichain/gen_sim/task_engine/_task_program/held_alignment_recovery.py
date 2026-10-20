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

"""Invocation-owned robot-proxy screening for GenSim yaw-free held alignment."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import math
import random
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.plans import PlanningFailure
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
)
from embodichain.lab.sim.atomic_actions.runtime_commands import JointPositionPayload
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanResult, PlanState

from .articulation_collision import check_robot_motion, robot_collision_scope
from .motion import (
    _cartesian_samples,
    _joint_velocity_limits,
    resolve_continuous_ik_path,
    _velocity_validity,
)

__all__: list[str] = []


class AlignmentCollisionUnavailable(ValueError):
    """The current invocation cannot obtain a qualified robot-proxy certificate."""


@dataclass
class _RandomState:
    python: Any
    numpy: Any
    cpu: torch.Tensor
    cuda: dict[int, torch.Tensor]

    @classmethod
    def capture(cls, devices: set[int]) -> _RandomState:
        return cls(
            random.getstate(),
            np.random.get_state(),
            torch.random.get_rng_state(),
            {index: torch.cuda.get_rng_state(index) for index in devices},
        )

    def restore(self) -> None:
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.random.set_rng_state(self.cpu)
        for index, value in self.cuda.items():
            torch.cuda.set_rng_state(value, index)


def _cuda_devices(*values: Any) -> set[int]:
    result = set()
    for value in values:
        device = torch.device(value)
        if device.type == "cuda":
            result.add(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
    return result


@dataclass
class _MotionCall:
    targets: list[PlanState]
    options: Any
    raw_success: bool


@dataclass
class PlanningScope:
    robot: Any
    context: Any
    part: str
    arm_ids: tuple[int, ...]
    hand_ids: tuple[int, ...]
    hand_qpos: torch.Tensor
    generator: Any = None
    calls: list[_MotionCall] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    checkpoint: _RandomState | None = None
    rejected_raw_success: int = 0
    replay_call: _MotionCall | None = None
    replay_result: PlanResult | None = None
    replay_used: bool = False
    fallback_attempted: bool = False

    def inspect(self, positions: torch.Tensor) -> dict[str, Any]:
        devices = _cuda_devices(
            self.context.robot.qpos.device, self.robot.device, positions.device
        )
        state = _RandomState.capture(devices)
        try:
            acceptance = check_robot_motion(self.robot, positions[0], self.part)
        except (
            ImportError,
            ValueError,
            RuntimeError,
            AttributeError,
            KeyError,
            TypeError,
        ) as error:
            raise AlignmentCollisionUnavailable(str(error)) from error
        finally:
            state.restore()
        if type(acceptance) is not dict or type(acceptance.get("valid")) is not bool:
            raise AlignmentCollisionUnavailable(
                "Robot-proxy collision acceptance is unavailable."
            )
        return acceptance

    def full_arm_path(self, positions: torch.Tensor) -> torch.Tensor:
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or positions.shape[1] == 0
            or positions.shape[-1] != len(self.arm_ids)
        ):
            raise AlignmentCollisionUnavailable(
                "Alignment arm trajectory has an unqualified shape."
            )
        full = (
            self.context.robot.qpos.to(positions)[:, None]
            .expand(-1, positions.shape[1], -1)
            .clone()
        )
        full[:, :, list(self.arm_ids)] = positions
        full[:, :, list(self.hand_ids)] = self.hand_qpos.to(full)[:, None]
        return full

    def replay(self, targets: list[PlanState], options: Any) -> PlanResult:
        call, result = self.replay_call, self.replay_result
        if (
            call is None
            or result is None
            or self.replay_used
            or options is None
            or options.strategy != call.options.strategy
            or options.sample_count != call.options.sample_count
            or options.interpolation_dt != call.options.interpolation_dt
            or options.control_part != call.options.control_part
            or options.preserve_cartesian_samples
            != call.options.preserve_cartesian_samples
            or options.is_linear != call.options.is_linear
            or options.velocity_limit != call.options.velocity_limit
            or options.acceleration_limit != call.options.acceleration_limit
            or options.plan_opts is not call.options.plan_opts
            or not torch.equal(options.start_qpos, call.options.start_qpos)
            or len(targets) != len(call.targets)
            or any(
                new.move_type is not old.move_type
                or new.xpos is None
                or new.xpos.shape != old.xpos.shape
                or not torch.allclose(new.xpos, old.xpos, atol=1e-6, rtol=0)
                for new, old in zip(targets, call.targets)
            )
        ):
            raise AlignmentCollisionUnavailable(
                "Alignment recovery target, start or policy changed during replay."
            )
        self.replay_used = True
        return result


_SCOPE: ContextVar[PlanningScope | None] = ContextVar(
    "gen_sim_held_alignment", default=None
)


def motion_scope(generator: Any) -> PlanningScope | None:
    scope = _SCOPE.get()
    return scope if scope is not None and scope.generator is generator else None


@contextmanager
def world_yaw_free_collision_scope(
    robot: Any, request: Any, context: Any
) -> Iterator[PlanningScope | None]:
    """Keep original safe planning streams; roll back only newly added searches."""
    if (
        type(request.goal) is not HeldObjectPoseGoal
        or request.goal.world_yaw_free is not True
    ):
        yield None
        return
    if (
        context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or context.robot.qpos.ndim != 2
    ):
        raise AlignmentCollisionUnavailable(
            "Robot-proxy screening requires one environment at index zero."
        )
    motion = request.binding.endpoint("primary", "motion")
    grasp = request.binding.endpoint("primary", "grasp")
    target = motion.require_target(JointPositionTarget)
    hand = grasp.require_target(JointPositionTarget)
    ids = (*target.joint_ids, *hand.joint_ids)
    if (
        not target.joint_ids
        or not hand.joint_ids
        or len(set(ids)) != len(ids)
        or min(ids) < 0
        or max(ids) >= context.robot.qpos.shape[-1]
        or motion.task_state_key != grasp.task_state_key
    ):
        raise AlignmentCollisionUnavailable(
            "Alignment endpoints do not own a qualified held-hand scope."
        )
    hand_qpos = grasp.joint_positions(
        "grasp",
        num_envs=1,
        device=context.robot.qpos.device,
        dtype=context.robot.qpos.dtype,
    )
    scope = PlanningScope(
        robot, context, target.control_part, target.joint_ids, hand.joint_ids, hand_qpos
    )
    token = _SCOPE.set(scope)
    try:
        with robot_collision_scope():
            yield scope
    finally:
        try:
            if scope.checkpoint is not None:
                scope.checkpoint.restore()
        finally:
            _SCOPE.reset(token)


def screen_alignment_result(
    generator: Any,
    result: PlanResult,
    options: Any,
    *,
    targets: list[PlanState] | None = None,
) -> PlanResult:
    """Reject raw-success robot-proxy paths outside the density-retry loop."""
    scope = _SCOPE.get()
    if scope is None or generator.robot is not scope.robot:
        return result
    if options is None or options.control_part != scope.part:
        raise AlignmentCollisionUnavailable(
            "Alignment motion does not match its bound arm."
        )
    if scope.generator is None:
        scope.generator = generator
    elif scope.generator is not generator:
        raise AlignmentCollisionUnavailable(
            "Alignment used an unexpected motion generator."
        )
    if not isinstance(result.success, torch.Tensor) or result.success.shape != (1,):
        raise AlignmentCollisionUnavailable(
            "Alignment planning did not retain its single-row scope."
        )
    raw_success = bool(result.success.all())
    if targets is not None:
        if options.start_qpos is None or any(
            item.move_type is not MoveType.EEF_MOVE or item.xpos is None
            for item in targets
        ):
            raise AlignmentCollisionUnavailable(
                "Alignment motion cannot preserve its Cartesian targets."
            )
        scope.calls.append(
            _MotionCall(
                [replace(item, xpos=item.xpos.clone()) for item in targets],
                replace(options, start_qpos=options.start_qpos.clone()),
                raw_success,
            )
        )
    if not raw_success:
        scope.attempts.append({"raw_success": False})
        return result
    if result.positions is None:
        raise AlignmentCollisionUnavailable(
            "Successful alignment has no joint trajectory."
        )
    acceptance = scope.inspect(scope.full_arm_path(result.positions))
    scope.attempts.append({"raw_success": True, "collision": acceptance})
    if acceptance["valid"]:
        return result
    scope.rejected_raw_success += 1
    if scope.checkpoint is None:
        scope.checkpoint = _RandomState.capture(
            _cuda_devices(
                scope.context.robot.qpos.device,
                scope.robot.device,
                result.positions.device,
            )
        )
    return replace(result, success=torch.zeros_like(result.success))


def recover(
    scope: PlanningScope | None,
    request: Any,
    context: Any,
    original: Any,
    build: Callable[[Any], Any],
) -> Any:
    """Recover only the last original allowed heading along its unchanged path."""
    if scope is None:
        return original
    report = {
        "robot_proxy_attempts": scope.attempts,
        "rejected_raw_success": scope.rejected_raw_success,
    }
    if (
        original.plan_success.any()
        or (
            not scope.rejected_raw_success
            and any(call.raw_success for call in scope.calls)
        )
        or len(scope.calls) != 8
        or scope.fallback_attempted
    ):
        return _annotate(original, report)
    call = scope.calls[-1]
    options = call.options
    if (
        options.strategy != "ik_interp"
        or options.start_qpos is None
        or type(options.sample_count) is not int
        or options.sample_count < len(call.targets) + 1
        or (
            options.preserve_cartesian_samples
            and len(call.targets) != options.sample_count - 1
        )
        or options.interpolation_dt is None
        or options.interpolation_dt <= 0
        or not isinstance(request.goal.object_target_pose, torch.Tensor)
    ):
        return _annotate(original, {**report, "local_fallback": "unsupported"})
    if scope.checkpoint is None:
        scope.checkpoint = _RandomState.capture(
            _cuda_devices(
                context.robot.qpos.device,
                scope.robot.device,
                options.start_qpos.device,
            )
        )
    scope.fallback_attempted = True
    # Certify the measured hand too, not only the commanded retained grasp.
    initial = scope.inspect(context.robot.qpos[:, None])
    if not initial["valid"]:
        return _annotate(
            original,
            {
                **report,
                "local_fallback": {
                    "accepted": False,
                    "reason": "initial_state_collision",
                    "collision": initial,
                },
            },
        )
    start = scope.robot.compute_fk(
        qpos=options.start_qpos, name=scope.part, to_matrix=True
    )
    targets = (
        call.targets
        if options.preserve_cartesian_samples
        else _cartesian_samples(start, call.targets, options.sample_count)
    )
    calls = 0

    def predicate(qpos: torch.Tensor) -> bool:
        nonlocal calls
        calls += 1
        return scope.inspect(scope.full_arm_path(qpos[:, None]))["valid"]

    result = resolve_continuous_ik_path(
        scope.generator, targets, options, candidate_predicate=predicate
    )
    report["local_fallback"] = {
        "predicate_calls": calls,
        "target_count": len(targets),
        "sample_count": options.sample_count,
        "accepted": False,
    }
    if result is None or not result.success.all():
        return _annotate(original, report)
    if (
        result.positions is None
        or result.positions.shape != (1, options.sample_count, len(scope.arm_ids))
        or result.dt.shape != (1, options.sample_count)
        or not torch.equal(result.positions[:, 0], options.start_qpos)
        or not torch.equal(result.dt[:, 0], result.dt.new_zeros(1))
        or not torch.equal(
            result.dt[:, 1:],
            result.dt.new_full((1, options.sample_count - 1), options.interpolation_dt),
        )
    ):
        raise AlignmentCollisionUnavailable(
            "Local alignment recovery changed its start, frame count or timing."
        )
    poses = request.goal.object_target_pose.clone()
    final = poses[:, -1] if poses.ndim == 4 else poses
    final[:, :3, :3] = (
        final.new_tensor(
            [
                [math.cos(math.pi), -math.sin(math.pi), 0.0],
                [math.sin(math.pi), math.cos(math.pi), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        @ final[:, :3, :3]
    )
    selected = replace(
        request,
        goal=replace(request.goal, object_target_pose=poses, world_yaw_free=False),
    )
    scope.replay_call, scope.replay_result, scope.replay_used = call, result, False
    try:
        candidate = build(selected)
        candidate = qualify_alignment_plan(selected, context, candidate, scope.robot)
        if candidate.plan_success.all() and scope.replay_used:
            report["local_fallback"]["accepted"] = True
            report["local_fallback"]["world_yaw_offset_rad"] = math.pi
            return _annotate(candidate, report)
        return _annotate(original, report)
    finally:
        scope.replay_call, scope.replay_result = None, None


def _annotate(plan: Any, report: dict[str, Any]) -> Any:
    return replace(
        plan,
        diagnostics=replace(
            plan.diagnostics,
            metadata={
                **plan.diagnostics.metadata,
                "gen_sim_held_alignment": {
                    **plan.diagnostics.metadata.get("gen_sim_held_alignment", {}),
                    **report,
                },
            },
        ),
    )


def qualify_alignment_plan(request: Any, context: Any, plan: Any, robot: Any) -> Any:
    """Recheck emitted held-arm commands after full composition or a transform."""
    if not plan.plan_success.any():
        return plan
    acceptance: dict[str, Any] = {"valid": False}
    try:
        targets = tuple(
            request.binding.endpoint("primary", name).require_target(
                JointPositionTarget
            )
            for name in ("motion", "grasp")
        )
        trajectory = plan.joint_trajectory
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or trajectory is None
            or trajectory.positions.ndim != 3
            or trajectory.positions.shape[0] != 1
            or trajectory.positions.shape[-1] != context.robot.qpos.shape[-1]
            or trajectory.positions.shape[1] == 0
            or trajectory.dt.shape != trajectory.positions.shape[:2]
            or not torch.isfinite(trajectory.positions).all()
            or not torch.isfinite(trajectory.dt).all()
            or (trajectory.dt < 0).any()
        ):
            raise AlignmentCollisionUnavailable(
                "Unqualified full alignment trajectory."
            )
        positions, dt = trajectory.positions, trajectory.dt
        active_ids = [index for target in targets for index in target.joint_ids]
        if (
            len(set(active_ids)) != len(active_ids)
            or min(active_ids) < 0
            or max(active_ids) >= positions.shape[-1]
        ):
            raise AlignmentCollisionUnavailable(
                "Unqualified alignment control endpoints."
            )
        passive = [
            index for index in range(positions.shape[-1]) if index not in active_ids
        ]
        if passive and not torch.equal(
            positions[:, :, passive],
            context.robot.qpos[:, None, passive].expand(-1, positions.shape[1], -1),
        ):
            raise AlignmentCollisionUnavailable(
                "Alignment changed an uncommanded joint."
            )
        hand_qpos = request.binding.endpoint("primary", "grasp").joint_positions(
            "grasp", num_envs=1, device=positions.device, dtype=positions.dtype
        )
        if not torch.equal(
            positions[:, :, list(targets[1].joint_ids)],
            hand_qpos[:, None].expand(-1, positions.shape[1], -1),
        ):
            raise AlignmentCollisionUnavailable(
                "Alignment changed the retained grasp command."
            )
        if not torch.equal(
            positions[:, 0, list(targets[0].joint_ids)],
            context.robot.qpos[:, list(targets[0].joint_ids)],
        ):
            raise AlignmentCollisionUnavailable(
                "Alignment changed its zero-time arm start."
            )
        limits = _joint_velocity_limits(robot, None).to(positions)[context.env_ids]
        if not _velocity_validity(positions, dt, limits).all():
            raise AlignmentCollisionUnavailable(
                "Alignment violates full-robot joint velocity limits."
            )
        for target in targets:
            bounds = robot.get_qpos_limits(
                name=target.control_part, env_ids=context.env_ids
            ).to(positions)
            if (
                bounds.shape != (1, len(target.joint_ids), 2)
                or torch.isnan(bounds).any()
                or (bounds[..., 0] > bounds[..., 1]).any()
            ):
                raise AlignmentCollisionUnavailable(
                    "Alignment joint limits are unavailable."
                )
            values = positions[:, :, list(target.joint_ids)]
            if (
                (values < bounds[:, None, :, 0] - 1e-5)
                | (values > bounds[:, None, :, 1] + 1e-5)
            ).any():
                raise AlignmentCollisionUnavailable(
                    "Alignment violates an endpoint joint limit."
                )
        if (
            plan.commands.frame_count != positions.shape[1]
            or not torch.equal(plan.commands.env_ids, context.env_ids)
            or not torch.equal(trajectory.env_ids, context.env_ids)
        ):
            raise AlignmentCollisionUnavailable(
                "Alignment command/trajectory scope differs."
            )
        for index, frame in enumerate(plan.commands.frames):
            if (
                len(frame.commands) != len(targets)
                or not torch.equal(frame.env_ids, context.env_ids)
                or not torch.equal(frame.active_mask, plan.plan_success)
                or not torch.equal(
                    frame.hold_duration, dt[:, min(index + 1, positions.shape[1] - 1)]
                )
            ):
                raise AlignmentCollisionUnavailable(
                    "Alignment emitted a different command schedule."
                )
            seen = set()
            for command in frame.commands:
                if (
                    type(command.target) is not JointPositionTarget
                    or command.target not in targets
                    or command.target.control_part in seen
                    or type(command.payload) is not JointPositionPayload
                ):
                    raise AlignmentCollisionUnavailable(
                        "Alignment emitted an unexpected command target."
                    )
                seen.add(command.target.control_part)
                ids = list(command.target.joint_ids)
                if not torch.equal(command.payload.positions, positions[:, index, ids]):
                    raise AlignmentCollisionUnavailable(
                        "Alignment command positions differ from its trajectory."
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
                    raise AlignmentCollisionUnavailable(
                        "Alignment command velocities are unqualified."
                    )
        state = _RandomState.capture(
            _cuda_devices(context.robot.qpos.device, robot.device, positions.device)
        )
        try:
            collision = check_robot_motion(robot, positions[0], targets[0].control_part)
        finally:
            state.restore()
        if type(collision) is not dict or type(collision.get("valid")) is not bool:
            raise AlignmentCollisionUnavailable(
                "Alignment collision certificate is unavailable."
            )
        acceptance.update({"valid": collision["valid"], "collision": collision})
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        acceptance["reason"] = str(error)
    details = dict(plan.diagnostics.metadata.get("gen_sim_held_alignment", {}))
    details["final_command_check"] = acceptance
    return replace(
        plan,
        plan_success=(
            plan.plan_success
            if acceptance["valid"]
            else torch.zeros_like(plan.plan_success)
        ),
        diagnostics=replace(
            plan.diagnostics,
            metadata={**plan.diagnostics.metadata, "gen_sim_held_alignment": details},
            **(
                {
                    "failure": PlanningFailure(
                        "held_alignment_safety_failed", retryable=False
                    )
                }
                if not acceptance["valid"]
                else {}
            ),
        ),
    )
