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

"""Bounded joint vias only for canonical released-hand Park scene failures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import (
    JointPositionGoal,
    JointPositionTarget,
    MoveJoints,
)
from embodichain.lab.sim.atomic_actions.control import PARK_COMMAND
from embodichain.lab.sim.atomic_actions.plans import PlanningFailure
from embodichain.lab.task_program.semantics import (
    EffectAssurance,
    RegisteredSemanticCall,
)

from .articulation_collision import robot_collision_scope
from .clearance_recovery import _ParkPreview, _command_trajectory_check, validate_motion
from .held_alignment_recovery import _RandomState, _cuda_devices
from .motion import _joint_velocity_limits

__all__: list[str] = []

PARK_RECOVERY_REVISION = 1
_UNAVAILABLE = (
    ImportError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    TypeError,
)


def eligible(engine: Any, request: Any, context: Any) -> bool:
    """Retain the factory's exact canonical release/cleanup/Park authority."""
    try:
        owns_articulation = getattr(
            engine, "_articulation_park_after_short_withdrawal", lambda _: False
        )(request.invocation_id)
        preview = engine._park_recovery_previews.get(request.invocation_id)
        if not owns_articulation and (
            not isinstance(preview, _ParkPreview)
            or preview.compiler.integration.engine is not engine
        ):
            return False
        endpoint = request.binding.endpoint("primary", "motion")
        if owns_articulation:
            reference, resource = endpoint, endpoint.resource_id
            owned = True
        else:
            bound = preview.workflow.calls[preview.call_index].bound
            call = bound.linked.call
            reference, resource = (
                preview.binding.endpoint("primary", "motion"),
                call.resources["primary"],
            )
            owned = (
                type(call) is RegisteredSemanticCall
                and call.call_id == "simulation.park"
                and not call.arguments
                and set(call.resources) == {"primary"}
                and bound.preset.effect_assurance is EffectAssurance.VERIFIED
            )
        return bool(
            owned
            and isinstance(engine._actions.get(request.skill_id), MoveJoints)
            and request.skill_id == "move_joints"
            and type(request.goal) is JointPositionGoal
            and isinstance(request.goal.target, str)
            and request.goal.target == PARK_COMMAND
            and len(request.binding.endpoints) == 1
            and endpoint.target == reference.target
            and endpoint.task_state_key == reference.task_state_key
            and endpoint.resource_id == resource
            and endpoint.commands[PARK_COMMAND].equivalent_to(
                reference.commands[PARK_COMMAND]
            )
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


def basic_check(
    engine: Any, request: Any, context: Any, plan: Any, *, frames: int | None = None
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
            limits = _joint_velocity_limits(engine.robot, None).to(velocities)[
                context.env_ids
            ]
            if (
                velocities.shape != positions.shape
                or not torch.isfinite(velocities).all()
                or (velocities.abs() > limits[:, None] + 1e-5).any()
            ):
                return {"valid": False, "reason": "velocity_payload_rejected"}
        return _command_trajectory_check(
            plan, context, engine.robot, target.control_part
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


def recover(
    engine: Any,
    request: Any,
    context: Any,
    original: Any,
    *,
    acceptance: dict[str, Any],
    build: Callable[[Any, Any], Any],
    simulation: Any,
) -> Any:
    """Use the same four p32 vias in previews and fresh actual Park planning."""
    if not eligible(engine, request, context):
        return original
    basic = basic_check(engine, request, context, original)
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
        or collision.get("kind") != "scene"
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
    robot = engine.robot
    stop = _RandomState.capture(_cuda_devices(robot.device, context.robot.qpos.device))
    before = robot.get_qpos().detach().clone()
    try:
        with robot_collision_scope():
            endpoint = request.binding.endpoint("primary", "motion")
            target = endpoint.require_target(JointPositionTarget)
            ids, part = list(target.joint_ids), target.control_part
            current = context.robot.qpos[:, ids].clone()
            ready = endpoint.joint_positions(
                PARK_COMMAND, num_envs=1, device=current.device, dtype=current.dtype
            )
            start_pose = (
                robot.compute_fk(qpos=current, name=part, to_matrix=True)
                .detach()
                .clone()
            )
            ready_pose = (
                robot.compute_fk(qpos=ready, name=part, to_matrix=True).detach().clone()
            )
            for kind, height in (("midpoint", 0.05), ("over_ready", 0.10)):
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
                        selected = replace(
                            request,
                            goal=JointPositionGoal(torch.stack((via, ready), dim=1)),
                        )
                        candidate = build(selected, context)
                        check = basic_check(
                            engine, selected, context, candidate, frames=frames
                        )
                        if check["valid"]:
                            check = validate_motion(
                                candidate, context, robot, simulation, part
                            )
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
                    except _UNAVAILABLE as error:
                        entry["reason"] = str(error)
                    finally:
                        if not torch.equal(before, robot.get_qpos()):
                            raise RuntimeError(
                                "Park planning changed native robot state."
                            )
            return _annotate(original, report, failed=True)
    finally:
        stop.restore()
