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
"""Fresh E8 pre-contact suffix planning from a separate issued command seed."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from embodichain.gen_sim.task_engine._task_program.motion import (
    _joint_velocity_limits,
    _velocity_validity,
)
from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    JointPositionTarget,
    OPEN_COMMAND,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    interpolate_hand_qpos,
    resolve_pose_target,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import (
        PlanningContext,
        ResolvedActionRequest,
        TwistGoal,
        TwistOptions,
    )
    from .twist_runtime import GenSimTwist

__all__ = ["plan_continuation"]

_LOCAL_SAMPLES = 16
_ALIGNMENT_HOLD_STEPS = 12


def _fixed_pose(
    value: np.ndarray, bias: np.ndarray, reference: torch.Tensor
) -> torch.Tensor:
    pose = np.array(value, dtype=float, copy=True)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Continuation requires a finite nominal TCP pose.")
    pose[:3, 3] += bias
    return torch.as_tensor(pose, device=reference.device, dtype=torch.float32)[None]


def plan_continuation(
    action: GenSimTwist,
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    context: PlanningContext,
    continuation: Any,
    *,
    command_seed: torch.Tensor,
    bias: np.ndarray,
) -> ActionPlan:
    """Replan a checked pre-contact correction and its complete turn suffix.

    The instance-owned caller owns observation freshness, no-contact admission,
    correction budgets and the original deadline. This planner does not replace
    the native context with a commanded state or perform simulation writes.
    A same-tick scene joint observation is mandatory; diagnostics expose it for
    the caller's final geometry qualification on an independently copied request.
    """
    if context.batch_size != 1 or request.revision < 1:
        raise ValueError("Continuation requires one environment and a new revision.")
    phase = continuation.resume_phase
    if phase not in {"reach", "close"}:
        raise ValueError("Continuation is restricted to reach/close entry.")
    bias = np.array(bias, dtype=float, copy=True)
    if bias.shape != (3,) or not np.isfinite(bias).all():
        raise ValueError("Continuation bias must be a finite world-frame vector.")
    if np.linalg.norm(bias) > 0.020 + 1e-9:
        raise ValueError("Continuation bias exceeds the existing 20 mm bound.")
    observed = context.robot.qpos
    if (
        not isinstance(command_seed, torch.Tensor)
        or command_seed.shape != observed.shape
        or not bool(torch.isfinite(command_seed).all())
        or not bool(torch.isfinite(observed).all())
        or float((command_seed - observed).abs().max()) > 0.05
    ):
        raise ValueError("Continuation command seed exceeds native observation bounds.")
    qpos = command_seed.clone()
    motion = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    grasp = request.binding.endpoint("primary", "grasp")
    hand = grasp.require_target(JointPositionTarget)
    arm_ids, hand_ids = list(motion.joint_ids), list(hand.joint_ids)
    if tuple(continuation.hand_ids) != tuple(hand.joint_ids):
        raise ValueError("Continuation must retain the original grasp endpoint.")
    closed = continuation.closed_qpos
    if (
        not isinstance(closed, torch.Tensor)
        or closed.shape != qpos.shape
        or not bool(torch.isfinite(closed).all())
    ):
        raise ValueError("Continuation requires the original finite closed target.")
    open_hand = grasp.joint_positions(
        OPEN_COMMAND,
        num_envs=context.batch_size,
        device=qpos.device,
        dtype=qpos.dtype,
    )
    closed_hand = closed[:, hand_ids].to(qpos).clone()
    grasp_pose = _fixed_pose(continuation.desired, bias, qpos)
    standoff_pose = _fixed_pose(continuation.desired_standoff, bias, qpos)

    # PlanningContext's joint accessor is verified symbolic state, not a live
    # reading. Revisions use only the owned scene observation from this tick.
    geometry = dict(request.goal.semantics.geometry)
    articulation_id = geometry["gen_sim_twist_articulation_id"]
    joint = geometry["gen_sim_twist_joint"]
    affordance = action._require_twist_affordance(request.goal.semantics)
    if affordance.joint_name != joint or affordance.joint_limits is None:
        raise ValueError("Continuation requires its exact source joint and limits.")
    if (
        not math.isfinite(context.scene.timestamp)
        or not math.isfinite(context.robot.timestamp)
        or abs(context.scene.timestamp - context.robot.timestamp) > 1e-9
    ):
        raise ValueError("Continuation requires a same-tick scene joint observation.")
    state = context.scene.get_articulation_joint_state(articulation_id, joint)
    if state is None:
        raise ValueError("Continuation requires a fresh scene joint angle.")
    position, valid = state.position, state.valid_mask
    if (
        not isinstance(position, torch.Tensor)
        or position.shape not in {(1,), (1, 1)}
        or not position.is_floating_point()
        or not bool(torch.isfinite(position).all())
        or (
            valid is not None
            and (
                not isinstance(valid, torch.Tensor)
                or valid.shape != (1,)
                or valid.dtype != torch.bool
                or not bool(valid.all())
            )
        )
    ):
        raise ValueError("Continuation scene joint observation is invalid/incomplete.")
    from .twist_runtime import _bounded_twist_qpos

    current = _bounded_twist_qpos(position.reshape(-1)[0], affordance.joint_limits)
    if current is None:
        raise ValueError("Continuation scene joint angle exceeds its source limits.")
    sign = geometry["gen_sim_twist_axis_sign"]
    if not math.isfinite(sign) or sign == 0:
        raise ValueError("Continuation requires a finite nonzero axis sign.")
    geometry["gen_sim_twist_measured_qpos"] = float(current)
    angle = (geometry["gen_sim_twist_target_qpos"] - current) / sign
    maximum = geometry.get("gen_sim_twist_chunk_angle", math.pi)
    if not math.isfinite(angle) or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Continuation requires a finite positive angular chunk bound.")
    angle = math.copysign(min(abs(angle), maximum), angle)
    if abs(angle) > math.pi:
        raise ValueError("Continuation turn exceeds the existing pi-radian bound.")
    revised = replace(
        request,
        goal=replace(
            request.goal, semantics=replace(request.goal.semantics, geometry=geometry)
        ),
        skill_options=replace(request.skill_options, twist_angle=angle),
    )
    options = revised.skill_options
    dt = context.require_control_dt()
    _, n_reach, n_twist, n_retract = action._motion_segment_lengths(
        request.motion_policy.sample_count, options.hand_interp_steps
    )
    settle = geometry["gen_sim_twist_settle_steps"]
    if type(settle) is not int or settle < 0:
        raise ValueError("Continuation settling steps must be a nonnegative integer.")
    target_pose = resolve_pose_target(
        resolve_pose_goal(request.goal.target_pose, context, name="target_pose"),
        num_envs=context.batch_size,
        device=qpos.device,
    )
    twist_poses = action._twisted_grasp_poses(
        target_pose,
        grasp_pose,
        affordance.twist_axis,
        affordance.require_axis_origin(),
        angle,
        options.twist_waypoint_count,
    )
    parts: list[torch.Tensor] = []
    lengths: dict[str, int] = {}
    start = qpos[:, arm_ids].clone()
    success = torch.ones(context.batch_size, dtype=torch.bool, device=qpos.device)

    def append_arm(
        name: str, target: torch.Tensor, count: int, hand_qpos: torch.Tensor, hold: int
    ) -> None:
        nonlocal start, success
        good, arm = action._plan_pose_segment(
            target, start, motion.control_part, revised, count, interpolation_dt=dt
        )
        success &= good
        if arm.shape != (context.batch_size, count, len(arm_ids)):
            raise ValueError("Continuation arm planner returned an unexpected shape.")
        if not bool(torch.allclose(arm[:, 0], start, atol=1e-6, rtol=0)):
            raise ValueError("Continuation arm path must preserve its command seed.")
        if hold:
            arm = torch.cat((arm, arm[:, -1:].expand(-1, hold, -1)), dim=1)
        full = qpos[:, None].expand(-1, arm.shape[1], -1).clone()
        full[:, :, arm_ids] = arm
        full[:, :, hand_ids] = hand_qpos[:, None]
        parts.append(full)
        lengths[name] = full.shape[1]
        start = arm[:, -1].clone()

    if phase == "reach":
        append_arm(
            "approach", standoff_pose, _LOCAL_SAMPLES, open_hand, _ALIGNMENT_HOLD_STEPS
        )
    append_arm(
        "reach",
        grasp_pose,
        n_reach if phase == "reach" else _LOCAL_SAMPLES,
        open_hand,
        _ALIGNMENT_HOLD_STEPS,
    )

    def append_hand(
        name: str, beginning: torch.Tensor, ending: torch.Tensor, hold: int
    ) -> None:
        hand_path = interpolate_hand_qpos(
            beginning, ending, n_waypoints=options.hand_interp_steps
        )
        if hold:
            hand_path = torch.cat(
                (hand_path, hand_path[:, -1:].expand(-1, hold, -1)), dim=1
            )
        full = qpos[:, None].expand(-1, hand_path.shape[1], -1).clone()
        full[:, :, arm_ids] = start[:, None]
        full[:, :, hand_ids] = hand_path
        parts.append(full)
        lengths[name] = full.shape[1]

    append_hand("close", open_hand, closed_hand, settle)
    append_arm("twist", twist_poses, n_twist, closed_hand, settle)
    append_hand("open", closed_hand, open_hand, 0)
    append_arm("retract", standoff_pose, n_retract, open_hand, 0)
    positions = torch.cat(parts, dim=1)
    positions[:, 0] = qpos
    if not bool(success.all()):
        raise ValueError(
            "Continuation IK rejected the complete correction/turn/retract path."
        )
    trajectory = TimedTrajectory.from_uniform_step(
        positions, env_ids=context.env_ids, step_dt=dt
    )
    if not bool(
        _velocity_validity(
            trajectory.positions,
            trajectory.dt,
            _joint_velocity_limits(action.robot, None).to(positions),
        ).all()
    ):
        raise ValueError(
            "Continuation exceeds existing full-robot joint velocity limits."
        )
    metadata = deepcopy(continuation.selected_metadata)
    metadata.update(
        gen_sim_twist_continuation=True,
        continuation_phase=phase,
        continuation_local_samples=_LOCAL_SAMPLES,
        continuation_angle_source="scene_observed_joint",
        continuation_joint_observed_at=context.scene.timestamp,
        continuation_measured_qpos=geometry.get("gen_sim_twist_measured_qpos"),
        continuation_angle_rad=angle,
        continuation_bias_m=bias.tolist(),
        native_context_preserved=True,
        continuation_start_kind="separate_last_issued_command_seed",
    )
    return action.build_plan(
        revised,
        context,
        success=success,
        trajectory=trajectory,
        expected_effects=StateDelta(),
        diagnostics=PlannerDiagnostics(
            backend=action.planning_services.planner_name, metadata=metadata
        ),
        segment_lengths=lengths,
        scene_dependency_end_segment="reach",
    )
