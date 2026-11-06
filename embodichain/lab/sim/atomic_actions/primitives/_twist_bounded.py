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

"""Bounded source-compatible Twist candidate planning on the canonical skill."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, TYPE_CHECKING
from uuid import uuid4
import torch

from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.search import measure_search
from ..core import PlanQualification
from ..effects import StateDelta
from ..goals import SceneEntityPose
from ..plans import ActionPlan, TimedTrajectory, PlannerDiagnostics, PlanningFailure
from ..state import PlanningContext
from ..invocation import ResolvedActionRequest
from ..twist_planning import TwistPlanningSession, TwistGeometryQuery
from .twist_recovery import TwistPlanningRecovery, TwistPlanningOutcome

if TYPE_CHECKING:
    from .twist import Twist, TwistGoal, TwistOptions

__all__: list[str] = []


def candidate_table(
    bite: float, fixed_roll: float | None
) -> tuple[tuple[int, float, float], ...]:
    rolls = (
        (float(fixed_roll),)
        if fixed_roll is not None
        else (0.0, math.pi / 2, -math.pi / 2, math.pi)
    )
    values = [
        (roll, (fraction - 1.0) * bite)
        for fraction in (1.0 / 3.0, 2.0 / 3.0, 1.0)
        for roll in rolls
    ]
    return tuple((index, roll, shift) for index, (roll, shift) in enumerate(values))


def _preflight(
    action: Twist,
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    context: PlanningContext,
) -> float:
    rule = request.recovery_policy.planning_recovery
    if type(rule) is not TwistPlanningRecovery:
        raise TypeError("Bounded Twist requires TwistPlanningRecovery.")
    if (
        context.batch_size != 1
        or action.num_envs != 1
        or action.robot.num_instances != 1
        or context.robot.qpos.dtype != torch.float32
        or type(request.invocation_id) is not str
        or not request.invocation_id
        or context.env_ids.tolist() != [0]
        or request.goal.joint_target is None
        or request.goal.contact_reference is None
        or type(request.goal.target_pose) is not SceneEntityPose
        or request.goal.target_pose.entity_id != request.goal.semantics.entity_id
        or request.goal.target_pose.relative_pose is not None
        or request.goal.target_pose.world_displacement is not None
        or request.goal.target_pose.world_orientation is not None
        or request.skill_options.contact_bite_depth is None
        or request.motion_policy.sample_count < rule.minimum_sample_count
        or request.motion_policy.strategy != "ik_interp"
        or request.motion_policy.interpolation_mode != "fixed_grid"
        or request.motion_policy.ik_seed_mode != "sequential"
        or request.motion_policy.dynamic_collision_mode != "off"
        or request.recovery_policy.max_replans != 0
        or request.recovery_policy.max_action_retries != 0
        or any(
            count > request.motion_policy.sample_count
            for count in request.motion_policy.sample_count_retries
        )
    ):
        raise ValueError(
            "Bounded Twist requires native float32 B1/env0, an observed joint target and bite, fixed-grid sequential IK with at least 320 samples, and no outer replans/retries."
        )
    if action.planning_services.twist_planning_provider is None:
        raise ValueError("Bounded Twist requires an explicit planning provider.")
    affordance = action._require_twist_affordance(request.goal.semantics)
    if (
        tuple(
            torch.tensor(request.goal.contact_reference, dtype=torch.float32).tolist()
        )
        != affordance.grasp_position
    ):
        raise ValueError("Twist contact reference differs from the nominal affordance.")
    angle, _ = action._joint_displacement(
        request.goal.joint_target,
        affordance,
        context,
        request.skill_options.max_twist_angle,
    )
    return float(angle[0]) if request.skill_options.rotation_enabled else 0.0


def _prepare(
    action: Twist,
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    context: PlanningContext,
) -> TwistPlanningSession:
    session = action.planning_services.twist_planning_provider.prepare(request, context)
    if type(session) is not TwistPlanningSession:
        raise TypeError("Twist provider must return exactly TwistPlanningSession.")
    return session


def _check_session(
    action: Twist,
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    context: PlanningContext,
    session: TwistPlanningSession,
) -> None:
    scope = session.query.scope
    if (
        session.query.robot is not action.robot
        or scope.observation_timestamp != context.robot.timestamp
        or scope.scene_version != context.scene.version
        or scope.env_id != 0
        or session.state.scope.env_id != 0
        or session.invocation_id != request.invocation_id
        or session.invocation_revision != request.revision
    ):
        raise ValueError("Twist query and cursor must match the exact B1 observation.")
    if (
        session.state.last_feedback_timestamp is not None
        and session.state.last_feedback_timestamp > context.robot.timestamp + 1e-9
    ):
        raise ValueError(
            "Twist planning observation precedes its adopted measured feedback."
        )
    if (
        request.goal.semantics.entity_id == session.parent_entity_id
        or session.parent_entity_id != request.goal.joint_target.articulation_id
        or session.parent_entity_id not in context.scene.entities
    ):
        raise ValueError(
            "Twist parent must be the explicitly observed articulation dependency."
        )
    axis = request.goal.semantics.affordance.twist_axis
    if not torch.equal(axis, axis.new_tensor(session.query.source_axis)):
        raise ValueError("Twist affordance differs from the source-calibrated axis.")


def _metadata(plan: ActionPlan, **values: Any) -> ActionPlan:
    return replace(
        plan,
        diagnostics=replace(
            plan.diagnostics, metadata={**plan.diagnostics.metadata, **values}
        ),
    )


def _free_frames(plan: ActionPlan) -> tuple[int, ...]:
    if plan.joint_trajectory is None:
        raise ValueError("Twist qualification requires a joint trajectory.")
    count = plan.joint_trajectory.positions.shape[1]
    frames = []
    for name in ("approach", "reach", "retract"):
        matched = [segment for segment in plan.segments if segment.name == name]
        if len(matched) != 1 or not 0 <= matched[0].start < matched[0].stop <= count:
            raise ValueError("Twist requires complete free-motion phases.")
        frames.extend(range(matched[0].start, matched[0].stop))
    if len(frames) != len(set(frames)):
        raise ValueError("Twist free-motion phases overlap.")
    return tuple(frames)


def _filter_candidate(
    action: Twist,
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    context: PlanningContext,
    plan: ActionPlan,
    query: TwistGeometryQuery,
) -> ActionPlan:
    if not plan.plan_success.any():
        return plan
    report = query.inspect_free_path(
        plan.joint_trajectory.positions, _free_frames(plan)
    )
    if not report.accepted:
        plan = action.failed_plan(
            request,
            context,
            message="E8 candidate failed table free-motion clearance.",
            failure_code="e8_table_clearance_rejected",
            retryable=False,
        )
    return _metadata(plan, gen_sim_twist_table_clearance=report.metadata())


def _root_dependency(
    plan: ActionPlan,
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    query: TwistGeometryQuery,
    parent_id: str,
) -> ActionPlan:
    if not plan.plan_success.any():
        return plan
    affordance = request.goal.semantics.affordance
    report = query.qualify_root_dependency(
        plan.joint_trajectory.positions,
        grasp_point=affordance.grasp_position,
        axis=affordance.twist_axis,
        roll=affordance.grasp_roll,
    )
    child = request.goal.semantics.entity_id
    changed = report.qualified and child in plan.scene_dependencies
    if changed:
        dependencies = tuple(
            sorted(
                {
                    parent_id if name == child else name
                    for name in plan.scene_dependencies
                }
            )
        )
        cutoff = dict(plan.scene_dependency_monitor_until)
        first, second = cutoff.pop(child, None), cutoff.pop(parent_id, None)
        if first is not None and (
            parent_id not in plan.scene_dependencies or second is not None
        ):
            cutoff[parent_id] = max(first, second or 0)
        plan = replace(
            plan, scene_dependencies=dependencies, scene_dependency_monitor_until=cutoff
        )
    evidence = (
        {"qualified": True, "changed": False}
        if report.qualified and not changed
        else {**report.metadata(), "changed": changed}
    )
    return _metadata(plan, gen_sim_twist_root_dependency=evidence)


def _candidate_request(
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    roll: float,
    shift: float,
    axis: tuple[float, float, float],
) -> ResolvedActionRequest[TwistGoal, TwistOptions]:
    affordance = request.goal.semantics.affordance
    point = tuple(
        p + shift * a for p, a in zip(request.goal.contact_reference, axis, strict=True)
    )
    return replace(
        request,
        goal=replace(
            request.goal,
            semantics=replace(
                request.goal.semantics,
                affordance=replace(
                    affordance,
                    grasp_position=point,
                    grasp_roll=roll,
                    grasp_roll_range=None,
                ),
            ),
        ),
    )


def plan_bounded_twist(
    action: Twist,
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    context: PlanningContext,
) -> ActionPlan:
    angle = _preflight(action, request, context)
    session = _prepare(action, request, context)
    try:
        _check_session(action, request, context, session)
        state = session.state
        if state.goal_complete:
            targets = session.completed_hold_qpos
            if targets is None or targets.shape != context.robot.qpos.shape:
                raise ValueError(
                    "A completed Twist cursor requires observed drive-target holds."
                )
            plan = action.build_plan(
                request,
                context,
                success=torch.ones(1, dtype=torch.bool, device=action.device),
                trajectory=TimedTrajectory.from_uniform_step(
                    targets[:, None].to(context.robot.qpos),
                    env_ids=context.env_ids,
                    step_dt=context.require_control_dt(),
                ),
                expected_effects=StateDelta(),
                diagnostics=None,
                segment_lengths={"approach": 1},
            )
            plan = _root_dependency(
                plan, request, session.query, session.parent_entity_id
            )
            # Even a completed-operation hold must be adopted under its own
            # occurrence before new physical feedback can be committed. This
            # proposal changes no candidate or measured success state.
            outcome = TwistPlanningOutcome(
                "twist",
                plan.invocation_id,
                plan.invocation_revision,
                uuid4().hex,
                replace(state, version=state.version + 1),
            )
            return replace(plan, planning_recovery_outcome=outcome)
        bite = request.skill_options.contact_bite_depth
        table = candidate_table(bite, request.skill_options.contact_roll)
        candidates = tuple(
            item
            for item in table
            if state.forced_index is None or item[0] == state.forced_index
        )
        if (
            len(candidates)
            > request.recovery_policy.planning_recovery.maximum_contact_candidates
        ):
            raise ValueError("Twist candidate table exceeds its declared budget.")
        if not candidates:
            raise ValueError(
                "Twist forced candidate is absent from the declared table."
            )
        axis = session.query.source_axis
        base_request = replace(
            request,
            goal=replace(request.goal, joint_target=None),
            skill_options=replace(request.skill_options, twist_angle=angle),
        )
        ranked, first_failed = [], None
        for index, roll, shift in candidates:
            selected = _candidate_request(base_request, roll, shift, axis)
            with measure_search("twist_candidate", budget_checkpoint=False):
                plan = action._plan_nominal(selected, context)
            plan = _metadata(
                plan, gen_sim_twist_roll=roll, gen_sim_twist_roll_candidate=index
            )
            plan = _filter_candidate(action, selected, context, plan, session.query)
            if first_failed is None:
                first_failed = index, plan, selected
            if plan.plan_success.any():
                close = plan.segment("close")
                parent_fraction, pad_distance, motion = session.query.contact_metrics(
                    plan.joint_trajectory.positions[:, close.stop - 1],
                    context.robot.qpos,
                )
                score = (
                    float(parent_fraction > 0.05),
                    parent_fraction,
                    pad_distance,
                    motion,
                )
                if not all(math.isfinite(value) for value in score):
                    raise ValueError("Twist candidate score must be finite.")
                ranked.append((score, index, plan, selected))
        by_index = {item[1]: item for item in ranked}
        if ranked:
            feasible = sorted(
                (item for item in table if item[0] in by_index),
                key=lambda item: (item[2], by_index[item[0]][0], item[0]),
            )
            chosen_roll = feasible[0][1]
            order = tuple(index for index, roll, _ in feasible if roll == chosen_roll)
            score, index, plan, selected = by_index[order[0]]
            if state.forced_index is None:
                state = replace(state, order=order, candidates=tuple(feasible))
            elif state.candidates:
                state = replace(
                    state,
                    order=tuple(i for i, r, _ in state.candidates if r == chosen_roll),
                )
            roll, shift = {i: (r, s) for i, r, s in table}[index]
            plan = _metadata(
                plan,
                gen_sim_twist_candidate_order=list(state.order),
                gen_sim_twist_candidate_score=list(score),
                gen_sim_twist_reference_bite=bite,
                gen_sim_twist_candidate_bite=bite + shift,
                gen_sim_twist_fixed_roll=roll,
                gen_sim_twist_candidate_fallback_limit=2,
                gen_sim_twist_parent_contact_score_limit=0.05,
            )
        else:
            index, plan, selected = first_failed
            roll, shift = {i: (r, s) for i, r, s in table}[index]
        state = replace(
            state,
            version=state.version + 1,
            selected_index=index,
            selected_roll=roll,
            selected_bite_depth=bite + shift,
        )
        # Selection is retained even if settle/velocity/root qualification fails.
        if plan.plan_success.any():
            settle = request.skill_options.contact_settle_steps
            if settle:
                trajectory = plan.joint_trajectory
                parts, lengths = [], {}
                for segment in plan.segments:
                    part = trajectory.positions[:, segment.start : segment.stop]
                    if segment.name in {"close", "twist"}:
                        part = torch.cat((part, part[:, -1:].expand(-1, settle, -1)), 1)
                    parts.append(part)
                    lengths[segment.name] = part.shape[1]
                plan = action.build_plan(
                    selected,
                    context,
                    success=plan.plan_success,
                    trajectory=TimedTrajectory.from_uniform_step(
                        torch.cat(parts, 1),
                        env_ids=trajectory.env_ids,
                        step_dt=(
                            float(trajectory.dt[0, 1].item())
                            if trajectory.dt.shape[1] > 1
                            else context.require_control_dt()
                        ),
                    ),
                    expected_effects=plan.expected_effects,
                    diagnostics=plan.diagnostics,
                    segment_lengths=lengths,
                )
            trajectory = plan.joint_trajectory
            limits = authored_joint_velocity_limits(action.robot, None).to(
                trajectory.positions
            )
            if not joint_velocity_validity(
                trajectory.positions, trajectory.dt, limits
            ).all():
                plan = action.failed_plan(
                    selected,
                    context,
                    message="E8 Twist exceeds resampled joint velocity limits.",
                )
            elif plan.scene_dependencies:
                plan = replace(plan, scene_dependency_end_segment="reach")
        plan = _root_dependency(plan, selected, session.query, session.parent_entity_id)
        outcome = TwistPlanningOutcome(
            "twist", plan.invocation_id, plan.invocation_revision, uuid4().hex, state
        )
        return replace(plan, planning_recovery_outcome=outcome)
    finally:
        session.query.close()


def qualify_transformed_twist(
    action: Twist,
    request: ResolvedActionRequest[TwistGoal, TwistOptions],
    context: PlanningContext,
    plan: ActionPlan,
) -> PlanQualification:
    _preflight(action, request, context)
    if not plan.plan_success.any():
        return PlanQualification(plan.plan_success, plan.diagnostics)
    session = _prepare(action, request, context)
    try:
        _check_session(action, request, context, session)
        # A transformed completed hold has no contact phases to qualify.
        if session.state.goal_complete:
            valid = session.completed_hold_qpos is not None and torch.equal(
                plan.joint_trajectory.positions,
                session.completed_hold_qpos[:, None].to(
                    plan.joint_trajectory.positions
                ),
            )
            report = {
                "supported": bool(valid),
                "accepted": bool(valid),
                "scope": "unchanged_completed_hold",
            }
        else:
            result = session.query.inspect_free_path(
                plan.joint_trajectory.positions, _free_frames(plan)
            )
            valid = result.accepted
            report = result.metadata()
            if plan.diagnostics.metadata.get("gen_sim_twist_root_dependency", {}).get(
                "changed"
            ):
                outcome = plan.planning_recovery_outcome
                if type(outcome) is not TwistPlanningOutcome:
                    raise ValueError(
                        "Transformed Twist lost its selected candidate outcome."
                    )
                table = {
                    i: (r, s)
                    for i, r, s in candidate_table(
                        request.skill_options.contact_bite_depth,
                        request.skill_options.contact_roll,
                    )
                }
                roll, shift = table[outcome.next_state.selected_index]
                selected = _candidate_request(
                    request, roll, shift, session.query.source_axis
                )
                affordance = selected.goal.semantics.affordance
                root = session.query.qualify_root_dependency(
                    plan.joint_trajectory.positions,
                    grasp_point=affordance.grasp_position,
                    axis=affordance.twist_axis,
                    roll=affordance.grasp_roll,
                )
                valid = valid and root.qualified
                report["root_dependency"] = root.metadata()
        positions, dt = plan.joint_trajectory.positions, plan.joint_trajectory.dt
        valid = bool(valid) and bool(
            joint_velocity_validity(
                positions,
                dt,
                authored_joint_velocity_limits(action.robot, None).to(positions),
            ).all()
        )
        mask = plan.plan_success & valid
        return PlanQualification(
            mask,
            PlannerDiagnostics(
                backend=plan.diagnostics.backend,
                metadata={"twist_transformed_path": report},
                failure=(
                    None
                    if valid
                    else PlanningFailure(
                        "twist_transformed_path_rejected", retryable=False
                    )
                ),
            ),
        )
    finally:
        session.query.close()
