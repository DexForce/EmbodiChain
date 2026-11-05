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

"""Action-owned goal-seeded Place phase construction and emitted validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING
import torch

from embodichain.compute.trajectory import interpolate_with_distance
from embodichain.compute.trajectory.interpolation import interpolate_velocity_grid
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.goal_seeded import solve_goal_seeded_waypoints
from embodichain.lab.sim.motion.ik_branches import CpuIKBranchSearch
from embodichain.lab.sim.motion.search import PlanningRandomState, record_search
from embodichain.lab.sim.motion.solvers.tolerances import pose_validation_tolerances
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanState
from ..bindings import JointPositionTarget
from ..goals import resolve_pose_goal
from ..plans import ActionPlan, TimedTrajectory
from ..planning_probes import PlaceSceneInputs
from ._pick_path import _pose_path_valid
from ._place_scene import inspect_place_plan

if TYPE_CHECKING:
    from ._place_transport import _ContactPhases

__all__: list[str] = []


@dataclass
class PlaceBranchBuild:
    action: Any
    request: Any
    context: Any
    inputs: PlaceSceneInputs
    groups: list[torch.Tensor] = field(default_factory=list)
    qposes: torch.Tensor | None = None
    receipt: dict[str, Any] = field(default_factory=dict)
    solves: int = 0
    preserved_transport: bool = False
    velocity_phase_allocation: bool = False
    release_branch_index: int = 0
    release_branches: list[torch.Tensor] | None = None
    contact: _ContactPhases | None = None

    def generate(self, down, back, motion_options):
        self.solves += 1
        if self.solves != 1:
            raise ValueError("Place branch permits one required-waypoint solve.")
        self.groups = [down.detach().clone(), back.detach().clone()]
        poses = torch.cat(self.groups, 1)
        part, start = motion_options.control_part, motion_options.start_qpos
        if (
            self.preserved_transport
            and self.release_branches is None
            and CpuIKBranchSearch.supports(self.action.robot, part)
        ):
            search = CpuIKBranchSearch.capture(
                self.action.robot, part, self.context.require_control_dt()
            )
            self.release_branches = [
                q.to(start)
                for _, q in search.endpoints(
                    search.to_root(down[:, -1].to(start)), start
                )
            ]
        selected = None
        if self.release_branches is not None:
            if self.release_branch_index >= len(self.release_branches):
                raise ValueError("No qualified Place release branch remains.")
            selected = self.release_branches[self.release_branch_index]
            self.receipt["release_branch_index"] = self.release_branch_index
            self.receipt["release_branch_count"] = len(self.release_branches)
        ok, qposes, measured = solve_goal_seeded_waypoints(
            self.action.robot,
            motion_options.control_part,
            poses,
            motion_options.start_qpos,
            goal_index=down.shape[1] - 1,
            qualified_goal_seed=selected,
        )
        self.receipt.update(measured)
        if ok.all():
            if self.velocity_phase_allocation:
                from ._place_transport import prepare_contact

                arm = self.request.binding.endpoint("primary", "motion").require_target(
                    JointPositionTarget
                )
                qposes, self.contact = prepare_contact(self, poses, qposes, arm)
            self.qposes = qposes.detach().clone()
        targets = [
            PlanState(move_type=MoveType.JOINT_MOVE, qpos=qposes[:, i])
            for i in range(qposes.shape[1])
        ]
        result = self.action.motion_generator.generate(
            targets,
            options=replace(
                motion_options,
                is_linear=False,
                segment_sample_count_retries=(),
                segment_continuous_ik_recovery=False,
            ),
        )
        result.success &= ok
        return result


def _tolerances(robot, part):
    return pose_validation_tolerances(robot.get_solver(part))


def _phases(plan: ActionPlan) -> tuple[Any, Any, Any]:
    approach, release, retract = plan.segments
    if (
        (approach.name, release.name, retract.name)
        != ("approach", "release", "retract")
        or approach.start != 0
        or approach.stop != release.start
        or release.stop != retract.start
        or retract.stop != plan.joint_trajectory.waypoint_count
    ):
        raise ValueError(
            "Place must retain contiguous approach/release/retract phases."
        )
    return approach, release, retract


def inspect_emitted(
    action: Any, request: Any, context: Any, plan: ActionPlan, inputs: PlaceSceneInputs
) -> dict[str, Any]:
    """Qualify final commands and the exact release target, never physical release."""
    try:
        if not plan.plan_success.all() or plan.joint_trajectory is None:
            return {"valid": False, "reason": "plan_failed"}
        _, release, _ = _phases(plan)
        endpoint = request.binding.endpoint("primary", "motion")
        effects = plan.expected_effects
        if (
            dict(effects.held_object_updates) != {endpoint.task_state_key: None}
            or effects.coordinated_held_object_updates
            or effects.articulation_joint_updates
        ):
            return {"valid": False, "reason": "release_effect_resource_mismatch"}
        checked = inspect_place_plan(action, request, context, plan, inputs)
        if not checked["valid"]:
            return checked
        arm = endpoint.require_target(JointPositionTarget)
        desired = resolve_pose_goal(
            request.goal.xpos, context, name="place_release_target"
        )
        if desired.ndim == 4:
            desired = desired[:, -1]
        elif desired.ndim == 2:
            desired = desired[None]
        actual = action.robot.compute_fk(
            qpos=plan.joint_trajectory.positions[:, release.start, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )
        valid = _pose_path_valid(
            actual, desired, *_tolerances(action.robot, arm.control_part)
        )
        return {
            **checked,
            "valid": valid,
            "release_pose_valid": valid,
            **({} if valid else {"reason": "release_pose_fk_rejected"}),
            "physical_evidence": False,
        }
    except (ValueError, RuntimeError) as error:
        return {
            "valid": False,
            "reason": "emitted_evidence_unavailable",
            "unavailable": str(error),
        }


def _rebuild(scope: PlaceBranchBuild, original: ActionPlan) -> ActionPlan:
    if scope.solves != 1 or scope.qposes is None or not original.plan_success.all():
        raise ValueError("Place shared builder did not qualify every required pose.")
    action, request, context = scope.action, scope.request, scope.context
    arm = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    ids = list(arm.joint_ids)
    approach, release, retract = _phases(original)
    n_down = scope.groups[0].shape[1]
    down, back = scope.qposes[:, :n_down], scope.qposes[:, n_down:]
    anchor = down[:, -1:]
    full = original.joint_trajectory.positions.clone()
    lengths = {s.name: s.stop - s.start for s in original.segments}
    times = original.joint_trajectory.dt
    if scope.contact is not None:
        from ._place_transport import compose

        full, lengths, times = compose(scope, original, arm)
        scope.groups = list(scope.contact.desired)
        scope.qposes = scope.contact.qposes
        n_down = scope.groups[0].shape[1]
        down, back = scope.qposes[:, :n_down], scope.qposes[:, n_down:]
    else:
        if scope.velocity_phase_allocation:
            steps = original.joint_trajectory.dt[:, 1:]
            if (
                not steps.numel()
                or not (steps > 0).all()
                or not (steps == steps[0, 0]).all()
            ):
                raise ValueError(
                    "Post-handover Place requires its uniform control grid."
                )
            limits = authored_joint_velocity_limits(action.robot, None).to(full)[:, ids]
            step = float(steps[0, 0])
            scope.receipt["velocity_phase_intervals"] = {}
        for segment, keys in (
            (approach, torch.cat((context.robot.qpos[:, None, ids], down), 1)),
            (retract, torch.cat((anchor, back), 1)),
        ):
            samples = segment.stop - segment.start
            if scope.velocity_phase_allocation:
                values, intervals = interpolate_velocity_grid(
                    keys, samples, limits, step
                )
                scope.receipt["velocity_phase_intervals"][
                    segment.name
                ] = intervals.tolist()
            else:
                values = interpolate_with_distance(keys, samples, device=full.device)
            full[:, segment.start : segment.stop, ids] = values
        full[:, release.start : release.stop, ids] = anchor
    candidate = action.build_plan(
        request,
        context,
        success=original.plan_success,
        trajectory=TimedTrajectory.from_positions(
            full,
            env_ids=original.joint_trajectory.env_ids,
            dt=times,
        ),
        expected_effects=original.expected_effects,
        effect_candidates=original.effect_candidates,
        effect_verification=original.effect_verification,
        replannable=original.replannable,
        diagnostics=original.diagnostics,
        segment_lengths=lengths,
        scene_dependency_monitor_until=(
            scope.contact.monitor_until
            if scope.contact is not None
            else original.scene_dependency_monitor_until
        ),
        scene_dependency_end_segment=original.scene_dependency_end_segment,
    )
    candidate = replace(candidate, scene_dependencies=original.scene_dependencies)
    approach, release, retract = _phases(candidate)
    # Validate mandatory samples after phase composition, not the discarded split.
    scope.receipt["mandatory_indices"] = {}
    for segment, qkeys, desired in (
        (approach, down, scope.groups[0]),
        (retract, back, scope.groups[1]),
    ):
        values = candidate.joint_trajectory.positions[
            :, segment.start : segment.stop, ids
        ]
        indices, cursor = [], 1
        for key in qkeys[0]:
            matches = (values[0, cursor:] == key).all(-1).nonzero().flatten()
            if not matches.numel():
                raise ValueError("Place emitted phase dropped a mandatory waypoint.")
            index = cursor + int(matches[0])
            indices.append(index)
            cursor = index + 1
        scope.receipt["mandatory_indices"][segment.name] = [
            segment.start + index for index in indices
        ]
        fk = action.robot.compute_batch_fk(
            qpos=values[:, indices], name=arm.control_part, to_matrix=True
        )
        if not _pose_path_valid(
            fk, desired, *_tolerances(action.robot, arm.control_part)
        ):
            raise ValueError("Place emitted mandatory waypoint failed FK.")
    scope.receipt["emitted_mandatory_fk_valid"] = True
    scope.receipt["inspection"] = inspect_emitted(
        action, request, context, candidate, scope.inputs
    )
    if not scope.receipt["inspection"]["valid"]:
        raise ValueError("Place emitted trajectory failed its original safety gates.")
    return candidate


def recover_goal_seeded_place(action, request, context, inputs):
    """Keep a qualified nominal path, otherwise attempt one complete branch."""
    original = action._plan_place(request, context, defer_segment_recovery=True)
    accepted = inspect_emitted(action, request, context, original, inputs)
    if accepted["valid"]:
        return original
    return _plan_branch(action, request, context, inputs, preserved=False)


def recover_preserved_lateral(action, request, context, inputs, rejected):
    """Recover only the declared lateral route's rejected self-collision probe."""
    acceptance = rejected.diagnostics.metadata.get("place_waypoint_probe", {}).get(
        "acceptance", {}
    )
    if (
        rejected.plan_success.any()
        or acceptance.get("reason") != "robot_proxy_rejected"
        or acceptance.get("collision", {}).get("kind") != "self"
    ):
        return rejected
    candidate = _plan_branch(action, request, context, inputs, preserved=True)
    receipt = candidate.diagnostics.metadata["place_branch"]
    chosen = candidate if candidate.plan_success.all() else rejected
    return replace(
        chosen,
        diagnostics=replace(
            chosen.diagnostics,
            metadata={
                **chosen.diagnostics.metadata,
                "place_branch": receipt,
                **(
                    {"place_waypoint_recovery": receipt["inspection"]}
                    if candidate.plan_success.all()
                    else {}
                ),
            },
        ),
    )


def plan_retained_end_grip(action, request, context, inputs):
    """Retain nominal end-grip release; qualify one failed-plan phase reconstruction."""
    from ._place_recovery import PlaceBuildTrace, enforce_continuous_velocity

    trace = PlaceBuildTrace()
    original = action._plan_place(request, context, planning_trace=trace)
    original = enforce_continuous_velocity(
        action, request, context, original, trace.used_continuity
    )
    if original.plan_success.any():
        return original
    return _plan_branch(
        action, request, context, inputs, preserved=False, velocity_allocation=True
    )


def _plan_branch(
    action, request, context, inputs, *, preserved, velocity_allocation=False
):
    state = PlanningRandomState.capture(
        (action.robot.device, context.robot.qpos.device)
    )
    chosen, branches, attempts = None, None, []
    try:
        for index in range(
            request.recovery_policy.planning_recovery.max_release_branches
            if preserved
            else 1
        ):
            record_search("place_branch_complete_plans", budget_checkpoint=False)
            state.restore()
            scope = PlaceBranchBuild(
                action,
                request,
                context,
                inputs,
                preserved_transport=preserved,
                velocity_phase_allocation=preserved or velocity_allocation,
                release_branch_index=index,
                release_branches=branches,
            )
            try:
                chosen = _rebuild(
                    scope, action._plan_place(request, context, branch_build=scope)
                )
            except (ValueError, RuntimeError) as error:
                scope.receipt["unavailable"] = str(error)
            branches = scope.release_branches
            attempts.append(dict(scope.receipt))
            if chosen is not None or branches is None or index + 1 >= len(branches):
                break
    finally:
        state.restore()
    accepted = chosen is not None
    if not accepted:
        chosen = action.failed_plan(
            request,
            context,
            message="Place goal-seeded waypoint branch was not qualified.",
            failure_code="place_branch_unqualified",
            retryable=False,
        )
    return replace(
        chosen,
        diagnostics=replace(
            chosen.diagnostics,
            metadata={
                **chosen.diagnostics.metadata,
                "place_branch": {
                    **scope.receipt,
                    "accepted": accepted,
                    "physical_evidence": False,
                    "scope": "goal_conditioned_waypoint_ik_and_emitted_phase_checks",
                    "mandatory_counts": [p.shape[1] for p in scope.groups],
                    "solve_requests": scope.solves,
                    "retries": len(attempts) - 1,
                    **({"release_branch_attempts": attempts} if preserved else {}),
                    "timing_unchanged": scope.receipt.get("timing_unchanged", True),
                },
            },
        ),
    )
