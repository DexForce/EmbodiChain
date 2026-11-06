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

"""Failure-only ordinary Place retries with source budgets and RNG boundaries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING
import torch
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.lab.sim.motion.solvers.tolerances import pose_validation_tolerances
from ..bindings import JointPositionTarget
from ..planning_probes import PlaceSceneInputs
from ..goals import SceneEntityPose, resolve_pose_goal
from ..trajectory_ops import resolve_pose_target
from ._manipulation_checks import inspect_robot_motion
from ._pick_path import _pose_path_valid
from ._place_scene import inspect_place_plan

__all__: list[str] = []

if TYPE_CHECKING:
    from .place import PlaceOptions


def _matches_waypoint_recovery_scope(options: PlaceOptions, variant: str) -> bool:
    """Share the existing declarative waypoint rule with static preflight."""
    return options.qualify_required_waypoints == (variant in {"waypoint", "lateral"})


@dataclass
class PlaceBuildTrace:
    kinematic_mask: torch.Tensor | None = None
    used_continuity: bool = False


def prepare_recovery_inputs(
    action: Any, request: Any, context: Any
) -> PlaceSceneInputs:
    from .place import PlaceGoal

    options = request.skill_options
    variant = request.recovery_policy.planning_recovery.variant
    goal_seeded = variant == "goal_seeded_on"
    if (
        context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or not isinstance(request.goal, PlaceGoal)
        or request.goal.tcp_symmetry != "none"
        or options.preserve_current_object_orientation == goal_seeded
        or options.max_approach_retract_z is not None
        or not options.linear_segments
        or not _matches_waypoint_recovery_scope(options, variant)
        or (
            goal_seeded
            and (
                not isinstance(request.goal.xpos, SceneEntityPose)
                or request.goal.xpos.world_orientation is not None
            )
        )
        or request.motion_policy.strategy != "ik_interp"
        or request.motion_policy.dynamic_collision_mode != "off"
    ):
        raise ValueError(
            "Place waypoint recovery requires its declared B1 orientation/waypoint scope and Cartesian ik_interp/off."
        )
    resource = request.binding.endpoint("primary", "motion").task_state_key
    held = context.get_held_object(resource)
    if held is None or not context.task.exclusive_held_object_mask(resource).all():
        raise ValueError("Place waypoint recovery requires an exclusively held object.")
    target, _ = action._resolve_place_xpos(
        request.goal, context, resource, request=request
    )
    if target.shape not in ((1, 4, 4), (1, 1, 4, 4)):
        raise ValueError(
            "Place waypoint recovery currently requires one required release pose."
        )
    provider = action.planning_services.place_scene_provider
    if provider is None:
        raise ValueError(
            "Place waypoint recovery requires its declared scene input provider."
        )
    inputs = provider.prepare(request, context)
    if type(inputs) is not PlaceSceneInputs:
        raise TypeError("Place scene provider must return owned PlaceSceneInputs.")
    if (
        held.semantics.entity_id not in inputs.geometry
        or inputs.geometry.unavailable
        or held.semantics.entity_id in {inputs.floor_entity_id, inputs.support_owner_id}
    ):
        raise ValueError(
            "Place recovery requires distinct complete held/floor/support input geometry."
        )
    if goal_seeded and inputs.support_owner_id is None:
        raise ValueError(
            "Goal-seeded on-Place requires an explicit intended support owner."
        )
    if request.recovery_policy.planning_recovery.variant == "lateral":
        if (
            options.cartesian_waypoint_count != 2
            or not isinstance(request.goal.xpos, SceneEntityPose)
            or inputs.support_owner_id is not None
        ):
            raise ValueError(
                "Lateral Place recovery requires a preserved two-waypoint scene-relative release without support exclusion."
            )
        if action.planning_services.curobo_queries is None:
            raise ValueError(
                "Lateral Place recovery requires its observed payload query service."
            )
    if variant == "retained_end_grip":
        if (
            not isinstance(request.goal.xpos, SceneEntityPose)
            or inputs.support_owner_id is not None
        ):
            raise ValueError(
                "Retained end-grip Place requires its scene-relative release without support exclusion."
            )
        if (
            options.cartesian_waypoint_count == 2
            and action.planning_services.curobo_queries is None
        ):
            raise ValueError(
                "Retained end-grip Cartesian contact requires its observed payload query service."
            )
    return inputs


def enforce_continuous_velocity(
    action: Any, request: Any, context: Any, plan: Any, used: bool
) -> Any:
    if used and plan.plan_success.any():
        trajectory = plan.joint_trajectory
        limits = authored_joint_velocity_limits(action.robot, None).to(
            trajectory.positions
        )
        if not joint_velocity_validity(
            trajectory.positions, trajectory.dt, limits
        ).all():
            failed = action.failed_plan(
                request,
                context,
                message="Recovered Place exceeds final resampled joint velocity limits.",
            )
            return replace(
                failed,
                diagnostics=replace(
                    failed.diagnostics, metadata=dict(plan.diagnostics.metadata)
                ),
            )
    return plan


def recover_place(
    action: Any, request: Any, context: Any, inputs: PlaceSceneInputs
) -> Any:
    """Retain successful nominal motion and reproduce the two fallback batches."""
    rule = request.recovery_policy.planning_recovery
    builds = 0
    waypoint_batches = 0
    routes = 0

    def build(selected=request, *, waypoint=False, defer=False):
        nonlocal builds
        builds += 1
        if builds > rule.max_complete_plans:
            raise RuntimeError(
                "Place recovery exceeded its declared complete-plan budget."
            )
        trace = PlaceBuildTrace()
        result = action._plan_place(
            selected,
            context,
            waypoint_recovery=waypoint,
            defer_segment_recovery=defer,
            planning_trace=trace,
        )
        return result, trace

    def waypoint_batch(first):
        nonlocal waypoint_batches
        if first.plan_success.any():
            return first
        waypoint_batches += 1
        if waypoint_batches > rule.max_waypoint_batches:
            raise RuntimeError(
                "Place recovery exceeded its declared waypoint-batch budget."
            )
        state = PlanningRandomState.capture(
            (action.robot.device, context.robot.qpos.device)
        )
        attempts = []
        evidence = []
        chosen = None
        acceptance = None
        try:
            for _ in range(rule.waypoint_attempts_per_batch):
                candidate, trace = build(waypoint=True)
                if trace.kinematic_mask is not None:
                    evidence.append(trace.kinematic_mask)
                measured = inspect_place_plan(
                    action, request, context, candidate, inputs
                )
                attempts.append(measured)
                if not measured["valid"]:
                    continue
                score = (measured["rotation_max_rad"], measured["rotation_path_rad"])
                if acceptance is None or score < (
                    acceptance["rotation_max_rad"],
                    acceptance["rotation_path_rad"],
                ):
                    chosen, acceptance = candidate, measured
                if measured["orientation_preserved"]:
                    break
            if chosen is None:
                if not evidence:
                    return first
                return replace(
                    first,
                    diagnostics=replace(
                        first.diagnostics,
                        metadata={
                            **first.diagnostics.metadata,
                            "place_waypoint_probe": {
                                "kinematic_valid": any(bool(v.all()) for v in evidence),
                                "acceptance": attempts[-1],
                                "attempts": attempts,
                            },
                        },
                    ),
                )
            return replace(
                chosen,
                diagnostics=replace(
                    chosen.diagnostics,
                    metadata={
                        **chosen.diagnostics.metadata,
                        "place_waypoint_recovery": {**acceptance, "attempts": attempts},
                    },
                ),
            )
        finally:
            state.restore()

    def finish(result):
        return replace(
            result,
            diagnostics=replace(
                result.diagnostics,
                metadata={
                    **result.diagnostics.metadata,
                    "place_recovery_cost": {
                        "complete_plans": builds,
                        "waypoint_batches": waypoint_batches,
                        "clearance_routes": routes,
                        "scope": "one_complete_action_planning_attempt",
                    },
                },
            ),
        )

    before = PlanningRandomState.capture(
        (action.robot.device, context.robot.qpos.device)
    )
    plan, trace = build(defer=True)
    used = trace.used_continuity
    early = None
    if not plan.plan_success.any():
        candidate = waypoint_batch(plan)
        if rule.variant == "lateral":
            from ._place_branch import recover_preserved_lateral

            candidate = recover_preserved_lateral(
                action, request, context, inputs, candidate
            )
        if candidate.plan_success.all():
            early = candidate
        receipt = candidate.diagnostics.metadata.get("place_waypoint_recovery", {})
        if candidate.plan_success.all() and receipt["rotation_max_rad"] <= torch.pi / 4:
            plan = candidate
        else:
            before.restore()
            plan, trace = build()
            used = trace.used_continuity
    if not plan.plan_success.any():
        plan = waypoint_batch(plan)
    checked = enforce_continuous_velocity(action, request, context, plan, used)
    if checked is not plan:
        if early is None:
            return finish(checked)
        plan = early
    if not plan.plan_success.any():
        endpoint = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        xpos = resolve_pose_target(
            resolve_pose_goal(request.goal.xpos, context, name="place_candidate_xpos"),
            num_envs=1,
            device=action.device,
        )
        if xpos.ndim == 3:
            xpos = xpos[:, None]
        start = action.robot.compute_fk(
            qpos=context.robot.qpos[:, list(endpoint.joint_ids)],
            name=endpoint.control_part,
            to_matrix=True,
        )
        if inspect_robot_motion(
            action.robot, context.robot.qpos, endpoint.control_part
        )["valid"]:
            tolerances = pose_validation_tolerances(
                action.robot.get_solver(endpoint.control_part)
            )
            state = PlanningRandomState.capture(
                (action.robot.device, context.robot.qpos.device)
            )
            attempts = []
            first = plan
            try:
                for clearance in rule.clearance_routes:
                    state.restore()
                    routes += 1
                    via = start.clone()
                    via[:, :2, 3] = (start[:, :2, 3] + xpos[:, 0, :2, 3]) / 2
                    via[:, 2, 3] = (
                        torch.maximum(start[:, 2, 3], xpos[:, :, 2, 3].amax(1))
                        + clearance
                    )
                    selected = replace(
                        request,
                        goal=replace(
                            request.goal, xpos=torch.cat((via[:, None], xpos), 1)
                        ),
                    )
                    candidate, _ = build(selected)
                    acceptance = inspect_place_plan(
                        action, selected, context, candidate, inputs
                    )
                    record = {"clearance_m": clearance, "acceptance": acceptance}
                    attempts.append(record)
                    if not acceptance["valid"]:
                        continue
                    release = next(s for s in candidate.segments if s.name == "release")
                    actual = action.robot.compute_fk(
                        qpos=candidate.joint_trajectory.positions[
                            :, release.start, list(endpoint.joint_ids)
                        ],
                        name=endpoint.control_part,
                        to_matrix=True,
                    )
                    record["release_pose_valid"] = _pose_path_valid(
                        actual, xpos[:, -1], *tolerances
                    )
                    if record["release_pose_valid"]:
                        plan = replace(
                            candidate,
                            scene_dependencies=first.scene_dependencies,
                            scene_dependency_monitor_until=first.scene_dependency_monitor_until,
                            scene_dependency_end_segment=first.scene_dependency_end_segment,
                        )
                        break
            finally:
                state.restore()
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "place_candidate_retry": {
                            "budget": len(rule.clearance_routes),
                            "attempts": attempts,
                            "accepted": bool(plan.plan_success.all()),
                            "release_pose_unchanged": True,
                        },
                    },
                ),
            )
    if early is not None:
        measured = plan.diagnostics.metadata.get("place_waypoint_recovery")
        previous = early.diagnostics.metadata["place_waypoint_recovery"]
        if not plan.plan_success.any() or (
            measured is not None
            and (measured["rotation_max_rad"], measured["rotation_path_rad"])
            > (previous["rotation_max_rad"], previous["rotation_path_rad"])
        ):
            plan = early
    return finish(plan)
