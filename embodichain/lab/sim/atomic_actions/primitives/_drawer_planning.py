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

"""Explicit drawer skill orchestration over shared Motion and Scene services."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import torch
from embodichain.compute.geometry.mesh_bounds import aabb_sphere_cover
from embodichain.compute.trajectory.limits import (
    joint_velocity_validity,
    joint_velocity_diagnostics,
)
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.drawer_ranking import prefilter_drawer_entry_paths
from embodichain.lab.sim.motion.planners.curobo.payload import CuroboPayload
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import CuroboPlanOptions
from ..bindings import JointPositionTarget
from ..goals import SceneEntityPose
from ..plans import PlanningFailure
from ..trajectory_ops import build_pose_plan_states
from .place import PlaceGoal
from .move_held_object import HeldObjectPoseGoal

__all__: list[str] = []


def _inputs(action, request, context, external_nominal):
    if external_nominal or context.batch_size != 1 or context.env_ids.tolist() != [0]:
        raise ValueError(
            "Drawer planning requires B1/env0 and no external nominal source."
        )
    provider = action.planning_services.drawer_planning_provider
    if provider is None:
        raise ValueError("Drawer planning requires its declared physical provider.")
    inputs = provider.inputs(request, context)
    if not inputs:
        raise ValueError("Drawer occurrence has no physical input binding.")
    return provider, inputs


def validate_close(action, request, context, *, external_nominal):
    _, inputs = _inputs(action, request, context, external_nominal)
    target = request.goal.joint_target
    if target is None:
        raise ValueError("Drawer containment requires an explicit articulation target.")
    for item in inputs:
        binding = item.observation.binding
        if (
            binding.object_id != target.articulation_id
            or binding.joint != target.joint_name
            or abs(target.position - binding.target("closed"))
            > max(target.tolerance, 0.01)
        ):
            raise ValueError(
                "Drawer containment binding does not identify this closing target."
            )
        report = item.observation.containment_report(lateral_margin=0.0)
        if not report["accepted"]:
            raise ValueError(
                f"Drawer close rejected: payload containment failed; report={report}"
            )


def _selected_input(action, request, context, external_nominal, target):
    provider, inputs = _inputs(action, request, context, external_nominal)
    if (
        len(inputs) != 1
        or not isinstance(target, SceneEntityPose)
        or target.entity_id != inputs[0].target_id
    ):
        raise ValueError("Drawer motion requires its exact declared scene target.")
    item = inputs[0]
    endpoint = request.binding.endpoint("primary", "motion")
    held = context.task.get_held_object(endpoint.task_state_key)
    if held is None or held.semantics.entity_id != item.observation.object_id:
        raise ValueError("Drawer motion requires the declared object to be held.")
    return provider, item, held


def _candidates(request, context, item, held, release_lift):
    requests = []
    for entry in item.observation.entry_pose_candidates(
        preserve_vertical=request.skill_options.preserve_vertical,
        other_payloads=item.other_payloads,
        release_lift=release_lift,
    ):
        target = torch.stack(entry, dim=1).to(context.robot.qpos)
        requests.append(
            replace(
                request,
                goal=PlaceGoal(
                    target @ held.object_to_eef.to(context.robot.qpos)[:, None],
                    tcp_symmetry=request.goal.tcp_symmetry,
                ),
                skill_options=replace(request.skill_options, lift_height=0.0),
            )
        )
    return tuple(requests)


def _rank(action, request, context, candidates, checker):
    motion = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    seed = context.robot.qpos[:, list(motion.joint_ids)].clone()

    def collisions(stage, qpos):
        full = context.robot.qpos.clone()
        full[:, list(motion.joint_ids)] = qpos
        records = [{"stage": stage, **checker(full)}]
        if stage == "release":
            grasp = request.binding.endpoint("primary", "grasp")
            hand = grasp.require_target(JointPositionTarget)
            full[:, list(hand.joint_ids)] = grasp.joint_positions(
                "open", num_envs=1, device=full.device, dtype=full.dtype
            )
            records.append({"stage": "release_open", **checker(full)})
        return records

    indices, records = prefilter_drawer_entry_paths(
        tuple(candidate.goal.xpos for candidate in candidates),
        seed,
        lambda pose, selected_seed: action.robot.compute_ik(
            pose=pose, name=motion.control_part, joint_seed=selected_seed
        ),
        endpoint_collision=collisions if checker is not None else None,
    )
    return tuple((index, candidates[index]) for index in indices), records


def _velocity(plan, robot):
    trajectory = plan.joint_trajectory
    if trajectory is None:
        if plan.plan_success.any():
            raise ValueError(
                "Drawer planning requires final joint trajectory evidence."
            )
        validity, diagnostics = plan.plan_success.clone(), []
    else:
        limits = authored_joint_velocity_limits(robot, None).to(trajectory.positions)
        validity = joint_velocity_validity(trajectory.positions, trajectory.dt, limits)
        diagnostics = joint_velocity_diagnostics(
            trajectory.positions, trajectory.dt, limits
        )
    accepted = plan.plan_success & validity
    if accepted.all():
        return plan
    return replace(
        plan,
        plan_success=accepted,
        diagnostics=replace(
            plan.diagnostics,
            failure=plan.diagnostics.failure
            or PlanningFailure("drawer_place_velocity_limit", retryable=True),
            metadata={
                **plan.diagnostics.metadata,
                "drawer_final_velocity": {
                    "valid_mask": validity.detach().cpu().tolist(),
                    "diagnostics": diagnostics,
                },
            },
        ),
    )


def _articulation(plan, checker):
    if checker is None or not plan.plan_success.all():
        return plan
    report = checker(plan.joint_trajectory.positions[0])
    return replace(
        plan,
        plan_success=plan.plan_success & report["valid"],
        diagnostics=replace(
            plan.diagnostics,
            metadata={**plan.diagnostics.metadata, "drawer_robot_articulation": report},
            **(
                {
                    "failure": PlanningFailure(
                        "drawer_articulation_collision", retryable=True
                    )
                }
                if not report["valid"]
                else {}
            ),
        ),
    )


def _direct_place(action, request, context):
    reverse_path = torch.cat(
        (request.goal.xpos, request.goal.xpos[:, :2].flip(1)), dim=1
    )

    def generate(_targets, options):
        return action.motion_generator.generate(
            build_pose_plan_states(reverse_path), options=options
        )

    plan = action._plan_place(request, context, motion_build=generate)
    resource = request.binding.endpoint("primary", "motion").task_state_key
    updates = dict(plan.expected_effects.held_object_updates)
    if plan.plan_success.any() and not (
        set(updates) == {resource} and updates[resource] is None
    ):
        plan = replace(
            plan,
            plan_success=torch.zeros_like(plan.plan_success),
            diagnostics=replace(
                plan.diagnostics,
                failure=PlanningFailure(
                    "place_release_contract_missing", retryable=False
                ),
            ),
        )
    if plan.scene_dependencies:
        approach = next(
            (segment for segment in plan.segments if segment.name == "approach"), None
        )
        if approach is not None:
            plan = replace(
                plan,
                scene_dependency_monitor_until={
                    identity: min(
                        approach.stop,
                        plan.scene_dependency_monitor_until.get(
                            identity, approach.stop
                        ),
                    )
                    for identity in plan.scene_dependencies
                },
            )
    return plan


def plan_place(action, request, context, *, build, external_nominal):
    if not isinstance(request.goal, PlaceGoal):
        raise ValueError("Drawer Place requires PlaceGoal.")
    provider, item, held = _selected_input(
        action, request, context, external_nominal, request.goal.xpos
    )
    with ExitStack() as stack:
        checker = None
        if request.skill_options.preserve_vertical:
            query = stack.enter_context(
                provider.prepare_articulation_query(request, context)
            )
            checker = lambda positions: query.inspect(positions).metadata()
        return _place_candidates(action, request, context, build, item, held, checker)


def _place_candidates(action, request, current, build, item, held, checker):
    plans = []
    candidate_results = []
    clearance_attempts = []
    upright = request.skill_options.preserve_vertical
    release_lifts = (0.0, 0.025, 0.05, 0.12) if upright else (0.0, 0.12)
    for release_lift in release_lifts:
        if len(plans) >= 2 or any(bool(p.plan_success.all()) for p in plans):
            break
        all_prepared_candidates = _candidates(
            request, current, item, held, release_lift
        )
        if not all_prepared_candidates:
            raise ValueError("Drawer placement produced no candidate targets.")
        selected_candidates, prefilter_results = _rank(
            action, request, current, all_prepared_candidates, checker
        )
        clearance_attempts.append(
            {
                "release_lift_m": release_lift,
                "prefilter": prefilter_results,
                "target_positions": [
                    p.goal.xpos[:, :, :3, 3].detach().cpu().tolist()
                    for p in all_prepared_candidates
                ],
            }
        )
        if upright:
            clear = tuple(
                (index, candidate)
                for index, candidate in selected_candidates
                if prefilter_results[index]["accepted"]
                and all(
                    record["valid"]
                    for record in prefilter_results[index].get(
                        "articulation_endpoints", ()
                    )
                )
            )
            if clear:
                selected_candidates = clear
            elif release_lift != release_lifts[-1]:
                clearance_attempts[-1][
                    "full_plan_deferred"
                ] = "all_release_endpoints_collide"
                continue
        remaining = 2 - len(plans)
        for index, prepared in selected_candidates[:remaining]:
            plan = build(
                prepared,
                apply_transform=False,
                planner=lambda selected, current_context: _direct_place(
                    action, selected, current_context
                ),
            )
            plan = _velocity(plan, action.robot)
            plan = _articulation(plan, checker)
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "drawer_placement_candidate": index,
                        "drawer_placement_candidate_count": len(
                            all_prepared_candidates
                        ),
                        "drawer_release_lift_m": release_lift,
                    },
                ),
            )
            candidate_results.append(
                {
                    "candidate_id": f"drawer_candidate_{index}",
                    "index": index,
                    "release_lift_m": release_lift,
                    "plan_success": plan.plan_success.detach().cpu().tolist(),
                    "failure": (
                        None
                        if plan.diagnostics.failure is None
                        else plan.diagnostics.failure.code
                    ),
                }
            )
            plans.append(plan)
            if bool(plan.plan_success.all().item()):
                break

    plan = plans[-1]
    selected_index = candidate_results[-1]["index"]
    selected_request = all_prepared_candidates[selected_index]
    plan = build.finalize_selected(selected_request, plan)
    plan = _velocity(plan, action.robot)
    plan = _articulation(plan, checker)
    plan = replace(
        plan,
        diagnostics=replace(
            plan.diagnostics,
            metadata={
                **plan.diagnostics.metadata,
                "drawer_placement_prefilter": prefilter_results,
                "drawer_placement_candidates": candidate_results,
                "drawer_release_clearance_attempts": clearance_attempts,
                "drawer_candidate_target_positions": [
                    prepared.goal.xpos[:, :, :3, 3].detach().cpu().tolist()
                    for prepared in all_prepared_candidates
                ],
            },
        ),
    )
    return plan


def plan_move_held_object(action, request, context, *, build, external_nominal):
    if not isinstance(request.goal, HeldObjectPoseGoal):
        raise ValueError("Drawer transport requires HeldObjectPoseGoal.")
    if (
        request.motion_policy.strategy != "motion_gen"
        or request.skill_options.linear_segments
        or request.skill_options.lift_clearance is not None
        or request.recovery_policy.planning_recovery is not None
    ):
        raise ValueError(
            "Drawer transport requires an explicit direct motion_gen policy."
        )
    provider, item, held = _selected_input(
        action, request, context, external_nominal, request.goal.object_target_pose
    )
    queries = action.planning_services.curobo_queries
    if queries is None:
        raise ValueError("Drawer transport requires observed cuRobo payload queries.")
    observation = item.observation
    target = observation.entry_pose_candidates(
        preserve_vertical=request.skill_options.preserve_vertical,
        other_payloads=item.other_payloads,
    )[0][0].to(context.robot.qpos)
    selection = None
    if request.skill_options.preserve_vertical:
        from embodichain.utils.math import pose_inv

        candidates = tuple(
            replace(
                request,
                goal=PlaceGoal(
                    torch.stack(entry, dim=1).to(context.robot.qpos)
                    @ held.object_to_eef.to(context.robot.qpos)[:, None]
                ),
            )
            for entry in observation.entry_pose_candidates(
                preserve_vertical=True,
                release_lift=0.0,
                other_payloads=item.other_payloads,
            )
        )
        with provider.prepare_articulation_query(request, context) as query:
            ranked, records = _rank(
                action,
                request,
                context,
                candidates,
                lambda positions: query.inspect(positions).metadata(),
            )
        index, candidate = ranked[0]
        target = candidate.goal.xpos[:, 0] @ pose_inv(
            held.object_to_eef.to(context.robot.qpos)
        )
        selection = {
            "candidate": index,
            "prefilter": records,
            "scope": "endpoint_ranking_not_full_place_preview",
        }
    prepared = replace(request, goal=HeldObjectPoseGoal(target))
    motion = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    metadata = {}

    def generate(targets, options):
        # Source transport owns a fresh payload backend for every actual generation.
        payload = CuroboPayload(
            object_id=observation.object_id,
            control_part=motion.control_part,
            spheres=aabb_sphere_cover(torch.as_tensor(observation.vertices)),
            object_pose=observation.payload_pose(),
        )
        native_options = deepcopy(options)
        native_options.strategy = "motion_gen"
        native_options.sample_count = None
        native_options.plan_opts = CuroboPlanOptions()
        with queries.open(
            context=context,
            control_part=motion.control_part,
            full_qpos=context.robot.qpos,
            payload=payload,
            world_metadata=metadata,
        ) as generator:
            result = generator.generate(targets, native_options)
        metadata.update(
            {
                "backend": "curobo",
                "payload": observation.object_id,
                "payload_spheres": 27,
                "world_update": "live_snapshot_per_plan",
                "simulator_attachment": False,
                "articulation_geometry": "link_local_collision_component_boxes",
                "payload_geometry": "object_local_box_sphere_cover",
                "plan_success": result.success.detach().cpu().tolist(),
            }
        )
        return result

    plan = build(
        prepared,
        planner=lambda selected, current: action._plan_transport(
            selected, current, motion_build=generate
        ),
    )
    return replace(
        plan,
        scene_dependencies=tuple(
            dict.fromkeys((*plan.scene_dependencies, item.link_entity_id))
        ),
        diagnostics=replace(
            plan.diagnostics,
            backend="curobo",
            metadata={
                **plan.diagnostics.metadata,
                "drawer_transport": metadata,
                **(
                    {"drawer_transport_heading": selection}
                    if selection is not None
                    else {}
                ),
            },
        ),
    )
