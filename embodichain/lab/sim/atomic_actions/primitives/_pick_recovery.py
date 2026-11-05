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

"""Shared-budget ordinary Pick candidates after the qualified first path fails."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from embodichain.lab.sim.motion.search import PlanningRandomState
from ..bindings import JointPositionTarget
from ._pick_adaptive import PickPlanningState
from ._pick_path import qualify_pick_path, _free_motion
from ._pick_candidate_path import geometric_variants, trajectory
from ._manipulation_checks import inspect_manipulation_plan

__all__: list[str] = []


def _qualify_candidate(action, request, context, state, pose, direction):
    exact, receipt = trajectory(action, request, context, pose, direction)
    record = {"trajectory": receipt}
    if exact is None:
        return None, record
    prior = state.request, state.trajectory, state.direction
    state.request, state.trajectory, state.direction = request, exact, direction
    try:
        plan = action._finish_native_plan(
            request, context, action._plan_pick(request, context)
        )
        record["emitted_plan"] = inspect_manipulation_plan(
            action.robot, request, context, plan
        )
        if not plan.plan_success.all() or not record["emitted_plan"]["valid"]:
            return None, record
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        stop = next(
            segment.stop for segment in plan.segments if segment.name == "approach"
        )
        record["approach_scene"] = _free_motion(
            action, plan.joint_trajectory.positions[0, :stop], arm.control_part
        )
        if not record["approach_scene"]["valid"]:
            return None, record
        from ._place_continuation import prepare_pick_continuation, preview_place

        preview = prepare_pick_continuation(action, request, context)
        if preview is not None and preview.route.requires_full_preview:
            record["place_preview"] = preview_place(
                action, request, context, plan, preview, safety=True
            )
            if not record["place_preview"]["valid"]:
                return None, record
        return plan, record
    finally:
        state.request, state.trajectory, state.direction = prior


def retry_filtered_candidates(
    action, request: Any, context: Any, first: Any, selection: PickPlanningState
) -> Any:
    """Replan at most four already-filtered grasps with their own approach."""
    budget = max(0, 4 - selection.depth_recovery_attempts)
    if first.plan_success.any() or not selection.candidate_routes or not budget:
        return first

    key = request.binding.endpoint("primary", "motion").task_state_key
    original = first.expected_effects.held_object_updates.get(key)
    seen = [] if original is None else [original.grasp_xpos]
    sampled = first.diagnostics.metadata.get("affordance_sample", {}).get("grasp", {})
    if sampled.get("selected_poses") is not None:
        seen.append(torch.as_tensor(sampled["selected_poses"], device=action.device))
    state = PlanningRandomState.capture((action.device, action.robot.device))
    prior = (selection.request, selection.trajectory, selection.direction)
    attempts: list[dict[str, Any]] = []
    chosen = first

    def candidates():
        deferred = []
        for strategy, pool, direction in selection.candidate_routes:
            order = torch.argsort(
                pool.costs[0].masked_fill(~pool.valid[0], torch.inf), stable=True
            )
            for index in order[pool.valid[0, order]].tolist():
                pose = pool.poses[:, index]
                if any(torch.equal(pose, value) for value in seen):
                    continue
                # Spend the bounded retry budget on different wrist branches
                # before millimetre perturbations of the same grasp.
                similar = any(
                    bool(
                        torch.linalg.vector_norm(
                            pose[:, :3, 3] - value[:, :3, 3], dim=-1
                        ).amax()
                        < 0.01
                    )
                    and bool(
                        (
                            (pose[:, :3, :3].transpose(-1, -2) @ value[:, :3, :3])
                            .diagonal(dim1=-2, dim2=-1)
                            .sum(-1)
                            .amin()
                            - 1
                        )
                        / 2
                        > 0.9396926208
                    )
                    for value in seen
                )
                if similar:
                    deferred.append((strategy, index, pose.clone(), direction.clone()))
                    continue
                seen.append(pose)
                yield strategy, index, pose.clone(), direction.clone()
        yield from deferred

    try:
        for strategy, index, pose, direction in candidates():
            if len(attempts) >= budget:
                break
            state.restore()
            selected = replace(
                request,
                goal=replace(request.goal, grasp_xpos=pose),
                skill_options=replace(
                    request.skill_options, approach_direction=direction[0]
                ),
            )
            selection.request, selection.trajectory, selection.direction = (
                selected,
                None,
                direction,
            )
            candidate = action._finish_native_plan(
                selected, context, action._plan_pick(selected, context)
            )
            candidate = qualify_pick_path(
                action, selected, context, candidate, selection
            )
            candidate = action._finish_native_plan(selected, context, candidate)
            record = {
                "strategy": strategy,
                "candidate_id": index,
                "selected_pose": pose.detach().cpu().tolist(),
                "approach_direction": direction.detach().cpu().tolist(),
                "path_success": candidate.plan_success.tolist(),
                "failure_code": (
                    None
                    if candidate.diagnostics.failure is None
                    else candidate.diagnostics.failure.code
                ),
            }
            attempts.append(record)
            if not candidate.plan_success.all():
                continue
            from ._place_continuation import prepare_pick_continuation, preview_place

            preview = prepare_pick_continuation(action, selected, context)
            if preview is not None and preview.route.requires_full_preview:
                record["place_preview"] = preview_place(
                    action, selected, context, candidate, preview, safety=True
                )
                if not record["place_preview"]["valid"]:
                    continue
            samples = dict(candidate.diagnostics.metadata.get("affordance_sample", {}))
            samples["grasp"] = {
                "candidate_ids": [index],
                "selected_poses": pose.detach().cpu().tolist(),
                "adaptive_grasp": {
                    "strategy_indices": [strategy],
                    "approach_directions": direction.tolist(),
                },
            }
            chosen = replace(
                candidate,
                diagnostics=replace(
                    candidate.diagnostics,
                    metadata={
                        **candidate.diagnostics.metadata,
                        "affordance_sample": samples,
                    },
                ),
            )
            break
    finally:
        selection.request, selection.trajectory, selection.direction = prior
        state.restore()
    return replace(
        chosen,
        diagnostics=replace(
            chosen.diagnostics,
            metadata={
                **chosen.diagnostics.metadata,
                "pick_candidate_retry": {
                    "budget": 4,
                    "attempts": attempts,
                    "accepted": bool(chosen.plan_success.all()),
                },
            },
        ),
    )


def retry_raw_candidates(action, request, context, first, selection):
    """Qualify legal geometry discarded by the coarse endpoint-IK screen."""
    from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

    previous = first.diagnostics.metadata.get("pick_candidate_retry", {})
    budget = max(
        0, 4 - len(previous.get("attempts", ())) - selection.depth_recovery_attempts
    )
    if first.plan_success.any() or budget == 0:
        return first
    state = PlanningRandomState.capture((action.device, action.robot.device))
    prior = selection.request, selection.trajectory, selection.direction
    attempts = []
    chosen = first
    seen = [
        torch.as_tensor(item["selected_pose"], device=action.device)
        for item in previous.get("attempts", ())
        if item.get("selected_pose") is not None
    ]
    try:
        routes = []
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        start = action.robot.compute_fk(
            qpos=context.robot.qpos[:, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )
        for strategy, raw, direction in selection.raw_candidate_routes:
            pool = geometric_variants(
                action,
                raw,
                request,
                context,
                direction,
            )
            rotation = quat_from_matrix(pool.poses[0, :, :3, :3])
            rotation_error = quat_error_magnitude(
                rotation, quat_from_matrix(start[0, :3, :3]).expand_as(rotation)
            )
            # Preserve shared Pick's nearest-roll preference within each
            # geometric cost, without discarding the other legal branch.
            preferred = torch.argsort(rotation_error, stable=True)
            order = torch.argsort(
                pool.costs[0, preferred].masked_fill(
                    ~pool.valid[0, preferred], torch.inf
                ),
                stable=True,
            )
            order = preferred[order]
            routes.append(
                (
                    strategy,
                    pool,
                    direction,
                    iter(order[pool.valid[0, order]].tolist()),
                )
            )
        # Share the same bounded budget across declared approach strategies.
        while routes and len(attempts) < budget:
            remaining = []
            for strategy, pool, direction, indices in routes:
                if len(attempts) >= budget:
                    break
                index = next(indices, None)
                while index is not None:
                    pose = pool.poses[:, index]
                    duplicate = any(torch.equal(pose, value) for value in seen)
                    if not duplicate and bool(
                        action._approach_alignment_mask(
                            pose[:, None, None],
                            request.skill_options,
                            direction[:, None, None],
                        ).all()
                    ):
                        break
                    index = next(indices, None)
                if index is None:
                    continue
                remaining.append((strategy, pool, direction, indices))
                if len(attempts) >= budget:
                    break
                pose = pool.poses[:, index].clone()
                seen.append(pose)
                state.restore()
                selected = replace(
                    request,
                    goal=replace(request.goal, grasp_xpos=pose),
                    skill_options=replace(
                        request.skill_options,
                        approach_direction=direction[0],
                        rotate_upright=None,
                        obj_upright_direction=None,
                    ),
                )
                candidate, qualification = _qualify_candidate(
                    action, selected, context, selection, pose, direction
                )
                record = {
                    "strategy": strategy,
                    "candidate_id": index // 2,
                    "roll_variant": index % 2,
                    "selected_pose": pose.detach().cpu().tolist(),
                    **qualification,
                }
                attempts.append(record)
                if candidate is None:
                    continue
                chosen = replace(
                    candidate,
                    diagnostics=replace(
                        candidate.diagnostics,
                        metadata={
                            **candidate.diagnostics.metadata,
                            "affordance_sample": {
                                "grasp": {
                                    "candidate_ids": [index // 2],
                                    "selected_poses": pose.detach().cpu().tolist(),
                                    "adaptive_grasp": {
                                        "strategy_indices": [strategy],
                                        "roll_variants": [index % 2],
                                        "approach_directions": direction.tolist(),
                                    },
                                }
                            },
                        },
                    ),
                )
                break
            if chosen.plan_success.all():
                break
            routes = remaining
    except (
        ImportError,
        ValueError,
        RuntimeError,
        AttributeError,
        KeyError,
        TypeError,
    ) as error:
        attempts.append({"unavailable": str(error)})
    finally:
        selection.request, selection.trajectory, selection.direction = prior
        state.restore()
    return replace(
        chosen,
        diagnostics=replace(
            chosen.diagnostics,
            metadata={
                **chosen.diagnostics.metadata,
                "pick_raw_candidate_retry": {
                    "budget": budget,
                    "attempts": attempts,
                    "accepted": bool(chosen.plan_success.all()),
                    "source": "geometry_before_endpoint_ik",
                    "qualification_scope": "continuous_pick_full_robot_limits_and_approach_scene_discrete_meshes",
                },
            },
        ),
    )
