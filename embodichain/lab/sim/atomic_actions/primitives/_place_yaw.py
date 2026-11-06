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

"""Failure-only inside release heading recovery; no physical-success claims."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import replace
import math
from typing import Any
from embodichain.compute.geometry.rigid import rotate_held_object_world_z
from embodichain.lab.sim.motion.search import PlanningRandomState, measure_search
from embodichain.utils.math import pose_inv
from ..goals import SceneEntityPose, resolve_pose_goal
from ..plans import ActionPlan
from ..planning_probes import (
    PlaceSceneInputs,
    PlaceInsideLandingProvider,
)
from .place import PlaceGoal
from ._place_branch import inspect_emitted

__all__: list[str] = []

_YAW_FRACTIONS = (0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0)


def recover_inside_yaw(
    action: Any,
    request: Any,
    context: Any,
    first: ActionPlan,
    build: Callable[[Any, Any], ActionPlan],
) -> ActionPlan:
    """Keep previews and successful plans unchanged; reuse Place for each yaw."""
    options = request.skill_options
    if (
        first.plan_success.any()
        or context.is_preview
        or context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or not isinstance(request.goal, PlaceGoal)
        or not isinstance(request.goal.xpos, SceneEntityPose)
        or request.goal.xpos.world_orientation is not None
        or request.goal.tcp_symmetry != "none"
        or options.lift_height != 0.05
        or options.max_approach_retract_z is not None
        or not options.preserve_current_object_orientation
        or request.motion_policy.strategy != "ik_interp"
        or request.motion_policy.dynamic_collision_mode != "off"
    ):
        return first
    provider = action.planning_services.place_scene_provider
    if provider is None:
        raise ValueError("Inside yaw requires its declared Place scene provider.")
    if isinstance(provider, PlaceInsideLandingProvider):
        inputs = provider.prepare_inside_landing(request, context)
    else:
        inputs = provider.prepare(request, context)
    if type(inputs) is not PlaceSceneInputs or inputs.inside_landing is None:
        raise ValueError("Inside yaw requires its exact native landing binding.")
    landing_binding = inputs.inside_landing
    subject = landing_binding.subject_id
    endpoint = request.binding.endpoint("primary", "motion")
    held = context.get_held_object(endpoint.task_state_key)
    if (
        held is None
        or held.semantics.entity_id != subject
        or request.goal.xpos.entity_id != landing_binding.container_id
    ):
        return first
    eef = resolve_pose_goal(request.goal.xpos, context, name="yaw_nominal_release")
    if eef.shape != (1, 4, 4):
        return first
    reference = resolve_pose_goal(
        SceneEntityPose(landing_binding.container_id),
        context,
        name="yaw_container_target",
    )
    parent = resolve_pose_goal(
        SceneEntityPose(landing_binding.parent_id), context, name="yaw_container_parent"
    )
    state = PlanningRandomState.capture((action.robot.device, eef.device))
    attempts: list[dict[str, Any]] = []
    chosen = first
    try:
        for fraction in _YAW_FRACTIONS:
            state.restore()
            target, object_target = rotate_held_object_world_z(
                eef, held.object_to_eef, math.pi * fraction
            )
            landing = object_target.clone()
            landing[:, :3, 3] -= landing_binding.release_clearance * reference[:, :3, 2]
            footprint = landing_binding.geometry.measure(landing, parent)
            fits = [
                inside and -0.01 <= gap <= 0.05
                for inside, gap in zip(
                    footprint["full_footprint_inside"],
                    footprint["support_gap_m"],
                    strict=True,
                )
            ]
            footprint = {
                **footprint,
                "support_tolerance_m": 0.05,
                "penetration_tolerance_m": 0.01,
            }
            row: dict[str, Any] = {
                "angle_deg": 180 * fraction,
                "landing_footprint_valid": all(fits),
            }
            attempts.append(row)
            if not all(fits):
                row["footprint"] = footprint
                continue
            selected = replace(
                request,
                goal=replace(
                    request.goal,
                    xpos=SceneEntityPose(
                        request.goal.xpos.entity_id,
                        relative_pose=pose_inv(reference) @ target,
                        minimum_confidence=request.goal.xpos.minimum_confidence,
                    ),
                ),
            )
            with measure_search("inside_place_yaw_candidate"):
                candidate = build(selected, context)
                inspection = inspect_emitted(
                    action, selected, context, candidate, inputs
                )
            row.update(
                plan_success=candidate.plan_success.tolist(),
                inspection=inspection,
                failure=(
                    None
                    if candidate.diagnostics.failure is None
                    else candidate.diagnostics.failure.code
                ),
            )
            if inspection["valid"]:
                chosen = candidate
                break
    finally:
        state.restore()
    return replace(
        chosen,
        diagnostics=replace(
            chosen.diagnostics,
            metadata={
                **chosen.diagnostics.metadata,
                "inside_world_yaw": {
                    "attempts": attempts,
                    "accepted": chosen is not first,
                    "release_clearance_m": landing_binding.release_clearance,
                    "lift_height_m": options.lift_height,
                    "heights_unchanged": True,
                    "physical_evidence": False,
                },
            },
        ),
    )
