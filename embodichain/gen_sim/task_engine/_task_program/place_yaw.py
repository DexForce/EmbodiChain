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

"""Explicit world-heading freedom for failed, observed rigid inside releases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import math
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose, resolve_pose_goal
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal
from embodichain.lab.task_program.semantics import Pick, Place, SceneAffordanceRef
from embodichain.utils.math import pose_inv

from ._search_cost import measure_search
from .alignment_continuation import _PREVIEW
from .held_alignment_recovery import _RandomState, _cuda_devices
from .motion import _CARTESIAN_APPROACH
from .place_branch import inspect_emitted
from .placement_evidence import _prepare_placement_geometry, measure_planar_placement

__all__: list[str] = []

# Keep the existing MoveHeldObject heading order, not a task-selected angle.
_YAW_FRACTIONS = (0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0)


def declarations(value: object) -> tuple[str, ...]:
    """Validate an explicit permission, never infer freedom from inside alone."""
    if (
        type(value) not in (list, tuple)
        or any(
            type(name) is not str or not name or name != name.strip() for name in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError("World yaw freedom requires unique exact Place names.")
    return tuple(value)


def bind_calls(value: object, program: Any) -> tuple[str, ...]:
    """Resolve declared ordinary inside segments before runtime assembly."""
    remaining = set(declarations(value))
    result = []
    for segment in program.iter_segments():
        if segment.name not in remaining:
            continue
        if len(segment.calls) != 1:
            raise ValueError("World yaw freedom requires one exact inside Place.")
        call = segment.calls[0]
        if (
            type(call.call) is not Place
            or type(call.call.inside) is not SceneAffordanceRef
        ):
            raise ValueError("World yaw freedom requires one exact inside Place.")
        result.append(
            f"{program.program_id}/{segment.segment_id}:{call.segment_call_index}"
        )
        remaining.remove(segment.name)
    if remaining:
        raise ValueError("World yaw freedom references a missing Place.")
    return tuple(result)


def bind_routes(factory: Any) -> dict[str, tuple[Any, ...]]:
    """Bind declared same-object/resource releases to measured rigid geometry."""
    selected = set(factory._world_yaw_free_places)
    bindings = factory.task_program_registration.scene_binding
    rigid = {item.entity_id: item for item in bindings.rigid_objects}
    containers = {item.entity_id: item for item in bindings.containers}
    purposes = dict(factory._pick_purposes)
    routes = {}
    for route in factory._pick_continuations:
        if route.place_id not in selected:
            continue
        pick, place = route.pick_call, route.place_call
        if (
            type(pick) is not Pick
            or type(place) is not Place
            or purposes.get(route.pick_id) != "ordinary"
            or type(place.inside) is not SceneAffordanceRef
            or place.object.entity_id != pick.object.entity_id
            or dict(place.resources) != dict(pick.resources)
            or set(place.resources) != {"primary"}
        ):
            continue
        container = containers.get(place.inside.entity_id)
        subject = place.object.entity_id
        if (
            container is None
            or container.parent_id not in rigid
            or subject not in rigid
            or container.release_clearance != 0.12
        ):
            continue
        object_binding, reference_binding = rigid[subject], rigid[container.parent_id]
        geometry = _prepare_placement_geometry(
            factory._simulation, object_binding, reference_binding, 1
        )
        routes[route.place_id] = (
            subject,
            container,
            object_binding,
            reference_binding,
            geometry,
        )
    if selected != routes.keys():
        raise ValueError(
            "World yaw freedom requires a bound ordinary rigid inside route."
        )
    return routes


def rotated_target(
    eef: torch.Tensor, held: torch.Tensor, angle: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the object about world Z, keeping its origin and held transform."""
    object_pose = eef @ pose_inv(held)
    yaw = eef.new_tensor(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    changed = object_pose.clone()
    changed[:, :3, :3] = yaw @ object_pose[:, :3, :3]
    return changed @ held, changed


def recover(
    action: Any,
    request: Any,
    context: Any,
    first: ActionPlan,
    build: Callable[[Any, Any], ActionPlan],
) -> ActionPlan:
    """Keep previews and successful plans unchanged; reuse Place for each yaw."""
    route = (action.world_yaw_free_routes or {}).get(request.invocation_id)
    options = request.skill_options
    if (
        first.plan_success.any()
        or route is None
        or _PREVIEW.get()
        or _CARTESIAN_APPROACH.get()
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
        or getattr(action.motion_generator, "direct_place_planning", False)
    ):
        return first
    subject, container, object_binding, reference_binding, geometry = route
    endpoint = request.binding.endpoint("primary", "motion")
    held = context.get_held_object(endpoint.task_state_key)
    if (
        held is None
        or held.semantics.entity_id != subject
        or request.goal.xpos.entity_id != container.entity_id
    ):
        return first
    eef = resolve_pose_goal(request.goal.xpos, context, name="yaw_nominal_release")
    if eef.shape != (1, 4, 4):
        return first
    reference = resolve_pose_goal(
        SceneEntityPose(container.entity_id), context, name="yaw_container_target"
    )
    parent = resolve_pose_goal(
        SceneEntityPose(container.parent_id), context, name="yaw_container_parent"
    )
    state = _RandomState.capture(_cuda_devices(action.robot.device, eef.device))
    attempts: list[dict[str, Any]] = []
    chosen = first
    try:
        for fraction in _YAW_FRACTIONS:
            state.restore()
            target, object_target = rotated_target(
                eef, held.object_to_eef, math.pi * fraction
            )
            landing = object_target.clone()
            landing[:, :3, 3] -= container.release_clearance * reference[:, :3, 2]
            fits, footprint = measure_planar_placement(
                action.world_yaw_simulation,
                object_binding,
                reference_binding,
                landing,
                parent,
                _geometry=geometry,
            )
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
                inspection = inspect_emitted(action, selected, context, candidate)
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
                "gen_sim_inside_world_yaw": {
                    "attempts": attempts,
                    "accepted": chosen is not first,
                    "release_clearance_m": container.release_clearance,
                    "lift_height_m": options.lift_height,
                    "heights_unchanged": True,
                    "physical_evidence": False,
                },
            },
        ),
    )
