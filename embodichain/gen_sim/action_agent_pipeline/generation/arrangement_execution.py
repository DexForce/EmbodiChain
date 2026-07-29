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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import permutations, product
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    defaults_section,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_intent import (
    _arrangement_order_is_constrained,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_layout import (
    _ArrangementFootprint,
    _arrangement_object_footprint,
    _slot_xy_bounds,
    _xy_bounds_overlap,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    ArrangementLineSpec,
    ArrangementLineStepSpec,
    SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _clean_vector3,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _arm_side_for_position,
)

__all__ = [
    "_arrangement_arm_side_for_motion",
    "_arrangement_initial_occupancy_schedule",
    "_arrangement_plan_execution",
    "_arrangement_slot_allowed_sides",
]

_DEFAULTS = defaults_section("arrangement")
_CENTER_SAFE_HALF_WIDTH = float(_DEFAULTS["center_safe_half_width"])


def _arrangement_arm_side_for_motion(
    init_position: Sequence[float],
    target_xy: Sequence[float],
) -> str:
    init_y = float(init_position[1])
    target_y = float(target_xy[1])
    reference_y = init_y if abs(init_y) > _CENTER_SAFE_HALF_WIDTH else target_y
    return _arm_side_for_position([0.0, reference_y, 0.0])


def _arrangement_motion_metadata(
    init_position: Sequence[float],
    target_xy: Sequence[float],
) -> tuple[str, bool, float]:
    init_y = float(init_position[1])
    target_y = float(target_xy[1])
    init_side = 0 if init_y == 0.0 else (1 if init_y > 0 else -1)
    target_side = 0 if target_y == 0.0 else (1 if target_y > 0 else -1)
    cross_side = init_side != 0 and target_side != 0 and init_side != target_side
    return (
        _arrangement_arm_side_for_motion(init_position, target_xy),
        cross_side,
        abs(init_y - target_y),
    )


def _arrangement_slot_allowed_sides(slot_index: int, slot_count: int) -> frozenset[str]:
    """Return semantic arms allowed to place into a world-y ordered slot."""
    if slot_count < 1:
        raise ValueError("Arrangement requires at least one slot.")
    if not 0 <= slot_index < slot_count:
        raise ValueError(
            f"Arrangement slot index {slot_index} is outside [0, {slot_count})."
        )

    center = slot_count // 2
    if slot_count % 2 == 0:
        shared_start, shared_end = center - 1, center
    else:
        shared_start, shared_end = max(0, center - 1), min(slot_count - 1, center + 1)
    if shared_start <= slot_index <= shared_end:
        return frozenset({"left", "right"})
    if slot_index < shared_start:
        return frozenset({"right"})
    return frozenset({"left"})


def _arrangement_assignment_side(
    init_position: Sequence[float],
    target_xy: Sequence[float],
    *,
    slot_index: int,
    slot_count: int,
) -> str | None:
    """Choose a pickup arm only when it is permitted for the target slot."""
    init_y = float(init_position[1])
    pickup_sides = (
        frozenset({"left", "right"})
        if init_y == 0.0
        else frozenset({"left" if init_y > 0.0 else "right"})
    )
    compatible_sides = pickup_sides & _arrangement_slot_allowed_sides(
        slot_index, slot_count
    )
    if not compatible_sides:
        return None
    preferred_side = _arm_side_for_position([0.0, float(target_xy[1]), 0.0])
    if preferred_side in compatible_sides:
        return preferred_side
    return min(compatible_sides)


def _arrangement_plan_execution(
    spec: ArrangementLineSpec,
    slots: Sequence[Sequence[float]],
    *,
    generated_objects: Sequence[SceneObject],
    rigid_configs: Mapping[str, Mapping[str, Any]],
) -> tuple[str, list[ArrangementLineStepSpec]] | None:
    order_is_constrained = _arrangement_order_is_constrained(
        spec.order_by,
        task_description=spec.task_description,
    )
    groups = [
        [step for step in spec.steps if step.category == category]
        for category in spec.category_order
    ]
    if not groups or any(not group for group in groups):
        groups = [list(spec.steps)]
    group_orders = [
        (
            [tuple(group)]
            if order_is_constrained
            else list(permutations(sorted(group, key=lambda step: step.runtime_uid)))
        )
        for group in groups
    ]
    footprint_by_uid = {
        obj.source_uid: _arrangement_object_footprint(obj, scene_dir=Path("."))
        for obj in generated_objects
    }
    best = None
    for grouped_order in product(*group_orders):
        semantic_steps = [step for group in grouped_order for step in group]
        for spatial_direction, candidate_slots, physical_slot_indices in (
            ("right_to_left", slots, range(len(slots))),
            ("left_to_right", list(reversed(slots)), reversed(range(len(slots)))),
        ):
            assigned_steps = []
            assignment_is_valid = True
            for step, target_xy, physical_slot_index in zip(
                semantic_steps, candidate_slots, physical_slot_indices
            ):
                init_position = _clean_vector3(
                    rigid_configs[step.runtime_uid].get("init_pos", [0.0, 0.0, 0.0])
                )
                active_side = _arrangement_assignment_side(
                    init_position,
                    target_xy,
                    slot_index=physical_slot_index,
                    slot_count=len(slots),
                )
                if active_side is None:
                    assignment_is_valid = False
                    break
                _, cross_side, _ = _arrangement_motion_metadata(
                    init_position, target_xy
                )
                assigned_steps.append(
                    replace(
                        step,
                        slot_index=physical_slot_index,
                        active_side=active_side,
                        target_xy=[float(target_xy[0]), float(target_xy[1])],
                        cross_side=cross_side,
                    )
                )
            if not assignment_is_valid:
                continue
            scheduled = _arrangement_initial_occupancy_schedule(
                assigned_steps,
                rigid_configs=rigid_configs,
                footprint_by_uid=footprint_by_uid,
                clearance=spec.layout_clearance,
            )
            if scheduled is None:
                continue
            execution_steps, blockers, conflict_count = scheduled
            direction_cost = _arrangement_direction_cost(
                assigned_steps,
                candidate_slots,
                rigid_configs=rigid_configs,
            )
            cost = (
                direction_cost[0],
                direction_cost[1],
                direction_cost[2],
                conflict_count,
                0 if spatial_direction == "left_to_right" else 1,
                direction_cost[3],
            )
            finalized_steps = [
                replace(
                    step,
                    execution_index=index,
                    blocked_by=blockers[step.runtime_uid],
                )
                for index, step in enumerate(execution_steps)
            ]
            candidate = (cost, spatial_direction, finalized_steps)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        if order_is_constrained:
            # A hard semantic order may change execution scheduling, but it
            # must never silently degrade to an initial-position ordering.
            return None
        return _arrangement_initial_side_order_fallback(
            spec,
            slots,
            rigid_configs=rigid_configs,
            footprint_by_uid=footprint_by_uid,
        )
    return best[1], best[2]


def _arrangement_initial_side_order_fallback(
    spec: ArrangementLineSpec,
    slots: Sequence[Sequence[float]],
    *,
    rigid_configs: Mapping[str, Mapping[str, Any]],
    footprint_by_uid: Mapping[str, _ArrangementFootprint],
) -> tuple[str, list[ArrangementLineStepSpec]] | None:
    """Match initial world-y order to slot order when category planning fails."""
    categories = {step.category for step in spec.steps}
    if spec.axis != "world_y" or spec.order_by != "explicit" or len(categories) < 2:
        return None

    ordered_steps = sorted(
        spec.steps,
        key=lambda step: (
            float(
                _clean_vector3(
                    rigid_configs[step.runtime_uid].get("init_pos", [0.0, 0.0, 0.0])
                )[1]
            ),
            step.runtime_uid,
        ),
    )
    ordered_slots = sorted(
        enumerate(slots),
        key=lambda item: (float(item[1][1]), float(item[1][0]), item[0]),
    )
    assigned_steps = []
    for step, (slot_index, target_xy) in zip(ordered_steps, ordered_slots):
        init_position = _clean_vector3(
            rigid_configs[step.runtime_uid].get("init_pos", [0.0, 0.0, 0.0])
        )
        active_side, cross_side, _ = _arrangement_motion_metadata(
            init_position, target_xy
        )
        assigned_steps.append(
            replace(
                step,
                slot_index=slot_index,
                active_side=active_side,
                target_xy=[float(target_xy[0]), float(target_xy[1])],
                cross_side=cross_side,
            )
        )

    scheduled = _arrangement_initial_occupancy_schedule(
        assigned_steps,
        rigid_configs=rigid_configs,
        footprint_by_uid=footprint_by_uid,
        clearance=spec.layout_clearance,
    )
    if scheduled is None:
        # The fallback preserves historical behavior when no acyclic assignment
        # exists, rather than introducing a new planning rejection.
        execution_steps = assigned_steps
        blockers = {step.runtime_uid: () for step in assigned_steps}
    else:
        execution_steps, blockers, _ = scheduled
    return (
        "initial_side_order",
        [
            replace(
                step,
                execution_index=index,
                blocked_by=blockers[step.runtime_uid],
            )
            for index, step in enumerate(execution_steps)
        ],
    )


def _arrangement_initial_occupancy_schedule(
    steps: Sequence[ArrangementLineStepSpec],
    *,
    rigid_configs: Mapping[str, Mapping[str, Any]],
    footprint_by_uid: Mapping[str, _ArrangementFootprint],
    clearance: float,
) -> tuple[list[ArrangementLineStepSpec], dict[str, tuple[str, ...]], int] | None:
    target_bounds = {
        step.runtime_uid: _slot_xy_bounds(
            step.target_xy,
            max_half_extent=footprint_by_uid[step.runtime_uid].half_extent,
        )
        for step in steps
    }
    initial_bounds = {
        uid: footprint.xy_bounds for uid, footprint in footprint_by_uid.items()
    }
    # An edge other -> uid means the other object must leave uid's destination
    # before uid can enter it.
    dependencies = {step.runtime_uid: set() for step in steps}
    blockers = {}
    for step in steps:
        uid = step.runtime_uid
        blocked_by = []
        for other in steps:
            other_uid = other.runtime_uid
            if other_uid == uid:
                continue
            if _xy_bounds_overlap(
                target_bounds[uid],
                initial_bounds[other_uid],
                clearance=clearance,
            ):
                dependencies[uid].add(other_uid)
                blocked_by.append(other_uid)
        blockers[uid] = tuple(sorted(blocked_by))

    successors = {step.runtime_uid: set() for step in steps}
    for uid, dependency_uids in dependencies.items():
        for dependency_uid in dependency_uids:
            successors[dependency_uid].add(uid)
    by_uid = {step.runtime_uid: step for step in steps}
    remaining = set(by_uid)
    execution_steps = []
    previous_arm = None
    while remaining:
        # A missing ready node is a physical occupancy cycle. The caller must
        # therefore try another slot assignment instead of breaking the order.
        ready = [uid for uid in remaining if not (dependencies[uid] & remaining)]
        if not ready:
            return None

        def priority(uid: str) -> tuple[int, bool, int, float, str]:
            step = by_uid[uid]
            init_position = _clean_vector3(
                rigid_configs[uid].get("init_pos", [0.0, 0.0, 0.0])
            )
            _, _, distance = _arrangement_motion_metadata(init_position, step.target_xy)
            return (
                -len(successors[uid] & remaining),
                step.cross_side,
                0 if previous_arm is None or step.active_side == previous_arm else 1,
                round(distance, 9),
                uid,
            )

        selected_uid = min(ready, key=priority)
        selected_step = by_uid[selected_uid]
        execution_steps.append(selected_step)
        previous_arm = selected_step.active_side
        remaining.remove(selected_uid)
    return execution_steps, blockers, sum(len(value) for value in blockers.values())


def _arrangement_direction_cost(
    steps: Sequence[ArrangementLineStepSpec],
    slots: Sequence[Sequence[float]],
    *,
    rigid_configs: Mapping[str, Mapping[str, Any]],
) -> tuple[float, float, int, tuple[str, ...]]:
    max_y_distance = 0.0
    total_y_distance = 0.0
    reverse_shared_use_count = 0
    slot_count = len(slots)
    for step, slot in zip(steps, slots):
        init_position = _clean_vector3(
            rigid_configs[step.runtime_uid].get("init_pos", [0.0, 0.0, 0.0])
        )
        init_y = float(init_position[1])
        target_y = float(slot[1])
        distance = abs(init_y - target_y)
        max_y_distance = max(max_y_distance, distance)
        total_y_distance += distance
        allowed_sides = _arrangement_slot_allowed_sides(step.slot_index, slot_count)
        target_preferred_side = _arm_side_for_position([0.0, target_y, 0.0])
        if len(allowed_sides) > 1 and step.active_side != target_preferred_side:
            reverse_shared_use_count += 1
    return (
        round(max_y_distance, 9),
        round(total_y_distance, 9),
        reverse_shared_use_count,
        tuple(step.runtime_uid for step in steps),
    )
