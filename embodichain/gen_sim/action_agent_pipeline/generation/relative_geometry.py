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

from collections.abc import Mapping
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.protocol.actions import DUAL_ARM_NAME

from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    RelativePlacementSpec,
    RelativePlacementStepSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _load_mesh_vertices,
)
from embodichain.gen_sim.action_agent_pipeline.generation import (
    relative_surface_geometry as _relative_surface_geometry,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_offsets import (
    _POSE_SENSITIVE_STAGING_Z_DELTA,
    _STAGING_Z_DELTA,
    _offset_position,
    _relative_release_offset,
    _side_relation_xy_offsets,
    _with_final_auto_arm_sides,
    _with_self_relative_absolute_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_slot_geometry import (
    _coordinated_payload_grid_offsets,
    _inside_container_axis_offsets,
    _inside_container_slot_axis_and_distance,
    _with_inside_container_slot_offsets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_surface_geometry import (
    _with_on_surface_release_offsets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_transport_geometry import (
    _with_coordinated_side_release_height_offsets,
    _with_coordinated_transport_geometry,
)

__all__ = [
    "_POSE_SENSITIVE_STAGING_Z_DELTA",
    "_STAGING_Z_DELTA",
    "_inside_container_axis_offsets",
    "_inside_container_slot_axis_and_distance",
    "_make_relative_summary",
    "_offset_position",
    "_relative_release_offset",
    "_side_relation_xy_offsets",
    "_with_on_surface_release_offsets",
    "_with_inside_container_slot_offsets",
    "_with_coordinated_side_release_height_offsets",
    "_with_coordinated_transport_geometry",
    "_with_final_auto_arm_sides",
    "_with_self_relative_absolute_targets",
]


def _target_local_zmin_for_orientation(
    obj_config: Mapping[str, Any],
    orientation_goal: str,
) -> float | None:
    """Preserve the legacy mesh-loader patch point while delegating policy."""
    original_loader = _relative_surface_geometry._load_mesh_vertices
    _relative_surface_geometry._load_mesh_vertices = _load_mesh_vertices
    try:
        return _relative_surface_geometry._target_local_zmin_for_orientation(
            obj_config,
            orientation_goal,
        )
    finally:
        _relative_surface_geometry._load_mesh_vertices = original_loader


def _make_relative_summary(spec: RelativePlacementSpec) -> dict[str, Any]:
    if spec.intent == "coordinated_pickment":
        summary = {
            "mode": "coordinated_pickment",
            "intent": spec.intent,
            "moved_object": spec.moved_runtime_uid,
            "reference_object": spec.reference_runtime_uid,
            "relation": spec.relation,
            "active_arm": DUAL_ARM_NAME,
            "release_offset": spec.release_offset,
            "target_position": spec.release_position,
            "orientation_goal": spec.orientation_goal,
            "orientation_axis": spec.orientation_axis,
            "orientation_align_to": spec.orientation_align_to_runtime_uid,
        }
        if spec.coordinated_terminal_behavior is not None:
            summary["direction"] = spec.coordinated_direction
            summary["terminal_behavior"] = spec.coordinated_terminal_behavior
            summary["payloads"] = [
                placement.moved_runtime_uid
                for placement in spec.placements
                if placement.intent == "place_relative"
            ]
        if spec.relation == "on" and not spec.reference_is_initial_pose:
            summary["surface_clearance"] = spec.surface_clearance
        return summary
    if len(spec.placements) == 1:
        summary = {
            "mode": "object_manipulation",
            "intent": spec.intent,
            "moved_object": spec.moved_runtime_uid,
            "reference_object": spec.reference_runtime_uid,
            "relation": spec.relation,
            "active_arm": f"{spec.active_side}_arm",
            "release_offset": spec.release_offset,
            "hover_height": spec.hover_height,
            "orientation_goal": spec.orientation_goal,
            "orientation_axis": spec.orientation_axis,
            "orientation_align_to": spec.orientation_align_to_runtime_uid,
        }
        if spec.relation == "on" and not spec.reference_is_initial_pose:
            summary["surface_clearance"] = spec.surface_clearance
        if spec.upright_in_place:
            summary["upright_in_place"] = True
        if spec.pickup_upright_direction is not None:
            summary["pickup_upright_direction"] = spec.pickup_upright_direction
        if spec.pickup_rotate_upright is not None:
            summary["pickup_rotate_upright"] = spec.pickup_rotate_upright
        return summary
    return {
        "mode": "dual_arm_object_manipulation",
        "manipulations": [
            _relative_placement_summary(placement) for placement in spec.placements
        ],
    }


def _relative_placement_summary(
    placement: RelativePlacementStepSpec,
) -> dict[str, Any]:
    summary = {
        "intent": placement.intent,
        "moved_object": placement.moved_runtime_uid,
        "reference_object": placement.reference_runtime_uid,
        "relation": placement.relation,
        "active_arm": f"{placement.active_side}_arm",
        "release_offset": placement.release_offset,
        "hover_height": placement.hover_height,
        "orientation_goal": placement.orientation_goal,
        "orientation_axis": placement.orientation_axis,
        "orientation_align_to": placement.orientation_align_to_runtime_uid,
    }
    if placement.relation == "on" and not placement.reference_is_initial_pose:
        summary["surface_clearance"] = placement.surface_clearance
    if placement.upright_in_place:
        summary["upright_in_place"] = True
    if placement.pickup_upright_direction is not None:
        summary["pickup_upright_direction"] = placement.pickup_upright_direction
    if placement.pickup_rotate_upright is not None:
        summary["pickup_rotate_upright"] = placement.pickup_rotate_upright
    return summary
