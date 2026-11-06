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

"""Source-compatible semantic action-option defaults and effect declarations."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from .semantic_graph import SemanticTaskGraph

__all__ = ["resolve_action_profile_declaration", "source_pick_option_declaration"]

_AXIS_ALIGN_CALL_ID = "simulation.axis_align"
_COORDINATED_TRANSPORT_CALL_ID = "simulation.coordinated_transport"
_COORDINATED_DETACH_DISTANCE = 0.08
_COORDINATED_RETREAT_MARGIN = 0.04
_DEFAULT_PICK_LIFT_HEIGHT = 0.16
_PLACE_RELATIVE_CALL_ID = "simulation.place_relative"
_UPRIGHT_PLACE_CALL_ID = "gen_sim.place_upright"
_STACK_PLACE_CALL_ID = "gen_sim.stack_place"
_MOVE_HELD_OBJECT_CALL_ID = "simulation.move_held_object"
_POUR_CALL_ID = "simulation.pour"
_COORDINATED_HOLD_CALL_ID = "simulation.coordinated_hold"
DRAWER_TRANSPORT_CALL = "gen_sim.drawer_transport"
IDLE_CALL = "gen_sim.clear_idle_arm"
PARK_CALL = "gen_sim.articulation_park"
SLIDE_CALL = "gen_sim.articulation_slide"
WITHDRAW_CALL = "gen_sim.articulation_withdraw"


def source_pick_option_declaration() -> dict[str, Any]:
    """Return an independent neutral source Pick configuration declaration."""
    return {
        "kind": "pick_up",
        "pick_object_part": "center",
        "pre_grasp_distance": 0.15,
        "lift_height": _DEFAULT_PICK_LIFT_HEIGHT,
        "approach_alignment_max_angle": 0.1,
        "hand_interp_steps": 12,
        "grasp_settle_steps": 0,
        "grasp_commit_fraction": 1.0,
    }


def resolve_action_profile_declaration(
    graph: SemanticTaskGraph,
    *,
    default_resources: Mapping[str, Mapping[str, str]],
    registered_pick_options: Mapping[str, Mapping[str, Any]],
    pour_geometry_revision: int | None = None,
    slide_preshape_fraction: float = 0.85,
    slide_release_retreat_distance: float = 0.04,
) -> dict[str, Any]:
    """Select unchanged recipe declarations without installing skill implementations.

    Args:
        graph: Ordered canonical semantic calls.
        default_resources: Explicit owner-resolved skill resource slots. Custom
            resource names remain intact; no robot model or control-part inference.
        registered_pick_options: Exact typed-option-shaped declarations provided
            by the source grasp-route owner for registered Pick aliases.
        pour_geometry_revision: Explicit candidate-dependent Pour geometry mode.
        slide_preshape_fraction: Explicit source Slide recipe preshape fraction.
        slide_release_retreat_distance: Explicit source Slide retreat distance.

    Returns:
        Existing task ``profile`` mapping containing defaults/options and effect
        monitor references. This does not choose planning algorithms, mutate
        private skills or evaluate physical effects. Shared-runtime opt-in options
        and per-occurrence execution policies are composed separately.
    """
    calls = [node["call"] for node in graph["nodes"]]
    ids = {call.get("call_id") for call in calls if call["kind"] == "registered"}
    supplied = deepcopy(dict(registered_pick_options))
    required = {
        identity
        for identity in ids
        if identity == "simulation.pick"
        or isinstance(identity, str)
        and identity.startswith("gen_sim.pick.")
    }
    if set(supplied) != required:
        raise ValueError(
            "Registered Pick options must match the exact declared alias IDs."
        )
    pick_options = supplied
    pick_routes = supplied
    has_relative_place = bool(
        ids & {_PLACE_RELATIVE_CALL_ID, _UPRIGHT_PLACE_CALL_ID, _STACK_PLACE_CALL_ID}
    )
    upright_lowerer_routes = _UPRIGHT_PLACE_CALL_ID in ids
    move_held_routes = _MOVE_HELD_OBJECT_CALL_ID in ids
    drawer_transport_routes = DRAWER_TRANSPORT_CALL in ids
    pour_objects = _POUR_CALL_ID in ids
    hold_lowerer_routes = _COORDINATED_HOLD_CALL_ID in ids
    axis_align_objects = _AXIS_ALIGN_CALL_ID in ids
    has_park_call = "simulation.park" in ids
    idle_routes = IDLE_CALL in ids
    has_articulation_park_call = PARK_CALL in ids
    articulation_bindings = bool(ids & {SLIDE_CALL, WITHDRAW_CALL})
    lowerer_routes = _COORDINATED_TRANSPORT_CALL_ID in ids
    PRESHAPE_FRACTION = slide_preshape_fraction
    RELEASE_RETREAT_DISTANCE = slide_release_retreat_distance
    default_pick_options = source_pick_option_declaration()
    result = {
        "defaults": deepcopy(
            {str(skill): dict(slots) for skill, slots in default_resources.items()}
        ),
        "action_options": {
            "pick": default_pick_options,
            "place": {
                "kind": "place",
                "hand_interp_steps": 12,
                "release_settle_steps": 60,
                "lift_height": 0.05,
                "cartesian_waypoint_count": 2,
                "preserve_current_object_orientation": True,
            },
            "hand_over": {
                "kind": "hand_over",
                "pre_grasp_distance": 0.15,
                "lift_height": 0.0,
                "hand_interp_steps": 16,
                "hold_steps": 8,
                "retreat_steps": 36,
                "retreat_distance": 0.1,
                "receive_pick_object_part": "center",
                "release_at_target": False,
                "arm_selection": "bound",
            },
            **pick_options,
            **(
                {
                    call_id: {
                        "kind": "place",
                        "hand_interp_steps": 12,
                        "release_settle_steps": 60,
                        "lift_height": 0.1,
                        "cartesian_waypoint_count": 2,
                        "preserve_current_object_orientation": True,
                    }
                    for call_id in (
                        _PLACE_RELATIVE_CALL_ID,
                        *((_UPRIGHT_PLACE_CALL_ID,) if upright_lowerer_routes else ()),
                    )
                }
                if has_relative_place
                else {}
            ),
            **(
                {_MOVE_HELD_OBJECT_CALL_ID: {"kind": "move_held_object"}}
                if move_held_routes
                else {}
            ),
            **(
                {DRAWER_TRANSPORT_CALL: {"kind": "move_held_object"}}
                if drawer_transport_routes
                else {}
            ),
            **(
                {
                    _POUR_CALL_ID: {
                        "kind": "pour",
                        "dwell_duration": 0.0,
                        "rotate_angle": -1.0471975512,
                    }
                }
                if pour_objects
                else {}
            ),
            **(
                {
                    _COORDINATED_HOLD_CALL_ID: {
                        "kind": "coordinated_pickment",
                        "object_motion_keyframes": 8,
                        "pre_grasp_distance": 0.1,
                        "lift_height": 0.08,
                        "hand_interp_steps": 10,
                        "hold_steps": 4,
                        "release": False,
                        "approach_direction": [0.0, 0.0, -1.0],
                        "left_to_right_arm_direction": [0.0, 1.0, 0.0],
                        "middle_empty_ratio": 0.4,
                    }
                }
                if hold_lowerer_routes
                else {}
            ),
            **(
                {
                    _AXIS_ALIGN_CALL_ID: {
                        "kind": "axis_align",
                        "pre_grasp_distance": 0.15,
                        "lift_height": 0.16,
                        "hand_interp_steps": 12,
                        "grasp_settle_steps": 0,
                        "grasp_commit_fraction": 1.0,
                        "target_axis": [0.0, 0.0, 1.0],
                    }
                }
                if axis_align_objects
                else {}
            ),
            **({"simulation.park": {"kind": "move_joints"}} if has_park_call else {}),
            **({IDLE_CALL: {"kind": "move_joints"}} if idle_routes else {}),
            **(
                {PARK_CALL: {"kind": "move_joints"}}
                if has_articulation_park_call
                else {}
            ),
            **(
                {
                    SLIDE_CALL: {
                        "kind": "slide",
                        "approach_distance": 0.1,
                        "hand_interp_steps": 12,
                        "preshape_fraction": PRESHAPE_FRACTION,
                        "release_retreat_distance": RELEASE_RETREAT_DISTANCE,
                        "approach_along_grasp_axis": True,
                    },
                    WITHDRAW_CALL: {"kind": "move_end_effector"},
                }
                if articulation_bindings
                else {}
            ),
            **(
                {
                    "simulation.coordinated_transport": {
                        "kind": "coordinated_pickment",
                        "object_motion_keyframes": 8,
                        "pre_grasp_distance": 0.1,
                        "lift_height": 0.08,
                        "hand_interp_steps": 10,
                        "hold_steps": 4,
                        "release": True,
                        "release_steps": 10,
                        "retreat_distance": _COORDINATED_DETACH_DISTANCE
                        + _COORDINATED_RETREAT_MARGIN,
                        "retreat_steps": 12,
                        "approach_direction": [0.0, 0.0, -1.0],
                        "left_to_right_arm_direction": [0.0, 1.0, 0.0],
                        "middle_empty_ratio": 0.4,
                    }
                }
                if lowerer_routes
                else {}
            ),
        },
        "effect_monitors": {
            semantic_id: {
                "monitor_id": "builtin.composite_effect",
                "revision": "1",
                "params": {
                    "consecutive_samples": 3,
                    "attached_translation_threshold": (
                        0.02
                        if semantic_id
                        in {_PLACE_RELATIVE_CALL_ID, _UPRIGHT_PLACE_CALL_ID}
                        else 0.06
                    ),
                    "attached_rotation_threshold": 3.0,
                    "detached_translation_threshold": (
                        0.03
                        if semantic_id
                        in {_PLACE_RELATIVE_CALL_ID, _UPRIGHT_PLACE_CALL_ID}
                        else (
                            _COORDINATED_DETACH_DISTANCE
                            if semantic_id
                            in {
                                _COORDINATED_TRANSPORT_CALL_ID,
                                _COORDINATED_HOLD_CALL_ID,
                            }
                            else 0.08
                        )
                    ),
                    "detached_rotation_threshold": 3.141592653589793,
                },
            }
            for semantic_id in (
                "pick",
                "place",
                "hand_over",
                "simulation.coordinated_transport",
                _COORDINATED_HOLD_CALL_ID,
                _AXIS_ALIGN_CALL_ID,
                _MOVE_HELD_OBJECT_CALL_ID,
                *((_POUR_CALL_ID,) if pour_geometry_revision in (1, 2) else ()),
                *pick_routes,
                _PLACE_RELATIVE_CALL_ID,
                _UPRIGHT_PLACE_CALL_ID,
            )
            if any(
                (
                    node["call"].get("kind") == semantic_id
                    or node["call"].get("call_id") == semantic_id
                    or (
                        semantic_id == "pick"
                        and node["call"].get("kind") in {"hand_over", "place"}
                    )
                    for node in graph["nodes"]
                )
            )
        },
        "grounding_providers": {"hand_over": "simulation.configured_handover_pose"},
    }
    return result
