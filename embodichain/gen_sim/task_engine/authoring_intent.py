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
"""Source-grounded graph expressions without execution contracts."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from itertools import groupby
import math
import re
from typing import Any

from ..task_spec import TaskSpec
from .scene_context import (
    _SCENE_ROLE_ID,
    require_articulation,
    scene_objects_requirement,
)
from .semantic_graph import (
    _AUTHORING_INTENT_PREFIX,
    SEMANTIC_TASK_GRAPH_SCHEMA,
    validate_semantic_task_graph,
)

__all__: list[str] = []

_ROLE_KINDS = set(
    "object target surface container tool articulation button robot".split()
)
_ARMS = {"left", "right", "auto"}
_ARM_VERBS = {"return_arms", "open_grippers", "move_gripper"}
_ACQUIRE_VERBS = {"pick", "grasp"}
_HELD_VERBS = {
    "move_held",
    "place",
    "release",
    "handover",
    "transfer_contents",
    "articulate",
}
_DIRECTIONAL_VERBS = {
    "pick",
    "move_held",
    "move_gripper",
    "press",
    "push",
    "articulate",
}
_DIRECTIONS = {"up", "down", "toward", "away", "left", "right"}
_LIFT_AND_DEPART = re.compile(
    r"(?P<lift>\b(?:lift|raise)\b[^.!?;,]*?)"
    r"(?:\s*,?\s*(?:and(?:\s+then)?|then)\s+|[.!?;]\s+then\s+)"
    r"(?P<depart>(?:move|shift|carry|translate)\s+it\b[^.!?;,]*?\baway\b)",
    re.IGNORECASE,
)
_NEGATED_ACTION_PREFIX = re.compile(
    r"\b(?:not|never|avoid|don['’]t)\s+(?:\w+\s+){0,3}$", re.IGNORECASE
)
_SHORT_DISTANCE = re.compile(
    r"\b(?:(?:short|small)\s+distance|slightly)\b", re.IGNORECASE
)
_GRIPPER_RELATIONS = {"above", "touching", "clear_of"}
_OTHER_CONTROL_POSITION = "other_from_initial"
_VERBS = (
    _ARM_VERBS
    | _ACQUIRE_VERBS
    | set(
        "move_held place push transfer_contents orient turn release handover actuate press articulate".split()
    )
)
_FRAMES = {"head_camera", "table", "robot", "world", "reference"}
_LATERAL = {"left", "right", "center", "far_left", "far_right"}
_NEAR_REGIONS = {"near_center_of", "near_left_end", "near_right_end"}
_AXIS_INVERSES = {
    "left_of": "right_of",
    "right_of": "left_of",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "above": "below",
    "below": "above",
}
_AXIS_RELATIONS = set(_AXIS_INVERSES)
_GROUP_RELATIONS = {"between", "opposite_sides", "opposite_ends", "spans"}
_RELATIONS = (
    _AXIS_RELATIONS
    | _NEAR_REGIONS
    | _GROUP_RELATIONS
    | set(
        (
            "at on inside not_inside next_to near centered_on centered_above separate_from "
            "supported_by on_top_faces aligned_with seated_on tightened_on compatible_with fixed_to touching"
        ).split()
    )
)
_REFERENCED_POINTING = {"pointing_toward", "pointing_away_from"}
_ORIENTATIONS = _REFERENCED_POINTING | set(
    (
        "upright vertical flat level opening_up opening_down tilted left_to_right "
        "front_to_back aligned_long_axis upright_cross_section lying_on_side "
        "pointing_left pointing_right pointing_up pointing_down pointing_forward pointing_backward"
    ).split()
)
_SLOT_RELATIONS = (
    _NEAR_REGIONS
    | _AXIS_RELATIONS
    | {"near", "next_to", "on", "inside", "centered_on", "centered_above"}
)
_STRONGER_RELATIONS = {
    "above": ("centered_above",),
    "near": (*_NEAR_REGIONS, "next_to"),
    "on": ("on_top_faces", "centered_on", "supported_by", "seated_on"),
}

_FACT_FIELDS = {
    "relation": ({"subjects", "references", "value"}, {"frame"}),
    "orientation": ({"subjects", "value"}, {"references", "frame"}),
    "pose_match": ({"subjects", "references", "epoch", "components"}, set()),
    "vertical_rise": ({"subjects", "comparison", "meters"}, set()),
    "distance_compare": ({"subjects", "references", "comparison"}, set()),
    "distance_change": ({"subjects", "references", "comparison", "epoch"}, set()),
    "ordered_row": ({"subjects", "frame"}, {"references"}),
    "view_location": ({"subjects", "frame"}, {"lateral", "depth", "references"}),
    "transferred": ({"subjects", "references", "material", "all", "quantity"}, set()),
    "hung_by": ({"subjects", "references", "part"}, set()),
    "held": ({"subjects", "arms"}, set()),
    "grasped_at": ({"subjects", "part", "arms"}, set()),
    "gripper_relation": ({"arms", "references", "value"}, set()),
    "released": ({"subjects"}, set()),
    "arms_at_initial_pose": ({"arms"}, set()),
    "grippers_open": ({"arms"}, set()),
    "untouched": ({"subjects"}, set()),
    "untouched_except": ({"subjects"}, set()),
    "stable": ({"subjects"}, set()),
    "intact": ({"subjects"}, set()),
    "mutually_separate": ({"subjects"}, set()),
    "spaced_apart": ({"subjects"}, set()),
    "unoccupied": ({"subjects"}, set()),
    "movable": ({"subjects"}, set()),
    "actuated": ({"subjects", "value"}, set()),
    "articulated": ({"subjects", "references", "mechanism", "value"}, set()),
}

# Only independent members can be projected out of a grouped state.
_INDEPENDENT_STATE_MEMBERS = {
    "actuated": "subjects",
    "distance_change": "subjects",
    "held": "subjects",
    "intact": "subjects",
    "movable": "subjects",
    "orientation": "subjects",
    "pose_match": "subjects",
    "released": "subjects",
    "unoccupied": "subjects",
    "untouched": "subjects",
    "vertical_rise": "subjects",
    "view_location": "subjects",
    "arms_at_initial_pose": "arms",
    "grippers_open": "arms",
}

_FACT_CHOICES = {
    "relation": {"value": _RELATIONS},
    "orientation": {"value": _ORIENTATIONS},
    "view_location": {"lateral": _LATERAL, "depth": ("near", "far", "middle")},
    "pose_match": {
        "epoch": ("initial", "current"),
        "components": ("position", "orientation"),
    },
    "vertical_rise": {"comparison": ("eq", "gt", "ge", "lt", "le")},
    "distance_compare": {"comparison": ("gt", "lt")},
    "distance_change": {"comparison": ("gt", "lt"), "epoch": ("initial",)},
    "gripper_relation": {"value": _GRIPPER_RELATIONS},
    "articulated": {"mechanism": ("hinge", "slide"), "value": ("open", "closed")},
    "transferred": {"material": ("solid", "liquid")},
    "actuated": {
        "value": ("open", "closed", "pressed", "turned", _OTHER_CONTROL_POSITION)
    },
}

_PROMPT = {
    "shape": {
        "roles": [
            {
                "id": "ASCII identifier; preserve caller-declared case",
                "kind": "one role kind string",
                "description": "concise image/source identity",
                "part_of": "owner role ID for a named part, grip site or joint; omit for standalone roles",
            }
        ],
        "steps": [
            {
                "verb": "one verb string",
                "subjects": ["role_id"],
                "arms": ["auto"],
                "effects": ["fact"],
                "quote": "source span ID or exact source span",
                "parallel_group": "omit except for explicitly simultaneous independent picks",
                "extent": "short only for a source-requested short move_held displacement; otherwise omit",
            }
        ],
        "conditions": [
            {
                "scope": "initial|always|final",
                "fact": "fact",
                "quote": "source span ID or exact source span",
            }
        ],
    },
    "facts": {
        kind: {"required_fields": sorted(required), "optional_fields": sorted(optional)}
        for kind, (required, optional) in _FACT_FIELDS.items()
    },
    "spatial_claim_values": sorted(_RELATIONS),
    "values": {
        "role.kind": sorted(_ROLE_KINDS),
        "step.verb": sorted(_VERBS),
        "step.direction": sorted(_DIRECTIONS),
        "relation.value": sorted(_RELATIONS),
        "orientation.value": sorted(_ORIENTATIONS),
        "frame": sorted(_FRAMES),
        "view_location.lateral": sorted(_LATERAL),
        "view_location.depth": list(_FACT_CHOICES["view_location"]["depth"]),
        "pose_match.epoch": list(_FACT_CHOICES["pose_match"]["epoch"]),
        "pose_match.components": list(_FACT_CHOICES["pose_match"]["components"]),
        "vertical_rise.comparison": list(_FACT_CHOICES["vertical_rise"]["comparison"]),
        "distance_compare.comparison": list(
            _FACT_CHOICES["distance_compare"]["comparison"]
        ),
        "distance_change.comparison": list(
            _FACT_CHOICES["distance_change"]["comparison"]
        ),
        "distance_change.epoch": list(_FACT_CHOICES["distance_change"]["epoch"]),
        "gripper_relation.value": sorted(_GRIPPER_RELATIONS),
        "articulated.mechanism": list(_FACT_CHOICES["articulated"]["mechanism"]),
        "articulated.value": list(_FACT_CHOICES["articulated"]["value"]),
        "transferred.material": list(_FACT_CHOICES["transferred"]["material"]),
        "transferred.quantity": "positive integer count, or null only if no item count is specified",
        "actuated.value": list(_FACT_CHOICES["actuated"]["value"]),
    },
    "semantics": (
        "A fact is a JSON object with a single kind string and exactly its listed fields. "
        "For each quote, use one source_spans key, or copy one contiguous source span verbatim. "
        "Subjects/references are lists of stable role IDs. Use distinct role IDs for separate instances "
        "and destination slots. A role may have optional part_of: owning_role_id. Do not guess poses. "
        "Declare both the moving part and its joint, and every specified grip site, as roles with "
        "explicit part_of links. Mentioning ownership only in a description is not a binding. "
        "Use the caller's scene_context for asset modeling; do not infer asset_kind on roles. "
        "Arms belong in the arms field, not object roles. "
        "Keep stated defining attributes such as color and material in role descriptions. If rank by size is the selector, "
        "use that rank, not an extra guessed location. Only encode camera locations as typed facts when required. "
        "Ordered_row lists members from left to right in the stated frame. A relation holds for every "
        "subject and reference, except opposite_sides/opposite_ends jointly relate exactly two subjects to one reference. "
        "When a row is positioned by a specified central member, locate that anchor and keep "
        "the row order and side relations. Do not impose a separate center target on its neighbors. "
        "Multiple references in supported_by/on_top_faces mean simultaneous support. Spans means bridge across. "
        "The spaced_apart fact is a positive pairwise gap among its subjects; do not supply a surface as a reference. "
        "If the source names a contact surface, represent that surface, e.g. on_top_faces rather than "
        "generic supported_by for a top-face requirement. "
        "When the source positions something relative to a named part, the operand is the "
        "part role, whose part_of identifies its owner. Do not add its owner as another operand. "
        "For example, positioning above an opening references the opening alone; transfer uses the container. "
        "Near_left_end and near_right_end refer to the ends of the reference along its main extent. "
        "Use view_location for an initial side/depth selector in the applicable reference frame. "
        "With frame:reference, supply exactly one reference role. "
        "Extreme lateral positions stay lateral, not far depth. An image midpoint is not the center "
        "of a partially visible table. Use the explicitly stated frame and locations for the same roles. "
        "Use directional pointing (on a part_of role) when the orientation specifies which end points where. "
        "For pointing toward, facing, or pointing away from a named object, use pointing_toward or "
        "pointing_away_from with that role in references, not a forward/backward axis guess. "
        "When pointing a rigid part of a held object, orient the held owner with its holding arm; "
        "the orientation effect names the part, not the owner. Link the part with part_of. "
        "Flat describes a lying posture on a support without choosing a particular resting face. "
        "Use lying_on_side only when the source specifies a side-lying posture; "
        "lying on top of a support identifies the support, not which face of the object rests on it. "
        "Use unoccupied for requested free destinations, "
        "and fixed_to rather than on for a fixture attached to a support. At means the subject is placed at the "
        "declared destination slot, not merely near it. For a placement with multiple simultaneous spatial "
        "requirements, state them directly for the subject or use an exact slot with those typed conditions. "
        "A slot description is not evidence of its required location, support, or clearance. "
        "Putting something on the center of a surface is centered_on, or at a slot centered_on it. "
        "Near_center_of is for proximity to the center, not exact centering or support. "
        "If support and center proximity are both requested, state both at the placement. "
        "Use near_center_of with the referenced surface or object for proximity to its center, "
        "or near with an explicit center target centered_on that reference. Being in the middle "
        "along only one axis does not establish proximity to the center. "
        "Near is not transitive. Joint effects can be stated once all referenced participants are in place. "
        "Pose_match initial is the referenced role's pose before any action, not its current pose. "
        "Refer to the original object with epoch:initial, not a new role named after its starting pose. "
        "To keep an object fixed, match its own initial pose (the same object role as subject "
        "and reference), not the initial pose of its supporting surface or coordinate frame. "
        "For precise matching to a reference pose, use pose_match with the required components, "
        "not just an alignment direction. "
        "Use vertical_rise only when the source specifies a metric threshold. "
        "For an unquantified lift use direction up and the requested destination, never invent a numeric rise. "
        "Distance_compare compares one subject with another item in the same state: references "
        "are [common anchor, comparison item], and gt means the subject is farther from the anchor. "
        "Use this for a stated initial distance comparison, not a depth category or a time change. "
        "Distance_change compares the subject's distance to the same reference before the entire task "
        "(epoch:initial) and after the event (gt means farther, lt means closer); do not replace it with far. "
        "Transferred subjects are the contents; references are [source container, recipient] in that order. "
        "Put any given item count in transferred.quantity even if the subjects are grouped or enumerated. "
        "The count in a label, description or quote is not enough. "
        "A specified long-axis match needs an orientation:aligned_long_axis with the actual "
        "reference object, not a generic aligned_with relation. "
        "Hung_by references one hook; part is the hanging part of the subject. "
        "Do not add an on-support effect for the same hanging outcome. "
        "Different fixtures use distinct role IDs; only assert physical separation or spacing if requested. "
        "Put completed step outcomes in effects and task conditions in initial/always/final. "
        "Encode explicit terminal states in final; if no separate terminal state is requested, "
        "completion is the last requested step and no extra terminal state is needed. "
        "Use optional step maintain:[fact,...] for a condition that must hold throughout that step, "
        "such as keeping an object level while carrying it. A task-wide continuous condition belongs "
        "in always. A final effect alone does not encode preservation during movement. "
        "Encode required initial contents, positions, and fixed support that actions use or must preserve "
        "as facts. Preserve initial descriptive details used only to identify roles in the role descriptions "
        "or in matching typed initial facts. Do not omit or contradict their attributes, relative sides, "
        "or orientation, and do not infer additional frame constraints. If the source directs identification "
        "by initial spatial position or a particular viewpoint, encode those selectors as typed initial facts. "
        "Use movable for explicitly loose and movable roles when the source distinguishes them from fixtures. "
        "Do not turn appearance/identity selectors into additional task constraints. "
        "Give all requested simultaneous relations, orientations, quantitative constraints, retention, order and "
        "negations structured facts. An exact quote or role description alone does not express a fact. "
        "Declare stable only when the source asks for stability or balance. Stacking, nesting "
        "or release alone does not add a separate stability requirement. If it is requested, "
        "list the actual assembled roles in stable.subjects. "
        "For two objects requested on opposite sides of a future placement, the reference is the "
        "other object's installed pose, not its initial/current pose before placement. Add the "
        "opposite_sides effect on the step that places the reference (or completes the relationship "
        "afterward); do not use initial or earlier side facts as the only evidence. "
        "Untouched_except protects all other scene objects; intact excludes disassembly; mutually_separate "
        "means the listed subjects do not intersect, contain or nest within one another. Use untouched_except "
        "for every other unspecified object; untouched only protects named individual roles, "
        "not a prose role describing all remaining objects. "
        "Group subjects that share the same fact instead of repeating individual facts. "
        "Pick acquires and lifts a free object in the ordinary unquantified pickup; "
        "it may state direction:up. Use a separate move_held for distinct requested "
        "height, location, or clearance. Move_held/orient/transfer_contents retain the grasp; "
        "Use grasp to acquire an attached part without detaching it. Use grasped_at on pick/grasp only "
        "for a grip site specified by the source. A named part used for contact, placement, or hanging "
        "is not thereby a grip site; use held alone for an unspecified grip. "
        "For two arms acquiring the SAME object at once, use one pick with both arms and separate "
        "grasped_at facts for each specified site. Keep both arms on a bimanual hold/move. When the "
        "source explicitly requires simultaneous acquisition of DIFFERENT objects, use two adjacent "
        "pick steps with the same parallel_group ID. Each has one explicit arm, its own distinct "
        "subject, and only its own held or grasped_at effects. Both complete before the next step. "
        "'Use both arms' and separate hand-object assignments do not specify simultaneous timing; "
        "without explicit timing language, omit parallel_group and use ordinary independent picks. "
        "A later place still has one object per step; do not make it simultaneous unless required. "
        "For distinct held objects, use one move_held step per object with its actual holding arm. "
        "Do not combine their subjects in one step or silently serialize an explicit timing constraint. "
        "For articulated movement, grasp the moving part at the specified site, articulate that same held "
        "part with articulated referencing a distinct articulation role for its joint, part_of the "
        "moving part or its owner, plus the mechanism, then release if requested. "
        "Reuse the owning asset's scene_context ID when it is supplied. A turn of a free object is not joint actuation. "
        "For a bistable control explicitly moved to its other position, use a push or actuate action "
        "on that specific control part, with an actuated effect valued other_from_initial. This means "
        "the other stable position relative to the same control's initial position, not just pressed or "
        "turned. Link the control part to its caller-declared articulated owner. "
        "For a press without grasping, move_gripper with gripper_relation:above at the actual site, "
        "press that site (actuated:pressed), then move_gripper with gripper_relation:clear_of if requested. "
        "When the source describes a button on a surface, the press effect names that button, "
        "not the whole surface used for approach. "
        "Use optional step direction on move_held, move_gripper, press, push or articulate when the "
        "source specifies a direction. Lifting away has direction up; clear_of describes leaving contact, "
        "not its direction. Move_gripper has no object subjects. For pressing a held rigid tool against "
        "a surface, use move_held with the whole held tool as subject and relation:touching from its "
        "specified contact part to the surface. Preserve the grasp during contact; do not use "
        "press, actuated, or place for that contact. "
        "For non-displacement while pressing, keep the parent object's initial pose instead of claiming it is untouched. "
        "a transfer has the held source container as its step subject. Place moves a held subject to its "
        "declared destination and releases it "
        "(including seating, nesting or hanging); its released effect can witness release after placement. "
        "Use move_held for a held destination or lift. When told to lift an object and then move "
        "it away, emit two ordered move_held steps with the same held subject and arm(s): first "
        "direction:up, then direction:away. Cite the whole source clause or the respective action "
        "phrases. Put extent:short on the second motion for an explicitly short displacement; do not "
        "invent a metric threshold. Keep any stated distance reference on that second step. "
        "Transferring contents does not restore the source container's orientation unless requested. "
        "Push does not pick up. To tighten a part against a fixed reference, first seat it with a "
        "seated_on effect while held, then turn with tightened_on for the same pair, then release if requested. "
        "Handover arms are [source, destination]. Use auto for unspecified arms and retain explicit sides. "
        "Auto still occupies one available arm until that object is released. A bimanual grasp "
        "occupies both. If no simultaneous hold is required, complete and release individual "
        "items in sequence; a requirement to release all does not mean hold them all at once. "
        "Return_arms and open_grippers have empty subjects and explicit arms. Return_arms always means each "
        "arm's own recorded initial pose, never an invented pose; it does not release. "
        "Final held is only a task condition when the source requires continued holding. "
        "Holding after an acquisition is not an always condition: it was false before the grasp. "
        "Use held as an acquisition effect, a during-step maintain condition or an explicit final "
        "condition, not whole-task scope unless held is also explicitly true initially. "
        "For a grasp used by intermediate steps with unspecified terminal handling, preserve the actions "
        "and do not invent a terminal hold, put-down or release. "
        "Open_grippers is only for an explicit gripper-state requirement, not redundant with place or release. "
        "Prefer specific operative directions over a conflicting general summary while retaining compatible goals. "
        "Include only requested actions and necessary implicit acquisitions; "
        "omit a separate move_held when a place already covers that motion with no distinct intermediate effect. "
        "do not add extra put-downs, intermediate releases, parks or moves. Preserve all source obligations, "
        "including identities referenced by initial location and later manipulation of the same object."
    ),
}


def _shape(value: Any, required: set[str], optional: set[str], label: str) -> None:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    missing = required - value.keys()
    unexpected = value.keys() - required - optional
    if missing or unexpected:
        raise ValueError(
            f"{label} requires {sorted(required)}; missing {sorted(missing)}; "
            f"unexpected {sorted(unexpected)}; optional {sorted(optional)}"
        )


def _values(value: Any, allowed: set[str], label: str, *, empty: bool = False) -> None:
    if (
        type(value) is not list
        or (not value and not empty)
        or any(not isinstance(item, str) or item not in allowed for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"{label} must be distinct allowed IDs: {sorted(allowed)}; got {str(value)[:180]}"
        )


def source_spans(instruction: str) -> dict[str, str]:
    """Index exact excerpts so a model can cite them without retranscription."""
    return {
        f"source_span_{index:02d}": span.strip()
        for index, span in enumerate(re.split(r"(?<=[.!?])\s+", instruction), 1)
        if span.strip()
    }


def _quote(value: Any, instruction: str, spans: dict[str, str]) -> str:
    if isinstance(value, str) and (value := value.strip()):
        if value in spans:
            return spans[value]
        if value in instruction:
            return value
        matches = {
            match[0]
            for match in re.finditer(re.escape(value), instruction, re.IGNORECASE)
            if match[0].casefold() == value.casefold()
        }
        if len(matches) == 1:
            return matches.pop()
    raise ValueError(f"Not a verbatim source span: {str(value)[:160]!r}")


def _expand_fact_shorthand(fact: Any) -> None:
    if type(fact) is not dict:
        return
    if len(fact) == 1:
        name, fields = next(iter(fact.items()))
        if (
            name in (_FACT_FIELDS.keys() | _RELATIONS | _ORIENTATIONS)
            and type(fields) is dict
            and "kind" not in fields
        ):
            fact.clear()
            fact.update(kind=name, **fields)
    if not isinstance(fact.get("kind"), str):
        return
    kind = fact["kind"]
    if fact.get("value", kind) != kind:
        return
    if kind in _RELATIONS:
        fact.update(kind="relation", value=kind)
    elif kind in _ORIENTATIONS:
        fact.update(kind="orientation", value=kind)


_OPERAND_COUNTS = {
    "pose_match": (None, 1),
    "distance_compare": (1, 2),
    "distance_change": (None, 1),
    "transferred": (None, 2),
    "hung_by": (1, 1),
    "grasped_at": (1, None),
    "articulated": (1, 1),
}
_DISTINCT_OPERANDS = {
    "relation",
    "distance_compare",
    "distance_change",
    "transferred",
    "articulated",
}


def _fact(fact: Any, roles: set[str]) -> None:
    if type(fact) is not dict or not isinstance(fact.get("kind"), str):
        raise ValueError(f"A fact requires a kind: {str(fact)[:160]}")
    kind = fact["kind"]
    if kind not in _FACT_FIELDS:
        raise ValueError(f"Unknown authoring fact {kind!r}")
    required, optional = _FACT_FIELDS[kind]
    _shape(fact, required | {"kind"}, optional, kind)
    for field in ("subjects", "references"):
        if field in fact:
            _values(fact[field], roles, f"{kind}.{field}")
    for field, values in _FACT_CHOICES.get(kind, {}).items():
        if field == "components":
            _values(fact[field], set(values), f"{kind}.{field}")
        elif field in fact and (
            not isinstance(fact[field], str) or fact[field] not in values
        ):
            raise ValueError(f"Invalid {kind}.{field}: {fact[field]!r}")
    if "arms" in fact:
        arms = fact["arms"]
        allowed = (
            _ARMS
            if kind in {"held", "grasped_at", "gripper_relation"}
            else _ARMS - {"auto"}
        )
        _values(arms, allowed, kind)
        if "auto" in arms and len(arms) != 1 or kind == "grasped_at" and len(arms) != 1:
            raise ValueError(
                f"{kind} requires compatible arms and one arm per grasp site"
            )
    if "frame" in fact and (
        not isinstance(fact["frame"], str) or fact["frame"] not in _FRAMES
    ):
        raise ValueError(f"Invalid reference frame for {kind}")

    subjects, refs = fact.get("subjects", []), fact.get("references", [])
    expected = _OPERAND_COUNTS.get(kind, (None, None))
    if kind == "relation" and fact["value"] in {"opposite_sides", "opposite_ends"}:
        expected = (2, 1)
    if kind == "orientation":
        if fact["value"] in _REFERENCED_POINTING:
            expected = (None, 1)
        if fact["value"] == "aligned_long_axis" and not refs:
            raise ValueError("aligned_long_axis requires a distinct reference role")
    if any(
        count is not None and len(ids) != count
        for ids, count in zip((subjects, refs), expected)
    ):
        raise ValueError(f"Invalid operand count for {kind}: {expected}")
    if _is_control_transition(fact) and len(subjects) != 1:
        raise ValueError("A control transition needs exactly one control part")
    if set(subjects) & set(refs) and (
        kind in _DISTINCT_OPERANDS
        or kind == "pose_match"
        and fact["epoch"] == "current"
        or kind == "orientation"
        and fact["value"] in _REFERENCED_POINTING | {"aligned_long_axis"}
    ):
        raise ValueError(f"{kind} requires distinct subject and reference roles")
    if "part" in fact and (
        fact["part"] not in roles or fact["part"] in subjects + refs
    ):
        raise ValueError(f"{kind} needs a distinct part role; declare its part_of link")
    if kind == "vertical_rise" and (
        type(fact["meters"]) not in (int, float)
        or not math.isfinite(fact["meters"])
        or fact["meters"] <= 0
    ):
        raise ValueError("vertical_rise requires a positive finite threshold")
    if (
        kind in {"ordered_row", "mutually_separate", "spaced_apart"}
        and len(subjects) < 2
    ):
        raise ValueError(f"{kind} needs multiple distinct subjects")
    if kind == "view_location" and (
        not {"lateral", "depth"} & fact.keys()
        or ("references" in fact or fact["frame"] == "reference")
        and len(refs) != 1
    ):
        raise ValueError(
            "view_location requires a side or depth and one role for a reference frame"
        )
    if kind == "transferred":
        quantity = fact["quantity"]
        if type(fact["all"]) is not bool or (
            quantity is not None and (type(quantity) is not int or quantity < 1)
        ):
            raise ValueError(
                "transferred requires boolean all and positive quantity or null"
            )


def _relation_pairs(fact: dict[str, Any], value: str) -> set[tuple[str, str]]:
    if fact["kind"] != "relation":
        return set()
    direct = fact["value"] == value
    # Swapping operands also swaps the anchor of a reference-local frame.
    if not direct and (
        fact.get("frame") == "reference" or _AXIS_INVERSES.get(fact["value"]) != value
    ):
        return set()
    return {
        (subject, reference) if direct else (reference, subject)
        for subject in fact["subjects"]
        for reference in fact["references"]
    }


class AuthoringEvidenceError(ValueError):
    """A well-formed source check has no matching structured graph evidence."""


def verify_spatial_claim(
    claim: Any, facts: list[dict[str, Any]], roles: dict[str, str]
) -> None:
    """Match a source spatial claim to typed relations between the named roles."""
    if type(claim) is dict:
        claim = dict(claim)
        _expand_fact_shorthand(claim)
    _shape(
        claim, {"kind", "value", "subjects", "references"}, {"frame"}, "spatial claim"
    )
    _fact(claim, set(roles))
    if claim["kind"] != "relation":
        raise ValueError("Source spatial claim must be a typed relation")
    value = claim["value"]
    frame = claim.get("frame")
    expected = _relation_pairs(claim, value)
    relations = [fact for fact in facts if fact["kind"] == "relation"]
    matching_frame = [
        fact for fact in relations if frame is None or fact.get("frame") == frame
    ]
    locations = {
        (subject, slot)
        for fact in relations
        if frame is None or fact.get("frame") in {None, frame}
        for subject, slot in _relation_pairs(fact, "at")
        if value in _SLOT_RELATIONS and roles[slot] == "target"
    }
    matches = [_relation_pairs(fact, value) for fact in matching_frame]
    # One joint relationship cannot be assembled from different groups.
    if value in _GROUP_RELATIONS:
        direct = expected if expected in matches else set()
    else:
        direct = set().union(*matches)
    direct.update(
        pair
        for relation in _STRONGER_RELATIONS.get(value, ())
        for fact in matching_frame
        for pair in _relation_pairs(fact, relation)
    )
    if value == "near_center_of":
        near = set().union(
            *(
                _relation_pairs(fact, "near")
                for fact in relations
                if frame is None or fact.get("frame") in {None, frame}
            )
        )
        centers = set().union(
            *(_relation_pairs(fact, "centered_on") for fact in matching_frame)
        )
        direct.update(
            (subject, reference)
            for anchor, reference in centers
            if roles[anchor] == "target"
            for subject, center in near
            if center == anchor
        )
    if any(
        (subject, reference) not in direct
        and not any(
            (other, reference) in direct for item, other in locations if item == subject
        )
        for subject, reference in expected
    ):
        raise AuthoringEvidenceError(
            f"Source spatial claim {value!r} needs matching typed relations for "
            f"{claim['subjects']} and {claim['references']}"
            + (f" in the {frame!r} frame" if frame else "")
            + "; view_location or labels are not enough"
        )


def verify_state_claim(
    claim: dict[str, Any],
    facts: list[dict[str, Any]],
    roles: dict[str, str],
    *,
    combine_released: bool = False,
) -> None:
    """Match a typed source state to the graph facts cited for its scope."""
    _fact(claim, set(roles))
    kind = claim["kind"]
    candidates = [fact for fact in facts if fact["kind"] == kind]
    # Unheld subjects are independent, but only combine facts at one snapshot.
    if (
        combine_released
        and kind == "released"
        and set(claim["subjects"])
        <= {subject for fact in candidates for subject in fact["subjects"]}
    ):
        return
    unordered = {"subjects", "arms", "references", "components"}
    if kind == "ordered_row":
        unordered.remove("subjects")
    if kind in {"distance_compare", "transferred"}:
        unordered.remove("references")
    member_field = _INDEPENDENT_STATE_MEMBERS.get(kind)

    def matches(fact: dict[str, Any]) -> bool:
        for key, value in claim.items():
            if key not in fact:
                return False
            actual = fact[key]
            if key == member_field:
                if not set(value).issubset(actual):
                    return False
            elif key in unordered:
                if set(value) != set(actual):
                    return False
            elif value != actual:
                return False
        return True

    if not any(matches(fact) for fact in candidates):
        raise AuthoringEvidenceError(
            f"Source state claim {claim['kind']!r} lacks matching evidence for "
            f"{claim.get('subjects', claim.get('arms'))!r} in the required scope"
        )


def _has_step_effect(step: dict[str, Any], fact: dict[str, Any]) -> bool:
    verb, kind, subjects = step["verb"], fact["kind"], step["subjects"]
    if verb == "place":
        return (
            kind in {"relation", "pose_match", "hung_by", "view_location"}
            and subjects[0] in fact["subjects"]
        )
    if verb == "transfer_contents":
        return kind == "transferred" and fact["references"][0] == subjects[0]
    if verb == "move_gripper":
        return kind == "gripper_relation" and set(fact["arms"]) == set(step["arms"])
    if verb == "articulate":
        return kind == "articulated" and fact["subjects"] == subjects
    return (
        kind == "actuated"
        and fact["value"] == "pressed"
        and fact["subjects"] == subjects
    )


def _is_control_transition(fact: dict[str, Any]) -> bool:
    return (
        fact.get("kind") == "actuated" and fact.get("value") == _OTHER_CONTROL_POSITION
    )


@dataclass(frozen=True)
class _ArmState:
    grips: frozenset[tuple[str, str]] = frozenset()
    parked: frozenset[str] = frozenset()
    opened: frozenset[str] = frozenset()

    def holders(self, subject: str) -> set[str]:
        return {arm for arm, role in self.grips if role == subject}


def _arm_options(arms: list[str]) -> tuple[tuple[str, ...], ...]:
    return (("left",), ("right",)) if arms == ["auto"] else (tuple(arms),)


def _advance_arms(
    state: _ArmState, verb: str, subjects: list[str], choice: tuple[str, ...]
) -> _ArmState | None:
    arms = set(choice)
    occupied = dict(state.grips)
    grips, parked, opened = state.grips, state.parked, state.opened
    if verb == "return_arms":
        return _ArmState(grips, parked | arms, opened)
    if verb == "open_grippers":
        return (
            None if arms & occupied.keys() else _ArmState(grips, parked, opened | arms)
        )
    if verb in _ACQUIRE_VERBS:
        subject = subjects[0]
        if subject in occupied.values() or arms & occupied.keys():
            return None
        grips |= {(arm, subject) for arm in arms}
    elif verb in _HELD_VERBS:
        subject = subjects[0]
        required = {choice[0]} if verb == "handover" else arms
        if state.holders(subject) != required:
            return None
        if verb in {"place", "release", "handover"}:
            grips -= {(arm, subject) for arm in required}
        if verb == "handover":
            if len(choice) != 2 or choice[1] in occupied:
                return None
            grips |= {(choice[1], subject)}
    else:
        held_subjects = set(occupied.values()) & set(subjects)
        if (
            (verb == "push" and held_subjects)
            or any(state.holders(subject) != arms for subject in held_subjects)
            or any(arm in occupied and occupied[arm] not in subjects for arm in arms)
        ):
            return None
    return _ArmState(
        grips, parked - arms, opened if verb in _ARM_VERBS else opened - arms
    )


def _intent_step_arms(plan: dict[str, Any]) -> list[set[str]]:
    """Check ordered arm facts and return the arms possible at each step."""
    states = {_ArmState()}
    sites: dict[str, dict[str, str]] = {}
    invariants = []
    transitions = []

    def require_fact(fact: dict[str, Any]) -> None:
        nonlocal states
        kind = fact["kind"]
        if kind == "held":
            required = set(fact["arms"]) - {"auto"}
            states = {
                state
                for state in states
                if all(
                    (owners := state.holders(subject)) and required <= owners
                    for subject in fact["subjects"]
                )
            }
            message = (
                "Held fact contradicts ordered acquisition/release or arm identity"
            )
        elif kind == "released":
            states = {
                state
                for state in states
                if not any(state.holders(subject) for subject in fact["subjects"])
            }
            message = "Released fact contradicts ordered holdings"
        elif kind == "grasped_at":
            subject, arm = fact["subjects"][0], fact["arms"][0]
            grasp = sites.get(subject, {})
            states = {
                state
                for state in states
                if any(
                    (arm == "auto" or side == arm)
                    and grasp.get(side, grasp.get("auto")) == fact["part"]
                    for side in state.holders(subject)
                )
            }
            message = "Grasp site is not witnessed by the acquisition of that arm"
        elif kind in {"arms_at_initial_pose", "grippers_open"}:
            required = set(fact["arms"])
            states = {
                state
                for state in states
                if required
                <= (state.parked if kind == "arms_at_initial_pose" else state.opened)
            }
            message = f"{kind} requires its explicit arm step"
        else:
            return
        if not states:
            raise ValueError(message)

    for condition in plan["conditions"]:
        scope, fact = condition["scope"], condition["fact"]
        if scope == "always":
            invariants.append(fact)
        elif scope == "initial" and fact["kind"] == "held":
            for subject in fact["subjects"]:
                feasible = set()
                for state in states:
                    owners = state.holders(subject)
                    for choice in _arm_options(fact["arms"]):
                        if owners == set(choice):
                            feasible.add(state)
                        elif not owners:
                            acquired = _advance_arms(state, "pick", [subject], choice)
                            if acquired is not None:
                                feasible.add(acquired)
                if not feasible:
                    raise ValueError(
                        "Initial held conditions disagree on arm ownership or capacity"
                    )
                states = feasible
    for fact in invariants:
        try:
            require_fact(fact)
        except ValueError as error:
            raise ValueError(
                f"An always condition must hold before the first step: {error}"
            ) from error

    for step in plan["steps"]:
        verb, subjects, effects = step["verb"], step["subjects"], step["effects"]
        for fact in step.get("maintain", ()):
            require_fact(fact)
        edges = [
            (state, choice, next_state)
            for state in states
            for choice in _arm_options(step["arms"])
            if (next_state := _advance_arms(state, verb, subjects, choice)) is not None
        ]
        if not edges:
            if verb in _ACQUIRE_VERBS:
                if all(len(state.grips) + len(step["arms"]) > 2 for state in states):
                    raise ValueError(
                        "More simultaneous grasps than available arms, including auto. "
                        "Release an earlier item before acquiring another if no arm is free."
                    )
                raise ValueError(
                    "Pick would reacquire a held subject or occupy an already holding arm"
                )
            if verb == "open_grippers":
                raise ValueError(
                    "Opening grippers with a held object requires a preceding release or placement"
                )
            if verb in _HELD_VERBS:
                raise ValueError(
                    f"{verb} requires an earlier acquisition of {subjects[0]!r} by the same arm(s)"
                    + (" and a free destination arm" if verb == "handover" else "")
                )
            if verb == "push" and any(
                role in subjects for state in states for _, role in state.grips
            ):
                raise ValueError("A push cannot move a held object")
            raise ValueError(f"{verb} contradicts the current arm holdings")
        states = {end for _, _, end in edges}
        if verb in _ACQUIRE_VERBS:
            sites[subjects[0]] = {
                fact["arms"][0]: fact["part"]
                for fact in effects
                if fact["kind"] == "grasped_at"
            }
        if verb in {"place", "release", "handover"}:
            sites.pop(subjects[0], None)
        for fact in (*effects, *invariants):
            require_fact(fact)
        transitions.append(
            [(start, choice, end) for start, choice, end in edges if end in states]
        )
    for condition in plan["conditions"]:
        if condition["scope"] == "final":
            require_fact(condition["fact"])

    # A later use of a held object can resolve an earlier auto choice.
    touched = []
    for edges in reversed(transitions):
        viable = [(start, choice) for start, choice, end in edges if end in states]
        touched.append({arm for _, choice in viable for arm in choice})
        states = {start for start, _ in viable}
    return list(reversed(touched))


def _role_ancestry(role: str, parents: dict[str, str]) -> set[str]:
    lineage = {role}
    while role in parents:
        role = parents[role]
        lineage.add(role)
    return lineage


def _intent_phases(steps: list[dict[str, Any]]) -> Iterator[tuple[int, int]]:
    """Return the inclusive start and exclusive end of each authoring phase."""
    for _, phase in groupby(
        enumerate(steps), key=lambda row: row[1].get("parallel_group", row[0])
    ):
        indices = [index for index, _ in phase]
        yield indices[0], indices[-1] + 1


def _verify_lift_then_depart(instruction: str, steps: list[dict[str, Any]]) -> None:
    for match in _LIFT_AND_DEPART.finditer(instruction):
        if _NEGATED_ACTION_PREFIX.search(instruction[: match.start()]):
            continue
        lift, depart = (phrase.strip() for phrase in match.groups())
        short = bool(_SHORT_DISTANCE.search(depart))
        lifted: set[tuple[str, tuple[str, ...]]] = set()
        for step in steps:
            verb = step["verb"]
            if verb in {"release", "place", "handover"}:
                lifted = {key for key in lifted if key[0] != step["subjects"][0]}
            if verb != "move_held":
                continue
            key = (step["subjects"][0], tuple(sorted(step["arms"])))
            direction, quote = step.get("direction"), step["quote"]
            if direction == "up" and lift in quote:
                lifted.add(key)
            if (
                direction == "away"
                and key in lifted
                and depart in quote
                and (not short or step.get("extent") == "short")
            ):
                break
        else:
            raise ValueError(
                "The quoted lift and move away require separate move_held steps: up, then away "
                "with the same held object and arms, each citing its source action or the full clause"
                + (
                    "; mark the requested short displacement with extent:short"
                    if short
                    else ""
                )
            )


def _normalize_held_contact(
    step: dict[str, Any],
    parents: dict[str, str],
    role_kinds: dict[str, str],
    scene_objects: dict[str, Any],
) -> None:
    # Some proposals already state the exact contact, but also label it as joint actuation.
    part = step["subjects"][0]
    owner = parents.get(part)
    if (
        owner is None
        or role_kinds[part] in {"button", "articulation"}
        or any(
            scene_objects.get(role, {}).get("asset_kind") == "articulated"
            for role in _role_ancestry(part, parents)
        )
        or dict(kind="held", subjects=[owner], arms=step["arms"])
        not in step.get("maintain", ())
    ):
        return
    effects = step["effects"]
    press = dict(kind="actuated", subjects=[part], value="pressed")
    if len(effects) != 2 or press not in effects:
        return
    contact = effects[1] if effects[0] == press else effects[0]
    if (
        isinstance(contact, dict)
        and contact.get("kind") == "relation"
        and contact.get("value") == "touching"
        and contact.get("subjects") == [part]
    ):
        step.update(verb="move_held", subjects=[owner], effects=[contact])


def validate_intent(
    plan: Any, instruction: str, *, scene_objects: dict[str, Any]
) -> dict[str, Any]:
    """Validate typed authoring syntax, role identity and explicit holdings."""
    _shape(plan, {"roles", "steps", "conditions"}, set(), "semantic_plan")
    plan = deepcopy(plan)
    spans = source_spans(instruction)
    for name in plan:
        if type(plan[name]) is not list or (name != "conditions" and not plan[name]):
            raise ValueError(
                f"semantic_plan.{name} must be an array (nonempty for roles and steps)"
            )

    role_kinds: dict[str, str] = {}
    parents: dict[str, str] = {}
    for role in plan["roles"]:
        _shape(role, {"id", "kind", "description"}, {"part_of", "asset_kind"}, "role")
        name = role["id"]
        if (
            not isinstance(name, str)
            or _SCENE_ROLE_ID.fullmatch(name) is None
            or name in role_kinds
            or not isinstance(role["kind"], str)
            or role["kind"] not in _ROLE_KINDS
            or not isinstance(role["description"], str)
            or not role["description"].strip()
        ):
            raise ValueError(
                "Roles require distinct ASCII identifiers, one scalar kind string and a description"
            )
        role_kinds[name] = role["kind"]
        # Older candidates can carry a model hint, not a caller declaration.
        role.pop("asset_kind", None)
        parent = role.get("part_of")
        if parent is None:
            role.pop("part_of", None)
        elif not isinstance(parent, str):
            raise ValueError("Role part_of must be one owning role ID")
        else:
            parents[name] = parent
    roles = set(role_kinds)
    for name in parents:
        parent: str | None = parents[name]
        seen = {name}
        while parent is not None:
            if parent in seen or parent not in roles:
                raise ValueError("Part ownership needs a declared, acyclic parent role")
            seen.add(parent)
            parent = parents.get(parent)
    for name, role in scene_objects.items():
        if name not in roles and any(
            item["id"] not in scene_objects
            and item["description"].casefold() == role["description"].casefold()
            for item in plan["roles"]
        ):
            raise ValueError(
                f"Use scene-context ID {name!r} for its corresponding role"
            )

    grasp_sites: dict[str, dict[str, str]] = {}
    manipulated: set[str] = set()
    seated: set[tuple[str, str]] = set()
    tightened: set[tuple[str, str]] = set()
    invariants = []

    def check_tightening(fact: dict[str, Any]) -> None:
        if not _relation_pairs(fact, "tightened_on") <= tightened:
            raise ValueError(
                "Tightened state has no prior seating and turn for the same role pair"
            )

    def check_fact(fact: Any) -> None:
        _expand_fact_shorthand(fact)
        _fact(fact, roles)
        if (
            fact["kind"] == "hung_by"
            and parents.get(fact["part"], fact["subjects"][0]) != fact["subjects"][0]
        ):
            raise ValueError("The hanging part belongs to a different subject")
        if (
            fact["kind"] == "grasped_at"
            and parents.get(fact["part"]) != fact["subjects"][0]
        ):
            raise ValueError("The grasp site must be part_of the acquired subject")
        if fact["kind"] == "articulated":
            part, joint = fact["subjects"][0], fact["references"][0]
            if (
                role_kinds[joint] != "articulation"
                or joint not in parents
                or (parents[joint] not in {part, parents.get(part)})
            ):
                raise ValueError(
                    "Articulation must reference a distinct joint of the moving part or its owning object; "
                    f"declare the part_of links for {part!r} and joint {joint!r} explicitly"
                )
        if fact["kind"] in {"articulated", "actuated"}:
            for subject in fact["subjects"]:
                require_articulation(subject, scene_objects, parents)
        if _is_control_transition(fact) and fact["subjects"][0] not in parents:
            raise ValueError("Link the specific control part to its articulated owner")

    for condition in plan["conditions"]:
        _shape(condition, {"scope", "fact", "quote"}, set(), "condition")
        condition["quote"] = _quote(condition["quote"], instruction, spans)
        scope, fact = condition["scope"], condition["fact"]
        if scope not in {"initial", "always", "final"}:
            raise ValueError("Condition scope must be initial, always or final")
        check_fact(fact)
        if _is_control_transition(fact) and scope != "final":
            raise ValueError(
                "A transition from the initial position is an effect or final goal"
            )
        if scope == "always":
            invariants.append(fact)
        if scope == "initial":
            if fact["kind"] == "distance_change":
                raise ValueError(
                    "distance_change compares against initial distance; use distance_compare for a comparison in the initial state"
                )
            seated.update(_relation_pairs(fact, "seated_on"))
            tightened.update(_relation_pairs(fact, "tightened_on"))
    seated.update(tightened)
    for fact in invariants:
        try:
            check_tightening(fact)
        except ValueError as error:
            raise ValueError(
                f"An always condition must hold before the first step: {error}"
            ) from error

    ordered_steps = []
    for step in plan["steps"]:
        _shape(
            step,
            {"verb", "subjects", "arms", "effects", "quote"},
            {"direction", "extent", "maintain", "release_quote", "parallel_group"},
            "step",
        )
        verb = step["verb"]
        if not isinstance(verb, str) or verb not in _VERBS:
            raise ValueError(f"Unknown authoring step {verb!r}")
        if "direction" in step and (
            verb not in _DIRECTIONAL_VERBS
            or step["direction"] not in _DIRECTIONS
            or (verb == "pick" and step["direction"] != "up")
        ):
            raise ValueError(f"Invalid movement direction for {verb}")
        if "extent" in step and (
            verb != "move_held" or step["extent"] != "short" or "direction" not in step
        ):
            raise ValueError("extent:short requires a directed move_held")
        step["quote"] = _quote(step["quote"], instruction, spans)
        _values(step["subjects"], roles, f"{verb}.subjects", empty=verb in _ARM_VERBS)
        _values(step["arms"], _ARMS, f"{verb}.arms")
        subjects, arms = step["subjects"], set(step["arms"])
        if "auto" in arms and (
            len(arms) != 1 or verb in {"return_arms", "open_grippers", "handover"}
        ):
            raise ValueError(
                f"{verb} requires either auto alone or valid explicit arms"
            )
        if verb in _ARM_VERBS and subjects:
            raise ValueError(f"{verb} uses arms, not object subjects")
        if verb in _ACQUIRE_VERBS | _HELD_VERBS | {"press"} and len(subjects) != 1:
            hint = (
                "; explicitly simultaneous picks of different objects use a shared parallel_group"
                if verb == "pick"
                else ""
            )
            raise ValueError(f"{verb} needs one object identity per occurrence{hint}")
        effects, maintained = step["effects"], step.get("maintain", [])
        if type(effects) is not list or type(maintained) is not list:
            raise ValueError("step.effects and step.maintain must be lists of facts")
        grouped = "parallel_group" in step
        if grouped and (
            not isinstance(step["parallel_group"], str)
            or not _SCENE_ROLE_ID.fullmatch(step["parallel_group"])
            or verb != "pick"
            or len(arms) != 1
            or "auto" in arms
            or maintained
        ):
            raise ValueError(
                "A parallel group contains independent single-arm picks with explicit arms"
            )
        for fact in maintained:
            check_fact(fact)
            if _is_control_transition(fact):
                raise ValueError("A control transition must be produced as an effect")
            check_tightening(fact)
        if verb == "press":
            _normalize_held_contact(step, parents, role_kinds, scene_objects)
            verb, subjects, effects = step["verb"], step["subjects"], step["effects"]
        for fact in effects:
            check_fact(fact)
            if grouped and (
                fact["kind"] not in {"held", "grasped_at"}
                or fact["subjects"] != subjects
                or fact["arms"] != step["arms"]
            ):
                raise ValueError(
                    "A parallel pick can only establish its own held or grasped_at effect"
                )
            if _is_control_transition(fact) and (
                verb not in {"push", "actuate"}
                or fact["subjects"] != subjects
                or len(arms) != 1
            ):
                raise ValueError(
                    "Only a single-arm push or actuate on the same control can produce its transition"
                )
        if verb in {
            "place",
            "transfer_contents",
            "articulate",
            "move_gripper",
            "press",
        } and not any(_has_step_effect(step, fact) for fact in effects):
            raise ValueError(
                f"{verb} requires a matching typed effect for its subject or arms"
            )
        if verb == "actuate":
            for subject in subjects:
                require_articulation(subject, scene_objects, parents)
        if "release_quote" in step:
            step["release_quote"] = _quote(step["release_quote"], instruction, spans)
            if verb != "place":
                raise ValueError("A release_quote belongs to placement")
        if verb == "release" and ordered_steps and not maintained:
            previous = ordered_steps[-1]
            released = dict(kind="released", subjects=subjects)
            if (
                previous["verb"] == "place"
                and previous["subjects"] == subjects
                and set(previous["arms"]) == arms
                and "release_quote" not in previous
                and all(fact == released for fact in effects)
            ):
                # The preceding placement already releases the same object.
                if released not in previous["effects"]:
                    previous["effects"].append(released)
                previous["release_quote"] = step["quote"]
                continue
        if verb == "place" and "release_quote" in step:
            released = dict(kind="released", subjects=subjects)
            if released not in effects:
                effects.append(released)
        sites = [fact for fact in effects if fact["kind"] == "grasped_at"]
        if sites and (
            verb not in _ACQUIRE_VERBS
            or any(
                fact["subjects"] != subjects or fact["arms"][0] not in arms
                for fact in sites
            )
            or len({fact["arms"][0] for fact in sites}) != len(sites)
        ):
            raise ValueError(
                "Assign each grasp site to an acquiring arm in the acquisition step"
            )
        if verb in {"move_held", "place", "push"}:
            moved = set(subjects)
            seated = {pair for pair in seated if moved.isdisjoint(pair)}
            tightened = {pair for pair in tightened if moved.isdisjoint(pair)}
        new_tightened = set().union(
            *(_relation_pairs(fact, "tightened_on") for fact in effects)
        )
        if new_tightened and (verb != "turn" or not new_tightened <= seated):
            raise ValueError(
                "Tightening requires a turn after seating the same part on the same reference"
            )
        tightened.update(new_tightened)
        seated.update(*(_relation_pairs(fact, "seated_on") for fact in effects))

        if verb in _ACQUIRE_VERBS:
            grasp_sites[subjects[0]] = {fact["arms"][0]: fact["part"] for fact in sites}
        # A fully specified grasp does not imply contact with sibling parts.
        contacts = (
            grasp_sites.get(subjects[0], {})
            if len(subjects) == 1
            and (
                verb in _ACQUIRE_VERBS
                or verb in {"move_held", "orient", "transfer_contents", "release"}
            )
            else {}
        )
        manipulated.update(
            contacts.values() if len(contacts) == len(arms) else subjects
        )
        if verb in {"place", "release", "handover"}:
            grasp_sites.pop(subjects[0], None)
        for fact in (*effects, *invariants):
            check_tightening(fact)
        ordered_steps.append(step)

    seen_groups = set()
    for start, end in _intent_phases(ordered_steps):
        group = ordered_steps[start].get("parallel_group")
        if group is None:
            continue
        if group in seen_groups or end - start != 2:
            raise ValueError("A parallel group must contain exactly two adjacent picks")
        first, second = ordered_steps[start:end]
        if first["arms"] == second["arms"] or not _role_ancestry(
            first["subjects"][0], parents
        ).isdisjoint(_role_ancestry(second["subjects"][0], parents)):
            raise ValueError(
                "A parallel group must use distinct arms and independent objects"
            )
        seen_groups.add(group)
    _verify_lift_then_depart(instruction, ordered_steps)
    plan["steps"] = ordered_steps
    for condition in plan["conditions"]:
        scope, fact = condition["scope"], condition["fact"]
        if scope == "final":
            check_tightening(fact)
        if scope != "initial" and fact["kind"] in {"untouched", "untouched_except"}:
            protected = set(fact["subjects"])
            if fact["kind"] == "untouched":
                owners = set().union(
                    *(_role_ancestry(role, parents) for role in protected)
                )
                conflict = {
                    role
                    for role in manipulated
                    if role in owners or _role_ancestry(role, parents) & protected
                }
            else:
                conflict = {
                    role
                    for role in manipulated
                    if not _role_ancestry(role, parents) & protected
                }
            if conflict:
                raise ValueError(
                    f"Ordered operations touch protected roles: {sorted(conflict)}"
                )
    _intent_step_arms(plan)
    return plan


def compile_intent(
    plan: dict[str, Any], instruction: str, task_id: str, scene_objects: dict[str, Any]
) -> tuple[TaskSpec, dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Project validated intent into a normative TaskSpec and graph-only nodes."""
    roles = {role["id"]: role for role in plan["roles"]}
    descriptions = {
        name: scene_objects.get(name, {}).get("description", role["description"])
        for name, role in roles.items()
    }
    definitions = {
        section: [] for section in ("init", "invariants", "goal", "milestones")
    }
    nodes, groups, clauses = [], [], []

    def predicate(identifier: str, fact: dict[str, Any]) -> dict[str, Any]:
        return dict(id=identifier, predicate="semantic_goal", arguments=fact)

    def clause(
        identifier: str, target: str, source: str, *, pending: bool = False
    ) -> None:
        clauses.append(
            dict(
                id=f"source.{identifier}",
                target=target,
                quote=source.strip(),
                origin="explicit",
                resolution="pending_binding" if pending else "resolved",
            )
        )

    graph_conditions: dict[str, list[dict[str, Any]]] = {
        "initial": [],
        "always": [],
        "final": [],
    }
    for index, condition in enumerate(plan["conditions"], 1):
        scope, fact = condition["scope"], condition["fact"]
        identifier = f"condition_{index:02d}"
        graph_conditions[scope].append(fact)
        key = {"initial": "init", "always": "invariants", "final": "goal"}[scope]
        definitions[key].append(predicate(identifier, fact))
        target = (
            identifier if scope == "final" else (fact.get("subjects") or ["step_01"])[0]
        )
        clause(
            identifier,
            target,
            condition["quote"],
            pending=fact["kind"]
            in {"arms_at_initial_pose", "pose_match", "view_location"},
        )

    prior: list[str] = []
    predecessors = {}
    for start, end in _intent_phases(plan["steps"]):
        members = [f"step_{index + 1:02d}" for index in range(start, end)]
        predecessors.update(dict.fromkeys(members, prior))
        prior = members
    for index, step in enumerate(plan["steps"], 1):
        identifier = f"step_{index:02d}"
        arguments = dict(step)
        verb = arguments.pop("verb")
        quote = arguments.pop("quote")
        release_quote = arguments.pop("release_quote", None)
        if verb == "return_arms":
            arguments.update(
                pose_reference="each_arm_initial_pose", binding_status="pending"
            )
        incoming = predecessors[identifier]
        metadata = dict(id=identifier, task_type=verb, depends_on=incoming)
        nodes.append(
            dict(
                **metadata,
                call=dict(
                    kind="registered",
                    call_id=f"{_AUTHORING_INTENT_PREFIX}{verb}",
                    arguments=dict(arguments, quote=quote.strip()),
                    resources={},
                ),
                task_instance_id=identifier,
                role=verb,
            )
        )
        groups.append(dict(**metadata, node_ids=[identifier], success={}))
        definitions["milestones"].append(
            dict(
                id=identifier,
                after=incoming,
                achieve=[
                    predicate(f"achieve_{identifier}", dict(verb=verb, **arguments))
                ],
            )
        )
        clause(identifier, identifier, quote, pending=verb == "return_arms")
        if release_quote:
            clause(f"{identifier}.release", identifier, release_quote)
    terminal = (
        {"terminal_step": prior[0]} if len(prior) == 1 else {"terminal_steps": prior}
    )
    if not definitions["goal"]:
        completion = "final_step_completed"
        definitions["goal"].append(predicate(completion, terminal))
        clause(completion, completion, plan["steps"][-1]["quote"])
    spec = TaskSpec.from_dict(
        dict(
            task_id=task_id,
            instruction=instruction,
            roles=[
                dict(
                    id=name,
                    kind=role["kind"],
                    properties=(
                        {"part_of": role["part_of"]} if "part_of" in role else {}
                    ),
                )
                for name, role in roles.items()
            ],
            requirements=[
                scene_objects_requirement(
                    scene_objects,
                    {
                        name: descriptions[name]
                        for name, role in roles.items()
                        if "part_of" not in role
                        and role["kind"] not in {"target", "robot"}
                    },
                )
            ],
            **definitions,
        )
    )
    graph = validate_semantic_task_graph(
        dict(
            schema_version=SEMANTIC_TASK_GRAPH_SCHEMA,
            task_id=task_id,
            instruction=instruction,
            planner_route="hybrid",
            integration_fingerprint="0" * 64,
            targets={},
            nodes=nodes,
            task_groups=groups,
            success=dict(authoring_only=True, conditions=graph_conditions, **terminal),
        )
    )
    return spec, graph, clauses, descriptions
