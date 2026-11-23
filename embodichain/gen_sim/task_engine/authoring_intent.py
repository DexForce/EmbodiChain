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

from copy import deepcopy
import math
import re
from typing import Any

from ..task_spec import TaskSpec, validate_task_spec
from .semantic_graph import (
    _AUTHORING_INTENT_PREFIX,
    SEMANTIC_TASK_GRAPH_SCHEMA,
    validate_semantic_task_graph,
)

__all__: list[str] = []

_ID = re.compile(r"[a-z][a-z0-9_]*")
_ROLE_KINDS = {
    "object",
    "target",
    "surface",
    "container",
    "tool",
    "articulation",
    "button",
    "robot",
}
_ARMS = {"left", "right", "auto"}
_ARM_VERBS = {"return_arms", "open_grippers", "move_gripper"}
_ACQUIRE_VERBS = {"pick", "grasp"}
_DIRECTIONAL_VERBS = {
    "pick",
    "move_held",
    "move_gripper",
    "press",
    "push",
    "articulate",
}
_DIRECTIONS = {"up", "down", "toward", "away", "left", "right"}
_GRIPPER_RELATIONS = {"above", "touching", "clear_of"}
_VERBS = _ARM_VERBS | {
    *_ACQUIRE_VERBS,
    "move_held",
    "place",
    "push",
    "transfer_contents",
    "orient",
    "turn",
    "release",
    "handover",
    "actuate",
    "press",
    "articulate",
}
_RELATIONS = {
    "at",
    "on",
    "inside",
    "not_inside",
    "above",
    "below",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "next_to",
    "near",
    "near_center_of",
    "centered_on",
    "centered_above",
    "between",
    "opposite_sides",
    "opposite_ends",
    "near_left_end",
    "near_right_end",
    "separate_from",
    "supported_by",
    "on_top_faces",
    "spans",
    "aligned_with",
    "seated_on",
    "tightened_on",
    "compatible_with",
    "fixed_to",
    "touching",
}
_ORIENTATIONS = {
    "upright",
    "vertical",
    "flat",
    "level",
    "opening_up",
    "opening_down",
    "tilted",
    "left_to_right",
    "front_to_back",
    "aligned_long_axis",
    "upright_cross_section",
    "lying_on_side",
    "pointing_left",
    "pointing_right",
    "pointing_up",
    "pointing_down",
    "pointing_forward",
    "pointing_backward",
    "pointing_toward",
    "pointing_away_from",
}
_REFERENCED_POINTING = {"pointing_toward", "pointing_away_from"}
_FRAMES = {"head_camera", "table", "robot", "world", "reference"}
_LATERAL = {"left", "right", "center", "far_left", "far_right"}
_NEAR_REGIONS = {"near_center_of", "near_left_end", "near_right_end"}
_AXIS_RELATIONS = {"left_of", "right_of", "in_front_of", "behind", "above", "below"}
_SLOT_RELATIONS = {
    "near",
    "next_to",
    "on",
    "inside",
    "centered_on",
    "centered_above",
    *_NEAR_REGIONS,
    *_AXIS_RELATIONS,
}
_GROUP_RELATIONS = {"between", "opposite_sides", "opposite_ends", "spans"}

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

# Predicate vocabulary is shared by the proposing model, validator and audit.
_PROMPT = {
    "shape": {
        "roles": [
            {
                "id": "snake_case",
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
        "view_location.depth": ["near", "far", "middle"],
        "pose_match.epoch": ["initial", "current"],
        "pose_match.components": ["position", "orientation"],
        "vertical_rise.comparison": ["eq", "gt", "ge", "lt", "le"],
        "distance_compare.comparison": ["gt", "lt"],
        "distance_change.comparison": ["gt", "lt"],
        "distance_change.epoch": ["initial"],
        "gripper_relation.value": sorted(_GRIPPER_RELATIONS),
        "articulated.mechanism": ["hinge", "slide"],
        "articulated.value": ["open", "closed"],
        "transferred.material": ["solid", "liquid"],
        "transferred.quantity": "positive integer count, or null only if no item count is specified",
        "actuated.value": ["open", "closed", "pressed", "turned"],
    },
    "semantics": (
        "A fact is a JSON object with a single kind string and exactly its listed fields. "
        "For each quote, use one source_spans key, or copy one contiguous source span verbatim. "
        "Subjects/references are lists of stable role IDs. Use distinct role IDs for separate instances "
        "and destination slots. A role may have optional part_of: owning_role_id. Do not guess poses. "
        "Declare both the moving part and its joint, and every specified grip site, as roles with "
        "explicit part_of links. Mentioning ownership only in a description is not a binding. "
        "Use optional role asset_kind:rigid|articulated for explicit asset declarations. "
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
        "For stability of a structure assembled by the task, list the actual assembled roles "
        "in stable.subjects; do not substitute a new role describing the finished structure. "
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
        "A simultaneous acquisition is one pick with both arms "
        "and separate grasped_at facts for each specified site. Keep both arms on a bimanual hold/move. "
        "For articulated movement, grasp the moving part at the specified site, articulate that same held "
        "part with articulated referencing a distinct articulation role for its joint, part_of the "
        "moving part or its owner, plus the mechanism, then release if requested. "
        "For a press without grasping, move_gripper with gripper_relation:above at the actual site, "
        "press that site (actuated:pressed), then move_gripper with gripper_relation:clear_of if requested. "
        "When the source describes a button on a surface, the press effect names that button, "
        "not the whole surface used for approach. "
        "Use optional step direction on move_held, move_gripper, press, push or articulate when the "
        "source specifies a direction. Lifting away has direction up; clear_of describes leaving contact, "
        "not its direction. Move_gripper has no object subjects. A held tool can touch a "
        "specific face with relation:touching between its part and the contact target; this is not place. "
        "For non-displacement while pressing, keep the parent object's initial pose instead of claiming it is untouched. "
        "a transfer has the held source container as its step subject. Place moves a held subject to its "
        "declared destination and releases it "
        "(including seating, nesting or hanging); its released effect can witness release after placement. "
        "Use move_held for a held destination or lift. "
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
    if "arms" in fact:
        _values(
            fact["arms"],
            (
                _ARMS
                if kind in {"held", "grasped_at", "gripper_relation"}
                else _ARMS - {"auto"}
            ),
            kind,
        )
        if "auto" in fact["arms"] and len(fact["arms"]) != 1:
            raise ValueError("auto cannot accompany an explicit arm")
    if "frame" in fact and fact["frame"] not in _FRAMES:
        raise ValueError(f"Invalid reference frame for {kind}")
    if kind == "relation":
        if fact["value"] not in _RELATIONS:
            raise ValueError("Unknown spatial relation")
        if set(fact["subjects"]) & set(fact["references"]):
            raise ValueError("Spatial relations must link distinct roles")
        if fact["value"] in {"opposite_sides", "opposite_ends"} and (
            len(fact["subjects"]) != 2 or len(fact["references"]) != 1
        ):
            raise ValueError(f"{fact['value']} requires two subjects and one reference")
    if kind == "orientation":
        if fact["value"] not in _ORIENTATIONS:
            raise ValueError("Unknown orientation intent")
        if fact["value"] in _REFERENCED_POINTING and (
            len(fact.get("references", ())) != 1
            or set(fact["subjects"]) & set(fact["references"])
        ):
            raise ValueError(f"{fact['value']} requires one distinct target role")
        if fact["value"] == "aligned_long_axis" and (
            not fact.get("references")
            or set(fact["subjects"]) & set(fact["references"])
        ):
            raise ValueError(
                "aligned_long_axis requires the distinct reference role whose axis is matched"
            )
    if kind == "pose_match":
        _values(
            fact["components"], {"position", "orientation"}, "pose_match.components"
        )
        if (
            fact["epoch"] not in {"initial", "current"}
            or len(fact["references"]) != 1
            or (
                fact["epoch"] == "current"
                and set(fact["subjects"]) & set(fact["references"])
            )
        ):
            raise ValueError("pose_match needs a single referenced pose and its epoch")
    if kind == "vertical_rise" and (
        fact["comparison"] not in {"eq", "gt", "ge", "lt", "le"}
        or type(fact["meters"]) not in (int, float)
        or not math.isfinite(fact["meters"])
        or fact["meters"] <= 0
    ):
        raise ValueError(
            "vertical_rise needs a positive metric threshold and comparison"
        )
    if kind == "distance_compare" and (
        fact["comparison"] not in {"gt", "lt"}
        or len(fact["subjects"]) != 1
        or len(fact["references"]) != 2
        or fact["subjects"][0] in fact["references"]
    ):
        raise ValueError(
            "distance_compare needs distinct subject, anchor and comparison item"
        )
    if kind == "distance_change" and (
        fact["comparison"] not in {"gt", "lt"}
        or fact["epoch"] != "initial"
        or len(fact["references"]) != 1
        or set(fact["subjects"]) & set(fact["references"])
    ):
        raise ValueError(
            "distance_change needs one distinct reference and comparison to the initial distance"
        )
    if kind == "ordered_row" and len(fact["subjects"]) < 2:
        raise ValueError("ordered_row needs multiple distinct ordered subjects")
    if kind in {"mutually_separate", "spaced_apart"} and len(fact["subjects"]) < 2:
        raise ValueError(f"{kind} needs at least two distinct subjects")
    if kind == "view_location" and (
        not {"lateral", "depth"} & fact.keys()
        or ("lateral" in fact and fact["lateral"] not in _LATERAL)
        or ("depth" in fact and fact["depth"] not in {"near", "far", "middle"})
        or ("references" in fact and len(fact["references"]) != 1)
        or (fact["frame"] == "reference" and "references" not in fact)
    ):
        raise ValueError(
            "view_location needs a valid side or depth and one role for a reference frame"
        )
    if kind == "transferred" and (
        fact["material"] not in {"solid", "liquid"}
        or len(fact["references"]) != 2
        or set(fact["subjects"]) & set(fact["references"])
        or type(fact["all"]) is not bool
        or (
            fact["quantity"] is not None
            and (type(fact["quantity"]) is not int or fact["quantity"] < 1)
        )
    ):
        raise ValueError(
            "transferred needs distinct contents, source and recipient, with a valid quantity"
        )
    if kind == "hung_by" and (
        len(fact["subjects"]) != 1
        or len(fact["references"]) != 1
        or fact["part"] not in roles
        or fact["part"] in fact["subjects"] + fact["references"]
    ):
        raise ValueError("hung_by needs a subject, a distinct part and one hook role")
    if kind == "grasped_at" and (
        len(fact["subjects"]) != 1
        or len(fact["arms"]) != 1
        or fact["part"] not in roles
        or fact["part"] in fact["subjects"]
    ):
        raise ValueError(
            "grasped_at needs a subject, one arm and a distinct site role; "
            f"declare {fact['part']!r} with part_of set to its owner in {fact['subjects']!r}"
        )
    if kind == "gripper_relation" and fact["value"] not in _GRIPPER_RELATIONS:
        raise ValueError("Unknown gripper relation")
    if kind == "articulated" and (
        fact["mechanism"] not in {"hinge", "slide"}
        or fact["value"] not in {"open", "closed"}
        or len(fact["subjects"]) != 1
        or len(fact["references"]) != 1
        or fact["subjects"] == fact["references"]
    ):
        raise ValueError(
            "articulated needs one moving part, one distinct joint and its mechanism and state"
        )
    if kind == "actuated" and fact["value"] not in {
        "open",
        "closed",
        "pressed",
        "turned",
    }:
        raise ValueError("Unknown actuation state")


def _relation_pairs(fact: dict[str, Any], value: str) -> set[tuple[str, str]]:
    if fact["kind"] != "relation" or fact["value"] != value:
        return set()
    return {
        (subject, other) for subject in fact["subjects"] for other in fact["references"]
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
    _shape(claim, {"kind", "value", "subjects", "references"}, set(), "spatial claim")
    _fact(claim, set(roles))
    if claim["kind"] != "relation":
        raise ValueError("Source spatial claim must be a typed relation")
    value = claim["value"]
    expected = _relation_pairs(claim, value)
    relations = [fact for fact in facts if fact["kind"] == "relation"]
    locations = {
        (subject, slot)
        for fact in relations
        for subject, slot in _relation_pairs(fact, "at")
        if value in _SLOT_RELATIONS and roles[slot] == "target"
    }
    matches = [_relation_pairs(fact, value) for fact in relations]
    # One joint relationship cannot be assembled from different groups.
    if value in _GROUP_RELATIONS:
        direct = expected if expected in matches else set()
    else:
        direct = set().union(*matches)
    if value == "above":
        direct.update(
            pair
            for fact in relations
            for pair in _relation_pairs(fact, "centered_above")
        )
    if value == "near":
        direct.update(
            pair
            for fact in relations
            for relation in (*_NEAR_REGIONS, "next_to")
            for pair in _relation_pairs(fact, relation)
        )
    if value == "on":
        direct.update(
            pair
            for fact in relations
            for relation in ("on_top_faces", "centered_on", "supported_by", "seated_on")
            for pair in _relation_pairs(fact, relation)
        )
    if value == "near_center_of":
        near = set().union(*(_relation_pairs(fact, "near") for fact in relations))
        centers = set().union(
            *(_relation_pairs(fact, "centered_on") for fact in relations)
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
            f"{claim['subjects']} and {claim['references']}; view_location or labels are not enough"
        )


def verify_state_claim(
    claim: dict[str, Any], facts: list[dict[str, Any]], roles: dict[str, str]
) -> None:
    """Match a typed source state to the graph facts cited for its scope."""
    _fact(claim, set(roles))
    unordered = {"subjects", "arms", "references", "components"}
    if claim["kind"] == "ordered_row":
        unordered.remove("subjects")
    if claim["kind"] in {"distance_compare", "transferred"}:
        unordered.remove("references")

    def matches(fact: dict[str, Any]) -> bool:
        return all(
            key in fact
            and (
                set(value) == set(fact[key]) if key in unordered else value == fact[key]
            )
            for key, value in claim.items()
        )

    if not any(matches(fact) for fact in facts if fact["kind"] == claim["kind"]):
        raise AuthoringEvidenceError(
            f"Source state claim {claim['kind']!r} lacks matching evidence for "
            f"{claim.get('subjects', claim.get('arms'))!r} in the required scope"
        )


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
            or _ID.fullmatch(name) is None
            or name in role_kinds
            or not isinstance(role["kind"], str)
            or role["kind"] not in _ROLE_KINDS
            or not isinstance(role["description"], str)
            or not role["description"].strip()
        ):
            raise ValueError(
                "Roles require distinct snake_case IDs, one scalar kind string and a description"
            )
        role_kinds[name] = role["kind"]
        if "asset_kind" in role and (
            role["asset_kind"] not in {"rigid", "articulated"}
            or scene_objects.get(name, {}).get("asset_kind", role["asset_kind"])
            != role["asset_kind"]
        ):
            raise ValueError(
                "Role asset_kind must agree with the explicit scene context"
            )
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
            item["description"].casefold() == role["description"].casefold()
            for item in plan["roles"]
        ):
            raise ValueError(
                f"Use scene-context ID {name!r} for its corresponding role"
            )

    held: dict[str, set[str]] = {}
    grasp_sites: dict[str, dict[str, str]] = {}
    parked: set[str] = set()
    opened: set[str] = set()
    seated: set[tuple[str, str]] = set()
    tightened: set[tuple[str, str]] = set()
    invariants = []

    def check_capacity() -> None:
        explicit = [arm for owner in held.values() for arm in owner if arm != "auto"]
        if sum(map(len, held.values())) > len(_ARMS - {"auto"}):
            raise ValueError(
                "More simultaneous grasps than available arms, including auto. "
                "Release an earlier item before acquiring another if no arm is free."
            )
        if len(explicit) != len(set(explicit)):
            raise ValueError("One explicit arm is holding more than one role")

    def check_fact_state(fact: dict[str, Any]) -> None:
        kind = fact["kind"]
        if kind == "held" and any(
            subject not in held
            or ("auto" not in fact["arms"] and not set(fact["arms"]) <= held[subject])
            for subject in fact["subjects"]
        ):
            raise ValueError(
                "Held fact contradicts ordered acquisition/release or arm identity"
            )
        if kind == "released" and held.keys() & set(fact["subjects"]):
            raise ValueError("Released fact contradicts ordered holdings")
        if kind == "grasped_at":
            subject, arm = fact["subjects"][0], fact["arms"][0]
            owner = held.get(subject, set())
            sites = grasp_sites.get(subject, {})
            if (
                not owner
                or (arm != "auto" and arm not in owner)
                or sites.get(arm, sites.get("auto")) != fact["part"]
            ):
                raise ValueError(
                    "Grasp site is not witnessed by the acquisition of that arm"
                )
        if kind in {"arms_at_initial_pose", "grippers_open"} and not set(
            fact["arms"]
        ) <= (parked if kind == "arms_at_initial_pose" else opened):
            raise ValueError(f"{kind} requires its explicit arm step")
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

    for condition in plan["conditions"]:
        _shape(condition, {"scope", "fact", "quote"}, set(), "condition")
        condition["quote"] = _quote(condition["quote"], instruction, spans)
        scope, fact = condition["scope"], condition["fact"]
        if scope not in {"initial", "always", "final"}:
            raise ValueError("Condition scope must be initial, always or final")
        check_fact(fact)
        if scope == "always":
            invariants.append(fact)
        if scope == "initial":
            if fact["kind"] == "distance_change":
                raise ValueError(
                    "distance_change compares against initial distance; use distance_compare for a comparison in the initial state"
                )
            seated.update(_relation_pairs(fact, "seated_on"))
            tightened.update(_relation_pairs(fact, "tightened_on"))
            if fact["kind"] == "held":
                owner = set(fact["arms"])
                for subject in fact["subjects"]:
                    if subject in held and held[subject] != owner:
                        raise ValueError(
                            "Initial held conditions disagree on arm ownership"
                        )
                    held[subject] = owner
    seated.update(tightened)
    check_capacity()
    for fact in invariants:
        try:
            check_fact_state(fact)
        except ValueError as error:
            raise ValueError(
                f"An always condition must hold before the first step: {error}"
            ) from error

    ordered_steps = []
    for step in plan["steps"]:
        _shape(
            step,
            {"verb", "subjects", "arms", "effects", "quote"},
            {"direction", "maintain", "release_quote"},
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
        if (
            verb
            in {
                "pick",
                "grasp",
                "move_held",
                "place",
                "release",
                "handover",
                "transfer_contents",
                "press",
                "articulate",
            }
            and len(subjects) != 1
        ):
            raise ValueError(f"{verb} needs one object identity per occurrence")
        if type(step["effects"]) is not list:
            raise ValueError("step.effects must be a list of facts")
        maintained = step.get("maintain", [])
        if type(maintained) is not list:
            raise ValueError("step.maintain must be a list of typed facts")
        for fact in maintained:
            check_fact(fact)
            check_fact_state(fact)
        for fact in step["effects"]:
            check_fact(fact)
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
                and all(fact == released for fact in step["effects"])
            ):
                # The preceding placement already releases the same object.
                if released not in previous["effects"]:
                    previous["effects"].append(released)
                previous["release_quote"] = step["quote"]
                continue
        if verb == "place" and "release_quote" in step:
            released = dict(kind="released", subjects=subjects)
            if released not in step["effects"]:
                step["effects"].append(released)
        sites = [fact for fact in step["effects"] if fact["kind"] == "grasped_at"]
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
            *(_relation_pairs(fact, "tightened_on") for fact in step["effects"])
        )
        if new_tightened and (verb != "turn" or not new_tightened <= seated):
            raise ValueError(
                "Tightening requires a turn after seating the same part on the same reference"
            )
        tightened.update(new_tightened)
        seated.update(*(_relation_pairs(fact, "seated_on") for fact in step["effects"]))

        active_arms = arms if "auto" not in arms else _ARMS - {"auto"}
        if verb not in {"return_arms", "open_grippers"}:
            parked -= active_arms
        if verb not in _ARM_VERBS:
            opened -= active_arms
        if verb == "return_arms":
            parked |= arms
        elif verb == "open_grippers":
            if any(arms & owner or "auto" in owner for owner in held.values()):
                raise ValueError(
                    "Opening grippers with a held object requires a preceding release or placement"
                )
            opened |= arms
        elif verb in _ACQUIRE_VERBS:
            subject = subjects[0]
            if subject in held or any(
                (arms - {"auto"}) & owner for owner in held.values()
            ):
                raise ValueError(
                    "Pick would reacquire a held subject or occupy an already holding arm"
                )
            held[subject] = arms
            grasp_sites[subject] = {fact["arms"][0]: fact["part"] for fact in sites}
        elif verb in {
            "move_held",
            "place",
            "release",
            "handover",
            "transfer_contents",
            "articulate",
        }:
            subject = subjects[0]
            owner = held.get(subject)
            required = {step["arms"][0]} if verb == "handover" else arms
            if (
                owner is None
                or len(required) != len(owner)
                or ({"auto"} not in (required, owner) and required != owner)
            ):
                raise ValueError(
                    f"{verb} requires an earlier acquisition of {subject!r} by the same arm(s)"
                )
            if verb in {"place", "release"}:
                del held[subject]
                grasp_sites.pop(subject, None)
            elif verb == "handover":
                if len(arms) != 2 or any(key != subject for key in held):
                    raise ValueError(
                        "Handover needs two distinct arms and an unoccupied destination"
                    )
                held[subject] = {step["arms"][1]}
                grasp_sites.pop(subject, None)
            elif owner == {"auto"} and arms != {"auto"}:
                if any(arms & other for key, other in held.items() if key != subject):
                    raise ValueError("A specified holding arm is already occupied")
                held[subject] = arms
            if verb == "place" and not any(
                subject in fact.get("subjects", [])
                and fact["kind"]
                in {"relation", "pose_match", "hung_by", "view_location"}
                for fact in step["effects"]
            ):
                raise ValueError(
                    "Place requires a typed destination effect for its subject"
                )
            if verb == "transfer_contents" and not any(
                fact["kind"] == "transferred" and fact["references"][0] == subject
                for fact in step["effects"]
            ):
                raise ValueError(
                    "Transfer requires a typed effect from the held source"
                )
            if verb == "articulate" and not any(
                fact["kind"] == "articulated" and fact["subjects"] == subjects
                for fact in step["effects"]
            ):
                raise ValueError(
                    "Articulate requires the moving part's joint and mechanism"
                )
        else:
            if verb == "push" and held.keys() & set(subjects):
                raise ValueError("A push cannot move a held object")
            if "auto" not in arms and any(
                (key in subjects and "auto" not in owner and owner != arms)
                or (key not in subjects and arms & owner)
                for key, owner in held.items()
            ):
                raise ValueError(f"{verb} contradicts the current arm holdings")
            if verb == "move_gripper" and not any(
                fact["kind"] == "gripper_relation" and set(fact["arms"]) == arms
                for fact in step["effects"]
            ):
                raise ValueError(
                    "Move_gripper requires an arm position relative to a declared target"
                )
            if verb == "press" and not any(
                fact["kind"] == "actuated"
                and fact["value"] == "pressed"
                and fact["subjects"] == subjects
                for fact in step["effects"]
            ):
                raise ValueError("Press requires a pressed effect at its declared site")
        check_capacity()
        for fact in (*step["effects"], *invariants):
            check_fact_state(fact)
        ordered_steps.append(step)

    plan["steps"] = ordered_steps
    manipulated = {subject for step in plan["steps"] for subject in step["subjects"]}
    for condition in plan["conditions"]:
        scope, fact = condition["scope"], condition["fact"]
        if scope == "final":
            check_fact_state(fact)
        if fact["kind"] in {"untouched", "untouched_except"}:
            protected = set(fact["subjects"])
            conflict = (
                manipulated & protected
                if fact["kind"] == "untouched"
                else manipulated - protected
            )
            if scope != "initial" and conflict:
                raise ValueError(
                    f"Ordered operations touch protected roles: {sorted(conflict)}"
                )
    return plan


def compile_intent(
    plan: dict[str, Any], instruction: str, task_id: str, scene_objects: dict[str, Any]
) -> tuple[TaskSpec, dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """Project validated intent into a normative TaskSpec and graph-only nodes."""
    role_rows = []
    descriptions = {}
    for role in plan["roles"]:
        name = role["id"]
        context = scene_objects.get(name, {})
        descriptions[name] = context.get("description", role["description"])
        properties = {
            key: role[key] for key in ("asset_kind", "part_of") if key in role
        }
        if "asset_kind" in context:
            properties["asset_kind"] = context["asset_kind"]
        role_rows.append(
            dict(
                id=name,
                kind=role["kind"],
                properties=properties,
            )
        )
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
    for index, step in enumerate(plan["steps"], 1):
        identifier = f"step_{index:02d}"
        verb = step["verb"]
        arguments = {
            k: step[k]
            for k in ("subjects", "arms", "effects", "direction", "maintain")
            if k in step
        }
        if verb == "return_arms":
            arguments["pose_reference"] = "each_arm_initial_pose"
            arguments["binding_status"] = "pending"
        arguments["quote"] = step["quote"].strip()
        nodes.append(
            dict(
                id=identifier,
                call=dict(
                    kind="registered",
                    call_id=f"{_AUTHORING_INTENT_PREFIX}{verb}",
                    arguments=arguments,
                    resources={},
                ),
                depends_on=prior,
                task_instance_id=identifier,
                task_type=verb,
                role=verb,
            )
        )
        groups.append(
            dict(
                id=identifier,
                task_type=verb,
                node_ids=[identifier],
                depends_on=prior,
                success={},
            )
        )
        definitions["milestones"].append(
            dict(
                id=identifier,
                after=prior,
                achieve=[
                    predicate(
                        f"achieve_{identifier}",
                        dict(
                            verb=verb,
                            **{k: v for k, v in arguments.items() if k != "quote"},
                        ),
                    )
                ],
            )
        )
        clause(identifier, identifier, step["quote"], pending=verb == "return_arms")
        if "release_quote" in step:
            clause(f"{identifier}.release", identifier, step["release_quote"])
        prior = [identifier]
    if not definitions["goal"]:
        completion = "final_step_completed"
        definitions["goal"].append(predicate(completion, {"terminal_step": prior[0]}))
        clause(completion, completion, plan["steps"][-1]["quote"])
    spec = TaskSpec.from_dict(
        dict(task_id=task_id, instruction=instruction, roles=role_rows, **definitions)
    )
    validate_task_spec(spec)
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
            success=dict(
                authoring_only=True, conditions=graph_conditions, terminal_step=prior[0]
            ),
        )
    )
    return spec, graph, clauses, descriptions
