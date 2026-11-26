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

import json
import re
from copy import deepcopy
from pathlib import Path

from PIL import Image
import pytest

from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg, compat
from embodichain.gen_sim.task_engine._review import main as review_main, write_review
from embodichain.gen_sim.task_engine.authoring_intent import (
    AuthoringEvidenceError,
    compile_intent,
    validate_intent,
    verify_spatial_claim,
    verify_state_claim,
)
from embodichain.gen_sim.task_engine.hybrid import (
    _validate_intent_audit,
    publish_hybrid,
)
from embodichain.gen_sim.task_engine.semantic_graph import validate_semantic_task_graph
from embodichain.gen_sim.workflow import load_task_plan

_SOURCE = "Execute the described task."


def _role(name, kind="object", **fields):
    return dict(id=name, kind=kind, description=name.replace("_", " "), **fields)


def _fact(kind, subjects, references="", **fields):
    fact = dict(kind=kind, subjects=subjects.split(), **fields)
    if references:
        fact["references"] = references.split()
    return fact


def _relation(value, subjects, references, **fields):
    return _fact("relation", subjects, references, value=value, **fields)


def _step(verb, subjects, *effects, arms="auto", quote=_SOURCE, **fields):
    return dict(
        verb=verb,
        subjects=subjects.split(),
        arms=arms.split(),
        effects=list(effects),
        quote=quote,
        **fields,
    )


def _plan(roles, *steps, conditions=()):
    return dict(
        roles=(
            [_role(name) for name in roles.split()] if isinstance(roles, str) else roles
        ),
        steps=list(steps),
        conditions=list(conditions),
    )


def _condition(scope, fact):
    return dict(scope=scope, fact=fact, quote=_SOURCE)


def _check(*refs, **fields):
    return {
        **dict(quote=_SOURCE, requirement="requested behavior", evidence=list(refs)),
        **fields,
    }


def _audit(*checks):
    return dict(issues=[], checks=list(checks))


def _validate(plan, context=None, text=_SOURCE):
    return validate_intent(plan, text, scene_objects=context or {})


@pytest.fixture(scope="module")
def robotwin_1287():
    path = Path(__file__).with_name("data") / "robotwin_1287_replies.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def test_recorded_held_tool_press_preserves_the_explicit_contact(robotwin_1287):
    case = robotwin_1287["stamp_seal"]
    original = case["proposal"]["semantic_plan"]
    source = case["instruction"]
    parsed = _validate(original, text=source)
    contact = parsed["steps"][2]
    assert contact["verb"] == "move_held" and contact["subjects"] == ["stamp"]
    assert contact["arms"] == ["right"] and contact["direction"] == "down"
    assert contact["effects"] == [_relation("touching", "stamp_base", "purple_mat")]
    assert parsed["steps"][3]["verb"] == "release"
    spec, graph, _, _ = compile_intent(parsed, source, "stamp_seal", {})
    assert not any(entry["is_articulated"] for entry in spec.requirements[0].value)
    assert (
        _fact("orientation", "stamp", value="upright")
        in graph["success"]["conditions"]["final"]
    )

    incomplete = deepcopy(original)
    incomplete["steps"][2]["effects"].pop(0)
    with pytest.raises(ValueError, match="stamp.*requires articulation"):
        _validate(incomplete, text=source)


def test_recorded_switch_press_does_not_prove_the_other_position(robotwin_1287):
    case = robotwin_1287["turn_switch"]
    context = {obj["id"]: obj for obj in case["scene_context"]["objects"]}
    source = case["instruction"]
    plan = deepcopy(case["proposal"]["semantic_plan"])
    audit = deepcopy(case["audit"])
    transition = _fact("actuated", "switch", value="other_from_initial")
    state_check = next(
        row for row in audit["checks"] if row.get("state", {}).get("kind") == "actuated"
    )
    state_check.update(
        scope="final",
        requirement="Leave the switch in its other position.",
        state=transition,
    )
    pressed = _validate(plan, context, source)
    with pytest.raises(AuthoringEvidenceError, match="actuated.*matching evidence"):
        _validate_intent_audit(source, pressed, audit)
    pressed["conditions"].append(
        dict(scope="final", fact=transition, quote=state_check["quote"])
    )
    with pytest.raises(AuthoringEvidenceError, match="surviving action effect"):
        _validate_intent_audit(source, pressed, audit)

    plan["steps"][1]["effects"] = [transition]
    parsed = _validate(plan, context, source)
    producers, new_goals = _validate_intent_audit(source, parsed, audit)
    assert new_goals == [
        dict(scope="final", fact=transition, quote=state_check["quote"])
    ]
    assert list(producers.values()) == [["step_02"]]
    with pytest.raises(ValueError, match="requires articulation"):
        _validate(plan, text=source)


def test_recorded_two_object_pick_needs_independent_parallel_branches(robotwin_1287):
    case = robotwin_1287["place_bread_basket"]
    source = case["instruction"]
    original = case["proposal"]["semantic_plan"]
    with pytest.raises(ValueError, match="one object identity per occurrence"):
        _validate(original, text=source)

    plan = deepcopy(original)
    pick, place = plan["steps"]
    plan["steps"] = [
        dict(
            pick,
            subjects=[subject],
            arms=[arm],
            effects=[grip],
            parallel_group="pickup",
        )
        for subject, arm, grip in zip(pick["subjects"], pick["arms"], pick["effects"])
    ] + [
        dict(
            place,
            subjects=[subject],
            arms=[arm],
            effects=[dict(fact, subjects=[subject]) for fact in place["effects"]],
        )
        for subject, arm in zip(place["subjects"], place["arms"])
    ]
    audit = _audit(
        _check(
            "step_01",
            "step_02",
            scope="action",
            quote="the left arm picks up the left piece of bread and the right arm picks up the right piece of bread at the same time",
            parallel_picks={"left": "bread_left", "right": "bread_right"},
        )
    )
    parsed = _validate(plan, text=source)
    _validate_intent_audit(source, parsed, audit)
    _, graph, _, _ = compile_intent(parsed, source, "place_bread_basket", {})
    assert [node["depends_on"] for node in graph["nodes"]] == [
        [],
        [],
        ["step_01", "step_02"],
        ["step_03"],
    ]
    assert graph["success"]["authoring_only"] is True
    hold_source = source.rsplit(" Then place", 1)[0]
    completed_together = _validate(
        dict(plan, steps=plan["steps"][:2]), text=hold_source
    )
    _, terminal_graph, _, _ = compile_intent(
        completed_together, hold_source, "hold_both", {}
    )
    assert terminal_graph["success"]["terminal_steps"] == ["step_01", "step_02"]

    swapped = deepcopy(audit)
    swapped["checks"][0]["parallel_picks"] = {
        "left": "bread_right",
        "right": "bread_left",
    }
    with pytest.raises(AuthoringEvidenceError, match="parallel"):
        _validate_intent_audit(source, parsed, swapped)

    for step in plan["steps"][:2]:
        del step["parallel_group"]
    sequential = _validate(plan, text=source)
    with pytest.raises(AuthoringEvidenceError, match="parallel"):
        _validate_intent_audit(source, sequential, audit)


@pytest.mark.parametrize("ending", [".", ", but not at the same time."])
def test_bilateral_assignments_do_not_authorize_parallel_picks(ending):
    source = (
        "Use both arms: the left arm picks up the cup and the right arm picks up the card"
        + ending
    )
    plan = _validate(
        _plan(
            "cup card",
            _step("pick", "cup", arms="left", quote=source, parallel_group="pair"),
            _step("pick", "card", arms="right", quote=source, parallel_group="pair"),
        ),
        text=source,
    )
    unsupported = _check(
        "step_01",
        "step_02",
        quote=source,
        scope="action",
        parallel_picks={"left": "cup", "right": "card"},
    )
    with pytest.raises(AuthoringEvidenceError, match="explicit.*timing"):
        _validate_intent_audit(source, plan, _audit(unsupported))
    for step in plan["steps"]:
        del step["parallel_group"]
    _validate_intent_audit(
        source, plan, _audit(_check("step_01", "step_02", quote=source, scope="action"))
    )


@pytest.mark.parametrize("earlier_arm", ["right", "auto"])
def test_departure_requires_a_distinct_held_motion_after_lifting(earlier_arm):
    source = "Grasp the container, lift it and move it a short distance away from where it was."
    moved = _fact(
        "distance_change", "container", "origin", comparison="gt", epoch="initial"
    )
    grasp = _step("grasp", "container", arms=earlier_arm, quote=source)
    lift = _step(
        "move_held", "container", moved, arms=earlier_arm, quote=source, direction="up"
    )
    plan = _plan("container origin", grasp, lift)
    with pytest.raises(ValueError, match="separate.*move_held"):
        _validate(plan, text=source)
    depart = dict(lift, arms=["right"], direction="away", extent="short")
    plan["steps"].append(depart)
    checked = _validate(plan, text=source)
    _, graph, _, _ = compile_intent(checked, source, "held_motion", {})
    assert graph["nodes"][1]["call"]["arguments"]["arms"] == [earlier_arm]
    assert graph["nodes"][2]["depends_on"] == ["step_02"]
    assert graph["nodes"][2]["call"]["arguments"]["extent"] == "short"
    with pytest.raises(ValueError, match="separate.*move_held"):
        _validate(dict(plan, steps=[grasp, depart, lift]), text=source)


def test_departure_cannot_reuse_a_lift_from_before_release():
    source = "Lift the container and move it away."
    pick = _step("pick", "container", quote=source)
    lift = _step("move_held", "container", quote=source, direction="up")
    release = _step("release", "container", quote=source)
    depart = _step("move_held", "container", quote=source, direction="away")
    with pytest.raises(ValueError, match="separate.*move_held"):
        _validate(_plan("container", pick, lift, release, pick, depart), text=source)


@pytest.mark.parametrize(
    "placement,quote_only_timing",
    [
        ("and place both in the tray at the same time", False),
        ("and with both arms place them in the tray at the same time", True),
        ("and they place both in the tray at the same time", False),
        ("and the robot together places both in the tray at the same time", False),
    ],
)
def test_placement_timing_does_not_authorize_parallel_picks(
    placement, quote_only_timing
):
    source = (
        "Pick up the cup with the left arm first and pick up the card with the "
        f"right arm second, {placement}."
    )
    picks = [
        _step("pick", obj, arms=arm, quote=source, parallel_group="pickup")
        for arm, obj in [("left", "cup"), ("right", "card")]
    ]
    places = [
        _step("place", obj, _relation("inside", obj, "tray"), arms=arm, quote=source)
        for arm, obj in [("left", "cup"), ("right", "card")]
    ]
    plan = _plan("cup card tray", *picks, *places)
    supported = _check("step_01", "step_02", scope="action", quote=source)
    unsupported = dict(
        supported,
        quote="at the same time" if quote_only_timing else source,
        parallel_picks={"left": "cup", "right": "card"},
    )
    with pytest.raises(AuthoringEvidenceError, match="explicit.*timing"):
        _validate_intent_audit(
            source, _validate(plan, text=source), _audit(unsupported)
        )

    for step in picks:
        del step["parallel_group"]
    parsed = _validate(plan, text=source)
    _validate_intent_audit(source, parsed, _audit(supported))
    _, graph, _, _ = compile_intent(parsed, source, "ordered_acquisition", {})
    assert graph["nodes"][1]["depends_on"] == ["step_01"]


def test_negated_lift_and_depart_does_not_require_motion():
    grasp = "Grasp the container."
    source = grasp + " Do not lift it and move it away."
    plan = _plan("container", _step("grasp", "container", arms="right", quote=grasp))
    assert _validate(plan, text=source)["steps"] == plan["steps"]


def test_future_side_relation_needs_evidence_after_reference_is_placed():
    relation_quote = "Put the two supports one on each side of the future crosspiece."
    source = relation_quote + " Put the crosspiece down."
    sides = _relation("opposite_sides", "left_base right_base", "crosspiece")
    plan = _validate(
        _plan(
            "left_base right_base crosspiece table",
            _step("pick", "crosspiece", quote="Put the crosspiece down."),
            _step(
                "place",
                "crosspiece",
                _relation("on", "crosspiece", "table"),
                quote="Put the crosspiece down.",
            ),
            conditions=[dict(scope="initial", fact=sides, quote=relation_quote)],
        ),
        text=source,
    )
    placement = _check("step_02", scope="action", quote="Put the crosspiece down.")
    initial = _check(
        "condition_01", scope="initial", quote=relation_quote, spatial=sides
    )
    with pytest.raises(AuthoringEvidenceError, match="after.*placed"):
        _validate_intent_audit(source, plan, _audit(placement, initial))
    plan["conditions"] = [dict(scope="final", fact=sides, quote=relation_quote)]
    final = _check("condition_01", scope="final", quote=relation_quote, spatial=sides)
    with pytest.raises(AuthoringEvidenceError, match="after.*placed"):
        _validate_intent_audit(source, plan, _audit(placement, final))
    plan["steps"][1]["effects"].append(sides)
    final["evidence"].append("step_02")
    producers, _ = _validate_intent_audit(source, plan, _audit(placement, final))
    assert producers == {"condition_01": ["step_02"]}


def test_future_side_effect_survives_audit_at_its_checkpoint_but_not_as_a_final_goal():
    relation_quote = "Put the two supports one on each side of the future crosspiece."
    install = "Install the crosspiece on the table."
    move = "Then move the crosspiece into the tray."
    source = " ".join((relation_quote, install, move))
    sides = _relation("opposite_sides", "left_base right_base", "crosspiece")
    plan = _validate(
        _plan(
            "left_base right_base crosspiece table tray",
            _step("pick", "crosspiece", quote=install),
            _step(
                "place",
                "crosspiece",
                _relation("on", "crosspiece", "table"),
                sides,
                quote=install,
            ),
            _step("pick", "crosspiece", quote=move),
            _step(
                "place",
                "crosspiece",
                _relation("inside", "crosspiece", "tray"),
                quote=move,
            ),
        ),
        text=source,
    )
    effect = _check("step_02", scope="effect", quote=relation_quote, spatial=sides)
    assert _validate_intent_audit(source, plan, _audit(effect)) == ({}, [])
    with pytest.raises(AuthoringEvidenceError, match="final|evidence"):
        _validate_intent_audit(source, plan, _audit(dict(effect, scope="final")))


def test_auto_grasps_reserve_capacity_until_release():
    plan = _plan("a b c", *(_step("pick", role) for role in "abc"))
    with pytest.raises(ValueError, match="More simultaneous grasps"):
        _validate(plan)
    press = _step("press", "c", _fact("actuated", "c", value="pressed"), arms="left")
    with pytest.raises(ValueError, match="contradicts the current arm holdings"):
        _validate(
            _plan("a b c", *plan["steps"][:2], press),
            {"c": {"asset_kind": "articulated"}},
        )
    plan["steps"].insert(2, _step("release", "a"))
    assert len(_validate(plan)["steps"]) == 4


@pytest.mark.parametrize("release_other", [False, True])
def test_auto_owner_cannot_change_without_a_handover(release_other):
    plan = _plan("a b", _step("pick", "a"), _step("pick", "b", arms="left"))
    if release_other:
        plan["steps"].append(_step("release", "b", arms="left"))
    plan["steps"].append(_step("release", "a", arms="left"))
    with pytest.raises(ValueError, match="same arm"):
        _validate(plan)
    plan["steps"][-1]["arms"] = ["right"]
    assert _validate(plan)["steps"][0]["arms"] == ["auto"]


def test_auto_owner_is_constrained_by_free_arm_actions_and_terminal_hold():
    def reach(arm):
        return _step(
            "move_gripper",
            "",
            dict(
                kind="gripper_relation", arms=[arm], references=["site"], value="above"
            ),
            arms=arm,
        )

    plan = _plan("a site", _step("pick", "a"), reach("left"), reach("right"))
    with pytest.raises(ValueError, match="arm holdings"):
        _validate(plan)
    plan["steps"].pop()
    plan["conditions"] = [_condition("final", _fact("held", "a", arms=["right"]))]
    assert _validate(plan)["conditions"] == plan["conditions"]
    plan["steps"].append(_step("handover", "a", arms="right left"))
    plan["conditions"][0]["fact"]["arms"] = ["left"]
    assert _validate(plan)["conditions"] == plan["conditions"]


def test_release_merges_only_for_the_same_arm_and_preserves_quotes():
    text = "Pick the parcel and set it on the shelf. Release it."
    support = _relation("on_top_faces", "parcel", "shelf")
    pick = _step("pick", "parcel", arms="left", quote="pick the parcel", direction="up")
    place = _step(
        "place",
        "parcel",
        {"on_top_faces": {k: support[k] for k in ("subjects", "references")}},
        arms="left",
        quote="set it on the shelf",
    )
    release = _step("release", "parcel", arms="left", quote="source_span_02")
    plan = _plan("parcel shelf", pick, place, release)
    normalized = _validate(plan, text=text)
    assert len(normalized["steps"]) == 2 and len(plan["steps"]) == 3
    assert normalized["steps"][0]["quote"] == "Pick the parcel"
    assert normalized["steps"][1]["effects"] == [support, _fact("released", "parcel")]
    spec, graph, clauses, _ = compile_intent(normalized, text, "parcel", {})
    assert graph["success"]["terminal_step"] == "step_02"
    assert spec.to_dict()["goal"][0]["arguments"] == {"terminal_step": "step_02"}
    assert any(
        c["target"] == "step_02" and c["quote"] == "Release it." for c in clauses
    )
    assert _validate(normalized, text=text) == normalized
    release["arms"] = ["right"]
    with pytest.raises(ValueError, match="requires an earlier acquisition"):
        _validate(plan, text=text)
    pick["quote"] = "Acquire it"
    with pytest.raises(ValueError, match="Not a verbatim source span"):
        _validate(plan, text=text)


@pytest.mark.parametrize(
    "value,wrong_value",
    [
        ("left_of", "right_of"),
        ("inside", "not_inside"),
        ("centered_on", "near_center_of"),
    ],
)
def test_spatial_claim_refuses_a_different_operand_or_weaker_relation(
    value, wrong_value
):
    roles = dict(item="object", base="surface", slot="target")
    expected = _relation(value, "item", "base")
    wrong = _relation(wrong_value, "item", "base")
    with pytest.raises(AuthoringEvidenceError, match="matching typed relations"):
        verify_spatial_claim(expected, [wrong], roles)
    verify_spatial_claim(expected, [expected], roles)


def test_spatial_claim_through_slot_needs_typed_exact_center():
    roles = dict(item="object", base="surface", slot="target")
    expected = _relation("centered_on", "item", "base")
    at = _relation("at", "item", "slot")
    with pytest.raises(AuthoringEvidenceError, match="matching typed relations"):
        verify_spatial_claim(
            expected,
            [at, _fact("view_location", "slot", frame="table", depth="middle")],
            roles,
        )
    verify_spatial_claim(
        expected, [at, _relation("centered_on", "slot", "base")], roles
    )


@pytest.mark.parametrize("through_slot", [False, True])
def test_spatial_claim_preserves_an_explicit_reference_frame(through_slot):
    roles = dict(item="object", base="surface", slot="target")
    evidence_subject = "slot" if through_slot else "item"
    prefix = [_relation("at", "item", "slot")] if through_slot else []
    claim = _relation("left_of", "item", "base", frame="robot")
    relation = _relation("left_of", evidence_subject, "base")
    verify_spatial_claim(claim, [*prefix, {**relation, "frame": "robot"}], roles)
    for wrong in (relation, {**relation, "frame": "world"}):
        with pytest.raises(AuthoringEvidenceError, match="matching typed relations"):
            verify_spatial_claim(claim, [*prefix, wrong], roles)
    if through_slot:
        with pytest.raises(AuthoringEvidenceError, match="matching typed relations"):
            verify_spatial_claim(
                claim,
                [{**prefix[0], "frame": "world"}, {**relation, "frame": "robot"}],
                roles,
            )
    verify_spatial_claim(
        _relation("left_of", "item", "base"), [*prefix, relation], roles
    )


@pytest.mark.parametrize(
    "value,inverse",
    [("left_of", "right_of"), ("in_front_of", "behind"), ("above", "below")],
)
def test_axis_relations_match_inverse_evidence_and_reject_conflicts(value, inverse):
    effect = _relation(value, "cup", "bowl", frame="table")
    inverse_claim = _relation(inverse, "bowl", "cup", frame="table")
    plan = _plan(
        "cup bowl",
        _step("pick", "cup"),
        _step("place", "cup", effect),
    )
    _validate_intent_audit(
        _SOURCE,
        _validate(plan),
        _audit(_check("step_02", scope="final", spatial=inverse_claim)),
    )
    at = _relation("at", "cup", "slot")
    slotted = _validate(
        _plan(
            [_role("cup"), _role("bowl"), _role("slot", kind="target")],
            _step("pick", "cup"),
            _step("place", "cup", at),
            conditions=[
                _condition("initial", _relation(inverse, "bowl", "slot", frame="table"))
            ],
        )
    )
    _validate_intent_audit(
        _SOURCE,
        slotted,
        _audit(_check("condition_01", "step_02", scope="final", spatial=effect)),
    )
    roles = dict(cup="object", bowl="object", slot="target")
    for frame in ("world", "reference"):
        with pytest.raises(AuthoringEvidenceError):
            verify_spatial_claim({**inverse_claim, "frame": frame}, [effect], roles)
    with pytest.raises(AuthoringEvidenceError):
        verify_spatial_claim(
            {**inverse_claim, "frame": "reference"},
            [{**effect, "frame": "reference"}],
            roles,
        )
    conflict = _relation(value, "bowl", "cup", frame="table")
    plan["conditions"] = [_condition("final", conflict)]
    with pytest.raises(AuthoringEvidenceError, match="contradicts"):
        _validate_intent_audit(
            _SOURCE,
            _validate(plan),
            _audit(_check("condition_01", scope="final", spatial=conflict)),
        )


def test_joint_relation_cannot_be_assembled_from_separate_groups():
    roles = dict.fromkeys("abcd", "object")
    claim = _relation("between", "a", "b c")
    with pytest.raises(AuthoringEvidenceError):
        verify_spatial_claim(
            claim, [_relation("between", "a", pair) for pair in ("b d", "c d")], roles
        )


def test_initial_audit_can_use_one_member_of_a_grouped_state():
    grouped = _fact("orientation", "a b", value="upright", frame="head_camera")
    plan = _validate(
        _plan(
            "a b c",
            _step("push", "c"),
            conditions=[_condition("initial", grouped)],
        )
    )
    member = dict(grouped, subjects=["b"])

    def audit(claim):
        return _audit(_check("condition_01", scope="initial", state=claim))

    _validate_intent_audit(_SOURCE, plan, audit(member))
    for different in (
        dict(member, subjects=["c"]),
        dict(member, value="flat"),
        dict(member, frame="table"),
    ):
        with pytest.raises(AuthoringEvidenceError, match="matching evidence"):
            _validate_intent_audit(_SOURCE, plan, audit(different))
    with pytest.raises(AuthoringEvidenceError, match="matching evidence"):
        verify_state_claim(
            grouped,
            [dict(grouped, subjects=["a"]), member],
            dict.fromkeys("abc", "object"),
        )


@pytest.mark.parametrize(
    "grouped",
    [
        _fact("ordered_row", "a b c", frame="table"),
        _fact("stable", "a b c"),
        _fact("mutually_separate", "a b c"),
        _fact("untouched_except", "a b c"),
        _fact("transferred", "a b c", "x y", material="solid", all=True, quantity=3),
    ],
    ids=lambda fact: fact["kind"],
)
def test_joint_state_cannot_be_proven_by_changing_its_members(grouped):
    roles = dict.fromkeys("abcxy", "object")
    verify_state_claim(grouped, [grouped], roles)
    with pytest.raises(AuthoringEvidenceError, match="matching evidence"):
        verify_state_claim(dict(grouped, subjects=["a", "b"]), [grouped], roles)
    if grouped["kind"] == "ordered_row":
        with pytest.raises(AuthoringEvidenceError, match="matching evidence"):
            verify_state_claim(
                dict(grouped, subjects=["c", "b", "a"]), [grouped], roles
            )


@pytest.mark.parametrize(
    "kind,value,refs,wrong",
    [
        ("orientation", "aligned_long_axis", "b", "c"),
        ("orientation", "pointing_toward", "b", "c"),
        ("pose_match", None, "a", "b"),
        ("distance_compare", None, "b c", "c b"),
    ],
)
def test_state_claim_keeps_reference_identity_and_order(kind, value, refs, wrong):
    roles = dict.fromkeys("abc", "object")
    fields = (
        {"value": value}
        if value
        else (
            dict(epoch="initial", components=["position", "orientation"])
            if kind == "pose_match"
            else dict(comparison="gt")
        )
    )
    claim = _fact(kind, "a", refs, **fields)
    with pytest.raises(AuthoringEvidenceError, match="matching evidence"):
        verify_state_claim(claim, [_fact(kind, "a", wrong, **fields)], roles)
    verify_state_claim(claim, [claim], roles)


def test_audit_keeps_scope_and_does_not_accept_identity_as_action_evidence():
    relation = _relation("on", "item", "surface")
    plan = _plan(
        "item surface",
        _step("push", "item", relation),
        conditions=[_condition("always", _fact("untouched", "surface"))],
    )
    spatial = _check("step_01", "condition_01", scope="always", spatial=relation)
    with pytest.raises(AuthoringEvidenceError, match="matching typed relations"):
        _validate_intent_audit(_SOURCE, plan, _audit(spatial))
    plan["conditions"][0]["fact"] = relation
    _validate_intent_audit(_SOURCE, plan, _audit(spatial))
    with pytest.raises(AuthoringEvidenceError, match="initial condition"):
        _validate_intent_audit(_SOURCE, plan, _audit({**spatial, "scope": "initial"}))
    with pytest.raises(ValueError, match="not roles alone"):
        _validate_intent_audit(_SOURCE, plan, _audit(_check("role_item")))
    _validate_intent_audit(_SOURCE, plan, _audit(_check("item"), _check("step_01")))
    _validate_intent_audit(_SOURCE, plan, _audit(_check("step_01", scope="action")))
    with pytest.raises(ValueError, match="scope"):
        _validate_intent_audit(
            _SOURCE, plan, _audit(_check("condition_01", scope="action"))
        )


def test_typed_action_uses_cited_step_effects_without_becoming_final():
    left = _relation("left_of", "item", "base", frame="robot")
    right = _relation("right_of", "item", "base", frame="robot")
    plan = _validate(
        _plan(
            "item base",
            _step("push", "item", left),
            _step("push", "item", right, maintain=[left]),
            conditions=[_condition("initial", left)],
        )
    )
    action = _check("step_01", scope="action", spatial=left)
    _validate_intent_audit(_SOURCE, plan, _audit(action))

    with pytest.raises(AuthoringEvidenceError, match="step producing it"):
        _validate_intent_audit(
            _SOURCE, plan, _audit({**action, "evidence": ["condition_01"]})
        )
    for wrong in (
        {**action, "evidence": ["step_02", "condition_01"]},
        {**action, "spatial": {**left, "frame": "world"}},
        {**action, "scope": "final"},
    ):
        with pytest.raises(AuthoringEvidenceError):
            _validate_intent_audit(_SOURCE, plan, _audit(wrong))


def test_plain_initial_role_check_is_identity_only():
    plan = _validate(_plan("item", _step("pick", "item")))
    identity = _check("item", "role_item", scope="initial")
    action = _check("step_01", scope="action")
    _validate_intent_audit(_SOURCE, plan, _audit(identity, action))

    with pytest.raises(ValueError, match="not roles alone"):
        _validate_intent_audit(_SOURCE, plan, _audit(identity))
    for wrong in (
        {**identity, "state": _fact("held", "item", arms=["auto"])},
        {**identity, "evidence": ["item", "step_01"]},
    ):
        with pytest.raises(AuthoringEvidenceError, match="initial condition"):
            _validate_intent_audit(_SOURCE, plan, _audit(wrong, action))


def test_before_audit_checks_the_position_at_start_of_regrasp():
    center = _relation("centered_on", "item", "table", frame="table")
    plan = _validate(
        _plan(
            "item table",
            _step("pick", "item", arms="left"),
            _step("place", "item", center, arms="left"),
            _step("pick", "item", arms="right"),
            conditions=[_condition("initial", center)],
        )
    )
    check = _check("step_02", "step_03", scope="before", spatial=center)
    _validate_intent_audit(_SOURCE, plan, _audit(check))
    _validate_intent_audit(
        _SOURCE,
        plan,
        _audit(_check("condition_01", "step_01", scope="before", spatial=center)),
    )

    for wrong in (
        {**check, "evidence": ["step_02"]},
        {**check, "evidence": ["condition_01", "step_03"]},
        {**check, "scope": "effect"},
        {**check, "scope": "final"},
        {**check, "spatial": {**center, "frame": "head_camera"}},
    ):
        with pytest.raises(AuthoringEvidenceError):
            _validate_intent_audit(_SOURCE, plan, _audit(wrong))


def test_released_audit_combines_only_live_per_subject_facts():
    plan = _validate(
        _plan(
            "a b table",
            _step("pick", "a"),
            _step("place", "a", _relation("on", "a", "table"), _fact("released", "a")),
            _step("pick", "b"),
            _step("place", "b", _relation("on", "b", "table"), _fact("released", "b")),
            _step("pick", "a"),
        )
    )
    check = _check("step_02", "step_04", scope="effect", state=_fact("released", "a b"))
    _validate_intent_audit(_SOURCE, plan, _audit(check))

    for wrong in (
        {**check, "evidence": ["step_04"]},
        {**check, "evidence": ["step_02", "step_04", "step_05"]},
        {**check, "scope": "final"},
    ):
        with pytest.raises(AuthoringEvidenceError, match="released"):
            _validate_intent_audit(_SOURCE, plan, _audit(wrong))
    plan["steps"][2]["maintain"] = [_fact("released", "a")]
    plan["steps"][4]["maintain"] = [_fact("released", "b")]
    with pytest.raises(AuthoringEvidenceError, match="released"):
        _validate_intent_audit(
            _SOURCE,
            _validate(plan),
            _audit({**check, "scope": "step", "evidence": ["step_03", "step_05"]}),
        )


def test_grouped_facts_keep_unaffected_subjects_and_references():
    plan = _validate(
        _plan(
            "a b table marker",
            _step("pick", "a", _fact("orientation", "a", value="left_to_right")),
            _step("return_arms", "", arms="left right"),
            conditions=[
                _condition("initial", _relation("on", "a b", "table")),
                _condition("initial", _fact("released", "a b")),
                _condition("initial", _relation("left_of", "b", "a marker")),
                _condition("initial", _relation("on_top_faces", "b", "a marker")),
                _condition("initial", _fact("orientation", "a b", value="upright")),
                _condition("initial", _relation("supported_by", "a b", "table marker")),
                _condition(
                    "initial",
                    _fact("view_location", "a b", frame="table", lateral="left"),
                ),
            ],
        )
    )
    _validate_intent_audit(
        _SOURCE,
        plan,
        _audit(
            _check(
                "step_02",
                "condition_01",
                scope="before",
                spatial=_relation("on", "b", "table"),
            ),
            _check(
                "step_02", "condition_02", scope="before", state=_fact("released", "b")
            ),
            _check(
                "step_02",
                "condition_03",
                scope="before",
                spatial=_relation("left_of", "b", "marker"),
            ),
            _check(
                "step_02",
                "condition_05",
                scope="before",
                state=_fact("orientation", "b", value="upright"),
            ),
            _check(
                "step_02",
                "condition_06",
                scope="before",
                spatial=_relation("supported_by", "b", "table marker"),
            ),
            _check(
                "step_02",
                "condition_07",
                scope="before",
                state=_fact("view_location", "b", frame="table", lateral="left"),
            ),
        ),
    )
    for index, field, fact in (
        (1, "spatial", _relation("on", "a", "table")),
        (2, "state", _fact("released", "a")),
        (3, "spatial", _relation("left_of", "b", "a")),
        (4, "spatial", _relation("on_top_faces", "b", "marker")),
        (5, "state", _fact("orientation", "a", value="upright")),
        (6, "spatial", _relation("supported_by", "a", "table marker")),
        (7, "state", _fact("view_location", "a", frame="table", lateral="left")),
    ):
        with pytest.raises(AuthoringEvidenceError):
            _validate_intent_audit(
                _SOURCE,
                plan,
                _audit(
                    _check(
                        "step_02",
                        f"condition_{index:02d}",
                        scope="before",
                        **{field: fact},
                    )
                ),
            )


@pytest.mark.parametrize("components", [["position"], ["position", "orientation"]])
@pytest.mark.parametrize("declare_final", [False, True])
def test_turn_invalidates_only_orientation_bearing_pose_evidence(
    components, declare_final
):
    pose = _fact(
        "pose_match", "bottle", "bottle", epoch="initial", components=components
    )
    final = dict(pose, components=list(reversed(components)))
    plan = _validate(
        _plan(
            "bottle",
            _step("pick", "bottle", arms="left"),
            _step("move_held", "bottle", pose, arms="left"),
            _step("turn", "bottle", arms="left"),
            conditions=[_condition("final", final)] if declare_final else [],
        )
    )
    refs = ("step_02", "condition_01") if declare_final else ("step_02",)
    check = _audit(_check(*refs, scope="final", state=final))
    if "orientation" in components:
        with pytest.raises(
            AuthoringEvidenceError, match="still valid at the end|outlives"
        ):
            _validate_intent_audit(_SOURCE, plan, check)
    else:
        _validate_intent_audit(_SOURCE, plan, check)


@pytest.mark.parametrize("moved", ["bowl", "other"])
@pytest.mark.parametrize("declare_final", [False, True])
def test_gripper_relation_loses_evidence_when_its_reference_moves(moved, declare_final):
    hover = dict(
        kind="gripper_relation", arms=["left"], references=["rim"], value="above"
    )
    plan = _validate(
        _plan(
            [_role("bowl"), _role("rim", part_of="bowl"), _role("other")],
            _step("move_gripper", "", hover, arms="left"),
            _step("pick", moved, arms="right"),
            _step("move_held", moved, arms="right"),
            conditions=[_condition("final", hover)] if declare_final else [],
        )
    )
    refs = ("step_01", "condition_01") if declare_final else ("step_01",)
    check = _audit(_check(*refs, scope="final", state=hover))
    if moved == "bowl":
        with pytest.raises(
            AuthoringEvidenceError, match="still valid at the end|outlives"
        ):
            _validate_intent_audit(_SOURCE, plan, check)
    else:
        _validate_intent_audit(_SOURCE, plan, check)


@pytest.mark.parametrize("verb", ["place", "move_held"])
@pytest.mark.parametrize(
    "previous,current,frame,supersedes",
    [
        ("left_to_right", "front_to_back", "table", True),
        ("upright", "left_to_right", "table", True),
        ("vertical", "flat", "table", True),
        ("pointing_up", "pointing_left", "table", True),
        ("upright_cross_section", "left_to_right", "table", False),
        ("left_to_right", "front_to_back", "head_camera", False),
    ],
)
def test_later_explicit_orientation_supersedes_the_same_axis(
    verb, previous, current, frame, supersedes
):
    old = _fact("orientation", "rod", value=previous, frame="table")
    updated = dict(old, value=current, frame=frame)
    plan = _validate(
        _plan(
            "rod table",
            _step("pick", "rod"),
            _step("orient", "rod", old),
            _step(verb, "rod", _relation("on", "rod", "table"), updated),
        )
    )
    _validate_intent_audit(
        _SOURCE, plan, _audit(_check("step_03", scope="final", state=updated))
    )
    previous_claim = _audit(_check("step_02", scope="final", state=old))
    if supersedes:
        with pytest.raises(
            AuthoringEvidenceError, match="effect still valid at the end"
        ):
            _validate_intent_audit(_SOURCE, plan, previous_claim)
    else:
        _validate_intent_audit(_SOURCE, plan, previous_claim)


@pytest.mark.parametrize(
    "verb,kind",
    [("return_arms", "arms_at_initial_pose"), ("open_grippers", "grippers_open")],
)
def test_grouped_arm_evidence_keeps_the_unaffected_arm(verb, kind):
    both = dict(kind=kind, arms=["left", "right"])
    plan = _plan(
        "item",
        _step(verb, "", both, arms="left right"),
        _step("pick", "item", arms="right"),
    )
    checked = _validate(plan)

    def claim(arm):
        return _audit(_check("step_01", scope="final", state=dict(both, arms=[arm])))

    _validate_intent_audit(
        _SOURCE,
        checked,
        _audit(_check("step_01", scope="effect", state=dict(both, arms=["left"]))),
    )
    _validate_intent_audit(_SOURCE, checked, claim("left"))
    with pytest.raises(AuthoringEvidenceError):
        _validate_intent_audit(_SOURCE, checked, claim("right"))
    plan["steps"][1]["arms"] = ["auto"]
    with pytest.raises(AuthoringEvidenceError):
        _validate_intent_audit(_SOURCE, _validate(plan), claim("left"))
    for effects, later in (
        ([_fact("held", "item", arms=["right"])], []),
        ([], [_step("move_held", "item", arms="right")]),
    ):
        plan["steps"][1]["effects"] = effects
        constrained = _validate(dict(plan, steps=[*plan["steps"], *later]))
        _validate_intent_audit(_SOURCE, constrained, claim("left"))
        with pytest.raises(AuthoringEvidenceError):
            _validate_intent_audit(_SOURCE, constrained, claim("right"))


@pytest.mark.parametrize("phase", ["initial", "maintain", "effect"])
def test_invariant_rejects_a_conflicting_fact_even_if_restored_later(phase):
    upright = _fact("orientation", "vase", value="upright")
    plan = _plan(
        "vase other",
        _step("push", "other", _fact("orientation", "other", value="flat")),
        _step("push", "vase", upright),
        conditions=[_condition("always", upright)],
    )
    audit = _audit(_check("condition_01", scope="always", state=upright))
    _validate_intent_audit(_SOURCE, _validate(plan), audit)
    flat = dict(upright, value="flat")
    if phase == "initial":
        plan["conditions"].append(_condition("initial", flat))
    elif phase == "maintain":
        plan["steps"][-1]["maintain"] = [flat]
    else:
        plan["steps"].insert(1, _step("push", "vase", flat))
    with pytest.raises(AuthoringEvidenceError, match="always condition"):
        _validate_intent_audit(_SOURCE, _validate(plan), audit)


@pytest.mark.parametrize(
    "components,epoch,reference,moved,comparison,conflict",
    [
        (["position", "orientation"], "initial", "panel", "panel", "gt", True),
        (["position"], "initial", "panel", "panel", "eq", True),
        (["position"], "initial", "other", "panel", "ge", True),
        (["orientation"], "initial", "panel", "panel", "gt", False),
        (["position"], "initial", "panel", "panel", "lt", False),
        (["position"], "initial", "panel", "other", "gt", False),
        (["position"], "current", "other", "panel other", "gt", False),
    ],
)
def test_fixed_position_invariant_rejects_only_required_displacement(
    components, epoch, reference, moved, comparison, conflict
):
    pose = _fact("pose_match", "panel", reference, epoch=epoch, components=components)
    rise = _fact("vertical_rise", moved, comparison=comparison, meters=0.1)
    plan = _validate(
        _plan(
            "panel other",
            _step("push", moved, rise),
            conditions=[_condition("always", pose)],
        )
    )
    audit = _audit(
        _check("condition_01", scope="always", state=pose),
        _check("step_01", scope="effect", state=rise),
    )
    if conflict:
        with pytest.raises(
            AuthoringEvidenceError,
            match="step_01 effect contradicts an always condition",
        ):
            _validate_intent_audit(_SOURCE, plan, audit)
    else:
        _validate_intent_audit(_SOURCE, plan, audit)


@pytest.mark.parametrize(
    "source", ["initial", "maintain", "stale_effect", "final", "stale_final"]
)
def test_terminal_audit_rejects_nonterminal_or_conflicting_evidence(source):
    left = _relation("left_of", "item", "base")
    right = _relation("right_of", "item", "base")
    moved = _step("push", "item", right)
    plan = _plan("item base", moved)
    reference = "step_01"
    if source in {"initial", "final"}:
        plan["conditions"] = [_condition(source, left)]
        reference = "condition_01"
    elif source == "stale_final":
        plan["conditions"] = [_condition("initial", left), _condition("final", left)]
        moved["effects"] = []
        reference = "condition_02"
    elif source == "maintain":
        moved["maintain"] = [left]
    else:
        plan["steps"].insert(0, _step("push", "item", left))
    with pytest.raises(AuthoringEvidenceError):
        _validate_intent_audit(
            _SOURCE,
            _validate(plan),
            _audit(_check(reference, scope="final", spatial=left)),
        )


def test_final_effect_persists_until_an_operand_moves():
    claim = _relation("left_of", "item", "base")
    plan = _plan(
        "item base",
        _step("pick", "item"),
        _step("place", "item", claim),
        _step("return_arms", "", arms="left right"),
    )
    final = _audit(_check("step_02", scope="final", spatial=claim))
    _validate_intent_audit(_SOURCE, _validate(plan), final)
    with pytest.raises(ValueError, match="typed claim") as error:
        _validate_intent_audit(
            _SOURCE, _validate(plan), _audit(_check("step_02", scope="final"))
        )
    assert not isinstance(error.value, AuthoringEvidenceError)
    plan["steps"].append(_step("push", "base"))
    with pytest.raises(AuthoringEvidenceError):
        _validate_intent_audit(_SOURCE, _validate(plan), final)
    _validate_intent_audit(
        _SOURCE,
        _validate(plan),
        _audit(_check("step_02", scope="effect", spatial=claim)),
    )
    with pytest.raises(ValueError, match="scope"):
        _validate_intent_audit(
            _SOURCE, _validate(plan), _audit(_check("step_02", spatial=claim))
        )


@pytest.mark.parametrize(
    "claim,wrong,mode",
    [
        (_fact("untouched_except", "item"), _fact("untouched", "site"), "always"),
        (
            _fact("actuated", "site", value="pressed"),
            _fact("actuated", "item", value="pressed"),
            None,
        ),
    ],
)
def test_audit_requires_the_complement_invariant_or_actual_actuation(
    claim, wrong, mode
):
    step = _step("press", "site", wrong)
    plan = _plan(
        "item site",
        step,
        conditions=[
            _condition("always" if mode else "final", wrong if mode else claim)
        ],
    )
    check = _check("condition_01", "step_01", state=claim, scope=mode or "effect")
    with pytest.raises(AuthoringEvidenceError, match="matching evidence"):
        _validate_intent_audit(_SOURCE, plan, _audit(check))
    if mode:
        plan["conditions"][0]["fact"] = claim
    else:
        step["effects"] = [claim]
    _validate_intent_audit(_SOURCE, plan, _audit(check))


def test_articulation_uses_declared_owner_and_a_distinct_joint():
    owner = _role("cabinet", asset_kind="articulated")
    roles = [
        owner,
        _role("flap", part_of="cabinet"),
        _role("hinge", "articulation", part_of="cabinet"),
    ]
    opened = _fact("articulated", "flap", "hinge", mechanism="hinge", value="open")
    plan = _plan(roles, _step("grasp", "flap"), _step("articulate", "flap", opened))
    with pytest.raises(ValueError, match="cabinet.*requires articulation"):
        _validate(plan)
    explicit = {**_role("cabinet"), "asset_kind": "articulated"}
    context = {"cabinet": explicit}
    rigid_flap = {**_role("flap"), "asset_kind": "rigid"}
    with pytest.raises(ValueError, match="flap.*rigid.*requires articulation"):
        _validate(plan, {**context, "flap": rigid_flap})
    parsed = _validate(plan, context)
    spec, _, _, _ = compile_intent(parsed, _SOURCE, "hinged", context)
    assert spec.requirements[0].value[0]["is_articulated"] is True
    opened["references"] = ["cabinet"]
    with pytest.raises(ValueError, match="distinct joint"):
        _validate(plan, context)


@pytest.mark.parametrize(
    "kind,subject,valid",
    [
        ("untouched", "device", False),
        ("untouched", "sibling", True),
        ("untouched_except", "device", True),
        ("untouched_except", "sibling", False),
    ],
)
def test_untouched_contract_respects_nested_part_ownership(kind, subject, valid):
    plan = _plan(
        [
            _role("device"),
            _role("panel", part_of="device"),
            _role("button", part_of="panel"),
            _role("sibling", part_of="device"),
        ],
        _step(
            "press", "button", _fact("actuated", "button", value="pressed"), arms="left"
        ),
        conditions=[_condition("always", _fact(kind, subject))],
    )
    context = {"device": dict(description="device", asset_kind="articulated")}
    if valid:
        _validate(plan, context)
    else:
        with pytest.raises(ValueError, match="protected roles.*button"):
            _validate(plan, context)


@pytest.mark.parametrize(
    "site,kind,subject,move,valid",
    [
        ("handle", "untouched", "rim", False, True),
        ("handle", "untouched", "rim", True, True),
        ("rim", "untouched", "rim", False, False),
        (None, "untouched", "rim", False, False),
        ("handle", "untouched", "mug", False, False),
        ("handle", "untouched_except", "handle", False, True),
    ],
)
def test_untouched_uses_explicit_grasp_sites(site, kind, subject, move, valid):
    roles = [_role("mug"), _role("handle", part_of="mug"), _role("rim", part_of="mug")]
    effects = [_fact("grasped_at", "mug", part=site, arms=["left"])] if site else []
    steps = [_step("pick", "mug", *effects, arms="left")]
    if move:
        steps.append(_step("move_held", "mug", arms="left"))
    plan = _plan(roles, *steps, conditions=[_condition("always", _fact(kind, subject))])
    if valid:
        _validate(plan)
    else:
        with pytest.raises(
            ValueError, match="Ordered operations touch protected roles"
        ):
            _validate(plan)


def test_bimanual_sites_and_terminal_hold_match_acquisition():
    arms = "left right"
    roles = [
        _role("carrier"),
        *(_role(s, part_of="carrier") for s in ("left_site", "right_site")),
    ]
    grasps = [
        _fact("grasped_at", "carrier", part=f"{arm}_site", arms=[arm])
        for arm in arms.split()
    ]
    pickup = _step("pick", "carrier", *grasps, arms=arms, direction="up")
    raised = _step(
        "move_held",
        "carrier",
        arms=arms,
        maintain=[_fact("orientation", "carrier", value="level")],
    )
    condition = _condition("final", _fact("held", "carrier", arms=arms.split()))
    plan = _plan(roles, pickup, raised, conditions=[condition])
    parsed = _validate(plan)
    assert parsed["steps"][0]["effects"] == grasps
    assert parsed["steps"][1]["maintain"] == raised["maintain"]
    raised["arms"] = ["auto"]
    with pytest.raises(ValueError, match="same arm"):
        _validate(plan)
    raised["arms"] = arms.split()
    plan["steps"].append(_step("release", "carrier", arms=arms))
    with pytest.raises(ValueError, match="Held fact contradicts"):
        _validate(plan)


def test_tightening_requires_prior_seating():
    seat = _step("move_held", "nut", _relation("seated_on", "nut", "stud"))
    turn = _step("turn", "nut", _relation("tightened_on", "nut", "stud"))
    plan = _plan("nut stud", _step("pick", "nut"), seat, turn)
    _validate(plan)
    plan["steps"][1:] = [turn, seat]
    with pytest.raises(ValueError, match="turn after seating"):
        _validate(plan)


def test_contents_transfer_preserves_count_and_independent_visual_observation(hybrid):
    transfer = _fact(
        "transferred", "beads", "scoop dish", material="solid", all=True, quantity=4
    )
    plan = _plan(
        "beads scoop dish",
        _step("pick", "scoop"),
        _step("transfer_contents", "scoop", transfer),
    )
    _validate(plan)
    count = transfer.pop("quantity")
    with pytest.raises(ValueError, match="requires .*quantity"):
        _validate(plan)
    transfer["quantity"] = count
    observations = {"transfers": [{"source": "scoop", "recipient": "dish"}]}
    wrong = deepcopy(plan)
    wrong["roles"][1]["description"] = "dish"

    def review(request):
        if "steps" not in request:
            assert set(request) == {"instruction", "request"}
            return observations
        assert request["independent_transfer_observations"] == observations
        if request["roles"]["role_scoop"]["description"] == "dish":
            return dict(
                issues=[dict(quote=_SOURCE, reason="Source and recipient are reversed")]
            )
        return _audit(_check("step_02", state=transfer, scope="effect"))

    result, proposals, reviews = hybrid([wrong, plan], review)
    assert proposals[-1]["previous_semantic_plan"] == wrong and len(reviews) == 3
    assert result["semantic_task_graph"]["nodes"][-1]["call"]["arguments"][
        "effects"
    ] == [transfer]


@pytest.fixture
def hybrid(tmp_path, monkeypatch):
    image = tmp_path / "source.png"
    Image.new("RGB", (10, 10)).save(image)
    output = tmp_path / "plan"

    def run(plans, review, *, instruction=_SOURCE, **options):
        candidates = iter(plans)
        proposals, reviews = [], []

        def propose(*args, **kwargs):
            proposals.append(json.loads(kwargs["user_prompt"]))
            return json.dumps({"semantic_plan": next(candidates)})

        def audit(**kwargs):
            assert kwargs["image_path"] == image
            reviews.append(json.loads(kwargs["prompt"]))
            return review(reviews[-1])

        monkeypatch.setattr(compat.VisionFacade, "complete", propose)
        monkeypatch.setattr(compat, "structured_call", audit)
        with GenSimServices(GenSimServicesCfg()) as services:
            result = publish_hybrid(
                instruction, "test", image, output, services, **options
            )
        return result, proposals, reviews

    return run


@pytest.mark.parametrize(
    "invalid_field,diagnostic",
    [
        (None, None),
        ("image_sha256", "different source image"),
        ("role", "Unknown preview box role"),
        ("bbox", "normalized bbox"),
    ],
)
def test_preview_boxes_are_bound_to_the_source_without_rewriting_the_plan(
    hybrid, tmp_path, invalid_field, diagnostic
):
    plan_dir = tmp_path / "plan"
    intent = _plan(
        "board shelf",
        _step("pick", "board"),
        _step("place", "board", _relation("on", "board", "shelf")),
    )
    result, _, _ = hybrid([intent], lambda _: _audit(_check("step_02")))
    sealed = {path.name: path.read_bytes() for path in plan_dir.iterdir()}
    sha256 = load_task_plan(plan_dir)["image_sha256"]
    boxes = {"board": [0.1, 0.2, 0.8, 0.6], "shelf": None}
    packet = {"image_sha256": sha256, "boxes": boxes}
    if invalid_field == "image_sha256":
        packet["image_sha256"] = "0" * 64
    elif invalid_field == "role":
        boxes["undeclared"] = boxes["board"]
    elif invalid_field == "bbox":
        boxes["board"] = [0.8, 0.2, 0.1, 0.6]
    sidecar = tmp_path / "boxes.json"
    sidecar.write_text(json.dumps(packet), encoding="utf-8")
    output = tmp_path / "overlay.html"
    argv = [
        "--plan",
        str(plan_dir),
        "--output",
        str(output),
        "--preview-boxes",
        str(sidecar),
    ]
    if diagnostic:
        with pytest.raises(ValueError, match=diagnostic):
            review_main(argv)
        assert not output.exists()
    else:
        review_main(argv)
        match = re.search(
            r'<script id="graph-data" type="application/json">(.*?)</script>',
            output.read_text(encoding="utf-8"),
            re.S,
        )
        assert match is not None
        payload = json.loads(match[1])
        assert payload["review"]["preview_boxes"] == boxes
        assert "source_boxes" not in payload["review"]
        assert payload["review"]["spec"] == result["task_spec_v1"]
        assert payload["graph"] == result["semantic_task_graph"]
    assert {path.name: path.read_bytes() for path in plan_dir.iterdir()} == sealed
    assert not load_task_plan(plan_dir)["execution_ready"]


@pytest.mark.parametrize(
    "invalid", [False, True], ids=["visible-and-missing", "invalid-box"]
)
def test_auto_preview_boxes_localize_frozen_roles_without_rewriting_plan(
    hybrid, tmp_path, monkeypatch, invalid
):
    plan_dir = tmp_path / "plan"
    intent = _plan(
        "board shelf",
        _step("pick", "board"),
        _step("place", "board", _relation("on", "board", "shelf")),
    )
    result, _, _ = hybrid([intent], lambda _: _audit(_check("step_02")))
    sealed = {path.name: path.read_bytes() for path in plan_dir.iterdir()}
    spec = result["task_spec_v1"]
    requests = []

    def localize(**kwargs):
        requests.append(kwargs)
        assert kwargs["role"] == "task_image_grounding"
        assert kwargs["image_path"] == plan_dir / spec["inputs"]["image"]["ref"]
        assert set(json.loads(kwargs["prompt"])["roles"]) == {"board", "shelf"}
        return {
            "board": [700, 200, 200, 900] if invalid else [100, 200, 750, 900],
            "shelf": None,
        }

    monkeypatch.setattr(compat, "structured_call", localize)
    output = tmp_path / "auto-preview.html"
    argv = ["--plan", str(plan_dir), "--output", str(output), "--auto-preview-boxes"]
    if invalid:
        with pytest.raises(ValueError, match="preview bbox"):
            review_main(argv)
        assert not output.exists()
    else:
        review_main(argv)
        match = re.search(
            r'<script id="graph-data" type="application/json">(.*?)</script>',
            output.read_text(encoding="utf-8"),
            re.S,
        )
        assert match is not None
        review = json.loads(match[1])["review"]
        assert review["preview_boxes"] == {
            "board": [0.1, 0.2, 0.75, 0.9],
            "shelf": None,
        }
        assert "source_boxes" not in review
        assert review["spec"] == spec
        original = output.read_bytes()
        with pytest.raises(FileExistsError):
            review_main(argv)
        assert output.read_bytes() == original
    assert len(requests) == 1
    assert {path.name: path.read_bytes() for path in plan_dir.iterdir()} == sealed
    assert not load_task_plan(plan_dir)["execution_ready"]


@pytest.mark.parametrize(
    "terminal", ["implicit", "effect_only", "audited_effect", "declaration_only"]
)
def test_authoring_publication_is_readable_without_execution(
    hybrid, tmp_path, terminal
):
    support = _relation("supported_by", "board", "short tall")
    parked = dict(kind="arms_at_initial_pose", arms=["left", "right"])
    plan = _plan(
        "board short tall",
        _step("pick", "board"),
        _step("place", "board", support),
        _step("return_arms", "", parked, arms="left right"),
    )
    expected_goals = {"final_step_completed": ["step_03"]}
    checks = [_check("step_02", "step_03")]
    if terminal == "effect_only":
        checks = [_check("step_02", "step_03", spatial=support, scope="final")]
        expected_goals = {"condition_01": ["step_02"]}
    elif terminal == "audited_effect":
        plan["conditions"] = [_condition("final", fact) for fact in (support, parked)]
        checks = [
            _check(
                "condition_01", "step_02", "step_03", spatial=support, scope="final"
            ),
            _check("condition_02", "step_03", state=parked, scope="final"),
        ]
        expected_goals = {"condition_01": ["step_02"], "condition_02": ["step_03"]}
    elif terminal == "declaration_only":
        plan["conditions"] = [_condition("final", parked)]
        checks = [_check("condition_01", state=parked, scope="final")]
        expected_goals = {}
    background = dict(id="stapler", description="stapler", asset_kind="articulated")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            dict(
                schema_version="embodichain.task-scene-context/v1", objects=[background]
            )
        )
    )
    result, _, _ = hybrid([plan], lambda _: _audit(*checks), scene_context=context)
    published = load_task_plan(tmp_path / "plan")
    assert published["stage"] == "authoring_only" and not published["execution_ready"]
    assert "stapler" not in {row["id"] for row in result["task_spec"]["roles"]}
    assert {**background, "is_articulated": True} in published["scene_requirements"][
        "configuration"
    ]["requirements"][0]["value"]
    preview = tmp_path / "review.html"
    write_review(tmp_path / "plan", preview)
    for html in (tmp_path / "plan/semantic_task_graph.html", preview):
        data = re.search(
            r'<script id="graph-data" type="application/json">(.*?)</script>',
            html.read_text(),
            re.S,
        )
        assert data is not None
        payload = json.loads(data[1])
        assert payload["task_spec"] == result["task_spec"]
        assert payload["review"]["source_image"].startswith("data:image/jpeg;base64,")
        pairing = payload["review"]["pairing"]
        assert pairing["role_entities"] == {
            r["id"]: [r["id"]] for r in result["task_spec"]["roles"]
        }
        assert pairing["milestone_groups"] == {
            "step_01": "step_01",
            "step_02": "step_02",
            "step_03": "step_03",
        }
        assert pairing["goal_groups"] == expected_goals
        assert pairing["goal_calls"] == expected_goals
        assert pairing["milestone_calls"] == pairing["milestone_groups"]
    graph = result["semantic_task_graph"]
    if terminal == "effect_only":
        assert graph["success"]["conditions"]["final"] == [support]
        assert result["task_spec"]["goal"][0]["id"] == "condition_01"
    assert graph["nodes"][1]["call"]["arguments"]["effects"] == [support]
    terminal = graph["nodes"][-1]
    assert terminal["depends_on"] == ["step_02"]
    args = terminal["call"]["arguments"]
    assert args["arms"] == ["left", "right"]
    assert (
        args["pose_reference"] == "each_arm_initial_pose"
        and args["binding_status"] == "pending"
    )
    with pytest.raises(ValueError, match="authoring-only"):
        validate_semantic_task_graph({**graph, "integration_fingerprint": "1" * 64})


def test_typed_authoring_keeps_the_scene_context_id(hybrid, tmp_path):
    context = tmp_path / "scene.json"
    button = dict(id="Button", description="button", asset_kind="articulated")
    context.write_text(
        json.dumps(
            dict(schema_version="embodichain.task-scene-context/v1", objects=[button])
        )
    )
    hover = dict(
        kind="gripper_relation", arms=["left"], references=["Button"], value="above"
    )
    plan = _plan(
        [_role("Button", kind="button") | {"description": "button"}],
        _step("move_gripper", "", hover, arms="left"),
        _step(
            "press", "Button", _fact("actuated", "Button", value="pressed"), arms="left"
        ),
    )
    alias = _plan([_role("button", kind="button")], _step("pick", "button"))
    with pytest.raises(ValueError, match="Use scene-context ID 'Button'"):
        _validate(alias, {"Button": button})
    _validate(plan, {"Button": button, "button": dict(button, id="button")})
    result, _, _ = hybrid(
        [plan], lambda _: _audit(_check("step_01", "step_02")), scene_context=context
    )
    assert result["task_spec"]["roles"][0]["id"] == "Button"
    assert result["semantic_task_graph"]["nodes"][0]["call"]["arguments"][
        "effects"
    ] == [hover]


def test_invalid_positive_review_citation_is_repaired_without_replanning(hybrid):
    plan = _plan("item", _step("pick", "item"))

    def review(request):
        quote = (
            "source_span_01"
            if "review_format_feedback" in request
            else "Not the source"
        )
        return _audit(_check("step_01", quote=quote))

    result, proposals, reviews = hybrid([plan], review)
    assert len(proposals) == 1 and len(reviews) == 2
    assert reviews[0]["steps"] == reviews[1]["steps"]
    assert result["source_audits"][-1]["checks"][0]["quote"] == _SOURCE


@pytest.mark.parametrize("explicit", [False, True])
def test_assembly_stability_is_only_required_when_requested(hybrid, explicit):
    clause = "Nest the small bowl in the large bowl and release it."
    source = clause + (" Leave the stack stable." if explicit else "")
    inside = _relation("inside", "small", "large")
    plan = _plan(
        "small large",
        _step("pick", "small", quote=clause),
        _step("place", "small", inside, quote=clause),
    )
    issue = dict(
        quote="Leave the stack stable." if explicit else clause,
        reason="The graph establishes the nesting order but does not encode that the assembled stack is stable.",
    )
    requests = []

    def review(request):
        requests.append(request)
        if "review_format_feedback" in request:
            return _audit(
                _check("step_02", quote=clause, scope="effect", spatial=inside)
            )
        return dict(issues=[issue])

    if explicit:
        with pytest.raises(ValueError, match="stable"):
            hybrid([plan] * 4, review, instruction=source)
        assert len(requests) == 1
    else:
        result, proposals, reviews = hybrid([plan] * 4, review, instruction=source)
        assert len(proposals) == 1 and len(reviews) == 2
        assert reviews[0]["steps"] == reviews[1]["steps"]
        assert result["source_audits"][0]["issues"] == [issue]
        assert result["source_audits"][-1]["issues"] == []


@pytest.mark.parametrize("repair", [False, True])
def test_recorded_scan_review_uses_the_original_clause(hybrid, repair):
    path = Path(__file__).with_name("data") / "robotwin_scan_object_reply.json"
    packet = json.loads(path.read_text(encoding="utf-8"))
    plan = packet["proposal"]["semantic_plan"]
    checks = packet["audit"]["checks"]
    bad_index = next(
        i for i, row in enumerate(checks) if row["evidence"] == ["step_02"]
    )
    exact = plan["steps"][1]["quote"]

    def review(request):
        audit = deepcopy(packet["audit"])
        if (not repair or "review_format_feedback" in request) and (
            request.get("source_spans", {}).get("source_span_02") == exact
        ):
            audit["checks"][bad_index]["quote"] = "source_span_02"
        return audit

    result, proposals, reviews = hybrid(
        [plan] * 4, review, instruction=packet["instruction"]
    )
    assert len(proposals) == 1 and len(reviews) == 1 + repair
    if repair:
        feedback = reviews[1]["review_format_feedback"]
        assert f"checks[{bad_index}].quote" in feedback["diagnostic"]
        assert checks[bad_index]["quote"] in feedback["diagnostic"]
    assert result["source_audits"][-1]["checks"][bad_index]["quote"] == exact
    assert result["semantic_intent"]["steps"][-1]["effects"] == [
        _fact("orientation", "scanner_head", "box", value="pointing_toward")
    ]
    assert result["coverage"]["goal_producers"] == {
        "condition_08": ["step_01"],
        "condition_09": ["step_02"],
    }


def test_citation_repair_cannot_downgrade_final_scope(hybrid, tmp_path):
    left = _relation("left_of", "item", "base")
    plan = _plan("item base", _step("push", "item", left), _step("push", "item"))
    reviews = []

    def review(request):
        repairs = "review_format_feedback" in request
        reviews.append(repairs)
        return _audit(
            _check(
                "step_01",
                spatial=left,
                scope="effect" if repairs else "final",
                quote=_SOURCE if repairs else "Not the source",
            )
        )

    with pytest.raises(ValueError, match="Unchanged rejected graph"):
        hybrid([plan] * 4, review)
    assert reviews == [False, True] and not (tmp_path / "plan").exists()


def test_quote_only_repair_cannot_clear_a_source_rejection(hybrid, tmp_path):
    plan = _plan("item", _step("pick", "item", quote="Execute"))
    revised = deepcopy(plan)
    revised["steps"][0]["quote"] = _SOURCE
    reviews = []

    def reject(request):
        reviews.append(request)
        return dict(issues=[dict(quote=_SOURCE, reason="missing a requirement")])

    with pytest.raises(ValueError, match="Unchanged rejected graph"):
        hybrid([plan, revised, revised, revised], reject)
    assert len(reviews) == 1 and not (tmp_path / "plan").exists()
