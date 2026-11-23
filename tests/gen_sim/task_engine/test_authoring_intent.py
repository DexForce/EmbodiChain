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

from PIL import Image
import pytest

from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg, compat
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


def _role(name):
    return {"id": name, "kind": "object", "description": name.replace("_", " ")}


def _step(verb, quote, subjects, arms, *effects):
    return dict(
        verb=verb, quote=quote, subjects=subjects, arms=arms, effects=list(effects)
    )


def test_auto_grasps_reserve_arm_capacity_until_release():
    instruction = "Pick up and release all three packages."
    names = ("first", "second", "third")
    plan = dict(
        roles=[_role(name) for name in names],
        steps=[
            *[_step("pick", instruction, [name], ["auto"]) for name in names],
            *[_step("release", instruction, [name], ["auto"]) for name in names],
        ],
        conditions=[
            dict(
                scope="final",
                fact=dict(kind="released", subjects=list(names)),
                quote=instruction,
            )
        ],
    )
    with pytest.raises(ValueError, match="More simultaneous grasps"):
        validate_intent(plan, instruction, scene_objects={})
    plan["steps"][2], plan["steps"][3] = plan["steps"][3], plan["steps"][2]
    validate_intent(plan, instruction, scene_objects={})


def test_directed_pick_and_named_relation_shorthand_keep_the_same_semantics():
    text = "Pick up the parcel and set it on top of the shelf."
    shorthand = dict(kind="on_top_faces", subjects=["parcel"], references=["shelf"])
    plan = dict(
        roles=[_role("parcel"), {**_role("shelf"), "kind": "surface"}],
        steps=[
            {**_step("pick", text, ["parcel"], ["left"]), "direction": "up"},
            _step("place", text, ["parcel"], ["left"], shorthand),
        ],
        conditions=[dict(scope="final", fact=shorthand, quote=text)],
    )
    normalized = validate_intent(plan, text, scene_objects={})
    assert normalized["steps"][1]["effects"] == [
        {**shorthand, "kind": "relation", "value": "on_top_faces"}
    ]
    assert shorthand["kind"] == "on_top_faces"
    assert normalized["conditions"][0]["fact"] == normalized["steps"][1]["effects"][0]
    shorthand["value"] = "on_top_faces"
    assert validate_intent(plan, text, scene_objects={}) == normalized
    shorthand["value"] = "inside"
    with pytest.raises(ValueError, match="Unknown authoring fact"):
        validate_intent(plan, text, scene_objects={})
    shorthand["value"] = "on_top_faces"
    fields = {"subjects": ["parcel"], "references": ["shelf"]}
    plan["steps"][1]["effects"] = [{"on_top_faces": fields}]
    plan["conditions"][0]["fact"] = {"relation": {**fields, "value": "on_top_faces"}}
    assert validate_intent(plan, text, scene_objects={}) == normalized
    plan["steps"][0]["direction"] = "down"
    with pytest.raises(ValueError, match="Invalid movement direction for pick"):
        validate_intent(plan, text, scene_objects={})


def test_immediate_release_after_placement_retains_its_clause_without_a_duplicate_step():
    text = "Pick up the parcel and set it on the shelf. Release it."
    on_shelf = dict(
        kind="relation", value="on", subjects=["parcel"], references=["shelf"]
    )
    recorded_release = dict(kind="released", subjects=["parcel"])
    plan = dict(
        roles=[{**_role("parcel"), "part_of": None}, _role("shelf")],
        steps=[
            _step("pick", "pick up the parcel", ["parcel"], ["left"]),
            _step("place", "set it on the shelf", ["parcel"], ["left"], on_shelf),
            _step("release", "source_span_02", ["parcel"], ["left"], recorded_release),
        ],
        conditions=[],
    )
    normalized = validate_intent(plan, text, scene_objects={})
    _, graph, clauses, _ = compile_intent(normalized, text, "parcel", {})
    assert [node["task_type"] for node in graph["nodes"]] == ["pick", "place"]
    assert recorded_release in graph["nodes"][-1]["call"]["arguments"]["effects"]
    assert any(
        c["target"] == "step_02" and c["quote"] == "Release it." for c in clauses
    )
    assert validate_intent(normalized, text, scene_objects={}) == normalized
    assert "part_of" not in normalized["roles"][0]
    assert normalized["steps"][0]["quote"] == "Pick up the parcel"
    assert len(plan["steps"]) == 3
    plan["steps"][-1]["arms"] = ["right"]
    with pytest.raises(ValueError, match="requires an earlier acquisition"):
        validate_intent(plan, text, scene_objects={})
    plan["steps"][0]["quote"] = "Acquire the parcel"
    with pytest.raises(ValueError, match="Not a verbatim source span"):
        validate_intent(plan, text, scene_objects={})


def test_proximity_review_requires_a_typed_reference_not_a_one_axis_selector():
    instruction = "Put the package near the center of the shelf."
    at_slot = dict(
        kind="relation", value="at", subjects=["package"], references=["slot"]
    )
    near_center = dict(
        kind="relation",
        value="near_center_of",
        subjects=["package"],
        references=["shelf"],
    )
    plan = dict(
        roles=[
            _role("package"),
            {**_role("slot"), "kind": "target"},
            {**_role("center"), "kind": "target"},
            {**_role("shelf"), "kind": "surface"},
        ],
        steps=[
            _step("pick", instruction, ["package"], ["auto"]),
            _step("place", instruction, ["package"], ["auto"], at_slot),
        ],
        conditions=[
            dict(
                scope="initial",
                fact=dict(
                    kind="view_location",
                    frame="table",
                    depth="middle",
                    subjects=["slot"],
                ),
                quote=instruction,
            ),
            dict(scope="final", fact=at_slot, quote=instruction),
        ],
    )
    check = dict(
        quote=instruction,
        requirement="package near the center of the shelf",
        evidence=["step_02", "condition_01"],
        spatial=near_center,
    )
    audit = dict(issues=[], checks=[check])
    with pytest.raises(ValueError, match="view_location or labels are not enough"):
        _validate_intent_audit(instruction, plan, audit)
    plan["conditions"][0]["fact"] = {**near_center, "subjects": ["slot"]}
    _validate_intent_audit(instruction, plan, audit)
    plan["conditions"][0]["fact"] = {
        **near_center,
        "value": "near",
        "subjects": ["slot"],
        "references": ["center"],
    }
    plan["conditions"].append(
        dict(
            scope="initial",
            fact={**near_center, "value": "centered_on", "subjects": ["center"]},
            quote=instruction,
        )
    )
    check["evidence"].append("condition_03")
    _validate_intent_audit(instruction, plan, audit)
    check["spatial"] = {**near_center, "value": "on"}
    check["evidence"] = ["step_02"]
    with pytest.raises(ValueError, match="needs matching typed relations"):
        _validate_intent_audit(instruction, plan, audit)
    plan["steps"][1]["effects"].append({**near_center, "value": "centered_on"})
    _validate_intent_audit(instruction, plan, audit)


def test_exact_center_and_named_axis_are_not_generic_proximity_and_alignment():
    roles = {"board": "object", "base": "surface", "slot": "target"}
    centered = dict(
        kind="relation", value="centered_on", subjects=["board"], references=["base"]
    )
    at_slot = {**centered, "value": "at", "references": ["slot"]}
    slot_centered = {**centered, "subjects": ["slot"]}
    with pytest.raises(AuthoringEvidenceError, match="centered_on"):
        verify_spatial_claim(
            centered, [at_slot, {**slot_centered, "value": "near_center_of"}], roles
        )
    verify_spatial_claim(centered, [at_slot, slot_centered], roles)

    axis = {
        **centered,
        "kind": "orientation",
        "value": "aligned_long_axis",
    }
    with pytest.raises(AuthoringEvidenceError, match="orientation.*matching evidence"):
        verify_state_claim(axis, [{**centered, "value": "aligned_with"}], roles)
    verify_state_claim(axis, [axis], roles)


def test_initial_distance_comparison_keeps_ordered_operands_and_scope():
    instruction = (
        "The pad is farther from the robot than the parcel. Pick up the parcel."
    )
    comparison = dict(
        kind="distance_compare",
        subjects=["pad"],
        references=["robot", "parcel"],
        comparison="gt",
    )
    plan = dict(
        roles=[_role(name) for name in ("pad", "parcel", "robot")],
        steps=[_step("pick", "source_span_02", ["parcel"], ["left"])],
        conditions=[dict(scope="initial", fact=comparison, quote="source_span_01")],
    )
    normalized = validate_intent(plan, instruction, scene_objects={})
    audit = dict(
        issues=[],
        checks=[
            dict(
                quote="farther from the robot than the parcel",
                requirement="relative distance at the start",
                evidence=["condition_01", "step_01"],
                scope="initial",
                state=comparison,
            )
        ],
    )
    _validate_intent_audit(instruction, normalized, audit)
    condition = normalized["conditions"][0]
    condition["fact"]["references"].reverse()
    with pytest.raises(
        AuthoringEvidenceError, match="distance_compare.*matching evidence"
    ):
        _validate_intent_audit(instruction, normalized, audit)
    condition["fact"]["references"].reverse()
    condition["scope"] = "final"
    with pytest.raises(AuthoringEvidenceError, match="needs an initial condition"):
        _validate_intent_audit(instruction, normalized, audit)

    plan["conditions"][0]["fact"] = {
        **comparison,
        "kind": "distance_change",
        "references": ["robot"],
        "epoch": "initial",
    }
    with pytest.raises(ValueError, match="use distance_compare"):
        validate_intent(plan, instruction, scene_objects={})


def test_pointing_toward_a_named_role_is_not_a_frame_axis():
    instruction = "The arrow points toward the marker. Pick up the arrow."
    direction = dict(
        kind="orientation",
        value="pointing_toward",
        subjects=["arrow"],
        references=["marker"],
    )
    plan = dict(
        roles=[_role("arrow"), _role("marker")],
        steps=[_step("pick", "Pick up the arrow.", ["arrow"], ["left"])],
        conditions=[
            dict(
                scope="initial",
                fact=dict(
                    kind="orientation",
                    value="pointing_forward",
                    subjects=["arrow"],
                    frame="table",
                ),
                quote="The arrow points toward the marker.",
            )
        ],
    )
    audit = dict(
        issues=[],
        checks=[
            dict(
                quote="points toward the marker",
                requirement="arrow initially points at marker",
                evidence=["condition_01"],
                scope="initial",
                state=direction,
            )
        ],
    )
    parsed = validate_intent(plan, instruction, scene_objects={})
    with pytest.raises(AuthoringEvidenceError, match="orientation.*matching evidence"):
        _validate_intent_audit(instruction, parsed, audit)
    plan["conditions"][0]["fact"] = {**direction, "references": ["arrow"]}
    with pytest.raises(ValueError, match="requires one distinct target role"):
        validate_intent(plan, instruction, scene_objects={})
    plan["conditions"][0]["fact"] = direction
    parsed = validate_intent(plan, instruction, scene_objects={})
    _validate_intent_audit(instruction, parsed, audit)


@pytest.mark.parametrize(
    ("relation", "evidence_relation", "wrong_relation", "wrong_reference"),
    [
        ("left_of", "left_of", "right_of", "opening"),
        ("above", "centered_above", "above", "container"),
        ("inside", "inside", "not_inside", "opening"),
    ],
)
def test_directional_review_requires_the_named_spatial_operand(
    relation, evidence_relation, wrong_relation, wrong_reference
):
    roles = {"object": "object", "opening": "target", "container": "container"}
    claim = dict(
        kind="relation", value=relation, subjects=["object"], references=["opening"]
    )
    with pytest.raises(AuthoringEvidenceError, match="matching typed relations"):
        verify_spatial_claim(
            claim,
            [{**claim, "value": wrong_relation, "references": [wrong_reference]}],
            roles,
        )
    verify_spatial_claim(claim, [{**claim, "value": evidence_relation}], roles)


def test_joint_relation_evidence_cannot_mix_distinct_groups():
    roles = {name: "object" for name in ("parcel", "a", "b", "c", "d")}
    claim = dict(
        kind="relation", value="between", subjects=["parcel"], references=["a", "b"]
    )
    with pytest.raises(AuthoringEvidenceError, match="matching typed relations"):
        verify_spatial_claim(
            claim,
            [{**claim, "references": list(refs)} for refs in (("a", "c"), ("b", "d"))],
            roles,
        )
    verify_spatial_claim(claim, [{**claim, "references": ["b", "a"]}], roles)


def test_complement_protection_cannot_be_witnessed_by_a_descriptive_group():
    instruction = "Pick the marker. Leave the other objects untouched."
    state = dict(kind="untouched_except", subjects=["marker"])
    plan = dict(
        roles=[_role("marker"), _role("others")],
        steps=[_step("pick", "Pick the marker", ["marker"], ["right"])],
        conditions=[
            dict(scope="always", fact=dict(kind="untouched", subjects=["others"]))
        ],
    )
    audit = dict(
        issues=[],
        checks=[
            dict(
                quote="Leave the other objects untouched.",
                requirement="protect the complement",
                evidence=["condition_01"],
                scope="always",
                state=state,
            )
        ],
    )
    with pytest.raises(
        AuthoringEvidenceError, match="untouched_except.*matching evidence"
    ):
        _validate_intent_audit(instruction, plan, audit)
    plan["conditions"][0]["fact"] = state
    _validate_intent_audit(instruction, plan, audit)


def test_actuation_proof_requires_effect_at_the_named_site():
    instruction = "Press the button on the face."
    state = dict(kind="actuated", subjects=["button"], value="pressed")
    plan = dict(
        roles=[_role("button"), _role("face")],
        steps=[
            _step(
                "press",
                instruction,
                ["face"],
                ["left"],
                {**state, "subjects": ["face"]},
            )
        ],
        conditions=[dict(scope="final", fact=state)],
    )
    audit = dict(
        issues=[],
        checks=[
            dict(
                quote=instruction,
                requirement="press the button, not the face",
                evidence=["step_01", "condition_01"],
                state=state,
            )
        ],
    )
    with pytest.raises(AuthoringEvidenceError, match="actuated.*matching evidence"):
        _validate_intent_audit(instruction, plan, audit)
    plan["steps"][0] = _step("press", instruction, ["button"], ["left"], state)
    _validate_intent_audit(instruction, plan, audit)


def test_fixed_pose_proof_cannot_substitute_the_supports_initial_pose():
    roles = {"parcel": "object", "table": "surface"}
    claim = dict(
        kind="pose_match",
        subjects=["parcel"],
        references=["parcel"],
        components=["position", "orientation"],
        epoch="initial",
    )
    with pytest.raises(AuthoringEvidenceError, match="pose_match.*matching evidence"):
        verify_state_claim(claim, [{**claim, "references": ["table"]}], roles)
    verify_state_claim(claim, [claim], roles)


def test_bimanual_grasp_preserves_each_site_and_cannot_turn_into_auto():
    text = "Grasp both handles together, then raise the carrier while holding both handles."
    carrier = _role("carrier")
    sites = [
        {**_role(f"{arm}_handle"), "part_of": "carrier"} for arm in ("left", "right")
    ]
    grasps = [
        dict(kind="grasped_at", subjects=["carrier"], part=site["id"], arms=[arm])
        for site, arm in zip(sites, ("left", "right"))
    ]
    arms = ["left", "right"]
    level = dict(kind="orientation", value="level", subjects=["carrier"])
    plan = dict(
        roles=[carrier, *sites, _role("table")],
        steps=[
            _step("pick", "Grasp both handles together", ["carrier"], arms, *grasps),
            {
                **_step(
                    "move_held",
                    "raise the carrier",
                    ["carrier"],
                    arms,
                    dict(
                        kind="relation",
                        value="above",
                        subjects=["carrier"],
                        references=["table"],
                    ),
                ),
                "direction": "up",
                "maintain": [level],
            },
        ],
        conditions=[
            dict(
                scope="final",
                fact=dict(kind="held", subjects=["carrier"], arms=arms),
                quote=text,
            )
        ],
    )
    validate_intent(plan, text, scene_objects={})
    _, graph, _, _ = compile_intent(plan, text, "carry", {})
    assert graph["nodes"][0]["call"]["arguments"]["effects"] == grasps
    assert graph["nodes"][1]["call"]["arguments"]["direction"] == "up"
    assert graph["nodes"][1]["call"]["arguments"]["maintain"] == [level]
    plan["steps"][1]["arms"] = ["auto"]
    with pytest.raises(ValueError, match="same arm"):
        validate_intent(plan, text, scene_objects={})


def test_explicit_continuity_is_not_proved_by_a_final_fact():
    text = "Keep the parcel level."
    fact = dict(kind="orientation", value="level", subjects=["parcel"])
    plan = dict(
        roles=[_role("parcel")],
        steps=[_step("pick", text, ["parcel"], ["auto"])],
        conditions=[dict(scope="final", fact=fact, quote=text)],
    )
    audit = dict(
        issues=[],
        checks=[
            dict(
                quote=text,
                requirement="maintain level",
                scope="always",
                evidence=["condition_01"],
            )
        ],
    )
    with pytest.raises(ValueError, match="needs an always condition"):
        _validate_intent_audit(text, plan, audit)
    plan["conditions"][0]["scope"] = "always"
    _validate_intent_audit(text, plan, audit)


def test_role_evidence_accepts_raw_identity_without_replacing_action_evidence():
    text = "Pick up the red parcel."
    plan = dict(
        roles=[_role("parcel")],
        steps=[_step("pick", text, ["parcel"], ["left"])],
        conditions=[],
    )
    identity = dict(
        quote="red parcel", requirement="object identity", evidence=["parcel"]
    )
    action = dict(
        quote="Pick up", requirement="acquire the parcel", evidence=["step_01"]
    )
    _validate_intent_audit(text, plan, dict(issues=[], checks=[identity, action]))
    identity["evidence"] = ["role_parcel"]
    _validate_intent_audit(text, plan, dict(issues=[], checks=[identity, action]))
    with pytest.raises(ValueError, match="not roles alone"):
        _validate_intent_audit(text, plan, dict(issues=[], checks=[identity]))


def test_articulation_needs_the_part_joint_not_the_stationary_body():
    text = "Grasp the cabinet flap, open it about the hinge, and release it."
    opened = dict(
        kind="articulated",
        subjects=["flap"],
        references=["hinge"],
        mechanism="hinge",
        value="open",
    )
    plan = dict(
        roles=[
            _role("cabinet"),
            {**_role("flap"), "kind": "articulation", "part_of": "cabinet"},
            {**_role("hinge"), "kind": "articulation", "part_of": "cabinet"},
        ],
        steps=[
            _step("grasp", text, ["flap"], ["auto"]),
            _step("articulate", text, ["flap"], ["auto"], opened),
            _step("release", text, ["flap"], ["auto"]),
        ],
        conditions=[dict(scope="final", fact=opened, quote=text)],
    )
    validate_intent(plan, text, scene_objects={})
    opened["references"] = ["cabinet"]
    with pytest.raises(ValueError, match="distinct joint"):
        validate_intent(plan, text, scene_objects={})


def test_implicit_completion_does_not_invent_a_terminal_state():
    text = "Press the switch down."
    plan = dict(
        roles=[_role("switch")],
        steps=[
            {
                **_step(
                    "press",
                    text,
                    ["switch"],
                    ["left"],
                    dict(kind="actuated", subjects=["switch"], value="pressed"),
                ),
                "direction": "down",
            }
        ],
        conditions=[],
    )
    validate_intent(plan, text, scene_objects={})
    spec, graph, _, _ = compile_intent(plan, text, "switch_press", {})
    assert graph["success"]["conditions"]["final"] == []
    assert graph["success"]["terminal_step"] == "step_01"
    assert spec.to_dict()["goal"][0]["arguments"] == {"terminal_step": "step_01"}


def test_hybrid_preserves_simultaneous_support_and_initial_arm_poses(
    tmp_path, monkeypatch
):
    place = "Rest the board on both supports"
    restore = "return both arms to their initial poses"
    instruction = f"{place}, then {restore}.\n"
    support = dict(
        kind="relation",
        value="supported_by",
        subjects=["board"],
        references=["short_support", "tall_support"],
    )
    parked = dict(kind="arms_at_initial_pose", arms=["left", "right"])
    plan = dict(
        roles=[_role(name) for name in ("board", "short_support", "tall_support")],
        steps=[
            _step("pick", place, ["board"], ["auto"]),
            _step("place", place, ["board"], ["auto"], support),
            _step("return_arms", restore, [], ["left", "right"]),
        ],
        conditions=[
            dict(scope="final", fact=support, quote=place),
            dict(scope="final", fact=parked, quote=restore),
        ],
    )
    monkeypatch.setattr(
        compat.VisionFacade,
        "complete",
        lambda *a, **kw: json.dumps({"semantic_plan": plan}),
    )
    monkeypatch.setattr(
        compat,
        "structured_call",
        lambda **kwargs: dict(
            issues=[],
            checks=[
                dict(
                    quote=place,
                    requirement="simultaneous support",
                    evidence=["step_02", "condition_01"],
                ),
                dict(
                    quote=restore,
                    requirement="both arms at their own starts",
                    evidence=["step_03", "condition_02"],
                ),
            ],
        ),
    )
    image, output = tmp_path / "input.png", tmp_path / "graph"
    Image.new("RGB", (10, 10)).save(image)
    with GenSimServices(GenSimServicesCfg()) as services:
        result = publish_hybrid(instruction, "two_supports", image, output, services)
    graph = result["semantic_task_graph"]
    assert graph["nodes"][1]["call"]["arguments"]["effects"] == [support]
    terminal = graph["nodes"][-1]
    assert terminal["depends_on"] == ["step_02"]
    assert terminal["call"]["arguments"]["arms"] == ["left", "right"]
    assert terminal["call"]["arguments"]["pose_reference"] == "each_arm_initial_pose"
    assert terminal["call"]["arguments"]["binding_status"] == "pending"
    with pytest.raises(ValueError, match="authoring-only"):
        validate_semantic_task_graph({**graph, "integration_fingerprint": "1" * 64})
    verified = load_task_plan(output)
    assert verified["instruction"] == instruction.rstrip()
    assert (
        verified["stage"] == "authoring_only" and verified["execution_ready"] is False
    )
    assert result["coverage"]["status"] == "source_reviewed"
    assert parked in [
        p["arguments"] for p in verified["task_document"]["objectives"]["goal"]
    ]
    assert (
        "task_engine.intent.return_arms"
        in (output / "semantic_task_graph.html").read_text()
    )
    with (
        GenSimServices(GenSimServicesCfg()) as services,
        pytest.raises(ValueError, match="operations"),
    ):
        publish_hybrid(
            instruction,
            "two_supports",
            image,
            tmp_path / "deployment",
            services,
            deployment=tmp_path / "deployment.yaml",
        )


def test_terminal_hold_is_incompatible_with_earlier_release():
    instruction = (
        "Pick up and lift the part by more than 0.1 m, keeping it in the right hand."
    )
    hold = dict(kind="held", subjects=["part"], arms=["right"])
    rise = dict(kind="vertical_rise", subjects=["part"], comparison="gt", meters=0.1)
    plan = dict(
        roles=[_role("part")],
        steps=[
            _step("pick", "Pick up", ["part"], ["right"]),
            _step(
                "move_held",
                "lift the part by more than 0.1 m",
                ["part"],
                ["right"],
                rise,
            ),
        ],
        conditions=[
            dict(scope="final", fact=hold, quote="keeping it in the right hand")
        ],
    )
    validate_intent(plan, instruction, scene_objects={})
    plan["conditions"].append({**plan["conditions"][0], "scope": "always"})
    with pytest.raises(ValueError, match="must hold before the first step"):
        validate_intent(plan, instruction, scene_objects={})
    plan["conditions"].pop()
    plan["steps"].append(
        _step("release", "keeping it in the right hand", ["part"], ["right"])
    )
    with pytest.raises(ValueError, match="Held fact contradicts"):
        validate_intent(plan, instruction, scene_objects={})


def test_tighten_requires_seating_the_same_pair_first():
    instruction = "Seat the locknut on the fixed stud and tighten it before release."
    seated = dict(
        kind="relation", value="seated_on", subjects=["locknut"], references=["stud"]
    )
    tight = {**seated, "value": "tightened_on"}
    plan = dict(
        roles=[_role("locknut"), _role("stud")],
        steps=[
            _step("pick", "Seat the locknut", ["locknut"], ["auto"]),
            _step(
                "move_held",
                "Seat the locknut on the fixed stud",
                ["locknut"],
                ["auto"],
                seated,
            ),
            _step("turn", "tighten it", ["locknut"], ["auto"], tight),
            _step("release", "release", ["locknut"], ["auto"]),
        ],
        conditions=[dict(scope="final", fact=tight, quote="tighten it")],
    )
    validate_intent(plan, instruction, scene_objects={})
    plan["steps"][1:3] = plan["steps"][1:3][::-1]
    with pytest.raises(ValueError, match="turn after seating"):
        validate_intent(plan, instruction, scene_objects={})


def test_transfer_has_a_structured_count_operand():
    instruction = "Transfer all four beads from the scoop to the dish."
    transfer = dict(
        kind="transferred",
        subjects=["beads"],
        references=["scoop", "dish"],
        material="solid",
        all=True,
        quantity=4,
    )
    plan = dict(
        roles=[_role(name) for name in ("beads", "scoop", "dish")],
        steps=[
            _step("pick", instruction, ["scoop"], ["auto"]),
            _step("transfer_contents", instruction, ["scoop"], ["auto"], transfer),
        ],
        conditions=[dict(scope="final", fact=transfer, quote=instruction)],
    )
    validate_intent(plan, instruction, scene_objects={})
    del transfer["quantity"]
    with pytest.raises(ValueError, match="requires .*quantity"):
        validate_intent(plan, instruction, scene_objects={})


def test_transfer_review_gets_candidate_independent_endpoint_evidence(
    tmp_path, monkeypatch
):
    instruction = "Transfer the beads from the scoop to the dish."
    endpoint_observations = {
        "transfers": [{"source": "oval blue scoop", "recipient": "round glass dish"}]
    }
    transfer = dict(
        kind="transferred",
        subjects=["beads"],
        references=["scoop", "dish"],
        material="solid",
        all=True,
        quantity=None,
    )
    observations = []
    proposals = []
    audits = []

    def propose(*args, **kwargs):
        proposals.append(1)
        scoop, dish = "oval blue scoop", "round glass dish"
        if len(proposals) == 1:
            scoop, dish = dish, scoop
        return json.dumps(
            {
                "semantic_plan": dict(
                    roles=[
                        _role("beads"),
                        {**_role("scoop"), "description": scoop},
                        {**_role("dish"), "description": dish},
                    ],
                    steps=[
                        _step("pick", instruction, ["scoop"], ["auto"]),
                        _step(
                            "transfer_contents",
                            instruction,
                            ["scoop"],
                            ["auto"],
                            transfer,
                        ),
                    ],
                    conditions=[],
                )
            }
        )

    def review(**kwargs):
        request = json.loads(kwargs["prompt"])
        assert kwargs["image_path"] == image
        if "steps" not in request:
            assert set(request) == {"instruction", "request"}
            observations.append(request)
            return endpoint_observations
        audits.append(request)
        assert request["independent_transfer_observations"] == endpoint_observations
        if request["roles"]["role_scoop"]["description"] == "round glass dish":
            return {
                "issues": [
                    dict(quote=instruction, reason="Container roles are reversed")
                ]
            }
        return {
            "issues": [],
            "checks": [
                dict(quote=instruction, requirement="transfer", evidence=["step_02"])
            ],
        }

    monkeypatch.setattr(compat.VisionFacade, "complete", propose)
    monkeypatch.setattr(compat, "structured_call", review)
    image = tmp_path / "input.png"
    Image.new("RGB", (10, 10)).save(image)
    output = tmp_path / "plan"
    with GenSimServices(GenSimServicesCfg()) as services:
        publish_hybrid(instruction, "transfer", image, output, services)
    spec = json.loads((output / "task_spec_v1.json").read_bytes())
    assert spec["configuration"]["descriptions"]["scoop"] == "oval blue scoop"
    assert len(observations) == 1
    assert len(audits) == len(proposals) == 2


def test_quote_only_repair_cannot_clear_an_intent_source_rejection(
    tmp_path, monkeypatch
):
    instruction = "Slide the puck onto the pad without lifting it."
    on_pad = dict(kind="relation", value="on", subjects=["puck"], references=["pad"])
    plan = dict(
        roles=[_role("puck"), _role("pad")],
        steps=[
            _step("push", "Slide the puck onto the pad", ["puck"], ["auto"], on_pad)
        ],
        conditions=[
            dict(scope="final", fact=on_pad, quote="Slide the puck onto the pad")
        ],
    )
    calls = []
    reviews = []

    def proposal(*args, **kwargs):
        calls.append(json.loads(kwargs["user_prompt"]))
        if len(calls) > 1:
            plan["steps"][0]["quote"] = instruction
        return json.dumps({"semantic_plan": plan})

    def review(**kwargs):
        assert kwargs["image_path"] == image
        reviews.append(kwargs["role"])
        return dict(
            issues=[
                dict(quote="without lifting it", reason="missing on-surface invariant")
            ]
        )

    monkeypatch.setattr(compat.VisionFacade, "complete", proposal)
    monkeypatch.setattr(compat, "structured_call", review)
    image, output = tmp_path / "input.png", tmp_path / "graph"
    Image.new("RGB", (10, 10)).save(image)
    with (
        GenSimServices(GenSimServicesCfg()) as services,
        pytest.raises(
            ValueError, match="Unchanged rejected graph: .*missing on-surface invariant"
        ),
    ):
        publish_hybrid(instruction, "puck", image, output, services)
    assert len(calls) == 4 and reviews == ["task_image_grounding"]
    assert not output.exists()


def test_semantic_repair_keeps_the_last_typed_candidate(tmp_path, monkeypatch):
    instruction = "Hang the marker by its eyelet on the rail."
    hanging = dict(
        kind="hung_by", subjects=["marker"], references=["rail"], part="eyelet"
    )
    extras = [
        dict(kind="relation", value="on", subjects=["marker"], references=["rail"]),
        dict(kind="orientation", value="lying_on_side", subjects=["marker"]),
        dict(kind="grasped_at", subjects=["marker"], part="eyelet", arms=["left"]),
    ]
    plans = [
        dict(
            roles=[
                _role("marker"),
                _role("rail"),
                {**_role("eyelet"), "part_of": "marker"},
            ],
            steps=[
                _step("pick", instruction, ["marker"], ["left"], *extras[2:n]),
                _step(
                    "place", instruction, ["marker"], ["left"], hanging, *extras[:n][:2]
                ),
            ],
            conditions=[],
        )
        for n in (3, 2, 1, 0)
    ]
    attempts = []

    def propose(*args, **kwargs):
        prompt = json.loads(kwargs["user_prompt"])
        if attempts:
            assert prompt["previous_semantic_plan"] == attempts[-1]
        plan = plans[len(attempts)]
        attempts.append(plan)
        return json.dumps({"semantic_plan": plan})

    def review(**kwargs):
        if len(attempts) < len(plans):
            return {
                "issues": [
                    {"quote": instruction, "reason": "An extra fact is unsupported"}
                ]
            }
        return {
            "issues": [],
            "checks": [
                dict(
                    quote=instruction,
                    requirement="hang by the eyelet",
                    evidence=["step_02"],
                )
            ],
        }

    monkeypatch.setattr(compat.VisionFacade, "complete", propose)
    monkeypatch.setattr(compat, "structured_call", review)
    image = tmp_path / "input.png"
    Image.new("RGB", (10, 10)).save(image)
    with GenSimServices(GenSimServicesCfg()) as services:
        result = publish_hybrid(
            instruction, "marker", image, tmp_path / "plan", services
        )
    nodes = result["semantic_task_graph"]["nodes"]
    assert [node["call"]["arguments"]["effects"] for node in nodes] == [[], [hanging]]


def test_invalid_positive_review_citation_is_repaired_without_replanning(
    tmp_path, monkeypatch
):
    instruction = "Slide the puck onto the pad."
    on_pad = dict(kind="relation", value="on", subjects=["puck"], references=["pad"])
    plan = dict(
        roles=[_role("puck"), _role("pad")],
        steps=[_step("push", instruction, ["puck"], ["auto"], on_pad)],
        conditions=[],
    )
    proposals = []
    reviews = []

    def propose(*args, **kwargs):
        proposals.append(1)
        return json.dumps({"semantic_plan": plan})

    def review(**kwargs):
        request = json.loads(kwargs["prompt"])
        reviews.append(request)
        return {
            "issues": [],
            "checks": [
                dict(
                    quote=(
                        "Slide it onto the pad"
                        if len(reviews) == 1
                        else "source_span_01"
                    ),
                    requirement="slide onto pad",
                    evidence=["step_01"],
                )
            ],
        }

    monkeypatch.setattr(compat.VisionFacade, "complete", propose)
    monkeypatch.setattr(compat, "structured_call", review)
    image = tmp_path / "input.png"
    Image.new("RGB", (10, 10)).save(image)
    with GenSimServices(GenSimServicesCfg()) as services:
        result = publish_hybrid(instruction, "puck", image, tmp_path / "plan", services)
    assert result["coverage"]["status"] == "source_reviewed"
    assert proposals == [1] and len(reviews) == 2
    assert reviews[1]["steps"] == reviews[0]["steps"]
    assert reviews[1]["review_format_feedback"]["source_spans"] == {
        "source_span_01": instruction
    }
    assert result["source_audits"][-1]["checks"][0]["quote"] == instruction
