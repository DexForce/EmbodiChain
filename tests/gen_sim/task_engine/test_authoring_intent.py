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
from copy import deepcopy

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

_SOURCE = "Execute the described task."


def _role(name, kind="object", **fields):
    return dict(id=name, kind=kind, description=name.replace("_", " "), **fields)


def _fact(kind, subjects, references="", **fields):
    fact = dict(kind=kind, subjects=subjects.split(), **fields)
    if references:
        fact["references"] = references.split()
    return fact


def _relation(value, subjects, references):
    return _fact("relation", subjects, references, value=value)


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


def test_joint_relation_cannot_be_assembled_from_separate_groups():
    roles = dict.fromkeys("abcd", "object")
    claim = _relation("between", "a", "b c")
    with pytest.raises(AuthoringEvidenceError):
        verify_spatial_claim(
            claim, [_relation("between", "a", pair) for pair in ("b d", "c d")], roles
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
    check = _check("condition_01", "step_01", state=claim)
    if mode:
        check["scope"] = mode
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
        return _audit(_check("step_02", state=transfer))

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

    def run(plans, review, **options):
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
            result = publish_hybrid(_SOURCE, "test", image, output, services, **options)
        return result, proposals, reviews

    return run


def test_authoring_publication_is_readable_without_execution(hybrid, tmp_path):
    support = _relation("supported_by", "board", "short tall")
    plan = _plan(
        "board short tall",
        _step("pick", "board"),
        _step("place", "board", support),
        _step("return_arms", "", arms="left right"),
    )
    background = dict(id="stapler", description="stapler", asset_kind="articulated")
    context = tmp_path / "context.json"
    context.write_text(
        json.dumps(
            dict(
                schema_version="embodichain.task-scene-context/v1", objects=[background]
            )
        )
    )
    result, _, _ = hybrid(
        [plan], lambda _: _audit(_check("step_02", "step_03")), scene_context=context
    )
    published = load_task_plan(tmp_path / "plan")
    assert published["stage"] == "authoring_only" and not published["execution_ready"]
    assert "stapler" not in {row["id"] for row in result["task_spec"]["roles"]}
    assert {**background, "is_articulated": True} in published["scene_requirements"][
        "configuration"
    ]["requirements"][0]["value"]
    assert (tmp_path / "plan/semantic_task_graph.html").is_file()
    graph = result["semantic_task_graph"]
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
