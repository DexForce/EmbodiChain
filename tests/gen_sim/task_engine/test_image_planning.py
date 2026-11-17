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

"""Catalog-driven planning preserves intent and unresolved evidence offline."""

from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import pytest
from PIL import Image
from embodichain.gen_sim.task_engine.image_planning import (
    generate_semantic_task_graph_from_image,
    SemanticPlanningError,
)
from embodichain.gen_sim.task_engine.semantic_plan_validation import (
    validate_semantic_plan,
)
from embodichain.lab.task_program.language.semantic_contracts import (
    semantic_planning_catalog,
    validate_semantic_call_payload,
)


def _selector(reference=""):
    return dict(
        kind="scene_ref" if reference else "none",
        step_id="",
        reference=reference,
        quantifier="one",
        count=0,
    )


def _fixture(task_type="E1"):
    step = dict(
        id="step_01",
        task_type=task_type,
        object=_selector("source"),
        target=_selector(),
        relation="none",
        required_arm="auto",
        transfer_arm="none",
        receive_arm="none",
        orientation_goal="none",
        target_state="none",
        target_setting=0,
        layout="none",
        axis="none",
        direction="none",
        terminal_behavior="none",
        depends_on=[],
    )
    if task_type in {"E1", "E3", "E4", "E5"}:
        step.update(target=_selector("tray"), relation="inside")
    if task_type == "E2":
        step["orientation_goal"] = "upright"
    if task_type == "E3":
        step.update(pour_goal_mode="motion", relation="above")
    if task_type == "E4":
        step.update(
            transfer_arm="right_arm", receive_arm="left_arm", terminal_behavior="place"
        )
    if task_type == "E5":
        step["terminal_behavior"] = "place"
    if task_type in {"E6", "E7"}:
        step["target_state"] = "open"
    if task_type == "E8":
        step["twist_goal_mode"] = "coarse"
    if task_type == "E9":
        step["target_state"] = "activated"
    bindings = {"step_01.object": ["source"]}
    if step["target"]["kind"] == "scene_ref":
        bindings["step_01.target"] = ["tray"]
    grounding = dict(
        objects=[
            dict(
                id="source",
                description="source",
                bbox=[100, 200, 300, 500],
                provenance="observed",
            ),
            dict(
                id="tray",
                description="tray",
                bbox=[500, 200, 900, 600],
                provenance="observed",
            ),
        ],
        bindings=bindings,
    )

    def pick(arm=None):
        return dict(
            kind="pick",
            object="source",
            resources={} if arm is None else {"primary": arm},
        )

    def place(arm=None):
        return dict(
            kind="place",
            object="source",
            inside="inside__tray__source",
            resources={} if arm is None else {"primary": arm},
        )

    def registered(cid, args):
        return dict(kind="registered", call_id=cid, arguments=args, resources={})

    calls = {
        "E1": [pick(), place()],
        "E2": [registered("simulation.axis_align", {"object": "source"}), place()],
        "E3": [
            pick(),
            registered("simulation.move_held_object", {"target": "pour_target"}),
            registered("simulation.pour", {"object": "source"}),
            place(),
        ],
        "E4": [
            pick("right"),
            dict(
                kind="hand_over",
                object="source",
                resources={"source": "right", "destination": "left"},
            ),
            place("left"),
        ],
        "E5": [
            dict(
                kind="registered",
                call_id="simulation.coordinated_transport",
                arguments={"object": "source", "target": "tray_target"},
                resources={"left": "left", "right": "right"},
            )
        ],
        "E6": [registered("simulation.articulation_link_slide", {"handle": "source"})],
        "E7": [
            registered("simulation.open_door", {"object": "source", "state": "open"})
        ],
        "E8": [registered("simulation.twist", {"object": "source"})],
        "E9": [registered("simulation.press", {"object": "source"})],
    }[task_type]
    intent = {
        k: v
        for k, v in step.items()
        if k
        in {
            "relation",
            "orientation_goal",
            "target_state",
            "terminal_behavior",
            "pour_goal_mode",
            "twist_goal_mode",
        }
        and v != "none"
    }
    nodes = [
        dict(
            id=f"n{i}",
            step_id="step_01",
            call=call,
            depends_on=[] if i == 0 else [f"n{i-1}"],
            semantic=dict(
                object="source",
                reference="tray",
                relation=step["relation"],
                intent=intent,
            ),
        )
        for i, call in enumerate(calls)
    ]
    for node in nodes:
        if node["call"]["kind"] == "place":
            node["semantic"]["relation"] = "inside"
    plan = dict(
        nodes=nodes,
        requirements=[
            dict(id="r1", quote="执行测试任务", node_ids=[n["id"] for n in nodes])
        ],
    )
    return step, grounding, plan


@pytest.fixture
def source_image(tmp_path):
    image = tmp_path / "scene.png"
    Image.new("RGB", (100, 80), "white").save(image)
    return image


def _plan(image, task_type="E1", mutate=None, instruction="执行测试任务"):
    step, grounding, plan = _fixture(task_type)
    if mutate:
        mutate(step, grounding, plan)
    return generate_semantic_task_graph_from_image(
        image,
        instruction,
        task_id="case",
        instruction_caller=lambda **_: dict(steps=[step]),
        image_caller=lambda **_: json.dumps(grounding),
        planning_caller=lambda **_: json.dumps(plan),
    )


@pytest.mark.parametrize("task_type", [f"E{i}" for i in range(1, 10)])
def test_all_task_families_plan_using_real_catalog_without_simulation(
    source_image, task_type
):
    before = source_image.read_bytes()
    result = _plan(source_image, task_type)
    coverage = result["semantic_graph_validation"]["coverage"]
    assert coverage["semantic_status"] == "valid"
    assert coverage["execution_ready"] is False
    assert coverage["runtime_evidence_status"] == "not_run"
    assert result["binding_requirements"]
    assert result["image_grounding"]["objects"][0]["bbox"] == [0.1, 0.2, 0.3, 0.5]
    assert source_image.read_bytes() == before


def test_variation_permission_and_coverage_have_independent_identities(source_image):
    from embodichain.gen_sim.task_spec.publication import (
        project_expansion_requirements,
        project_scene_requirements,
    )

    def document(mode):
        def variation(step, grounding, plan):
            grounding["generalization"] = [
                dict(
                    domain="background",
                    scope="visual",
                    dimension="kitchen_background",
                    mode=mode,
                    quote="执行测试任务",
                )
            ]

        return _plan(source_image, mutate=variation)["task_spec_v1"]

    permitted, required = document("allow"), document("cover")
    assert permitted["task_semantic_hash"] == required["task_semantic_hash"]
    assert permitted["generalization_hash"] != required["generalization_hash"]
    assert not project_expansion_requirements(permitted, set())["required_coverage"]
    assert (
        project_expansion_requirements(required, set())["unsupported"]
        == required["generalization"]
    )
    scene = project_scene_requirements(required)
    assert scene["initial_conditions"] == []
    assert any(p.get("relation") == "inside" for p in scene["achievable_goals"])
    required["configuration"]["roles"].clear()
    with pytest.raises(ValueError, match="identity"):
        project_scene_requirements(required)


def test_task_document_requires_fresh_observations_for_each_attempt(source_image):
    from embodichain.gen_sim.task_spec.publication import evaluate_attempt_observations

    document = _plan(source_image)["task_spec_v1"]
    identity = dict(
        run_id="run",
        attempt_id="one",
        spec_digest=document["spec_digest"],
        scene_digest="1" * 64,
        program_digest="2" * 64,
        bundle_digest="3" * 64,
    )
    capture = dict(identity=identity, step=7, observations={})
    assert (
        evaluate_attempt_observations(document, identity, capture)["status"]
        == "unavailable"
    )
    with pytest.raises(ValueError, match="Stale"):
        evaluate_attempt_observations(
            document, {**identity, "attempt_id": "two"}, capture
        )


def test_ambiguous_goal_blocks_freeze_before_planning(source_image):
    def ambiguous(step, grounding, plan):
        grounding["clauses"] = [
            dict(
                id="meaning",
                quote="执行测试任务",
                target="goal_001",
                origin="explicit",
                resolution="ambiguous_goal",
            )
        ]

    with pytest.raises(ValueError, match="instruction obligation"):
        _plan(source_image, mutate=ambiguous)


def test_graph_must_account_for_explicit_qualifier(source_image):
    def omit(step, grounding, plan):
        grounding["clauses"] = [
            dict(
                id="qualifier",
                quote="测试",
                target="role_001",
                origin="explicit",
                resolution="resolved",
            )
        ]
        plan["requirements"][0]["quote"] = "执行"

    with pytest.raises(ValueError, match="omits instruction qualifier"):
        _plan(source_image, mutate=omit)


def test_explicit_role_properties_survive_standalone_scene_projection(source_image):
    from embodichain.gen_sim.task_spec.publication import (
        decode_task_document,
        project_scene_requirements,
    )

    def constrain(step, grounding, plan):
        grounding["role_properties"] = {"role_001": {"color": "red"}}
        grounding["clauses"] = [
            dict(
                id="red",
                target="role_001",
                quote="红色",
                origin="explicit",
                resolution="resolved",
            )
        ]
        plan["requirements"][0]["quote"] = "执行测试任务，物体必须是红色"

    doc = _plan(
        source_image, mutate=constrain, instruction="执行测试任务，物体必须是红色"
    )["task_spec_v1"]
    assert (
        doc["task_semantic_hash"]
        != _plan(source_image)["task_spec_v1"]["task_semantic_hash"]
    )
    roles = project_scene_requirements(decode_task_document(doc))["configuration"][
        "roles"
    ]
    assert (
        next(r for r in roles if r["id"] == "role_001")["properties"]["color"] == "red"
    )

    def structured(step, grounding, plan):
        constrain(step, grounding, plan)
        grounding["clauses"][0]["constraint"] = {
            "kind": "role_property",
            "key": "color",
            "value": "red",
        }

    explicit = _plan(
        source_image, mutate=structured, instruction="执行测试任务，物体必须是红色"
    )["task_spec_v1"]
    assert explicit["task_semantic_hash"] == doc["task_semantic_hash"]
    assert explicit["spec_digest"] != doc["spec_digest"]

    def contradictory(step, grounding, plan):
        structured(step, grounding, plan)
        grounding["clauses"][0]["constraint"]["value"] = "blue"

    with pytest.raises(ValueError, match="property constraint"):
        _plan(
            source_image,
            mutate=contradictory,
            instruction="执行测试任务，物体必须是红色",
        )


@pytest.mark.parametrize(
    "damage", ["remove_restoration", "quote_only_pick", "move_after_restoration"]
)
def test_frozen_restoration_qualifier_checks_node_semantics_on_revision_and_load(
    source_image,
    tmp_path,
    monkeypatch,
    damage,
):
    import hashlib
    import shutil
    from embodichain.gen_sim.task_engine import _image_run
    from embodichain.gen_sim.task_engine.image_planning import _assemble
    from embodichain.gen_sim.task_spec import canonical_hash
    from embodichain.gen_sim.workflow import load_task_plan, revise_task_plan

    instruction = "Pour into tray then return to original position"
    proposal = {}

    def restoration(step, grounding, plan):
        grounding["clauses"] = [
            dict(
                id="restore",
                quote="return to original position",
                target="step_01",
                origin="explicit",
                resolution="resolved",
                constraint={"kind": "return_to_origin"},
            )
        ]
        plan["nodes"][-1]["call"] = dict(
            kind="place", object="source", on="initial_support", resources={}
        )
        ann = plan["nodes"][-1]["semantic"]
        ann.update(
            reference="source",
            relation="on",
            intent={**ann["intent"], "return_to_origin": True},
        )
        plan["requirements"] = [
            dict(id="pour", quote="Pour into tray", node_ids=["n0", "n1", "n2"]),
            dict(id="restore", quote="return to original position", node_ids=["n3"]),
        ]
        proposal.update(deepcopy(plan))

    result = _plan(source_image, "E3", mutate=restoration, instruction=instruction)
    assert (
        result["semantic_graph_validation"]["qualifier_coverage"][0]["status"]
        == "covered"
    )
    monkeypatch.setattr(
        _image_run,
        "generate_semantic_task_graph_from_image",
        lambda *a, **kw: deepcopy(result),
    )
    parent = _image_run.plan_image_task(
        source_image, instruction, tmp_path / "original", task_id="case"
    )
    original = load_task_plan(parent)
    broken = deepcopy(proposal)
    if damage == "remove_restoration":
        broken["nodes"][-1]["call"] = dict(
            kind="place", object="source", inside="tray_slot", resources={}
        )
        ann = broken["nodes"][-1]["semantic"]
        ann.update(
            reference="tray",
            relation="inside",
            intent={k: v for k, v in ann["intent"].items() if k != "return_to_origin"},
        )
    elif damage == "quote_only_pick":
        broken["requirements"][1]["node_ids"] = ["n0"]
        broken["requirements"][0]["node_ids"].append("n3")
    else:
        pick = deepcopy(proposal["nodes"][0])
        pick.update(id="later_pick", depends_on=["n3"])
        place = deepcopy(pick)
        place.update(
            id="later_place",
            depends_on=["later_pick"],
            call=dict(kind="place", object="source", inside="tray_slot", resources={}),
        )
        place["semantic"].update(relation="inside", reference="tray")
        broken["nodes"].extend([pick, place])
        broken["requirements"][0]["node_ids"].extend(["later_pick", "later_place"])
    rejected = tmp_path / "rejected-revision"
    with pytest.raises(ValueError, match="qualifier.*return_to_origin"):
        revise_task_plan(
            parent,
            broken,
            rejected,
            expected_spec_digest=original["spec_digest"],
            expected_graph_digest=original["semantic_graph_sha256"],
            expected_plan_digest=original["plan_digest"],
        )
    assert not rejected.exists()

    # Even a correctly inventoried, self-consistent semantic report cannot
    # replace the frozen qualifier with a matching quote and different behavior.
    unsafe = tmp_path / "unsafe"
    shutil.copytree(parent, unsafe)
    validation = deepcopy(result["semantic_graph_validation"])
    graph, annotations, bindings = _assemble(broken, validation["source"]["candidate"])
    validation["source"].update(
        annotations=annotations, bindings=bindings, requirements=broken["requirements"]
    )
    validation.update(
        graph_digest=canonical_hash(graph),
        coverage=validate_semantic_plan(graph, **validation["source"]),
    )
    pairing = json.loads((parent / "plan_coverage.json").read_text())
    pairing["graph_digest"] = canonical_hash(graph)
    for name, value in dict(
        semantic_task_graph=graph,
        semantic_graph_validation=validation,
        plan_coverage=pairing,
    ).items():
        (unsafe / (name + ".json")).write_text(json.dumps(value))
    manifest = json.loads((unsafe / "task_plan_manifest.json").read_text())
    manifest["semantic_graph_sha256"] = canonical_hash(graph)
    manifest["artifacts"] = {
        name: hashlib.sha256((unsafe / name).read_bytes()).hexdigest()
        for name in manifest["artifacts"]
    }
    (unsafe / "task_plan_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="qualifier.*return_to_origin"):
        load_task_plan(unsafe)


@pytest.mark.parametrize("quote_pick", [False, True])
def test_orientation_qualifier_requires_the_restoring_primitive(
    source_image, quote_pick
):
    def qualify(step, grounding, plan):
        grounding["clauses"] = [
            dict(
                id="upright",
                quote="回正",
                target="step_01",
                origin="explicit",
                resolution="resolved",
                constraint={"kind": "restore_upright"},
            )
        ]
        plan["requirements"] = [
            dict(id="all", quote="倾倒", node_ids=[n["id"] for n in plan["nodes"]]),
            dict(id="upright", quote="回正", node_ids=["n0" if quote_pick else "n2"]),
        ]

    if quote_pick:
        with pytest.raises(SemanticPlanningError, match="restore_upright"):
            _plan(source_image, "E3", mutate=qualify, instruction="倾倒后回正")
    else:
        result = _plan(source_image, "E3", mutate=qualify, instruction="倾倒后回正")
        assert (
            result["semantic_graph_validation"]["qualifier_coverage"][0]["status"]
            == "covered"
        )


def test_unsupported_frozen_qualifier_remains_unverified(
    source_image, tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import _image_run
    from embodichain.gen_sim.workflow import (
        _check_scene_roles,
        load_task_plan,
        revise_task_plan,
    )

    def unsupported(step, grounding, plan):
        grounding["clauses"] = [
            dict(
                id="opaque",
                quote="执行测试任务",
                target="role_001",
                origin="explicit",
                resolution="pending_binding",
                constraint={"kind": "unsupported"},
            )
        ]

    result = _plan(source_image, mutate=unsupported)
    monkeypatch.setattr(
        _image_run, "generate_semantic_task_graph_from_image", lambda *a, **kw: result
    )
    parent = _image_run.plan_image_task(
        source_image, "执行测试任务", tmp_path / "unverified", task_id="case"
    )
    plan = load_task_plan(parent)
    assert plan["qualifier_coverage"][0]["status"] == "unverified"
    _, issues = _check_scene_roles(
        plan,
        {"objects": {"source": "source", "tray": "tray"}},
        {"source": {}, "tray": {}},
        asset_root=tmp_path,
        asset_sources={},
    )
    assert issues[0]["code"] == "unverified_instruction_qualifier"
    with pytest.raises(ValueError, match="Unverified instruction qualifiers"):
        revise_task_plan(
            parent,
            _fixture()[2],
            tmp_path / "blocked",
            expected_spec_digest=plan["spec_digest"],
            expected_graph_digest=plan["semantic_graph_sha256"],
            expected_plan_digest=plan["plan_digest"],
        )


def _terminal_hold(task_type, *, release=False):
    def mutate(s, g, p):
        s["terminal_behavior"] = "hold"
        if task_type == "E1":
            # A held tool ends above its target instead of being placed there.
            s["relation"] = "above"
        transport = dict(
            kind="registered",
            call_id="simulation.move_held_object",
            arguments={"target": "hold_pose"},
            resources={},
        )
        # A pick and one held transport replace the releasing placement; E2
        # holds its object upright where it was picked.
        nodes = p["nodes"]
        nodes[0]["call"] = dict(kind="pick", object="source", resources={})
        nodes[1]["call"] = transport
        if release:
            place = dict(
                kind="place",
                object="source",
                inside="inside__tray__source",
                resources={},
            )
            nodes.append({**deepcopy(nodes[1]), "id": "n2", "call": place})
            nodes[2]["depends_on"] = ["n1"]
            nodes[2]["semantic"]["relation"] = "inside"
            p["requirements"][0]["node_ids"].append("n2")
        if task_type == "E2":
            nodes[1]["semantic"]["reference"] = None
        for node in p["nodes"]:
            if node["call"]["kind"] != "place":
                node["semantic"]["relation"] = s["relation"]
            node["semantic"]["intent"]["terminal_behavior"] = "hold"
            if task_type == "E1":
                node["semantic"]["intent"]["relation"] = "above"

    return mutate


@pytest.mark.parametrize("task_type", ["E1", "E2"])
def test_terminal_hold_plans_keep_the_object_in_the_gripper(source_image, task_type):
    result = _plan(source_image, task_type, _terminal_hold(task_type))
    coverage = result["semantic_graph_validation"]["coverage"]
    assert coverage["semantic_status"] == "valid"


def test_terminal_hold_rejects_a_releasing_plan(source_image):
    with pytest.raises(SemanticPlanningError, match="hold is not retained"):
        _plan(source_image, "E2", _terminal_hold("E2", release=True))


@pytest.mark.parametrize("planned_height", [0.10, None, 0.20])
def test_metric_lift_keeps_the_source_height_on_the_held_transport(
    source_image, planned_height
):
    instruction = "Pick up the figurine, then lift it vertically by more than 10 centimetres. Keep it held."

    def mutate(step, grounding, plan):
        _terminal_hold("E1")(step, grounding, plan)
        # The episode's original model output declared the motion but did not
        # have a field for its height.
        step.update(target=_selector(), relation="none", direction="up")
        grounding["bindings"].pop("step_01.target")
        grounding["objects"] = [
            item for item in grounding["objects"] if item["id"] == "source"
        ]
        plan["requirements"][0][
            "quote"
        ] = "lift it vertically by more than 10 centimetres"
        for node in plan["nodes"]:
            node["semantic"].update(reference=None, relation="none")
            node["semantic"]["intent"].pop("relation", None)
        held_intent = plan["nodes"][1]["semantic"]["intent"]
        held_intent["direction"] = "up"
        if planned_height is not None:
            held_intent["lift_height_m"] = planned_height

    if planned_height != 0.10:
        with pytest.raises(SemanticPlanningError, match="lift_height_m=0.1"):
            _plan(source_image, "E1", mutate, instruction)
        return
    result = _plan(source_image, "E1", mutate, instruction)
    candidate = result["task_candidates"]["candidates"][0]
    (step,) = candidate["draft"]["steps"]
    assert step["lift_height_m"] == 0.10
    assert step["target"]["kind"] == "none"
    assert result["semantic_graph_validation"]["coverage"]["semantic_status"] == "valid"


def test_e1_placement_still_requires_a_target():
    from embodichain.gen_sim.task_engine.interpretation import (
        validate_instruction_intent,
    )

    step, _, _ = _fixture("E1")
    step["target"] = _selector()
    with pytest.raises(ValueError, match="E1 requires a target selector"):
        validate_instruction_intent({"steps": [step]})


def test_terminal_hold_acquires_with_pick_not_axis_alignment(source_image):
    def align_first(s, g, p):
        _terminal_hold("E2")(s, g, p)
        p["nodes"][0]["call"] = dict(
            kind="registered",
            call_id="simulation.axis_align",
            arguments={"object": "source"},
            resources={},
        )

    with pytest.raises(SemanticPlanningError, match="acquire it with pick"):
        _plan(source_image, "E2", align_first)


def test_terminal_hold_transport_carries_the_orientation_goal(source_image):
    def upright_on_pick_only(s, g, p):
        _terminal_hold("E2")(s, g, p)
        del p["nodes"][1]["semantic"]["intent"]["orientation_goal"]

    with pytest.raises(SemanticPlanningError, match="declare it in semantic.intent"):
        _plan(source_image, "E2", upright_on_pick_only)


@pytest.mark.parametrize("second_upright", [True, False])
def test_terminal_hold_orientation_is_checked_for_every_subject(
    source_image, second_upright
):
    def two_held(s, g, p):
        _terminal_hold("E2")(s, g, p)
        s["object"].update(quantifier="count", count=2)
        g["objects"].append(
            dict(
                id="second",
                description="second source",
                bbox=[300, 200, 400, 400],
                provenance="observed",
            )
        )
        g["bindings"]["step_01.object"].append("second")
        # Each arm picks and holds one object.
        more = deepcopy(p["nodes"])
        for node in p["nodes"]:
            node["call"]["resources"] = {"primary": "left"}
        for i, node in enumerate(more):
            node.update(id=f"n{i+2}", depends_on=[f"n{i+1}"])
            node["call"]["resources"] = {"primary": "right"}
            node["semantic"]["object"] = "second"
        more[0]["call"]["object"] = "second"
        if not second_upright:
            # The first object's transport must not satisfy the second's goal.
            del more[1]["semantic"]["intent"]["orientation_goal"]
        p["nodes"].extend(more)
        p["requirements"][0]["node_ids"].extend(["n2", "n3"])

    if second_upright:
        result = _plan(source_image, "E2", two_held)
        coverage = result["semantic_graph_validation"]["coverage"]
        assert coverage["semantic_status"] == "valid"
    else:
        with pytest.raises(SemanticPlanningError, match="ends holding second"):
            _plan(source_image, "E2", two_held)


def test_hidden_objects_have_no_invented_image_boxes(source_image):
    def hidden(s, g, p):
        g["objects"][0].update(provenance="instruction_declared", bbox=None)

    result = _plan(source_image, mutate=hidden)
    assert result["semantic_graph_validation"]["coverage"]["visual_status"] == "pending"
    assert result["image_grounding"]["objects"][0]["bbox"] is None


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_call",
        "wrong_argument",
        "missing_step",
        "wrong_object",
        "wrong_destination",
        "cycle",
        "wrong_arm",
        "unheld_place",
        "missing_intent",
        "unknown_reference",
    ],
)
def test_invalid_plans_fail_with_retained_attempt_diagnostics(source_image, damage):
    def mutate(s, g, p):
        if damage == "unknown_call":
            p["nodes"][1]["call"] = {
                "kind": "registered",
                "call_id": "invented.magic",
                "arguments": {},
                "resources": {},
            }
        elif damage == "wrong_argument":
            p["nodes"][0]["call"]["extra"] = 123
        elif damage == "missing_step":
            p["nodes"] = []
        elif damage == "wrong_object":
            p["nodes"][0]["semantic"]["object"] = "tray"
        elif damage == "wrong_destination":
            p["nodes"][1]["call"]["inside"] = "source"
        elif damage == "cycle":
            p["nodes"][0]["depends_on"] = ["n1"]
        elif damage == "wrong_arm":
            p["nodes"][1]["call"]["resources"] = {"primary": "left"}
        elif damage == "unheld_place":
            p["nodes"] = p["nodes"][1:]
            p["nodes"][0]["depends_on"] = []
            p["requirements"][0]["node_ids"] = ["n1"]
        elif damage == "missing_intent":
            s["orientation_goal"] = "upright"
            for node in p["nodes"]:
                node["semantic"]["intent"] = {}
        elif damage == "unknown_reference":
            p["nodes"][1]["semantic"]["reference"] = "missing"

    with pytest.raises(SemanticPlanningError) as failure:
        _plan(source_image, mutate=mutate)
    attempts = failure.value.artifacts["model_attempts"]
    assert len([a for a in attempts if a["stage"] == "planning"]) == 2
    assert attempts[-1]["errors"]


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [-1, 2, 4, 5], [0.1, 0.2, 0.3, 0.4]])
def test_invalid_grounding_rejects_before_graph_planning(source_image, box):
    def mutate(s, g, p):
        g["objects"][0]["bbox"] = box

    with pytest.raises(SemanticPlanningError) as failure:
        _plan(source_image, mutate=mutate)
    assert all(
        a["stage"] == "grounding" for a in failure.value.artifacts["model_attempts"]
    )


def test_empty_visual_side_tables_preserve_frozen_contract(source_image):
    baseline = _plan(source_image)

    def empty(step, grounding, plan):
        grounding.update(milestones=[], constraints=[], cross_with=[])

    actual = _plan(source_image, mutate=empty)
    assert actual["task_spec_v1"] == baseline["task_spec_v1"]
    assert actual["image_grounding"] == baseline["image_grounding"]


@pytest.mark.parametrize(
    ("damage", "diagnostic"),
    [
        ("unattached_constraint", "Unexpected grounding keys"),
        ("observed_with_candidates", "Ambiguous references"),
        ("unknown_entity_field", "Malformed entity evidence"),
    ],
)
def test_visual_compatibility_never_drops_obligations_or_invalid_evidence(
    source_image, damage, diagnostic
):
    def mutate(step, grounding, plan):
        grounding.update(constraints=[], cross_with=[])
        if damage == "unattached_constraint":
            grounding["constraints"] = [{"kind": "restore_upright"}]
        elif damage == "observed_with_candidates":
            grounding["objects"][0]["candidates"] = ["tray"]
        else:
            grounding["objects"][0]["invented"] = "tray"

    with pytest.raises(SemanticPlanningError, match=diagnostic):
        _plan(source_image, mutate=mutate)


def test_handover_direction_is_checked_against_task_contract(source_image):
    def mutate(s, g, p):
        p["nodes"][1]["call"]["resources"] = {"source": "left", "destination": "right"}

    with pytest.raises(SemanticPlanningError, match="Handover"):
        _plan(source_image, "E4", mutate)


def test_catalog_is_detached_and_does_not_import_runtime(tmp_path):
    code = "import sys; from embodichain.lab.task_program.language.semantic_contracts import semantic_planning_catalog; assert 'simulation.twist' in semantic_planning_catalog()['calls']; assert not any(n.startswith('embodichain.lab.sim') or n.startswith('dexsim') for n in sys.modules)"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    a = semantic_planning_catalog()
    a["calls"].clear()
    assert semantic_planning_catalog()["calls"]


@pytest.mark.parametrize("home_path", [False, True])
def test_preview_and_existing_output_remain_read_only(
    source_image, tmp_path, monkeypatch, home_path
):
    from embodichain.cli.main import main
    from embodichain.gen_sim.task_engine import _image_run as cli

    def extra(step, grounding, plan):
        grounding["objects"].append(
            dict(
                id="extra",
                description="unbound decoration",
                bbox=[0, 0, 100, 100],
                provenance="observed",
            )
        )

    result = _plan(source_image, mutate=extra)
    monkeypatch.setenv("HOME", str(source_image.parent))

    def generate(*a, **kw):
        assert a[0] == source_image.resolve()
        assert "model" not in kw
        return deepcopy(result)

    monkeypatch.setattr(cli, "generate_semantic_task_graph_from_image", generate)
    output = tmp_path / "preview"
    argv = [
        "task-engine",
        "--route",
        "template",
        "--image",
        "~/scene.png" if home_path else str(source_image),
        "--instruction",
        "执行测试任务",
        "--task-id",
        "case",
        "--output",
        str(output),
    ]
    main(argv)
    html = (output / "semantic_task_graph.html").read_text()
    assert "extra" in html
    assert "semantic_plan_validation/v1" in html and "data:image/png;base64," in html
    with pytest.raises(FileExistsError):
        main(argv)
    assert (output / "semantic_task_graph.html").read_text() == html
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["execution_ready"] is False and manifest["pending_bindings"] > 0
    from embodichain.gen_sim.workflow import load_task_plan

    plan = load_task_plan(output)
    assert plan["schema_version"] == "embodichain.task-plan-artifact/v2"
    assert plan["scene_requirements"]["schema_version"] == (
        "embodichain.scene-requirements/v2"
    )
    assert plan["scene_requirements"]["image_sha256"] == manifest["image_sha256"]
    from embodichain.gen_sim.workflow import revise_task_plan

    proposal = _fixture()[2]
    proposal["nodes"][-1]["id"] = "revised_place"
    proposal["requirements"][0]["node_ids"][-1] = "revised_place"
    target = tmp_path / "revised"
    with pytest.raises(ValueError, match="Stale"):
        revise_task_plan(
            output,
            proposal,
            target,
            expected_spec_digest=plan["spec_digest"],
            expected_graph_digest="0" * 64,
            expected_plan_digest=plan["plan_digest"],
        )
    assert not target.exists()
    revise_task_plan(
        output,
        proposal,
        target,
        expected_spec_digest=plan["spec_digest"],
        expected_graph_digest=plan["semantic_graph_sha256"],
        expected_plan_digest=plan["plan_digest"],
    )
    revision = load_task_plan(target)
    assert revision["semantic_graph_sha256"] != plan["semantic_graph_sha256"]
    assert revision["spec_digest"] == plan["spec_digest"]
    assert revision["scene_requirements"] == plan["scene_requirements"]


def test_publication_normalizes_identity_and_uses_its_verified_source_image(
    source_image, tmp_path, monkeypatch
):
    import base64
    from io import BytesIO
    import re
    from embodichain.gen_sim.task_engine import _image_run
    from embodichain.gen_sim.task_engine._review import write_review
    from embodichain.gen_sim.workflow import _source_image, load_task_plan

    Image.new("RGB", (100, 80), "red").save(source_image)
    result = _plan(source_image)
    monkeypatch.setattr(
        _image_run, "generate_semantic_task_graph_from_image", lambda *a, **kw: result
    )
    output = tmp_path / "normalized"
    _image_run.plan_image_task(
        source_image, "  执行测试任务\n", output, task_id=" case "
    )
    plan = load_task_plan(output)
    assert plan["instruction"] == "执行测试任务"
    assert plan["task_id"] == "case"
    Image.new("RGB", (100, 80), "blue").save(output / "source_image.jpg")
    view = tmp_path / "verified-image.html"
    write_review(output, view)
    payload = json.loads(
        re.search(
            r'<script id="graph-data" type="application/json">(.*?)</script>',
            view.read_text(),
            re.S,
        )[1]
    )
    encoded = payload["review"]["source_image"].split(",", 1)[1]
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        red, green, blue = image.getpixel((0, 0))
        assert red > 200 and green < 30 and blue < 30
    assert (
        _source_image(
            output, {k: v for k, v in plan.items() if k != "task_document"}
        ).name
        == "source_image.png"
    )


def test_plan_identity_covers_annotation_only_revisions(
    source_image, tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import _image_run
    from embodichain.gen_sim.workflow import load_task_plan, revise_task_plan

    result = _plan(source_image)
    monkeypatch.setattr(
        _image_run, "generate_semantic_task_graph_from_image", lambda *a, **kw: result
    )
    parent = _image_run.plan_image_task(
        source_image, "执行测试任务", tmp_path / "parent", task_id="case"
    )
    original = load_task_plan(parent)
    proposal = _fixture()[2]
    proposal["nodes"][0]["semantic"]["intent"][
        "description"
    ] = "revised deployment annotation"
    revised = revise_task_plan(
        parent,
        proposal,
        tmp_path / "revision",
        expected_spec_digest=original["spec_digest"],
        expected_graph_digest=original["semantic_graph_sha256"],
        expected_plan_digest=original["plan_digest"],
    )
    current = load_task_plan(revised)
    assert current["semantic_graph_sha256"] == original["semantic_graph_sha256"]
    assert current["plan_digest"] != original["plan_digest"]
    target = tmp_path / "stale"
    with pytest.raises(ValueError, match="Stale"):
        revise_task_plan(
            revised,
            proposal,
            target,
            expected_spec_digest=original["spec_digest"],
            expected_graph_digest=original["semantic_graph_sha256"],
            expected_plan_digest=original["plan_digest"],
        )
    assert not target.exists()


def test_ambiguous_reference_preserves_observed_candidates(source_image):
    def mutate(step, grounding, plan):
        grounding["objects"][0].update(
            provenance="runtime_required",
            bbox=None,
            candidates=["candidate_a", "candidate_b"],
        )
        grounding["objects"].extend(
            [
                dict(
                    id="candidate_a",
                    description="one possible source",
                    bbox=[0, 0, 100, 100],
                    provenance="observed",
                ),
                dict(
                    id="candidate_b",
                    description="another possible source",
                    bbox=[200, 0, 300, 100],
                    provenance="observed",
                ),
            ]
        )

    result = _plan(source_image, mutate=mutate)
    assert result["image_grounding"]["objects"][0]["candidates"] == [
        "candidate_a",
        "candidate_b",
    ]
    assert result["semantic_graph_validation"]["coverage"]["visual_status"] == "pending"


def test_single_repair_receives_original_response_and_error(source_image):
    step, grounding, plan = _fixture()
    invalid = deepcopy(plan)
    invalid["nodes"][0]["call"]["kind"] = "invented"
    calls = []

    def provider(**kwargs):
        payload = json.loads(kwargs["user_prompt"])
        calls.append(payload)
        return json.dumps(invalid if len(calls) == 1 else plan)

    result = generate_semantic_task_graph_from_image(
        source_image,
        "执行测试任务",
        task_id="case",
        instruction_caller=lambda **_: dict(steps=[step]),
        image_caller=lambda **_: json.dumps(grounding),
        planning_caller=provider,
    )
    assert len(calls) == 2
    assert (
        calls[1]["repair_errors"]
        and json.loads(calls[1]["previous_response"]) == invalid
    )
    assert result["semantic_graph_validation"]["coverage"]["semantic_status"] == "valid"


@pytest.mark.parametrize("max_repairs", [1, 2])
def test_repair_budget_is_explicit_and_revalidates_every_attempt(
    source_image, max_repairs
):
    step, grounding, plan = _fixture()
    invalid = deepcopy(plan)
    invalid["nodes"] = []
    calls = []

    def provider(**kwargs):
        calls.append(json.loads(kwargs["user_prompt"]))
        return json.dumps(invalid if len(calls) <= 2 else plan)

    def run():
        return generate_semantic_task_graph_from_image(
            source_image,
            "执行测试任务",
            task_id="case",
            instruction_caller=lambda **_: dict(steps=[step]),
            image_caller=lambda **_: json.dumps(grounding),
            planning_caller=provider,
            max_repairs=max_repairs,
        )

    if max_repairs == 1:
        with pytest.raises(SemanticPlanningError, match="after one repair"):
            run()
        assert len(calls) == 2
    else:
        result = run()
        assert len(calls) == 3
        assert json.loads(calls[2]["previous_response"]) == invalid
        assert (
            result["semantic_graph_validation"]["coverage"]["semantic_status"]
            == "valid"
        )


def test_one_diagnostic_names_every_placement_relation_mismatch(source_image):
    def mutate(step, grounding, plan):
        pick, place = deepcopy(plan["nodes"][0]), deepcopy(plan["nodes"][1])
        pick.update(id="n2", depends_on=["n1"])
        place.update(id="n3", depends_on=["n2"])
        plan["nodes"] += [pick, place]
        plan["requirements"][0]["node_ids"] += ["n2", "n3"]
        for node in (plan["nodes"][1], place):
            node["semantic"]["relation"] = "right_of"

    with pytest.raises(SemanticPlanningError) as failure:
        _plan(source_image, mutate=mutate)
    message = failure.value.artifacts["model_attempts"][-1]["errors"][0]["message"]
    assert "n1 uses place.inside" in message
    assert "n3 uses place.inside" in message
    assert "simulation.place_relative" in message


def test_role_id_subject_names_the_grounded_entities(source_image):
    def mutate(step, grounding, plan):
        for node in plan["nodes"]:
            node["semantic"]["object"] = "role_001"

    with pytest.raises(SemanticPlanningError) as failure:
        _plan(source_image, mutate=mutate)
    message = failure.value.artifacts["model_attempts"][-1]["errors"][0]["message"]
    assert "'role_001' is not a grounded entity ID" in message
    assert "'source'" in message and "'tray'" in message


def test_saved_report_rejects_tampered_runtime_binding(source_image):
    from embodichain.gen_sim.task_engine.semantic_graph_html import (
        render_semantic_task_graph_html,
    )

    result = _plan(source_image)
    validation = deepcopy(result["semantic_graph_validation"])
    validation["source"]["bindings"][0]["subject"] = "tray"
    with pytest.raises(ValueError):
        render_semantic_task_graph_html(
            result["semantic_task_graph"],
            task_spec=result["task_spec"],
            validation=validation,
        )


@pytest.mark.parametrize("annotate_parent", [True, False])
def test_control_part_is_linked_to_its_instruction_parent(
    source_image, annotate_parent
):
    def mutate(step, grounding, plan):
        grounding["objects"].append(
            dict(
                id="handle",
                description="drawer handle",
                parent="source",
                bbox=None,
                provenance="runtime_required",
            )
        )
        plan["nodes"][0]["call"]["arguments"]["handle"] = "handle"
        plan["nodes"][0]["semantic"]["object"] = (
            "source" if annotate_parent else "handle"
        )
        grounding["role_properties"] = {"role_001": {"color": "red"}}
        grounding["clauses"] = [
            dict(
                id="red",
                quote="红色",
                target="role_001",
                origin="explicit",
                resolution="resolved",
                constraint={"kind": "role_property", "key": "color", "value": "red"},
            )
        ]
        plan["requirements"][0]["quote"] = "执行红色抽屉测试任务"

    result = _plan(source_image, "E6", mutate, instruction="执行红色抽屉测试任务")
    assert result["semantic_task_graph"]["nodes"][0]["call"]["arguments"] == {
        "handle": "handle"
    }
    assert result["semantic_graph_validation"]["coverage"]["visual_status"] == "pending"


def test_cross_step_result_keeps_the_previous_object_identity(source_image):
    step, grounding, plan = _fixture()
    second = deepcopy(step)
    second.update(id="step_02", depends_on=["step_01"])
    second["object"] = {
        "kind": "step_result",
        "reference": "",
        "step_id": "step_01",
        "quantifier": "one",
        "count": 0,
    }
    grounding["bindings"]["step_02.target"] = ["tray"]
    more = deepcopy(plan["nodes"])
    for i, node in enumerate(more):
        node.update(id=f"n{i+2}", step_id="step_02", depends_on=[f"n{i+1}"])
    plan["nodes"].extend(more)
    plan["requirements"][0]["node_ids"].extend(["n2", "n3"])
    result = generate_semantic_task_graph_from_image(
        source_image,
        "执行测试任务",
        task_id="case",
        instruction_caller=lambda **_: dict(steps=[step, second]),
        image_caller=lambda **_: json.dumps(grounding),
        planning_caller=lambda **_: json.dumps(plan),
    )
    assert result["image_grounding"]["bindings"]["step_02.object"] == ["source"]
    assert result["semantic_graph_validation"]["coverage"]["semantic_status"] == "valid"


def test_each_member_of_a_declared_set_is_planned(source_image):
    step, grounding, plan = _fixture()
    step["object"].update(quantifier="count", count=2)
    grounding["objects"].append(
        dict(
            id="second",
            description="second source",
            bbox=[300, 200, 400, 400],
            provenance="observed",
        )
    )
    grounding["bindings"]["step_01.object"].append("second")
    more = deepcopy(plan["nodes"])
    for i, node in enumerate(more):
        node.update(id=f"n{i+2}", depends_on=[f"n{i+1}"])
        node["call"]["object"] = "second"
        node["semantic"]["object"] = "second"
        if "inside" in node["call"]:
            node["call"]["inside"] = "inside__tray__second"
    plan["nodes"].extend(more)
    plan["requirements"][0]["node_ids"].extend(["n2", "n3"])
    result = generate_semantic_task_graph_from_image(
        source_image,
        "执行测试任务",
        task_id="case",
        instruction_caller=lambda **_: dict(steps=[step]),
        image_caller=lambda **_: json.dumps(grounding),
        planning_caller=lambda **_: json.dumps(plan),
    )
    assert len(result["semantic_task_graph"]["nodes"]) == 4
    assert result["semantic_graph_validation"]["coverage"]["semantic_status"] == "valid"


def test_dense_dependencies_finish_with_correct_transitive_closure():
    code = """from embodichain.gen_sim.task_engine.semantic_plan_validation import _ancestor_sets
for dense in (False, True):
    nodes = [{"id": str(i), "depends_on": [str(j) for j in range(i)] if dense else ([str(i-1)] if i else [])} for i in range(64)]
    result = _ancestor_sets(nodes)
    assert all(result[str(i)] == {str(j) for j in range(i)} for i in range(64))
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
    )
    assert run.returncode == 0, run.stderr


def test_single_repair_gets_all_independent_symbolic_state_errors(source_image):
    def mutate(step, grounding, plan):
        plan["nodes"][1]["call"]["resources"] = {"primary": "left"}
        duplicate = deepcopy(plan["nodes"][1])
        duplicate.update(id="n2", depends_on=["n1"])
        plan["nodes"].append(duplicate)
        plan["requirements"][0]["node_ids"].append("n2")

    with pytest.raises(SemanticPlanningError) as failure:
        _plan(source_image, mutate=mutate)
    diagnostics = failure.value.artifacts["model_attempts"][-1]["errors"][0]
    assert {item["code"] for item in diagnostics["errors"]} == {
        "holder_mismatch",
        "missing_held_object",
    }
    assert {item["node_id"] for item in diagnostics["errors"]} == {"n1", "n2"}
    mismatch = next(
        item for item in diagnostics["errors"] if item["code"] == "holder_mismatch"
    )
    assert mismatch["repair_operations"] == [
        {"operation": "remove", "node_id": "n1", "path": "/call/resources/primary"}
    ]


@pytest.mark.parametrize("arm", ["left", "right"])
def test_named_arm_requests_keep_canonical_resource_ids(source_image, arm):
    def mutate(step, grounding, plan):
        step["required_arm"] = arm + "_arm"
        for node in plan["nodes"]:
            node["call"]["resources"] = {"primary": arm}

    assert (
        _plan(source_image, mutate=mutate)["semantic_graph_validation"]["coverage"][
            "semantic_status"
        ]
        == "valid"
    )


@pytest.mark.parametrize("wrong_relation", [False, True])
def test_pick_annotation_cannot_cover_wrong_placement(source_image, wrong_relation):
    def mutate(step, grounding, plan):
        plan["nodes"][0]["semantic"].update(reference="tray", relation="inside")
        placement = plan["nodes"][1]
        if wrong_relation:
            placement["call"].pop("inside")
            placement["call"]["on"] = "on_tray"
            placement["semantic"]["relation"] = "on"
        else:
            grounding["objects"].append(
                dict(
                    id="elsewhere",
                    description="another destination",
                    bbox=[0, 0, 100, 100],
                    provenance="observed",
                )
            )
            placement["call"]["inside"] = "inside_elsewhere"
            placement["semantic"]["reference"] = "elsewhere"

    with pytest.raises(SemanticPlanningError, match="target/relation"):
        _plan(source_image, mutate=mutate)


def test_coordinated_release_cannot_satisfy_requested_hold(source_image):
    def mutate(step, grounding, plan):
        step["terminal_behavior"] = "hold"
        plan["nodes"][0]["semantic"]["intent"]["terminal_behavior"] = "hold"

    with pytest.raises(SemanticPlanningError, match="Coordinated hold"):
        _plan(source_image, "E5", mutate)


@pytest.mark.parametrize("malformed", [None, [], "invalid"])
def test_malformed_intent_gets_one_diagnostic_repair(source_image, malformed):
    step, grounding, plan = _fixture()
    broken = deepcopy(plan)
    broken["nodes"][0]["semantic"]["intent"] = malformed
    attempts = []

    def provider(**kwargs):
        attempts.append(json.loads(kwargs["user_prompt"]))
        return json.dumps(broken if len(attempts) == 1 else plan)

    result = generate_semantic_task_graph_from_image(
        source_image,
        "执行测试任务",
        instruction_caller=lambda **_: dict(steps=[step]),
        image_caller=lambda **_: json.dumps(grounding),
        planning_caller=provider,
    )
    assert len(attempts) == 2 and attempts[1]["repair_errors"]
    assert result["semantic_graph_validation"]["coverage"]["semantic_status"] == "valid"


def test_handover_repair_points_to_existing_unbound_acquisition(source_image):
    def damage(step, grounding, plan):
        plan["nodes"][0]["call"]["resources"] = {}

    with pytest.raises(SemanticPlanningError) as failure:
        _plan(source_image, "E4", damage)
    diagnostics = failure.value.artifacts["model_attempts"][-1]["errors"][0]
    handover = next(
        item for item in diagnostics["errors"] if item["code"] == "handover_state"
    )
    assert handover["repair_operations"] == [
        {
            "operation": "set",
            "node_id": "n0",
            "path": "/call/resources/primary",
            "value": "right",
        }
    ]


def test_repair_receives_explicit_intent_to_logical_resource_mapping(source_image):
    step, grounding, plan = _fixture("E4")
    wrong = deepcopy(plan)
    for node in wrong["nodes"]:
        node["call"]["resources"] = {
            slot: value + "_arm" for slot, value in node["call"]["resources"].items()
        }
    requests = []

    def provider(**kwargs):
        requests.append(json.loads(kwargs["user_prompt"]))
        return json.dumps(wrong if len(requests) == 1 else plan)

    result = generate_semantic_task_graph_from_image(
        source_image,
        "执行测试任务",
        instruction_caller=lambda **_: {"steps": [step]},
        image_caller=lambda **_: json.dumps(grounding),
        planning_caller=provider,
    )
    assert len(requests) == 2
    assert requests[1]["resource_identity"]["intent_side_to_logical_resource"] == {
        "left_arm": "left",
        "right_arm": "right",
    }
    assert "logical resource 'right'" in str(requests[1]["repair_errors"])
    assert result["semantic_task_graph"]["nodes"][1]["call"]["resources"] == {
        "source": "right",
        "destination": "left",
    }
    assert (
        result["task_candidates"]["candidates"][0]["draft"]["steps"][0]["transfer_arm"]
        == "right_arm"
    )


def test_hybrid_cli_compiles_pick_place_and_rejects_hinge(
    source_image, tmp_path, monkeypatch
):
    from embodichain.gen_sim.services import compat
    from embodichain.gen_sim.task_engine.cli import main
    from embodichain.gen_sim.task_spec import canonical_hash
    from embodichain.gen_sim.task_spec.publication import decode_task_document

    root = Path(__file__).resolve().parents[3]
    deployment = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task.cobotmagic.yaml"
    )
    instruction = "把第二块积木放在第一块积木顶面"
    from embodichain.gen_sim.task_engine import _image_run

    def fixed_template(step, grounding, plan):
        step["relation"] = "on"
        grounding["objects"][0]["description"] = "第二块积木"
        grounding["objects"][1]["description"] = "第一块积木顶面"
        plan["requirements"][0]["quote"] = instruction
        for node in plan["nodes"]:
            node["semantic"].update(relation="on")
            node["semantic"]["intent"]["relation"] = "on"
            if node["call"]["kind"] == "place":
                del node["call"]["inside"]
                node["call"]["on"] = "on__tray__source"

    monkeypatch.setattr(
        _image_run,
        "generate_semantic_task_graph_from_image",
        lambda image, text, **_: _plan(image, instruction=text, mutate=fixed_template),
    )
    default = tmp_path / "default"
    main(
        [
            "--route",
            "template",
            "--image",
            str(source_image),
            "--instruction",
            instruction,
            "--output",
            str(default),
            "--task-id",
            "case",
        ]
    )
    legacy = json.loads((default / "task_plan_manifest.json").read_text())
    assert legacy["semantic_graph_sha256"] == canonical_hash(
        json.loads((default / legacy["graph_path"]).read_text())
    )
    assert (
        legacy["spec_digest"]
        == decode_task_document(
            json.loads((default / legacy["spec_path"]).read_text())
        )["spec_digest"]
    )
    op = dict(
        template="E1",
        object="block_2",
        reference="block_1_top",
        relation="on",
        quote=instruction,
    )
    monkeypatch.setattr(
        compat.VisionFacade, "complete", lambda *a, **kw: json.dumps([op])
    )
    monkeypatch.setattr(
        compat,
        "structured_call",
        lambda **kw: (
            {
                r: {
                    "description": r,
                    "bbox": (
                        [100, 200, 300, 400] if r == "block_2" else [50, 600, 550, 900]
                    ),
                }
                for r in ("block_2", "block_1_top")
            }
            if kw["role"] == "task_image_grounding"
            else {"issues": []}
        ),
    )
    out = tmp_path / "hybrid"
    main(
        [
            "--image",
            str(source_image),
            "--instruction",
            instruction,
            "--deployment",
            str(deployment),
            "--output",
            str(out),
        ]
    )

    def read(name):
        return json.loads((out / (name + ".json")).read_text())

    manifest = read("run_manifest")
    graph = read("semantic_task_graph")
    assert manifest["route"] == "hybrid"
    assert (
        manifest["spec_digest"]
        == decode_task_document(read("task_spec_v1"))["spec_digest"]
    )
    assert manifest["graph_digest"] == canonical_hash(graph)
    pick, place = [node["call"] for node in graph["nodes"]]
    assert pick["kind"] == "pick" and place["kind"] == "place"
    assert (
        pick["object"] == place["object"] == "block_2" and place["on"] == "block_1_top"
    )
    assert pick["resources"] == place["resources"] == {}
    check = read("deployment_preflight")
    assert check["compilation"] == "passed", check
    assert check["physical_verdict"] == "not_evaluated" and not check["execution_ready"]
    out = tmp_path / "hinge"
    main(
        [
            "--route",
            "goal",
            "--instruction",
            "关上铰链门",
            "--deployment",
            str(deployment),
            "--proposal",
            str(root / "examples/gensim/goal_authoring/close_hinge.json"),
            "--output",
            str(out),
        ]
    )
    check = read("deployment_preflight")
    assert check["compilation"] == "rejected" and not check["execution_ready"]
    assert "closing_direction_unsupported" in {i["code"] for i in check["issues"]}


@pytest.mark.parametrize("version", [0, 1])
def test_frozen_hold_projection_stays_viewable_and_revisable(
    source_image, tmp_path, monkeypatch, version
):
    from embodichain.gen_sim.task_spec import TaskSpec, canonical_hash
    from embodichain.gen_sim.task_spec.publication import freeze_task_spec
    from embodichain.gen_sim.task_engine import _image_run
    from embodichain.gen_sim.task_engine._review import write_review
    from embodichain.gen_sim.workflow import load_task_plan, revise_task_plan

    frozen = json.loads(
        (Path(__file__).with_name("data") / "hold_projection_history.json").read_text()
    )
    result = _plan(source_image, "E2", _terminal_hold("E2"))
    assert result["task_candidates"]["candidates"][0] == frozen["candidate"]
    spec = TaskSpec.from_dict(frozen["projections"][version]["task_spec"])
    old = result["task_spec_v1"]
    result["task_spec"] = spec.to_dict()
    result["task_spec_v1"] = freeze_task_spec(
        spec,
        image_ref=old["inputs"]["image"]["ref"],
        image_sha256=old["inputs"]["image"]["sha256"],
        role_descriptions=old["configuration"]["descriptions"],
        interpretation=old["interpretation"]["references"],
        generalization=[],
        interpreted_steps=frozen["candidate"]["draft"]["steps"],
    )
    result["semantic_graph_validation"]["task_spec_digest"] = canonical_hash(
        spec.to_dict()
    )
    monkeypatch.setattr(
        _image_run,
        "generate_semantic_task_graph_from_image",
        lambda *a, **kw: deepcopy(result),
    )
    parent = _image_run.plan_image_task(
        source_image, "执行测试任务", tmp_path / "old", task_id="case"
    )
    plan = load_task_plan(parent)
    write_review(parent, tmp_path / "old.html")
    step, grounding, proposal = _fixture("E2")
    _terminal_hold("E2")(step, grounding, proposal)
    revised = revise_task_plan(
        parent,
        proposal,
        tmp_path / "revised",
        expected_spec_digest=plan["spec_digest"],
        expected_graph_digest=plan["semantic_graph_sha256"],
        expected_plan_digest=plan["plan_digest"],
    )
    assert load_task_plan(revised)["spec_digest"] == plan["spec_digest"]


def test_grounded_alias_release_retires_previous_grasp(source_image):
    first, grounding, plan = _fixture("E2")
    _terminal_hold("E2")(first, grounding, plan)
    first["object"] = _selector("cube")
    second, extra, placement = _fixture("E1")
    second.update(id="step_02", object=_selector("red cube"), depends_on=["step_01"])
    grounding["bindings"].update(
        {"step_02.object": ["source"], "step_02.target": ["tray"]}
    )
    node = placement["nodes"][-1]
    node.update(id="n2", step_id="step_02", depends_on=["n1"])
    plan["nodes"].append(node)
    plan["requirements"][0]["node_ids"].append("n2")
    result = generate_semantic_task_graph_from_image(
        source_image,
        "执行测试任务",
        task_id="case",
        instruction_caller=lambda **kw: dict(steps=[first, second]),
        image_caller=lambda **kw: json.dumps(grounding),
        planning_caller=lambda **kw: json.dumps(plan),
    )
    from embodichain.gen_sim.task_spec import TaskSpec, evaluate_task_spec

    spec = TaskSpec.from_dict(result["task_spec"])
    assert all(p.name != "held" for p in spec.goal)
    assert any(p.name == "held" for p in spec.milestones[0].achieve)
    observations = {p.id: True for p in spec.goal}
    observations["trajectory"] = [
        {p.id: True for p in m.achieve} for m in spec.milestones
    ]
    assert evaluate_task_spec(spec, observations).status.value == "satisfied"


@pytest.mark.parametrize("repair", [False, True])
def test_grounding_retries_keep_original_goal_meaning(source_image, repair):
    text = (
        "Put the cube on the table, then lift the red cube upright and keep holding it."
    )
    first, grounding, plan = _fixture("E1")
    first.update(object=_selector("cube"), relation="on")
    for node in plan["nodes"]:
        node["semantic"]["relation"] = "on"
        node["semantic"]["intent"]["relation"] = "on"
        if node["call"]["kind"] == "place":
            node["call"].pop("inside")
            node["call"]["on"] = "on__tray__source"
    second, _, hold = _fixture("E2")
    _terminal_hold("E2")(second, {}, hold)
    second.update(id="step_02", object=_selector("red cube"), depends_on=["step_01"])
    grounding["bindings"]["step_02.object"] = ["source"]
    for index, node in enumerate(hold["nodes"], 2):
        node.update(id=f"n{index}", step_id="step_02", depends_on=[f"n{index-1}"])
        plan["nodes"].append(node)
    plan["requirements"][0].update(quote=text, node_ids=["n0", "n1", "n2", "n3"])
    grounding["clauses"] = [
        dict(
            id="source_placement",
            quote="on the table",
            target="goal_001",
            origin="explicit",
            resolution="resolved",
        )
    ]
    requests = []

    def respond(**kwargs):
        requests.append(json.loads(kwargs["user_prompt"]))
        response = deepcopy(grounding)
        if repair and len(requests) == 2:
            response["clauses"][0].update(quote="upright", target="goal_002")
        return json.dumps(response)

    def generate():
        return generate_semantic_task_graph_from_image(
            source_image,
            text,
            task_id="case",
            instruction_caller=lambda **kw: dict(steps=[first, second]),
            image_caller=respond,
            planning_caller=lambda **kw: json.dumps(plan),
        )

    if repair:
        result = generate()
        clause = result["task_spec_v1"]["interpretation"]["clauses"][0]
        assert clause["quote"] == "upright"
        goal = next(
            p
            for p in result["task_spec_v1"]["objectives"]["goal"]
            if p["id"] == clause["target"]
        )
        assert goal["predicate"] == "object_upright"
    else:
        with pytest.raises(SemanticPlanningError, match="superseded") as caught:
            generate()
        original = caught.value.artifacts["task_spec"]
        placement = next(p for p in original["goal"] if p["id"] == "goal_001")
        assert placement["predicate"] == "semantic_goal"
        assert placement["relation"] == "on"
    assert len(requests) == 2
    assert requests[0]["objectives"] == requests[1]["objectives"]
