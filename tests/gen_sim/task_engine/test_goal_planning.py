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

import pytest
import json
from copy import deepcopy
from pathlib import Path

from embodichain.gen_sim.task_engine.goal_planning import author_goal_plan
from embodichain.gen_sim.task_engine.goal_planning import publish
from embodichain.gen_sim.task_spec.publication import decode_task_document


@pytest.mark.parametrize(
    "text,proposed,reviewed",
    [
        (
            "Tilt the cup toward the bowl without transferring its contents.",
            dict(template="E3", reference="bowl", mode="fluid"),
            dict(template="E3", reference="bowl", mode="motion"),
        ),
        (
            "Lower the cup into the recess in the holder.",
            dict(template="E1", reference="holder", relation="on"),
            dict(template="E1", reference="holder", relation="inside"),
        ),
        (
            "Mimic pouring into the bowl, then leave the cup in the gripper.",
            dict(template="E3", reference="bowl", mode="motion"),
            dict(
                template="E3",
                reference="bowl",
                mode="motion",
                terminal_behavior="hold",
                terminal_quote="leave the cup in the gripper",
            ),
        ),
    ],
)
def test_structured_source_audit_rejects_changed_meaning(text, proposed, reviewed):
    from embodichain.gen_sim.task_engine.hybrid import (
        _operation_rows,
        _validate_source_audit,
    )

    operations = _operation_rows([dict(object="cup", quote=text, **proposed)], text)
    audit = {
        "issues": [],
        "source_operations": [dict(object="cup", quote=text, **reviewed)],
    }
    with pytest.raises(ValueError, match="source_audit.*operation 1"):
        _validate_source_audit(text, operations, audit)


def test_source_audit_requires_an_independent_operation_for_each_input():
    from embodichain.gen_sim.task_engine.hybrid import (
        _operation_rows,
        _validate_source_audit,
    )

    instruction = "Put the cup inside the box."
    operations = _operation_rows(
        [
            dict(
                template="E1",
                object="cup",
                reference="box",
                relation="inside",
                quote=instruction,
            )
        ],
        instruction,
    )
    with pytest.raises(ValueError, match="source_operations"):
        _validate_source_audit(instruction, operations, {"issues": []})


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"mode": "solid"}, "Unsupported or ambiguous pour mode"),
        ({"terminal_behavior": "hold"}, "Missing verbatim terminal behavior source"),
        (
            {"terminal_behavior": "hold", "terminal_quote": "not in the instruction"},
            "Missing verbatim terminal behavior source",
        ),
    ],
)
def test_operation_rows_rejects_unsupported_or_unquoted_intent(fields, error):
    from embodichain.gen_sim.task_engine.hybrid import _operation_rows

    instruction = "Mimic pouring from the cup toward the bowl."
    operation = dict(
        template="E3",
        object="cup",
        reference="bowl",
        mode="motion",
        quote=instruction,
    )
    with pytest.raises(ValueError, match=error):
        _operation_rows([{**operation, **fields}], instruction)


def test_failed_source_audit_prevents_publication(tmp_path, monkeypatch):
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid
    from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg
    from embodichain.gen_sim.services import compat
    from PIL import Image

    image = tmp_path / "source.png"
    Image.new("RGB", (12, 12)).save(image)
    instruction = "把杯子放在垫上，再按按钮"
    operations = [
        dict(
            template="E1",
            object="cup",
            reference="mat",
            relation="on",
            quote="把杯子放在垫上",
        )
    ]
    seen = []

    def interpret(self, **kwargs):
        seen.append(json.loads(kwargs["user_prompt"]))
        return json.dumps({"operations": operations})

    monkeypatch.setattr(compat.VisionFacade, "complete", interpret)
    monkeypatch.setattr(
        compat,
        "structured_call",
        lambda **kw: {"issues": [{"quote": "再按按钮", "reason": "missing press"}]},
    )
    with (
        GenSimServices(GenSimServicesCfg()) as services,
        pytest.raises(ValueError, match="missing press"),
    ):
        publish_hybrid(instruction, "test", image, tmp_path / "plan", services)
    assert len(seen) == 2 and "missing press" in seen[1]["diagnostic"]
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize(
    "revision,reviewed",
    [
        (dict(kind="place", object="cup", on="bowl", resources={}), True),
        (
            dict(
                kind="registered",
                call_id="simulation.press",
                arguments={"object": "cup"},
                resources={},
            ),
            False,
        ),
    ],
)
def test_hybrid_validates_a_repair_before_selecting_it(
    tmp_path, monkeypatch, revision, reviewed
):
    from PIL import Image
    from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg, compat
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid

    text = "Empty the water from the cup into the bowl."
    source = [
        dict(template="E3", object="cup", reference="bowl", mode="fluid", quote=text)
    ]
    audited = []
    monkeypatch.setattr(
        compat.VisionFacade,
        "complete",
        lambda *a, **kw: json.dumps({"operations": source}),
    )

    def respond(**kwargs):
        prompt = json.loads(kwargs["prompt"])
        if kwargs["role"] == "task_semantic_planning":
            return {"calls": [*prompt["previous"], revision]}
        audited.append(prompt["ordered_calls"])
        if revision in audited[-1]:
            return {"issues": [dict(quote=text, reason="unsupported action")]}
        return {"issues": [], "source_operations": source}

    monkeypatch.setattr(compat, "structured_call", respond)
    image = tmp_path / "scene.png"
    Image.new("RGB", (20, 20)).save(image)
    with GenSimServices(GenSimServicesCfg()) as services:
        result = publish_hybrid(text, "pour", image, tmp_path / "plan", services)
    assert result["coverage"]["status"] == "uncovered"
    rejection = result["planning_history"][-1]["coverage"]
    assert rejection["status"] == "source_rejected"
    assert (revision in audited[-1]) == reviewed
    assert len(audited) == (2 if reviewed else 1)
    if not reviewed:
        assert "requires articulation" in rejection["error"]
    assert revision not in [
        node["call"] for node in result["semantic_task_graph"]["nodes"]
    ]


@pytest.mark.parametrize(
    "family,params,expected",
    [
        ("E2", {"orientation": "upright"}, "covered"),
        (
            "E3",
            {
                "reference": "cup",
                "mode": "motion",
                "terminal_behavior": "return_and_release",
                "terminal_quote": "request",
            },
            "covered",
        ),
        ("E3", {"reference": "cup", "mode": "fluid"}, "uncovered"),
        ("E6", {"state": "closed"}, "covered"),
        ("E7", {"state": "open"}, "covered"),
        ("E8", {"mode": "setting", "setting": 3}, "covered"),
        ("E9", {}, "covered"),
    ],
)
def test_hybrid_family_coverage_requires_matching_bound_operation(
    family, params, expected
):
    from embodichain.gen_sim.task_engine.hybrid import (
        _operation_rows,
        _supported_program,
        verify_witness,
    )
    from embodichain.gen_sim.task_spec import TaskSpec

    bindings = {}
    objective, _, calls = _supported_program(
        _operation_rows(
            [{"template": family, "object": "part", "quote": "request", **params}],
            "request",
        ),
        bindings=bindings,
    )
    result = author_goal_plan("request", "case", objective, calls)
    graph = result["semantic_task_graph"]
    spec = TaskSpec.from_dict(result["task_spec"])
    assert (
        verify_witness(spec, graph, support_targets={}, bindings=bindings)["status"]
        == expected
    )
    if family in {"E2", "E3", "E6", "E8"}:
        assert (
            verify_witness(spec, graph, support_targets={}, bindings={})["status"]
            == "uncovered"
        )


def test_hybrid_transfer_and_transport_reject_wrong_witness():
    from embodichain.gen_sim.task_engine.hybrid import (
        _operation_rows,
        _supported_program,
        verify_witness,
    )
    from embodichain.gen_sim.task_spec import TaskSpec

    instruction = "交接苹果，放盘，搬盘到垫子"
    ops = [
        dict(
            template="E4",
            object="apple",
            source="right",
            destination="left",
            quote="交接苹果",
        ),
        dict(
            template="E1",
            object="apple",
            primary="left",
            reference="plate",
            relation="inside",
            quote="放盘",
        ),
        dict(
            template="E5",
            object="plate",
            reference="mat",
            relation="on",
            quote="搬盘到垫子",
        ),
    ]
    objective, _, calls = _supported_program(_operation_rows(ops, instruction))
    result = author_goal_plan(instruction, "pilot", objective, calls)
    spec = TaskSpec.from_dict(result["task_spec"])
    graph = result["semantic_task_graph"]

    def check(g):
        return verify_witness(spec, g, support_targets={"mat": "mat"})

    valid = check(graph)
    assert valid["status"] == "covered"
    assert valid["producers"]["m3"] == ("call_4", "acquire")
    assert valid["producers"]["m4"] == ("call_4", "release")
    wrong = deepcopy(graph)
    wrong["nodes"][2]["call"]["inside"] = "mat"
    assert "final_apple" in check(wrong)["uncovered"]
    wrong = deepcopy(graph)
    wrong["nodes"][1]["call"]["resources"] = {"source": "left", "destination": "right"}
    assert check(wrong)["resource_conflicts"][0]["node_id"] == "call_2"


def test_repair_fixes_resource_conflict_with_frozen_quoted_goal(monkeypatch, tmp_path):
    from embodichain.gen_sim.services import compat

    instruction = "右臂将苹果递给左臂"
    objective = dict(
        roles=[{"id": "apple"}],
        goal=[
            dict(
                id="transfer",
                predicate="handover_complete",
                subject="apple",
                arguments=dict(transfer_arm="right", receive_arm="left"),
            )
        ],
        sources={"transfer": instruction},
    )
    give = dict(
        kind="hand_over",
        object="apple",
        resources=dict(source="right", destination="left"),
    )
    calls = []

    def respond(*, role, prompt, **kwargs):
        if role == "task_interpretation":
            return objective
        request = json.loads(prompt)
        calls.append(request)
        if len(calls) == 1:
            return {"calls": [give]}
        assert request["frozen_spec"] == calls[0]["frozen_spec"]
        assert request["repair_feedback"]["resource_conflicts"][0]["held_before"] == {}
        return {
            "calls": [
                dict(kind="pick", object="apple", resources={"primary": "right"}),
                give,
            ]
        }

    monkeypatch.setattr(compat, "structured_call", respond)
    result = publish(instruction, "handover", tmp_path / "result", None, None)
    assert result["coverage"]["status"] == "covered"
    assert result["planning_history"]["selected_round"] == 1
    assert (
        result["task_spec_v1"]["interpretation"]["clauses"][0]["quote"] == instruction
    )
    objective["sources"]["transfer"] = "左臂递给右臂"
    with pytest.raises(ValueError, match="verbatim"):
        publish(
            instruction,
            "handover",
            tmp_path / "invalid",
            {"objective": objective, "calls": [give]},
            None,
        )


@pytest.mark.parametrize(
    "target,state,status",
    [
        ("door", "closed", "covered"),
        ("door", "open", "uncovered"),
        ("other", "closed", "uncovered"),
    ],
)
def test_hinge_endpoint_requires_exact_producer(target, state, status):
    objective = dict(
        roles=[
            {"id": "door", "kind": "articulation"},
            {"id": "other", "kind": "articulation"},
        ],
        goal=[
            dict(
                id="endpoint",
                predicate="articulation_joint_near",
                subject="door",
                arguments={"joint_kind": "revolute", "target_state": "closed"},
            )
        ],
    )
    call = dict(
        kind="registered",
        call_id="simulation.open_door",
        arguments={"object": target, "state": state},
        resources={},
    )
    plan = author_goal_plan("关上铰链门", "close", objective, [call])
    assert plan["coverage"]["status"] == status
    assert not plan["deployment_preflight"]["execution_ready"]
    if state == "closed":
        assert (
            plan["deployment_preflight"]["issues"][0]["code"]
            == "closing_direction_unsupported"
        )
    assert (
        decode_task_document(plan["task_spec_v1"])["inputs"]["instruction"]
        == "关上铰链门"
    )


def test_handover_requires_preceding_acquisition_and_correct_direction():
    objective = dict(
        roles=[{"id": "apple"}],
        goal=[
            dict(
                id="transfer",
                predicate="handover_complete",
                subject="apple",
                arguments={"transfer_arm": "right", "receive_arm": "left"},
            )
        ],
    )
    give = dict(
        kind="hand_over",
        object="apple",
        resources={"source": "right", "destination": "left"},
    )
    pick = dict(kind="pick", object="apple", resources={"primary": "right"})
    assert author_goal_plan("hand over", "test", objective, [give])["coverage"][
        "resource_conflicts"
    ]
    assert (
        author_goal_plan("hand over", "test", objective, [pick, give])["coverage"][
            "status"
        ]
        == "covered"
    )
    give["resources"] = {"source": "left", "destination": "right"}
    assert (
        author_goal_plan("hand over", "test", objective, [pick, give])["coverage"][
            "status"
        ]
        == "uncovered"
    )


def test_hybrid_compound_drawer_clause_accepts_bare_array():
    from embodichain.gen_sim.task_engine.hybrid import _operation_rows

    ops = [
        dict(
            template="E1",
            object="cube",
            quote="把魔方放进抽屉",
            reference="drawer",
            relation="inside",
        )
    ]
    text = "把魔方放进抽屉后把抽屉关好。"
    assert _operation_rows(ops, text)[0]["reference"] == "drawer"
    with pytest.raises(ValueError, match=r"operations\[0\]"):
        _operation_rows({"operations": [ops]}, text)


def test_hybrid_held_vessel_pours_without_reacquisition():
    from embodichain.gen_sim.task_engine.hybrid import (
        _operation_rows,
        _supported_program,
        verify_witness,
    )
    from embodichain.gen_sim.task_spec import TaskSpec

    text = "扶正瓶子，演示倾倒动作，再放回原位。"
    ops = [
        dict(
            template="E2",
            object="bottle",
            primary="left",
            quote="扶正瓶子",
            orientation="upright",
        ),
        dict(
            template="E3",
            object="bottle",
            reference="cup",
            mode="motion",
            terminal_behavior="return_and_release",
            terminal_quote="再放回原位",
            quote="演示倾倒动作，再放回原位",
        ),
    ]
    bindings = {}
    operations = _operation_rows(ops, text)
    obj, _, calls = _supported_program(operations, bindings=bindings)
    assert all(c["resources"] == {"primary": "left"} for c in calls)
    assert not any(c["kind"] == "pick" for c in calls)
    plan = author_goal_plan(text, "pour", obj, calls)
    assert (
        verify_witness(
            TaskSpec.from_dict(plan["task_spec"]),
            plan["semantic_task_graph"],
            support_targets={},
            bindings=bindings,
        )["status"]
        == "covered"
    )
    operations[-1]["primary"] = "right"
    with pytest.raises(ValueError, match="occupied"):
        _supported_program(operations)


@pytest.mark.parametrize(
    "family,terminal,clause,terminal_quote,expected_calls",
    [
        (
            "E3",
            "hold",
            "Tilt the cup toward the bowl without emptying it, then keep the cup in your grip.",
            "keep the cup in your grip",
            ("pick", "simulation.move_held_object", "simulation.pour"),
        ),
        (
            "E3",
            "return_and_release",
            "Tilt the cup toward the bowl without emptying it, then put the cup back where it began.",
            "put the cup back where it began",
            (
                "pick",
                "simulation.move_held_object",
                "simulation.pour",
                "simulation.move_held_object",
                "place",
            ),
        ),
        (
            "E3",
            "none",
            "Tilt the cup toward the bowl without emptying it, then bring it upright again.",
            None,
            ("pick", "simulation.move_held_object", "simulation.pour"),
        ),
        (
            "E2",
            "hold",
            "Stand the cup upright and keep holding it.",
            "keep holding it",
            ("simulation.axis_align",),
        ),
        (
            "E2",
            "return_and_release",
            "Stand the cup upright and put it back where it began.",
            "put it back where it began",
            ("simulation.axis_align", "simulation.move_held_object", "place"),
        ),
        (
            "E4",
            "hold",
            "Hand the cup from the right to the left arm and keep it in the left gripper.",
            "keep it in the left gripper",
            ("pick", "hand_over"),
        ),
        (
            "E4",
            "none",
            "Hand the cup from the right to the left arm.",
            None,
            ("pick", "hand_over"),
        ),
    ],
    ids=[
        "pour-hold",
        "pour-return",
        "pour-upright-only",
        "align-hold",
        "align-return",
        "handover-hold",
        "handover-unspecified",
    ],
)
def test_hybrid_terminal_follows_instruction(
    tmp_path, monkeypatch, family, terminal, clause, terminal_quote, expected_calls
):
    from PIL import Image
    from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg, compat
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid, verify_witness
    from embodichain.gen_sim.task_spec import TaskSpec
    from embodichain.gen_sim.workflow import load_task_plan

    fields = {
        "E2": {"orientation": "upright", "primary": "right"},
        "E3": {"reference": "bowl", "mode": "motion", "primary": "right"},
        "E4": {"source": "right", "destination": "left"},
    }[family]
    operation = dict(template=family, object="cup", quote=clause, **fields)
    operation["terminal_behavior"] = terminal
    if terminal_quote is not None:
        operation["terminal_quote"] = terminal_quote
    monkeypatch.setattr(
        compat.VisionFacade,
        "complete",
        lambda *a, **kw: json.dumps({"operations": [operation]}),
    )

    def review(**kwargs):
        assert kwargs["role"] == "task_interpretation"
        return {"issues": [], "source_operations": [operation]}

    monkeypatch.setattr(compat, "structured_call", review)
    image = tmp_path / "scene.png"
    Image.new("RGB", (20, 20)).save(image)
    output = tmp_path / "plan"
    with GenSimServices(GenSimServicesCfg()) as services:
        result = publish_hybrid(clause, "terminal", image, output, services)

    calls = [node["call"] for node in result["semantic_task_graph"]["nodes"]]
    assert tuple(call.get("call_id", call["kind"]) for call in calls) == expected_calls
    assert result["coverage"]["status"] == "covered"
    document = load_task_plan(output)
    assert document["stage"] == "authoring_only"
    objectives = document["task_document"]["objectives"]
    holds = [goal for goal in objectives["goal"] if goal["predicate"] == "held"]
    if terminal == "hold":
        assert len(holds) == 1
        assert holds[0]["subject"] == "cup"
        assert holds[0]["arguments"]["required_arm"] == (
            "left" if family == "E4" else "right"
        )
    else:
        assert not holds
    if terminal == "return_and_release":
        assert calls[-2]["arguments"]["target"] == "original_support_cup"
        assert calls[-1]["on"] == "original_support_cup"
        assert any(
            p.get("arguments", {}).get("operation") == "return_and_release"
            for m in objectives["milestones"]
            for p in m["achieve"]
        )
    elif family == "E3" and terminal == "hold":
        mutated = deepcopy(result["semantic_task_graph"])
        last = mutated["nodes"][-1]
        mutated["nodes"].append(
            {
                **last,
                "id": "extra_place",
                "depends_on": [last["id"]],
                "call": dict(
                    kind="place",
                    object="cup",
                    on="bowl",
                    resources={"primary": "right"},
                ),
            }
        )
        mutated["task_groups"][0]["node_ids"].append("extra_place")
        verdict = verify_witness(
            TaskSpec.from_dict(result["task_spec"]),
            mutated,
            support_targets={},
            bindings=result["pending_operation_bindings"],
        )
        assert "final_cup_held" in verdict["uncovered"]


@pytest.mark.parametrize(
    "instruction",
    [
        "把杯子放在托盘上，不要倒水。",
        "Place the pouring jug on the table.",
    ],
)
def test_nonaffirmative_pour_reaches_publication_and_review(
    tmp_path, monkeypatch, instruction
):
    from PIL import Image
    from embodichain.gen_sim.services import compat
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid
    from embodichain.gen_sim.task_engine._review import write_review
    from embodichain.gen_sim.workflow import load_task_plan
    from embodichain.gen_sim.task_spec import canonical_hash

    image = tmp_path / "scene.png"
    Image.new("RGB", (20, 20)).save(image)
    ops = [
        dict(
            template="E1",
            object="jug" if instruction.startswith("Place") else "cup",
            reference="table" if instruction.startswith("Place") else "tray",
            relation="on",
            quote=instruction,
        )
    ]
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        compat.VisionFacade,
        "complete",
        lambda *a, **kw: json.dumps(dict(operations=ops)),
    )

    def respond(**kw):
        if kw["role"] == "task_interpretation":
            return {"issues": [], "source_operations": ops}
        return {
            r: dict(description=r, bbox=[10, 10, 200, 200])
            for r in kw["schema"]["required"]
        }

    monkeypatch.setattr(compat, "structured_call", respond)
    out = tmp_path / "plan"
    publish_hybrid(instruction, "example", Path("~/scene.png"), Path("~/plan"))
    plan = load_task_plan(out)
    assert plan["schema_version"] == "embodichain.hybrid-task-plan/v1"
    if ops[0]["object"] == "jug":
        return  # The native evidence fixture does not depend on the source wording.
    native = tmp_path / "attempt"
    native.mkdir()
    identities = {
        k: "measured"
        for k in (
            "bundle_digest",
            "scene_digest",
            "program_digest",
            "policy_sha256",
            "run_id",
            "attempt_id",
        )
    }
    identities.update(
        spec_digest=plan["spec_digest"],
        graph_digest=plan["semantic_graph_sha256"],
        plan_digest=plan["plan_digest"],
    )
    report = dict(
        status="succeeded",
        segments=[
            dict(
                name="call_1",
                success=True,
                start_step=0,
                end_step=12,
                active=[True],
                successes=[True],
                start_steps=[0],
                end_steps=[12],
            )
        ],
    )
    for file, data in {
        "run_manifest": identities,
        "attempt_feedback": dict(
            identities,
            schema_version="embodichain.attempt-feedback/v1",
            execution="succeeded",
            physical_verdict="not_evaluated",
            execution_report_digest=canonical_hash(report),
        ),
        "execution_report": report,
    }.items():
        (native / (file + ".json")).write_text(json.dumps(data))
    html = tmp_path / "review.html"
    write_review(out, html, native)
    import re

    data = json.loads(
        re.search(
            r'<script[^>]*id="graph-data"[^>]*>(.*?)</script>', html.read_text(), re.S
        ).group(1)
    )
    assert data["review"]["attempt"]["calls"]["call_1"]["end_step"] == 12
    assert data["review"]["source_image"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("family", ["E1", "E5"])
def test_hybrid_two_mats_project_each_occurrence(tmp_path, monkeypatch, family):
    from PIL import Image
    from embodichain.gen_sim.services import compat
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid
    from embodichain.gen_sim.workflow import load_task_plan

    instruction = "Using both arms, place the cube on the first mat, then place the cube on the second mat."
    image = tmp_path / "scene.png"
    Image.new("RGB", (20, 20)).save(image)
    ops = [
        dict(
            template=family,
            object="cube",
            reference=r,
            relation="on",
            quote=instruction,
        )
        for r in ("first_mat", "second_mat")
    ]
    monkeypatch.setattr(
        compat.VisionFacade,
        "complete",
        lambda *a, **kw: json.dumps(dict(operations=ops)),
    )
    monkeypatch.setattr(
        compat,
        "structured_call",
        lambda **kw: (
            {"issues": [], "source_operations": ops}
            if kw["role"] == "task_interpretation"
            else {
                r: dict(description=r, bbox=[10, 10, 200, 200])
                for r in kw["schema"]["required"]
            }
        ),
    )
    result = publish_hybrid(instruction, "example", image, tmp_path / "plan")
    load_task_plan(tmp_path / "plan")
    annotations = result["semantic_graph_validation"]["source"]["annotations"]
    assert (
        annotations["call_2" if family == "E1" else "call_1"]["reference"]
        == "first_mat"
    )
    assert (
        annotations["call_4" if family == "E1" else "call_2"]["reference"]
        == "second_mat"
    )


@pytest.mark.parametrize(
    "tray,description,box",
    [
        ("labeled_tray", "blue tray with a white label", [50, 400, 400, 800]),
        ("plain_tray", "blue tray without a label", [600, 400, 950, 800]),
        ("tray", None, [50, 400, 400, 800]),
    ],
)
def test_hybrid_scene_context_selects_and_preserves_destination(
    tmp_path, monkeypatch, tray, description, box
):
    from PIL import Image
    from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg, compat
    from embodichain.gen_sim.task_engine.cli import main
    from embodichain.gen_sim.workflow import load_task_plan

    instruction = "Put the cup inside the tray."
    image = tmp_path / "scene.png"
    Image.new("RGB", (20, 20)).save(image)
    output = tmp_path / "plan"
    argv = [
        "--image",
        str(image),
        "--instruction",
        instruction,
        "--output",
        str(output),
    ]
    context = None
    if description:
        context = {
            "schema_version": "embodichain.task-scene-context/v1",
            "description": "There are two blue trays in the scene.",
            "objects": [
                dict(id=tray, description=description, asset_kind="rigid"),
                dict(id="stapler", description="a stapler in the background"),
            ],
        }
        source = tmp_path / "context.json"
        source.write_text(json.dumps(context), encoding="utf-8")
        argv.extend(["--scene-context", str(source)])

    def interpret(self, **kwargs):
        prompt = json.loads(kwargs["user_prompt"])
        assert prompt["instruction"] == instruction
        assert prompt.get("scene_context") == context
        target = prompt["scene_context"]["objects"][0]["id"] if context else "tray"
        return json.dumps(
            dict(
                operations=[
                    dict(
                        template="E1",
                        object="cup",
                        reference=target,
                        relation="inside",
                        quote=instruction,
                    )
                ]
            )
        )

    def respond(**kwargs):
        prompt = json.loads(kwargs["prompt"])
        assert prompt["instruction"] == instruction
        if kwargs["role"] == "task_interpretation":
            assert prompt.get("scene_context") == context
            return {
                "issues": [],
                "source_operations": [
                    dict(
                        template="E1",
                        object="cup",
                        reference=tray,
                        relation="inside",
                        quote=instruction,
                    )
                ],
            }
        assert prompt.get("scene_context") == context
        return {
            role: dict(description=f"observed {role}", bbox=coords)
            for role, coords in {"cup": [450, 100, 550, 250], tray: box}.items()
        }

    monkeypatch.setattr(compat.VisionFacade, "complete", interpret)
    monkeypatch.setattr(compat, "structured_call", respond)
    with GenSimServices(GenSimServicesCfg()) as services:
        main(argv, services=services)

    graph = json.loads((output / "semantic_task_graph.json").read_text())
    assert [node["call"] for node in graph["nodes"]] == [
        dict(kind="pick", object="cup", resources={}),
        dict(kind="place", object="cup", inside=tray, resources={}),
    ]
    assert graph["nodes"][1]["depends_on"] == [graph["nodes"][0]["id"]]
    grounding = json.loads((output / "image_grounding.json").read_text())
    assert grounding["bindings"][tray] == [tray]
    assert {item["id"]: item["bbox"] for item in grounding["objects"]}[tray] == [
        coord / 1000 for coord in box
    ]
    plan = load_task_plan(output)
    assert plan["task_document"]["configuration"]["descriptions"][tray] == (
        description or tray
    )
    configuration = plan["scene_requirements"]["configuration"]
    assert {role["id"] for role in configuration["roles"]} == {"cup", tray}
    objects = next(
        entry["value"]
        for entry in configuration["requirements"]
        if (entry["consumer"], entry["key"]) == ("scene", "scene_objects")
    )
    assert {row["id"]: row["is_articulated"] for row in objects} == {
        "cup": False,
        tray: False,
        **({"stapler": False} if context else {}),
    }
    snapshot = output / "scene_context.json"
    if context is None:
        assert not snapshot.exists()
    else:
        assert snapshot.name in plan["artifacts"]
        assert json.loads(snapshot.read_text()) == context


@pytest.mark.parametrize(
    "object_id,asset_kind,error",
    [
        ("drawer", "rigid", "drawer.*rigid.*articulation"),
        (None, None, "drawer.*requires articulation.*scene-context"),
        ("top_drawer", "articulated", "drawer.*requires articulation.*scene-context"),
        ("drawer", "container", "asset_kind"),
        ("cabinet", "articulated", None),
    ],
)
def test_hybrid_scene_context_gates_native_joint_actions(
    tmp_path, monkeypatch, object_id, asset_kind, error
):
    from PIL import Image
    from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg, compat
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid

    instruction = "Open the drawer."
    image = tmp_path / "scene.png"
    Image.new("RGB", (20, 20)).save(image)
    source = None
    if object_id is not None:
        source = tmp_path / "context.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": "embodichain.task-scene-context/v1",
                    "objects": [
                        dict(id=object_id, description=object_id, asset_kind=asset_kind)
                    ],
                }
            ),
            encoding="utf-8",
        )
    op = dict(template="E6", object="drawer", state="open", quote=instruction)
    if object_id == "cabinet":
        op["part_of"] = "cabinet"
    interpretations = []

    def interpret(self, **kwargs):
        assert asset_kind != "container", "Invalid context must fail before model calls"
        interpretations.append(kwargs)
        return json.dumps([op])

    def audit(**kwargs):
        assert error is None, "Conflicting declarations must fail before audit"
        return {"issues": [], "source_operations": [op]}

    monkeypatch.setattr(compat.VisionFacade, "complete", interpret)
    monkeypatch.setattr(compat, "structured_call", audit)
    output = tmp_path / "plan"
    with GenSimServices(GenSimServicesCfg()) as services:
        if error:
            with pytest.raises(ValueError, match=error):
                publish_hybrid(
                    instruction,
                    "example",
                    image,
                    output,
                    services,
                    scene_context=source,
                )
            assert len(interpretations) <= 1
            assert not output.exists()
        else:
            result = publish_hybrid(
                instruction, "example", image, output, services, scene_context=source
            )
            roles = result["task_spec"]["roles"]
            assert [role["id"] for role in roles] == ["drawer"]
            assert roles[0]["properties"]["part_of"] == "cabinet"
            configuration = result["task_spec_v1"]["configuration"]
            assert {
                "consumer": "scene",
                "key": "scene_objects",
                "value": [
                    dict(
                        id="cabinet",
                        description="cabinet",
                        asset_kind="articulated",
                        is_articulated=True,
                    )
                ],
            } in configuration["requirements"]
