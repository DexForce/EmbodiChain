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
    "template,fields,wrong",
    [
        ("E1", dict(reference="holder", relation="inside"), {"relation": "on"}),
        ("E3", dict(reference="bowl", mode="motion"), {"mode": "fluid"}),
        ("E2", dict(orientation="upright"), {"terminal_behavior": "hold"}),
    ],
)
def test_structured_audit_rechecks_the_source(template, fields, wrong):
    from embodichain.gen_sim.task_engine.hybrid import (
        _operation_rows,
        _validate_source_audit,
    )

    text = "Follow the source instruction."
    source = dict(template=template, object="cup", quote=text, **fields)
    proposed = _operation_rows(
        (
            [dict(source, terminal_quote=text, **wrong)]
            if "terminal_behavior" in wrong
            else [dict(source, **wrong)]
        ),
        text,
    )
    with pytest.raises(ValueError, match="source_audit.*operation 1"):
        _validate_source_audit(
            text, proposed, dict(issues=[], source_operations=[source])
        )
    with pytest.raises(ValueError, match="source_operations"):
        _validate_source_audit(text, proposed, dict(issues=[]))


@pytest.mark.parametrize(
    "wrong",
    [{"mode": "solid"}, {"terminal_behavior": "hold", "terminal_quote": "Missing"}],
)
def test_operation_rows_reject_invalid_mode_or_source(wrong):
    from embodichain.gen_sim.task_engine.hybrid import _operation_rows

    text = "Mimic pouring from the cup toward the bowl."
    op = dict(template="E3", object="cup", reference="bowl", mode="motion", quote=text)
    with pytest.raises(ValueError, match="mode|verbatim"):
        _operation_rows([op | wrong], text)


@pytest.fixture
def native_hybrid(tmp_path, monkeypatch):
    from PIL import Image
    from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg, compat
    from embodichain.gen_sim.task_engine.hybrid import publish_hybrid

    image, output = tmp_path / "scene.png", tmp_path / "plan"
    Image.new("RGB", (20, 20)).save(image)

    def run(ops, *, context=None, review=None):
        text = ops[0]["quote"]
        source = None
        if context is not None:
            source = tmp_path / "context.json"
            source.write_text(json.dumps(context))

        def proposal(*args, **kwargs):
            assert json.loads(kwargs["user_prompt"]).get("scene_context") == context
            return json.dumps(ops)

        def audit(**kwargs):
            request = json.loads(kwargs["prompt"])
            if review is not None:
                return review(kwargs["role"], request)
            assert request.get("scene_context") == context
            if kwargs["role"] == "task_interpretation":
                return dict(issues=[], source_operations=ops)
            return {
                r: dict(description=r, bbox=[10, 10, 500, 500])
                for r in kwargs["schema"]["required"]
            }

        monkeypatch.setattr(compat.VisionFacade, "complete", proposal)
        monkeypatch.setattr(compat, "structured_call", audit)
        with GenSimServices(GenSimServicesCfg()) as services:
            return publish_hybrid(
                text, "test", image, output, services, scene_context=source
            )

    return run


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


@pytest.mark.parametrize("control", [False, True])
def test_repair_cannot_insert_an_unsupported_action(native_hybrid, control):
    text = "Empty the water from the cup into the bowl."
    op = dict(template="E3", object="cup", reference="bowl", mode="fluid", quote=text)
    extra = (
        dict(
            kind="registered",
            call_id="simulation.press",
            arguments={"object": "cup"},
            resources={},
        )
        if control
        else dict(kind="place", object="cup", on="bowl", resources={})
    )

    def review(role, request):
        if role == "task_semantic_planning":
            return {"calls": [*request["previous"], extra]}
        return (
            dict(issues=[dict(quote=text, reason="unsupported action")])
            if extra in request["ordered_calls"]
            else dict(issues=[], source_operations=[op])
        )

    result = native_hybrid([op], review=review)
    rejection = result["planning_history"][-1]["coverage"]
    assert rejection["status"] == "source_rejected"
    assert ("requires articulation" if control else "unsupported action") in rejection[
        "error"
    ]
    assert all(node["call"] != extra for node in result["semantic_task_graph"]["nodes"])


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
        ("E3", {"reference": "cup", "mode": "motion"}, "covered"),
        ("E3", {"reference": "cup", "mode": "fluid"}, "uncovered"),
        (
            "E2",
            {
                "orientation": "upright",
                "terminal_behavior": "hold",
                "terminal_quote": "request",
            },
            "covered",
        ),
        (
            "E4",
            {
                "source": "right",
                "destination": "left",
                "terminal_behavior": "hold",
                "terminal_quote": "request",
            },
            "covered",
        ),
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
    terminal = params.get("terminal_behavior")
    assert any(call["kind"] == "place" for call in calls) == (
        terminal == "return_and_release"
    )
    assert any(goal.name == "held" for goal in spec.goal) == (terminal == "hold")
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


def _context(*objects):
    return dict(
        schema_version="embodichain.task-scene-context/v1", objects=list(objects)
    )


def test_hybrid_keeps_explicit_destination_without_adding_background_roles(
    native_hybrid, tmp_path
):
    from embodichain.gen_sim.workflow import load_task_plan

    tray = "labeled_tray"

    entry = dict(id=tray, description=tray.replace("_", " "), asset_kind="rigid")
    context = _context(entry, dict(id="stapler", description="background stapler"))
    result = native_hybrid(
        [
            dict(
                template="E1",
                object="cup",
                reference=tray,
                relation="inside",
                quote="Put the cup in the tray.",
            )
        ],
        context=context,
    )
    plan = load_task_plan(tmp_path / "plan")
    config = plan["scene_requirements"]["configuration"]
    assert result["semantic_task_graph"]["nodes"][-1]["call"]["inside"] == tray
    assert result["image_grounding"]["bindings"][tray] == [tray]
    assert (
        plan["task_document"]["configuration"]["descriptions"][tray]
        == entry["description"]
    )
    assert {row["id"] for row in config["roles"]} == {"cup", tray}
    objects = next(
        row["value"] for row in config["requirements"] if row["key"] == "scene_objects"
    )
    assert {row["id"]: row["is_articulated"] for row in objects} == {
        "cup": False,
        tray: False,
        "stapler": False,
    }
    assert json.loads((tmp_path / "plan" / "scene_context.json").read_text()) == context


@pytest.mark.parametrize(
    "name,kind,error",
    [
        ("drawer", "rigid", "drawer.*rigid.*articulation"),
        (None, None, "requires articulation.*scene-context"),
        ("cabinet", "articulated", None),
    ],
)
def test_native_joint_action_requires_the_declared_owner(
    native_hybrid, tmp_path, name, kind, error
):
    context = (
        _context(dict(id=name, description=name, asset_kind=kind)) if name else None
    )
    op = dict(template="E6", object="drawer", state="open", quote="Open the drawer.")
    if name == "cabinet":
        op["part_of"] = name
    if error:
        with pytest.raises(ValueError, match=error):
            native_hybrid([op], context=context)
        assert not (tmp_path / "plan").exists()
    else:
        result = native_hybrid([op], context=context)
        assert result["task_spec"]["roles"][0]["properties"]["part_of"] == name
        declared = next(
            row["value"]
            for row in result["task_spec"]["requirements"]
            if row["key"] == "scene_objects"
        )
        assert declared == [
            dict(id=name, description=name, asset_kind=kind, is_articulated=True)
        ]
