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

from embodichain.gen_sim.task_engine.goal_planning import author_goal_plan
from embodichain.gen_sim.task_engine.goal_planning import publish
from embodichain.gen_sim.task_spec.publication import decode_task_document


@pytest.mark.parametrize(
    "text,operations,error",
    [
        (
            "完成一次倾倒动作",
            [{"template": "E3", "quote": "倾倒动作", "mode": "fluid"}],
            "pour_mode_mismatch",
        ),
        (
            "把水倒进碗里",
            [{"template": "E3", "quote": "水倒进碗里", "mode": "motion"}],
            "pour_mode_mismatch",
        ),
        (
            "朝碗倾倒一次",
            [{"template": "E3", "quote": "倾倒一次", "mode": "fluid"}],
            "ambiguous_pour",
        ),
        (
            "扶正杯子再移入托盘",
            [{"template": "E1", "relation": "inside", "quote": "移入托盘"}],
            "source_omission",
        ),
        (
            "把杯子移入托盘",
            [{"template": "E1", "relation": "on", "quote": "把杯子移入托盘"}],
            "source_omission",
        ),
        (
            "把餐碗放在垫上",
            [{"template": "E5", "relation": "on"}],
            "unsupported_addition",
        ),
    ],
)
def test_source_gate_rejects_wrong_meaning_before_freeze(text, operations, error):
    from embodichain.gen_sim.task_engine.hybrid import _check_source_intent

    with pytest.raises(ValueError, match=error):
        _check_source_intent(text, operations)


def test_negated_transfer_does_not_require_liquid_evidence():
    from embodichain.gen_sim.task_engine.hybrid import _check_source_intent

    _check_source_intent(
        "演示倒水姿势，不需要倒出液体",
        [{"template": "E3", "quote": "倒水姿势", "mode": "motion"}],
    )


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
    "family,params,expected",
    [
        ("E2", {"orientation": "upright"}, "covered"),
        (
            "E3",
            {"reference": "cup", "mode": "motion", "restore_required": True},
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
        _supported_program,
        verify_witness,
    )
    from embodichain.gen_sim.task_spec import TaskSpec

    bindings = {}
    objective, _, calls = _supported_program(
        [{"template": family, "object": "part", "quote": "request", **params}],
        "request",
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
    objective, _, calls = _supported_program(ops, instruction)
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
    from embodichain.gen_sim.task_engine.hybrid import (
        _check_source_intent,
        _operation_rows,
    )

    ops = [
        dict(
            template="E1",
            object="cube",
            quote="把魔方放进抽屉",
            reference="drawer",
            relation="inside",
        )
    ]
    _check_source_intent("把魔方放进抽屉后把抽屉关好。", _operation_rows(ops))
    with pytest.raises(ValueError, match=r"operations\[0\]"):
        _operation_rows({"operations": [ops]})


def test_hybrid_held_vessel_pours_without_reacquisition():
    from embodichain.gen_sim.task_engine.hybrid import (
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
            continue_held=True,
        ),
        dict(
            template="E3",
            object="bottle",
            reference="cup",
            mode="motion",
            restore_required=True,
            quote="演示倾倒动作，再放回原位",
        ),
    ]
    bindings = {}
    obj, _, calls = _supported_program(ops, text, bindings=bindings)
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
    ops[-1]["primary"] = "right"
    with pytest.raises(ValueError, match="occupied"):
        _supported_program(ops, text)
