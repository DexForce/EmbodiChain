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

from copy import deepcopy
import json
from pathlib import Path
import pytest
import yaml

from embodichain.gen_sim.task_engine.program_declarations import (
    ProgramSegmentBinding,
    build_program_declaration,
)


def _frozen():
    fixture = json.loads(
        (Path(__file__).parent / "data/e1_bundle_declarations.json").read_text()
    )["files"]
    return json.loads(fixture["semantic_task_graph.json"]), yaml.safe_load(
        fixture["task_program/program.yaml"]
    )


def test_explicit_acceptance_copy_preserves_frozen_program_and_inputs():
    graph, expected = _frozen()
    before = deepcopy((graph, expected))
    place, park = expected["program"]["items"][1:]
    binding = ProgramSegmentBinding(
        place["name"],
        post=tuple(place["post"]),
        validators=tuple(place["validators"]),
        terminal_post=tuple(park["post"]),
        terminal_validators=tuple(park["validators"]),
    )
    actual = build_program_declaration(
        graph, expected["program_id"], bindings=(binding,)
    )
    assert actual == expected
    assert (graph, expected) == before
    actual["program"]["items"][1]["post"][0]["preset"] = "caller-copy"
    assert binding.post[0]["preset"] == "rigid_object"


def test_explicit_deferral_moves_every_check_without_changing_calls():
    graph, expected = _frozen()
    place = expected["program"]["items"][1]
    binding = ProgramSegmentBinding(
        place["name"],
        post=tuple(place["post"]),
        validators=tuple(place["validators"]),
        terminal_post=tuple(place["post"]),
        terminal_validators=tuple(place["validators"]),
        defer_to_terminal=True,
    )
    actual = build_program_declaration(
        graph, expected["program_id"], bindings=(binding,)
    )
    assert "post" not in actual["program"]["items"][1]
    assert actual["program"]["items"][2]["post"] == place["post"]
    assert actual["program"]["items"][2]["validators"] == place["validators"]
    assert [x["steps"] for x in actual["program"]["items"]] == [
        x["steps"] for x in expected["program"]["items"]
    ]
    with pytest.raises(ValueError, match="retain every"):
        build_program_declaration(
            graph,
            expected["program_id"],
            bindings=(
                ProgramSegmentBinding(
                    place["name"], post=tuple(place["post"]), defer_to_terminal=True
                ),
            ),
        )


def test_bindings_cannot_change_occurrences_or_semantic_calls():
    graph, program = _frozen()
    node = graph["nodes"][0]
    binding = ProgramSegmentBinding(node["id"])
    for bindings in ((binding, binding), (ProgramSegmentBinding("absent"),)):
        with pytest.raises(ValueError, match="unique existing"):
            build_program_declaration(graph, program["program_id"], bindings=bindings)
    with pytest.raises(ValueError, match="exact.*projection"):
        build_program_declaration(
            graph,
            program["program_id"],
            bindings=(ProgramSegmentBinding(node["id"], call={}),),
        )


def test_explicit_empty_checks_survive_and_terminal_checks_are_not_dropped():
    graph, program = _frozen()
    last = graph["nodes"][-1]["id"]
    result = build_program_declaration(
        graph,
        program["program_id"],
        bindings=(ProgramSegmentBinding(last, validators=()),),
    )
    assert result["program"]["items"][-1]["validators"] == []
    with pytest.raises(ValueError, match="declared directly"):
        build_program_declaration(
            graph,
            program["program_id"],
            bindings=(
                ProgramSegmentBinding(
                    last,
                    terminal_post=(
                        {
                            "kind": "wait_stable",
                            "entity": "cylinder",
                            "preset": "rigid_object",
                        },
                    ),
                ),
            ),
        )


def _source_scene(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    import trimesh

    objects = []
    for uid, size in (("cylinder", [0.125, 0.125, 0.25]), ("pad", [0.5, 0.5, 0.03125])):
        mesh = trimesh.creation.box(extents=size)
        mesh.vertices = np.asarray(mesh.vertices)[:, [0, 2, 1]] * [1, 1, -1]
        path = tmp_path / f"{uid}.glb"
        mesh.export(path)
        objects.append(
            {
                "uid": uid,
                "runtime_uid": uid,
                "role": "rigid_object",
                "shape": {"fpath": str(path)},
                "init_pos": [0.0, 0.0, 0.5],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        )
    return SimpleNamespace(planner_objects=objects, articulations=(), table_top_z=0.4)


@pytest.mark.parametrize("strong", [False, True])
@pytest.mark.parametrize("tolerance", [0.05, 0.1])
def test_semantic_binding_keeps_release_checks_and_terminal_acceptance(
    tmp_path, strong, tolerance
):
    from embodichain.gen_sim.task_engine.semantic_program import (
        build_semantic_program_declaration,
    )
    from embodichain.gen_sim.task_engine.stability_declarations import (
        resolve_stability_conditions,
    )
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SupportSensorDeclarationCfg,
    )

    graph, expected = _frozen()
    scene = _source_scene(tmp_path)
    conditions = resolve_stability_conditions(
        graph,
        scene,
        {},
        support_contract_revision=1 if strong else None,
        support_sensors={
            "gen_sim.step_01__call_02.stable": SupportSensorDeclarationCfg(
                uid="explicit_support"
            )
        },
    )
    before = deepcopy(graph)
    actual = build_semantic_program_declaration(
        graph,
        expected["program_id"],
        scene,
        stability_conditions=conditions,
        operation_groups={},
        relative_position_tolerance=tolerance,
    )
    for segment in expected["program"]["items"]:
        for validator in segment.get("validators", ()):
            validator["displacement"] = [0.0, 0.0, 0.150625]
            validator["position_tolerance"] = tolerance
    if strong:
        release, terminal = expected["program"]["items"][1:]
        release.pop("post")
        release.pop("validators")
    assert actual == expected
    assert graph == before
    assert [item["steps"]["call"] for item in actual["program"]["items"]] == [
        node["call"] for node in graph["nodes"]
    ]


def test_control_groups_require_complete_owner_bindings_and_recheck_final_joint(
    tmp_path,
):
    from embodichain.gen_sim.task_engine.semantic_program import (
        build_semantic_program_declaration,
    )
    from embodichain.gen_sim.task_engine.stability_declarations import (
        resolve_stability_conditions,
    )
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SupportSensorDeclarationCfg,
    )

    graph, program = _frozen()
    operation = deepcopy(graph["task_groups"][0])
    operation["id"] = "control"
    operation["task_type"] = "E6"
    operation["node_ids"] = ["control_open", "control_withdraw", "control_park"]
    operation["depends_on"] = []
    operation_nodes = []
    for index, (name, call_id) in enumerate(
        zip(
            operation["node_ids"],
            [
                "gen_sim.articulation_slide",
                "gen_sim.articulation_withdraw",
                "gen_sim.articulation_park",
            ],
            strict=True,
        )
    ):
        node = deepcopy(graph["nodes"][index])
        node.update(
            id=name,
            task_instance_id="control",
            task_type="E6",
            depends_on=[] if not index else [operation["node_ids"][index - 1]],
        )
        node["call"] = {
            "kind": "registered",
            "call_id": call_id,
            "arguments": {"object": "drawer"},
        }
        operation_nodes.append(node)
    graph["nodes"][0]["depends_on"] = [operation["node_ids"][-1]]
    graph["task_groups"][0]["depends_on"] = ["control"]
    graph["nodes"] = operation_nodes + graph["nodes"]
    graph["task_groups"].insert(0, operation)
    scene = _source_scene(tmp_path)
    conditions = resolve_stability_conditions(
        graph,
        scene,
        {},
        support_sensors={
            "gen_sim.step_01__call_02.stable": SupportSensorDeclarationCfg(
                uid="support"
            )
        },
    )
    with pytest.raises(ValueError, match="explicit owner-resolved"):
        build_semantic_program_declaration(
            graph,
            program["program_id"],
            scene,
            stability_conditions=conditions,
            operation_groups={},
        )
    joint = {
        "kind": "articulation_joint_position",
        "articulation": "drawer",
        "joint": "slide",
        "minimum_position": 0.2,
        "maximum_position": 0.3,
    }
    bindings = tuple(
        ProgramSegmentBinding(
            name,
            post=(
                {
                    "kind": "wait_stable",
                    "entity": "drawer",
                    "preset": "explicit_joint_window",
                },
            ),
            validators=(joint,),
        )
        for name in operation["node_ids"]
    )
    actual = build_semantic_program_declaration(
        graph,
        program["program_id"],
        scene,
        stability_conditions=conditions,
        operation_groups={"control": bindings},
    )
    assert actual["program"]["items"][-1]["validators"][-1] == joint
    assert (
        len(
            [
                v
                for v in actual["program"]["items"][-1]["validators"]
                if v["kind"] == "articulation_joint_position"
            ]
        )
        == 1
    )
    with pytest.raises(ValueError, match="exact ordered group"):
        build_semantic_program_declaration(
            graph,
            program["program_id"],
            scene,
            stability_conditions=conditions,
            operation_groups={"control": bindings[:-1]},
        )


def test_semantic_program_qualifies_repeated_placement_occurrences(tmp_path):
    from embodichain.gen_sim.task_engine.semantic_program import (
        build_semantic_program_declaration,
    )
    from embodichain.gen_sim.task_engine.stability_declarations import (
        resolve_stability_conditions,
    )
    from embodichain.lab.task_program.integrations.simulation.stability import (
        SupportSensorDeclarationCfg,
    )

    graph, program = _frozen()
    later = deepcopy(graph["nodes"])
    renamed = {node["id"]: node["id"] + "_again" for node in later}
    for node in later:
        node["id"] = renamed[node["id"]]
        node["task_instance_id"] = "second"
        node["depends_on"] = [renamed[name] for name in node["depends_on"]]
    later[0]["depends_on"] = [graph["nodes"][-1]["id"]]
    group = deepcopy(graph["task_groups"][0])
    group.update(
        id="second",
        node_ids=[node["id"] for node in later],
        depends_on=[graph["task_groups"][0]["id"]],
    )
    graph["nodes"].extend(later)
    graph["task_groups"].append(group)
    scene = _source_scene(tmp_path)
    sensors = {
        f"gen_sim.{name}.stable": SupportSensorDeclarationCfg(uid="support_" + name)
        for name in (graph["nodes"][1]["id"], later[1]["id"])
    }
    conditions = resolve_stability_conditions(graph, scene, {}, support_sensors=sensors)
    actual = build_semantic_program_declaration(
        graph,
        program["program_id"],
        scene,
        stability_conditions=conditions,
        operation_groups={},
    )
    calls = [actual["program"]["items"][index]["steps"]["call"] for index in (1, 4)]
    assert [call["arguments"]["stage_id"] for call in calls] == ["step_01", "second"]
    assert all(
        "stage_id" not in graph["nodes"][index]["call"]["arguments"] for index in (1, 4)
    )
