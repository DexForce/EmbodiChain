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
from types import SimpleNamespace
from copy import deepcopy
import pytest
from embodichain.gen_sim.task_engine import prismatic_declarations as owner
from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding


def _graph():
    args = {"object": "cabinet", "state": "open", "part": "part_drawer"}
    calls = [
        dict(
            kind="registered",
            call_id=identity,
            arguments=deepcopy(args),
            resources={"primary": "custom_resource"},
        )
        for identity in ("gen_sim.articulation_slide", "gen_sim.articulation_withdraw")
    ] + [
        dict(
            kind="registered",
            call_id="gen_sim.articulation_park",
            arguments={},
            resources={"primary": "custom_resource"},
        )
    ]
    nodes = [
        dict(
            id=name,
            task_type="E6",
            task_instance_id="operation",
            role="primary" if index == 0 else "cleanup",
            depends_on=[] if not index else [f"node{index-1}"],
            call=call,
        )
        for index, (name, call) in enumerate(
            zip(("node0", "node1", "node2"), calls, strict=True)
        )
    ]
    return dict(
        nodes=nodes,
        task_groups=[
            dict(
                id="operation", task_type="E6", node_ids=[node["id"] for node in nodes]
            )
        ],
    )


def test_explicit_prismatic_recipe_generates_shared_bindings_and_source_checks(
    monkeypatch,
):
    source = PrismaticBinding(
        "cabinet",
        "rail",
        "drawer",
        "base",
        "/root/drawer/handle",
        "a" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.2),
        part_id="part_drawer",
        closed_position=0.0,
    )
    captured = []
    monkeypatch.setattr(owner, "inspect_prismatic_part", lambda config, part: source)
    monkeypatch.setattr(
        owner,
        "validate_prismatic_placement",
        lambda binding, config, height: captured.append((binding, config, height)),
    )
    graph = _graph()
    before = deepcopy(graph)
    config = {"uid": "cabinet", "fpath": "explicit.usda"}
    result = owner.resolve_prismatic_program_declarations(
        graph, SimpleNamespace(articulations=(config,), table_top_z=0.75)
    )
    assert captured == [(source, config, 0.75)]
    assert result.link_declarations() == [
        dict(
            entity_id="cabinet__drawer",
            articulation_id="cabinet",
            native_link_name="drawer",
        )
    ]
    condition = result.conditions["gen_sim.articulation.cabinet.part_drawer.open"]
    assert condition.target == 0.2 and condition.tolerance == 0.03
    assert (
        condition.duration == 1.0
        and condition.timeout == 5.0
        and condition.position_drift == 0.001
    )
    expected = dict(
        kind="articulation_joint_position",
        articulation="cabinet",
        joint="rail",
        minimum_position=0.17,
        maximum_position=0.23,
    )
    for binding in result.operation_groups["operation"]:
        assert binding.validators[0] == expected
        assert binding.post[0]["entity"] == "cabinet"
    lowerers = result.lowerer_declarations(table_uid="explicit_table")
    assert [item["kind"] for item in lowerers] == [
        "prismatic_slide",
        "prismatic_withdraw",
        "articulation_park",
    ]
    assert lowerers[0]["bindings"][0]["source"] == source.payload()
    assert graph == before
    broken = deepcopy(graph)
    broken["nodes"][1]["depends_on"] = []
    with pytest.raises(ValueError, match="preceding action"):
        owner.resolve_prismatic_program_declarations(
            broken, SimpleNamespace(articulations=(config,), table_top_z=0.75)
        )
    assert len(captured) == 1


def test_drawer_declarations_use_owner_link_and_per_occurrence_vertical_policy(
    monkeypatch,
):
    import numpy as np
    from embodichain.gen_sim.task_engine import drawer_declarations as drawer
    from embodichain.lab.task_program.integrations.simulation.prismatic_lowering import (
        PrismaticControlBinding,
    )

    source = PrismaticBinding(
        "cabinet",
        "rail",
        "drawer",
        "base",
        "/drawer/grip",
        "0" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.4),
        part_id="part_drawer",
        closed_position=0.0,
    )
    target = "inside__cabinet__payload__part_drawer"
    route_call = dict(
        kind="place", object="payload", inside=target, resources={"primary": "custom"}
    )
    graph = {
        "nodes": [
            {
                "id": "align",
                "task_instance_id": "vertical",
                "call": {
                    "kind": "registered",
                    "call_id": "gen_sim.align_held",
                    "arguments": {"object": "payload", "target": "current_object_pose"},
                },
            },
            {"id": "first", "task_instance_id": "vertical", "call": route_call},
            {
                "id": "second",
                "task_instance_id": "ordinary",
                "call": deepcopy(route_call),
            },
        ]
    }
    scene = SimpleNamespace(
        articulations=({"uid": "cabinet"},),
        planner_objects=({"runtime_uid": "payload"},),
    )
    binding = owner.PrismaticProgramDeclarations(
        (PrismaticControlBinding("owner_selected_link", source),), {}, {}
    )
    monkeypatch.setattr(
        drawer, "read_drawer_region", lambda cfg, binding: (np.zeros(3), np.ones(3))
    )
    monkeypatch.setattr(drawer, "source_drawer_container_pose", lambda *args: np.eye(4))
    (route,) = drawer.resolve_drawer_declarations(graph, scene, binding)
    assert route.vertical_nodes == ("first",)
    assert route.container_declaration()["parent_id"] == "owner_selected_link"
    assert (
        route.invocation_declaration("program/close:0", "close")["target_id"]
        == "owner_selected_link"
    )
    assert (
        route.invocation_declaration("program/place:0", "place")["target_id"] == target
    )
    assert (
        route.invocation_declaration("program/place:0", "place")["object_id"]
        == "payload"
    )
    with pytest.raises(ValueError):
        route.invocation_declaration("program/place:0", "unknown")


def test_drawer_binding_defers_only_its_acquisition_and_checks_all_close_payloads():
    from embodichain.gen_sim.task_engine.drawer_declarations import (
        DrawerDeclaration,
        bind_drawer_occurrences,
    )

    source = PrismaticBinding(
        "cabinet",
        "rail",
        "drawer",
        "base",
        "/drawer/grip",
        "0" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.4),
        part_id="part_drawer",
        closed_position=0.0,
    )
    routes = tuple(
        DrawerDeclaration(
            f"inside__cabinet__{obj}__part_drawer",
            obj,
            "drawer_link",
            source,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            (),
        )
        for obj in ("payload", "other")
    )
    target = routes[0].target_id
    calls = [
        dict(kind="pick", object="payload", resources={"primary": "active"}),
        dict(
            kind="registered",
            call_id="gen_sim.drawer_transport",
            arguments={"object": "payload", "target": target},
            resources={"primary": "active"},
        ),
        dict(
            kind="place",
            object="payload",
            inside=target,
            resources={"primary": "active"},
        ),
        dict(
            kind="registered",
            call_id="gen_sim.articulation_slide",
            arguments={"object": "cabinet", "part": "part_drawer", "state": "closed"},
            resources={"primary": "closing"},
        ),
        dict(kind="pick", object="payload", resources={"primary": "unrelated"}),
    ]
    nodes = [
        dict(
            id=str(index), call=call, depends_on=[] if index == 0 else [str(index - 1)]
        )
        for index, call in enumerate(calls)
    ]
    rows, deferred = bind_drawer_occurrences(
        {"nodes": nodes}, routes, {str(i): f"program/{i}:0" for i in range(len(nodes))}
    )
    assert deferred == frozenset({"0"})
    assert len(rows["3"]) == 2
    assert {row["object_id"] for row in rows["3"]} == {"payload", "other"}
    assert all(row["target_id"] == "drawer_link" for row in rows["3"])
    assert "4" not in rows


@pytest.mark.parametrize("transport", ["insert", "legacy", "conflict"])
def test_composite_drawer_graph_declares_canonical_transport_and_close_resource(
    monkeypatch, transport
):
    from dataclasses import replace
    from embodichain.gen_sim.task_engine import drawer_graph_declarations as drawer
    from embodichain.lab.task_program.integrations.simulation.prismatic_lowering import (
        PrismaticControlBinding,
    )

    original = PrismaticBinding(
        "cabinet",
        "rail",
        "drawer",
        "base",
        "/drawer/handle",
        "a" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.2),
        closed_position=0.0,
    )
    canonical = replace(original, part_id="part_drawer")
    monkeypatch.setattr(
        drawer,
        "resolve_prismatic_program_declarations",
        lambda graph, scene: owner.PrismaticProgramDeclarations(
            (PrismaticControlBinding("cabinet__drawer", original),), {}, {}
        ),
    )
    monkeypatch.setattr(
        drawer,
        "discover_prismatic_parts",
        lambda cfg: (SimpleNamespace(joint="rail", part_id="part_drawer"),),
    )
    monkeypatch.setattr(drawer, "inspect_prismatic_part", lambda cfg, part: canonical)
    graph = _graph()
    for node in graph["nodes"]:
        node["call"]["resources"]["primary"] = "left"
        node["call"]["arguments"].pop("part", None)
    release = dict(
        id="release",
        task_type="E1",
        task_instance_id="landing",
        role="primary",
        depends_on=[graph["nodes"][-1]["id"]],
        call=dict(
            kind="place",
            object="payload",
            inside="inside__cabinet__payload",
            resources={"primary": "left"},
        ),
    )
    expected_call = dict(
        kind="registered",
        call_id="gen_sim.drawer_transport",
        arguments=dict(
            object="payload", target="inside__cabinet__payload__part_drawer"
        ),
        resources={"primary": "left"},
    )
    landing_ids = [release["id"]]
    if transport != "insert":
        call = deepcopy(expected_call)
        call["call_id"] = "simulation.move_held_object"
        if transport == "conflict":
            call["arguments"]["object"] = "other"
        graph["nodes"].append(dict(release, id="release__transport", call=call))
        release["depends_on"] = ["release__transport"]
        landing_ids.insert(0, "release__transport")
    graph["nodes"].append(release)
    graph["task_groups"].append(
        dict(id="landing", task_type="E1", node_ids=landing_ids)
    )
    before = deepcopy(graph)
    scene = SimpleNamespace(articulations=({"uid": "cabinet"},))
    if transport == "conflict":
        with pytest.raises(ValueError, match="transport conflicts"):
            drawer.prepare_source_drawer_graph(graph, scene)
    else:
        result = drawer.prepare_source_drawer_graph(graph, scene)
        assert (
            result["nodes"][-1]["call"]["inside"]
            == "inside__cabinet__payload__part_drawer"
        )
        assert result["nodes"][-2]["call"] == expected_call
        assert result["nodes"][-1]["depends_on"] == ["release__transport"]
        assert result["task_groups"][-1]["node_ids"] == [
            "release__transport",
            "release",
        ]
        assert result["nodes"][0]["call"]["arguments"]["part"] == "part_drawer"
        assert drawer.prepare_source_drawer_graph(result, scene) == result
        for node in result["nodes"][:2]:
            node["call"]["arguments"]["state"] = "closed"
        closed = drawer.rewrite_source_drawer_close_resources(
            result, drawer_bindings=frozenset({("cabinet", "part_drawer")})
        )
        assert all(
            n["call"]["resources"]["primary"] == "right" for n in closed["nodes"][:3]
        )
        assert closed["nodes"][-1]["call"]["resources"]["primary"] == "left"
        assert (
            drawer.rewrite_source_drawer_close_resources(
                closed, drawer_bindings=frozenset({("cabinet", "part_drawer")})
            )
            == closed
        )
    assert graph == before
