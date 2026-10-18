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

"""Explicit region-Pour producer contracts, never runtime or liquid qualification."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import trimesh

from embodichain.gen_sim.task_engine._bundle_runner import (
    _verify_integration_fingerprint,
)
from embodichain.gen_sim.task_engine._task_program.configured import decode_task_lowerer
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.utils.utility import load_config


def _case(tmp_path):
    objects = []
    for uid, size, position in (
        ("bottle", (0.06, 0.12, 0.06), [0.1, -0.2, 0.85]),
        ("receiver", (0.24, 0.04, 0.24), [0.2, 0.2, 0.78]),
    ):
        path = tmp_path / f"{uid}.glb"
        trimesh.creation.box(extents=size).export(path)
        objects.append(
            {
                "uid": uid,
                "shape": {"shape_type": "Mesh", "fpath": str(path)},
                "init_pos": position,
                "init_rot": [0.0, 0.0, 0.0],
            }
        )
    table = {
        "uid": "table",
        "shape": {"shape_type": "Cube", "size": [1.4, 1.0, 0.72]},
        "init_pos": [0.0, 0.0, 0.36],
    }
    scene = PreparedScene(
        source_config_path=tmp_path / "scene.yaml",
        scene_dir=tmp_path,
        planner_objects=tuple(
            {**item, "runtime_uid": item["uid"], "role": "rigid_object"}
            for item in objects
        )
        + ({**table, "runtime_uid": "table", "role": "background"},),
        background=(table,),
        rigid_objects=tuple(objects),
        articulations=(),
        uid_map={item["uid"]: item["uid"] for item in (*objects, table)},
        table_top_z=0.72,
        z_rotation_degrees=0.0,
        body_scale_policy="preserve",
        body_scale=(1.0, 1.0, 1.0),
        asset_hashes={},
    )
    calls = [
        {"kind": "pick", "object": "bottle"},
        {
            "kind": "registered",
            "call_id": "simulation.move_held_object",
            "arguments": {
                "object": "bottle",
                "target": "stage",
                "reference": "receiver",
            },
        },
        {
            "kind": "registered",
            "call_id": "simulation.pour",
            "arguments": {"object": "bottle"},
        },
        {
            "kind": "registered",
            "call_id": "simulation.move_held_object",
            "arguments": {"object": "bottle", "target": "home"},
        },
        {
            "kind": "place",
            "object": "bottle",
            "at": {"kind": "target_ref", "target": "home"},
        },
        {"kind": "registered", "call_id": "simulation.park", "arguments": {}},
    ]
    graph = {
        "schema_version": "semantic_task_graph/v1",
        "task_id": "region_pour",
        "instruction": "Pour toward the receiver, restore and return the bottle.",
        "planner_route": "offline",
        "integration_fingerprint": "0" * 64,
        "targets": {
            "home": {
                "kind": "cyclic_pose",
                "values": [
                    {"position": [0.1, -0.2, 0.85], "quaternion_xyzw": [0, 0, 0, 1]}
                ],
            }
        },
        "nodes": [
            {
                "id": f"call_{i}",
                "call": {**call, "resources": {"primary": "left"}},
                "depends_on": [] if i == 0 else [f"call_{i - 1}"],
                "task_instance_id": "pour",
                "task_type": "E3",
                "role": "cleanup" if i == 5 else "primary",
            }
            for i, call in enumerate(calls)
        ],
        "task_groups": [
            {
                "id": "pour",
                "task_type": "E3",
                "node_ids": [f"call_{i}" for i in range(6)],
                "depends_on": [],
                "success": {"kind": "call_completed"},
            }
        ],
        "success": {"kind": "all_task_groups"},
    }
    candidate = {
        "draft": {
            "steps": [
                {
                    "task_type": "E3",
                    "orientation_goal": "none",
                    "axis": "none",
                    "direction": "none",
                }
            ]
        }
    }
    return scene, graph, candidate


def _generate(tmp_path, scene, graph, candidate, **kwargs):
    return generate_task_program_bundle(
        graph,
        scene,
        tmp_path,
        robot_profile="dual_franka",
        task_candidate=candidate,
        verify_pick_phases=True,
        **kwargs,
    )


@pytest.mark.parametrize("revision", [6, 7, 8])
def test_upright_quality_native_revision_survives_strict_bundle_preflight(
    tmp_path, revision
):
    scene, graph, candidate = _case(tmp_path)
    _, bundle = _generate(
        tmp_path / f"native{revision}",
        scene,
        graph,
        candidate,
        pick_contact_revision=revision,
        upright_container_picks={"call_0": 30.0},
    )
    constraints = load_config(bundle.integration.parent / "constraints.json")
    assert constraints["pick_contact_revision"] == revision
    assert constraints["upright_container_picks"] == {"call_0": 30.0}


def test_region_pour_is_explicit_and_preserves_program_options_constraints_and_v1(
    tmp_path,
):
    from embodichain.gen_sim.task_engine._task_program.pour_services import (
        PourRegionTransportFactory,
        PourTransportFactory,
    )

    scene, graph, candidate = _case(tmp_path)
    old_graph, old = _generate(
        tmp_path / "center", scene, graph, candidate, pour_geometry_revision=1
    )
    new_graph, new = _generate(
        tmp_path / "region",
        scene,
        graph,
        candidate,
        pour_geometry_revision=2,
        pour_proxy_position_tolerance_m=0.02,
    )
    old_root, new_root = old.integration.parent.parent, new.integration.parent.parent
    for name in (
        "task_program/program.yaml",
        "task_program/constraints.json",
        "components/execution_policy.yaml",
        "task_program_deployment.yaml",
    ):
        assert (old_root / name).read_bytes() == (new_root / name).read_bytes()
    before, after = load_config(old.integration), load_config(new.integration)
    old_lowerers = before["runtime_services"]["registered_semantic_lowerers"]
    new_lowerers = after["runtime_services"]["registered_semantic_lowerers"]
    a = next(v for v in old_lowerers if v["kind"] == "pour_transport")
    b = next(v for v in new_lowerers if v["kind"] == "pour_region_transport")
    assert {
        **a,
        "kind": "pour_region_transport",
        "proxy_position_tolerance_m": 0.02,
    } == b
    assert type(decode_task_lowerer(a, path="old")) is PourTransportFactory
    assert type(decode_task_lowerer(b, path="new")) is PourRegionTransportFactory
    old_lowerers[old_lowerers.index(a)] = b
    assert before == after
    assert old_graph["integration_fingerprint"] != new_graph["integration_fingerprint"]
    import json

    for root, generated in ((old_root, old_graph), (new_root, new_graph)):
        _verify_integration_fingerprint(
            root,
            root / "task_program_deployment.yaml",
            generated,
            json.loads((root / "integration_fingerprint.json").read_text()),
        )


@pytest.mark.parametrize("value", [None, 0.0, -0.02, True, float("nan"), float("inf")])
def test_region_pour_requires_an_explicit_finite_positive_proxy_budget(tmp_path, value):
    scene, graph, candidate = _case(tmp_path)
    with pytest.raises(
        (TypeError, ValueError), match="proxy.*position|position.*tolerance"
    ):
        _generate(
            tmp_path / "bad",
            scene,
            graph,
            candidate,
            pour_geometry_revision=2,
            pour_proxy_position_tolerance_m=value,
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("revision", [None, 1])
def test_proxy_budget_cannot_silently_change_a_legacy_consumer(tmp_path, revision):
    scene, graph, candidate = _case(tmp_path)
    with pytest.raises(ValueError, match="revision 2"):
        _generate(
            tmp_path / "bad",
            scene,
            graph,
            candidate,
            pour_geometry_revision=revision,
            pour_proxy_position_tolerance_m=0.02,
        )
    assert not (tmp_path / "bad").exists()


def test_region_pour_does_not_replace_an_explicit_stage_point_or_orientation(tmp_path):
    scene, graph, candidate = _case(tmp_path)
    explicit = deepcopy(graph)
    explicit["targets"]["stage"] = deepcopy(graph["targets"]["home"])
    with pytest.raises(ValueError, match="explicit.*stag|stag.*explicit"):
        _generate(
            tmp_path / "point",
            scene,
            explicit,
            candidate,
            pour_geometry_revision=2,
            pour_proxy_position_tolerance_m=0.02,
        )
    candidate["draft"]["steps"][0]["orientation_goal"] = "upright"
    with pytest.raises(ValueError, match="ordinary E3"):
        _generate(
            tmp_path / "orientation",
            scene,
            graph,
            candidate,
            pour_geometry_revision=2,
            pour_proxy_position_tolerance_m=0.02,
        )


@pytest.mark.parametrize("region_owned", [False, True])
def test_assembly_injects_the_same_runtime_region_owner_only_for_new_lowerers(
    monkeypatch, region_owned
):
    from types import SimpleNamespace
    import torch

    from embodichain.gen_sim.task_engine._task_program.actions import GenSimPickUp
    from embodichain.gen_sim.task_engine._task_program.assembly import _TaskFactory
    from embodichain.gen_sim.task_engine._task_program.pour_continuation import (
        PourContinuation,
    )
    from embodichain.gen_sim.task_engine._task_program.pour_services import (
        PourRegionGeometry,
        PourStageRoute,
        _PourRegionTransportLowerer,
        _PourTransportLowerer,
    )
    from embodichain.gen_sim.task_engine._task_program.services import (
        _MoveHeldObjectRoute,
        _SceneEntityTarget,
    )
    from embodichain.lab.task_program.integrations.simulation.environment import (
        SimulationTaskProgramFactory,
    )

    mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.04))
    vertices = {"source": mesh.vertices, "receiver": mesh.vertices}
    geometry = PourRegionGeometry(
        vertices, {"receiver": mesh.faces}, (0, 0, -9.81), 0.02
    )
    base = _PourTransportLowerer(
        (_MoveHeldObjectRoute("source", "stage", _SceneEntityTarget("receiver")),),
        (PourStageRoute("source", "stage", "receiver", 0.1),),
        vertices,
        {"source": "source", "receiver": "receiver"},
        (0, 0, -9.81),
    )
    lowerer = (
        _PourRegionTransportLowerer(base, geometry, object()) if region_owned else base
    )
    monkeypatch.setattr(
        SimulationTaskProgramFactory,
        "create_registered_semantic_lowerers",
        lambda self, **kwargs: (lowerer,),
    )
    factory = object.__new__(_TaskFactory)
    factory._clearance_routes = ()
    factory._alignment_continuation_routes = ()
    factory._pick_continuations = ()
    factory._pour_continuations = (
        PourContinuation(
            "pick",
            "stage",
            "pour",
            "source",
            "receiver",
            "left",
            torch.eye(4),
            0.1,
            1.0,
            140,
            0.02 if region_owned else None,
        ),
    )
    pick = GenSimPickUp()
    result = factory.create_registered_semantic_lowerers(
        scene_registry=object(), engine=SimpleNamespace(actions={"pick_up": pick})
    )
    assert result == (lowerer,)
    if region_owned:
        assert pick.pour_region_geometries == {"pick": geometry}
        assert pick.pour_region_geometries["pick"] is lowerer.region_geometry
    else:
        assert pick.pour_region_geometries is None
