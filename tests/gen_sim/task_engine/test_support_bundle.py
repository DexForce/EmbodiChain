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
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest
import trimesh

from .test_articulation_slide import _graph as drawer_graph
from .test_articulation_slide import scene as drawer_scene
from .test_semantic_graph import _graph, _prepared_axis_scene
from .test_twist import graph as twist_graph
from .test_twist import knob_scene
from embodichain.gen_sim.task_engine import task_program_bundle as bundle
from embodichain.gen_sim.task_engine._task_program.assembly import load_deployment
from embodichain.gen_sim.task_engine._task_program.stability import StabilityConstraint
from embodichain.gen_sim.task_engine._task_program.physical_inputs import (
    load_frozen_physical_input,
)
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.utils.utility import load_config, save_config


def _placement_graph(*, inside: bool = False, reference: str = "table") -> dict:
    graph = _graph()
    graph["task_id"] = "support_bundle"
    graph["nodes"] = [
        {
            "id": "pick",
            "task_type": "E1",
            "task_instance_id": "support_group",
            "role": "primary",
            "depends_on": [],
            "call": {
                "kind": "pick",
                "object": "bottle",
                "resources": {"primary": "left"},
            },
        },
        {
            "id": "place",
            "task_type": "E1",
            "task_instance_id": "support_group",
            "role": "primary",
            "depends_on": ["pick"],
            "call": (
                {
                    "kind": "place",
                    "object": "bottle",
                    "inside": f"inside__{reference}__bottle",
                    "resources": {"primary": "left"},
                }
                if inside
                else {
                    "kind": "registered",
                    "call_id": "simulation.place_relative",
                    "arguments": {
                        "object": "bottle",
                        "reference": reference,
                        "relation": "on",
                    },
                    "resources": {"primary": "left"},
                }
            ),
        },
    ]
    graph["task_groups"] = [
        {
            "id": "support_group",
            "task_type": "E1",
            "node_ids": ["pick", "place"],
            "depends_on": [],
            "success": {"kind": "call_completed"},
        }
    ]
    return graph


@pytest.fixture
def support_scene(tmp_path: Path) -> PreparedScene:
    scene = _prepared_axis_scene(tmp_path)
    path = tmp_path / "tray.glb"
    trimesh.creation.box(extents=(0.3, 0.2, 0.02)).export(path)
    tray = {
        "uid": "tray",
        "shape": {"shape_type": "Mesh", "fpath": str(path)},
        "init_pos": [0.25, 0.15, 0.74],
    }
    return replace(
        scene,
        rigid_objects=(*scene.rigid_objects, tray),
        planner_objects=(
            *scene.planner_objects,
            {**tray, "runtime_uid": "tray", "role": "rigid_object"},
        ),
    )


def _constraints(graph: dict, scene: PreparedScene, *, revision: int | None) -> dict:
    return bundle._task_stability_payload(
        graph,
        scene,
        {"skill_profile": {"resources": []}},
        support_contract_revision=revision,
    )


@pytest.mark.parametrize("inside", [False, True])
def test_explicit_e4_placement_support_is_not_skipped(support_scene, inside):
    graph = _placement_graph(inside=inside, reference="tray")
    for node in graph["nodes"]:
        node["task_type"] = "E4"
    graph["task_groups"][0]["task_type"] = "E4"
    legacy = _constraints(graph, support_scene, revision=None)
    strong = _constraints(graph, support_scene, revision=1)
    assert "support_contract_revision" not in legacy
    assert strong["presets"]["gen_sim.place.stable"]["support_contract_revision"] == 1
    assert strong["presets"]["gen_sim.place.stable"]["reference"] == "tray"


def _load_bundle(paths: Any) -> Any:
    return load_deployment(
        task_program=load_config(paths.deployment)["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )


def test_legacy_bundle_keeps_unversioned_constraints_and_generic_post(
    support_scene: PreparedScene, tmp_path: Path
) -> None:
    _, paths = bundle.generate_task_program_bundle(
        _placement_graph(),
        support_scene,
        tmp_path / "legacy",
        robot_profile="dual_franka",
    )
    raw = load_config(paths.root / "task_program" / "constraints.json")
    assert "support_contract_revision" not in raw
    assert raw["presets"] == {
        "gen_sim.place.stable": {
            "kind": "supported_region",
            "entity": "bottle",
            "reference": "table",
        }
    }
    place = load_config(paths.program)["program"]["items"][-1]
    assert [post["preset"] for post in place["post"]] == [
        "rigid_object",
        "gen_sim.place.stable",
    ]
    loaded = _load_bundle(paths).integration.adapter_factory
    assert loaded.support_contract_revision is None
    assert all(cfg.support_contract_revision is None for _, cfg in loaded.constraints)


def test_typed_inside_exact_rigid_tray_has_one_strong_support_post(
    support_scene: PreparedScene,
) -> None:
    graph = _placement_graph(inside=True, reference="tray")
    legacy = _constraints(graph, support_scene, revision=None)
    assert legacy["presets"] == {}
    strong = _constraints(graph, support_scene, revision=1)
    cfg = strong["presets"]["gen_sim.place.stable"]
    assert cfg == {
        "kind": "supported_region",
        "entity": "bottle",
        "reference": "tray",
        "support_contract_revision": 1,
    }
    assert StabilityConstraint.decode(cfg).support_contract_revision == 1
    place = bundle._program_payload(
        graph, "support_bundle", support_scene, stability_presets=strong["presets"]
    )["program"]["items"][-1]
    assert place["post"] == [
        {
            "kind": "wait_stable",
            "entity": "bottle",
            "preset": "gen_sim.place.stable",
        }
    ]


@pytest.mark.parametrize("oriented", [False, True])
def test_relative_on_exact_reference_preserves_existing_pose_contract(
    support_scene: PreparedScene, oriented: bool
) -> None:
    graph = _placement_graph(reference="tray")
    if oriented:
        graph["nodes"].insert(
            1,
            {
                "id": "align",
                "task_type": "E1",
                "task_instance_id": "support_group",
                "call": {
                    "kind": "registered",
                    "call_id": "gen_sim.align_held",
                    "arguments": {"object": "bottle", "target": "current_object_pose"},
                    "resources": {"primary": "left"},
                },
            },
        )
        graph["task_groups"][0]["node_ids"].insert(1, "align")
    legacy = _constraints(graph, support_scene, revision=None)
    strong = _constraints(graph, support_scene, revision=1)
    original = legacy["presets"]["gen_sim.place.stable"]
    cfg = strong["presets"]["gen_sim.place.stable"]
    assert cfg["reference"] == "tray"
    assert cfg["support_contract_revision"] == 1
    assert cfg["displacement"] == original["displacement"]
    assert cfg["position_tolerance"] == original["position_tolerance"]
    if oriented:
        assert cfg["local_axis"] == original["local_axis"]
    assert StabilityConstraint.decode(cfg).reference == "tray"
    old_place = bundle._program_payload(
        graph, "legacy", support_scene, stability_presets=legacy["presets"]
    )["program"]["items"][-1]
    new_place = bundle._program_payload(
        graph, "strong", support_scene, stability_presets=strong["presets"]
    )["program"]["items"][-1]
    assert new_place["validators"] == old_place["validators"]
    assert len(new_place["post"]) == 1


@pytest.mark.parametrize("inside", [False, True])
@pytest.mark.parametrize("clearance", [False, True])
def test_cleanup_terminal_reuses_one_strong_window_without_generic_wait(
    support_scene: PreparedScene, clearance: bool, inside: bool
) -> None:
    graph = _placement_graph(reference="tray", inside=inside)
    cleanup = [("park", "simulation.park", {})]
    if clearance:
        cleanup.insert(
            0,
            ("clear", "gen_sim.clear_released", {"object": "bottle", "target": "away"}),
        )
    for name, call_id, arguments in cleanup:
        graph["nodes"].append(
            {
                "id": name,
                "task_type": "E1",
                "task_instance_id": "support_group",
                "call": {
                    "kind": "registered",
                    "call_id": call_id,
                    "arguments": arguments,
                },
            }
        )
        graph["task_groups"][0]["node_ids"].append(name)
    constraints = _constraints(graph, support_scene, revision=1)
    items = bundle._program_payload(
        graph, "cleanup", support_scene, stability_presets=constraints["presets"]
    )["program"]["items"]
    assert items[-1]["post"] == [
        {
            "kind": "wait_stable",
            "entity": "bottle",
            "preset": "gen_sim.place.stable",
        }
    ]
    if not inside:
        assert items[-1]["validators"][0]["reference"] == "tray"
    assert "post" not in items[1]
    assert "validators" not in items[1]
    owners = [
        item["name"]
        for item in items
        for policy in item.get("post", [])
        if policy.get("preset") == "gen_sim.place.stable"
    ]
    assert owners == ["park"]
    legacy = _constraints(graph, support_scene, revision=None)
    old_items = bundle._program_payload(
        graph, "legacy", support_scene, stability_presets=legacy["presets"]
    )["program"]["items"]
    assert bool(old_items[1].get("post")) is not clearance


@pytest.mark.parametrize("unsupported", ["moving_link", "wrong_object"])
def test_global_marker_does_not_promote_unsupported_inside_pair(
    support_scene: PreparedScene, unsupported: str
) -> None:
    graph = _placement_graph(inside=True, reference="tray")
    if unsupported == "moving_link":
        graph["nodes"][-1]["call"]["inside"] += "__part_slide"
    else:
        graph["nodes"][-1]["call"]["inside"] = "inside__tray__other"
    constraints = _constraints(graph, support_scene, revision=1)
    assert constraints["support_contract_revision"] == 1
    assert constraints["presets"] == {}


@pytest.mark.parametrize("with_bundle_marker", [False, True])
def test_strong_bundle_round_trips_through_production_loader(
    support_scene: PreparedScene, tmp_path: Path, with_bundle_marker: bool
) -> None:
    _, paths = bundle.generate_task_program_bundle(
        _placement_graph(reference="tray"),
        support_scene,
        tmp_path / "strong",
        robot_profile="dual_franka",
        support_contract_revision=1,
    )
    if not with_bundle_marker:
        constraints_path = paths.root / "task_program" / "constraints.json"
        constraints = load_config(constraints_path)
        constraints.pop("support_contract_revision")
        save_config(constraints_path, constraints)
        with pytest.raises(
            ValueError,
            match="Strong support presets require the bundle contract marker",
        ):
            _load_bundle(paths)
        return
    loaded = _load_bundle(paths)
    adapter = loaded.integration.adapter_factory
    assert adapter.support_contract_revision == 1
    assert len(adapter.constraints) == 1
    cfg = dict(adapter.constraints)["gen_sim.place.stable"]
    assert (cfg.entity, cfg.reference, cfg.support_contract_revision) == (
        "bottle",
        "tray",
        1,
    )
    assert len(load_config(paths.program)["program"]["items"][-1]["post"]) == 1


@pytest.mark.parametrize("revision", [None, 1, 2, 3, 4])
def test_pick_contact_contract_is_explicit_and_independent_of_support(
    support_scene: PreparedScene, tmp_path: Path, revision: int | None
) -> None:
    _, paths = bundle.generate_task_program_bundle(
        _placement_graph(reference="tray"),
        support_scene,
        tmp_path / "pick_contact",
        robot_profile="dual_franka",
        pick_contact_revision=revision,
    )
    constraints = load_config(paths.root / "task_program/constraints.json")
    assert constraints.get("pick_contact_revision") == revision
    assert "support_contract_revision" not in constraints
    assert (
        _load_bundle(paths).integration.adapter_factory.pick_contact_revision
        == revision
    )


def test_explicit_inside_yaw_survives_production_bundle_loading(
    support_scene: PreparedScene, tmp_path: Path
) -> None:
    # Source meshes are Y-up; provide a horizontal floor large enough for this bottle.
    trimesh.creation.box(extents=(0.5, 0.02, 0.5)).export(
        support_scene.rigid_objects[-1]["shape"]["fpath"]
    )
    graph = _placement_graph(inside=True, reference="tray")
    _, paths = bundle.generate_task_program_bundle(
        graph,
        support_scene,
        tmp_path / "yaw",
        robot_profile="dual_franka",
        world_yaw_free_places=["place"],
    )
    constraints = load_config(paths.root / "task_program/constraints.json")
    assert constraints["world_yaw_free_places"] == ["place"]
    loaded = _load_bundle(paths).integration.adapter_factory
    assert len(loaded.world_yaw_free_places) == 1
    assert loaded.world_yaw_free_places[0].endswith(":0")
    defaults = []
    for label, selection in (("legacy", None), ("empty", [])):
        _, default_paths = bundle.generate_task_program_bundle(
            graph,
            support_scene,
            tmp_path / label,
            robot_profile="dual_franka",
            world_yaw_free_places=selection,
        )
        raw = load_config(default_paths.root / "task_program/constraints.json")
        assert "world_yaw_free_places" not in raw
        default = _load_bundle(default_paths).integration
        assert default.adapter_factory.world_yaw_free_places == ()
        defaults.append(default.integration_fingerprint)
    assert defaults[0] == defaults[1]
    assert defaults[0] != _load_bundle(paths).integration.integration_fingerprint
    for invalid in (["pick"], ["missing"], ["place", "place"], "place"):
        with pytest.raises(ValueError, match="yaw"):
            bundle.generate_task_program_bundle(
                graph,
                support_scene,
                tmp_path / f"invalid_{repr(invalid)}",
                robot_profile="dual_franka",
                world_yaw_free_places=invalid,
            )


def test_per_call_contact_revision_survives_strict_bundle_loading(
    support_scene: PreparedScene, tmp_path: Path
) -> None:
    _, paths = bundle.generate_task_program_bundle(
        _placement_graph(reference="tray"),
        support_scene,
        tmp_path / "per_call",
        robot_profile="dual_franka",
        pick_contact_revisions={"pick": 2},
    )
    constraints = load_config(paths.root / "task_program/constraints.json")
    assert constraints["pick_contact_revisions"] == {"pick": 2}
    assert "pick_contact_revision" not in constraints
    loaded = _load_bundle(paths).integration.adapter_factory
    assert loaded.pick_contact_revision is None
    assert len(loaded.pick_contact_revisions) == 1
    assert loaded.pick_contact_revisions[0][0].endswith(":0")
    assert loaded.pick_contact_revisions[0][1] == 2
    for name in ("place", "missing"):
        with pytest.raises(ValueError, match="explicit Pick"):
            bundle.generate_task_program_bundle(
                _placement_graph(reference="tray"),
                support_scene,
                tmp_path / name,
                robot_profile="dual_franka",
                pick_contact_revisions={name: 2},
            )


@pytest.mark.parametrize("revision", [True, False, 0, 9, 3.0, 4.0, "3", "4"])
def test_pick_contact_contract_rejects_unknown_revisions(
    support_scene: PreparedScene, tmp_path: Path, revision: object
) -> None:
    with pytest.raises(ValueError, match="Pick contact revision"):
        bundle.generate_task_program_bundle(
            _placement_graph(reference="tray"),
            support_scene,
            tmp_path / "invalid_pick_contact",
            robot_profile="dual_franka",
            pick_contact_revision=revision,
        )


def test_pick_contact_revision_only_changes_its_explicit_constraint_marker(
    support_scene: PreparedScene, tmp_path: Path
) -> None:
    graph = _placement_graph(reference="tray")
    _, reference = bundle.generate_task_program_bundle(
        graph, support_scene, tmp_path / "reference", robot_profile="dual_franka"
    )
    frozen = load_frozen_physical_input(
        support_scene, reference.root, robot_profile="dual_franka"
    )
    original_components = None
    original_constraints = None
    for revision in (None, 1, 2, 3, 4):
        _, paths = bundle.generate_task_program_bundle(
            graph,
            frozen,
            tmp_path / f"revision_{revision}",
            robot_profile="dual_franka",
            support_contract_revision=1,
            pick_contact_revision=revision,
        )
        components = {
            name: getattr(paths, name).read_bytes()
            for name in (
                "program",
                "integration",
                "scene",
                "embodiment",
                "execution_policy",
                "deployment",
            )
        }
        constraints = load_config(paths.root / "task_program/constraints.json")
        assert constraints.pop("pick_contact_revision", None) == revision
        encoded = json.dumps(constraints, ensure_ascii=False, indent=2).encode()
        if revision is None:
            original_components, original_constraints = components, encoded
        else:
            assert components == original_components
            assert encoded == original_constraints
        frozen.assert_unchanged()


def _articulation_bundles(
    support_scene: PreparedScene,
    control_scene: PreparedScene,
    tmp_path: Path,
    family: str,
) -> tuple[Any, Any]:
    articulations = deepcopy(control_scene.articulations)
    if family == "E8":
        articulations[0]["init_pos"] = [-0.25, 0.25, 0.8]
    scene = replace(
        support_scene,
        articulations=articulations,
        planner_objects=(
            *support_scene.planner_objects,
            *(
                {**item, "runtime_uid": item["uid"], "role": "articulation"}
                for item in articulations
            ),
        ),
    )
    graph = _placement_graph(reference="tray")
    suffix = drawer_graph(scene) if family == "E6" else twist_graph(scene)
    composed = deepcopy(graph)
    composed["nodes"].extend(suffix["nodes"])
    composed["task_groups"].extend(suffix["task_groups"])
    _, before = bundle.generate_task_program_bundle(
        graph,
        scene,
        tmp_path / "prefix",
        robot_profile="dual_franka",
        support_contract_revision=1,
    )
    _, after = bundle.generate_task_program_bundle(
        composed,
        scene,
        tmp_path / "composed",
        robot_profile="dual_franka",
        support_contract_revision=1,
    )
    return before, after


@pytest.mark.parametrize("family", ["E6", "E8"])
def test_unrelated_articulation_suffix_preserves_e1_preset_and_physics(
    support_scene: PreparedScene,
    drawer_scene: PreparedScene,
    knob_scene: PreparedScene,
    tmp_path: Path,
    family: str,
) -> None:
    before, after = _articulation_bundles(
        support_scene, drawer_scene if family == "E6" else knob_scene, tmp_path, family
    )
    original = _load_bundle(before)
    combined = _load_bundle(after)
    assert (
        dict(original.integration.adapter_factory.constraints)["gen_sim.place.stable"]
        == dict(combined.integration.adapter_factory.constraints)[
            "gen_sim.place.stable"
        ]
    )
    assert (
        original.integration.registration.settle_presets["rigid_object"]
        == combined.integration.registration.settle_presets["rigid_object"]
    )
    first, second = load_config(before.deployment), load_config(after.deployment)
    for field in ("physics", "physics_config", "env", "max_episode_steps"):
        assert second[field] == first[field]
    assert load_config(after.execution_policy) == load_config(before.execution_policy)
    if family == "E8":
        assert (
            load_config(after.embodiment)["simulation"]["joint_drive_props"]
            == load_config(before.embodiment)["simulation"]["joint_drive_props"]
        )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Existing task_program_bundle.py E6 all-motion-endpoint stiffness calibration changes the unrelated E1 left arm; repair in a separate physics-profile change.",
)
def test_unrelated_e6_suffix_preserves_e1_left_arm_stiffness(
    support_scene: PreparedScene, drawer_scene: PreparedScene, tmp_path: Path
) -> None:
    before, after = _articulation_bundles(support_scene, drawer_scene, tmp_path, "E6")
    first_drive = load_config(before.embodiment)["simulation"]["joint_drive_props"]
    second_drive = load_config(after.embodiment)["simulation"]["joint_drive_props"]
    assert second_drive["stiffness"]["left_arm"] == first_drive["stiffness"]["left_arm"]
