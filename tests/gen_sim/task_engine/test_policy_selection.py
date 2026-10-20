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
from pathlib import Path
from types import SimpleNamespace

import pytest
import trimesh

from embodichain.gen_sim.task_engine._task_program.policy_selection import (
    select_manipulation_policy,
)
from embodichain.gen_sim.task_engine import task_program_bundle as bundle
from embodichain.utils.utility import load_config

from .test_support_bundle import _load_bundle, _placement_graph, support_scene


def candidate(*, orientation: str = "none", source: str = "scene_ref") -> dict:
    return {
        "draft": {
            "steps": [
                {"id": "upright", "task_type": "E2", "orientation_goal": "upright"},
                {
                    "id": "support_group",
                    "task_type": "E1",
                    "orientation_goal": orientation,
                    "axis": "none",
                    "direction": "none",
                    "layout": "none",
                    "object": {
                        "kind": source,
                        "step_id": "upright" if source == "step_result" else "",
                    },
                },
            ]
        }
    }


def scene(*, confined: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        planner_objects=(
            {
                "runtime_uid": "bottle",
                "role": "rigid_object",
                "attributes": {
                    "final_world_aabb": {"min": [0, 0, 0], "max": [0.05, 0.05, 0.1]}
                },
            },
            {"runtime_uid": "tray", "role": "rigid_object"},
            {
                "runtime_uid": "drawer",
                "role": "articulation",
                "attributes": {
                    "final_world_aabb": {
                        "min": [-0.1, -0.1, -0.1] if confined else [1, 1, 1],
                        "max": [0.2, 0.2, 0.2] if confined else [2, 2, 2],
                    }
                },
            },
        )
    )


def test_heading_free_rigid_inside_selects_existing_call_local_capabilities() -> None:
    graph = _placement_graph(inside=True, reference="tray")
    assert select_manipulation_policy(graph, candidate(), scene()) == {
        "world_yaw_free_places": ["place"],
        "measured_rigid_calls": ["pick", "place"],
    }


def test_reacquisition_of_upright_result_uses_qualified_closing_not_preshape() -> None:
    graph = _placement_graph(inside=True, reference="tray")
    assert select_manipulation_policy(
        graph, candidate(source="step_result"), scene()
    ) == {
        "pick_contact_revisions": {"pick": 2},
        "world_yaw_free_places": ["place"],
    }


@pytest.mark.parametrize("orientation", ["upright", "preserve", "inverted"])
def test_explicit_orientation_does_not_gain_heading_or_measured_permission(
    orientation: str,
) -> None:
    assert (
        select_manipulation_policy(
            _placement_graph(inside=True, reference="tray"),
            candidate(orientation=orientation),
            scene(),
        )
        == {}
    )


def test_nonrigid_receiver_on_placement_and_explicit_grasp_remain_legacy() -> None:
    assert select_manipulation_policy(_placement_graph(), candidate(), scene()) == {}
    assert (
        select_manipulation_policy(
            _placement_graph(inside=True, reference="drawer"), candidate(), scene()
        )
        == {}
    )
    graph = _placement_graph(inside=True, reference="tray")
    graph["nodes"][0]["call"]["grasp"] = "authored_grasp"
    assert select_manipulation_policy(graph, candidate(), scene()) == {}


@pytest.mark.parametrize("family", ["E2", "E3", "E4", "E5", "E6", "E8", "E9"])
def test_other_recipes_do_not_acquire_ordinary_inside_policy(family: str) -> None:
    graph = _placement_graph(inside=True, reference="tray")
    for node in graph["nodes"]:
        node["task_type"] = family
    assert select_manipulation_policy(graph, candidate(), scene()) == {}


def test_unrelated_suffix_and_node_reordering_preserve_selected_policies() -> None:
    graph = _placement_graph(inside=True, reference="tray")
    baseline = select_manipulation_policy(graph, candidate(), scene())
    graph["nodes"].append(
        {
            "id": "unrelated",
            "task_instance_id": "drawer_group",
            "task_type": "E6",
            "depends_on": ["place"],
            "call": {
                "kind": "registered",
                "call_id": "gen_sim.articulation_slide",
                "arguments": {"object": "drawer"},
                "resources": {"primary": "left"},
            },
        }
    )
    assert select_manipulation_policy(graph, candidate(), scene()) == baseline
    graph["nodes"].reverse()
    assert select_manipulation_policy(graph, candidate(), scene()) == baseline


def test_task_and_entity_names_are_not_policy_selectors() -> None:
    graph = _placement_graph(inside=True, reference="tray")
    graph["task_id"] = "unseen_task"
    graph["nodes"][0]["call"]["object"] = "unseen_object"
    graph["nodes"][1]["call"].update(
        object="unseen_object", inside="inside__unseen_receiver__unseen_object"
    )
    renamed = deepcopy(scene())
    renamed.planner_objects[0]["runtime_uid"] = "unseen_object"
    renamed.planner_objects[1]["runtime_uid"] = "unseen_receiver"
    assert select_manipulation_policy(graph, candidate(), renamed) == {
        "world_yaw_free_places": ["place"],
        "measured_rigid_calls": ["pick", "place"],
    }


def test_selected_capabilities_survive_normal_bundle_generation_and_loading(
    support_scene, tmp_path: Path
) -> None:
    trimesh.creation.box(extents=(0.5, 0.02, 0.5)).export(
        support_scene.rigid_objects[-1]["shape"]["fpath"]
    )
    graph = _placement_graph(inside=True, reference="tray")
    original = deepcopy(graph)
    policies = select_manipulation_policy(graph, candidate(), support_scene)
    assert graph == original
    _, paths = bundle.generate_task_program_bundle(
        graph,
        support_scene,
        tmp_path / "selected",
        robot_profile="dual_franka",
        **policies,
    )
    raw = load_config(paths.root / "task_program/constraints.json")
    assert raw["world_yaw_free_places"] == ["place"]
    assert "measured_rigid_calls" not in raw
    loaded = _load_bundle(paths).integration.adapter_factory
    assert len(loaded.measured_rigid_calls) == 0
    assert len(loaded.world_yaw_free_places) == 1


def test_tabletop_source_does_not_gain_measured_preshape_or_transport() -> None:
    assert select_manipulation_policy(
        _placement_graph(inside=True, reference="tray"),
        candidate(),
        scene(confined=False),
    ) == {"world_yaw_free_places": ["place"]}


def test_previous_body_operation_invalidates_initial_confinement_hint() -> None:
    graph = _placement_graph(inside=True, reference="tray")
    graph["nodes"][0]["depends_on"] = ["earlier_move"]
    graph["nodes"].append(
        {
            "id": "earlier_move",
            "task_type": "E1",
            "task_instance_id": "earlier",
            "depends_on": [],
            "call": {
                "kind": "registered",
                "call_id": "simulation.place_relative",
                "arguments": {
                    "object": "bottle",
                    "reference": "table",
                    "relation": "on",
                },
                "resources": {"primary": "left"},
            },
        }
    )
    assert select_manipulation_policy(graph, candidate(), scene()) == {
        "world_yaw_free_places": ["place"]
    }


def test_prior_confined_cargo_qualifies_only_later_shared_receiver_place() -> None:
    graph = _placement_graph(inside=True, reference="tray")
    second_pick = deepcopy(graph["nodes"][0])
    second_place = deepcopy(graph["nodes"][1])
    second_pick.update(id="pick2", task_instance_id="later", depends_on=["place"])
    second_pick["call"]["object"] = "other"
    second_place.update(id="place2", task_instance_id="later", depends_on=["pick2"])
    second_place["call"].update(object="other", inside="inside__tray__other")
    graph["nodes"].extend([second_pick, second_place])
    request = candidate()
    step = deepcopy(request["draft"]["steps"][1])
    step["id"] = "later"
    request["draft"]["steps"].append(step)
    selected_scene = scene()
    selected_scene.planner_objects += (
        {"runtime_uid": "other", "role": "rigid_object"},
    )
    selected = select_manipulation_policy(graph, request, selected_scene)
    assert selected["measured_rigid_calls"] == ["pick", "place", "place2"]
    second_pick["depends_on"] = []
    selected = select_manipulation_policy(graph, request, selected_scene)
    assert selected["measured_rigid_calls"] == ["pick", "place"]
