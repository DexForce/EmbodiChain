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

import pytest

from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.domain import (
    motion_policy,
    public_task_spec,
    seed_graph_hash,
    validate_public_task_spec,
    validate_scene_requirements,
    validate_seed_graph,
    validate_task_spec,
)
from embodichain.gen_sim.action_engine.protocol import (
    SCENE_REQUIREMENTS_SCHEMA,
    SEED_GRAPH_SCHEMA,
    TASK_SPEC_SCHEMA,
)
from embodichain.gen_sim.action_engine.planning.linker import link_seed_graph


def _seed_graph() -> dict:
    registry = build_atomic_capability_registry()
    draft = {
        "schema_version": SEED_GRAPH_SCHEMA,
        "task_id": "place-cup",
        "instruction": "Place the cup in the tray.",
        "level": "L1",
        "reasoning_type": "none",
        "planner_route": "offline",
        "nodes": [
            {
                "id": "pick_cup",
                "atomic_action": "PickUp",
                "object_uid": "cup",
                "actor": {"mode": "auto"},
                "control": "arm",
                "target_binding": {"kind": "object", "object": "cup"},
                "depends_on": [],
                "task_instance_id": "e1_001",
                "task_type": "E1",
                "role": "primary",
                "precondition": {"type": "object_not_fallen", "object": "cup"},
                "postcondition": {"type": "object_held", "object": "cup"},
                "motion_policy": motion_policy(),
            },
            {
                "id": "place_cup",
                "atomic_action": "Place",
                "object_uid": "cup",
                "actor": {"mode": "auto"},
                "control": "arm",
                "target_binding": {"kind": "current_held_pose"},
                "depends_on": ["pick_cup"],
                "task_instance_id": "e1_001",
                "task_type": "E1",
                "role": "primary",
                "precondition": {"type": "object_held", "object": "cup"},
                "postcondition": {
                    "type": "object_in_container",
                    "object": "cup",
                    "container": "tray",
                },
                "motion_policy": motion_policy(),
            },
        ],
        "task_groups": [
            {
                "id": "e1_001",
                "task_type": "E1",
                "role": "primary",
                "operator": "place_relative",
                "object_uid": "cup",
                "actor": {"mode": "auto"},
                "goal": {"relation": "inside", "reference_object": "tray"},
                "depends_on": [],
                "node_ids": ["pick_cup", "place_cup"],
                "success": {
                    "type": "object_in_container",
                    "object": "cup",
                    "container": "tray",
                },
            }
        ],
        "success": {
            "type": "object_in_container",
            "object": "cup",
            "container": "tray",
        },
        "capability_catalog_hash": registry.catalog_hash(),
        "metadata": {},
    }
    return link_seed_graph(
        draft,
        registry=registry,
        task_order=["e1_001"],
        known_objects={"cup", "tray"},
    )


def test_seed_graph_validates_direct_atomic_action_nodes() -> None:
    registry = build_atomic_capability_registry()
    graph = validate_seed_graph(
        _seed_graph(),
        known_objects={"cup", "tray"},
        known_actions=registry.names(),
        executable_actions=registry.executable_names(),
        require_executable=True,
    )
    assert [node["atomic_action"] for node in graph["nodes"]] == ["PickUp", "Place"]
    assert graph["task_groups"][0]["node_ids"] == ["pick_cup", "place_cup"]


def test_seed_graph_rejects_grounded_motion_and_cycles() -> None:
    grounded = _seed_graph()
    grounded["nodes"][0]["target_binding"]["target_pose"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="grounded motion data"):
        validate_seed_graph(grounded)

    cyclic = _seed_graph()
    cyclic["nodes"][0]["depends_on"] = ["place_cup"]
    with pytest.raises(ValueError, match="dependency cycle"):
        validate_seed_graph(cyclic)


def test_seed_graph_rejects_unknown_uids_illegal_groups_and_resource_conflicts() -> (
    None
):
    with pytest.raises(ValueError, match="unknown object"):
        validate_seed_graph(_seed_graph(), known_objects={"tray"})

    illegal_group = _seed_graph()
    illegal_group["task_groups"][0]["task_type"] = "E9"
    for node in illegal_group["nodes"]:
        node["task_type"] = "E9"
    with pytest.raises(ValueError, match="core actions"):
        validate_seed_graph(illegal_group)

    conflicting = _seed_graph()
    conflicting["nodes"][1]["depends_on"] = []
    conflicting["task_groups"][0]["contract"]["entry_node_ids"] = [
        "pick_cup",
        "place_cup",
    ]
    conflicting["task_groups"][0]["contract"]["terminal_node_ids"] = [
        "pick_cup",
        "place_cup",
    ]
    with pytest.raises(ValueError, match="resource conflicts"):
        validate_seed_graph(conflicting)


def test_seed_graph_hash_is_order_stable_and_detached() -> None:
    graph = _seed_graph()
    original = deepcopy(graph)
    first = seed_graph_hash(graph)
    second = seed_graph_hash({key: graph[key] for key in reversed(graph)})
    assert first == second
    assert graph == original


def test_task_spec_enforces_reasoning_level_and_repetition_shape() -> None:
    spec = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "upright-cans",
        "level": "L2",
        "instruction": "Stand both cans upright.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "e2_1",
                "task_type": "E2",
                "params": {"object_role": "can_1"},
                "depends_on": [],
            },
            {
                "id": "e2_2",
                "task_type": "E2",
                "params": {"object_role": "can_2"},
                "depends_on": [],
            },
        ],
        "success": {"type": "all_upright"},
        "oracle": {"object_roles": ["can_1", "can_2"]},
        "metadata": {},
    }
    assert validate_task_spec(spec)["level"] == "L2"
    spec["level"] = "L4"
    with pytest.raises(ValueError, match="non-'none'"):
        validate_task_spec(spec)


def test_public_l4_task_hides_oracle_and_reference_instances() -> None:
    spec = {
        "schema_version": TASK_SPEC_SCHEMA,
        "task_id": "complete-mouth",
        "level": "L4",
        "instruction": "Complete the missing mouth.",
        "reasoning_type": "visual_semantics",
        "task_instances": [
            {
                "id": "hidden_e1",
                "task_type": "E1",
                "params": {"object_role": "mouth", "target_role": "face"},
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "visual_part_complete"},
        "oracle": {"missing_part": "mouth"},
        "metadata": {},
    }

    public = public_task_spec(spec)

    assert "oracle" not in public
    assert "task_instances" not in public
    assert validate_public_task_spec(public) == public


def test_scene_requirements_validate_task_first_handoff() -> None:
    requirements = validate_scene_requirements(
        {
            "schema_version": SCENE_REQUIREMENTS_SCHEMA,
            "task_id": "upright-cans",
            "objects": [
                {
                    "role_id": "can",
                    "category": "can",
                    "count": 2,
                    "affordances": ["graspable", "orientable"],
                    "initial_state": {"orientation": "fallen"},
                    "attributes": {},
                }
            ],
            "cameras": [{"role": "overview", "requires_rgb": True}],
            "spatial_constraints": [],
            "distractor_count": 0,
            "metadata": {},
        }
    )
    assert requirements["objects"][0]["count"] == 2


def test_planning_only_capability_is_rejected_before_execution() -> None:
    registry = build_atomic_capability_registry()
    graph = _seed_graph()
    graph["nodes"][0]["atomic_action"] = "Pour"
    graph["nodes"][0]["target_binding"] = {
        "kind": "pour_goal",
        "object": "cup",
        "reference_object": "tray",
    }
    with pytest.raises(ValueError, match="planning-only"):
        validate_seed_graph(
            graph,
            known_actions=registry.names(),
            executable_actions=registry.executable_names(),
            require_executable=True,
        )
