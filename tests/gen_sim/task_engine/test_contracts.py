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
import importlib.util

import pytest

from embodichain.gen_sim.task_engine import (
    FORBIDDEN_SEMANTIC_GRAPH_FIELDS,
    SemanticTaskGraph,
    TaskSpec,
    decode_semantic_task_graph,
    decode_task_spec,
    semantic_task_graph_hash,
    task_spec_hash,
)
from embodichain.lab.gym.envs.expert_program import (
    PickCfg,
    PlaceCfg,
    encode_semantic_call,
)


def _task_spec() -> dict[str, object]:
    return {
        "schema_version": "task_spec/v1",
        "task_id": "place_cube",
        "level": "L1",
        "instruction": "Place the cube on the tray",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "e1_0",
                "task_type": "E1",
                "params": {"object": "cube", "target": "tray", "relation": "on"},
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {
            "kind": "object_supported_by",
            "object": "cube",
            "support": "tray",
        },
        "oracle": {"reference": "offline"},
        "metadata": {"dataset": "unit-test"},
    }


def _semantic_graph() -> dict[str, object]:
    return {
        "schema_version": "semantic_task_graph/v1",
        "task_id": "place_cube",
        "instruction": "Place the cube on the tray",
        "planner_route": "selected",
        "integration_fingerprint": "a" * 64,
        "nodes": [
            {
                "id": "pick_cube",
                "call": {"kind": "pick", "object": "cube"},
                "depends_on": [],
                "task_instance_id": "e1_0",
                "task_type": "E1",
                "role": "primary",
            },
            {
                "id": "place_cube",
                "call": {
                    "kind": "place",
                    "object": "cube",
                    "on": "tray",
                },
                "depends_on": ["pick_cube"],
                "task_instance_id": "e1_0",
                "task_type": "E1",
                "role": "primary",
            },
        ],
        "task_groups": [
            {
                "id": "e1_0",
                "task_type": "E1",
                "node_ids": ["pick_cube", "place_cube"],
                "depends_on": [],
                "success": {
                    "kind": "object_supported_by",
                    "object": "cube",
                    "support": "tray",
                },
                "role": "primary",
                "failure_policy": {
                    "max_attempts": 2,
                    "alternate_task_group_ids": [],
                },
            }
        ],
        "success": {"kind": "all_task_groups"},
        "planner_provenance": {
            "planner_id": "deterministic_recipe",
            "revision": "v1",
            "confidence": 1.0,
            "metadata": {"candidate_id": "offline_0"},
        },
        "metadata": {"candidate_rank": 0},
    }


def test_task_spec_is_immutable_json_safe_and_stably_hashed() -> None:
    source = _task_spec()
    task_spec = decode_task_spec(source)
    source["metadata"] = {"mutated": True}

    assert type(task_spec) is TaskSpec
    assert task_spec.metadata == {"dataset": "unit-test"}
    assert task_spec.to_public_dict().get("oracle") is None
    assert decode_task_spec(task_spec.to_dict()) == task_spec
    assert task_spec_hash(task_spec) == task_spec_hash(task_spec.to_dict())
    with pytest.raises(TypeError):
        task_spec.metadata["mutated"] = True  # type: ignore[index]


def test_semantic_graph_round_trips_through_the_canonical_call_codec() -> None:
    graph = decode_semantic_task_graph(_semantic_graph())

    assert type(graph) is SemanticTaskGraph
    assert type(graph.nodes[0].call) is PickCfg
    assert type(graph.nodes[1].call) is PlaceCfg
    assert encode_semantic_call(graph.nodes[1].call) == {
        "kind": "place",
        "object": "cube",
        "on": "tray",
    }
    assert decode_semantic_task_graph(graph.to_dict()) == graph


def test_semantic_graph_owns_call_payloads_and_returns_fresh_configs() -> None:
    source = _semantic_graph()
    graph = decode_semantic_task_graph(source)
    source["nodes"][0]["call"]["object"] = "mutated"  # type: ignore[index]
    first_call = graph.nodes[0].call
    assert type(first_call) is PickCfg
    first_call.object = "mutated"

    assert graph.nodes[0].call.object == "cube"


def test_semantic_graph_hash_is_key_order_independent() -> None:
    source = _semantic_graph()
    reordered = {key: source[key] for key in reversed(tuple(source))}

    assert semantic_task_graph_hash(source) == semantic_task_graph_hash(reordered)


@pytest.mark.parametrize("field_name", sorted(FORBIDDEN_SEMANTIC_GRAPH_FIELDS))
def test_semantic_graph_rejects_forbidden_fields_recursively(
    field_name: str,
) -> None:
    source = _semantic_graph()
    source["metadata"] = {"audit": [{"nested": {field_name: "leak"}}]}

    with pytest.raises(ValueError, match="forbidden"):
        decode_semantic_task_graph(source)


def test_task_spec_rejects_grounded_execution_fields_recursively() -> None:
    source = _task_spec()
    source["task_instances"][0]["params"] = {  # type: ignore[index]
        "nested": {"qpos": [0.0]}
    }

    with pytest.raises(ValueError, match="forbidden"):
        decode_task_spec(source)


def test_semantic_graph_rejects_noncanonical_call_payload() -> None:
    source = _semantic_graph()
    source["nodes"][0]["call"]["unexpected"] = True  # type: ignore[index]

    with pytest.raises(Exception, match="unexpected|unknown"):
        decode_semantic_task_graph(source)


def test_semantic_graph_rejects_invalid_fingerprint_and_task_group_membership() -> None:
    bad_fingerprint = _semantic_graph()
    bad_fingerprint["integration_fingerprint"] = "old-capability-hash"
    with pytest.raises(ValueError, match="SHA-256"):
        decode_semantic_task_graph(bad_fingerprint)

    missing_membership = deepcopy(_semantic_graph())
    missing_membership["task_groups"][0]["node_ids"] = ["pick_cube"]  # type: ignore[index]
    with pytest.raises(ValueError, match="missing TaskGroup membership"):
        decode_semantic_task_graph(missing_membership)


def test_semantic_graph_rejects_dependency_cycles() -> None:
    source = _semantic_graph()
    source["nodes"][0]["depends_on"] = ["place_cube"]  # type: ignore[index]

    with pytest.raises(ValueError, match="dependency cycle"):
        decode_semantic_task_graph(source)


def test_gen_sim_does_not_define_a_parallel_capability_registry() -> None:
    for module_name in ("atomic", "builtins", "registry"):
        assert (
            importlib.util.find_spec(
                f"embodichain.gen_sim.action_engine.capabilities.{module_name}"
            )
            is None
        )
