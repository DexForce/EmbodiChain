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

"""Deterministic contracts for the E3 approximate pouring workflow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.gen_sim.action_engine.domain import validate_seed_graph
from embodichain.gen_sim.action_engine.runtime.predicates import evaluate_predicate
from embodichain.gen_sim.action_engine.tasks import instantiate_seed_graph


def _task() -> dict:
    params = {
        "source_role": "source",
        "target_role": "target",
        "required_arm": "right_arm",
    }
    return {
        "schema_version": "action_engine_task_spec_v2",
        "task_id": "e3_single",
        "level": "L1",
        "instruction": "Pour from the source container into the target container.",
        "reasoning_type": "none",
        "task_instances": [
            {
                "id": "pour",
                "task_type": "E3",
                "params": params,
                "depends_on": [],
                "role": "primary",
            }
        ],
        "success": {"type": "poured"},
        "oracle": {},
        "metadata": {},
    }


def _graph() -> dict:
    return instantiate_seed_graph(
        _task(),
        {"source": "source_container", "target": "target_container"},
    )


def test_single_arm_mode_preserves_the_historical_auto_actor_contract() -> None:
    task = _task()
    task["task_instances"][0]["params"].pop("required_arm")

    graph = instantiate_seed_graph(
        task,
        {"source": "source_container", "target": "target_container"},
    )

    assert graph["task_groups"][0]["actor"] == {"mode": "auto"}
    assert all(node["actor"] == {"mode": "auto"} for node in graph["nodes"])
    assert [node["atomic_action"] for node in graph["nodes"]] == [
        "PickUp",
        "MoveHeldObject",
        "Pour",
        "MoveHeldObject",
        "Place",
        "MoveEndEffector",
        "MoveJoints",
    ]


def test_single_arm_recipe_binds_only_the_source_and_requested_arm() -> None:
    graph = _graph()

    assert all(node["object_uid"] == "source_container" for node in graph["nodes"])
    assert all(
        node["actor"] == {"mode": "required", "arm": "right_arm"}
        for node in graph["nodes"]
    )
    assert graph["task_groups"][0]["success"]["verification"] == "action_completion"


@pytest.mark.parametrize("field", ["pour_mode", "pouring_arm", "holding_arm"])
def test_single_arm_recipe_rejects_legacy_dual_arm_fields(field: str) -> None:
    task = _task()
    task["task_instances"][0]["params"][field] = "dual_arm"

    with pytest.raises(ValueError, match="Dual-arm E3 is not supported"):
        instantiate_seed_graph(
            task,
            {"source": "source_container", "target": "target_container"},
        )


def test_seed_graph_rejects_legacy_dual_arm_e3_goal() -> None:
    graph = _graph()
    graph["task_groups"][0]["goal"]["pour_mode"] = "dual_arm"

    with pytest.raises(ValueError, match="unsupported dual-arm E3 fields"):
        validate_seed_graph(graph)


def test_approximate_poured_predicate_needs_no_scene_or_content_state() -> None:
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
    )

    result = evaluate_predicate(
        env,
        {
            "type": "poured",
            "verification": "action_completion",
        },
    )

    assert result.tolist() == [True]
