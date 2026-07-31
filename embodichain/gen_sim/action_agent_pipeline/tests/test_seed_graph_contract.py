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
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.domain.seed_task_graph import (
    validate_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    _build_executable_seed,
    make_relative_seed_task_graph,
    make_stacking_seed_task_graph,
)


def test_executable_seed_build_does_not_mutate_semantic_input() -> None:
    semantic_steps = [_relative_semantic_step()]
    original = deepcopy(semantic_steps)

    graph = _build_executable_seed(
        task_name="immutable_input",
        route="object_manipulation",
        program="place_relative",
        semantic_steps=semantic_steps,
    )

    assert semantic_steps == original
    assert graph["semantic_steps"][0]["edge_ids"]


def test_object_manipulation_validator_rejects_route_invalid_operator() -> None:
    seed = make_relative_seed_task_graph("relative", _relative_spec())
    seed["semantic_steps"][0]["operator"] = "place_on_stack"

    with pytest.raises(ValueError, match="invalid operator"):
        validate_seed_task_graph(seed)


def test_stacking_validator_rejects_broken_support_chain() -> None:
    seed = make_stacking_seed_task_graph("stack", _stacking_spec())
    seed["semantic_steps"][1]["goal"]["reference_object"] = "basket"
    seed["semantic_steps"][1]["postcondition"]["reference_object"] = "basket"

    with pytest.raises(ValueError, match="prior layer"):
        validate_seed_task_graph(seed)


def test_task_text_alone_does_not_enable_parallel_pickup() -> None:
    spec = _relative_spec(
        second_step=True,
        auto_arms=True,
        parallel_pickup_requested=False,
    )
    spec.task_description = "Use both arms in parallel."

    seed = make_relative_seed_task_graph("structured_parallelism", spec)

    assert seed["allocation_groups"] == []


def _relative_semantic_step() -> dict:
    return {
        "id": "s01_place_object",
        "operator": "place_relative",
        "object": "object",
        "actor": {"mode": "required", "arm": "left_arm"},
        "goal": {
            "relation": "inside",
            "reference_object": "basket",
            "reference_state": "live",
            "orientation_goal": "preserve",
            "orientation_axis": "none",
        },
        "depends_on": [],
        "postcondition": {
            "type": "semantic_goal",
            "operator": "place_relative",
            "relation": "inside",
        },
    }


def _relative_spec(
    *,
    second_step: bool = False,
    auto_arms: bool = False,
    parallel_pickup_requested: bool = False,
) -> SimpleNamespace:
    def placement(
        step_id: str,
        object_uid: str,
        depends_on: tuple[str, ...],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            intent="place_relative",
            moved_runtime_uid=object_uid,
            reference_runtime_uid="basket",
            relation="inside",
            reference_is_initial_pose=False,
            orientation_goal="preserve",
            orientation_axis="none",
            orientation_align_to_runtime_uid=None,
            arm_request="auto" if auto_arms else "left",
            step_id=step_id,
            depends_on=depends_on,
        )

    placements = [placement("s01_object_a", "object_a", ())]
    if second_step:
        placements.append(placement("s02_object_b", "object_b", ("s01_object_a",)))
    return SimpleNamespace(
        intent="place_relative",
        task_description="Move the objects.",
        placements=tuple(placements),
        coordinated_direction=None,
        coordinated_terminal_behavior=None,
        parallel_pickup_requested=parallel_pickup_requested,
    )


def _stacking_spec() -> SimpleNamespace:
    return SimpleNamespace(
        stack_mode="on_top",
        anchor_runtime_uid="basket",
        steps=(
            SimpleNamespace(
                runtime_uid="object_a",
                support_runtime_uid="basket",
                layer_index=0,
                orientation_goal="preserve",
                orientation_axis="none",
            ),
            SimpleNamespace(
                runtime_uid="object_b",
                support_runtime_uid="object_a",
                layer_index=1,
                orientation_goal="preserve",
                orientation_axis="none",
            ),
        ),
    )
