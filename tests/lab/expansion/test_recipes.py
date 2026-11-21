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

"""Logical randomness is invariant to batching and unrelated operator fields."""

from __future__ import annotations
import json
from dataclasses import dataclass
from types import SimpleNamespace
import pytest
from embodichain.lab.expansion import PinnedExpertSource, UniformRangeOperator


def test_affordance_feedback_freezes_choices_and_keeps_exploring(tmp_path):
    from embodichain.lab.expansion.recipes import (
        BatchCompatibility,
        ExpansionRecipe,
        OperatorCapability,
        RecipeOperation,
    )
    from embodichain.lab.expansion.task_program_jobs import _task_expansion_extensions

    source = PinnedExpertSource("task_program", "v1")
    world = BatchCompatibility(
        "robot", ("j0",), "topology", "program", "default", 0.04, "obs", "world"
    )
    operation = RecipeOperation(
        "motion", "v1", OperatorCapability("motion", "plan", "environment"), "{}"
    )
    options = {
        "trajectory": {
            "affordance": {
                "branches": 4,
                "call_indices": [0],
                "branch_selection": "outcome_guided",
            }
        }
    }
    ports = _task_expansion_extensions(
        options=options, operator="affordance", seed=1, report={"recipe_revision": "v1"}
    )

    def recipe(ordinal):
        return ports.recipe_transform(
            ExpansionRecipe(source.context(1, ordinal), world, (operation,))
        )

    def branch(recipe):
        return json.loads(recipe.operations[0].payload_json)["affordance_branch"]

    def outcome(segments, samples=None):
        return SimpleNamespace(
            segments=([{"success": True}] * segments,),
            expansion_records=(({"samples": samples or {}},),),
        )

    first = [recipe(i) for i in range(3)]
    assert [branch(value) for value in first] == [1, 2, 3]
    ports.plan_outcome(2, False, outcome(1))
    ports.plan_outcome(0, False, outcome(1))
    ports.plan_outcome(1, False, outcome(3))
    assert [recipe(i) for i in range(3)] == first
    guided = recipe(3)
    assert branch(guided) == 2
    import numpy as np

    reference = np.eye(4)
    reference[0, 3] = 2
    grasp = reference.copy()
    grasp[0, 3] += 0.125
    sample = {
        "key": "pick:grasp",
        "pose_frame": "world",
        "success": [True],
        "reference_poses": [reference.tolist()],
        "selected_poses": [grasp.tolist()],
    }
    ports.plan_outcome(3, True, outcome(4, {"grasp": sample}))
    ports.plan_outcome(
        3, False, outcome(0)
    )  # A repeated notification cannot revoke the result.
    retry = recipe(4)
    assert branch(retry) == 2
    prior = json.loads(retry.operations[0].payload_json)["accepted_grasps"]
    assert prior["ordinal"] == 3 and prior["local_poses"]["pick:grasp"][3] == 0.125
    recipe(5)
    assert "accepted_grasps" not in json.loads(recipe(6).operations[0].payload_json)
    assert {branch(recipe(i)) for i in range(4, 17)} == {1, 2, 3}

    @dataclass
    class Context:
        batch_size: int = 1
        affordance_sampling: object = None

    row = SimpleNamespace(ordinal=4, directory=tmp_path, recipe=retry, scene_case=None)
    factory = ports.plan_transform_factory(None, row)
    prepared = factory.prepare_planning_context(
        SimpleNamespace(analysis_call_index=0), Context(), engine=None
    )
    assert prepared.affordance_sampling.episode_id == 4
    assert prepared.affordance_sampling.branch_override == 2
    assert (
        dict(prepared.affordance_sampling.preferred_local_poses)["pick:grasp"][3]
        == 0.125
    )


@pytest.mark.parametrize("batch", [1, 8, 32])
def test_random_streams_are_slot_and_completion_order_independent(batch):
    source = PinnedExpertSource("task_program", "source-v1")
    op = UniformRangeOperator("mass_scale", {"cube": (0.5, 2.0)})
    expected = [dict(op.sample(source.context(7, i))) for i in range(67)]
    actual = {}
    for start in range(0, 67, batch):
        for i in reversed(range(start, min(start + batch, 67))):
            actual[i] = dict(op.sample(source.context(7, i)))
    assert [actual[i] for i in range(67)] == expected
    assert len({r["cube"] for r in expected}) == 67
    other = UniformRangeOperator(
        "mass_scale", {"other": (0.0, 1.0), "cube": (0.5, 2.0)}
    )
    assert other.sample(source.context(7, 3))["cube"] == expected[3]["cube"]
    assert (
        op.sample(PinnedExpertSource("task_program", "source-v2").context(7, 3))["cube"]
        != expected[3]["cube"]
    )
    with pytest.raises(TypeError):
        op.sample(source.context(7, 0))["cube"] = 0
