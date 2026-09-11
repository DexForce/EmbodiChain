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


"""CPU checks for the bounded, observed upright TaskSpec evaluator."""

from __future__ import annotations

from copy import deepcopy
import math
import pytest
import torch

from embodichain.task_spec import semantic_hash
from embodichain.lab.task_evaluation import UprightTaskEvaluator


def template():
    value = {
        "schema_version": "taskspec/template/v0.1",
        "semantic_version": "0.1",
        "roles": {"item": {"kind": "object"}},
        "init": [],
        "invariants": [],
        "requirements": [],
        "goal": [
            {
                "predicate": "upright",
                "object": "item",
                "max_tilt": {"value": "0.1", "unit": "rad"},
            }
        ],
    }
    value["semantic_hash"] = semantic_hash(value)
    return value


def test_final_observation_rechecks_all_rows_independently():
    evaluator = UprightTaskEvaluator(template())
    poses = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    poses[1, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    result = evaluator.evaluate(poses, scope="task_goal")
    assert result["status"] == ["pass", "failed"]
    assert result["tilt_rad"] == pytest.approx([0, math.pi / 2])
    poses[0, :3, :3] = poses[1, :3, :3]
    assert evaluator.evaluate(poses, scope="task_goal")["status"] == [
        "failed",
        "failed",
    ]


def test_missing_or_invalid_observation_never_passes_even_under_not():
    value = template()
    value["init"] = [{"op": "not", "args": [deepcopy(value["goal"][0])]}]
    value["semantic_hash"] = semantic_hash(value)
    evaluator = UprightTaskEvaluator(value)
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[0, 0, 0] = float("nan")
    poses[1, :3, :3] = 0
    result = evaluator.evaluate(poses, scope="initial")
    assert result["status"] == ["unavailable", "unavailable"]
    assert result["tilt_rad"] == [None, None]


def test_unobserved_process_conditions_are_rejected_before_execution():
    value = template()
    value["invariants"] = deepcopy(value["goal"])
    value["semantic_hash"] = semantic_hash(value)
    with pytest.raises(ValueError, match="bounded"):
        UprightTaskEvaluator(value)


def test_small_float32_tilt_is_not_rounded_to_a_false_pass():
    value = template()
    value["goal"][0]["max_tilt"]["value"] = "0.0001"
    value["semantic_hash"] = semantic_hash(value)
    pose = torch.eye(4).unsqueeze(0)
    angle = 0.0002
    pose[0, :3, :3] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )
    result = UprightTaskEvaluator(value).evaluate(pose, scope="task_goal")
    assert result["status"] == ["failed"]
    assert result["tilt_rad"] == pytest.approx([angle])
