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
from embodichain.gen_sim.task_engine.task_program_bundle import (
    _calibrate_task_gripper_effort,
)


def _inputs(
    mass: float | None = 0.01, task_type: str = "E1"
) -> tuple[dict, SimpleNamespace, dict]:
    embodiment = {
        "simulation": {
            "joint_drive_props": {
                "max_effort": {
                    "left_arm": 10000.0,
                    "left_eef": 500.0,
                    "right_eef": 500.0,
                }
            }
        },
        "skill_profile": {
            "defaults": {"pick_up": {"primary": "left"}},
            "resources": [
                {
                    "resource_id": "left",
                    "endpoints": [{"endpoint_id": "grasp", "control_part": "left_eef"}],
                },
                {
                    "resource_id": "right",
                    "endpoints": [
                        {"endpoint_id": "grasp", "control_part": "right_eef"}
                    ],
                },
            ],
        },
    }
    scene = SimpleNamespace(
        rigid_objects=({"uid": "object", "attrs": {"mass_props": {"mass": mass}}},)
    )
    graph = {
        "nodes": [
            {
                "task_type": task_type,
                "call": {
                    "kind": "pick",
                    "object": "object",
                    "resources": {"primary": "left"},
                },
            }
        ]
    }
    return embodiment, scene, graph


@pytest.mark.parametrize("task_type", ["E1", "E2"])
@pytest.mark.parametrize("resource", ["left", "right"])
def test_light_rigid_grip_limits_only_the_used_gripper(
    task_type: str, resource: str
) -> None:
    embodiment, scene, graph = _inputs(task_type=task_type)
    graph["nodes"][0]["call"]["resources"]["primary"] = resource
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert embodiment["simulation"]["joint_drive_props"]["max_effort"] == {
        "left_arm": 10000.0,
        "left_eef": 0.5 if resource == "left" else 500.0,
        "right_eef": 0.5 if resource == "right" else 500.0,
    }
    assert "mimic_compliance" not in embodiment["simulation"]


@pytest.mark.parametrize(
    "task_type,resource",
    [("E3", "left"), ("E3", "right"), ("E4", "left"), ("E5", "left"), ("E6", "left")],
)
def test_other_recipe_families_retain_authored_control(
    task_type: str, resource: str
) -> None:
    embodiment, scene, graph = _inputs(task_type=task_type)
    graph["nodes"][0]["call"]["resources"]["primary"] = resource
    before = deepcopy(embodiment)
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert embodiment == before
