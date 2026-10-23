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
@pytest.mark.parametrize("profile,cap", [("default", 0.5), ("contact_experiment", 5.0)])
def test_light_rigid_grip_limits_only_the_used_gripper(
    task_type: str, resource: str, profile: str, cap: float
) -> None:
    embodiment, scene, graph = _inputs(task_type=task_type)
    graph["nodes"][0]["call"]["resources"]["primary"] = resource
    _calibrate_task_gripper_effort(embodiment, scene, graph, runtime_profile=profile)
    assert embodiment["simulation"]["joint_drive_props"]["max_effort"] == {
        "left_arm": 10000.0,
        "left_eef": cap if resource == "left" else 500.0,
        "right_eef": cap if resource == "right" else 500.0,
    }
    assert "mimic_compliance" not in embodiment["simulation"]


@pytest.mark.parametrize("limit", [0.25, 0.5, 2.0, 5.0])
def test_light_grip_calibration_never_raises_an_authored_lower_limit(
    limit: float,
) -> None:
    embodiment, scene, graph = _inputs()
    limits = embodiment["simulation"]["joint_drive_props"]["max_effort"]
    limits["left_eef"] = limit
    _calibrate_task_gripper_effort(
        embodiment, scene, graph, runtime_profile="contact_experiment"
    )
    assert limits["left_eef"] == limit


@pytest.mark.parametrize(
    "mass", [None, True, 0, -0.01, 0.0101, float("nan"), float("inf")]
)
def test_unqualified_payload_does_not_change_effort(mass: float | None) -> None:
    embodiment, scene, graph = _inputs(mass=mass)
    before = deepcopy(embodiment)
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert embodiment == before


@pytest.mark.parametrize(
    "task_type,resource",
    [("E4", "left"), ("E5", "left"), ("E6", "left")],
)
def test_other_recipe_families_retain_authored_control(
    task_type: str, resource: str
) -> None:
    embodiment, scene, graph = _inputs(task_type=task_type)
    graph["nodes"][0]["call"]["resources"]["primary"] = resource
    before = deepcopy(embodiment)
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert embodiment == before


@pytest.mark.parametrize("resource", ["left", "right"])
@pytest.mark.parametrize(
    "profile", ["default", "contact_experiment", "composition_experiment"]
)
def test_light_pour_caps_only_its_hand_including_empty_hand_press(
    resource: str, profile: str
) -> None:
    embodiment, scene, graph = _inputs(task_type="E3")
    graph["nodes"][0]["call"]["resources"]["primary"] = resource
    graph["nodes"].append(
        {
            "task_type": "E9",
            "call": {
                "kind": "registered",
                "call_id": "press",
                "resources": {"primary": resource},
            },
        }
    )
    _calibrate_task_gripper_effort(embodiment, scene, graph, runtime_profile=profile)
    limits = embodiment["simulation"]["joint_drive_props"]["max_effort"]
    assert limits[f"{resource}_eef"] == 5.0
    assert limits["right_eef" if resource == "left" else "left_eef"] == 500.0
    assert limits["left_arm"] == 10000.0


@pytest.mark.parametrize("mass", [None, True, 0.0101, float("inf")])
def test_unqualified_pour_keeps_canonical_effort(mass: float | None) -> None:
    embodiment, scene, graph = _inputs(mass=mass, task_type="E3")
    before = deepcopy(embodiment)
    _calibrate_task_gripper_effort(
        embodiment, scene, graph, runtime_profile="composition_experiment"
    )
    assert embodiment == before


@pytest.mark.parametrize("family", ["E4", "E6", "E8", "unknown"])
def test_pour_resource_with_unqualified_action_keeps_canonical_effort(
    family: str,
) -> None:
    embodiment, scene, graph = _inputs(task_type="E3")
    graph["nodes"].append(
        {
            "task_type": family,
            "call": {"kind": "registered", "resources": {"primary": "left"}},
        }
    )
    before = deepcopy(embodiment)
    _calibrate_task_gripper_effort(
        embodiment, scene, graph, runtime_profile="composition_experiment"
    )
    assert embodiment == before


def test_composition_without_pour_keeps_previously_qualified_effort() -> None:
    embodiment, scene, graph = _inputs()
    before = deepcopy(embodiment)
    _calibrate_task_gripper_effort(
        embodiment, scene, graph, runtime_profile="composition_experiment"
    )
    assert embodiment == before


@pytest.mark.parametrize("limit", [0.25, 0.5, 2.0, 5.0])
def test_pour_cap_preserves_lower_authored_effort(limit: float) -> None:
    embodiment, scene, graph = _inputs(task_type="E3")
    limits = embodiment["simulation"]["joint_drive_props"]["max_effort"]
    limits["left_eef"] = limit
    _calibrate_task_gripper_effort(
        embodiment, scene, graph, runtime_profile="composition_experiment"
    )
    assert limits["left_eef"] == limit


@pytest.mark.parametrize("heavy_owner", ["left", "right"])
def test_pour_cap_checks_all_payloads_on_its_own_hand(heavy_owner: str) -> None:
    embodiment, scene, graph = _inputs(task_type="E3")
    scene.rigid_objects += ({"uid": "heavy", "attrs": {"mass_props": {"mass": 1.0}}},)
    graph["nodes"].append(
        {
            "task_type": "E1",
            "call": {
                "kind": "pick",
                "object": "heavy",
                "resources": {"primary": heavy_owner},
            },
        }
    )
    _calibrate_task_gripper_effort(
        embodiment, scene, graph, runtime_profile="composition_experiment"
    )
    limits = embodiment["simulation"]["joint_drive_props"]["max_effort"]
    assert limits["left_eef"] == (500.0 if heavy_owner == "left" else 5.0)
    assert limits["right_eef"] == 500.0
