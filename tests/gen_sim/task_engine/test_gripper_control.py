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

import numpy as np
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
def test_light_rigid_grip_limits_only_the_used_gripper(task_type: str) -> None:
    embodiment, scene, graph = _inputs(task_type=task_type)
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert embodiment["simulation"]["joint_drive_props"]["max_effort"] == {
        "left_arm": 10000.0,
        "left_eef": 0.5,
        "right_eef": 500.0,
    }
    assert "mimic_compliance" not in embodiment["simulation"]


@pytest.mark.parametrize("mass", [None, 0.02, 1.0])
def test_unqualified_payload_retains_authored_control(mass: float | None) -> None:
    embodiment, scene, graph = _inputs(mass=mass)
    before = deepcopy(embodiment)
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert embodiment == before


@pytest.mark.parametrize("task_type", ["E3", "E4", "E5", "E6"])
def test_other_recipe_families_retain_authored_control(task_type: str) -> None:
    embodiment, scene, graph = _inputs(task_type=task_type)
    before = deepcopy(embodiment)
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert embodiment == before


def test_lower_existing_effort_is_not_increased() -> None:
    embodiment, scene, graph = _inputs()
    embodiment["simulation"]["joint_drive_props"]["max_effort"]["left_eef"] = 0.2
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert (
        embodiment["simulation"]["joint_drive_props"]["max_effort"]["left_eef"] == 0.2
    )


def test_registered_pick_uses_its_bound_resource() -> None:
    embodiment, scene, graph = _inputs(task_type="E2")
    graph["nodes"][0]["call"] = {
        "kind": "registered",
        "call_id": "gen_sim.pick.upright",
        "arguments": {"object": "object"},
        "resources": {"primary": "right"},
    }
    _calibrate_task_gripper_effort(embodiment, scene, graph)
    assert (
        embodiment["simulation"]["joint_drive_props"]["max_effort"]["right_eef"] == 0.5
    )
    assert (
        embodiment["simulation"]["joint_drive_props"]["max_effort"]["left_eef"] == 500.0
    )


def test_on_placement_generates_support_checks_without_relaxing_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.task_engine import task_program_bundle as bundle
    from embodichain.gen_sim.task_engine._task_program.stability import (
        StabilityConstraint,
    )

    objects = [
        {
            "runtime_uid": uid,
            "init_pos": [0.0, 0.0, 0.7],
            "init_rot": [90.0, 0.0, 0.0],
            "init_local_pose": np.eye(4).tolist(),
        }
        for uid in ("cup", "cube")
    ]
    monkeypatch.setattr(
        bundle,
        "_mesh_vertices",
        lambda obj: np.array(
            [
                [-0.05, -0.05, 0.0],
                [0.05, 0.05, 0.1 if obj["runtime_uid"] == "cube" else 0.12],
            ]
        ),
    )
    graph = {
        "nodes": [
            {
                "id": "place",
                "task_type": "E1",
                "task_instance_id": "step",
                "call": {
                    "kind": "registered",
                    "call_id": "simulation.place_relative",
                    "arguments": {
                        "object": "cup",
                        "reference": "cube",
                        "relation": "on",
                    },
                },
            }
        ],
        "task_groups": [{"node_ids": ["place"]}],
    }
    embodiment = {
        "skill_profile": {
            "resources": [
                {
                    "resource_id": "left",
                    "endpoints": [
                        {"endpoint_id": "motion", "control_part": "left_arm"}
                    ],
                }
            ]
        }
    }
    payload = bundle._task_stability_payload(
        graph, SimpleNamespace(planner_objects=objects, table_top_z=0.7), embodiment
    )
    cfg = StabilityConstraint.decode(payload["presets"]["gen_sim.place.stable"])
    assert cfg.kind == "supported_placement"
    assert cfg.position_tolerance == 0.05
    assert cfg.local_axis is None and cfg.reference_axis is None
    assert cfg.reference_half_extents is None
    assert cfg.displacement == pytest.approx((0.0, 0.0, 0.11))
