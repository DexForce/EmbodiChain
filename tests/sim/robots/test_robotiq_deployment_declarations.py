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

"""Explicit model deployment declarations retain source calibration and input ownership."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml
from embodichain_robots.robotiq.deployment import (
    source_robotiq_opening_declaration,
    source_contact_pad_declaration,
    source_robot_mount_declaration,
    source_grip_effort_declaration,
    source_ur10_idle_posture_declaration,
)


def component(name):
    path = (
        Path(__file__).parents[3]
        / "extensions/embodichain_tasks/configs/components/embodiments"
        / f"{name}_robotiq_arg2f_140.yaml"
    )
    return yaml.safe_load(path.read_text())


def test_source_opening_effort_and_contact_are_explicit_owned_overrides():
    value = component("dual_franka")
    before = deepcopy(value)
    generators = value["skill_profile"]["runtime_services"]["grasp_pose_generators"]
    next(iter(generators.values()))["model"]["max_opening_width"] = 0.1
    selected = source_robotiq_opening_declaration(value)
    assert [
        x["model"]["max_opening_width"]
        for x in selected["skill_profile"]["runtime_services"][
            "grasp_pose_generators"
        ].values()
    ] == [0.1, 0.128]
    capped = source_grip_effort_declaration(selected, {"left": 0.5})
    assert capped["simulation"]["joint_drive_props"]["max_effort"]["left_eef"] == 0.5
    assert (
        capped["simulation"]["joint_drive_props"]["max_effort"]["right_eef"]
        == before["simulation"]["joint_drive_props"]["max_effort"]["right_eef"]
    )
    contact = source_contact_pad_declaration(capped)
    assert contact["simulation"]["link_attrs"]["gen_sim_contact_pads"] == {
        "link_names_expr": [".*inner_finger_pad"],
        "attrs": {
            "material_props": {"static_friction": 2.85, "dynamic_friction": 2.70}
        },
    }
    assert contact["simulation"]["asset_physics_mode"] == "overlay"
    assert "gen_sim_contact_pads" not in capped["simulation"].get("link_attrs", {})
    with pytest.raises(ValueError, match="already declared"):
        source_contact_pad_declaration(contact)
    assert value["simulation"] == before["simulation"]


@pytest.mark.parametrize(
    "name,expected",
    [("dual_franka", [-0.7, 0.0, 0.38]), ("dual_ur10", [0.55, 0.0, 0.78])],
)
def test_bench_mount_uses_qualified_model_calibration_without_mutating_component(
    name, expected
):
    value = component(name)
    before = deepcopy(value)
    mounted = source_robot_mount_declaration(value, table_top_z=0.73)
    np.testing.assert_allclose(mounted["simulation"]["init_pos"], expected, atol=1e-15)
    assert value == before


def test_ur10_idle_seed_and_both_park_postures_preserve_source_selection():
    value = component("dual_ur10")
    before = deepcopy(value)
    selected, record = source_ur10_idle_posture_declaration(
        value, active_resource_id="right"
    )
    changes = [
        i
        for i, (a, b) in enumerate(
            zip(value["simulation"]["init_qpos"], selected["simulation"]["init_qpos"])
        )
        if a != b
    ]
    assert changes == [2]
    assert selected["simulation"]["init_qpos"][2] == pytest.approx(
        value["simulation"]["init_qpos"][2] - 0.5
    )
    old = {x["preset_id"]: x for x in value["skill_profile"]["command_presets"]}
    new = {x["preset_id"]: x for x in selected["skill_profile"]["command_presets"]}
    for name in ("left_arm_postures", "right_arm_postures"):
        assert new[name]["commands"]["park"][1] == pytest.approx(
            old[name]["commands"]["park"][1] - 0.5
        )
    assert record["tucked_initial_resource"] == "left"
    assert record["physical_clearance_certified"] is False
    assert value == before


def test_explicit_arm_stiffness_override_preserves_hand_drives():
    from embodichain_robots.robotiq.deployment import source_arm_stiffness_declaration

    value = component("dual_franka")
    before = deepcopy(value)
    selected = source_arm_stiffness_declaration(value, stiffness=50000.0)
    gains = selected["simulation"]["joint_drive_props"]["stiffness"]
    assert gains["left_arm"] == gains["right_arm"] == 50000.0
    assert (
        gains["left_eef"]
        == before["simulation"]["joint_drive_props"]["stiffness"]["left_eef"]
    )
    assert (
        gains["right_eef"]
        == before["simulation"]["joint_drive_props"]["stiffness"]["right_eef"]
    )
    assert value == before


def test_source_lateral_hints_retain_none_and_explicit_component_pose():
    from embodichain_robots.robotiq.deployment import source_robot_lateral_positions

    franka = "dual_franka_robotiq_arg2f_140"
    ur10 = "dual_ur10_robotiq_arg2f_140"
    assert source_robot_lateral_positions(franka) is None
    assert source_robot_lateral_positions(franka, component("dual_franka")) is None
    assert source_robot_lateral_positions(ur10) is None
    declared = component("dual_ur10")
    result = source_robot_lateral_positions(ur10, declared)
    assert result == pytest.approx({"left": -0.3, "right": 0.3})
    pose = np.eye(4)
    pose[1, 3] = 0.2
    declared["simulation"]["init_local_pose"] = pose.tolist()
    assert source_robot_lateral_positions(ur10, declared) == pytest.approx(
        {"left": 0.5, "right": -0.1}
    )


@pytest.mark.parametrize(
    "model",
    [
        "dual_franka_robotiq_arg2f_140",
        "dual_ur10_robotiq_arg2f_140",
    ],
)
def test_source_robot_preflight_constructs_no_simulator_or_robot(monkeypatch, model):
    from embodichain_robots.robotiq.deployment import require_source_robot_assets
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.objects import Robot

    monkeypatch.setattr(
        SimulationManager,
        "__init__",
        lambda *a, **k: pytest.fail("preflight created simulation"),
    )
    monkeypatch.setattr(
        Robot, "__init__", lambda *a, **k: pytest.fail("preflight created Robot")
    )
    require_source_robot_assets(model)


def test_explicit_deployment_profiles_keep_six_frozen_complete_declaration_hashes():
    import hashlib
    import json

    frozen = json.loads(
        (Path(__file__).parent / "data/robotiq_task_declarations_489.json").read_text()
    )
    from embodichain_robots.task_program import TaskGripEffortCandidate
    from embodichain_robots.robotiq.deployment import source_grip_effort_caps

    for case in frozen["cases"]:
        value = source_robotiq_opening_declaration(component(case["model"]))
        if case["profile"] == "contact_experiment":
            value = source_contact_pad_declaration(value)
        value = source_robot_mount_declaration(value, table_top_z=frozen["table_top_z"])
        caps = source_grip_effort_caps(
            value,
            [TaskGripEffortCandidate("left", True, False, (0.005,))],
            runtime_profile=case["profile"],
        )
        assert caps == case["caps"]
        value = source_grip_effort_declaration(value, caps)
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        assert (
            hashlib.sha256(payload.encode()).hexdigest() == case["declaration_sha256"]
        ), (case["model"], case["profile"])


@pytest.mark.parametrize(
    "masses,eligible,pouring,profile,expected",
    [
        ((0.01,), True, False, "default", {"left": 0.5}),
        ((0.010001,), True, False, "default", {}),
        ((None,), True, False, "default", {}),
        ((False,), True, False, "default", {}),
        ((0.005,), False, True, "composition_experiment", {}),
        ((0.005,), True, True, "composition_experiment", {"left": 5.0}),
    ],
)
def test_model_owned_grip_calibration_requires_qualified_neutral_payloads(
    masses, eligible, pouring, profile, expected
):
    from embodichain_robots.task_program import TaskGripEffortCandidate
    from embodichain_robots.robotiq.deployment import source_grip_effort_caps

    candidate = TaskGripEffortCandidate("left", eligible, pouring, masses)
    assert (
        source_grip_effort_caps(
            component("dual_franka"), [candidate], runtime_profile=profile
        )
        == expected
    )


def test_source_calibration_rejects_unknown_profiles_and_duplicate_resources():
    from embodichain_robots.task_program import TaskGripEffortCandidate
    from embodichain_robots.robotiq.deployment import source_grip_effort_caps

    candidate = TaskGripEffortCandidate("left", True, False, (0.005,))
    with pytest.raises(ValueError, match="explicit supported profile"):
        source_grip_effort_caps(
            component("dual_franka"), [candidate], runtime_profile="unknown"
        )
    with pytest.raises(ValueError, match="cannot repeat"):
        source_grip_effort_caps(
            component("dual_franka"), [candidate, candidate], runtime_profile="default"
        )
