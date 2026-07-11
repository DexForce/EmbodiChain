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

from types import SimpleNamespace

import pytest

from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_args import build_parser
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_dual_ur5_robot_config,
    make_dual_ur_dh_pgi_robot_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _make_observations_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _DUAL_FRANKA_TABLETOP_CLEARANCE,
)
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_basket_basic_background,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    available_robot_profile_choices,
    available_robot_profile_ids,
    resolve_robot_profile,
)

_DUAL_FRANKA_HOME_QPOS = [
    0.0,
    0.0,
    -0.569,
    -0.569,
    0.0,
    0.0,
    -2.810,
    -2.810,
    0.0,
    0.0,
    3.037,
    3.037,
    0.741,
    0.741,
    0.04,
    0.04,
    0.04,
    0.04,
]


def test_robot_profile_registry_exposes_default_and_switchable_profiles() -> None:
    assert DEFAULT_ROBOT_PROFILE_ID == "dual_ur5"
    assert set(available_robot_profile_ids()) == {
        "dual_ur3",
        "dual_ur5",
        "dual_ur10",
        "dual_franka",
    }
    assert "franka" in available_robot_profile_choices()
    assert "franka_v3" not in available_robot_profile_choices()
    assert resolve_robot_profile(None).id == "dual_ur5"
    assert resolve_robot_profile("ur10").id == "dual_ur10"
    assert resolve_robot_profile("panda").id == "dual_franka"

    with pytest.raises(ValueError, match="Unknown robot profile"):
        resolve_robot_profile("dual_unknown")
    with pytest.raises(ValueError, match="Unknown robot profile"):
        resolve_robot_profile("franka_v3")


def test_dual_ur_robot_templates_switch_arm_variant_without_mutating_ur5() -> None:
    ur5 = make_dual_ur5_robot_config(robot_init_z=0.42)
    ur3 = make_dual_ur_dh_pgi_robot_config(ur_type="ur3", robot_init_z=0.43)
    ur10 = make_dual_ur_dh_pgi_robot_config(ur_type="ur10", robot_init_z=0.44)

    assert ur5["uid"] == "DualUR5"
    assert ur5["init_pos"] == pytest.approx([-2.0, 0.0, 0.42])
    assert ur5["solver_cfg"]["left_arm"]["ur_type"] == "ur5"
    assert ur5["drive_pros"]["max_effort"]["left_arm"] == pytest.approx(100000.0)

    assert ur3["uid"] == "DualUR3"
    assert ur3["init_pos"] == pytest.approx([-2.0, 0.0, 0.43])
    assert ur3["solver_cfg"]["right_arm"]["ur_type"] == "ur3"
    assert ur3["drive_pros"]["max_effort"]["right_arm"] == pytest.approx(56.0)
    assert _arm_urdf_paths(ur3) == {
        "UniversalRobots/UR3/UR3.urdf",
    }

    assert ur10["uid"] == "DualUR10"
    assert ur10["init_pos"] == pytest.approx([-2.0, 0.0, 0.44])
    assert ur10["solver_cfg"]["left_arm"]["ur_type"] == "ur10"
    assert ur10["drive_pros"]["max_effort"]["left_arm"] == pytest.approx(330.0)
    assert _arm_urdf_paths(ur10) == {
        "UniversalRobots/UR10/UR10.urdf",
    }


def test_dual_ur_robotiq_tcp_rotates_grasp_x_onto_gripper_y_axis() -> None:
    robot = make_dual_ur5_robot_config(robot_init_z=0.42)

    assert robot["solver_cfg"]["left_arm"]["tcp"] == [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.16],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert robot["solver_cfg"]["right_arm"]["tcp"] == [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.21],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_dual_franka_profile_defines_robot_runtime_and_observation_contracts() -> None:
    profile = resolve_robot_profile("franka")
    robot = profile.robot_config_factory(0.45)
    extensions = profile.runtime_extensions()
    observations = _make_observations_config(robot)

    assert robot["uid"] == "DualFrankaPanda"
    assert robot["init_pos"] == pytest.approx([0.7, 0.0, 0.45])
    assert profile.tabletop_clearance == pytest.approx(_DUAL_FRANKA_TABLETOP_CLEARANCE)
    assert _arm_urdf_paths(robot) == {"Franka/Panda/PandaWithHand.urdf"}
    assert robot["init_qpos"] == pytest.approx(_DUAL_FRANKA_HOME_QPOS)
    assert robot["control_parts"]["left_arm"] == [
        "left_fr3_joint1",
        "left_fr3_joint2",
        "left_fr3_joint3",
        "left_fr3_joint4",
        "left_fr3_joint5",
        "left_fr3_joint6",
        "left_fr3_joint7",
    ]
    assert robot["control_parts"]["right_arm"] == [
        "right_fr3_joint1",
        "right_fr3_joint2",
        "right_fr3_joint3",
        "right_fr3_joint4",
        "right_fr3_joint5",
        "right_fr3_joint6",
        "right_fr3_joint7",
    ]
    assert robot["control_parts"]["left_eef"] == ["left_fr3_finger_joint[1-2]"]
    assert robot["control_parts"]["right_eef"] == ["right_fr3_finger_joint[1-2]"]
    assert robot["solver_cfg"]["left_arm"]["class_type"] == "PytorchSolver"
    assert robot["solver_cfg"]["left_arm"]["root_link_name"] == "left_base"
    assert robot["solver_cfg"]["left_arm"]["end_link_name"] == "left_fr3_hand_tcp"
    assert robot["solver_cfg"]["right_arm"]["class_type"] == "PytorchSolver"
    assert robot["solver_cfg"]["right_arm"]["root_link_name"] == "right_base"
    assert robot["solver_cfg"]["right_arm"]["end_link_name"] == "right_fr3_hand_tcp"
    expected_tcp = [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert robot["solver_cfg"]["left_arm"]["tcp"] == expected_tcp
    assert robot["solver_cfg"]["right_arm"]["tcp"] == expected_tcp
    assert robot["qpos_limits"]["(left|right)_fr3_finger_joint[1-2]"] == [
        0.0,
        0.06,
    ]
    assert observations["norm_robot_eef_joint"]["params"]["joint_ids"] == [
        14,
        15,
        16,
        17,
    ]
    assert extensions["agent_robot_profile"] == "dual_franka"
    assert extensions["gripper_open_state"] == [0.06, 0.06]
    assert extensions["gripper_close_state"] == [0.0, 0.0]
    assert extensions["agent_grasp_runtime_defaults"]["grasp_finger_length"] == (
        pytest.approx(0.058)
    )


def test_prompts_and_cli_accept_robot_profile_aliases() -> None:
    args = build_parser().parse_args(["--robot-profile", "franka"])
    prompt = make_basket_basic_background(
        "demo_project",
        _basket_roles(),
        robot_profile=args.robot_profile,
    )

    assert args.robot_profile == "franka"
    assert "Dual Franka Panda" in prompt
    assert "dual-UR5" not in prompt


def _arm_urdf_paths(robot_config: dict) -> set[str]:
    return {
        component["urdf_path"]
        for component in robot_config["urdf_cfg"]["components"]
        if str(component["component_type"]).endswith("_arm")
    }


def _basket_roles() -> SimpleNamespace:
    return SimpleNamespace(
        left_target_runtime_uid="left_apple",
        right_target_runtime_uid="right_apple",
        container_runtime_uid="basket",
        left_target_source_uid="apple_1",
        right_target_source_uid="apple_2",
        container_source_uid="basket_1",
        left_target_noun="apple",
        right_target_noun="apple",
    )
