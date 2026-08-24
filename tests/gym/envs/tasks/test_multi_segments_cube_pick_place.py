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

"""Tests for the declarative multi-segment cube task."""

from __future__ import annotations

import json
from pathlib import Path

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)
from embodichain.lab.sim.robots import URRobotCfg

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.expert_program.repeated_pick_place import (  # noqa: E402
    ExpertProgramRepeatedPickPlaceEnv,
)
from embodichain_tasks.multi_segments import cube_pick_place as cube_task  # noqa: E402
from embodichain_tasks.multi_segments.cube_pick_place import (  # noqa: E402
    CUBE_ROBOT_PROFILE_ID,
    CUBE_SCENE_REGISTRY_ID,
    MultiSegmentsCubePickPlaceEnv,
    _create_default_env_cfg,
)


def _gym_config_path() -> Path:
    """Return the installed-source cube Gym config path."""
    return (
        Path(__file__).parents[4]
        / "embodichain_tasks/configs/gym/multi_segments/cube_pick_place.json"
    )


def _gym_payload() -> dict[str, object]:
    """Load the runnable Gym configuration as inert JSON data."""
    payload = json.loads(_gym_config_path().read_text(encoding="utf-8"))
    assert type(payload) is dict
    return payload


def test_registered_task_reuses_shared_expert_program_integration() -> None:
    """The historical task ID delegates to the production reference task."""
    from embodichain_tasks.multi_segments import __all__

    assert "MultiSegmentsCubePickPlaceEnv" in __all__
    spec = REGISTERED_ENVS["MultiSegmentsCubePickPlace-v1"]
    assert spec.cls is MultiSegmentsCubePickPlaceEnv
    assert spec.max_episode_steps == 1200
    assert issubclass(MultiSegmentsCubePickPlaceEnv, ExpertProgramRepeatedPickPlaceEnv)
    assert issubclass(MultiSegmentsCubePickPlaceEnv, EmbodiedEnv)
    assert "create_demo_segments" not in MultiSegmentsCubePickPlaceEnv.__dict__


def test_gym_config_selects_packaged_expert_program_and_contact_evidence() -> None:
    """Normal Gym startup selects the program and physical grasp sensor."""
    payload = _gym_payload()

    assert payload["id"] == "MultiSegmentsCubePickPlace-v1"
    assert payload["expert_program_path"] == (
        "../../expert_program/multi_segments/repeated_cube_pick_place.yaml"
    )
    assert payload["env"]["extensions"] == {}
    settle = payload["env"]["events"]["settle_cube_on_reset"]
    assert settle["func"] == "wait_for_dynamic_objects_to_settle"
    assert settle["mode"] == "reset"
    assert settle["params"]["entity_cfgs"] == [{"uid": "cube"}]
    sensor = payload["sensor"][0]
    assert sensor["sensor_type"] == "ContactSensor"
    assert sensor["uid"] == "grasp_contacts"
    assert sensor["rigid_uid_list"] == ["cube"]
    assert sensor["articulation_cfg_list"][0]["link_name_list"] == [
        "gripper_finger1_link_1",
        "gripper_finger2_link_1",
    ]


def test_gym_config_keeps_scene_and_robot_configuration() -> None:
    """The migration changes the expert layer, not the physical environment."""
    cfg = config_to_cfg(_gym_payload(), source_path=_gym_config_path())

    assert isinstance(cfg.robot, URRobotCfg)
    assert cfg.robot.robot_type == "ur5"
    assert cfg.robot.control_parts["arm"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    assert cfg.robot.control_parts["hand"] == ["gripper_finger1_joint_1"]
    assert cfg.rigid_object[0].uid == "cube"


def test_direct_default_cfg_loads_the_same_typed_program_and_sensor(
    monkeypatch,
) -> None:
    """Direct Python construction and Gym startup share one integration."""
    config_root = _gym_config_path().parents[2]
    monkeypatch.setattr(
        cube_task,
        "get_config_path",
        lambda relative_path: config_root / relative_path,
    )
    cfg = _create_default_env_cfg()

    assert cfg.expert_program is not None
    assert cfg.expert_program.integration.scene_registry == CUBE_SCENE_REGISTRY_ID
    assert cfg.expert_program.integration.robot_profile == CUBE_ROBOT_PROFILE_ID
    assert cfg.expert_program.program_id == "repeated_cube_pick_place"
    assert cfg.sensor[0].uid == "grasp_contacts"
    assert cfg.sensor[0].rigid_uid_list == ["cube"]
    settle = cfg.events["settle_cube_on_reset"]
    assert settle.func is not None
    assert settle.params["entity_cfgs"][0].uid == "cube"


def test_task_does_not_replace_reference_effect_observation() -> None:
    """The migrated task inherits physical two-finger grasp evidence unchanged."""
    assert "_observe_grasp_constraint" not in MultiSegmentsCubePickPlaceEnv.__dict__


__all__: list[str] = []
