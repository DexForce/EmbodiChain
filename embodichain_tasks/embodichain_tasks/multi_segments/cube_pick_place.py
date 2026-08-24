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

"""Declarative repeated cube pick-and-place environment.

The legacy task ID reuses the production repeated-pick/place Expert Program
integration. Its packaged program owns the three semantic cycles, cyclic
targets, settling, and validation; this module only preserves the historical
scene configuration and direct-construction entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramCfg,
    load_expert_program,
)
from embodichain.lab.gym.envs.managers import EventCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import (
    wait_for_dynamic_objects_to_settle,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.cfg import LightCfg, RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.lab.sim.sensors import (
    ArticulationContactFilterCfg,
    ContactSensorCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain_tasks.configs import get_config_path
from embodichain_tasks.expert_program._common import GRIPPER_FINGER_LINKS
from embodichain_tasks.expert_program.repeated_pick_place import (
    CONTACT_SENSOR_UID,
    ROBOT_PROFILE_ID,
    SCENE_REGISTRY_ID,
    ExpertProgramRepeatedPickPlaceEnv,
)

__all__ = ["MultiSegmentsCubePickPlaceEnv"]

CUBE_UID = "cube"
CUBE_SIZE = 0.05
CUBE_SCENE_REGISTRY_ID = SCENE_REGISTRY_ID
CUBE_ROBOT_PROFILE_ID = ROBOT_PROFILE_ID
CUBE_EXPERT_PROGRAM_PATH = Path(
    "expert_program/multi_segments/repeated_cube_pick_place.yaml"
)

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "gripper_finger1_joint_1"
GRIPPER_TCP_Z = 0.15


def _create_default_robot_cfg() -> URRobotCfg:
    """Create the UR5 scene embodiment used by the declarative task."""
    return URRobotCfg.from_dict(
        {
            "robot_type": "ur5",
            "uid": "UR5",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "hand",
                        "urdf_path": GRIPPER_URDF_PATH,
                    },
                ],
            },
            "control_parts": {"hand": [GRIPPER_HAND_JOINT_PATTERN]},
            "drive_pros": {
                "stiffness": {GRIPPER_HAND_JOINT_PATTERN: 1e3},
                "damping": {GRIPPER_HAND_JOINT_PATTERN: 1e2},
                "max_effort": {GRIPPER_HAND_JOINT_PATTERN: 1e4},
            },
            "solver_cfg": {
                "arm": {
                    "tcp": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, GRIPPER_TCP_Z],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            },
            "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
        }
    )


def _load_default_expert_program() -> ExpertProgramCfg:
    """Decode the packaged semantic program for direct instantiation."""
    return load_expert_program(get_config_path(CUBE_EXPERT_PROGRAM_PATH))


def _create_default_env_cfg() -> EmbodiedEnvCfg:
    """Create a directly-instantiable task configuration."""
    cfg = EmbodiedEnvCfg()
    cfg.max_episode_steps = 1200
    cfg.robot = _create_default_robot_cfg()
    cfg.light = EmbodiedEnvCfg.EnvLightCfg(
        direct=[
            LightCfg(
                uid="main_light",
                color=(0.6, 0.6, 0.6),
                intensity=30.0,
                init_pos=(1.0, 0.0, 3.0),
            )
        ]
    )
    cfg.rigid_object = [
        RigidObjectCfg(
            uid=CUBE_UID,
            shape=CubeCfg(size=[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
            attrs=RigidBodyAttributesCfg(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
                enable_ccd=True,
                linear_damping=0.2,
                angular_damping=0.2,
            ),
            max_convex_hull_num=16,
            init_pos=(-0.42, -0.08, 0.5 * CUBE_SIZE),
        )
    ]
    cfg.sensor = [
        ContactSensorCfg(
            uid=CONTACT_SENSOR_UID,
            rigid_uid_list=[CUBE_UID],
            articulation_cfg_list=[
                ArticulationContactFilterCfg(
                    articulation_uid="UR5",
                    link_name_list=list(GRIPPER_FINGER_LINKS),
                )
            ],
            filter_need_both_actor=True,
            max_contacts_per_env=64,
        )
    ]
    cfg.events = {
        "settle_cube_on_reset": EventCfg(
            func=wait_for_dynamic_objects_to_settle,
            mode="reset",
            params={
                "entity_cfgs": [SceneEntityCfg(uid=CUBE_UID)],
                "min_steps": 10,
                "max_steps": 120,
                "check_interval_steps": 2,
                "required_stable_checks": 3,
                "timeout_behavior": "raise",
            },
        )
    }
    cfg.expert_program = _load_default_expert_program()
    return cfg


@register_env("MultiSegmentsCubePickPlace-v1", max_episode_steps=1200)
class MultiSegmentsCubePickPlaceEnv(ExpertProgramRepeatedPickPlaceEnv):
    """Run the historical cube task through the shared semantic runtime."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        """Initialize the task from Gym config or the packaged defaults.

        Args:
            cfg: Environment configuration. Defaults to the historical UR5
                cube scene with the packaged Expert Program.
            **kwargs: Additional arguments forwarded to the shared reference
                environment.
        """
        if cfg is None:
            cfg = _create_default_env_cfg()
        if cfg.expert_program is None:
            cfg.expert_program = _load_default_expert_program()
        super().__init__(cfg, **kwargs)
