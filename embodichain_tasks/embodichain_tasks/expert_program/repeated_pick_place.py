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

"""Repeated pick/place expressed by one declarative Expert Program."""

from __future__ import annotations

from typing import Any

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    AntipodalGraspAffordanceBinding,
    ExpertProgramEnvironmentAdapter,
    SimulationRigidObjectBinding,
    SimulationExpertProgramRegistration,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    create_simulation_expert_program_adapter,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import PickUpOptions, PlaceOptions
from embodichain.lab.sim.skills import SceneDynamics

from ._common import (
    HAND_CONTROL_PART,
    create_parallel_jaw_grasp_pose_generator,
    create_ur5_skill_profile_binding,
    load_bundled_expert_program,
)

__all__ = [
    "ExpertProgramRepeatedPickPlaceEnv",
    "REPEATED_PICK_PLACE_EXPERT_PROGRAM_REGISTRATION",
    "create_repeated_pick_place_robot_profile_binding",
    "create_repeated_pick_place_scene_binding",
]

ENV_ID = "ExpertProgramRepeatedPickPlace-v1"
PROGRAM_FILENAME = "repeated_pick_place.yaml"
CUBE_ENTITY_ID = "cube"
CUBE_GRASP_ENTITY_ID = "cube_grasp"
ROBOT_PROFILE_ID = "expert_program_ur5_pick_place"
SCENE_REGISTRY_ID = "expert_program_repeated_pick_place"
SAFE_MOTION_SAMPLE_COUNT = 120
DEFAULT_GRASP_SAMPLES = 10_000


def create_repeated_pick_place_scene_binding() -> SimulationSceneBinding:
    """Declare the provider-free cube scene integration."""
    return SimulationSceneBinding(
        registry_id=SCENE_REGISTRY_ID,
        rigid_objects=(
            SimulationRigidObjectBinding(
                entity_id=CUBE_ENTITY_ID,
                simulation_uid=CUBE_ENTITY_ID,
                dynamics=SceneDynamics.DYNAMIC,
                semantic_type="cube",
                default_grasp_affordance=CUBE_GRASP_ENTITY_ID,
            ),
        ),
        antipodal_grasps=(
            AntipodalGraspAffordanceBinding(
                entity_id=CUBE_GRASP_ENTITY_ID,
                object_id=CUBE_ENTITY_ID,
                native_name="cube_mesh",
                revision="cube-mesh-v1",
            ),
        ),
    )


def create_repeated_pick_place_robot_profile_binding() -> (
    SimulationRobotSkillProfileBinding
):
    """Declare the provider-free UR5 Pick/Place profile."""
    return create_ur5_skill_profile_binding(
        None,
        profile_id=ROBOT_PROFILE_ID,
        sample_count=SAFE_MOTION_SAMPLE_COUNT,
        skill_ids=("pick_up", "place"),
        action_option_templates={
            "pick": PickUpOptions(),
            "place": PlaceOptions(),
        },
    )


REPEATED_PICK_PLACE_EXPERT_PROGRAM_REGISTRATION = SimulationExpertProgramRegistration(
    scene_binding=create_repeated_pick_place_scene_binding(),
    robot_profile_binding=create_repeated_pick_place_robot_profile_binding(),
)


@register_env(
    ENV_ID,
    max_episode_steps=1200,
    expert_program_registration=REPEATED_PICK_PLACE_EXPERT_PROGRAM_REGISTRATION,
)
class ExpertProgramRepeatedPickPlaceEnv(EmbodiedEnv):
    """Run three declarative Pick/Place cycles through the semantic runtime.

    The bundled YAML program owns repetition, cyclic targets, settling, and
    segment validation. This class only declares the live scene, robot, and
    grasp-planning bindings.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize the scene bindings and production Expert Program adapter.

        Args:
            cfg: Environment configuration. The bundled program is installed
                when ``cfg.expert_program`` is unset.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.
        """
        if cfg.expert_program is None:
            cfg.expert_program = load_bundled_expert_program(PROGRAM_FILENAME)

        super().__init__(cfg, **kwargs)

        grasp_pose_generator = create_parallel_jaw_grasp_pose_generator(
            sample_count=DEFAULT_GRASP_SAMPLES,
            opening_margin=0.002,
        )
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            registration=REPEATED_PICK_PLACE_EXPERT_PROGRAM_REGISTRATION,
            grasp_pose_generators={HAND_CONTROL_PART: grasp_pose_generator},
        )

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the exact production adapter used by ``EmbodiedEnv``.

        Returns:
            Adapter that compiles and executes the configured Expert Program.
        """
        return self._expert_program_adapter
