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

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    AntipodalGraspAffordanceBinding,
    ExpertProgramEnvironmentAdapter,
    SimulationRigidObjectBinding,
    SimulationSceneBinding,
    create_simulation_expert_program_adapter,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.sensors import ContactSensor
from embodichain.lab.sim.skills import (
    BinaryEffectEvidenceQuery,
    BinaryEffectObservation,
    EffectEvidenceCollectionContext,
    HeldObjectStateExpectation,
    SceneDynamics,
)

from ._common import (
    GRIPPER_FINGER_LINKS,
    HAND_CONTROL_PART,
    create_parallel_jaw_grasp_pose_generator,
    create_ur5_skill_profile_binding,
    load_bundled_expert_program,
)

__all__ = ["ExpertProgramRepeatedPickPlaceEnv"]

ENV_ID = "ExpertProgramRepeatedPickPlace-v1"
PROGRAM_FILENAME = "repeated_pick_place.yaml"
CUBE_ENTITY_ID = "cube"
CUBE_GRASP_ENTITY_ID = "cube_grasp"
CONTACT_SENSOR_UID = "grasp_contacts"
ROBOT_PROFILE_ID = "expert_program_ur5_pick_place"
SCENE_REGISTRY_ID = "expert_program_repeated_pick_place"
SAFE_MOTION_SAMPLE_COUNT = 120
DEFAULT_GRASP_SAMPLES = 10_000


@register_env(ENV_ID, max_episode_steps=1200)
class ExpertProgramRepeatedPickPlaceEnv(EmbodiedEnv):
    """Run three verified Pick/Place cycles through the semantic runtime.

    Unlike
    :class:`~embodichain_tasks.multi_segments.MultiSegmentsCubePickPlaceEnv`,
    this environment contains no task-local action planning or segment
    generator. The bundled YAML program owns repetition, cyclic targets,
    settling, and segment validation. This class only declares the live
    scene/robot bindings and supplies physical two-finger contact evidence for
    Pick/Place effect verification.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize the scene bindings and production Expert Program adapter.

        Args:
            cfg: Environment configuration. The bundled program is installed
                when ``cfg.expert_program`` is unset.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.

        Raises:
            TypeError: If the contact sensor is invalid.
            ValueError: If native actor IDs are invalid.
            RuntimeError: If the configured cube is absent from the live scene.
        """
        if cfg.expert_program is None:
            cfg.expert_program = load_bundled_expert_program(PROGRAM_FILENAME)

        super().__init__(cfg, **kwargs)

        cube = self.sim.get_rigid_object(CUBE_ENTITY_ID)
        if cube is None:
            raise RuntimeError(f"{ENV_ID} requires rigid object {CUBE_ENTITY_ID!r}.")
        contact_sensor = self.get_sensor(CONTACT_SENSOR_UID)
        if not isinstance(contact_sensor, ContactSensor):
            raise TypeError(f"Sensor {CONTACT_SENSOR_UID!r} must be a ContactSensor.")
        self._contact_sensor = contact_sensor
        self._cube_user_ids = self._reshape_user_ids(
            cube.get_user_ids(), owner=CUBE_ENTITY_ID
        )
        self._finger_user_ids = tuple(
            self._reshape_user_ids(self.robot.get_user_ids(link_name), owner=link_name)
            for link_name in GRIPPER_FINGER_LINKS
        )
        grasp_pose_generator = create_parallel_jaw_grasp_pose_generator(
            sample_count=DEFAULT_GRASP_SAMPLES,
            opening_margin=0.002,
        )
        scene_binding = SimulationSceneBinding(
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
        profile_binding = create_ur5_skill_profile_binding(
            self.robot,
            profile_id=ROBOT_PROFILE_ID,
            sample_count=SAFE_MOTION_SAMPLE_COUNT,
            skill_ids=("pick_up", "place"),
        )
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            scene_binding=scene_binding,
            robot_profile_binding=profile_binding,
            grasp_pose_generators={HAND_CONTROL_PART: grasp_pose_generator},
            constraint_observer=self._observe_grasp_constraint,
        )

    def _reshape_user_ids(self, value: object, *, owner: str) -> torch.Tensor:
        """Normalize native actor IDs to ``(num_envs, actors_per_env)``."""
        ids = torch.as_tensor(value, dtype=torch.long, device=self.device)
        if ids.numel() == 0 or ids.numel() % self.num_envs != 0:
            raise ValueError(
                f"Native user IDs for {owner!r} must divide across {self.num_envs} "
                "environments."
            )
        return ids.reshape(self.num_envs, -1)

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the exact production adapter used by ``EmbodiedEnv``.

        Returns:
            Adapter that compiles and executes the configured Expert Program.
        """
        return self._expert_program_adapter

    def _observe_grasp_constraint(
        self,
        query: BinaryEffectEvidenceQuery,
        context: EffectEvidenceCollectionContext,
    ) -> BinaryEffectObservation:
        """Derive a physical two-finger grasp constraint from contact pairs.

        A row is constrained only while the cube contacts both gripper finger
        links. The raw sensor is refreshed at the effect-evidence tick, so this
        signal is independent from accepted controller commands.
        """
        expectation = query.expectation
        if (
            not isinstance(expectation, HeldObjectStateExpectation)
            or expectation.object_id != CUBE_ENTITY_ID
        ):
            values = torch.zeros(
                context.env_ids.shape,
                dtype=torch.bool,
                device=context.env_ids.device,
            )
            return BinaryEffectObservation(
                values=values,
                valid=torch.zeros_like(values),
                acquisition_errors=(
                    f"{ENV_ID} only observes grasp constraints for "
                    f"{CUBE_ENTITY_ID!r}.",
                )
                * values.numel(),
            )

        self._contact_sensor.update()
        contact_data = self._contact_sensor.get_data()
        pairs = contact_data["user_ids"]
        valid_contacts = contact_data["is_valid"]
        row_ids = context.env_ids.to(device=self.device, dtype=torch.long)
        if bool(((row_ids < 0) | (row_ids >= self.num_envs)).any().item()):
            raise IndexError("Effect evidence requested an unknown environment row.")
        pairs = pairs.index_select(0, row_ids).to(dtype=torch.long)
        valid_contacts = valid_contacts.index_select(0, row_ids)
        values = torch.zeros(
            row_ids.shape,
            dtype=torch.bool,
            device=self.device,
        )
        for output_row, simulation_row in enumerate(row_ids.tolist()):
            row_pairs = pairs[output_row]
            row_valid = valid_contacts[output_row]
            cube_ids = self._cube_user_ids[simulation_row]
            finger_contacts: list[torch.Tensor] = []
            for finger_ids_by_env in self._finger_user_ids:
                finger_ids = finger_ids_by_env[simulation_row]
                cube_first = torch.isin(row_pairs[:, 0], cube_ids) & torch.isin(
                    row_pairs[:, 1], finger_ids
                )
                cube_second = torch.isin(row_pairs[:, 1], cube_ids) & torch.isin(
                    row_pairs[:, 0], finger_ids
                )
                finger_contacts.append(row_valid & (cube_first | cube_second))
            values[output_row] = all(
                bool(contact.any().item()) for contact in finger_contacts
            )
        return BinaryEffectObservation(values=values)
