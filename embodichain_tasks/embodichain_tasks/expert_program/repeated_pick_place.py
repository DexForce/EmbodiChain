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

from dataclasses import dataclass
from typing import Any, ClassVar, TYPE_CHECKING

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    AntipodalGraspAffordanceBinding,
    ControlPartEvidenceProviderFactory,
    ExpertProgramEnvironmentAdapter,
    SimulationRigidObjectBinding,
    SimulationExpertProgramRegistration,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    create_simulation_expert_program_adapter,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    PickUpOptions,
    PlaceOptions,
    SceneProvider,
)
from embodichain.lab.sim.skills import (
    BinaryEffectEvidenceQuery,
    BinaryEffectObservation,
    BinaryEvidenceKind,
    CONSTRAINT_EFFECT_CHANNEL,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    ControlPartEvidenceAddress,
    ControlPartSimulationEvidenceProvider,
    EffectEvidenceCollectionContext,
    EffectEvidenceProvider,
    HeldObjectStateExpectation,
    SceneDynamics,
    SceneRegistry,
    WorkflowRecoveryPolicy,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject, Robot
    from embodichain.lab.sim.sensors import ContactSensor

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
SAFE_MOTION_SAMPLE_COUNT = 100
DEFAULT_GRASP_SAMPLES = 10_000
CUBE_CONTACT_SENSOR_UID = "cube_grasp_contacts"
FINGER_LINK_NAMES = ("gripper_finger1_link_1", "gripper_finger2_link_1")


class _CubeFingerConstraintObserver:
    """Observe a cube constrained only while both gripper fingers contact it."""

    def __init__(
        self,
        sensor: ContactSensor,
        cube: RigidObject,
        robot: Robot,
    ) -> None:
        self._sensor = sensor
        self._cube = cube
        self._robot = robot

    @staticmethod
    def _selected_user_ids(
        values: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        field_name: str,
    ) -> torch.Tensor:
        """Normalize one simulator user-ID vector for contact matching."""
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{field_name} must be a torch.Tensor.")
        normalized = values.reshape(-1)
        if normalized.shape != (batch_size,):
            raise ValueError(f"{field_name} must contain one user ID per row.")
        return normalized.to(device=device)

    def __call__(
        self,
        query: BinaryEffectEvidenceQuery,
        context: EffectEvidenceCollectionContext,
    ) -> BinaryEffectObservation:
        """Return true for rows with cube contact on both finger links."""
        if type(query) is not BinaryEffectEvidenceQuery:
            raise TypeError("Cube grasp evidence requires a binary query.")
        if query.clause.evidence_kind is not BinaryEvidenceKind.CONSTRAINT:
            raise ValueError("Cube grasp evidence serves only constraint queries.")
        address = query.source.address
        if (
            type(address) is not ControlPartEvidenceAddress
            or address.control_part != HAND_CONTROL_PART
            or address.channel != CONSTRAINT_EFFECT_CHANNEL
        ):
            raise ValueError(
                "Cube grasp evidence requires the exact hand constraint route."
            )
        expectation = query.expectation
        if (
            type(expectation) is not HeldObjectStateExpectation
            or expectation.object_id != CUBE_ENTITY_ID
        ):
            raise ValueError(
                "Cube grasp evidence requires the cube held-object expectation."
            )

        data = self._sensor.get_data()
        user_ids = data["user_ids"]
        valid_contacts = data["is_valid"]
        if (
            not isinstance(user_ids, torch.Tensor)
            or user_ids.dim() != 3
            or user_ids.shape[-1] != 2
        ):
            raise ValueError("Contact user_ids must have shape (N, C, 2).")
        if (
            not isinstance(valid_contacts, torch.Tensor)
            or valid_contacts.dtype != torch.bool
            or valid_contacts.shape != user_ids.shape[:2]
            or valid_contacts.device != user_ids.device
        ):
            raise ValueError("Contact is_valid must match user_ids rows and contacts.")

        sensor_rows = context.env_ids.to(device=user_ids.device)
        if (
            bool((sensor_rows < 0).any())
            or int(sensor_rows.max().item()) >= user_ids.shape[0]
        ):
            raise ValueError("Evidence env_ids must address valid contact-sensor rows.")
        contacts = user_ids.index_select(0, sensor_rows)
        valid = valid_contacts.index_select(0, sensor_rows)
        env_ids = context.env_ids.detach().cpu().tolist()
        batch_size = int(context.env_ids.numel())
        cube_ids = self._selected_user_ids(
            self._cube.get_user_ids(env_ids),
            batch_size=batch_size,
            device=contacts.device,
            field_name="cube user IDs",
        )
        finger_ids = tuple(
            self._selected_user_ids(
                self._robot.get_user_ids(link_name, env_ids),
                batch_size=batch_size,
                device=contacts.device,
                field_name=f"{link_name} user IDs",
            )
            for link_name in FINGER_LINK_NAMES
        )

        first_actor = contacts[..., 0]
        second_actor = contacts[..., 1]

        def touches_cube(other_ids: torch.Tensor) -> torch.Tensor:
            return (
                valid
                & (
                    (
                        (first_actor == cube_ids[:, None])
                        & (second_actor == other_ids[:, None])
                    )
                    | (
                        (second_actor == cube_ids[:, None])
                        & (first_actor == other_ids[:, None])
                    )
                )
            ).any(dim=1)

        values = (touches_cube(finger_ids[0]) & touches_cube(finger_ids[1])).to(
            device=context.env_ids.device
        )
        return BinaryEffectObservation(values=values)


@dataclass(frozen=True, slots=True)
class _CubeControlPartEvidenceProviderFactory(ControlPartEvidenceProviderFactory):
    """Create live control-part evidence from the task's contact sensor."""

    provider_id: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_REVISION
    sensor_uid: str = CUBE_CONTACT_SENSOR_UID

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        scene_provider: SceneProvider,
    ) -> EffectEvidenceProvider:
        """Bind the exact sensor, cube, robot, and shared scene provider."""
        from embodichain.lab.sim.objects import RigidObject, Robot
        from embodichain.lab.sim.sensors import ContactSensor

        del scene_registry
        if not isinstance(robot, Robot):
            raise TypeError("Cube grasp evidence requires a simulation Robot.")
        if engine.robot is not robot:
            raise ValueError("Cube grasp evidence requires the engine's exact robot.")
        get_sensor = getattr(simulation, "get_sensor", None)
        get_rigid_object = getattr(simulation, "get_rigid_object", None)
        if not callable(get_sensor) or not callable(get_rigid_object):
            raise TypeError("simulation must expose sensor and rigid-object lookup.")
        sensor = get_sensor(self.sensor_uid)
        if not isinstance(sensor, ContactSensor):
            raise TypeError(f"Sensor {self.sensor_uid!r} must be a ContactSensor.")
        cube = get_rigid_object(CUBE_ENTITY_ID)
        if not isinstance(cube, RigidObject):
            raise TypeError(f"Rigid object {CUBE_ENTITY_ID!r} is not available.")
        observer = _CubeFingerConstraintObserver(sensor, cube, robot)
        return ControlPartSimulationEvidenceProvider(
            robot,
            scene_provider=scene_provider,
            constraint_observer=observer,
        )


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
        workflow_recovery_policy=WorkflowRecoveryPolicy(
            max_recovery_attempts=2,
        ),
    )


REPEATED_PICK_PLACE_EXPERT_PROGRAM_REGISTRATION = SimulationExpertProgramRegistration(
    scene_binding=create_repeated_pick_place_scene_binding(),
    robot_profile_binding=create_repeated_pick_place_robot_profile_binding(),
    control_part_evidence_factory=_CubeControlPartEvidenceProviderFactory(),
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
