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

"""Declarative dual-UR5 can hand-over environment.

The task owns only the physical scene, robot-resource profile, and configured
object-space hand-over poses. The packaged Expert Program selects the unified
``hand_over`` semantic call; shared runtime components generate and execute all
pickup, transfer, placement, and release motion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    AntipodalGraspAffordanceBinding,
    ConfiguredHandOverPoseProvider,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    ExpertProgramCfg,
    ExpertProgramEnvironmentAdapter,
    SimulationExpertProgramRegistration,
    SimulationRigidObjectBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    create_simulation_expert_program_adapter,
)
from embodichain.lab.gym.envs.managers import EventCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import (
    wait_for_dynamic_objects_to_settle,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    ExecutionRunnerCfg,
    HandOverOptions,
    MotionPolicy,
    RecoveryPolicy,
    TrackingPolicy,
)
from embodichain.lab.sim.cfg import (
    LightCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    RobotCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.skills import (
    BinaryEffectEvidenceQuery,
    BinaryEffectObservation,
    BinaryObservationCallback,
    COMPOSITE_EFFECT_MONITOR_ID,
    COMPOSITE_EFFECT_MONITOR_REVISION,
    ControlPartEvidenceAddress,
    EffectEvidenceCollectionContext,
    EffectMonitorRef,
    SceneCollisionRole,
    SceneDynamics,
)
from embodichain.lab.sim.skills.profiles import (
    SkillPolicyPreset,
    WorkflowRecoveryPolicy,
)

from ._common import (
    create_parallel_jaw_grasp_pose_generator,
    load_bundled_expert_program,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "HandOverEnv",
    "HAND_OVER_EXPERT_PROGRAM_REGISTRATION",
    "HAND_OVER_POSE_PROVIDER",
    "create_hand_over_robot_profile_binding",
    "create_hand_over_scene_binding",
]

CAN_UID = "can"
CAN_SIMULATION_UID = "handover_object"
SUPPORT_SURFACE_UID = "support_surface"
HAND_OVER_SCENE_REGISTRY_ID = "dual_ur5_handover_v1"
HAND_OVER_ROBOT_PROFILE_ID = "dual_ur5_handover_v1"
HAND_OVER_GRASP_AFFORDANCE_ID = "can_antipodal_grasp"
HAND_OVER_EXPERT_PROGRAM_FILENAME = "hand_over.yaml"

CAN_MESH_PATH = "SodaCan/simple_cola_can.obj"
ARM_URDF_PATH = "UniversalRobots/UR5/UR5.urdf"
GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_TCP_Z = 0.155
GRIPPER_MAX_OPEN_WIDTH = 0.100
GRIPPER_FINGER_LENGTH = 0.12
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040
GRIPPER_OPEN_QPOS = 0.0
GRIPPER_GRASP_QPOS = 0.025
GRIPPER_MASTER_DRIVE_STIFFNESS = 1e3
GRIPPER_MASTER_DRIVE_DAMPING = 1e2
GRIPPER_MASTER_DRIVE_MAX_EFFORT = 1e4
GRIPPER_FINGER_DYNAMIC_FRICTION = 2.0
GRIPPER_FINGER_STATIC_FRICTION = 2.0
HAND_OVER_SAMPLE_COUNT = 200
HAND_OVER_GRASP_SAMPLE_COUNT = 10_000
HAND_OVER_APPROACH_DIRECTION_SAMPLES = 1
HAND_OVER_GRASP_OPENING_MARGIN = 0.03
HAND_OVER_TRACKING_ERROR_THRESHOLD = 1.0
HAND_OVER_CONTROL_DT = 0.04
HAND_OVER_EFFECT_CONSECUTIVE_SAMPLES = 10
GRIPPER_CONSTRAINT_QPOS_THRESHOLD = 0.004

SUPPORT_SURFACE_Z = 0.50
SUPPORT_SURFACE_SIZE = (0.8, 1.2, 0.02)
SUPPORT_SURFACE_CENTER = (0.0, 0.0, 0.49)
CAN_INITIAL_POSITION = (0.0, 0.02, 0.62)
CAN_INITIAL_ROTATION_DEG = (90.0, 0.0, 0.0)
CAN_SCALE = (0.56, 0.56, 0.56)
CAN_MASS = 0.33

_GRIPPER_TCP = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, GRIPPER_TCP_Z),
    (0.0, 0.0, 0.0, 1.0),
)
_UR_IK_NEAREST_WEIGHT = (1.0, 4.0, 1.0, 1.0, 1.0, 1.0)
_LEFT_ARM_HOME = (0.0, 0.0, -1.57, -1.57, 1.57, 1.57)
_RIGHT_ARM_HOME = (-1.57, -1.57, -1.57, -1.57, 0.0, 0.0)
_DUAL_UR5_INIT_QPOS = (*_LEFT_ARM_HOME, *_RIGHT_ARM_HOME, 0.0, 0.0, 0.0, 0.0)


HAND_OVER_POSE_PROVIDER = ConfiguredHandOverPoseProvider(
    middle_position=(0.0, 0.0, 0.7),
    middle_quaternion_wxyz=(0.7071067812, 0.7071067812, 0.0, 0.0),
    final_position=(0.0, -0.2, 0.6),
    final_quaternion_wxyz=(0.7071067812, 0.7071067812, 0.0, 0.0),
)


def _create_gripper_constraint_observer(
    robot: Robot,
) -> BinaryObservationCallback:
    """Create measured-aperture evidence for the task's PGI grippers.

    This task has no contact sensor, so phase gates use measured closure as the
    binary gripper signal. Final object placement remains independently checked
    by the program validator.
    """

    def observe(
        query: BinaryEffectEvidenceQuery,
        context: EffectEvidenceCollectionContext,
    ) -> BinaryEffectObservation:
        address = query.source.address
        if type(address) is not ControlPartEvidenceAddress:
            raise TypeError("Gripper constraint evidence requires a control part.")
        qpos = robot.get_qpos(name=address.control_part)
        if qpos.dim() != 2 or qpos.shape[1] == 0:
            raise ValueError("Gripper qpos must have non-empty shape (B, J).")
        env_ids = context.env_ids.to(device=qpos.device, dtype=torch.long)
        measured = qpos.index_select(0, env_ids)
        closure = torch.amax(torch.abs(measured - GRIPPER_OPEN_QPOS), dim=1)
        return BinaryEffectObservation(
            values=closure >= GRIPPER_CONSTRAINT_QPOS_THRESHOLD,
        )

    return observe


def _dual_ur5_robot_dict() -> dict[str, object]:
    """Return the shared serialized dual-UR5 embodiment declaration."""
    return {
        "uid": "DualUR5HandOver",
        "urdf_cfg": {
            "fname": "dual_ur5_hand_over",
            "name_case": {"joint": "lower", "link": "lower"},
            "components": [
                {
                    "component_type": "left_arm",
                    "urdf_path": ARM_URDF_PATH,
                    "transform": [
                        [0.0, -1.0, 0.0, -0.3],
                        [1.0, 0.0, 0.0, -1.45],
                        [0.0, 0.0, 1.0, 0.4],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                {
                    "component_type": "right_arm",
                    "urdf_path": ARM_URDF_PATH,
                    "transform": [
                        [0.0, -1.0, 0.0, 0.3],
                        [1.0, 0.0, 0.0, -1.45],
                        [0.0, 0.0, 1.0, 0.4],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                {
                    "component_type": "left_hand",
                    "urdf_path": GRIPPER_URDF_PATH,
                },
                {
                    "component_type": "right_hand",
                    "urdf_path": GRIPPER_URDF_PATH,
                },
            ],
        },
        "control_parts": {
            "left_arm": ["left_joint[0-9]"],
            "right_arm": ["right_joint[0-9]"],
            "dual_arm": ["left_joint[0-9]", "right_joint[0-9]"],
            "left_hand": ["left_gripper_finger1_joint_1"],
            "right_hand": ["right_gripper_finger1_joint_1"],
        },
        "drive_pros": {
            "stiffness": {
                "left_joint[0-9]": 1e4,
                "right_joint[0-9]": 1e4,
                "left_gripper_finger1_joint_1": GRIPPER_MASTER_DRIVE_STIFFNESS,
                "right_gripper_finger1_joint_1": GRIPPER_MASTER_DRIVE_STIFFNESS,
                "left_gripper_finger2_joint_1": 0.0,
                "right_gripper_finger2_joint_1": 0.0,
            },
            "damping": {
                "left_joint[0-9]": 1e3,
                "right_joint[0-9]": 1e3,
                "left_gripper_finger1_joint_1": GRIPPER_MASTER_DRIVE_DAMPING,
                "right_gripper_finger1_joint_1": GRIPPER_MASTER_DRIVE_DAMPING,
                "left_gripper_finger2_joint_1": 0.0,
                "right_gripper_finger2_joint_1": 0.0,
            },
            "max_effort": {
                "left_joint[0-9]": 1e5,
                "right_joint[0-9]": 1e5,
                "left_gripper_finger1_joint_1": GRIPPER_MASTER_DRIVE_MAX_EFFORT,
                "right_gripper_finger1_joint_1": GRIPPER_MASTER_DRIVE_MAX_EFFORT,
                "left_gripper_finger2_joint_1": 0.0,
                "right_gripper_finger2_joint_1": 0.0,
            },
            "drive_type": "force",
        },
        "link_attrs": {
            "gripper_fingers": {
                "link_names_expr": [
                    "(left|right)_gripper_finger[12]_link_1",
                ],
                "attrs": {
                    "dynamic_friction": GRIPPER_FINGER_DYNAMIC_FRICTION,
                    "static_friction": GRIPPER_FINGER_STATIC_FRICTION,
                },
            }
        },
        "solver_cfg": {
            "left_arm": {
                "class_type": "URSolver",
                "ur_type": "ur5",
                "root_link_name": "left_base_link",
                "end_link_name": "left_ee_link",
                "tcp": _GRIPPER_TCP,
                "ik_nearest_weight": _UR_IK_NEAREST_WEIGHT,
            },
            "right_arm": {
                "class_type": "URSolver",
                "ur_type": "ur5",
                "root_link_name": "right_base_link",
                "end_link_name": "right_ee_link",
                "tcp": _GRIPPER_TCP,
                "ik_nearest_weight": _UR_IK_NEAREST_WEIGHT,
            },
        },
        "init_pos": [1.95, 0.0, 0.1],
        "init_rot": [0.0, 0.0, -90.0],
        "init_qpos": _DUAL_UR5_INIT_QPOS,
    }


def _create_default_robot_cfg() -> RobotCfg:
    """Create the dual-UR5 and dual-PGI task embodiment."""
    return RobotCfg.from_dict(_dual_ur5_robot_dict())


def _load_default_expert_program() -> ExpertProgramCfg:
    """Decode and preflight the packaged semantic hand-over program."""
    program = load_bundled_expert_program(HAND_OVER_EXPERT_PROGRAM_FILENAME)
    HAND_OVER_EXPERT_PROGRAM_REGISTRATION.catalog.preflight(program)
    return program


def _create_default_env_cfg() -> EmbodiedEnvCfg:
    """Create a directly-instantiable physical and semantic task config."""
    cfg = EmbodiedEnvCfg()
    cfg.max_episode_steps = 1200
    cfg.robot = _create_default_robot_cfg()
    cfg.sensor = []
    cfg.light = EmbodiedEnvCfg.EnvLightCfg(
        direct=[
            LightCfg(
                uid="main_light",
                color=(0.6, 0.6, 0.6),
                intensity=30.0,
                init_pos=(0.0, -0.4, 3.0),
            )
        ]
    )
    cfg.background = [
        RigidObjectCfg(
            uid=SUPPORT_SURFACE_UID,
            shape=CubeCfg(size=list(SUPPORT_SURFACE_SIZE)),
            attrs=RigidBodyAttributesCfg(
                mass=10.0,
                dynamic_friction=0.9,
                static_friction=0.95,
                restitution=0.01,
            ),
            body_type="static",
            init_pos=list(SUPPORT_SURFACE_CENTER),
            init_rot=[0.0, 0.0, 0.0],
        )
    ]
    cfg.rigid_object = [
        RigidObjectCfg(
            uid=CAN_SIMULATION_UID,
            shape=MeshCfg(fpath=get_data_path(CAN_MESH_PATH), compute_uv=False),
            attrs=RigidBodyAttributesCfg(
                mass=CAN_MASS,
                dynamic_friction=0.97,
                static_friction=0.99,
                angular_damping=1.0,
                linear_damping=0.5,
                contact_offset=0.001,
                rest_offset=0.0,
                restitution=0.01,
                min_position_iters=32,
                min_velocity_iters=8,
                max_depenetration_velocity=2.0,
            ),
            max_convex_hull_num=1,
            init_pos=list(CAN_INITIAL_POSITION),
            init_rot=list(CAN_INITIAL_ROTATION_DEG),
            body_scale=CAN_SCALE,
        )
    ]
    cfg.extensions = {}
    cfg.events = {
        "settle_can_on_reset": EventCfg(
            func=wait_for_dynamic_objects_to_settle,
            mode="reset",
            params={
                "entity_cfgs": [SceneEntityCfg(uid=CAN_SIMULATION_UID)],
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


def create_hand_over_scene_binding() -> SimulationSceneBinding:
    """Declare the can, support slab, and antipodal grasp affordance."""
    return SimulationSceneBinding(
        registry_id=HAND_OVER_SCENE_REGISTRY_ID,
        rigid_objects=(
            SimulationRigidObjectBinding(
                entity_id=CAN_UID,
                simulation_uid=CAN_SIMULATION_UID,
                dynamics=SceneDynamics.DYNAMIC,
                collision_role=SceneCollisionRole.NONE,
                semantic_type="soda_can",
                default_grasp_affordance=HAND_OVER_GRASP_AFFORDANCE_ID,
            ),
            SimulationRigidObjectBinding(
                entity_id=SUPPORT_SURFACE_UID,
                simulation_uid=SUPPORT_SURFACE_UID,
                dynamics=SceneDynamics.STATIC,
                collision_role=SceneCollisionRole.NONE,
                semantic_type="support_surface",
            ),
        ),
        antipodal_grasps=(
            AntipodalGraspAffordanceBinding(
                entity_id=HAND_OVER_GRASP_AFFORDANCE_ID,
                object_id=CAN_UID,
                native_name="can_mesh_antipodal",
                revision="can-antipodal-v1",
            ),
        ),
    )


def create_hand_over_robot_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Declare left/right arm-and-gripper semantic resources."""
    motion_capabilities = frozenset(
        {
            BATCH_INVERSE_KINEMATICS_CAPABILITY,
            CARTESIAN_POSE_CAPABILITY,
            FORWARD_KINEMATICS_CAPABILITY,
        }
    )
    return SimulationRobotSkillProfileBinding(
        profile_id=HAND_OVER_ROBOT_PROFILE_ID,
        resources=tuple(
            ControlPartResourceBinding(
                resource_id=side,
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part=f"{side}_arm",
                        capabilities=motion_capabilities,
                    ),
                    ControlPartEndpointBinding(
                        endpoint_id="grasp",
                        control_part=f"{side}_hand",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        command_preset=f"{side}_parallel_gripper",
                    ),
                ),
            )
            for side in ("left", "right")
        ),
        command_presets=tuple(
            ControlPartCommandPreset(
                preset_id=f"{side}_parallel_gripper",
                control_part=f"{side}_hand",
                commands={
                    "open": (GRIPPER_OPEN_QPOS,),
                    "grasp": (GRIPPER_GRASP_QPOS,),
                },
            )
            for side in ("left", "right")
        ),
        defaults={
            "hand_over": {"source": "left", "destination": "right"},
        },
        presets=(
            SkillPolicyPreset(
                "safe",
                action_option_templates={
                    "hand_over": HandOverOptions(
                        pre_grasp_distance=0.08,
                        lift_height=0.15,
                        hand_interp_steps=10,
                    ),
                },
                motion_policy=MotionPolicy(sample_count=HAND_OVER_SAMPLE_COUNT),
                tracking_policy=TrackingPolicy.joint_position(
                    in_flight_max_abs_error=HAND_OVER_TRACKING_ERROR_THRESHOLD,
                    terminal_max_abs_error=HAND_OVER_TRACKING_ERROR_THRESHOLD,
                ),
                recovery_policy=RecoveryPolicy(max_action_retries=0),
                workflow_recovery_policy=WorkflowRecoveryPolicy(
                    max_recovery_attempts=2,
                ),
                runner_cfg=ExecutionRunnerCfg(
                    minimum_cycle_time=HAND_OVER_CONTROL_DT,
                    hold_during_effect_verification=False,
                    hold_on_completion=False,
                ),
                effect_monitors={
                    "hand_over": EffectMonitorRef(
                        COMPOSITE_EFFECT_MONITOR_ID,
                        COMPOSITE_EFFECT_MONITOR_REVISION,
                        {"consecutive_samples": (HAND_OVER_EFFECT_CONSECUTIVE_SAMPLES)},
                    )
                },
            ),
        ),
        default_preset="safe",
        grounding_providers={
            "hand_over": ConfiguredHandOverPoseProvider.provider_id,
        },
    )


HAND_OVER_EXPERT_PROGRAM_REGISTRATION = SimulationExpertProgramRegistration(
    scene_binding=create_hand_over_scene_binding(),
    robot_profile_binding=create_hand_over_robot_profile_binding(),
    handover_pose_providers=(HAND_OVER_POSE_PROVIDER,),
)


@register_env(
    "HandOver-v1",
    max_episode_steps=1200,
    expert_program_registration=HAND_OVER_EXPERT_PROGRAM_REGISTRATION,
)
class HandOverEnv(EmbodiedEnv):
    """Transfer a can between two UR5 arms through a semantic program."""

    def __init__(
        self,
        cfg: EmbodiedEnvCfg | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the configured scene without task-level motion code."""
        if cfg is None:
            cfg = _create_default_env_cfg()
        super().__init__(cfg, **kwargs)
        grasp_pose_generators = {
            f"{side}_hand": create_parallel_jaw_grasp_pose_generator(
                sample_count=HAND_OVER_GRASP_SAMPLE_COUNT,
                opening_margin=HAND_OVER_GRASP_OPENING_MARGIN,
                approach_direction_samples=HAND_OVER_APPROACH_DIRECTION_SAMPLES,
            )
            for side in ("left", "right")
        }
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            scene_binding=HAND_OVER_EXPERT_PROGRAM_REGISTRATION.scene_binding,
            robot_profile_binding=(
                HAND_OVER_EXPERT_PROGRAM_REGISTRATION.robot_profile_binding
            ),
            grasp_pose_generators=grasp_pose_generators,
            handover_pose_providers=(HAND_OVER_POSE_PROVIDER,),
            constraint_observer=_create_gripper_constraint_observer(self.robot),
        )

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the shared adapter assembled for this environment."""
        return self._expert_program_adapter
