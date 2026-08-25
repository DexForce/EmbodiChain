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

"""Runtime bindings for the dual-UR5 hand-over Expert Program."""

from __future__ import annotations

from embodichain.lab.gym.envs.expert_program import (
    AntipodalGraspAffordanceBinding,
    ConfiguredHandOverPoseProvider,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    SimulationExpertProgramRegistration,
    SimulationRigidObjectBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
)
from embodichain.lab.sim.atomic_actions import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    ExecutionRunnerCfg,
    HandOverOptions,
    MotionPolicy,
)
from embodichain.lab.sim.skills import SceneCollisionRole, SceneDynamics
from embodichain.lab.sim.skills.profiles import (
    SkillPolicyPreset,
    WorkflowRecoveryPolicy,
)

__all__ = [
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
GRIPPER_OPEN_QPOS = 0.0
GRIPPER_GRASP_QPOS = 0.011
HAND_OVER_SAMPLE_COUNT = 200
HAND_OVER_GRASP_SAMPLE_COUNT = 10_000


HAND_OVER_POSE_PROVIDER = ConfiguredHandOverPoseProvider(
    middle_position=(0.0, 0.0, 0.7),
    middle_quaternion_wxyz=(0.7071067812, 0.7071067812, 0.0, 0.0),
    final_position=(0.0, -0.2, 0.7),
    final_quaternion_wxyz=(0.7071067812, 0.7071067812, 0.0, 0.0),
)


def create_hand_over_scene_binding() -> SimulationSceneBinding:
    """Declare the can, support slab, and antipodal grasp affordance.

    Returns:
        Provider-free semantic scene binding for the hand-over task.
    """
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
    """Declare left/right arm-and-gripper semantic resources.

    Returns:
        Provider-free robot skill profile for the hand-over task.
    """
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
                        lift_height=0.08,
                        hand_interp_steps=10,
                    ),
                },
                motion_policy=MotionPolicy(sample_count=HAND_OVER_SAMPLE_COUNT),
                workflow_recovery_policy=WorkflowRecoveryPolicy(
                    max_recovery_attempts=2,
                ),
                runner_cfg=ExecutionRunnerCfg(
                    hold_during_effect_verification=False,
                    hold_on_completion=False,
                ),
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
