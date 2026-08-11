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

"""Declarative expert environment for opening a sliding drawer.

The task owns only scene and embodiment declarations. The packaged Expert
Program selects the semantic ``operate_articulation`` skill and its named
``open`` target; shared runtime components generate and execute all motion.
"""

from __future__ import annotations

from typing import Any

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    ArticulationOperationAffordanceBinding,
    ArticulationOperationTargetBinding,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    ExpertProgramEnvironmentAdapter,
    ExpertProgramEnvironmentMixin,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationExpertProgramRegistration,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    create_simulation_expert_program_adapter,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
    CARTESIAN_POSE_CAPABILITY,
    GRASP_CAPABILITY,
    JOINT_POSITION_CAPABILITY,
)
from embodichain.lab.sim.skills import SceneCollisionRole, SceneDynamics
from embodichain.lab.sim.skills.profiles import SkillPolicyPreset

__all__ = [
    "OpenDrawerEnv",
    "OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION",
    "create_open_drawer_robot_profile_binding",
    "create_open_drawer_scene_binding",
]

DRAWER_SCENE_REGISTRY_ID = "open_drawer_v1"
DRAWER_ROBOT_PROFILE_ID = "cobot_magic_right_manipulator_v1"
DRAWER_UID = "drawer"
DRAWER_HANDLE_LINK_ID = "drawer_handle_link"
DRAWER_HANDLE_AFFORDANCE_ID = "drawer_handle"
DRAWER_NATIVE_HANDLE_LINK = "handle_xpos"
DRAWER_NATIVE_SLIDE_JOINT = "slide_rails"
DRAWER_OPEN_POSITION = 0.11
DRAWER_OPEN_DISPLACEMENT = 0.11

# Rotation from the drawer handle frame to the historical right-arm TCP frame.
_HANDLE_POSE_OFFSET = (
    -0.023958006,
    -0.999453075,
    -0.022793945,
    0.0,
    0.999712744,
    -0.023966955,
    0.000119456,
    0.0,
    -0.000665692,
    -0.022784535,
    0.999740177,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _translation_pose(x: float, y: float, z: float) -> tuple[float, ...]:
    """Return a flattened identity-rotation pose with one translation."""
    return (
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def create_open_drawer_scene_binding() -> SimulationSceneBinding:
    """Declare the exact native drawer identities used by the semantic task."""
    approach = _translation_pose(-0.00442594, -0.00050044, -0.10508996)
    contact = _translation_pose(-0.00442594, -0.00050041, 0.00491005)
    retract = _translation_pose(-0.00442594, -0.00050044, -0.00508996)
    return SimulationSceneBinding(
        registry_id=DRAWER_SCENE_REGISTRY_ID,
        articulations=(
            SimulationArticulationBinding(
                entity_id=DRAWER_UID,
                simulation_uid=DRAWER_UID,
                dynamics=SceneDynamics.DYNAMIC,
                collision_role=SceneCollisionRole.NONE,
                semantic_type="sliding_drawer",
                default_operation_affordance=DRAWER_HANDLE_AFFORDANCE_ID,
            ),
        ),
        links=(
            SimulationArticulationLinkBinding(
                entity_id=DRAWER_HANDLE_LINK_ID,
                articulation_id=DRAWER_UID,
                native_link_name=DRAWER_NATIVE_HANDLE_LINK,
                dynamics=SceneDynamics.DYNAMIC,
                semantic_type="drawer_handle_link",
            ),
        ),
        articulation_operations=(
            ArticulationOperationAffordanceBinding(
                entity_id=DRAWER_HANDLE_AFFORDANCE_ID,
                articulation_id=DRAWER_UID,
                link_id=DRAWER_HANDLE_LINK_ID,
                joint_id=DRAWER_NATIVE_SLIDE_JOINT,
                revision="open-drawer-v1",
                semantic_targets={
                    "open": ArticulationOperationTargetBinding(
                        target_position=DRAWER_OPEN_POSITION,
                        displacement=DRAWER_OPEN_DISPLACEMENT,
                    ),
                },
                handle_pose_offset=_HANDLE_POSE_OFFSET,
                approach_offset=approach,
                contact_offset=contact,
                operation_offset=contact,
                retract_offset=retract,
                operation_axis=(0.0, 0.0, -1.0),
                position_scale=1.0,
            ),
        ),
    )


def create_open_drawer_robot_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Declare the CobotMagic right-arm and right-gripper skill resource."""
    return SimulationRobotSkillProfileBinding(
        profile_id=DRAWER_ROBOT_PROFILE_ID,
        resources=(
            ControlPartResourceBinding(
                resource_id="right_manipulator",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="right_arm",
                        capabilities=frozenset(
                            {
                                CARTESIAN_POSE_CAPABILITY,
                                JOINT_POSITION_CAPABILITY,
                            }
                        ),
                    ),
                    ControlPartEndpointBinding(
                        endpoint_id="interaction",
                        control_part="right_eef",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        command_preset="right_parallel_gripper",
                    ),
                ),
            ),
        ),
        command_presets=(
            ControlPartCommandPreset(
                preset_id="right_parallel_gripper",
                control_part="right_eef",
                commands={
                    "open": (0.05, 0.05),
                    "grasp": (0.0, 0.0),
                },
            ),
        ),
        defaults={
            "operate_articulation": {"primary": "right_manipulator"},
        },
        presets=(SkillPolicyPreset("safe"),),
        default_preset="safe",
    )


OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION = SimulationExpertProgramRegistration(
    scene_binding=create_open_drawer_scene_binding(),
    robot_profile_binding=create_open_drawer_robot_profile_binding(),
)


@register_env(
    "OpenDrawer-v1",
    max_episode_steps=300,
    expert_program_registration=OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION,
)
class OpenDrawerEnv(ExpertProgramEnvironmentMixin, EmbodiedEnv):
    """Open a drawer through a configured semantic Expert Program."""

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize the configured scene without task-level motion code."""
        super().__init__(cfg, **kwargs)
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            registration=OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION,
        )

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the shared adapter assembled for this environment."""
        return self._expert_program_adapter
