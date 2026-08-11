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

The task declares its simulation identities and robot resources, while the
packaged Expert Program defines the three semantic pick/place cycles. Shared
Expert Program components own motion generation, execution, settling, and
validation; extending the cycle count or destinations requires config only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers import EventCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import (
    wait_for_dynamic_objects_to_settle,
)
from embodichain.lab.gym.envs.expert_program import (
    AntipodalGraspAffordanceBinding,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    ExpertProgramCfg,
    ExpertProgramEnvironmentAdapter,
    ExpertProgramEnvironmentMixin,
    SimulationRigidObjectBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    create_simulation_expert_program_adapter,
    load_expert_program,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    RecoveryPolicy,
)
from embodichain.lab.sim.cfg import (
    LightCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.skills import SceneCollisionRole, SceneDynamics
from embodichain.lab.sim.skills.profiles import SkillPolicyPreset
from embodichain.toolkits.graspkit.pg_grasp import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
    GripperCollisionCfg,
)
from embodichain_tasks.configs import get_config_path

__all__ = [
    "MultiSegmentsCubePickPlaceEnv",
    "create_cube_robot_profile_binding",
    "create_cube_scene_binding",
]

CUBE_UID = "cube"
CUBE_SIZE = 0.05
CUBE_SCENE_REGISTRY_ID = "multi_segments_cube_v1"
CUBE_ROBOT_PROFILE_ID = "ur5_parallel_gripper_v1"
CUBE_GRASP_AFFORDANCE_ID = "cube_antipodal_grasp"
CUBE_EXPERT_PROGRAM_PATH = Path(
    "expert_program/multi_segments/repeated_cube_pick_place.yaml"
)

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "gripper_finger1_joint_1"
GRIPPER_TCP_Z = 0.15
GRIPPER_MAX_OPEN_WIDTH = 0.100
GRIPPER_FINGER_LENGTH = 0.12
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040
GRIPPER_OPEN_QPOS = 0.0
GRIPPER_GRASP_QPOS = 0.024


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
                }
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
    cfg.extensions = {
        "grasp_samples": 10000,
        "force_reannotate": False,
    }
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


def create_cube_scene_binding(
    *,
    grasp_samples: int = 10000,
    force_reannotate: bool = False,
) -> SimulationSceneBinding:
    """Declare the cube and its exact antipodal-grasp affordance."""
    if isinstance(grasp_samples, bool) or not isinstance(grasp_samples, int):
        raise TypeError("grasp_samples must be an integer.")
    if grasp_samples < 1:
        raise ValueError("grasp_samples must be positive.")
    if not isinstance(force_reannotate, bool):
        raise TypeError("force_reannotate must be a bool.")
    return SimulationSceneBinding(
        registry_id=CUBE_SCENE_REGISTRY_ID,
        rigid_objects=(
            SimulationRigidObjectBinding(
                entity_id=CUBE_UID,
                simulation_uid=CUBE_UID,
                dynamics=SceneDynamics.DYNAMIC,
                collision_role=SceneCollisionRole.NONE,
                semantic_type="cube",
                default_grasp_affordance=CUBE_GRASP_AFFORDANCE_ID,
            ),
        ),
        antipodal_grasps=(
            AntipodalGraspAffordanceBinding(
                entity_id=CUBE_GRASP_AFFORDANCE_ID,
                object_id=CUBE_UID,
                native_name="cube_mesh_antipodal",
                revision="cube-antipodal-v1",
                generator_cfg=GraspGeneratorCfg(
                    viser_port=11801,
                    antipodal_sampler_cfg=AntipodalSamplerCfg(
                        n_sample=grasp_samples,
                        max_length=GRIPPER_MAX_OPEN_WIDTH,
                        min_length=0.005,
                    ),
                    is_partial_annotate=False,
                    is_filter_ground_collision=False,
                ),
                gripper_collision_cfg=GripperCollisionCfg(
                    max_open_length=GRIPPER_MAX_OPEN_WIDTH,
                    finger_length=GRIPPER_FINGER_LENGTH,
                    y_thickness=GRIPPER_Y_THICKNESS,
                    root_z_width=GRIPPER_ROOT_Z_WIDTH,
                    open_check_margin=0.002,
                    point_sample_dense=0.012,
                ),
                force_reannotate=force_reannotate,
            ),
        ),
    )


def create_cube_robot_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Declare the UR5 arm and parallel-gripper semantic resource."""
    motion_capabilities = frozenset(
        {
            BATCH_INVERSE_KINEMATICS_CAPABILITY,
            CARTESIAN_POSE_CAPABILITY,
            FORWARD_KINEMATICS_CAPABILITY,
        }
    )
    return SimulationRobotSkillProfileBinding(
        profile_id=CUBE_ROBOT_PROFILE_ID,
        resources=(
            ControlPartResourceBinding(
                resource_id="manipulator",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="arm",
                        capabilities=motion_capabilities,
                    ),
                    ControlPartEndpointBinding(
                        endpoint_id="grasp",
                        control_part="hand",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        command_preset="parallel_gripper",
                    ),
                ),
            ),
        ),
        command_presets=(
            ControlPartCommandPreset(
                preset_id="parallel_gripper",
                control_part="hand",
                commands={
                    "open": (GRIPPER_OPEN_QPOS,),
                    "grasp": (GRIPPER_GRASP_QPOS,),
                },
            ),
        ),
        defaults={
            "pick_up": {"primary": "manipulator"},
            "place": {"primary": "manipulator"},
        },
        presets=(
            SkillPolicyPreset(
                "safe",
                recovery_policy=RecoveryPolicy(tracking_error_threshold=0.08),
            ),
        ),
        default_preset="safe",
    )


@register_env("MultiSegmentsCubePickPlace-v1", max_episode_steps=1200)
class MultiSegmentsCubePickPlaceEnv(ExpertProgramEnvironmentMixin, EmbodiedEnv):
    """Repeatedly pick and place a cube from a semantic config program."""

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        """Initialize the configured scene without task-level motion code."""
        if cfg is None:
            cfg = _create_default_env_cfg()
        super().__init__(cfg, **kwargs)
        self._expert_program_adapter = create_simulation_expert_program_adapter(
            self,
            scene_binding=create_cube_scene_binding(
                grasp_samples=getattr(self, "grasp_samples", 10000),
                force_reannotate=getattr(self, "force_reannotate", False),
            ),
            robot_profile_binding=create_cube_robot_profile_binding(),
        )

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the shared adapter assembled for this environment."""
        return self._expert_program_adapter
