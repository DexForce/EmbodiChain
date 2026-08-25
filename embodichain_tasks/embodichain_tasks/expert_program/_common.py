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

"""Shared UR5 embodiment declarations for Expert Program task examples."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import resources
from typing import TYPE_CHECKING

import torch

from embodichain.lab.gym.envs.expert_program import (
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    ExpertProgramCfg,
    SimulationRobotSkillProfileBinding,
    load_expert_program,
)
from embodichain.lab.sim.atomic_actions import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    MotionPolicy,
    TrackingPolicy,
)
from embodichain.lab.sim.skills import SkillPolicyPreset
from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

DH_PGI_140_80_GRIPPER_MODEL = ParallelJawGripperModelCfg(
    model_id="dh_pgi_140_80",
    min_opening_width=0.005,
    max_opening_width=0.100,
    finger_length=0.12,
    finger_width=0.040,
    finger_thickness=0.01,
    palm_depth=0.096,
)
"""Physical model of the concrete gripper used by these reference tasks."""
DEFAULT_GRIPPER_CLOSE_QPOS = 0.024
DEFAULT_TRACKING_ERROR_THRESHOLD = 0.1
MANIPULATOR_RESOURCE_ID = "manipulator"
ARM_CONTROL_PART = "arm"
HAND_CONTROL_PART = "hand"


def load_bundled_expert_program(filename: str) -> ExpertProgramCfg:
    """Load one packaged reference program from ``embodichain_tasks.configs``."""
    if (
        type(filename) is not str
        or not filename
        or filename != filename.strip()
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("filename must be one non-empty resource basename.")
    resource = resources.files("embodichain_tasks.configs").joinpath(
        "expert_program", filename
    )
    with resources.as_file(resource) as path:
        return load_expert_program(path)


def create_parallel_jaw_grasp_pose_generator(
    *,
    sample_count: int,
    opening_margin: float,
    force_refresh: bool = False,
) -> AntipodalGraspPoseGenerator:
    """Create the shared antipodal service for the configured parallel jaws."""
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGenerator,
        AntipodalGraspPoseGeneratorCfg,
        GraspAnnotationCfg,
        ParallelJawGraspCollisionCfg,
    )

    return AntipodalGraspPoseGenerator(
        DH_PGI_140_80_GRIPPER_MODEL,
        algorithm_cfg=AntipodalGraspPoseGeneratorCfg(sample_count=sample_count),
        collision_cfg=ParallelJawGraspCollisionCfg(
            opening_margin=opening_margin,
            point_sample_density=0.012,
            filter_ground_collision=False,
        ),
        annotation_cfg=GraspAnnotationCfg(
            selection_mode="whole_mesh",
            viser_port=11801,
            force_refresh=force_refresh,
        ),
    )


def create_ur5_skill_profile_binding(
    robot: Robot,
    *,
    profile_id: str,
    sample_count: int,
    skill_ids: Sequence[str],
) -> SimulationRobotSkillProfileBinding:
    """Bind semantic manipulation resources to the common live UR5 robot."""
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least 2.")
    normalized_skill_ids = tuple(skill_ids)
    if not normalized_skill_ids or not all(
        type(skill_id) is str and skill_id for skill_id in normalized_skill_ids
    ):
        raise ValueError("skill_ids must contain non-empty strings.")

    hand_limits = robot.get_qpos_limits(name=HAND_CONTROL_PART)[0].to(
        device=robot.device, dtype=torch.float32
    )
    hand_open_qpos = hand_limits[:, 0]
    hand_close_qpos = torch.clamp(
        torch.full_like(hand_limits[:, 1], DEFAULT_GRIPPER_CLOSE_QPOS),
        min=hand_limits[:, 0],
        max=hand_limits[:, 1],
    )
    motion_capabilities = frozenset(
        {
            BATCH_INVERSE_KINEMATICS_CAPABILITY,
            CARTESIAN_POSE_CAPABILITY,
            FORWARD_KINEMATICS_CAPABILITY,
        }
    )
    return SimulationRobotSkillProfileBinding(
        profile_id=profile_id,
        resources=(
            ControlPartResourceBinding(
                resource_id=MANIPULATOR_RESOURCE_ID,
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part=ARM_CONTROL_PART,
                        capabilities=motion_capabilities,
                    ),
                    ControlPartEndpointBinding(
                        endpoint_id="grasp",
                        control_part=HAND_CONTROL_PART,
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        command_preset="parallel_jaw_commands",
                    ),
                ),
            ),
        ),
        command_presets=(
            ControlPartCommandPreset(
                preset_id="parallel_jaw_commands",
                control_part=HAND_CONTROL_PART,
                commands={
                    "open": tuple(
                        float(value) for value in hand_open_qpos.detach().cpu().tolist()
                    ),
                    "grasp": tuple(
                        float(value)
                        for value in hand_close_qpos.detach().cpu().tolist()
                    ),
                },
            ),
        ),
        defaults={
            skill_id: {"primary": MANIPULATOR_RESOURCE_ID}
            for skill_id in normalized_skill_ids
        },
        presets=(
            SkillPolicyPreset(
                "safe",
                motion_policy=MotionPolicy(sample_count=sample_count),
                tracking_policy=TrackingPolicy.joint_position(
                    in_flight_max_abs_error=DEFAULT_TRACKING_ERROR_THRESHOLD,
                    terminal_max_abs_error=DEFAULT_TRACKING_ERROR_THRESHOLD,
                ),
            ),
        ),
        default_preset="safe",
    )


__all__: list[str] = []
