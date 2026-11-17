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
"""Observation functors for environment managers.

Existing imports and configuration names resolve from this package. The
implementation modules group general and locomotion observations.
"""

from __future__ import annotations

from ._common import *
from ._common import (
    _capture_proprioception,
    _get_deformable_centroid_state,
    _get_evaluation_state,
    _get_segment_start_state,
    _state_descriptor,
)
from ._locomotion import *

__all__ = [
    "get_entity_state",
    "get_object_pose",
    "get_rigid_object_pose",
    "get_object_body_scale",
    "get_object_uid",
    "get_rigid_object_velocity",
    "normalize_robot_joint_data",
    "get_sensor_pose_in_robot_frame",
    "get_sensor_intrinsics",
    "compute_semantic_mask",
    "get_robot_eef_pose",
    "target_position",
    "compute_exteroception",
    "get_rigid_object_physics_attributes",
    "get_articulation_joint_drive",
    "get_fluid_particle_state",
    "get_fluid_summary",
    "get_fluid_volume_in_region",
    "velocity_locomotion_observation",
]
