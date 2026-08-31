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
"""Shared waypoint modality ids/names (no simulator imports)."""

from __future__ import annotations

WP_TYPE_CARTESIAN = 0
WP_TYPE_POSITION_ONLY = 1
WP_TYPE_JOINT = 2
NUM_WP_TYPES = 3

WAYPOINT_MODALITY_NAMES = ("cartesian", "position_only", "joint")


WAYPOINT_TASK_FIELDS = (
    "initial_joint_q",
    "episode_num_waypoints",
    "waypoint_valid_mask",
    "waypoint_type",
    "waypoint_pos_mask",
    "waypoint_rot_mask",
    "waypoint_joint_mask",
    "waypoints",
    "waypoint_quats",
    "waypoint_joint_qs",
)

WAYPOINT_TASK_OPTIONAL_FIELDS = (
    "initial_eef_pose",
    "waypoint_motion_scale_h",
    "waypoint_distance_bucket",
    "waypoint_active_joint_count",
    "waypoint_direction_relation",
    "waypoint_se3_primitive",
)
