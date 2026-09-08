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

"""Compatibility aliases for ``embodichain.compute.kinematics._warp.srs``."""

from __future__ import annotations

from embodichain.compute.kinematics._warp.srs import (
    identity_mat44,
    identity_mat33,
    safe_acos,
    wrap_to_limit,
    safe_division,
    skew,
    dh_transform,
    transform_pose,
    transform_pose_kernel,
    calculate_arm_joint_angles,
    compute_reference_plane,
    compute_fk_kernel,
    compute_arm_angle_kernel,
    validate_fk_with_target,
    compute_ik_kernel,
    nearest_ik_kernel,
    check_success_kernel,
)

__all__ = [
    "identity_mat44",
    "identity_mat33",
    "safe_acos",
    "wrap_to_limit",
    "safe_division",
    "skew",
    "dh_transform",
    "transform_pose",
    "transform_pose_kernel",
    "calculate_arm_joint_angles",
    "compute_reference_plane",
    "compute_fk_kernel",
    "compute_arm_angle_kernel",
    "validate_fk_with_target",
    "compute_ik_kernel",
    "nearest_ik_kernel",
    "check_success_kernel",
]
