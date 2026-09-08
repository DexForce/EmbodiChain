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

"""Compatibility aliases for ``embodichain.compute.kinematics._warp.ur``."""

from __future__ import annotations

from embodichain.compute.kinematics._warp.ur import (
    wp_vec6f,
    wp_vec48f,
    normalize_to_pi,
    URParam,
    _safe_sqrt_ur,
    _ur_dh,
    ur_single_fk,
    _ur_transform_err,
    _ur_solve_234,
    _shift_to_limit,
    ur_ik_kernel,
)

__all__ = [
    "wp_vec6f",
    "wp_vec48f",
    "normalize_to_pi",
    "URParam",
    "ur_single_fk",
    "ur_ik_kernel",
]
