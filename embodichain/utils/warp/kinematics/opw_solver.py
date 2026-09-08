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

"""Compatibility aliases for ``embodichain.compute.kinematics._warp.opw``."""

from __future__ import annotations

from embodichain.compute.kinematics._warp.opw import (
    wp_vec48f,
    wp_vec6f,
    normalize_to_pi,
    normalize_in_limit,
    is_within_limit,
    safe_acos,
    th4_th6_for_branch,
    OPWparam,
    get_transform_err,
    opw_single_fk,
    opw_fk_kernel,
    opw_ik_kernel,
    opw_ik_select_kernel,
)

__all__ = [
    "wp_vec48f",
    "wp_vec6f",
    "normalize_to_pi",
    "normalize_in_limit",
    "is_within_limit",
    "safe_acos",
    "th4_th6_for_branch",
    "OPWparam",
    "get_transform_err",
    "opw_single_fk",
    "opw_fk_kernel",
    "opw_ik_kernel",
    "opw_ik_select_kernel",
]
