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

"""Compatibility imports for the packaged RP1 carrying implementation."""

from __future__ import annotations
from embodichain.extensions.models.policies.rp1_carry._rp_hand import (
    _numbers,
    add_rp_hands,
    _set_carrying_pose,
    _wrist_inertia,
    _set_combined_inertia,
    RP_HAND_ASSET,
)
