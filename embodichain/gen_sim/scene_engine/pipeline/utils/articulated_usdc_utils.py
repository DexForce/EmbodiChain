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

"""Compatibility imports for articulation asset normalization.

The implementation is owned by Asset Engine under
``asset_engine.utils.articulated_usdc_utils``.  Scene Engine keeps this
path for callers that still import the former utility location.
"""

from __future__ import annotations

from embodichain.gen_sim.asset_engine.utils.articulated_usdc_utils import (
    _articulation_root_bottom_z,
    _canonicalize_articulated_usdc_bottom_center,
    _read_revolute_qpos_limits,
)

__all__: list[str] = []
