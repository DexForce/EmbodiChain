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

"""Official RP1 humanoid descriptions and mesh assets."""

from __future__ import annotations

from .._assets import _RoboPartySourceAsset

__all__ = ["RoboPartyRP1"]


class RoboPartyRP1(_RoboPartySourceAsset):
    """Resolve the four pinned RP1 URDF models and their source assets."""

    cache_prefix = "RoboPartyRP1"
    repository = "rp1_description"
    revision = "7b7d807e52cbec23f22f36444ed3129f082d76ca"
    archive_md5 = "f2ebdac8c8978d2299be44b66f1558b2"
    urdf_names = ("rp1.urdf", "rp1_24dof.urdf", "rp1_hand.urdf", "rp1_hand_fixed.urdf")
