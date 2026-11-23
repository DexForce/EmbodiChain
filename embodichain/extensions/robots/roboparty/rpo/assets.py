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

"""Official Roboto Origin descriptions and mesh assets."""

from __future__ import annotations

from .._assets import _RoboPartySourceAsset

__all__ = ["RoboPartyRPO"]


class RoboPartyRPO(_RoboPartySourceAsset):
    """Resolve the pinned RPO model as ``RoboPartyRPO/rpo.urdf``."""

    cache_prefix = "RoboPartyRPO"
    repository = "rpo_description"
    revision = "8e0f8f78903dc147a5c094c0fdc14b4a78f1b667"
    archive_md5 = "da87a55aba3ca787a4c44ac2ec8cfbb2"
    urdf_names = ("rpo.urdf",)
