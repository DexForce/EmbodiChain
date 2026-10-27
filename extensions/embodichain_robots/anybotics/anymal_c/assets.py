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

"""Asset source and cache identity for ANYmalC."""

from __future__ import annotations

from embodichain.data._locomotion_asset import _LocomotionAsset

__all__ = ["ANYmalCLocomotion"]


class ANYmalCLocomotion(_LocomotionAsset):
    """Robot descriptions and dependencies for ANYmalCLocomotion."""

    archive_md5 = "6634d4b3c4b83e264e58e295ab3b5656"
