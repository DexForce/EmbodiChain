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

"""Pinned source URDF and meshes for the AgileX AGX parallel gripper."""

from __future__ import annotations

from .._arm_assets import _AgilexArmAsset, _REVISION

__all__ = ["AgxGripper"]

_GRIPPER_URDF = "agx_gripper/urdf/agx_gripper_description.urdf"
_SOURCE_FILES = {
    _GRIPPER_URDF: "e8ca1a5dad852e4afc88a56430b3f9a005b719c1c4fe648203b2bee0cff3b4c5",
    "agx_gripper/meshes/dae/gripper_base.dae": "aaea10ec06709d6a3343dd3378553eee3df40b43ee276df10c5b68ac7f61bae2",
    "agx_gripper/meshes/dae/gripper_link1.dae": "b4aaf0e024d09eb0454e70e0c5092638edcf7da7c253d1effdeeaec46f76427b",
    "agx_gripper/meshes/dae/gripper_link2.dae": "0da1350898b2787b76b84698253aeb1efd2ddb6360b2fa2ae03ecff389883af8",
    "agx_gripper/meshes/gripper_base.stl": "0b839dc0bcb7d94e3422c74f6396b54a0441ba4836448ab5586e7f372d1bdbf3",
    "agx_gripper/meshes/gripper_link1.stl": "475a7d8db056524d7659fadf5a2c307d62791047b6214d733dfc7846ed1c613d",
    "agx_gripper/meshes/gripper_link2.stl": "3f88d534a85e28a33ef7d2061adacbd1d25214aa9325f9e464aad0f9e4b74606",
}


class AgxGripper(_AgilexArmAsset):
    """Resolve the standalone AGX two-finger tool with its source mimic joints.

    Args:
        data_root: Optional override for the EmbodiChain asset cache root.
    """

    cache_prefix = f"AgxGripper_{_REVISION[:12]}_v1"

    def __init__(self, data_root: str | None = None) -> None:
        super().__init__(
            data_root,
            source_urdf=_GRIPPER_URDF,
            urdf_md5="263dd6f47bb943e13485702e6c5f9162",
            source_files=_SOURCE_FILES,
            output_urdf="agx_gripper.urdf",
        )
