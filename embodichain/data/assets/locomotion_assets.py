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


"""Download the robot assets used by the locomotion tasks."""

from __future__ import annotations

import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DEFAULT_DATA_ROOT,
    EMBODICHAIN_DOWNLOAD_PREFIX,
)

__all__ = [
    "ANYmalCLocomotion",
    "HumanoidRun",
    "MicroDuckLocomotion",
    "UnitreeG1Locomotion",
    "UnitreeGo1Locomotion",
    "UnitreeGo2Locomotion",
    "UnitreeH1_2Locomotion",
]


class _LocomotionAsset(EmbodiChainDataset):
    """Resolve one versioned archive through the standard dataset cache."""

    archive_md5: str

    def __init__(self, data_root: str | None = None) -> None:
        prefix = type(self).__name__
        filename = f"robot_assets/{prefix}.zip"
        descriptor = o3d.data.DataDescriptor(
            [
                f"https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/{filename}",
                f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}",
            ],
            self.archive_md5,
        )
        super().__init__(
            prefix,
            descriptor,
            EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root,
        )


class ANYmalCLocomotion(_LocomotionAsset):
    """Robot descriptions and dependencies for ANYmalCLocomotion."""

    archive_md5 = "6634d4b3c4b83e264e58e295ab3b5656"


class HumanoidRun(_LocomotionAsset):
    """Robot descriptions and dependencies for HumanoidRun."""

    archive_md5 = "846cb8db9d069d0eae41c8b799ad1872"


class MicroDuckLocomotion(_LocomotionAsset):
    """Robot descriptions and dependencies for MicroDuckLocomotion."""

    archive_md5 = "65c294dddfc8b0b21ccc4b8829c4d13c"


class UnitreeG1Locomotion(_LocomotionAsset):
    """Robot descriptions and dependencies for UnitreeG1Locomotion."""

    archive_md5 = "617102145c54e678bc46f6f9421ee497"


class UnitreeGo1Locomotion(_LocomotionAsset):
    """Robot descriptions and dependencies for UnitreeGo1Locomotion."""

    archive_md5 = "95e923163f4137b1dd10ac0e4a1f8238"


class UnitreeGo2Locomotion(_LocomotionAsset):
    """Robot descriptions and dependencies for UnitreeGo2Locomotion."""

    archive_md5 = "a6342c231788db4bf7e4dd6981400223"


class UnitreeH1_2Locomotion(_LocomotionAsset):
    """Robot descriptions and dependencies for UnitreeH1_2Locomotion."""

    archive_md5 = "7c9457022329a008bc72189228085bab"
