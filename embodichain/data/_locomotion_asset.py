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


"""Shared archive downloading for model-owned locomotion assets."""

from __future__ import annotations

import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DEFAULT_DATA_ROOT,
    EMBODICHAIN_DOWNLOAD_PREFIX,
)

__all__ = []

_HF_HUB_PREFIX = (
    "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
)


class _LocomotionAsset(EmbodiChainDataset):
    """Resolve one versioned archive through the standard dataset cache."""

    archive_md5: str

    def __init__(self, data_root: str | None = None) -> None:
        prefix = type(self).__name__
        filename = f"robot_assets/{prefix}.zip"
        # The configured prefix goes first; the Hugging Face Hub is only a fallback.
        urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}"]
        if EMBODICHAIN_DOWNLOAD_PREFIX != _HF_HUB_PREFIX:
            urls.append(f"{_HF_HUB_PREFIX}{filename}")
        descriptor = o3d.data.DataDescriptor(urls, self.archive_md5)
        super().__init__(
            prefix,
            descriptor,
            EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root,
        )
