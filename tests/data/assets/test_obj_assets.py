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

"""Provider-free coverage for the PourWater object archive descriptor."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d
import pytest

from embodichain.data.assets import obj_assets
from embodichain.data.dataset import EmbodiChainDataset

_HUB_PREFIX = (
    "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
)


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "prefix",
    [
        _HUB_PREFIX,
        "https://hf-mirror.com/datasets/DexForceAI/embodichain_data/resolve/main/",
        "https://assets.example.test/",
    ],
)
@pytest.mark.parametrize("custom_root", [False, True])
def test_pour_water_descriptor_preserves_cache_and_source_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
    custom_root: bool,
) -> None:
    """Respect configured hosting first, without duplicate Hub fallbacks."""
    calls = []

    class DownloadIntercepted(Exception):
        """Stop before the native backend performs external I/O."""

    def initialize(
        self: EmbodiChainDataset,
        name: str,
        descriptor: o3d.data.DataDescriptor,
        root: str,
    ) -> None:
        calls.append((name, descriptor, root))
        raise DownloadIntercepted

    default_root = str(tmp_path / "default")
    override = str(tmp_path / "override") if custom_root else None
    monkeypatch.setattr(obj_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", default_root)
    monkeypatch.setattr(obj_assets, "EMBODICHAIN_DOWNLOAD_PREFIX", prefix)
    monkeypatch.setattr(EmbodiChainDataset, "__init__", initialize)
    # Returning from a mocked constructor without initializing the C++ base is
    # rejected by pybind11. Intercept the real download boundary by raising.
    with pytest.raises(DownloadIntercepted):
        obj_assets.PourWaterAssets(data_root=override)
    assert len(calls) == 1
    name, descriptor, root = calls[0]
    assert name == "PourWaterAssets"
    assert root == (override if custom_root else default_root)
    filename = "obj_assets/PourWaterAssets.zip"
    expected_urls = [f"{prefix}{filename}"]
    if prefix != _HUB_PREFIX:
        expected_urls.append(f"{_HUB_PREFIX}{filename}")
    assert descriptor.urls == expected_urls
    assert descriptor.md5 == "7267053763ed8e1b84f3da3e49d39f01"
