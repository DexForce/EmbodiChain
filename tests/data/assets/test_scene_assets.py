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

"""Offline registration and hosting contracts for the SimReady room bundles."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d
import pytest

from embodichain.data.assets import scene_assets
from embodichain.data.dataset import EmbodiChainDataset, get_data_class
from embodichain.data.download import get_registry

_HUB = "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"


@pytest.mark.no_sim
@pytest.mark.parametrize("custom_root", [False, True])
@pytest.mark.parametrize("prefix", [_HUB, "https://assets.example.test/"])
@pytest.mark.parametrize(
    "asset, filename, md5",
    [
        ("WoodLab", "wood_lab.zip", "c549494006de4ca6d0f6cafd914df890"),
        (
            "IndustrialWorkshop",
            "industrial_workshop.zip",
            "a74f3fc598af42f0cba2947fc37ec186",
        ),
        ("DemoStudio", "demo_studio.zip", "0e74b55b03f980bcac2c3e254e9c9188"),
        ("CleanLab", "clean_lab.zip", "3e0c7a820aa5a72bddf6292be85ee6a4"),
    ],
)
def test_room_registration_and_download_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asset: str,
    filename: str,
    md5: str,
    prefix: str,
    custom_root: bool,
) -> None:
    """Room IDs resolve in the scene catalog and select verified archives."""
    cls = getattr(scene_assets, asset)
    assert get_data_class(asset) is cls
    assert dict(get_registry()["scene"])[asset] is cls
    captured = []

    class Intercepted(Exception):
        pass

    def initialize(
        self: EmbodiChainDataset,
        name: str,
        descriptor: o3d.data.DataDescriptor,
        root: str,
    ) -> None:
        captured.append((name, descriptor, root))
        raise Intercepted

    default = str(tmp_path / "default")
    selected = str(tmp_path / "custom") if custom_root else None
    monkeypatch.setattr(scene_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", default)
    monkeypatch.setattr(scene_assets, "EMBODICHAIN_DOWNLOAD_PREFIX", prefix)
    monkeypatch.setattr(EmbodiChainDataset, "__init__", initialize)
    with pytest.raises(Intercepted):
        cls(data_root=selected)
    name, descriptor, root = captured[0]
    assert name == asset
    assert root == (selected if custom_root else default)
    relative = "scene_assets/" + filename
    urls = [prefix + relative]
    if prefix != _HUB:
        urls.append(_HUB + relative)
    assert descriptor.urls == urls
    assert descriptor.md5 == md5
