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

"""Offline contracts for the registered tabletop demo archive."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import open3d as o3d
import pytest

from embodichain.data import download
from embodichain.data.assets import demo_assets
from embodichain.data.dataset import EmbodiChainDataset, get_data_class

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
def test_tabletop_descriptor_respects_source_and_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
    custom_root: bool,
) -> None:
    """Configured hosting is tried first and identical Hub URLs are deduplicated."""
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
    monkeypatch.setattr(demo_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", default)
    monkeypatch.setattr(demo_assets, "EMBODICHAIN_DOWNLOAD_PREFIX", prefix)
    monkeypatch.setattr(EmbodiChainDataset, "__init__", initialize)
    with pytest.raises(Intercepted):
        demo_assets.TabletopFourBases(data_root=selected)
    name, descriptor, root = captured[0]
    assert name == "TabletopFourBases"
    assert root == (selected if custom_root else default)
    urls = [prefix + "demo/tabletop_four_bases.zip"]
    if prefix != _HUB_PREFIX:
        urls.append(_HUB_PREFIX + "demo/tabletop_four_bases.zip")
    assert descriptor.urls == urls
    assert descriptor.md5 == "103d891625d1f9f90e1393d048ea87c2"


@pytest.mark.no_sim
def test_tabletop_asset_is_listed_and_resolves_through_the_download_cli(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Paths and CLI names select the same registered demo dataset."""
    cls = get_data_class("TabletopFourBases")
    assert dict(download.get_registry()["demo"])["TabletopFourBases"] is cls
    download.cmd_list(Namespace(category="demo"))
    assert "TabletopFourBases" in capsys.readouterr().out
    captured = []
    monkeypatch.setattr(
        download,
        "download_asset",
        lambda name, asset_cls: captured.append((name, asset_cls)) or True,
    )
    download.cmd_download(Namespace(all=False, category=None, name="tabletopfourbases"))
    assert captured == [("tabletopfourbases", cls)]
