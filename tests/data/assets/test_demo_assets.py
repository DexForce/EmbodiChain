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
@pytest.mark.parametrize(
    "asset_name,filename,checksum,cache_prefix",
    [
        (
            "TabletopFourBases",
            "demo/tabletop_four_bases.zip",
            "103d891625d1f9f90e1393d048ea87c2",
            "TabletopFourBases",
        ),
        (
            "Real2SimCarton",
            "demo/real2sim_carton/real2sim_carton_v1.zip",
            "cef38008a3bae2d17a422bf2698cb4fb",
            "Real2SimCartonV1",
        ),
    ],
)
def test_tabletop_descriptor_respects_source_and_cache_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str,
    custom_root: bool,
    asset_name: str,
    filename: str,
    checksum: str,
    cache_prefix: str,
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
        getattr(demo_assets, asset_name)(data_root=selected)
    name, descriptor, root = captured[0]
    assert name == cache_prefix
    assert root == (selected if custom_root else default)
    urls = [prefix + filename]
    if prefix != _HUB_PREFIX:
        urls.append(_HUB_PREFIX + filename)
    assert descriptor.urls == urls
    assert descriptor.md5 == checksum


@pytest.mark.no_sim
@pytest.mark.parametrize("asset_name", ["TabletopFourBases", "Real2SimCarton"])
def test_tabletop_asset_is_listed_and_resolves_through_the_download_cli(
    monkeypatch: pytest.MonkeyPatch, capsys, asset_name: str
) -> None:
    """Paths and CLI names select the same registered demo dataset."""
    cls = get_data_class(asset_name)
    assert dict(download.get_registry()["demo"])[asset_name] is cls
    download.cmd_list(Namespace(category="demo"))
    assert asset_name in capsys.readouterr().out
    captured = []
    monkeypatch.setattr(
        download,
        "download_asset",
        lambda name, asset_cls: captured.append((name, asset_cls)) or True,
    )
    download.cmd_download(Namespace(all=False, category=None, name=asset_name.lower()))
    assert captured == [(asset_name.lower(), cls)]


@pytest.mark.no_sim
def test_carton_resource_manifest_rejects_changed_extracted_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    root = tmp_path / "resource"
    (root / "raw").mkdir(parents=True)
    source = root / "raw/trajectory.h5"
    source.write_bytes(b"pinned original evidence")
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "raw/trajectory.h5",
                        "size": 24,
                        "sha256": hashlib.sha256(
                            b"pinned original evidence"
                        ).hexdigest(),
                    }
                ]
            }
        )
    )
    # This test trusts a tiny fixture manifest instead of downloading the real one.
    monkeypatch.setattr(
        demo_assets.Real2SimCarton,
        "_manifest_sha256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    demo_assets.Real2SimCarton._verify_files(root)
    source.write_bytes(b"changed original evidenc")
    with pytest.raises(ValueError, match="resource is missing or changed"):
        demo_assets.Real2SimCarton._verify_files(root)
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest is missing or changed"):
        demo_assets.Real2SimCarton._verify_files(root)
