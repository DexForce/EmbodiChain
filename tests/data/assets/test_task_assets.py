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

"""Task-asset descriptors, cache selection and public registry contracts."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import open3d as o3d
import pytest

from embodichain.data import download
from embodichain.data.assets import task_assets
from embodichain.data.dataset import EmbodiChainDataset, get_data_class

pytestmark = pytest.mark.no_sim

_HUB = "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
_ASSETS = (
    ("PourCupAssets", "f97857b639189be4daf717a873ab84cf"),
    ("PourCupDemoData", "8eb96cb756ac08efe403417f50ad2bf2"),
)


@pytest.mark.parametrize("name,md5", _ASSETS)
@pytest.mark.parametrize(
    "prefix", [_HUB, "https://hf-mirror.com/assets/", "https://example.test/"]
)
@pytest.mark.parametrize("custom_root", [False, True])
def test_task_archive_descriptor_uses_configured_source_and_locked_cache(
    name: str,
    md5: str,
    prefix: str,
    custom_root: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class DownloadIntercepted(Exception):
        """Stop before the Open3D base performs network I/O."""

    def initialize(
        self: EmbodiChainDataset,
        asset: str,
        descriptor: o3d.data.DataDescriptor,
        root: str,
    ) -> None:
        calls.append((asset, descriptor, root))
        raise DownloadIntercepted

    default = str(tmp_path / "default")
    override = str(tmp_path / "custom") if custom_root else None
    monkeypatch.setattr(task_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", default)
    monkeypatch.setattr(task_assets, "EMBODICHAIN_DOWNLOAD_PREFIX", prefix)
    monkeypatch.setattr(EmbodiChainDataset, "__init__", initialize)
    with pytest.raises(DownloadIntercepted):
        getattr(task_assets, name)(data_root=override)
    assert len(calls) == 1
    asset, descriptor, root = calls[0]
    assert asset == name
    assert root == (override or default)
    relative = f"task_assets/{name}.zip"
    assert descriptor.urls == [f"{prefix}{relative}"] + (
        [] if prefix == _HUB else [f"{_HUB}{relative}"]
    )
    assert descriptor.md5 == md5


@pytest.mark.parametrize("name,_", _ASSETS)
def test_task_assets_are_shared_by_config_and_cli_registry(
    name: str, _: str, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    registry = download.get_registry()
    cls = get_data_class(name)
    assert dict(registry["task"])[name] is cls
    assert all(not key.startswith("_") for key, _ in registry["task"])
    download.cmd_list(Namespace(category="task"))
    assert name in capsys.readouterr().out
    calls = []
    monkeypatch.setattr(
        download,
        "download_asset",
        lambda key, value: calls.append((key, value)) or True,
    )
    download.cmd_download(Namespace(all=False, category=None, name=name.lower()))
    assert calls == [(name.lower(), cls)]


@pytest.mark.parametrize(
    "source", [_HUB, _HUB.replace("huggingface.co", "hf-mirror.com")]
)
def test_rj45_fixture_uses_pinned_archive_and_versioned_cache(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.extensions.robots.franka import rj45_assets

    class DownloadIntercepted(Exception):
        """Stop before the Open3D base performs network I/O."""

    def initialize(
        self: EmbodiChainDataset,
        prefix: str,
        descriptor: o3d.data.DataDescriptor,
        root: str,
    ) -> None:
        assert prefix == "Rj45FixtureV1"
        assert root == str(tmp_path)
        archive = (
            "/datasets/DexForceAI/embodichain_data/resolve/"
            "dcb448da52ea3df5df5a249e9ff977acafead0e4/task_assets/Rj45FixtureV1.zip"
        )
        assert descriptor.urls == (
            ["https://huggingface.co" + archive]
            if source == _HUB
            else ["https://hf-mirror.com" + archive, "https://huggingface.co" + archive]
        )
        assert descriptor.md5 == "1d22a1fbad7a0ae10fd2c8da0cf0f8fd"
        raise DownloadIntercepted

    monkeypatch.setattr(task_assets, "EMBODICHAIN_DOWNLOAD_PREFIX", source)
    monkeypatch.setattr(EmbodiChainDataset, "__init__", initialize)
    cls = get_data_class("Rj45Fixture")
    assert cls is task_assets.Rj45Fixture is rj45_assets.Rj45Fixture
    assert dict(download.get_registry()["task"])["Rj45Fixture"] is cls
    with pytest.raises(DownloadIntercepted):
        cls(data_root=str(tmp_path))
