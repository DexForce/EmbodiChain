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

"""Download registration and mirror coverage for the Pika gripper."""

from __future__ import annotations

from pathlib import Path

import open3d as o3d
import pytest

from embodichain.data.assets import eef_assets
from embodichain.data.dataset import EmbodiChainDataset, get_data_class
from embodichain.data.download import find_asset_class, get_registry

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
def test_pika_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str, custom_root: bool
) -> None:
    """Keep the archive checksum, cache override and Hub fallback aligned."""
    calls = []

    class DownloadIntercepted(Exception):
        """Stop before native I/O."""

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
    monkeypatch.setattr(eef_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", default_root)
    monkeypatch.setattr(eef_assets, "EMBODICHAIN_DOWNLOAD_PREFIX", prefix)
    monkeypatch.setattr(EmbodiChainDataset, "__init__", initialize)
    with pytest.raises(DownloadIntercepted):
        eef_assets.PikaGripper(data_root=override)
    name, descriptor, root = calls[0]
    assert name == "PikaGripper"
    assert root == (override if custom_root else default_root)
    filename = "eef_assets/PikaGripper.zip"
    urls = [f"{prefix}{filename}"]
    if prefix != _HUB_PREFIX:
        urls.append(f"{_HUB_PREFIX}{filename}")
    assert descriptor.urls == urls
    assert descriptor.md5 == "8a9d0779e7dbfa7bf5d4713175df014e"


@pytest.mark.no_sim
def test_pika_is_resolvable_and_listed_as_eef() -> None:
    """Expose the same class through data IDs, package exports and the CLI."""
    from embodichain.data.assets import PikaGripper

    assert get_data_class("PikaGripper") is PikaGripper
    assert find_asset_class("PikaGripper", get_registry()) == ("eef", PikaGripper)
