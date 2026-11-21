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

"""Immutable transport and shared discovery for the captured GS packages."""

from __future__ import annotations

import re

import pytest

from embodichain.data.assets import scene_assets
from embodichain.data.dataset import EmbodiChainDataset, get_data_class
from embodichain.data.download import get_registry


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "name,archive,checksum",
    [
        (
            "RoboSplatterOfficeGS",
            "robosplatter_office_v1.zip",
            "c36ff7550a47677479f22a21a8ee3c40",
        ),
        (
            "DiscoverseLab3GS",
            "discoverse_lab3_v1.zip",
            "ec9f17d4610e0dd5347ddb4815c531ad",
        ),
        (
            "RoboSimGSTabletopGS",
            "robosimgs_tabletop_v1.zip",
            "b6fc69397d64ccff611bff5712931b1a",
        ),
    ],
)
def test_gaussian_scene_uses_pinned_official_descriptor_and_shared_registry(
    monkeypatch, name, archive, checksum
):
    captured = []

    class Intercepted(Exception):
        pass

    def initialize(self, prefix, descriptor, root):
        captured.append((prefix, descriptor, root))
        raise Intercepted

    monkeypatch.setattr(EmbodiChainDataset, "__init__", initialize)
    monkeypatch.setattr(
        scene_assets,
        "EMBODICHAIN_DOWNLOAD_PREFIX",
        "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/",
    )
    cls = get_data_class(name)
    assert dict(get_registry()["scene"])[name] is cls
    with pytest.raises(Intercepted):
        cls(data_root="/explicit/scene/cache")
    prefix, descriptor, root = captured[0]
    assert prefix == name + "V1"
    assert root == "/explicit/scene/cache"
    assert descriptor.md5 == checksum
    assert len(descriptor.urls) == 1
    assert re.fullmatch(
        r"https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/[0-9a-f]{40}/scene_assets/gaussian_backgrounds_v1/"
        + archive,
        descriptor.urls[0],
    )


@pytest.mark.no_sim
def test_gaussian_scene_custom_mirror_keeps_source_and_pinned_hub_fallback(monkeypatch):
    captured = []

    class Intercepted(Exception):
        pass

    def initialize(self, prefix, descriptor, root):
        captured.append((prefix, descriptor, root))
        raise Intercepted

    monkeypatch.setattr(EmbodiChainDataset, "__init__", initialize)
    monkeypatch.setattr(
        scene_assets, "EMBODICHAIN_DOWNLOAD_PREFIX", "https://assets.example.test/"
    )
    monkeypatch.setattr(scene_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", "/default/cache")
    with pytest.raises(Intercepted):
        scene_assets.RoboSplatterOfficeGS()
    _, descriptor, root = captured[0]
    assert root == "/default/cache"
    assert (
        descriptor.urls[0]
        == "https://assets.example.test/scene_assets/gaussian_backgrounds_v1/robosplatter_office_v1.zip"
    )
    assert "/resolve/main/" not in descriptor.urls[1]
    assert descriptor.urls[1].startswith(
        "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/"
    )
