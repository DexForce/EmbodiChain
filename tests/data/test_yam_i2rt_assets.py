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

"""YAM participates in the existing data registry and cache protocol."""

from __future__ import annotations

from pathlib import Path
import importlib

import pytest

from embodichain.data import get_data_path
from embodichain.data.assets import YamI2rt
from embodichain.data import dataset
from embodichain.data.download import find_asset_class, get_registry

robot_assets = importlib.import_module("embodichain.data.assets.robot_assets")

pytestmark = pytest.mark.no_sim


def test_yam_is_discoverable_from_data_path_and_robot_download_cli() -> None:
    assert dataset.get_data_class("YamI2rt") is YamI2rt
    assert find_asset_class("YamI2rt", get_registry()) == ("robot", YamI2rt)


@pytest.mark.parametrize("custom_root", [False, True])
def test_yam_uses_the_shared_checked_archive_and_cache_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, custom_root: bool
) -> None:
    captured = {}

    class Captured(Exception):
        pass

    def capture(self, prefix, descriptor, path):
        captured.update(prefix=prefix, descriptor=descriptor, root=path)
        raise Captured

    monkeypatch.setattr(dataset.EmbodiChainDataset, "__init__", capture)
    monkeypatch.setattr(robot_assets, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(tmp_path))
    root = str(tmp_path / "custom") if custom_root else None
    with pytest.raises(Captured):
        YamI2rt(data_root=root)
    assert captured["prefix"] == "YamI2rt"
    assert captured["root"] == (root or str(tmp_path))
    assert captured["descriptor"].md5 == YamI2rt.archive_md5
    assert captured["descriptor"].urls[0].endswith("robot_assets/YamI2rtV1.zip")


def test_prepared_offline_cache_is_reused_without_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from embodichain.data import constants

    urdf = tmp_path / "YamI2rt/yam_i2rt.urdf"
    urdf.parent.mkdir()
    urdf.write_text('<robot name="offline"/>')
    monkeypatch.setattr(constants, "EMBODICHAIN_DEFAULT_DATA_ROOT", str(tmp_path))

    def no_download(*args, **kwargs):
        raise AssertionError("A prepared offline cache must not download assets")

    monkeypatch.setattr(dataset, "get_data_class", no_download)
    assert get_data_path("YamI2rt/yam_i2rt.urdf") == str(urdf)
