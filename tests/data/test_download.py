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

"""Regression coverage for asset discovery through the download CLI."""

from __future__ import annotations

from argparse import Namespace

import pytest

from embodichain.data import download
from embodichain.data.dataset import get_data_class

LOCOMOTION_ASSETS = (
    "ANYmalCLocomotion",
    "HumanoidRun",
    "MicroDuckLocomotion",
    "UnitreeG1Locomotion",
    "UnitreeGo1Locomotion",
    "UnitreeGo2Locomotion",
    "UnitreeH1_2Locomotion",
)


@pytest.mark.no_sim
def test_rubiks_cube_asset_is_shared_by_config_and_download_registries() -> None:
    """Task configs and the data CLI resolve the same Rubik's-cube bundle."""
    demo_assets = dict(download.get_registry()["demo"])

    assert demo_assets["RubiksCube"] is get_data_class("RubiksCube")


@pytest.mark.no_sim
def test_robot_asset_listing_includes_locomotion_and_existing_robots(capsys) -> None:
    """Users see all public robot assets without exposing helper classes."""
    registry = download.get_registry()
    names = [name for name, _ in registry["robot"]]
    assert set(LOCOMOTION_ASSETS).issubset(names)
    assert {"UnitreeH1", "UnitreeH1Usd", "CartPole"}.issubset(names)
    assert names == sorted(set(names))
    assert all(not name.startswith("_") for name in names)
    download.cmd_list(Namespace(category="robot"))
    output = capsys.readouterr().out
    for name in LOCOMOTION_ASSETS:
        assert name in output
    assert "_LocomotionAsset" not in output


@pytest.mark.no_sim
@pytest.mark.parametrize("name", LOCOMOTION_ASSETS)
def test_download_by_name_uses_the_same_class_as_path_resolution(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI resolves every new asset through the shared public registry."""
    calls = []
    monkeypatch.setattr(
        download,
        "download_asset",
        lambda asset_name, cls: calls.append((asset_name, cls)) or True,
    )
    download.cmd_download(Namespace(all=False, category=None, name=name.lower()))
    assert calls == [(name.lower(), get_data_class(name))]


@pytest.mark.no_sim
def test_download_reports_extraction_failure(capsys) -> None:
    """A broken archive is not reported as a successful download."""

    class BrokenAsset:
        def __init__(self) -> None:
            raise RuntimeError("Extraction failed")

    assert download.download_asset("BrokenAsset", BrokenAsset) is False
    captured = capsys.readouterr()
    assert "Extraction failed" in captured.err
    assert "ready" not in captured.out


@pytest.mark.no_sim
def test_download_command_finishes_batch_and_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Scripts receive failure even when later assets download successfully."""
    calls = []
    monkeypatch.setattr(
        download,
        "get_registry",
        lambda: {"robot": [("Broken", object), ("Good", object)]},
    )

    def run_download(name: str, cls: type) -> bool:
        calls.append(name)
        return name == "Good"

    monkeypatch.setattr(download, "download_asset", run_download)
    with pytest.raises(SystemExit) as exc:
        download.cmd_download(Namespace(all=False, category="robot", name=None))
    assert exc.value.code == 1
    assert calls == ["Broken", "Good"]
    assert "Failed assets: Broken" in capsys.readouterr().err
