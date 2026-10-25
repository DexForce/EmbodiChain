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
from importlib import import_module
from types import ModuleType, SimpleNamespace
import subprocess
import sys

import pytest

from embodichain.data import download
from embodichain.data.dataset import get_data_class
from embodichain.data import _asset_registry


@pytest.fixture
def asset_catalogs(monkeypatch):
    original = _asset_registry.metadata.entry_points
    entries = list(original(group="embodichain.assets"))
    _asset_registry._asset_specs.cache_clear()
    monkeypatch.setattr(
        _asset_registry.metadata,
        "entry_points",
        lambda **kwargs: (
            entries
            if kwargs.get("group") == "embodichain.assets"
            else original(**kwargs)
        ),
    )
    yield entries
    _asset_registry._asset_specs.cache_clear()


def test_migrated_robot_assets_keep_legacy_class_identity() -> None:
    from embodichain_robots._asset_catalog import ROBOT_ASSETS
    import embodichain.data.assets as assets

    registry = download.get_registry()
    for name, spec in ROBOT_ASSETS.items():
        module, _, symbol = spec["class_path"].partition(":")
        canonical = getattr(import_module(module), symbol)
        legacy = (
            "w1_assets"
            if name.startswith("DexforceW1")
            else "so101_assets" if name == "SO101FollowerAssets" else "robot_assets"
        )
        assert (
            getattr(import_module("embodichain.data.assets." + legacy), name)
            is canonical
        )
        assert getattr(assets, name) is canonical
        assert get_data_class(name) is canonical
        assert dict(registry[spec["category"]])[name] is canonical
        assert canonical.__module__ == module
        assert (
            sum(
                asset_name == name
                for rows in registry.values()
                for asset_name, _ in rows
            )
            == 1
        )


@pytest.mark.parametrize("name", ["Franka", "franka"])
def test_asset_catalog_rejects_duplicate_ids(asset_catalogs, name) -> None:
    asset_catalogs.append(
        SimpleNamespace(
            name="duplicate",
            load=lambda: {
                name: {"class_path": "unavailable:Dataset", "category": "robot"},
            },
        )
    )
    with pytest.raises(ValueError, match="Duplicate asset ID"):
        _asset_registry._asset_specs()


def test_external_asset_category_is_available_to_download_cli(
    asset_catalogs, monkeypatch, capsys
) -> None:
    from embodichain.data.dataset import EmbodiChainDataset

    class ExternalAsset(EmbodiChainDataset):
        def __init__(self):
            pytest.fail("Listing must not construct or download assets")

    module = ModuleType("test_external_asset_provider")
    module.ExternalAsset = ExternalAsset
    monkeypatch.setitem(sys.modules, module.__name__, module)
    asset_catalogs.append(
        SimpleNamespace(
            name="external",
            load=lambda: {
                "ExternalAsset": {
                    "class_path": module.__name__ + ":ExternalAsset",
                    "category": "external",
                },
            },
        )
    )
    assert get_data_class("ExternalAsset") is ExternalAsset
    download.main(["list", "--category", "external"])
    assert "ExternalAsset" in capsys.readouterr().out


def test_asset_resolution_preserves_explicit_extra_modules(monkeypatch) -> None:
    module = ModuleType("test_extra_asset_module")
    module.CustomDataset = type("CustomDataset", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert get_data_class("CustomDataset", [module.__name__]) is module.CustomDataset
    with pytest.raises(AttributeError, match="CustomDataset"):
        get_data_class("CustomDataset")


@pytest.mark.parametrize(
    "first_module",
    [
        "embodichain_robots.arx.arx5.assets",
        "embodichain_robots.franka.assets",
        "embodichain_robots.the_robot_studio.so101.assets",
    ],
)
def test_asset_imports_do_not_load_robot_configuration(first_module: str) -> None:
    script = """
import importlib.abc
import sys
class RejectConfigs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('embodichain_robots.') and (fullname.endswith('.cfg') or fullname in ('embodichain_robots.franka.panda', 'embodichain_robots.universal_robots.ur_robot')):
            raise AssertionError('Asset discovery imported robot config: ' + fullname)
sys.meta_path.insert(0, RejectConfigs())
from embodichain_robots._asset_catalog import ROBOT_ASSETS
assert 'open3d' not in sys.modules
import importlib
importlib.import_module(sys.argv[1])
from embodichain.data.dataset import EmbodiChainDataset, get_data_class
def reject(*args, **kwargs):
    raise AssertionError('Asset lookup downloaded data')
EmbodiChainDataset.__init__ = reject
for name in ROBOT_ASSETS:
    assert issubclass(get_data_class(name), EmbodiChainDataset)
assert 'embodichain.lab.sim' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
def test_pour_water_assets_are_visible_to_configs_and_download_cli(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The task's asset prefix and object CLI select the same dataset class."""
    objects = dict(download.get_registry()["obj"])
    cls = get_data_class("PourWaterAssets")
    assert objects["PourWaterAssets"] is cls
    download.cmd_list(Namespace(category="obj"))
    assert "PourWaterAssets" in capsys.readouterr().out
    calls = []
    monkeypatch.setattr(
        download,
        "download_asset",
        lambda name, asset_cls: calls.append((name, asset_cls)) or True,
    )
    download.cmd_download(Namespace(all=False, category=None, name="pourwaterassets"))
    assert calls == [("pourwaterassets", cls)]


@pytest.mark.no_sim
def test_robot_asset_listing_includes_locomotion_and_existing_robots(capsys) -> None:
    """Users see all public robot assets without exposing helper classes."""
    registry = download.get_registry()
    names = [name for name, _ in registry["robot"]]
    assert set(LOCOMOTION_ASSETS).issubset(names)
    assert {
        "UnitreeH1",
        "UnitreeH1Usd",
        "CartPole",
        "FlexivRizon4S",
        "SO101FollowerAssets",
    }.issubset(names)
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


@pytest.mark.no_sim
def test_standard_w1_v030_asset_is_discoverable_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.data import get_data_class
    from embodichain.data.assets.w1_assets import DexforceW1V030
    from embodichain.data.dataset import EmbodiChainDataset

    def forbid_download(*args, **kwargs):
        raise AssertionError("Asset discovery must not construct a dataset")

    monkeypatch.setattr(EmbodiChainDataset, "__init__", forbid_download)
    assert get_data_class("DexforceW1V030") is DexforceW1V030
    assert ("DexforceW1V030", DexforceW1V030) in download.get_registry()["w1"]
