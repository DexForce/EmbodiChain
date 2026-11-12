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
from importlib import import_module, util
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
        if spec["category"] == "eef" or name == "YamI2rt":
            # EEFs and new model-owned assets have no legacy core import surface.
            continue
        module, _, symbol = spec["class_path"].partition(":")
        canonical = getattr(import_module(module), symbol)
        legacy = (
            "w1_assets"
            if name.startswith("DexforceW1")
            else (
                "locomotion_assets"
                if name in LOCOMOTION_ASSETS
                else "so101_assets" if name == "SO101FollowerAssets" else "robot_assets"
            )
        )
        if name in {"AgibotG1", "AgilexPiperX"}:
            assert not hasattr(import_module("embodichain.data.assets." + legacy), name)
            assert not hasattr(assets, name)
            assert not hasattr(assets, name)
        else:
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
        "embodichain_robots.brainco.revo1.assets",
        "embodichain_robots.agilex.pika.assets",
        "embodichain_robots.agibot.g1.assets",
        "embodichain_robots.franka.assets",
        "embodichain_robots.the_robot_studio.so101.assets",
        "embodichain_robots.unitree.go2.assets",
        "embodichain_robots.anybotics.anymal_c.assets",
        "embodichain_robots.pollen_robotics.microduck.assets",
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
        "AgibotG1",
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


# Pinned EEF descriptors. Existing task references retain their archive/cache
# identities; new bundles explicitly select their versioned cache prefixes.
EEF_ARCHIVES = {
    "DH_PGC_140_50": ("DH_PGC_140_50.zip", "c2a642308a76e99b1b8b7cb3a11c5df3"),
    "DH_PGI_140_80": ("DH_PGI_140_80.zip", "05a1a08b13c6250cc12affeeda3a08ba"),
    "DH_PGC_140_50_M": ("DH_PGC_140_50_M.zip", "3a9ab5f32639e03afb38dc033b44bb62"),
    "DH_AG95": ("DH_AG95.zip", "34b6f3c2f649697ea7f12814b6a50529"),
    "ZH_CTM2F110": ("ZH_CTM2F110.zip", "0e7c3310425609797fe010b2a76fe465"),
    "BrainCoHandRevo1": ("BrainCoHandRevo01.zip", "ff9ac77e7e1493fd32d40c87fecbee6c"),
    "InspireHand": ("InspireHand.zip", "c60132a6f03866fb021cca5b6d72845e"),
    "Robotiq": ("Robotiq_V2.zip", "9bcbf92946b200c903938d59323e7b2c"),
    "Robotiq2F85": ("Robotiq2F85.zip", "53ecbf2c953f43f1134aa7223e592292"),
    "WheelTecFA2F": ("WheelTecFA2F.zip", "feaf13f25b1c6ce58d011b1f2fa72f58"),
    "PikaGripper": ("PikaGripper.zip", "8a9d0779e7dbfa7bf5d4713175df014e"),
    "SharpaWave": ("SharpaWaveV1.zip", "6a03c433d16cfc307e0c8dda2b70006e"),
    "WujiHand": ("WujiHandV2.zip", "d00115e853cfe0e9c4b096c26740354b"),
}


@pytest.mark.no_sim
def test_eef_catalog_preserves_ids_without_legacy_exports(monkeypatch, capsys) -> None:
    """EEF IDs remain discoverable once after removing core Python imports."""
    import embodichain.data.assets as assets
    from embodichain.data.dataset import EmbodiChainDataset

    def unexpected_download(*args, **kwargs):
        pytest.fail("Asset discovery must not construct a dataset")

    monkeypatch.setattr(EmbodiChainDataset, "__init__", unexpected_download)
    assert util.find_spec("embodichain.data.assets.eef_assets") is None
    entries = download.get_registry()["eef"]
    expected_ids = EEF_ARCHIVES.keys() | {"AgxGripper"}
    assert {name for name, _ in entries} == expected_ids
    assert len(entries) == len(expected_ids)
    for name, cls in entries:
        assert cls is get_data_class(name)
        assert cls.__module__.startswith("embodichain_robots.")
        assert not hasattr(assets, name)
    assert get_data_class("PikaGripper").__module__ == (
        "embodichain_robots.agilex.pika.assets"
    )
    download.cmd_list(Namespace(category="eef"))
    output = capsys.readouterr().out
    for name in expected_ids:
        assert name in output


@pytest.mark.no_sim
@pytest.mark.parametrize("name", EEF_ARCHIVES)
def test_eef_download_descriptors_and_cache_prefix_are_preserved(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Existing asset references retain their exact archive and cache identity."""
    from embodichain.data.constants import (
        EMBODICHAIN_DEFAULT_DATA_ROOT,
        EMBODICHAIN_DOWNLOAD_PREFIX,
    )
    from embodichain.data.dataset import EmbodiChainDataset

    captured = []

    class Captured(Exception):
        pass

    def capture(self, prefix, descriptor, root):
        captured.append((prefix, descriptor, root))
        # Stop before the native DownloadDataset constructor performs I/O.
        raise Captured

    monkeypatch.setattr(EmbodiChainDataset, "__init__", capture)
    cls = get_data_class(name)
    with pytest.raises(Captured):
        cls()
    with pytest.raises(Captured):
        cls(data_root=str(tmp_path))
    filename, md5 = EEF_ARCHIVES[name]
    hub = "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
    expected_urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}eef_assets/{filename}"]
    if (
        name in {"PikaGripper", "SharpaWave", "WujiHand"}
        and EMBODICHAIN_DOWNLOAD_PREFIX != hub
    ):
        expected_urls.append(f"{hub}eef_assets/{filename}")
    for (prefix, descriptor, root), expected_root in zip(
        captured, (EMBODICHAIN_DEFAULT_DATA_ROOT, str(tmp_path))
    ):
        expected_prefix = {"SharpaWave": "SharpaWaveV1", "WujiHand": "WujiHandV2"}.get(
            name, name
        )
        assert prefix == expected_prefix
        assert list(descriptor.urls) == expected_urls
        assert descriptor.md5 == md5
        assert root == expected_root
