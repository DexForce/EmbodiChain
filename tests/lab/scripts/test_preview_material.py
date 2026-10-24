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

"""Material-preview CLI selection, offline discovery and resource boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from embodichain.lab.scripts import preview_material

pytestmark = pytest.mark.no_sim
_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("option", ["--help", "--list"])
def test_catalog_commands_need_no_native_imports_or_downloads(tmp_path, option) -> None:
    script = f"""
import importlib.abc
import socket
import sys
sys.path.insert(0, {_ROOT.as_posix()!r})
class NoRendering(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {{'dexsim', 'torch', 'numpy', 'cv2', 'PIL', 'matplotlib'}}:
            raise AssertionError('Catalog query imported ' + fullname)
sys.meta_path.insert(0, NoRendering())
def no_network(*args, **kwargs):
    raise AssertionError('Catalog query used the network')
socket.socket.connect = no_network
from embodichain.cli.main import main
main(['preview-material', {option!r}])
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "EMBODICHAIN_DATA_ROOT": str(tmp_path / "assets")},
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    if option == "--list":
        records = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(records) == 24
        assert {record["source"] for record in records} == {"builtin", "resource"}
        assert "SoapBubbleInterior" not in {record["name"] for record in records}
    else:
        assert "usage: embodichain preview-material" in result.stdout
        assert "--material-params" in result.stdout
    assert not (tmp_path / "assets").exists()
    assert not (tmp_path / "outputs").exists()


def test_list_source_filter_is_offline(capsys) -> None:
    preview_material.cli(["--list", "--source", "builtin"])
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(records) == 6
    assert all(record["requires_download"] is False for record in records)


@pytest.fixture
def captures(monkeypatch):
    from embodichain.lab.sim import SimulationManager

    calls = []
    monkeypatch.setattr(SimulationManager, "flush_cleanup_queue", lambda: None)

    def render(args, configs, skipped):
        calls.append((args, configs, skipped))
        return []

    monkeypatch.setattr(preview_material, "_render_materials", render)
    return calls


def test_default_preview_inherits_authored_fields_and_requests_window(captures) -> None:
    preview_material.cli([])
    args, configs, skipped = captures[0]
    assert args.renderer == "rt" and args.headless is False
    assert configs[0].preset == "GlassClear" and configs[0].ior is None
    assert configs[0].parameters == {} and skipped == []


def test_overrides_keep_explicit_zero_false_and_none(captures) -> None:
    preview_material.cli(
        [
            "--material",
            "GlassClear",
            "--headless",
            "--material-params",
            '{"ior":null,"roughness":0,"parameters":{"thin_walled":false,"volume_attenuation_distance":8}}',
        ]
    )
    args, configs, skipped = captures[0]
    cfg = configs[0]
    assert args.headless and not skipped
    assert cfg.ior is None and cfg.roughness == 0
    assert cfg.parameters == {"thin_walled": False, "volume_attenuation_distance": 8}


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--all", "--material", "GlassClear"], "not allowed"),
        (["--material-params", "[]"], "JSON object"),
        (["--material-params", "{broken"], "JSON object"),
        (["--material-params", '{"uid":"caller"}'], "preview UIDs"),
        (["--spp", "0"], "positive integer"),
        (["--renderer", "no-render"], "invalid choice"),
        (["--headless", "--max-steps", "1"], "native-window"),
        (["--source", "builtin"], "requires --list or --all"),
        (["--compatible-only"], "requires --all"),
        (["--compare-native", "--material-params", '{"ior":1.6}'], "authored presets"),
        (["--material", "unknown"], "Unknown default material"),
        (["--renderer", "hybrid"], "requires renderer='rt'"),
        (["--all", "--renderer", "fast-rt"], "requires renderer='rt'"),
        (
            ["--material-params", '{"roughness":0,"parameters":{"roughness":0.1}}'],
            "Duplicate",
        ),
        (["--material-params", '{"parameters":{"transmission":-1}}'], "Invalid visual"),
        (
            ["--material-params", '{"material_type":"BSDF"}'],
            "material_type",
        ),
        (
            [
                "--material",
                "PlasticMatteBlue",
                "--renderer",
                "hybrid",
                "--material-params",
                '{"parameters":{"clearcoat":0}}',
            ],
            "requires renderer='rt'",
        ),
        (
            [
                "--all",
                "--compatible-only",
                "--renderer",
                "hybrid",
                "--material-params",
                '{"parameters":{"clearcoat":0}}',
            ],
            "No selected presets",
        ),
    ],
)
def test_invalid_requests_fail_before_rendering(
    captures, capsys, argv, message
) -> None:
    with pytest.raises(SystemExit) as error:
        preview_material.cli(argv)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not captures


@pytest.mark.parametrize("renderer", ["hybrid", "fast-rt"])
def test_explicit_compatible_batch_reports_every_skipped_preset(
    captures, capsys, renderer
) -> None:
    preview_material.cli(["--all", "--compatible-only", "--renderer", renderer])
    args, configs, skipped = captures[0]
    assert args.all
    assert {cfg.preset for cfg in configs} == {
        "PlasticMatteBlue",
        "PlasticGlossyRed",
        "GoldMetal",
        "MirrorMetal",
        "BrickRed",
        "WoodOak",
        "CeramicTileWhite",
        "MarbleWhite",
        "AsphaltDark",
        "FabricGray",
        "LeatherBrown",
        "PlasticABSBlack",
        "RubberBlack",
        "SteelBrushed",
    }
    assert {item["name"] for item in skipped} == {
        "GlassClearSurface",
        "GlassFrostedSurface",
        "ChromePolished",
        "GoldPolished",
        "CopperPolished",
        "GlassClear",
        "GlassFrosted",
        "GlassSmoke",
        "SkinHumanLight",
        "SoapBubble",
    }
    stderr = capsys.readouterr().err
    assert all(f"Skipping {item['name']}:" in stderr for item in skipped)


def test_render_failure_still_drains_native_cleanup(monkeypatch) -> None:
    from embodichain.lab.sim import SimulationManager

    flushed = []

    def fail(*args):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(preview_material, "_render_materials", fail)
    monkeypatch.setattr(
        SimulationManager, "flush_cleanup_queue", lambda: flushed.append(True)
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        preview_material.cli(["--headless", "--material", "PlasticMatteBlue"])
    assert flushed == [True]
