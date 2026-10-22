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

"""Robot CLI listing stays lightweight while exposing every catalog selector."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from embodichain import robot_registry as registry
from embodichain.cli import list_robot
from embodichain.cli import main as cli


@pytest.fixture
def catalogs(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    entries: list[Any] = []
    registry._robot_specs.cache_clear()
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_: entries)
    yield entries
    registry._robot_specs.cache_clear()


def _column_text(output: str, column: int) -> str:
    """Reassemble column contents across wrapped table lines."""
    return "".join(
        cells[column].strip()
        for cells in (line.split("│") for line in output.splitlines())
        if len(cells) == 7
    )


@pytest.mark.parametrize("terminal_width", [80, 120])
def test_list_robot_dispatch_reads_installed_catalogs(
    catalogs: list[Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    terminal_width: int,
) -> None:
    """The dispatcher lists canonical IDs and the distinct UR variant presets."""
    monkeypatch.setenv("COLUMNS", str(terminal_width))
    catalogs.append(
        metadata.EntryPoint(
            name="embodichain_robots",
            value="embodichain_robots._catalog:ROBOT_CONFIGS",
            group="embodichain.robots",
        )
    )

    cli.main(["list-robot"])

    output = capsys.readouterr().out
    assert output.startswith("╭")
    assert output.rstrip().endswith("╯")
    assert "EmbodiChain · Robot Catalog · 8 robots" in output
    assert "Robot ID" in output
    assert "Config Class" not in output
    assert "Variants" in output
    assert "Mimic Joints" in output
    assert "Control DOF" in output
    assert "Control Parts" in output
    expected_ids = {
        "AlohaMini",
        "TianjiMarvin",
        "DexforceW1",
        "CobotMagic",
        "FrankaPanda",
        "FrankaPandaPgi",
        "URRobot",
        "DualArmRobot",
        "ur3",
        "ur3e",
        "ur5",
        "ur5e",
        "ur10",
        "ur10e",
    }
    for robot_id in expected_ids - {"ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e"}:
        assert re.search(rf"│\s*{re.escape(robot_id)}\s*│", output)
    assert set(registry.list_robot_cfgs()) == expected_ids
    for alias in (
        "aloha_mini",
        "tianji_marvin",
        "dexforce_w1",
        "cobotmagic",
        "franka",
        "franka_panda",
        "ur",
    ):
        assert not re.search(rf"│\s*{re.escape(alias)}\s*│", output)
    assert output.count("│ URRobot") == 1
    for model in ("ur3", "ur3e", "ur5", "ur5e", "ur10", "ur10e"):
        assert model in output
    assert "eef (1+1m)" in output
    assert "fr3_finger_joint2" in _column_text(output, 5)


@pytest.mark.parametrize("terminal_width", [80, 120])
def test_list_robot_lists_external_metadata_without_importing_configuration(
    catalogs: list[Any],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    terminal_width: int,
) -> None:
    """Provider classes need not be importable to list their catalog records."""
    monkeypatch.setenv("COLUMNS", str(terminal_width))
    spec = {
        "class_path": "absent_robots.arm:ArmCfg",
        "defaults": {"with_gripper": False, "variant": "standard"},
    }
    original = deepcopy(spec)
    catalogs.append(SimpleNamespace(name="vendor", load=lambda: {"vendor.arm": spec}))

    list_robot.main([])

    output = capsys.readouterr().out
    assert "vendor.arm" in output
    assert "absent_robots.arm:ArmCfg" not in output
    variants = _column_text(output, 2)
    assert "variant='standard'" in variants
    assert "with_gripper=False" in variants
    assert "absent_robots" not in sys.modules
    assert spec == original


def test_list_robot_keeps_simulation_and_robot_modules_unloaded() -> None:
    """Cold listing must never import simulation, robot classes, or asset loading."""
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys
from importlib import metadata

class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("embodichain.lab")
            or fullname.startswith("embodichain.data")
            or (fullname.startswith("embodichain_robots.") and fullname != "embodichain_robots._catalog")
        ):
            raise AssertionError("Listing imported runtime module: " + fullname)

sys.meta_path.insert(0, RejectRuntimeImports())
from embodichain import robot_registry as registry
registry.metadata.entry_points = lambda **_: [
    metadata.EntryPoint(
        name="embodichain_robots",
        value="embodichain_robots._catalog:ROBOT_CONFIGS",
        group="embodichain.robots",
    )
]
from embodichain.cli.main import main
main(["list-robot"])
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "8 robots" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("terminal_width", "expected_width"),
    [(40, 80), (80, 80), (100, 100), (140, 120)],
)
def test_list_robot_wraps_long_catalog_metadata(
    monkeypatch: pytest.MonkeyPatch,
    terminal_width: int,
    expected_width: int,
) -> None:
    """Long vendor IDs, paths, and values fit the bounded terminal width."""
    monkeypatch.setattr(
        shutil,
        "get_terminal_size",
        lambda fallback=(100, 24): os.terminal_size((terminal_width, 24)),
    )
    entries = {
        "vendor.robot_with_an_intentionally_long_selector": {
            "class_path": "vendor.robot_with_a_long_module_path:LongRobotConfiguration",
            "defaults": {"variant": "a_long_variant_name_with_additional_options"},
        }
    }

    output = list_robot._format_robot_entries(entries, color=False)

    assert max(len(line) for line in output.splitlines()) <= expected_width
    assert "1 robot" in output


def test_list_robot_prints_control_metadata_without_loading_robot(
    catalogs: list[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A provider's control summary is printed directly from its catalog."""
    spec = {
        "class_path": "absent_robots:RobotCfg",
        "dof": 7,
        "control_parts": {"arm": 6, "tool": 1},
    }
    catalogs.append(SimpleNamespace(name="vendor", load=lambda: {"vendor.arm": spec}))

    list_robot.main([])

    output = capsys.readouterr().out
    assert re.search(r"│\s*vendor.arm\s*│\s*-\s*│\s*7\s*│", output)
    assert "arm (6 total)" in output
    assert "tool (1 total)" in output
    assert "absent_robots" not in sys.modules


def test_dual_arm_variant_rows_use_each_base_dof(catalogs: list[Any]) -> None:
    """The default UR and Panda compositions expose different independent DOFs."""
    catalogs.append(
        metadata.EntryPoint(
            name="official",
            value="embodichain_robots._catalog:ROBOT_CONFIGS",
            group="embodichain.robots",
        )
    )
    dual = registry._robot_specs()["DualArmRobot"]
    profiles = list_robot._variant_profiles(dual)

    assert len(profiles) == 2
    assert profiles[0][1]["dof"] == 12
    assert "ur10*" in profiles[0][0]
    assert profiles[1][0] == "FrankaPanda"
    assert profiles[1][1]["dof"] == 16
    assert len(profiles[1][1]["mimic_joints"]) == 2
    assert "left_eef (1+1m)" in list_robot._format_control_parts(profiles[1][1])


def test_dual_arm_composite_part_marks_overlapping_mimics(
    catalogs: list[Any],
) -> None:
    """Composite control groups must retain their mimic annotation without double counting."""
    spec = {
        "class_path": "absent:RobotCfg",
        "dof": 1,
        "control_parts": {"arm": 2},
        "mimic_joints": {"follower": "leader"},
        "control_part_mimics": {"arm": ("follower",)},
        "supports_dual_arm": True,
    }
    catalogs.append(SimpleNamespace(name="vendor", load=lambda: {"vendor.arm": spec}))

    profile = registry._robot_specs()["DualArmRobot"]["variants"]["vendor.arm"]

    assert profile["dof"] == 2
    assert len(profile["mimic_joints"]) == 2
    assert "dual_arm (2+2m)" in list_robot._format_control_parts(profile)


def test_w1_variants_show_body_releases_and_hand_specific_dofs(
    catalogs: list[Any],
) -> None:
    """W1 versions share summaries only when their hand configuration matches."""
    catalogs.append(
        metadata.EntryPoint(
            name="official",
            value="embodichain_robots._catalog:ROBOT_CONFIGS",
            group="embodichain.robots",
        )
    )
    spec = registry._robot_specs()["DexforceW1"]
    profiles = list_robot._variant_profiles(spec)

    assert len(profiles) == 6
    assert profiles[0][0] == "v021*, v022, v025\nBrainCo"
    assert profiles[0][1]["dof"] == 32
    by_hand = {label.split("\n")[1]: profile for label, profile in profiles}
    assert by_hand["DH_PGC"]["dof"] == 22
    assert by_hand["DH_PGC_M"]["dof"] == 22
    assert by_hand["bare"]["dof"] == 20
    assert any("v030" in label and label.endswith("\nbare") for label, _ in profiles)
    assert by_hand["L=DH_PGC,R=DH_PGC_M"]["dof"] == 22
    assert not any("L=BrainCo" in label or "R=BrainCo" in label for label in by_hand)


def test_list_robot_sorts_ids_and_catalog_defaults() -> None:
    """Display order is stable regardless of provider registration order."""
    output = list_robot._format_robot_entries(
        {
            "zeta": {"class_path": "robots:ZetaCfg"},
            "Alpha": {
                "class_path": "robots:AlphaCfg",
                "defaults": {"z": 2, "a": 1},
            },
        },
        color=False,
    )

    assert output.index("Alpha") < output.index("zeta")
    assert "a=1, z=2" in output
    assert re.search(r"│\s*zeta\s*│\s*-\s*│\s*-\s*│\s*-\s*│\s*-\s*│", output)


def test_list_robot_respects_tty_color_and_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TTY colors match list-task and NO_COLOR disables those escape codes."""
    entries = {"panda": {"class_path": "robots:PandaCfg"}}
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.delenv("NO_COLOR", raising=False)
    colored = list_robot._format_robot_entries(entries)
    for code in ("1;36", "1;32", "1;33"):
        assert f"\033[{code}m" in colored

    monkeypatch.setenv("NO_COLOR", "1")
    plain = list_robot._format_robot_entries(entries)

    assert "\033[" not in plain
    assert re.sub(r"\033\[[0-9;]*m", "", colored) == plain


def test_list_robot_redirected_output_has_no_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redirected listings remain readable without ANSI escape codes."""
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: False))
    monkeypatch.delenv("NO_COLOR", raising=False)

    output = list_robot._format_robot_entries(
        {"panda": {"class_path": "robots:PandaCfg"}}
    )

    assert "\033[" not in output


def test_list_robot_help_does_not_load_catalogs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Help works before catalog discovery or optional provider dependencies."""

    def fail_if_loaded() -> None:
        raise AssertionError("Help loaded robot catalogs")

    monkeypatch.setattr(list_robot, "_robot_specs", fail_if_loaded)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["list-robot", "--help"])

    assert exc_info.value.code == 0
    assert "mimic" in capsys.readouterr().out


def test_list_robot_rejects_unknown_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Argument errors name the selected command."""
    with pytest.raises(SystemExit) as exc_info:
        list_robot.main(["--unknown"])

    assert exc_info.value.code == 2
    assert "embodichain list-robot: error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("catalog", "message"),
    [
        (
            {"DualArmRobot": {"class_path": "vendor:DualArmCfg"}},
            "Duplicate robot ID",
        ),
        ({"invalid": {}}, "must declare a class_path"),
    ],
)
def test_list_robot_reports_invalid_catalog_as_cli_error(
    catalogs: list[Any],
    capsys: pytest.CaptureFixture[str],
    catalog: dict[str, Any],
    message: str,
) -> None:
    """Registry validation failures produce an actionable argparse error."""
    catalogs.append(SimpleNamespace(name="invalid", load=lambda: catalog))

    with pytest.raises(SystemExit) as exc_info:
        list_robot.main([])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert message in error
    assert "Traceback" not in error


def test_list_robot_empty_catalog_has_clear_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Empty registries report no robots without printing a misleading table."""
    monkeypatch.setattr(list_robot, "_robot_specs", lambda: {})

    list_robot.main([])

    assert capsys.readouterr().out == "No registered robots found.\n"
