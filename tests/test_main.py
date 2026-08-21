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

from __future__ import annotations

from collections.abc import Sequence

import pytest

from embodichain import __main__ as cli

EXPECTED_COMMANDS = {
    "analyze-workspace",
    "annotate-grasp",
    "benchmark",
    "data",
    "decompose-urdf",
    "preview-asset",
    "preview_lerobot_data",
    "run-env",
    "scene-engine",
    "task-engine",
    "simready",
    "train-rl",
    "preview-scene",
    "workspace-cache",
}


def test_all_public_commands_are_registered() -> None:
    """Every documented public CLI should have one unified command."""
    assert {command.name for command in cli.COMMANDS} == EXPECTED_COMMANDS


def test_root_help_does_not_import_command_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level help should remain lightweight."""

    def fail_if_loaded(_: str) -> None:
        raise AssertionError("Top-level help loaded a command module")

    monkeypatch.setattr(cli, "_load_handler", fail_if_loaded)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0


def test_dispatch_forwards_subcommand_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected command should receive all arguments after its name."""
    received: list[str] = []

    def fake_handler(argv: Sequence[str] | None = None) -> None:
        received.extend(argv or [])

    monkeypatch.setattr(cli, "_load_handler", lambda _: fake_handler)

    cli.main(["preview-asset", "--asset_path", "robot.urdf", "--headless"])

    assert received == ["--asset_path", "robot.urdf", "--headless"]


def test_subcommand_help_uses_complete_command_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Subcommand help should show real arguments instead of an empty shell."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["simready", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--input_dir" in output
    assert "--output_root" in output
    assert "--category" in output


def test_preview_scene_help_includes_output_and_viser_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preview Scene should expose its required path and optional Viser settings."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["preview-scene", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--output_root" in output
    assert "--viser" in output


def test_preview_lerobot_data_help_includes_validation_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The unified dataset preview exposes its path and validation arguments."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["preview_lerobot_data", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: embodichain preview_lerobot_data" in output
    assert "dataset_root" in output
    assert "--expect-segments" in output
    assert "--latest" in output


def test_nested_benchmark_help_uses_suite_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Nested benchmark help should include suite-specific arguments."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["benchmark", "rl", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--tasks" in output
    assert "--algorithms" in output
    assert "--rebuild-report-only" in output
