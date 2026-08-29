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

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

import embodichain.cli.list_task as list_task
import embodichain.cli.main as cli
from embodichain import __main__ as entrypoint

EXPECTED_COMMANDS = {
    "analyze-workspace",
    "annotate-grasp",
    "benchmark",
    "data",
    "decompose-urdf",
    "list-task",
    "preview-asset",
    "preview_lerobot_data",
    "run-env",
    "scene-engine",
    "simready",
    "train-rl",
    "preview-scene",
    "workspace-cache",
}


def test_all_public_commands_are_registered() -> None:
    """Every documented public CLI should have one unified command."""
    assert {command.name for command in cli.COMMANDS} == EXPECTED_COMMANDS
    assert entrypoint.COMMANDS is cli.COMMANDS
    assert entrypoint.Command is cli.Command
    assert entrypoint.build_parser is cli.build_parser
    assert entrypoint.main is cli.main


def test_workspace_cache_command_uses_dedicated_cli_adapter() -> None:
    """The workspace cache command resolves its dedicated CLI adapter."""
    command = next(
        command for command in cli.COMMANDS if command.name == "workspace-cache"
    )

    assert command.target == "embodichain.cli.workspace_cache:main"


def test_list_task_command_uses_dedicated_cli_adapter() -> None:
    """The task listing command resolves its dedicated CLI adapter."""
    command = next(command for command in cli.COMMANDS if command.name == "list-task")

    assert command.target == "embodichain.cli.list_task:main"


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


def test_run_task_alias_dispatches_to_run_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run-task forwards to the canonical run-env handler unchanged."""
    loaded_targets: list[str] = []
    received: list[str] = []

    def fake_handler(argv: Sequence[str] | None = None) -> None:
        received.extend(argv or [])

    def load_handler(target: str):
        loaded_targets.append(target)
        return fake_handler

    monkeypatch.setattr(cli, "_load_handler", load_handler)

    cli.main(["run-task", "--gym_config", "task.yaml", "--headless"])

    assert loaded_targets == ["embodichain.lab.scripts.run_env:cli"]
    assert received == ["--gym_config", "task.yaml", "--headless"]


def test_list_task_discovers_and_prints_task_tree(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """List-task renders folder hierarchy and precise task capabilities."""
    from embodichain.lab.gym.utils import registration

    discovery_calls: list[None] = []
    monkeypatch.setattr(
        registration,
        "discover_task_packages",
        lambda: discovery_calls.append(None),
    )
    monkeypatch.setattr(
        list_task,
        "_collect_environment_entries",
        lambda: [
            list_task._EnvironmentListEntry(
                "CartPoleRL",
                ("classic_control", "cart_pole"),
                {list_task._RL},
            ),
            list_task._EnvironmentListEntry(
                "HandOver-v1",
                ("manipulation", "hand_over"),
                {list_task._TASK_PROGRAM},
            ),
            list_task._EnvironmentListEntry(
                "BlocksRankingRGB-v1",
                ("manipulation", "tableware", "blocks_ranking_rgb"),
                {list_task._HANDWRITTEN_DEMO},
            ),
            list_task._EnvironmentListEntry(
                "StackCups-v1",
                ("manipulation", "tableware", "stack_cups"),
                set(),
            ),
        ],
    )

    cli.main(["list-task"])

    assert discovery_calls == [None]
    assert capsys.readouterr().out == """\
+------------------------------------------------------------------------------------+
|                                     Tasks (4)                                      |
+------------------------+---------------------+-------------------------------------+
| Task                   | Environment ID      | Capability                          |
+------------------------+---------------------+-------------------------------------+
| classic_control/       |                     |                                     |
|   cart_pole            | CartPoleRL          | RL                                  |
| manipulation/          |                     |                                     |
|   hand_over            | HandOver-v1         | Expert Demo: Task Program           |
|   tableware/           |                     |                                     |
|     blocks_ranking_rgb | BlocksRankingRGB-v1 | Expert Demo: Handwritten Trajectory |
|     stack_cups         | StackCups-v1        | Environment Only                    |
+------------------------+---------------------+-------------------------------------+
"""


def test_list_task_help_explains_environment_only_label(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """List-task help defines the fallback capability label."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["list-task", "--help"])

    assert exc_info.value.code == 0
    assert "Environment Only" in capsys.readouterr().out


def test_config_environment_entries_use_task_paths_and_artifacts(
    tmp_path: Path,
) -> None:
    """Task-local configs provide hierarchy and explicit capabilities."""
    expert_task = tmp_path / "manipulation" / "pick_place"
    expert_task.mkdir(parents=True)
    (expert_task / "env.json").write_text(
        json.dumps(
            {
                "id": "PickPlace-v1",
                "task_program_dir": "task_program",
            }
        ),
        encoding="utf-8",
    )

    rl_task = tmp_path / "classic_control" / "point_mass"
    agents = rl_task / "agents"
    agents.mkdir(parents=True)
    (agents / "ppo.json").write_text(
        json.dumps(
            {
                "trainer": {
                    "learning_env": {"name": "PointMassRL"},
                }
            }
        ),
        encoding="utf-8",
    )

    entries = list_task._config_environment_entries([tmp_path])

    expert = entries["pickplace-v1"]
    assert expert.task_path == ("manipulation", "pick_place")
    assert expert.capabilities == {list_task._TASK_PROGRAM}
    learning = entries["pointmassrl"]
    assert learning.task_path == ("classic_control", "point_mass")
    assert learning.capabilities == {list_task._RL}


def test_handwritten_demo_detection_uses_environment_hooks() -> None:
    """Only task classes overriding a demo hook are handwritten demos."""
    from embodichain_tasks.manipulation.tableware.blocks_ranking_size import (
        BlocksRankingSizeEnv,
    )
    from embodichain_tasks.special.simple_task import SimpleTaskEnv

    assert list_task._implements_handwritten_demo(SimpleTaskEnv)
    assert not list_task._implements_handwritten_demo(BlocksRankingSizeEnv)


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
