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

from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from embodichain.gen_sim.task_engine import cli
from embodichain.gen_sim.task_engine.orchestration.artifacts import ArtifactTransaction
from embodichain.gen_sim.task_engine.orchestration.coordinator import (
    TaskEngineCoordinator,
)
from embodichain.gen_sim.task_engine.robot_profiles import _require_robot_assets
from embodichain.gen_sim.task_engine.workflow import _error_record


def test_workflow_preserves_original_exception():
    assert _error_record(ModuleNotFoundError("missing robot package")) == {
        "type": "ModuleNotFoundError",
        "failure_type": "terminal",
        "message": "missing robot package",
    }


@pytest.mark.parametrize("profile", ["dual_franka", "dual_ur10", "dual_tianji"])
@pytest.mark.parametrize("missing", ["embodichain_robots", "embodichain_tasks"])
def test_robot_preflight_rejects_missing_packages(monkeypatch, profile, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ValueError, match="Python environment"):
        _require_robot_assets(profile)


def test_artifact_transaction_rolls_back_and_preserves_existing_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "kept.txt").write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fail"):
        with ArtifactTransaction(output, overwrite=True) as transaction:
            assert transaction.staging_dir is not None
            (transaction.staging_dir / "partial.txt").write_text(
                "partial", encoding="utf-8"
            )
            raise RuntimeError("fail before commit")

    assert (output / "kept.txt").read_text(encoding="utf-8") == "old"
    assert not (output / "partial.txt").exists()


def test_prepare_rejects_output_overlapping_read_only_source(tmp_path: Path) -> None:
    source = tmp_path / "gym_project"
    source.mkdir()
    coordinator = TaskEngineCoordinator(
        task_agent=object(),
        scene_adapter=SimpleNamespace(robot_profile="franka"),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        coordinator.prepare(
            "task",
            "Pick up the object.",
            source,
            source / "task_run",
            overwrite=True,
        )


@pytest.mark.parametrize(
    "lower_layer_option",
    [
        ["--planner-mode", "ik_interp"],
        ["--ik-solver", "pytorch"],
        ["--show-grasp-poses"],
    ],
)
def test_unified_cli_rejects_lower_layer_execution_options(
    tmp_path: Path,
    lower_layer_option: list[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "prepare",
                "--mode",
                "scene",
                "--task-id",
                "task",
                "--instruction",
                "place the cup",
                "--scene",
                str(tmp_path / "project"),
                "--output-root",
                str(tmp_path / "history"),
                *lower_layer_option,
            ]
        )


@pytest.mark.parametrize(
    "prefix", ["", "attempts/scene_0001/action_attempts/action_0001"]
)
def test_cli_reports_only_committed_local_videos(tmp_path, prefix):
    root = tmp_path / prefix / "videos"
    (root / "env_000").mkdir(parents=True)
    expected = [root / "legacy.mp4", root / "env_000/episode.mp4"]
    for path in [*expected, root / "env_000/failed.partial.mp4"]:
        path.touch()
    foreign = tmp_path.parent / f"{tmp_path.name}-foreign.mp4"
    foreign.touch()
    (root / "env_000/foreign.mp4").symlink_to(foreign)
    assert cli._report_saved_videos(tmp_path) == sorted(str(p) for p in expected)
