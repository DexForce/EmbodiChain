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
from pathlib import Path
import sys

import pytest

from embodichain.gen_sim.agent_lab import __main__ as cli, _launch, session
from embodichain.gen_sim.agent_lab.catalog import Task, write_json
from embodichain.gen_sim.agent_lab.delivery import finalize_run


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        session.subprocess,
        "check_output",
        lambda *a, **k: "test" if k.get("text") else b"diff",
    )
    task = Task(
        "task1187",
        "OLD_INSTRUCTION",
        "OLD_ACCEPTANCE",
        "/assets",
        "/image.png",
        "/assets/scene.json",
        "exported",
        "L3",
    )
    return task, tmp_path


def test_override_preserves_assets_but_replaces_goal_acceptance_and_level(
    prepared,
) -> None:
    task, tmp = prepared
    root = session.create_run(
        task, repo=tmp, output_root=tmp / "runs", instruction="NEW_GOAL"
    )
    manifest = json.loads((root / "run.json").read_text())
    effective = json.loads((root / "workspace/task.json").read_text())
    assert effective["instruction"] == effective["acceptance"] == "NEW_GOAL"
    assert effective["level"] == ""
    for key in ("task_id", "source_dir", "scene_config", "image", "asset_status"):
        assert effective[key] == task.to_dict()[key]
    assert manifest["task_variant"] == "custom_instruction"
    assert manifest["acceptance_source"] == "instruction"
    assert manifest["source_task"] == task.to_dict()
    assert task.instruction == "OLD_INSTRUCTION"
    prompt = session._prompt(root, manifest, 1000)
    assert "OLD_ACCEPTANCE" not in prompt and "OLD_INSTRUCTION" not in prompt
    assert "NEW_GOAL" in prompt
    assert (
        json.loads((root / "workspace/lab.json").read_text())["scene"]
        == task.scene_config
    )


def test_no_override_keeps_default_task_unchanged(prepared) -> None:
    task, tmp = prepared
    root = session.create_run(task, repo=tmp, output_root=tmp / "runs")
    assert json.loads((root / "workspace/task.json").read_text()) == task.to_dict()
    assert json.loads((root / "run.json").read_text())["task_variant"] == "default"


def test_custom_tasks_create_separate_runs_and_reports(prepared) -> None:
    task, tmp = prepared
    first = session.create_run(
        task, repo=tmp, output_root=tmp / "runs", instruction="FIRST"
    )
    second = session.create_run(
        task, repo=tmp, output_root=tmp / "runs", instruction="SECOND"
    )
    assert first != second
    write_json(
        second / "workspace/task_interpretation.json",
        {"checks": ["Observe the requested result before claiming completion"]},
    )
    result = finalize_run(second)
    assert result["run"]["task_variant"] == "custom_instruction"
    assert result["run"]["instruction"] == result["run"]["acceptance"] == "SECOND"
    assert result["run"]["source_task"]["level"] == "L3"
    assert result["assessment"]["task_success"] is None
    assert result["assessment"]["task_interpretation"]["checks"]
    report = (second / "final/report.md").read_text()
    assert "自定义任务" in report and "OLD_ACCEPTANCE" not in report


@pytest.mark.parametrize("instruction", ["", "  \n "])
def test_empty_override_does_not_silently_run_default(
    prepared, instruction: str
) -> None:
    task, tmp = prepared
    with pytest.raises(ValueError, match="instruction"):
        session.create_run(
            task, repo=tmp, output_root=tmp / "runs", instruction=instruction
        )
    assert not (tmp / "runs").exists()


@pytest.mark.parametrize("entrypoint", ["prepare", "solve", "pipeline"])
@pytest.mark.parametrize("selection", ["catalog", "file"])
def test_cli_supports_one_override_flag(
    prepared,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    selection: str,
) -> None:
    task, tmp = prepared
    monkeypatch.setattr(cli, "read_catalog", lambda _: {task.task_id: task})
    monkeypatch.setattr(
        _launch,
        "_launch",
        lambda *a, **k: {"status": "exited", "process": {"returncode": 0}},
    )
    monkeypatch.setattr(cli, "solve", lambda *a, **k: {"status": "no_candidate"})
    task_file = tmp / "input.json"
    write_json(task_file, task.to_dict())
    selector = (
        ["--task", task.task_id]
        if selection == "catalog"
        else ["--task-file", str(task_file)]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent_lab",
            entrypoint,
            *selector,
            "--instruction",
            "CLI_GOAL",
            "--output-root",
            str(tmp / "runs"),
        ],
    )
    if entrypoint == "pipeline":
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert stopped.value.code == 0
    else:
        cli.main()
    root = next((tmp / "runs" / task.task_id).iterdir())
    assert (
        json.loads((root / "workspace/task.json").read_text())["instruction"]
        == "CLI_GOAL"
    )
    assert json.loads(task_file.read_text()) == task.to_dict()
