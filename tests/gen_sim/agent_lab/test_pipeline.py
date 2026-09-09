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

from embodichain.gen_sim.agent_lab import __main__ as cli
from embodichain.gen_sim.agent_lab import _launch, session
from embodichain.gen_sim.agent_lab.catalog import Task, write_json


@pytest.mark.parametrize(
    "flags,window,minutes,objective",
    [
        ([], False, 45, "solve"),
        (
            ["--batch", "--minutes", "10", "--objective", "cold_start"],
            False,
            10,
            "cold_start",
        ),
        (["--window"], True, 45, "solve"),
    ],
)
def test_pipeline_creates_workspace_and_launches_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: list[str],
    window: bool,
    minutes: int,
    objective: str,
) -> None:
    task = Task("task1187", "task", "acceptance", "/assets")
    root = tmp_path / "created"
    calls = []
    monkeypatch.setattr(cli, "read_catalog", lambda path: {task.task_id: task})

    def prepare(selected, **kwargs):
        assert selected == task
        assert kwargs["objective"] == objective
        calls.append("prepare")
        return root

    def launch(selected, **kwargs):
        assert selected == root
        assert kwargs == {
            "minutes": minutes,
            "batch": not window,
            "model": "gpt-6-astra",
            "effort": "xhigh",
        }
        calls.append("window" if window else "batch")
        return 0 if window else {"status": "exited", "process": {"returncode": 0}}

    monkeypatch.setattr(cli, "create_run", prepare)
    monkeypatch.setattr(_launch, "_launch_window" if window else "_launch", launch)
    monkeypatch.setattr(
        cli,
        "solve",
        lambda *a, **k: pytest.fail("Pipeline must not use the legacy solver"),
    )
    monkeypatch.setattr(
        sys, "argv", ["agent_lab", "pipeline", "--task", "task1187", *flags]
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 0
    assert calls == ["prepare", "window" if window else "batch"]


def test_pipeline_task_file_reaches_standard_delivery_without_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_file = tmp_path / "task.json"
    write_json(
        task_file,
        {
            "task_id": "custom",
            "instruction": "inspect",
            "acceptance": "report",
            "source_dir": "/assets",
        },
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        session.subprocess,
        "check_output",
        lambda *a, **k: "test" if k.get("text") else b"diff",
    )

    def codex(command, **kwargs):
        assert "resume" not in command
        assert kwargs["prompt"] == _launch._BOOTSTRAP_PROMPT
        assert not kwargs["interactive"]
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(_launch, "_process", codex)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent_lab",
            "pipeline",
            "--task-file",
            str(task_file),
            "--output-root",
            str(tmp_path / "runs"),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 0
    roots = list((tmp_path / "runs/custom").iterdir())
    assert len(roots) == 1
    root = roots[0]
    assert (root / "workspace/START.md").is_file()
    assert (root / "usage.json").is_file()
    assert (root / "final/report.md").is_file()
    result = json.loads((root / "final/result.json").read_text())
    assert result["assessment"]["scope"] == "solve"
    assert result["assessment"]["task_success"] is None


def test_pipeline_rejects_conflicting_modes_before_preparing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "create_run",
        lambda *a, **k: pytest.fail("Must not prepare on invalid arguments"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["agent_lab", "pipeline", "--task", "task1187", "--batch", "--window"],
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
