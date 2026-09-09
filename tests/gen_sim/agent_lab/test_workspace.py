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
import os
from pathlib import Path
import signal
import threading
import subprocess
import sys
import time

import psutil
import pytest

from embodichain.gen_sim.agent_lab import _launch, session
from embodichain.gen_sim.agent_lab._workspace import _write_workspace
from embodichain.gen_sim.agent_lab.catalog import Task


def test_bootstrap_does_not_depend_on_task_identity(tmp_path: Path) -> None:
    guides = []
    for task_id in ("custom_experiment", "another_scene"):
        root = tmp_path / task_id
        (root / "workspace").mkdir(parents=True)
        manifest = {
            "task": {"task_id": task_id, "instruction": task_id},
            "repo": "/repo",
            "python": sys.executable,
            "objective": "cold_start",
        }
        _write_workspace(root, manifest)
        workspace = root / "workspace"
        guides.append((workspace / "START.md").read_bytes())
        assert not (workspace / "solution.py").exists()
        assert not (workspace / "notes.md").exists()
        assert json.loads((workspace / "task.json").read_text()) == manifest["task"]
        assert (
            json.loads((workspace / "physics.json").read_text())[
                "approved_asset_changes"
            ]
            == []
        )
    assert guides[0] == guides[1]


def test_prepare_accepts_non_task100_identity_and_explicit_robot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session.subprocess,
        "check_output",
        lambda *a, **kw: "revision" if kw.get("text") else b"diff",
    )
    task = Task(
        "custom", "inspect", "observed", "/assets", "/image.png", None, "unprepared"
    )
    root = session.create_run(
        task,
        repo=tmp_path,
        output_root=tmp_path / "outputs",
        objective="cold_start",
        robot_component=Path("/robot.yaml"),
    )
    config = json.loads((root / "workspace/lab.json").read_text())
    assert config["robot_component"] == "/robot.yaml"
    assert (
        json.loads((root / "workspace/environment.json").read_text())["objective"]
        == "cold_start"
    )


@pytest.mark.parametrize("batch", [True, False])
def test_launch_command_is_fresh_with_memory_disabled(
    tmp_path: Path, batch: bool
) -> None:
    command = _launch._codex_command(
        tmp_path, tmp_path, batch=batch, model="gpt-6-astra", effort="xhigh"
    )
    assert "resume" not in command and "fork" not in command
    assert "memories.use_memories=false" in command
    assert "memories.generate_memories=false" in command
    assert 'model_reasoning_effort="xhigh"' in command
    assert command[command.index("--model") + 1] == "gpt-6-astra"
    assert ("exec" in command) is batch


def test_cold_launch_supplies_only_workspace_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "workspace").mkdir()
    (tmp_path / "run.json").write_text(
        json.dumps({"repo": str(tmp_path), "task": {"instruction": "SECRET TASK"}})
    )

    def fake_process(command, *, prompt, service, **kwargs):
        assert prompt == _launch._BOOTSTRAP_PROMPT
        assert "SECRET TASK" not in prompt
        assert service is not None
        assert not kwargs["interactive"]
        return {"timed_out": False, "returncode": 0}

    monkeypatch.setattr(_launch, "_process", fake_process)
    result = _launch._launch(
        tmp_path, minutes=1, batch=True, model="gpt-6-astra", effort="xhigh"
    )
    assert result["status"] == "exited"
    assert result["new_conversation"]


def test_missing_or_stale_host_fails_without_queueing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No live experiment host"):
        _launch._check_host(tmp_path)
    (tmp_path / "launch.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": os.getpid(),
                "created_at": 0,
                "deadline": time.time() + 60,
            }
        )
    )
    with pytest.raises(RuntimeError, match="No live experiment host"):
        _launch._check_host(tmp_path)


def test_live_host_cannot_be_replaced(tmp_path: Path) -> None:
    (tmp_path / "launch.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": os.getpid(),
                "created_at": psutil.Process().create_time(),
                "deadline": time.time() + 60,
            }
        )
    )
    with pytest.raises(RuntimeError, match="already has a live host"):
        _launch._check_host(tmp_path, required=False)


@pytest.mark.parametrize("exit_code", [0, 3])
def test_independent_script_does_not_require_lab_function(
    tmp_path: Path, exit_code: int
) -> None:
    repo = Path(__file__).resolve().parents[3]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "run.json").write_text(
        json.dumps({"repo": str(repo), "python": sys.executable})
    )
    script = workspace / "plain.py"
    script.write_text(
        "import os\nfrom pathlib import Path\nPath(os.environ['GENSIM_LAB_OUTPUT'], 'plain.txt').write_text('ran')\n"
        f"raise SystemExit({exit_code})\n"
    )
    result = session.execute_script(tmp_path, script, None, timeout=30)
    assert result["worker"]["execution_completed"] is (exit_code == 0)
    assert (Path(result["attempt"]) / "plain.txt").read_text() == "ran"
    assert result["worker"]["task_success"] is None


def test_workspace_help_needs_no_live_host(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    (tmp_path / "workspace").mkdir()
    _write_workspace(
        tmp_path, {"repo": str(repo), "python": sys.executable, "task": {}}
    )
    result = subprocess.run(
        [sys.executable, str(tmp_path / "workspace/lab.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "run,inspect,status,stop" in result.stdout


def test_workspace_client_discovers_host_without_inherited_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENSIM_LAB_BROKER", raising=False)
    monkeypatch.delenv("GENSIM_LAB_DEADLINE", raising=False)
    host = {
        "status": "running",
        "pid": os.getpid(),
        "created_at": psutil.Process().create_time(),
        "deadline": time.time() + 60,
        "output": "launch-test",
    }
    (tmp_path / "launch.json").write_text(json.dumps(host))
    _launch._write_heartbeat(tmp_path, host)
    monkeypatch.setattr(
        _launch.psutil,
        "Process",
        lambda *a: (_ for _ in ()).throw(
            AssertionError("Sandbox client must not inspect host PIDs")
        ),
    )

    def fake_execute(root, script, config, **kwargs):
        assert root == tmp_path
        assert config is None
        assert os.environ["GENSIM_LAB_BROKER"] == str(tmp_path)
        assert float(os.environ["GENSIM_LAB_DEADLINE"]) == host["deadline"]
        return {"worker": {"execution_completed": True}}

    monkeypatch.setattr(_launch, "execute_script", fake_execute)
    with pytest.raises(SystemExit) as caught:
        _launch._workspace_main(tmp_path, ["run", "--script", "plain.py"])
    assert caught.value.code == 0


def test_stop_uses_launch_scoped_request_not_a_host_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "launch.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": os.getpid(),
                "created_at": psutil.Process().create_time(),
                "deadline": time.time() + 60,
                "output": "launch-test",
            }
        )
    )
    _launch._write_heartbeat(tmp_path, {"output": "launch-test"})
    signals = []
    monkeypatch.setattr(_launch.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    _launch._workspace_main(tmp_path, ["stop"])
    assert signals == []
    assert json.loads((tmp_path / "stop.json").read_text()) == {"launch": "launch-test"}


def test_stale_heartbeat_is_not_a_live_host(tmp_path: Path) -> None:
    (tmp_path / "launch.json").write_text(
        json.dumps(
            {"status": "running", "output": "current", "deadline": time.time() + 60}
        )
    )
    (tmp_path / "heartbeat.json").write_text(
        json.dumps({"launch": "current", "time": time.time() - 30})
    )
    with pytest.raises(RuntimeError, match="No live experiment host"):
        _launch._check_host(tmp_path)


def test_heartbeat_services_stop_while_execution_loop_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped = threading.Event()
    signals = []
    (tmp_path / "stop.json").write_text(json.dumps({"launch": "current"}))
    monkeypatch.setattr(_launch.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    thread = threading.Thread(
        target=_launch._heartbeat, args=(tmp_path, {"output": "current"}, stopped)
    )
    thread.start()
    thread.join(timeout=3)
    stopped.set()
    thread.join()
    assert signals == [(os.getpid(), signal.SIGINT)]


def test_agent_requested_stop_is_not_task_success_or_launcher_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "workspace").mkdir()
    (tmp_path / "run.json").write_text(json.dumps({"repo": str(tmp_path)}))

    def fake_process(command, *, output, **kwargs):
        (tmp_path / "stop.json").write_text(json.dumps({"launch": str(output)}))
        (output / "process.json").write_text(
            json.dumps({"interrupted": True, "returncode": -15})
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(_launch, "_process", fake_process)
    result = _launch._launch(
        tmp_path, minutes=1, batch=True, model="gpt-6-astra", effort="xhigh"
    )
    assert result["status"] == "stopped"
    assert result["process"]["returncode"] == -15
    assert "task_success" not in result
