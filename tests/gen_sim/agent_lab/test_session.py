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
import sys

import psutil
import pytest

from embodichain.gen_sim.agent_lab import session
from embodichain.gen_sim.agent_lab.session import _process, _prompt


@pytest.mark.parametrize(
    "model,effort,expected",
    [
        (None, None, ("custom-model", "medium")),
        ("another-model", None, ("another-model", "medium")),
        (None, "high", ("custom-model", "high")),
        ("another-model", "low", ("another-model", "low")),
    ],
)
def test_settings_inherit_independently(
    tmp_path: Path, model, effort, expected
) -> None:
    previous = tmp_path / "codex/turn_003"
    previous.mkdir(parents=True)
    (previous / "agent_settings.json").write_text(
        json.dumps({"model": "custom-model", "reasoning_effort": "medium"})
    )
    assert session._resolve_agent_settings(tmp_path, model, effort) == expected


def test_settings_default_only_when_no_previous_configuration(tmp_path: Path) -> None:
    assert session._resolve_agent_settings(tmp_path, None, None) == (
        "gpt-6-astra",
        "xhigh",
    )
    with pytest.raises(ValueError, match="non-empty"):
        session._resolve_agent_settings(tmp_path, "", "high")


def test_resume_passes_inherited_settings_and_does_not_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace/candidate.json").write_text(
        json.dumps({"script": "old.py", "config": "old.json"})
    )
    previous = tmp_path / "codex/turn_001"
    previous.mkdir(parents=True)
    (previous / "agent_settings.json").write_text(
        json.dumps({"model": "custom-model", "reasoning_effort": "medium"})
    )
    (tmp_path / "session.json").write_text(
        json.dumps({"thread_id": "thread", "turns": 1})
    )
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "task": {
                    "instruction": "move",
                    "acceptance": "stable",
                    "source_dir": "/assets",
                    "image": "",
                    "asset_status": "exported",
                },
                "repo": str(tmp_path),
                "python": sys.executable,
            }
        )
    )
    commands = []

    def fake_process(command, *, output, **kwargs):
        commands.append(command)
        (output / "stdout.log").write_text("")
        return {"returncode": 1, "timed_out": False}

    monkeypatch.setattr(session, "_process", fake_process)

    def unexpected_replay(*args, **kwargs):
        raise AssertionError("A backend failure must not replay a stale candidate")

    monkeypatch.setattr(session, "execute_script", unexpected_replay)
    result = session.solve(tmp_path, minutes=1)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--model") + 1] == "custom-model"
    assert 'model_reasoning_effort="medium"' in commands[0]
    assert result["agent_error"]["returncode"] == 1


def test_resume_cli_leaves_omitted_settings_unset_and_propagates_backend_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from embodichain.gen_sim.agent_lab import __main__ as cli

    captured = {}

    def fake_solve(root, **kwargs):
        captured.update(kwargs)
        return {"status": "no_candidate", "agent_error": {"returncode": 1}}

    monkeypatch.setattr(cli, "solve", fake_solve)
    monkeypatch.setattr(
        sys, "argv", ["agent-lab", "resume", "--run-dir", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert captured["model"] is None
    assert captured["reasoning_effort"] is None


def test_process_preserves_nonzero_exit_and_both_streams(tmp_path: Path) -> None:
    result = _process(
        [
            sys.executable,
            "-c",
            "import sys; print('motion'); print('failed', file=sys.stderr); sys.exit(3)",
        ],
        cwd=tmp_path,
        output=tmp_path,
        timeout=5,
        env=dict(os.environ),
    )
    assert result["returncode"] == 3
    assert "motion" in (tmp_path / "stdout.log").read_text()
    assert "failed" in (tmp_path / "stderr.log").read_text()


def test_process_timeout_is_not_success(tmp_path: Path) -> None:
    result = _process(
        [sys.executable, "-c", "import time; time.sleep(20)"],
        cwd=tmp_path,
        output=tmp_path,
        timeout=0.1,
        env=dict(os.environ),
    )
    assert result["timed_out"]
    assert result["returncode"] != 0
    assert json.loads((tmp_path / "process.json").read_text())["timed_out"]


def test_prompt_opens_methods_but_does_not_claim_physical_success(
    tmp_path: Path,
) -> None:
    manifest = {
        "task": {
            "instruction": "lift",
            "acceptance": "stable",
            "source_dir": "/assets",
            "image": "/image.png",
            "asset_status": "exported",
        },
        "repo": "/repo",
        "python": sys.executable,
    }
    prompt = _prompt(tmp_path, manifest, 1000)
    assert "There is no action DSL" in prompt
    assert "real contact-driven object motion" in prompt
    assert "not authoritative task success" in prompt


def test_prompt_limits_mode_c_to_explicit_asset_approval(tmp_path: Path) -> None:
    manifest = {
        "task": {
            "instruction": "complete the mirrored pattern",
            "acceptance": "stable symmetric placement",
            "source_dir": "/assets",
            "image": "/image.png",
            "asset_status": "exported",
        },
        "repo": "/repo",
        "python": sys.executable,
        "mode": "C",
        "approved_asset_changes": [
            {"entity": "block_001", "operation": "runtime-copy material color only"}
        ],
    }
    prompt = _prompt(tmp_path, manifest, 1000)
    assert "Experiment mode: C" in prompt
    assert json.dumps(manifest["approved_asset_changes"]) in prompt
    assert "original object mass/friction" in prompt
    assert "Any further source asset repair requires" in prompt


def test_timeout_stops_a_child_in_a_separate_process_session(tmp_path: Path) -> None:
    command = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], start_new_session=True); "
        "open('child.pid','w').write(str(p.pid)); time.sleep(30)"
    )
    result = _process(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        output=tmp_path,
        timeout=0.5,
        env=dict(os.environ),
    )
    pid = int((tmp_path / "child.pid").read_text())
    assert result["timed_out"]
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


def test_candidate_helpers_are_snapshotted_with_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "run"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    (root / "run.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "repo": str(tmp_path),
                "mode": "C",
                "approved_asset_changes": [{"entity": "block_001", "color": "green"}],
            }
        )
    )
    script = workspace / "solution.py"
    script.write_text("from helper import VALUE\n")
    (workspace / "helper.py").write_text("VALUE = 42\n")
    config = workspace / "lab.json"
    config.write_text("{}")

    def fake_process(command, *, cwd, output, **kwargs):
        assert (cwd / "helper.py").read_text() == "VALUE = 42\n"
        assert str(cwd) in kwargs["env"]["PYTHONPATH"]
        (output / "result.json").write_text(
            json.dumps({"execution_completed": True, "task_success": None})
        )
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(session, "_process", fake_process)
    report = session.execute_script(root, script, config)
    (workspace / "helper.py").write_text("VALUE = 0\n")
    assert (Path(report["attempt"]) / "code/helper.py").read_text() == "VALUE = 42\n"
    (root / "run.json").write_text("{}")
    assert json.loads((Path(report["attempt"]) / "experiment.json").read_text()) == {
        "mode": "C",
        "approved_asset_changes": [{"entity": "block_001", "color": "green"}],
    }


def test_host_queue_executes_once_and_returns_a_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = tmp_path / "requests"
    queue.mkdir()
    (queue / "001.json").write_text(
        json.dumps({"script": "/s.py", "config": "/c.json", "timeout": 10})
    )
    calls = []

    def fake_execute(*args, **kwargs):
        calls.append(args)
        return {"worker": {"execution_completed": False}, "process": {"returncode": 1}}

    monkeypatch.setattr(session, "execute_script", fake_execute)
    session._serve_requests(tmp_path, float("inf"))
    session._serve_requests(tmp_path, float("inf"))
    assert len(calls) == 1
    assert (
        json.loads((queue / "001.reply.json").read_text())["process"]["returncode"] == 1
    )


def test_resume_uses_existing_session_without_unsupported_color_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "workspace").mkdir()
    manifest = {
        "task": {
            "instruction": "lift",
            "acceptance": "stable",
            "source_dir": "/assets",
            "image": "/image.png",
            "asset_status": "exported",
        },
        "repo": str(tmp_path),
        "python": sys.executable,
    }
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    (tmp_path / "session.json").write_text(
        json.dumps({"thread_id": "existing", "turns": 1})
    )

    def fake_process(command, *, output, **kwargs):
        assert command[4:6] == ["resume", "existing"]
        assert "--color" not in command
        assert command[command.index("--model") + 1] == "gpt-6-astra"
        assert 'model_reasoning_effort="xhigh"' in command
        assert json.loads((output / "agent_settings.json").read_text()) == {
            "model": "gpt-6-astra",
            "reasoning_effort": "xhigh",
        }
        assert kwargs["service"] is not None
        (output / "stdout.log").write_text("")
        return {"timed_out": False, "returncode": 1}

    monkeypatch.setattr(session, "_process", fake_process)
    result = session.solve(tmp_path, minutes=1)
    assert result["status"] == "no_candidate"
    assert result["thread_id"] == "existing"


def test_resuming_host_does_not_restart_a_live_gpu_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = tmp_path / "requests"
    queue.mkdir()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (queue / "001.json").write_text("{}")
    (queue / "001.active.json").write_text(json.dumps({"attempt": str(attempt)}))
    (attempt / "process.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "created_at": psutil.Process().create_time(),
            }
        )
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("A live existing worker must not be restarted")

    monkeypatch.setattr(session, "execute_script", unexpected)
    session._serve_requests(tmp_path, float("inf"))
    assert not (queue / "001.reply.json").exists()


def test_resuming_host_recovers_finished_worker_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = tmp_path / "requests"
    queue.mkdir()
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (queue / "001.json").write_text("{}")
    (queue / "001.active.json").write_text(json.dumps({"attempt": str(attempt)}))
    (attempt / "process.json").write_text(
        json.dumps({"pid": 99999999, "created_at": 0})
    )
    (attempt / "result.json").write_text(
        json.dumps({"execution_completed": True, "task_success": None})
    )
    session._serve_requests(tmp_path, float("inf"))
    assert json.loads((queue / "001.reply.json").read_text())["worker"][
        "execution_completed"
    ]


def test_interrupted_codex_session_is_recovered_from_durable_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "workspace").mkdir()
    manifest = {
        "task": {
            "instruction": "lift",
            "acceptance": "stable",
            "source_dir": "/assets",
            "image": "/image.png",
            "asset_status": "exported",
        },
        "repo": str(tmp_path),
        "python": sys.executable,
    }
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    previous = tmp_path / "codex/turn_001"
    previous.mkdir(parents=True)
    (previous / "stdout.log").write_text(
        '{"type":"thread.started","thread_id":"interrupted"}\n'
    )

    def fake_process(command, *, output, **kwargs):
        assert command[4:6] == ["resume", "interrupted"]
        assert output.name == "turn_002"
        (output / "stdout.log").write_text("")
        return {"timed_out": False, "returncode": 1}

    monkeypatch.setattr(session, "_process", fake_process)
    assert session.solve(tmp_path, minutes=1)["thread_id"] == "interrupted"


def test_new_sessions_explicitly_use_astra_xhigh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "workspace").mkdir()
    manifest = {
        "task": {
            "instruction": "mirror",
            "acceptance": "symmetry",
            "level": "L4",
            "source_dir": "/assets",
            "image": "/image.png",
            "asset_status": "exported",
        },
        "repo": str(tmp_path),
        "python": sys.executable,
    }
    (tmp_path / "run.json").write_text(json.dumps(manifest))

    def fake_process(command, *, output, prompt, **kwargs):
        assert "resume" not in command
        assert command[command.index("--model") + 1] == "gpt-6-astra"
        assert 'model_reasoning_effort="xhigh"' in command
        assert "Difficulty: L4" in prompt
        assert "scene_review.json" in prompt
        assert "For L3" not in prompt
        assert "For L4" not in prompt
        (output / "stdout.log").write_text("")
        return {"timed_out": False, "returncode": 1}

    monkeypatch.setattr(session, "_process", fake_process)
    session.solve(tmp_path, minutes=1)
