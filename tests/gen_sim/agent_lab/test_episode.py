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
import time
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.agent_lab.catalog import write_json
from embodichain.gen_sim.agent_lab.episode_worker import serve_episode
from embodichain.gen_sim.agent_lab.session import _serve_requests
from embodichain.gen_sim.agent_lab.episode import EpisodeHost


def _run_chunks(tmp_path: Path, scripts: list[str]) -> list[dict]:
    commands = tmp_path / "commands"
    commands.mkdir()
    physics = {"steps": 0}
    lab = SimpleNamespace(
        output=tmp_path,
        snapshot=lambda: dict(physics),
        capture=lambda name: None,
        step=lambda n: physics.update(steps=physics["steps"] + n),
    )
    for index, source in enumerate(scripts):
        script = tmp_path / f"chunk{index}.py"
        script.write_text(source)
        write_json(
            commands / f"{index}.json", {"operation": "exec", "script": str(script)}
        )
    write_json(commands / f"{len(scripts)}.json", {"operation": "close"})
    serve_episode(lab, commands)
    return [
        json.loads((commands / f"{i}.reply.json").read_text())
        for i in range(len(scripts))
    ]


@pytest.mark.parametrize(
    "error", ["RuntimeError('grasp did not hold')", "SystemExit(2)"]
)
def test_failed_chunk_returns_its_measurements_without_changing_failure(
    tmp_path, error
):
    reply = _run_chunks(
        tmp_path,
        [
            f"lab.step(2)\nresult = {{'height_error_m': 0.04, 'task_success': True}}\nraise {error}\n"
        ],
    )[0]
    assert reply["ok"] is False
    assert reply["result"] == {"height_error_m": 0.04, "task_success": True}
    assert reply["state"]["steps"] == 2
    assert reply["error"].startswith(error.split("(")[0] + ":")
    assert "task_success" not in reply


@pytest.mark.parametrize("value", ["object()", "float('nan')"])
@pytest.mark.parametrize("fails", [False, True])
def test_feedback_encoding_failure_does_not_mask_execution_error_or_kill_session(
    tmp_path, value, fails
):
    source = f"state['kept'] = 7\nresult = {value}\n"
    if fails:
        source += "raise RuntimeError('original failure')\n"
    first, second = _run_chunks(tmp_path, [source, "result = state['kept']\n"])
    assert first["ok"] is False
    assert first["result"] is None
    assert first["result_serialization_error"]
    if fails:
        assert first["error"] == "RuntimeError: original failure"
        assert "RuntimeError: original failure" in first["traceback"]
    else:
        assert first["error"] == first["result_serialization_error"]
    assert second["ok"] is True and second["result"] == 7


@pytest.mark.parametrize(
    "source", ["raise ValueError('before measurement')", "def incomplete("]
)
def test_failed_chunk_without_measurement_cannot_return_previous_result(
    tmp_path, source
):
    first, second = _run_chunks(tmp_path, ["result = {'previous': True}", source])
    assert first["ok"] is True
    assert second["ok"] is False and second["result"] is None


def test_script_can_use_failure_feedback_to_change_the_next_action(tmp_path):
    first, second = _run_chunks(
        tmp_path,
        [
            "result = {'remaining_steps': 3}\nraise RuntimeError('probe incomplete')\n",
            "import json\nfrom pathlib import Path\n"
            "reply = json.loads((Path(__file__).parent / 'commands/0.reply.json').read_text())\n"
            "remaining = reply['result']['remaining_steps']\n"
            "if remaining > 0:\n    lab.step(remaining)\n"
            "result = {'correction': remaining}\n",
        ],
    )
    assert first["state"]["steps"] == 0
    assert second["ok"] is True
    assert second["result"]["correction"] == second["state"]["steps"] == 3


def test_chunks_keep_namespace_and_physics_after_python_failure(tmp_path: Path) -> None:
    commands = tmp_path / "commands"
    commands.mkdir()
    physics = {"steps": 0}
    captures = []
    lab = SimpleNamespace(
        output=tmp_path,
        snapshot=lambda: dict(physics),
        capture=lambda name: captures.append(name),
        step=lambda n: physics.update(steps=physics["steps"] + n),
    )
    scripts = [
        "state['value'] = 7\nlab.step(3)\n",
        "raise ValueError('bad IK hypothesis')\n",
        "lab.step(state['value'])\nresult = {'value': state['value']}\n",
    ]
    for i, source in enumerate(scripts):
        script = tmp_path / f"chunk{i}.py"
        script.write_text(source)
        write_json(commands / f"{i}.json", {"operation": "exec", "script": str(script)})
    write_json(commands / "3.json", {"operation": "close"})
    result = serve_episode(lab, commands)
    replies = [json.loads((commands / f"{i}.reply.json").read_text()) for i in range(3)]
    assert replies[0]["state"]["steps"] == 3
    assert replies[1]["ok"] is False
    assert replies[1]["state"]["steps"] == 3
    assert replies[2]["result"]["value"] == 7
    assert replies[2]["state"]["steps"] == 10
    assert len(result["chunks"]) == len(captures) == 3


def test_interrupted_episode_job_is_not_replayed(tmp_path: Path) -> None:
    queue = tmp_path / "requests"
    queue.mkdir()
    write_json(queue / "one.json", {"episode_operation": "exec"})
    write_json(
        queue / "one.active.json", {"episode_operation": "exec", "attempt": "old"}
    )
    _serve_requests(tmp_path, float("inf"))
    result = json.loads((queue / "one.reply.json").read_text())
    assert (
        "not automatically replayed" in result["host_error"]
        or "no state or command" in result["host_error"]
    )


def test_standard_main_guard_and_zero_exit_only_finish_the_chunk(tmp_path):
    commands = tmp_path / "commands"
    commands.mkdir()
    lab = SimpleNamespace(
        output=tmp_path, snapshot=lambda: {}, capture=lambda name: None
    )
    script = tmp_path / "main.py"
    script.write_text(
        "import sys\nif __name__ == '__main__':\n    state['value'] = 17\n    sys.exit(0)\n"
    )
    followup = tmp_path / "followup.py"
    followup.write_text("result = state['value']\n")
    write_json(commands / "1.json", {"operation": "exec", "script": str(script)})
    write_json(commands / "2.json", {"operation": "exec", "script": str(followup)})
    write_json(commands / "3.json", {"operation": "close"})
    serve_episode(lab, commands)
    assert json.loads((commands / "1.reply.json").read_text())["ok"] is True
    assert json.loads((commands / "2.reply.json").read_text())["result"] == 17


def test_episode_requests_use_host_service_not_action_dispatch(tmp_path: Path) -> None:
    queue = tmp_path / "requests"
    queue.mkdir()
    write_json(queue / "one.json", {"episode_operation": "observe"})
    jobs = []
    host = SimpleNamespace(
        attempt=None, service=lambda job: jobs.append(job) or {"ok": True}
    )
    _serve_requests(tmp_path, float("inf"), host)
    assert jobs == [{"episode_operation": "observe"}]
    assert json.loads((queue / "one.reply.json").read_text())["ok"]


def test_dead_worker_closes_handles_and_retains_diagnostic_result(tmp_path):
    host = EpisodeHost(tmp_path, time.time() + 100)
    host.attempt = tmp_path / "attempts/episode_dead"
    host.attempt.mkdir(parents=True)
    host.started = time.monotonic()
    host.identity = {"started_at": time.time()}
    host.streams = [io.StringIO(), io.StringIO()]
    host.process = SimpleNamespace(poll=lambda: 2, returncode=2)
    with pytest.raises(RuntimeError, match=r"exited \(2\)"):
        host._wait(host.attempt / "ready.json", 1)
    assert host.process is None
    assert all(stream.closed for stream in host.streams)
    report = json.loads((tmp_path / "latest_attempt.json").read_text())
    assert report["worker"]["execution_completed"] is False


def test_no_implicit_restart_of_live_episode(tmp_path):
    host = EpisodeHost(tmp_path, time.time() + 100)
    host.process = SimpleNamespace(poll=lambda: None)
    with pytest.raises(RuntimeError, match="no implicit reset"):
        host.start(tmp_path / "unused.json", 1)


def test_chunk_snapshot_does_not_recursively_copy_its_destination(tmp_path):
    host = EpisodeHost(tmp_path, time.time() + 100)
    host.process = SimpleNamespace(poll=lambda: None)
    host.attempt = tmp_path / "attempts/episode"
    host.attempt.mkdir(parents=True)
    write_json(host.attempt / "lab.json", {})
    script = tmp_path / "probe.py"
    script.write_text("result = 1\n")
    host._wait = lambda path, timeout: {"ok": True}
    result = host.command("exec", script=script, timeout=1)
    code = host.attempt / "chunks/0001/code"
    assert result["ok"]
    assert (code / "probe.py").read_text() == "result = 1\n"
    assert not (code / "attempts/episode").exists()


def test_timed_out_chunk_stops_worker_instead_of_automatic_replay(
    tmp_path, monkeypatch
):
    host = EpisodeHost(tmp_path, time.time() - 1)
    host.attempt = tmp_path / "attempts/episode_timeout"
    host.attempt.mkdir(parents=True)
    host.started = time.monotonic()
    host.identity = {"started_at": time.time()}
    host.process = SimpleNamespace(poll=lambda: None, returncode=None)
    stopped = []
    monkeypatch.setattr(
        "embodichain.gen_sim.agent_lab.episode._stop_tree",
        lambda process: stopped.append(process),
    )
    with pytest.raises(TimeoutError):
        host._wait(host.attempt / "absent.json", 1)
    assert len(stopped) == 1 and host.process is None
    assert json.loads((host.attempt / "process.json").read_text())["timed_out"]
