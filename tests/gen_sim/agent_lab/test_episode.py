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
