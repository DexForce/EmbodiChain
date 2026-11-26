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

"""Bounded, connection-independent program revision pilot for the official DexUni UR5."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from uuid import uuid4

import yaml
from embodichain.utils.config_paths import resolve_config_path

__all__ = ["ExperimentStore"]

TASK = (
    Path(__file__).resolve().parents[1]
    / "extensions/tasks/configs/tasks/manipulation/workspace_pose_pick_place/task.ur5.dexuni.yaml"
)
GOAL = {
    "goal": "lift cube and release it at the scene goal",
    "reference": "goal",
    "expected_xy_m": [0.0, 0.0],
    "min_lift_m": 0.04,
    "xy_tolerance_m": 0.02,
    "seed": 7,
    "max_attempts": 2,
    "wall_seconds": 1800,
}
_ID = re.compile(r"[a-f0-9]{32}")
_PROGRESS_INTERVAL_S = 0.5


def _save(path: Path, data: Any) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(data, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _seal(root: Path, offset: float, parent: str | None = None) -> str:
    """Resolve only the trusted deployment and seal its referenced YAML declarations."""
    files = {}

    def copy(source: Path) -> str:
        source = source.resolve()

        def visit(value):
            if isinstance(value, dict):
                return {k: visit(v) for k, v in value.items()}
            if isinstance(value, list):
                return [visit(v) for v in value]
            if isinstance(value, str) and value.endswith((".yaml", ".yml")):
                return copy(resolve_config_path(value, base_dir=source.parent))
            return value

        doc = visit(yaml.safe_load(source.read_text()))
        if source.name == "program.dexuni.yaml":
            doc["targets"]["drop_pose"]["position"][0] = offset
        if source == TASK:
            doc["seed"] = GOAL["seed"]
            doc["num_envs"] = 1
        name = _hash(doc) + ".yaml"
        files[name] = yaml.safe_dump(doc, sort_keys=False)
        return name

    if parent is None:
        entry = copy(TASK)
    else:
        prior = root / "bundles" / parent
        declaration = yaml.safe_load(_validate(prior, parent).read_text())
        original = _load(prior / "manifest.json")
        files = {name: (prior / name).read_text() for name in original["files"]}
        program = yaml.safe_load(files.pop(declaration["task_program"]["program"]))
        program["targets"]["drop_pose"]["position"][0] = offset
        program_name = _hash(program) + ".yaml"
        files[program_name] = yaml.safe_dump(program, sort_keys=False)
        files.pop(original["entry"])
        declaration["task_program"]["program"] = program_name
        entry = _hash(declaration) + ".yaml"
        files[entry] = yaml.safe_dump(declaration, sort_keys=False)
    identity = _hash({"entry": entry, "files": files, "goal": GOAL})
    dest = root / "bundles" / identity
    if dest.exists():
        _validate(dest, identity)
    else:
        tmp = Path(tempfile.mkdtemp(dir=dest.parent))
        try:
            for name, content in files.items():
                (tmp / name).write_text(content)
            _save(
                tmp / "manifest.json",
                {"entry": entry, "bundle": identity, "files": sorted(files)},
            )
            _validate(tmp, identity)
            tmp.rename(dest)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    return identity


def _validate(directory: Path, expected_revision: str) -> Path:
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    m = _load(directory / "manifest.json")
    if m["bundle"] != expected_revision:
        raise ValueError("Bundle does not match the requested revision")
    if m["entry"] not in m["files"] or any(
        not re.fullmatch(r"[0-9a-f]{64}\.yaml", n) for n in m["files"]
    ):
        raise ValueError("Invalid bundle inventory")
    files = {n: (directory / n).read_text() for n in m["files"]}
    if _hash(dict(entry=m["entry"], files=files, goal=GOAL)) != m["bundle"]:
        raise ValueError("Changed sealed declarations")
    entry = directory / m["entry"]
    cfg = config_to_cfg(yaml.safe_load(entry.read_text()), source_path=entry)
    if (
        cfg.num_envs != 1
        or cfg.sim_cfg.physics_cfg.solver_cfg["solver_type"] != "dexuni"
    ):
        raise ValueError("Pilot requires one DexUni arena")
    return entry


class ExperimentStore:
    """Own one local GenSim run directory per frozen experimental goal."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self, run_id: str):
        if not _ID.fullmatch(run_id):
            raise ValueError("Invalid experiment ID")
        path = self.root / run_id
        with (path / "lock").open("a+b") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield path

    @staticmethod
    def run_id_for(agent_session: str, command_id: str) -> str:
        """Stable run identity for creation retries."""
        return _hash([agent_session, command_id])[:32]

    def create(self, agent_session: str, command_id: str) -> dict:
        """Create once per external session and command ID."""
        if not agent_session or not command_id:
            raise ValueError("Missing agent session or command ID")
        rid = self.run_id_for(agent_session, command_id)
        path = self.root / rid
        path.mkdir(mode=0o700, exist_ok=True)
        with self._locked(rid):
            if not (path / "run.json").exists():
                (path / "bundles").mkdir(exist_ok=True)
                (path / "jobs").mkdir(exist_ok=True)
                initial = _seal(path, 0.06)
                _save(
                    path / "run.json",
                    dict(
                        run_id=rid,
                        agent_session=agent_session,
                        goal=GOAL,
                        deadline=time.time() + GOAL["wall_seconds"],
                        revision=initial,
                        revision_history=[initial],
                        offset_x_m=0.06,
                        attempts=[],
                        intents={},
                    ),
                )
        return self.inspect(rid)

    def inspect(self, run_id: str) -> dict:
        with self._locked(run_id) as path:
            data = _load(path / "run.json")
            data.pop("intents")
            data["attempts"] = [self._job(path, j) for j in data["attempts"]]
            return data

    def revise(
        self, run_id: str, expected_revision: str, offset_x_m: float, command_id: str
    ) -> dict:
        """Publish a new target revision while retaining the original trusted goal."""
        if not command_id or not math.isfinite(offset_x_m) or abs(offset_x_m) > 0.15:
            raise ValueError("Offset exceeds pilot workspace")
        with self._locked(run_id) as path:
            d = _load(path / "run.json")
            key = _hash(["revise", expected_revision, offset_x_m])
            if command_id in d["intents"]:
                digest, result = d["intents"][command_id]
                if key != digest:
                    raise ValueError("Command ID input conflict")
                return result
            self._admit(path, d, expected_revision)
            digest = _seal(path, offset_x_m, parent=expected_revision)
            d.update(revision=digest, offset_x_m=offset_x_m)
            d["revision_history"].append(digest)
            result = {"revision": digest}
            d["intents"][command_id] = [key, result]
            _save(path / "run.json", d)
            return result

    def validate(self, run_id: str, expected_revision: str) -> dict:
        with self._locked(run_id) as path:
            data = _load(path / "run.json")
            if data["revision"] != expected_revision:
                raise ValueError("Stale program revision")
            _validate(path / "bundles" / expected_revision, expected_revision)
            return dict(
                valid=True, bundle=expected_revision, physical_verdict="unavailable"
            )

    def _admit(self, path, d, revision):
        if d["revision"] != revision:
            raise ValueError("Stale program revision")
        if time.time() >= d["deadline"] or len(d["attempts"]) >= GOAL["max_attempts"]:
            raise ValueError("Experiment budget exhausted")
        if d["attempts"] and self._job(path, d["attempts"][-1])["state"] in (
            "queued",
            "running",
        ):
            raise ValueError("Attempt still active")

    def submit(self, run_id: str, expected_revision: str, command_id: str) -> dict:
        """Persist the intent and start a fresh worker with a kernel-owned lifetime lock."""
        if not command_id:
            raise ValueError("Command ID required")
        with self._locked(run_id) as path:
            d = _load(path / "run.json")
            key = _hash(["submit", expected_revision])
            if command_id in d["intents"]:
                digest, result = d["intents"][command_id]
                if digest != key:
                    raise ValueError("Command ID input conflict")
                return self._job(path, result)
            self._admit(path, d, expected_revision)
            _validate(path / "bundles" / expected_revision, expected_revision)
            job = uuid4().hex
            folder = path / "jobs" / job
            folder.mkdir()
            with (
                (folder / "worker.lock").open("wb") as lock,
                (folder / "worker.log").open("wb") as log,
            ):
                fcntl.flock(lock, fcntl.LOCK_EX)
                status = dict(
                    run_id=run_id,
                    job_id=job,
                    attempt_id=len(d["attempts"]) + 1,
                    bundle=expected_revision,
                    state="queued",
                    submitted_s=time.time(),
                    physical_verdict="unavailable",
                    program_completed=None,
                    episode_accepted=None,
                    data_committed=False,
                )
                _save(folder / "receipt.json", status)
                d["intents"][command_id] = [key, job]
                d["attempts"].append(job)
                _save(path / "run.json", d)
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        (
                            __name__
                            if __name__ != "__main__"
                            else "embodichain.gen_sim.agent_experiment"
                        ),
                        str(folder),
                    ],
                    pass_fds=(lock.fileno(),),
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            return status

    @staticmethod
    def _job(path: Path, job_id: str) -> dict:
        if not _ID.fullmatch(job_id):
            raise ValueError("Invalid job ID")
        folder = path / "jobs" / job_id
        with (folder / "worker.lock").open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return _load(folder / "receipt.json")
            state = _load(folder / "receipt.json")
            if state["state"] in ("queued", "running"):
                state.update(
                    state="unknown",
                    error="Worker exited before final physical readback",
                )
                _save(folder / "receipt.json", state)
            return state

    def cancel(self, run_id: str, job_id: str) -> dict:
        with self._locked(run_id) as path:
            receipt = self._job(path, job_id)
            if receipt["state"] in ("queued", "running"):
                (path / "jobs" / job_id / "cancel").touch()
                receipt["cancel_requested"] = True
            return receipt


def _worker(folder: Path) -> None:
    from ._run_execution import _execute_deployment

    run = folder.parents[1]
    s = _load(folder / "receipt.json")
    d = _load(run / "run.json")
    s.update(state="running", started_s=time.time(), physics_steps_per_control=4)
    _save(folder / "receipt.json", s)
    count = 0

    def check(*_):
        nonlocal count
        count += 1
        if (folder / "cancel").exists() or time.time() >= d["deadline"]:
            raise InterruptedError("Execution stop requested")

    @contextmanager
    def world_scope(env):
        import dexsim
        from dexsim.runtime import SessionContext
        from dexsim.runtime.command_queue import CommandQueue
        from dexsim.runtime.errors import UnsupportedOperationError
        from dexsim.runtime.host import _publish_json
        from dexsim.kit.mcp import dispatch

        session_id = "dsws_" + s["job_id"]
        sim = env.sim
        ctx = SessionContext(
            sim.get_world(), sim._spawn_scene.builder, session_id=session_id
        )
        start_tick = sim.physics_step_index

        def reject(*_):
            raise UnsupportedOperationError(
                "This attempt World is owned by its Gym worker"
            )

        def read(context, params):
            kw = params.get("arguments", {})
            if params.get("module") == "dexsim.kit.mcp.dispatch" and kw.get(
                "method"
            ) in {
                "world.status",
                "scene.list_objects",
                "articulation.get_state",
                "simulation.get_rigid_body_state",
            }:
                return dispatch.run(context, **kw)
            return reject()

        sessions = folder / "sessions"
        sessions.mkdir(mode=0o700)
        record = sessions / (session_id + ".json")
        try:
            with CommandQueue(
                ctx,
                {"adapter.invoke": read, "session.shutdown": reject},
                journal=folder / "world.jsonl",
            ) as queue:
                info = dict(
                    session_id=session_id,
                    pid=os.getpid(),
                    host=queue.address[0],
                    port=queue.address[1],
                    journal=str(folder / "world.jsonl"),
                    request_timeout=30,
                    config={"backend": "newton", "solver": "dexuni"},
                    runtime={
                        "dexsim_version": dexsim.__version__,
                        "build_id": dexsim.__build_id__,
                    },
                )
                if getattr(queue, "token", None):
                    info["token"] = queue.token
                _publish_json(record, info)
                s.update(
                    session_id=session_id,
                    session_directory=str(sessions),
                    topology_revision=ctx.scene.topology_revision,
                    physics_step_origin=start_tick,
                    physics_dt=sim.sim_config.physics_dt,
                )
                _save(folder / "receipt.json", s)
                initial_q = None
                next_receipt_at = time.monotonic()

                def poll():
                    nonlocal initial_q, next_receipt_at
                    queue.poll(timeout=0)
                    pose = env.robot.get_qpos(name="eef").detach().cpu().tolist()[0]
                    if initial_q is None:
                        initial_q = pose
                    s.update(
                        physics_steps=sim.physics_step_index - start_tick,
                        simulation_time_s=sim.simulation_time,
                        initial_gripper=initial_q,
                        last_gripper=pose,
                    )
                    now = time.monotonic()
                    if now >= next_receipt_at:
                        _save(folder / "receipt.json", s)
                        next_receipt_at = now + _PROGRESS_INTERVAL_S

                yield poll
        finally:
            record.unlink(missing_ok=True)
            ctx.close()

    try:
        check()
        count = 0
        revision = s["bundle"]
        entry = _validate(run / "bundles" / revision, revision)
        result = _execute_deployment(
            entry,
            device="cuda:0",
            headless=True,
            recording=folder / "recording",
            step_observer=check,
            world_scope=world_scope,
        )
        _save(folder / "execution.json", result)
        trace = folder / "recording" / "states.jsonl"
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        base = rows[0]["objects"]["cube"]
        final = rows[-1]["objects"]["cube"]
        goal = rows[-1]["objects"]["goal"]
        error = [final[i] - goal[i] for i in range(2)]
        lift = max(row["objects"]["cube"][2] for row in rows) - base[2]
        release = all(
            abs(x - y) < 0.002
            for x, y in zip(s["initial_gripper"], s["last_gripper"], strict=True)
        )
        passed = (
            math.hypot(*error) <= GOAL["xy_tolerance_m"]
            and lift >= GOAL["min_lift_m"]
            and release
        )
        s.update(
            state="succeeded" if result["all_success"] and passed else "failed",
            program_completed=result.get("completed"),
            physical_verdict="pass" if passed else "fail",
            episode_accepted=bool(result["all_success"] and passed),
            data_committed=True,
            measurement=dict(horizontal_error_m=error, lift_m=lift, released=release),
            trajectory=str(trace),
            trajectory_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        )
    except BaseException as exc:
        stop = (folder / "cancel").exists() or time.time() >= d["deadline"]
        s.update(state="cancelled" if stop else "failed", error=str(exc))
    finally:
        s.update(finished_s=time.time(), observed_control_steps=count)
        _save(folder / "receipt.json", s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    _worker(parser.parse_args().folder.resolve())
