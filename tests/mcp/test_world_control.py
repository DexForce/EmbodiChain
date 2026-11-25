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

"""Qualification of control admission, partial effects and reply-loss fencing."""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import Popen
import sys
from time import monotonic, sleep

import pytest
from dexsim.runtime import Session
from dexsim.runtime.errors import SessionConnectionError, SessionRemoteError

from embodichain.lab.sim.dexsim_session.writer import WriterClient
from examples.sim.attached_world.fixture import ARM, native


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dexsim", "newton"])
def test_dedicated_writer(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "host"
    with Popen(
        [
            sys.executable,
            "-m",
            "examples.sim.attached_world.control_host",
            "--directory",
            str(directory),
            "--backend",
            backend,
        ]
    ) as host:
        try:
            deadline = monotonic() + 120
            while not list(directory.glob("dsws_*.json")):
                if host.poll() is not None or monotonic() > deadline:
                    pytest.fail("Dedicated host failed to start")
                sleep(0.2)
            (record,) = directory.glob("dsws_*.json")
            sid = record.stem
            key = directory / "writer.key"
            with Session.connect(sid, directory=directory) as session:
                writer = WriterClient(session, key)
                state = writer.call("status", {})
                req = dict(
                    command_id="one",
                    session_id=sid,
                    revision=state["revision"],
                    topology_revision=state["topology_revision"],
                    robot_uid="UR5",
                    control_part="arm",
                    units=["rad"] * 6,
                    target=[0.15, -0.15, 0.12, -0.15, 0.12, -0.15],
                    control_interval_s=0.04,
                )

                def read(name):
                    return native(
                        session,
                        "articulation.get_state",
                        object_ref=name,
                        joint_names=ARM,
                    )

                a0, b0 = read("robot_a"), read("robot_b")
                for changes in (
                    {"session_id": "wrong"},
                    {"topology_revision": -1},
                    {"revision": "old"},
                    {"control_part": "absent"},
                    {"units": ["m"] * 6},
                    {"target": [900.0] * 6},
                    {"control_interval_s": 0.03},
                ):
                    with pytest.raises(SessionRemoteError):
                        writer.call("target_and_step", {**req, **changes})
                bad_key = tmp_path / "wrong.key"
                bad_key.write_bytes(b"x" * 32)
                with pytest.raises(SessionRemoteError):
                    WriterClient(session, bad_key).call("target_and_step", req)
                for method in ("simulation.step", "articulation.set_state"):
                    with pytest.raises(SessionRemoteError):
                        native(session, method)
                with pytest.raises(SessionRemoteError):
                    session.evaluate("world.update(.01)")
                with pytest.raises(SessionRemoteError):
                    session.shutdown()
                assert (
                    read("robot_b")["joint_target_position"]
                    == b0["joint_target_position"]
                )
                assert writer.call("status", {})["control_step"] == 0

                invoke = session.invoke

                def drop_reply(*args, **kwargs):
                    invoke(*args, **kwargs)
                    raise SessionConnectionError("Dropped accepted reply")

                monkeypatch.setattr(session, "invoke", drop_reply)
                with pytest.raises(SessionConnectionError):
                    writer.call("target_and_step", req)
            with Session.connect(sid, directory=directory) as reconnected:
                writer = WriterClient(reconnected, key)
                with pytest.raises(RuntimeError, match="pending"):
                    writer.call("target_and_step", {**req, "command_id": "other"})
                # MCP supplies the default tolerance explicitly; Python may omit it.
                recovered = writer.call("target_and_step", {**req, "tolerance": 1e-3})
                assert writer.call("receipt", {"command_id": "one"}) == recovered
                assert recovered["state"] == "ready" and recovered["target_accepted"]
                assert writer.call("target_and_step", req) == recovered
                assert (
                    recovered["physics_steps"] == 4 and recovered["control_step"] == 1
                )
                assert recovered["target_attained"] == all(
                    abs(a - b) <= 0.001
                    for a, b in zip(req["target"], recovered["measured_after"]["qpos"])
                )
                stopped = writer.call("stop", {})
                assert stopped["state"] == "stopped" and stopped["control_step"] == 1
                with pytest.raises(SessionRemoteError):
                    writer.call(
                        "target_and_step",
                        {
                            **req,
                            "command_id": "future",
                            "revision": recovered["revision"],
                        },
                    )
                a = native(
                    reconnected,
                    "articulation.get_state",
                    object_ref="robot_a",
                    joint_names=ARM,
                )
                b = native(
                    reconnected,
                    "articulation.get_state",
                    object_ref="robot_b",
                    joint_names=ARM,
                )
                assert a["joint_target_position"] == a0["joint_target_position"]
                assert b["joint_target_position"] == pytest.approx(req["target"])
                assert b["joint_position"] != pytest.approx(b0["joint_position"])
                assert b["joint_position"] == pytest.approx(
                    recovered["measured_after"]["qpos"]
                )
                if backend == "newton":
                    assert b["step_index"] - b0["step_index"] == 4
                    assert b["simulation_time"] - b0[
                        "simulation_time"
                    ] == pytest.approx(0.04)
            # Leave native evidence for independent readback when --basetemp is supplied.
            (directory / "readback.json").write_text(
                json.dumps(
                    dict(before=b0, after=b, receipt=recovered, stop=stopped), indent=2
                )
            )
        finally:
            host.terminate()
            host.wait(timeout=30)


def test_partial_native_step_fences_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception after an accepted target must never liberate a second write."""
    from unittest.mock import Mock
    import torch
    from embodichain.lab.sim.dexsim_session import control

    ctx = Mock(session_id="example")
    ctx.scene.backend = "dexsim"
    ctx.world.is_physics_manually_update.return_value = True
    ctx.world.update.side_effect = [None, RuntimeError("native step error")]
    manager = Mock(_context=ctx, session_id="example", topology_revision=3)
    robot = manager.get_robot.return_value
    robot.mimic_ids = []
    robot.device = torch.device("cpu")
    robot.get_qpos_limits.return_value = torch.tensor([[[-2.0, 2.0]]])
    robot.get_qpos.return_value = torch.tensor([[0.3]])
    sample = dict(units=["rad"], dof_ids=[0], qpos=[0.0], step_index=None)
    monkeypatch.setattr(control, "read_robot_state", lambda *a, **kw: sample)
    owner = control.WorldControl(ctx, manager, physics_dt=0.01, steps_per_control=4)
    request = dict(
        command_id="once",
        session_id="example",
        topology_revision=3,
        revision=owner.status()["revision"],
        robot_uid="r",
        control_part="arm",
        target=[0.3],
        units=["rad"],
        control_interval_s=0.04,
    )
    first = owner.target_and_step(**request)
    assert (
        first["state"] == "effect_unknown"
        and first["physics_steps"] == 1
        and first["control_step"] == 0
    )
    assert first["target_accepted"] and first["target_attained"] is None
    assert owner.target_and_step(**request) == first
    assert owner.stop()["state"] == "effect_unknown"
    with pytest.raises(ValueError):
        owner.target_and_step(
            **{**request, "command_id": "again", "revision": first["revision"]}
        )
    manager.validate.side_effect = RuntimeError("scene replaced")
    assert owner.status()["binding_current"] is False
    assert owner.receipt("once") == first
    robot.set_qpos.assert_called_once()
    assert ctx.world.update.call_count == 2
