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

"""Check motion admission and the stop boundary observed by remote callers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.lab.sim.dexsim_session._motion_jobs import (
    MODULE,
    MotionJobs,
    executors,
    sign,
)


def _owner() -> Mock:
    return Mock(session_id="dsws_" + "a" * 32)


def test_command_replay_does_not_advance_and_cancel_stops_at_boundary(
    tmp_path: Path,
) -> None:
    context = _owner()
    consumed = []
    closed = []
    jobs = MotionJobs(context, tmp_path, lambda: {"control_step": len(consumed)})

    def work(_folder):
        try:
            for i in range(3):
                consumed.append(i)
                yield {"phase": "pick"}
        finally:
            closed.append(True)
        return {"passed": True}

    first = jobs.start("pick1", "skill", {"object": "cube"}, work)
    assert first["status"] == "queued"
    tick = context.register_update.call_args.args[1]
    tick(0)
    assert jobs.start("pick1", "skill", {"object": "cube"}, work)["control_step"] == 1
    with pytest.raises(ValueError, match="COMMAND_ID_CONFLICT"):
        jobs.start("pick1", "skill", {"object": "other"}, work)
    with pytest.raises(ValueError, match="WORLD_BUSY"):
        jobs.start("pick2", "skill", {"object": "cube"}, work)
    assert jobs.cancel(first["run_id"])["status"] == "cancelling"
    assert consumed == [0] and not closed
    tick(1)
    receipt = json.loads((tmp_path / first["run_id"] / "receipt.json").read_text())
    assert receipt["done"] and receipt["status"] == "cancelled"
    assert receipt["control_step"] == 1 and receipt["physical_verdict"] == "unverified"
    assert consumed == [0] and closed == [True]
    assert jobs.start("pick1", "skill", {"object": "cube"}, work) == receipt


def test_owner_grant_covers_the_bound_session_and_intent() -> None:
    from dexsim.runtime.errors import InvalidParamsError, UnsupportedOperationError

    context = _owner()
    secret = b"x" * 32
    execute = Mock(return_value={"command_id": "pick1"})
    table = executors(context, {"start_skill": execute}, secret)
    invoke = table["adapter.invoke"]
    args = {"command_id": "pick1"}
    signature = sign(secret, context.session_id, "start_skill", args)

    def call(arguments, proof):
        return invoke(
            context,
            {
                "module": MODULE,
                "arguments": {
                    "operation": "start_skill",
                    "arguments": arguments,
                    "signature": proof,
                },
            },
        )

    assert call(args, signature) == {"command_id": "pick1"}
    with pytest.raises(InvalidParamsError, match="Invalid motion writer grant"):
        call({"command_id": "pick2"}, signature)
    with pytest.raises(InvalidParamsError, match="Invalid motion writer grant"):
        call(args, sign(secret, "dsws_other", "start_skill", args))
    with pytest.raises(UnsupportedOperationError):
        invoke(context, {"module": "unrestricted_script", "arguments": {}})
    with pytest.raises(UnsupportedOperationError):
        invoke(
            context,
            {
                "module": "dexsim.kit.mcp.dispatch",
                "arguments": {"method": "simulation.step", "params": {"steps": 1}},
            },
        )
    execute.assert_called_once_with(command_id="pick1")


def test_grasp_is_revoked_when_measured_object_moves() -> None:
    import numpy as np

    from examples.learning.policy_runs._ur5_motion import Ur5Motion

    tcp = np.eye(4)
    cube = np.eye(4)
    session = Ur5Motion.__new__(Ur5Motion)
    session.state = "held"
    session.grasp = {"grasp_ref": "valid", "cube_in_tcp": cube.tolist()}
    session.episode = SimpleNamespace(
        manager=Mock(),
        grip_index=0,
        grip_open=0.0,
        grip_close=0.024,
        observe=lambda: {
            "tcp_pose": tcp,
            "cube_pose": cube,
            "physics_step": 12,
            "qpos": [0.015],
        },
    )
    assert session.get_grasp("valid")["verified"]
    cube[0, 3] += 0.05
    assert session.get_grasp("valid")["code"] == "GRASP_LOST"
    cube[0, 3] -= 0.05
    assert not session.get_grasp("valid")["verified"]


@pytest.mark.gpu
def test_ur5_world_rejects_place_after_measured_grasp_loss(tmp_path: Path) -> None:
    """Check the real borrowed UR5 before and after a host-side cube move."""
    from contextlib import ExitStack
    import numpy as np

    import dexsim
    from dexsim.runtime import SessionContext
    from dexsim.runtime.config import normalize_config
    from dexsim.runtime.host import _world_config

    from examples.learning.policy_runs._ur5_motion import Ur5Motion
    from examples.learning.policy_runs.ur5_motion_host import _DEPLOYMENT
    from examples.learning.policy_runs.ur5_pick_place import _Episode

    config = normalize_config(
        dict(backend="dexsim", renderer="fast-rt", device="cpu", headless=True, dt=0.01)
    )
    world = dexsim.World(_world_config(config))
    world.set_manual_update(True)
    world.enable_physics(True)
    context = SessionContext(
        world, dexsim.spawn.SceneBuilder(world), session_id="dsws_" + "b" * 32
    )
    with ExitStack() as stack:
        stack.callback(context.close)
        episode = _Episode(context, _DEPLOYMENT, False, stack)
        motion = Ur5Motion(episode, tmp_path)
        stack.callback(motion.close)

        def receipt(job):
            return json.loads((tmp_path / job["run_id"] / "receipt.json").read_text())

        joints = motion._arm_joint_fields("arm")
        joints["qpos"] = episode.robot.get_qpos("arm")[0].tolist()
        joints["qpos"][0] += 0.025
        preview = motion.plan_motion("arm", "ur5/trapezoidal@1", [joints])
        moving = motion.execute_motion(preview["motion_plan_id"], "short-motion")
        motion.jobs._tick(0)
        assert episode.physics_step == episode.substeps
        motion.jobs.cancel(moving["run_id"])
        motion.jobs._tick(0)
        assert receipt(moving)["status"] == "cancelled"
        assert episode.physics_step == episode.substeps

        pick = motion.start_skill("pick_up", "cube", "lift")
        for _ in range(2401):
            if motion.jobs.active is None:
                break
            motion.jobs._tick(0)
        result = receipt(pick)
        assert result["status"] == "succeeded", result
        grasp_ref = result["result"]["grasp_ref"]
        assert motion.get_grasp(grasp_ref)["verified"]
        steps = episode.physics_step

        cancelled = motion.start_skill(
            "place", "cube", "cancel-before-place", grasp_ref, "platform_top"
        )
        assert motion.jobs.cancel(cancelled["run_id"])["status"] == "cancelling"
        motion.jobs._tick(0)
        assert receipt(cancelled)["status"] == "cancelled"
        assert episode.physics_step == steps
        assert motion.get_grasp(grasp_ref)["verified"]

        # Simulate the owner externally displacing the cube while physics is paused.
        pose = np.asarray(episode.cube.get_world_pose()).copy()
        pose[0, 3] += 0.10
        episode.cube.set_world_pose(pose)
        stale = motion.start_skill(
            "place", "cube", "place-after-slip", grasp_ref, "platform_top"
        )
        motion.jobs._tick(0)
        assert receipt(stale)["error"]["code"] == "GRASP_LOST"
        assert episode.physics_step == steps
