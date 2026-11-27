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

"""Control-part ordering and stdio attachment to an existing two-UR5 World."""

from __future__ import annotations

import asyncio
import json
import sys
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


def test_host_imports_without_mcp_sdk() -> None:
    """Direct Session.invoke users can install just the World host dependencies."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["mcp"] = None
from embodichain.lab.observations.joints import ControlPartJointState, SessionJointState
from embodichain.lab.sim.dexsim_session import read_robot_state, run
assert "embodichain.mcp" not in sys.modules
""",
        ],
        check=True,
    )


def test_scalar_layout_mapping() -> None:
    from dexsim.scene import JointDofLayout
    from embodichain.lab.sim.dexsim_session import read_robot_state
    from embodichain.lab.sim.dexsim_session.robot_state import _select_dofs

    layout = [
        JointDofLayout("base", 0, 6, "translation_rotation_vector"),
        JointDofLayout("joint2", 6, 1, "angle"),
        JointDofLayout("slide", 7, 1, "translation"),
        JointDofLayout("joint1", 8, 1, "angle"),
    ]
    handle = Mock(path="test_robot", joint_dof_layout=layout)
    handle.get_current_qpos.return_value = [0] * 6 + [0.1, 0.2, 0.3]
    handle.get_current_qvel.return_value = [0] * 6 + [-1, -2, -3]
    scene = Mock(backend="dexsim", topology_revision=4)
    scene.get_articulation.return_value = handle
    state = read_robot_state(
        Mock(session_id="test_session", scene=scene),
        object_ref="test_robot",
        robot_cfg=Mock(control_parts={"group": ["slide", "joint1", "joint2"]}),
        control_part="group",
    )
    assert state["joint_names"] == ["slide", "joint1", "joint2"]
    assert state["dof_ids"] == [7, 8, 6]
    assert state["units"] == ["m", "rad", "rad"]
    assert state["qpos"] == [0.2, 0.3, 0.1]
    assert state["qvel"] == [-2, -3, -1]
    assert _select_dofs(layout, ["joint.*"])[1] == [6, 8]
    for invalid in (["joint3"], ["joint.*", "joint2"], ["base"]):
        with pytest.raises(ValueError):
            _select_dofs(layout, invalid)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dexsim", "newton"])
def test_attach_two_ur5_and_reconnect(
    tmp_path: Path, backend: str, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("mcp.server.mcpserver")
    import mcp
    from dexsim.runtime import Session
    from embodichain.data import get_data_path

    urdf = Path(get_data_path("UniversalRobots/UR5/UR5.urdf"))
    session = Session.create(
        directory=tmp_path,
        backend=backend,
        **({"solver": "mujoco_warp"} if backend == "newton" else {}),
        device="cuda:0",
        headless=True,
        asset_root=str(urdf.parent),
        request_timeout=60,
    )
    arm = [f"joint{i}" for i in range(1, 7)]
    pose_a = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    pose_b = [-0.6, 0.5, -0.4, 0.3, -0.2, 0.1]

    vel_a = [0.11, -0.12, 0.13, -0.14, 0.15, -0.16]
    vel_b = [-0.21, 0.22, -0.23, 0.24, -0.25, 0.26]

    def native(method: str, **params: Any) -> dict[str, Any]:
        return session.invoke("dexsim.kit.mcp.dispatch", method=method, params=params)

    async def read_b(expected: dict[str, Any], *, catalog: bool = False) -> None:
        parameters = mcp.StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "embodichain",
                "mcp",
                "--session",
                session.session_id,
                "--session-directory",
                str(tmp_path),
                *(["--catalog"] if catalog else []),
            ],
        )
        async with mcp.Client(parameters) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            if catalog:
                uri = f"embodichain://worlds/{session.session_id}/manifest"
                manifest = await client.read_resource(uri)
                assert json.loads(manifest.contents[0].text) == {
                    "schema_version": 1,
                    "session_id": session.session_id,
                    "backend": backend,
                    "solver": session.info["config"]["solver"],
                    "device": "cuda:0",
                    "timestep_s": session.info["config"]["dt"],
                    "dexsim_version": session.info["runtime"]["dexsim_version"],
                    "scene_attached": True,
                    "builder_finalized": True,
                    "shutdown_requested": False,
                }
                health = await client.call_tool("embodichain_health")
                assert not health.is_error
                assert health.structured_content["status"] == "ready"
                assert health.structured_content["session_id"] == session.session_id
                assert health.structured_content["host_state"] in ("idle", "busy")
            else:
                assert set(tools) == {"embodichain_get_robot_state"}
            tool = tools["embodichain_get_robot_state"]
            assert tool.annotations.read_only_hint
            assert "qvel" in tool.output_schema["required"]
            args = dict(
                robot_cfg="URRobot",
                variant="ur5",
                control_part="arm",
                object_ref="ur5_b",
            )
            error = await client.call_tool(tool.name, {**args, "object_ref": "missing"})
            assert error.is_error
            for _ in range(2):
                response = await client.call_tool(tool.name, args)
                assert not response.is_error, response.content
                state = response.structured_content
                assert state["object_ref"] == "ur5_b"
                assert state["session_id"] == session.session_id
                assert state["joint_names"] == arm
                assert state["units"] == ["rad"] * 6
                assert state["qpos"] == pytest.approx(
                    expected["joint_position"], abs=1e-6
                )
                assert state["qvel"] == pytest.approx(
                    expected["joint_velocity"], abs=1e-6
                )
                assert state["step_index"] == initial["step_index"]
                assert state["simulation_time_s"] == initial["simulation_time"]

    try:
        for name, x in (("ur5_a", 0), ("ur5_b", 2)):
            native(
                "articulation.import_urdf",
                name=name,
                asset_path=urdf.name,
                position=[x, 0, 0],
                fixed_base=True,
            )
        native("scene.finalize")
        for name, pose, velocity in (
            ("ur5_a", pose_a, vel_a),
            ("ur5_b", pose_b, vel_b),
        ):
            native(
                "articulation.set_state",
                object_ref=name,
                joint_names=arm,
                joint_position=pose,
                joint_velocity=velocity,
            )
        initial = native("articulation.get_state", object_ref="ur5_b", joint_names=arm)
        assert initial["joint_position"] == pytest.approx(pose_b)
        assert initial["joint_velocity"] == pytest.approx(vel_b)
        assert initial["joint_target_velocity"] != pytest.approx(vel_b)
        assert initial["joint_target_position"] != pytest.approx(pose_b)
        legacy = session.invoke(
            "embodichain.lab.sim.dexsim_session",
            object_ref="ur5_b",
            robot_cfg="URRobot",
            variant="ur5",
            control_part="arm",
        )
        assert legacy["qpos"] == pytest.approx(initial["joint_position"])
        assert legacy["qvel"] == pytest.approx(initial["joint_velocity"])
        asyncio.run(read_b(initial))
        # Change B on the host while the first stdio process is gone.
        pose_b[2] = 0.72
        vel_b[2] = 0.37
        native(
            "articulation.set_state",
            object_ref="ur5_b",
            joint_names=arm,
            joint_position=pose_b,
            joint_velocity=vel_b,
        )
        updated = native("articulation.get_state", object_ref="ur5_b", joint_names=arm)
        assert updated["joint_position"] == pytest.approx(pose_b)
        assert updated["joint_velocity"] == pytest.approx(vel_b)
        asyncio.run(read_b(updated, catalog=True))
        after = native("articulation.get_state", object_ref="ur5_b", joint_names=arm)
        for signal in ("joint_position", "joint_velocity"):
            assert after[signal] == pytest.approx(updated[signal])
        final = native("articulation.get_state", object_ref="ur5_a", joint_names=arm)
        assert final["step_index"] == initial["step_index"]
        assert final["joint_velocity"] == pytest.approx(vel_a)
        assert final["joint_position"] == pytest.approx(pose_a)
        assert "Failed to parse JSONRPC message from server" not in caplog.text
    finally:
        session.shutdown()
