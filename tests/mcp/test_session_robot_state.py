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
import sys
from pathlib import Path
from typing import Any

import pytest


def test_scalar_layout_mapping() -> None:
    from dexsim.scene import JointDofLayout
    from embodichain.lab.sim.dexsim_session import _select_dofs

    layout = [
        JointDofLayout("base", 0, 6, "translation_rotation_vector"),
        JointDofLayout("joint2", 6, 1, "angle"),
        JointDofLayout("slide", 7, 1, "translation"),
        JointDofLayout("joint1", 8, 1, "angle"),
    ]
    assert _select_dofs(layout, ["slide", "joint1", "joint2"]) == (
        ["slide", "joint1", "joint2"],
        [7, 8, 6],
        ["m", "rad", "rad"],
    )
    assert _select_dofs(layout, ["joint.*"])[1] == [6, 8]
    for invalid in (["joint3"], ["joint.*", "joint2"], ["base"]):
        with pytest.raises(ValueError):
            _select_dofs(layout, invalid)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dexsim", "newton"])
def test_attach_two_ur5_and_reconnect(tmp_path: Path, backend: str) -> None:
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

    def native(method: str, **params: Any) -> dict[str, Any]:
        return session.invoke("dexsim.kit.mcp.dispatch", method=method, params=params)

    async def read_b(expected: list[float]) -> None:
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
            ],
        )
        async with mcp.Client(parameters) as client:
            catalog = (await client.list_tools()).tools
            assert [tool.name for tool in catalog] == ["embodichain_get_robot_state"]
            tool = catalog[0]
            assert tool.annotations.read_only_hint
            assert tool.output_schema is not None
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
                assert state["qpos"] == pytest.approx(expected, abs=1e-6)
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
        for name, pose in (("ur5_a", pose_a), ("ur5_b", pose_b)):
            native(
                "articulation.set_state",
                object_ref=name,
                joint_names=arm,
                joint_position=pose,
            )
        initial = native("articulation.get_state", object_ref="ur5_b", joint_names=arm)
        assert initial["joint_position"] == pytest.approx(pose_b)
        assert initial["joint_target_position"] != pytest.approx(pose_b)
        asyncio.run(read_b(initial["joint_position"]))
        # Change B on the host while the first stdio process is gone.
        pose_b[2] = 0.72
        native(
            "articulation.set_state",
            object_ref="ur5_b",
            joint_names=arm,
            joint_position=pose_b,
        )
        asyncio.run(read_b(pose_b))
        final = native("articulation.get_state", object_ref="ur5_a", joint_names=arm)
        assert final["step_index"] == initial["step_index"]
        assert final["joint_position"] == pytest.approx(pose_a)
    finally:
        session.shutdown()
