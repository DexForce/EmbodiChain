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

"""Stdio MCP tools for the World owner's configured motion capabilities."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
from time import monotonic
from typing import Any, Literal

from dexsim.runtime import Session
from dexsim.runtime.errors import DexSimRuntimeError
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from typing_extensions import TypedDict

from embodichain.lab.sim.dexsim_session._motion_jobs import MODULE, RUN_ID, sign
from .tools.robot import register_robot_tools

__all__: list[str] = []


class JointPosition(TypedDict):
    joint_names: list[str]
    qpos: list[float]
    units: list[Literal["rad", "m"]]


class Pose(TypedDict):
    frame: Literal["arena"]
    position_m: list[float]
    quaternion_xyzw: list[float]


def serve(
    session_id: str,
    output: Path,
    key_file: Path,
    *,
    directory: str | None = None,
    labels: dict[str, Any] | None = None,
) -> None:
    """Serve one authorized World, with read-only polling from its journaled receipts."""
    key = key_file.read_bytes()
    if len(key) != 32:
        raise ValueError("Invalid motion writer grant file")
    client = {"kind": "embodichain-motion-mcp", **(labels or {})}
    with ExitStack() as stack:
        try:
            session = stack.enter_context(
                Session.connect(session_id, directory=directory, client=client)
            )
        except (DexSimRuntimeError, OSError):
            session = None
        _server(session_id, session, key, output.resolve()).run(transport="stdio")


def _server(
    session_id: str, session: Session | None, key: bytes, output: Path
) -> MCPServer:
    server = MCPServer("EmbodiChain motion")
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
    if session is not None:
        register_robot_tools(server, session)

    def invoke(operation: str, **arguments: Any) -> dict[str, Any]:
        if session is None:
            raise ToolError("WORLD_UNREACHABLE")
        try:
            return session.invoke(
                MODULE,
                operation=operation,
                arguments=arguments,
                signature=sign(key, session.session_id, operation, arguments),
            )
        except (DexSimRuntimeError, ValueError) as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=readonly)
    def embodichain_list_capabilities() -> dict[str, Any]:
        """Read the bound robot, control groups, models, skills, and planner profiles."""
        return invoke("capabilities")

    @server.tool(annotations=readonly)
    def embodichain_fk(
        control_part: str, joints: JointPosition | None = None
    ) -> dict[str, Any]:
        """Compute TCP pose in the arena frame. Omit joints to use measured state."""
        return invoke("fk", control_part=control_part, joints=joints)

    @server.tool(annotations=readonly)
    def embodichain_ik(
        control_part: str, target: Pose, seed: JointPosition | None = None
    ) -> dict[str, Any]:
        """Solve a named control group's IK using the scene TCP and joint limits."""
        return invoke("ik", control_part=control_part, target=target, seed=seed)

    @server.tool(annotations=readonly)
    def embodichain_plan_motion(
        control_part: str,
        planner_profile: str,
        waypoints: list[JointPosition | Pose],
    ) -> dict[str, Any]:
        """Plan timed motion for the current World snapshot. Read the qualified check scope."""
        return invoke(
            "plan_motion",
            control_part=control_part,
            planner_profile=planner_profile,
            waypoints=waypoints,
        )

    @server.tool()
    def embodichain_execute_motion(
        motion_plan_id: str, command_id: str
    ) -> dict[str, Any]:
        """Execute a motion plan after rechecking the live World and starting state."""
        return invoke(
            "execute_motion", motion_plan_id=motion_plan_id, command_id=command_id
        )

    @server.tool()
    def embodichain_start_skill(
        skill_id: str,
        object_ref: str,
        command_id: str,
        grasp_ref: str | None = None,
        target_ref: str | None = None,
    ) -> dict[str, Any]:
        """Execute one qualified skill. place requires the PickUp grasp_ref and a support target."""
        return invoke(
            "start_skill",
            skill_id=skill_id,
            object_ref=object_ref,
            grasp_ref=grasp_ref,
            target_ref=target_ref,
            command_id=command_id,
        )

    @server.tool(annotations=readonly)
    def embodichain_get_grasp(grasp_ref: str) -> dict[str, Any]:
        """Recheck a grasp against the current robot and object measurements."""
        return invoke("get_grasp", grasp_ref=grasp_ref)

    @server.tool(annotations=readonly)
    def embodichain_prepare_program(program_id: str) -> dict[str, Any]:
        """Resolve an operator-registered Task Program in the current World."""
        return invoke("prepare_program", program_id=program_id)

    @server.tool()
    def embodichain_start_run(plan_id: str, command_id: str) -> dict[str, Any]:
        """Start the prepared Task Program using the shared motion-job lifecycle."""
        return invoke("start_run", plan_id=plan_id, command_id=command_id)

    @server.tool(annotations=readonly)
    async def embodichain_get_run(
        run_id: str, wait_seconds: float = 0.0
    ) -> dict[str, Any]:
        """Read a durable receipt. Optional wait is capped at 20 seconds."""
        if not RUN_ID.fullmatch(run_id) or not 0 <= wait_seconds <= 20:
            raise ToolError("Invalid run ID or wait_seconds outside [0, 20]")
        path = output / run_id / "receipt.json"
        deadline = monotonic() + wait_seconds
        while True:
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ToolError("Run receipt is unavailable") from exc
            if receipt.get("session_id") != session_id:
                raise ToolError("The receipt belongs to another World")
            if receipt["done"]:
                return receipt
            if session is None:
                return {
                    **receipt,
                    "world_reachable": False,
                    "physical_verdict": "unverified",
                }
            if monotonic() >= deadline:
                return receipt
            await asyncio.sleep(min(0.2, max(0.0, deadline - monotonic())))

    @server.tool()
    def embodichain_cancel_run(run_id: str) -> dict[str, Any]:
        """Request cancellation. Poll until the host confirms the stop boundary."""
        return invoke("cancel_run", run_id=run_id)

    return server
