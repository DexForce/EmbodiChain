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

"""Opt-in MCP tools for a dedicated host's target-and-step controller."""

from __future__ import annotations

from typing import Any
from dexsim.runtime.errors import DexSimRuntimeError
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from embodichain.lab.sim.dexsim_session.writer import WriterClient

__all__: list[str] = []


def register_control_tools(server: MCPServer, writer: WriterClient) -> None:
    def invoke(operation: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return writer.call(operation, args)
        except (DexSimRuntimeError, ValueError, RuntimeError) as exc:
            raise ToolError(str(exc)) from exc

    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False)

    @server.tool(annotations=readonly)
    def embodichain_control_status() -> dict[str, Any]:
        """Read the writer binding and clock counters without advancing the World."""
        return invoke("status", {})

    @server.tool(annotations=readonly)
    def embodichain_control_receipt(command_id: str) -> dict[str, Any]:
        """Reconcile an intent by its original command ID without replaying it."""
        return invoke("receipt", {"command_id": command_id})

    @server.tool()
    def embodichain_control_target_and_step(
        command_id: str,
        session_id: str,
        topology_revision: int,
        revision: str,
        robot_uid: str,
        control_part: str,
        target: list[float],
        units: list[str],
        control_interval_s: float,
        tolerance: float = 1e-3,
    ) -> dict[str, Any]:
        """Set the selected group's joint targets and advance one authorized interval."""
        return invoke(
            "target_and_step",
            dict(
                command_id=command_id,
                session_id=session_id,
                topology_revision=topology_revision,
                revision=revision,
                robot_uid=robot_uid,
                control_part=control_part,
                target=target,
                units=units,
                control_interval_s=control_interval_s,
                tolerance=tolerance,
            ),
        )

    @server.tool()
    def embodichain_control_stop() -> dict[str, Any]:
        """Stop issuing writes and report the actual completed control boundary."""
        return invoke("stop", {})
