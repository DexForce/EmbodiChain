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

"""Serve EmbodiChain queries over stdio for an explicitly selected session."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ._types import RobotState

__all__ = ["cli"]


def cli(argv: Sequence[str] | None = None) -> None:
    """Attach a stdio MCP server to an existing DexSim runtime session."""
    parser = argparse.ArgumentParser(prog="embodichain mcp", description=__doc__)
    parser.add_argument(
        "--session", required=True, help="Existing DexSim Runtime session ID"
    )
    parser.add_argument(
        "--session-directory", help="Optional DexSim discovery directory"
    )
    args = parser.parse_args(argv)
    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        parser.exit(
            2, f"Install the MCP extra with pip install 'embodichain[mcp]': {exc}\n"
        )

    from dexsim.runtime import Session
    from dexsim.runtime.errors import DexSimRuntimeError

    with Session.connect(
        args.session,
        directory=args.session_directory,
        client={"kind": "embodichain-mcp"},
    ) as session:
        server = MCPServer("EmbodiChain")

        @server.tool(
            annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False)
        )
        def embodichain_get_robot_state(
            object_ref: str,
            robot_cfg: str,
            control_part: str,
            variant: str | None = None,
        ) -> RobotState:
            """Read measured qpos for a scene articulation in configured control-part order.

            Supply the scene object path, a registered robot config ID, and a control
            part such as arm. For the UR5 use robot_cfg=URRobot and variant=ur5.
            """
            try:
                return session.invoke(
                    "embodichain.lab.sim.dexsim_session",
                    object_ref=object_ref,
                    robot_cfg=robot_cfg,
                    variant=variant,
                    control_part=control_part,
                )
            except DexSimRuntimeError as exc:
                raise ToolError(f"{exc.code}: {exc}") from exc

        server.run(transport="stdio")
