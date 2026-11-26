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

"""Build the local stdio service and connect to a DexSim World when selected."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.server import MCPServer

if TYPE_CHECKING:
    from dexsim.runtime import Session

__all__ = ["create_server", "serve"]


def create_server(session: Session | None, *, catalog: bool = False) -> MCPServer:
    """Register read-only tools and optional discovery on a caller-owned session.

    Args:
        session: Borrowed DexSim session, or None for a standalone catalog.
        catalog: Include service discovery, catalog tools and resources.

    Returns:
        The configured MCP server.
    """
    if session is None and not catalog:
        raise ValueError("Supply a DexSim Session or enable the catalog")
    server = MCPServer("EmbodiChain")
    if session is not None:
        from .tools.robot import register_robot_tools

        register_robot_tools(server, session)
    if catalog:
        from ._discovery import _register_discovery

        _register_discovery(server, session)
    return server


def serve(
    session_id: str | None, *, directory: str | None = None, catalog: bool = False
) -> None:
    """Run the stdio server and release any borrowed connection on exit.

    Args:
        session_id: Existing DexSim Session ID, or None for offline discovery.
        directory: Optional directory containing the selected session record.
        catalog: Include service discovery, catalog tools and resources.
    """
    if session_id is None:
        create_server(None, catalog=catalog).run(transport="stdio")
        return

    from dexsim.runtime import Session

    with Session.connect(
        session_id, directory=directory, client={"kind": "embodichain-mcp"}
    ) as session:
        create_server(session, catalog=catalog).run(transport="stdio")
