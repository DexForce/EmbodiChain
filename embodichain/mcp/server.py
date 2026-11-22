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

"""Build and serve the stdio interface to a borrowed DexSim World."""

from __future__ import annotations

from dexsim.runtime import Session
from mcp.server import MCPServer

from .tools.robot import register_robot_tools

__all__ = ["create_server", "serve"]


def create_server(session: Session) -> MCPServer:
    """Register the available domain tools using a caller-owned session."""
    server = MCPServer("EmbodiChain")
    register_robot_tools(server, session)
    return server


def serve(session_id: str, *, directory: str | None = None) -> None:
    """Connect to an existing session and release the connection when stdio closes."""
    with Session.connect(
        session_id, directory=directory, client={"kind": "embodichain-mcp"}
    ) as session:
        create_server(session).run(transport="stdio")
