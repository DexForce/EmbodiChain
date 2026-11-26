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

"""Explicit Scene handoff lifecycle on a pre-authorized manual Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dexsim.runtime import Session
from dexsim.runtime.errors import DexSimRuntimeError
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

__all__: list[str] = []


def register_handoff_tools(server: MCPServer, session: Session, key_file: Path) -> None:
    from dexsim.runtime.handoff import MODULE, signature

    key = key_file.read_bytes()

    def invoke(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return session.invoke(
                MODULE,
                arguments=args,
                signature=signature(key, session.session_id, args),
            )
        except DexSimRuntimeError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    def embodichain_inspect_handoff() -> dict[str, Any]:
        """Read the host phase, Scene topology and current handoff generation."""
        return invoke({"operation": "describe"})

    @server.tool()
    def embodichain_bind_world(
        generation: str, topology_revision: int, binding: dict[str, Any]
    ) -> dict[str, Any]:
        """Bind existing scene roles to configured robots and bodies without physics updates."""
        return invoke(
            dict(
                operation="acquire",
                generation=generation,
                topology_revision=topology_revision,
                binding=binding,
            )
        )

    @server.tool()
    def embodichain_release_world(generation: str) -> dict[str, Any]:
        """Release control and return the unchanged host Session to scene authoring."""
        return invoke(dict(operation="release", generation=generation))
