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

"""MCP protocol adapter and command-line entry point.

The server owns protocol registration and delegates domain behavior to
injected adapters. Simulation and URDF assembly are the first registered
providers; future Task Program, RL, data, visualization, and device adapters
can register beside them.
The phase-one command intentionally serves stdio only. Streamable HTTP remains
behind an authenticated gateway boundary so this process cannot expose
stateful simulation and artifact-writing tools without authorization.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
from functools import wraps
from typing import Any

from .adapters import MCPAdapter, MCPAdapterRegistry
from .backend import SimulationManagerBackend
from .service import EmbodiChainMCPService
from .simulation import SimulationMCPAdapter
from .urdf import URDFAssemblyAdapter

__all__ = ["cli", "create_server", "serve"]


def _default_adapter_registry(service: EmbodiChainMCPService) -> MCPAdapterRegistry:
    """Create the default project adapters with optional sim verification."""
    verifier = None
    if isinstance(service.backend, SimulationManagerBackend):
        from .urdf_simulation import verify_urdf_in_simulation

        def verifier(path: Any, backend: str, seed: int | None) -> dict[str, Any]:
            return verify_urdf_in_simulation(
                path,
                backend,
                seed,
                simulation=service.backend,
            )

    return MCPAdapterRegistry(
        [
            SimulationMCPAdapter(service),
            URDFAssemblyAdapter(simulation_verifier=verifier),
        ]
    )


def create_server(
    service: EmbodiChainMCPService | None = None,
    *,
    adapters: MCPAdapterRegistry | Sequence[MCPAdapter] | None = None,
) -> Any:
    """Create an MCP server bound to an EmbodiChain service.

    Args:
        service: Optional service instance. A SimulationManager-backed service
            is created when omitted; inject ``InMemorySimulationBackend`` for
            protocol-only smoke tests.
        adapters: Optional project-domain adapters.  The default registers the
            simulation facade together with the pure URDF assembly adapter.

    Returns:
        An ``mcp.server.MCPServer`` instance.

    Raises:
        ModuleNotFoundError: If the optional ``mcp`` dependency is absent.
    """
    try:
        from mcp.server import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MCP support requires the optional dependency: "
            "pip install 'embodichain[mcp]'"
        ) from error

    class _StdioOnlyMCPServer(MCPServer):
        """Prevent public server objects from bypassing the transport policy."""

        def run(self, transport: str = "stdio", **kwargs: Any) -> None:
            if transport != "stdio":
                raise ValueError(
                    "streamable HTTP and SSE are deferred until an authenticated "
                    "MCP gateway is available; use stdio"
                )
            super().run(transport=transport, **kwargs)

    service = service or EmbodiChainMCPService(backend=SimulationManagerBackend())
    if adapters is None:
        adapter_registry = _default_adapter_registry(service)
    elif isinstance(adapters, MCPAdapterRegistry):
        adapter_registry = adapters
    else:
        adapter_registry = MCPAdapterRegistry(adapters)
    server = _StdioOnlyMCPServer(
        "embodichain",
        version="0.1.0",
        description="EmbodiChain project-level domain adapters over MCP.",
        instructions=(
            "Use explicit handles and revisions. URDF composition is asset-only; "
            "simulation verification is a separate optional operation. Results "
            "describe their provider scope and do not grant real-robot control."
        ),
    )

    def register_tool(
        name: str,
        function: Any,
        *,
        read_only: bool | None = None,
        destructive: bool | None = None,
        idempotent: bool | None = None,
        envelope: bool = False,
    ) -> None:
        def normalize_result(value: Any) -> Any:
            """Give stateful status tools the same metadata envelope."""
            if not envelope or not isinstance(value, dict):
                return value
            world_id = value.get("world_id")
            world_state = None
            if isinstance(world_id, str):
                try:
                    world_state = service.backend.get_world_state(world_id)
                except ValueError:
                    world_state = None
            artifacts = []
            for key, media_type in (
                ("artifact_path", "video/mp4"),
                ("depth_artifact_path", "application/x-npz"),
            ):
                path = value.get(key)
                if isinstance(path, str):
                    artifacts.append({"path": path, "media_type": media_type})
            return {
                "status": value.get("status", "succeeded"),
                "result": deepcopy(value),
                "backend": getattr(
                    service.backend, "name", type(service.backend).__name__
                ),
                "frame": "arena",
                "units": {"length": "m", "angle": "rad", "time": "s"},
                "diagnostics": [],
                "artifacts": artifacts,
                "world_id": world_id,
                "scene_revision": (
                    None if world_state is None else world_state.get("scene_revision")
                ),
                "seed": None if world_state is None else world_state.get("seed"),
            }

        @wraps(function)
        def guarded_tool(*args: Any, **kwargs: Any) -> Any:
            try:
                return normalize_result(function(*args, **kwargs))
            except (
                KeyError,
                ModuleNotFoundError,
                NotImplementedError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                raise ToolError(str(error)) from error

        read_only_hint = bool(read_only)
        destructive_hint = bool(destructive)
        idempotent_hint = read_only_hint if idempotent is None else bool(idempotent)
        server.add_tool(
            guarded_tool,
            name=name,
            structured_output=True,
            annotations=ToolAnnotations(
                readOnlyHint=read_only_hint,
                destructiveHint=destructive_hint,
                idempotentHint=idempotent_hint,
                openWorldHint=False,
            ),
        )

    def server_info() -> dict[str, Any]:
        """Return service metadata together with installed domain adapters."""
        result = dict(service.server_info())
        result["scope"] = "project_capabilities"
        result["adapters"] = list(adapter_registry.names)
        return result

    def health() -> dict[str, Any]:
        """Return service health together with installed domain adapters."""
        result = dict(service.health())
        result["scope"] = "project_capabilities"
        result["adapters"] = list(adapter_registry.names)
        return result

    def list_capabilities() -> list[str]:
        """Return core and adapter-provided MCP operation names."""
        result: list[str] = []
        seen: set[str] = set()
        for capability in [*service.capabilities(), *adapter_registry.capabilities()]:
            if capability not in seen:
                result.append(capability)
                seen.add(capability)
        return result

    register_tool("server_info", server_info, read_only=True, idempotent=True)
    register_tool("health", health, read_only=True, idempotent=True)
    register_tool(
        "list_capabilities", list_capabilities, read_only=True, idempotent=True
    )
    # Core health/introspection tools are registered here; all domain tools,
    # resources, and prompts are supplied by adapter registrations below.

    adapter_registry.register(server, register_tool=register_tool)

    return server


def serve(
    transport: str = "stdio",
    *,
    service: EmbodiChainMCPService | None = None,
    adapters: MCPAdapterRegistry | Sequence[MCPAdapter] | None = None,
) -> None:
    """Run the MCP server using stdio or Streamable HTTP.

    Args:
        transport: ``stdio``. Streamable HTTP is intentionally deferred until
            an authenticated gateway is available.
        service: Optional service instance for embedding and testing.
        adapters: Optional project-domain adapters.  A default URDF adapter is
            created when omitted.
    """
    if transport != "stdio":
        raise ValueError(
            "streamable-http is deferred until an authenticated MCP gateway "
            "is available; use stdio"
        )
    owned_service = service is None
    service = service or EmbodiChainMCPService(backend=SimulationManagerBackend())
    owned_adapters = not isinstance(adapters, MCPAdapterRegistry)
    if adapters is None:
        adapter_registry = _default_adapter_registry(service)
    elif isinstance(adapters, MCPAdapterRegistry):
        adapter_registry = adapters
    else:
        adapter_registry = MCPAdapterRegistry(adapters)
    try:
        server = create_server(service, adapters=adapter_registry)
        server.run(transport=transport)
    finally:
        if owned_service:
            service.close()
        if owned_adapters:
            adapter_registry.close()


def cli(argv: Sequence[str] | None = None) -> None:
    """Run ``embodichain mcp`` from the unified command line."""
    parser = argparse.ArgumentParser(
        prog="embodichain mcp",
        description="Expose EmbodiChain phase-one capabilities over MCP.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio",),
        default="stdio",
        help="MCP transport to use (default: stdio).",
    )
    args = parser.parse_args(argv)
    serve(args.transport)
