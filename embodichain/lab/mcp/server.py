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

"""MCP protocol adapter and command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from .backend import SimulationManagerBackend
from .service import EmbodiChainMCPService

__all__ = ["cli", "create_server", "serve"]


def create_server(service: EmbodiChainMCPService | None = None) -> Any:
    """Create an MCP server bound to an EmbodiChain service.

    Args:
        service: Optional service instance. A SimulationManager-backed service
            is created when omitted; inject ``InMemorySimulationBackend`` for
            protocol-only smoke tests.

    Returns:
        An ``mcp.server.MCPServer`` instance.

    Raises:
        ModuleNotFoundError: If the optional ``mcp`` dependency is absent.
    """
    try:
        from mcp.server import MCPServer
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "MCP support requires the optional dependency: "
            "pip install 'embodichain[mcp]'"
        ) from error

    service = service or EmbodiChainMCPService(backend=SimulationManagerBackend())
    server = MCPServer(
        "embodichain",
        version="0.1.0",
        description="EmbodiChain phase-one simulation and motion tools.",
        instructions=(
            "Use explicit world_id and scene revisions. Results describe the "
            "backend scope and do not grant real-robot control."
        ),
    )

    read_only_tools = {
        "server_info",
        "list_capabilities",
        "list_robots",
        "get_robot_info",
        "list_tasks",
        "get_task_info",
        "list_physics_backends",
        "get_world_state",
        "forward_kinematics",
        "solve_ik",
        "check_reachability",
        "check_collision",
        "validate_trajectory",
        "get_run_status",
        "get_run_metrics",
        "compare_runs",
    }

    def register_tool(name: str, function: Any) -> None:
        server.add_tool(
            function,
            name=name,
            structured_output=True,
            annotations=ToolAnnotations(
                readOnlyHint=name in read_only_tools,
                destructiveHint=False,
                idempotentHint=name in read_only_tools
                or name in {"reset_world", "cancel_run"},
                openWorldHint=False,
            ),
        )

    register_tool("server_info", service.server_info)
    register_tool("list_capabilities", service.capabilities)
    register_tool("list_robots", service.list_robots)
    register_tool("get_robot_info", service.get_robot_info)
    register_tool("list_tasks", service.list_tasks)
    register_tool("get_task_info", service.get_task_info)
    register_tool("list_physics_backends", service.list_physics_backends)
    register_tool("create_world", service.create_world)
    register_tool("load_task_or_scene", service.load_task_or_scene)
    register_tool("reset_world", service.reset_world)
    register_tool("destroy_world", service.destroy_world)
    register_tool("get_world_state", service.get_world_state)
    register_tool("snapshot_world", service.snapshot_world)
    register_tool("restore_world", service.restore_world)
    register_tool("step_simulation", service.step_simulation)
    register_tool("forward_kinematics", service.forward_kinematics)
    register_tool("solve_ik", service.solve_ik)
    register_tool("check_reachability", service.check_reachability)
    register_tool("check_collision", service.check_collision)
    register_tool("plan_motion", service.plan_motion)
    register_tool("validate_trajectory", service.validate_trajectory)
    register_tool("start_rollout", service.start_rollout)
    register_tool("get_run_status", service.get_run_status)
    register_tool("get_run_metrics", service.get_run_metrics)
    register_tool("cancel_run", service.cancel_run)
    register_tool("compare_runs", service.compare_runs)

    @server.resource("embodichain://tasks/catalog", mime_type="application/json")
    def task_catalog() -> str:
        return service.read_resource("embodichain://tasks/catalog")

    @server.resource(
        "embodichain://robots/{robot_id}/config", mime_type="application/json"
    )
    def robot_config(robot_id: str) -> str:
        return service.read_resource(f"embodichain://robots/{robot_id}/config")

    @server.resource(
        "embodichain://worlds/{world_id}/manifest", mime_type="application/json"
    )
    def world_manifest(world_id: str) -> str:
        return service.read_resource(f"embodichain://worlds/{world_id}/manifest")

    @server.resource("embodichain://runs/{run_id}/status", mime_type="application/json")
    def run_status(run_id: str) -> str:
        return service.read_resource(f"embodichain://runs/{run_id}/status")

    @server.resource(
        "embodichain://runs/{run_id}/metrics", mime_type="application/json"
    )
    def run_metrics(run_id: str) -> str:
        return service.read_resource(f"embodichain://runs/{run_id}/metrics")

    @server.prompt(name="inspect_robot")
    def inspect_robot(robot_id: str) -> str:
        return (
            f"Inspect robot {robot_id}. First read its configuration resource, "
            "then report degrees of freedom, limits, frames, and available "
            "kinematics capabilities."
        )

    @server.prompt(name="validate_motion")
    def validate_motion(world_id: str, robot_id: str) -> str:
        return (
            f"Validate a motion for robot {robot_id} in world {world_id}. "
            "Read the current world state, solve or inspect the target, plan a "
            "trajectory, and call validate_trajectory before reporting success."
        )

    return server


def serve(
    transport: str = "stdio", *, service: EmbodiChainMCPService | None = None
) -> None:
    """Run the MCP server using stdio or Streamable HTTP.

    Args:
        transport: ``stdio`` or ``streamable-http``.
        service: Optional service instance for embedding and testing.
    """
    if transport not in {"stdio", "streamable-http"}:
        raise ValueError("transport must be 'stdio' or 'streamable-http'")
    owned_service = service is None
    service = service or EmbodiChainMCPService(backend=SimulationManagerBackend())
    server = create_server(service)
    try:
        server.run(transport=transport)
    finally:
        if owned_service:
            service.close()


def cli(argv: Sequence[str] | None = None) -> None:
    """Run ``embodichain mcp`` from the unified command line."""
    parser = argparse.ArgumentParser(
        prog="embodichain mcp",
        description="Expose EmbodiChain phase-one capabilities over MCP.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport to use (default: stdio).",
    )
    args = parser.parse_args(argv)
    serve(args.transport)
