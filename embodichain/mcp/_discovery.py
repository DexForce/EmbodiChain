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

"""Read-only MCP discovery backed by the installed EmbodiChain catalogs."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal, get_args

from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import ToolAnnotations
from typing_extensions import TypedDict

from embodichain import __version__
from embodichain.cli._task_catalog import _discover_catalog, _select_task
from embodichain.robot_registry import _robot_specs, list_robot_cfgs

if TYPE_CHECKING:
    from dexsim.runtime import Session
    from mcp.server import MCPServer

    from embodichain.cli._task_catalog import _Task

__all__: list[str] = []

_PhysicsBackend = Literal["default", "newton"]
_TASK_CATALOG_URI = "embodichain://tasks/catalog"
_ROBOT_CONFIG_URI = "embodichain://robots/{robot_id}/config"


class _Result(TypedDict):
    schema_version: Literal[1]


class _ServerInfo(_Result):
    name: str
    version: str
    transport: Literal["stdio"]
    session_id: str | None


class _Health(_Result):
    status: Literal["ready", "unreachable"]
    session_id: str | None
    host_state: str | None


class _Capabilities(_Result):
    tools: list[str]
    resources: list[str]
    resource_templates: list[str]
    prompts: list[str]


class _RobotCatalog(_Result):
    robot_ids: list[str]


class _RobotInfo(_Result):
    robot_id: str
    variant: str | None
    default_variant: str | None
    variants: list[str]
    vendor: str | None
    morphology: str | None
    dof: int | None
    control_parts: dict[str, int] | None


class _TaskSummary(TypedDict):
    task_id: str
    title: str


class _TaskCatalog(_Result):
    tasks: list[_TaskSummary]


class _TaskDeployment(TypedDict):
    name: str
    env_id: str
    physics: str | None
    embodiments: list[str]
    capabilities: list[str]


class _TaskInfo(_Result, _TaskSummary):
    summary: str
    default_deployment: str | None
    deployments: list[_TaskDeployment]


class _Backends(_Result):
    backends: list[_PhysicsBackend]


class _SessionManifest(_Result):
    session_id: str
    backend: str | None
    solver: str | None
    device: str | None
    timestep_s: float | None
    dexsim_version: str | None
    scene_attached: bool
    builder_finalized: bool
    shutdown_requested: bool


def _register_discovery(server: MCPServer, session: Session | None) -> None:
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
    session_id = session.session_id if session is not None else None

    @cache
    def tasks() -> list[_Task]:
        return _discover_catalog(include_runtime=False)

    @server.tool(annotations=read_only)
    def embodichain_server_info() -> _ServerInfo:
        """Report the EmbodiChain version, transport and selected Session ID."""
        return {
            "schema_version": 1,
            "name": server.name,
            "version": __version__,
            "transport": "stdio",
            "session_id": session_id,
        }

    @server.tool(annotations=read_only)
    def embodichain_health() -> _Health:
        """Probe the selected host, or report offline catalog service readiness."""
        state = None if session is None else session.heartbeat()["state"]
        return {
            "schema_version": 1,
            "status": "ready" if state in (None, "idle", "busy") else "unreachable",
            "session_id": session_id,
            "host_state": state,
        }

    @server.tool(annotations=read_only)
    async def embodichain_list_service_capabilities() -> _Capabilities:
        """Discover the tools, resources and prompts installed for this connection."""
        return {
            "schema_version": 1,
            "tools": sorted(tool.name for tool in await server.list_tools()),
            "resources": sorted(
                str(resource.uri) for resource in await server.list_resources()
            ),
            "resource_templates": sorted(
                template.uri_template
                for template in await server.list_resource_templates()
            ),
            "prompts": sorted(prompt.name for prompt in await server.list_prompts()),
        }

    @server.tool(annotations=read_only)
    def embodichain_list_robots() -> _RobotCatalog:
        """List installed robot configuration IDs without instantiating a robot."""
        return {"schema_version": 1, "robot_ids": list(list_robot_cfgs())}

    @server.tool(annotations=read_only)
    def embodichain_get_robot_info(
        robot_id: str, variant: str | None = None
    ) -> _RobotInfo:
        """Read declared robot metadata and control-part counts for a preset."""
        spec = _robot_specs().get(robot_id)
        if spec is None:
            raise ToolError(f"Unknown robot ID {robot_id!r}")
        variants = spec.get("variants") or {}
        default_variant = spec.get("default_variant")
        selected = default_variant if variant is None else variant
        if selected is not None and selected not in variants:
            raise ToolError(f"Unknown variant {selected!r} for robot {robot_id!r}")
        profile = spec if selected is None else variants[selected]
        return {
            "schema_version": 1,
            "robot_id": robot_id,
            "variant": selected,
            "default_variant": default_variant,
            "variants": sorted(variants),
            "vendor": profile.get("vendor", spec.get("vendor")),
            "morphology": profile.get("morphology", spec.get("morphology")),
            "dof": profile.get("dof"),
            "control_parts": profile.get("control_parts"),
        }

    @server.tool(annotations=read_only)
    def embodichain_list_tasks() -> _TaskCatalog:
        """List packaged task definitions using their qualified IDs."""
        return {
            "schema_version": 1,
            "tasks": [
                {"task_id": task.qualified_key, "title": task.title}
                for task in sorted(tasks(), key=lambda item: item.qualified_key)
            ],
        }

    @server.tool(annotations=read_only)
    def embodichain_get_task_info(task_id: str) -> _TaskInfo:
        """Read packaged task metadata, deployments and declared capabilities."""
        try:
            task = _select_task(tasks(), task_id)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "schema_version": 1,
            "task_id": task.qualified_key,
            "title": task.title,
            "summary": task.summary,
            "default_deployment": task.default_deployment,
            "deployments": [
                {
                    "name": deployment.name,
                    "env_id": deployment.env_id,
                    "physics": deployment.physics,
                    "embodiments": list(deployment.embodiments),
                    "capabilities": sorted(deployment.capabilities),
                }
                for deployment in task.deployments
            ],
        }

    @server.tool(annotations=read_only)
    def embodichain_list_physics_backends() -> _Backends:
        """List EmbodiChain's supported physics configuration names."""
        return {"schema_version": 1, "backends": list(get_args(_PhysicsBackend))}

    @server.resource(_TASK_CATALOG_URI, mime_type="application/json")
    def task_catalog() -> _TaskCatalog:
        """Read the same packaged task inventory as the catalog tool."""
        return embodichain_list_tasks()

    @server.resource(_ROBOT_CONFIG_URI, mime_type="application/json")
    def robot_config(robot_id: str) -> _RobotInfo:
        """Read a registered robot's default profile metadata."""
        try:
            return embodichain_get_robot_info(robot_id)
        except ToolError as exc:
            raise ResourceError(str(exc)) from exc

    if session is not None:
        from dexsim.runtime.errors import DexSimRuntimeError

        @server.resource(
            f"embodichain://worlds/{session.session_id}/manifest",
            mime_type="application/json",
        )
        def session_manifest() -> _SessionManifest:
            """Read runtime identity and lifecycle state for the bound World."""
            try:
                state = session.status()
            except DexSimRuntimeError as exc:
                raise ResourceError(f"Session status unavailable ({exc.code})") from exc
            info = session.info
            config = info["config"]
            return {
                "schema_version": 1,
                "session_id": session.session_id,
                "backend": config.get("backend"),
                "solver": config.get("solver"),
                "device": config.get("device"),
                "timestep_s": config.get("dt"),
                "dexsim_version": info.get("runtime", {}).get("dexsim_version"),
                "scene_attached": state["scene_attached"],
                "builder_finalized": state["builder_finalized"],
                "shutdown_requested": state["shutdown_requested"],
            }

    @server.prompt()
    def embodichain_inspect_setup() -> str:
        """Discover a robot, task deployment and available Session observations."""
        return (
            "Call embodichain_list_service_capabilities to discover the available interfaces. "
            "Use the robot and task catalogs to select registered IDs and check the "
            "deployment's physics configuration. If a World manifest is listed, "
            "read it before querying measured robot state."
        )
