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

"""Simulation-domain MCP registrations for the shared project server."""

from __future__ import annotations

from typing import Any

from .adapters import ToolRegistrar
from .service import EmbodiChainMCPService

__all__ = ["SimulationMCPAdapter"]


class SimulationMCPAdapter:
    """Register the existing simulation service as one MCP domain adapter."""

    name = "simulation"

    def __init__(self, service: EmbodiChainMCPService) -> None:
        self.service = service

    def close(self) -> None:
        """Leave service lifetime ownership with the caller."""

    def capabilities(self) -> list[str]:
        """Return the simulation service operation names."""
        return self.service.capabilities()

    def register(self, server: Any, *, register_tool: ToolRegistrar) -> None:
        """Register simulation tools, resources, and prompts."""
        read_only_tools = {
            "list_robots",
            "get_robot_info",
            "list_tasks",
            "get_task_info",
            "list_physics_backends",
            "list_cameras",
            "get_world_state",
            "forward_kinematics",
            "solve_ik",
            "check_reachability",
            "check_collision",
            "validate_trajectory",
            "plan_motion",
            "get_run_status",
            "get_run_metrics",
            "compare_runs",
        }
        destructive_tools = {
            "create_world",
            "load_task_or_scene",
            "reset_world",
            "destroy_world",
            "restore_world",
            "step_simulation",
            "execute_trajectory",
            "record_trajectory",
            "start_rollout",
            "cancel_run",
        }
        envelope_tools = {
            "destroy_world",
            "start_rollout",
            "get_run_status",
            "get_run_metrics",
            "cancel_run",
            "compare_runs",
        }
        for name in (
            "list_robots",
            "get_robot_info",
            "list_tasks",
            "get_task_info",
            "list_physics_backends",
            "list_cameras",
            "create_world",
            "load_task_or_scene",
            "reset_world",
            "destroy_world",
            "get_world_state",
            "snapshot_world",
            "restore_world",
            "step_simulation",
            "forward_kinematics",
            "solve_ik",
            "check_reachability",
            "check_collision",
            "plan_motion",
            "generate_robot_trajectory",
            "execute_trajectory",
            "record_trajectory",
            "validate_trajectory",
            "start_rollout",
            "get_run_status",
            "get_run_metrics",
            "cancel_run",
            "compare_runs",
        ):
            register_tool(
                name,
                getattr(self.service, name),
                read_only=name in read_only_tools,
                destructive=name in destructive_tools,
                idempotent=name in read_only_tools
                or name in {"reset_world", "cancel_run"},
                envelope=name in envelope_tools,
            )

        @server.resource("embodichain://tasks/catalog", mime_type="application/json")
        def task_catalog() -> str:
            return self.service.read_resource("embodichain://tasks/catalog")

        @server.resource(
            "embodichain://demos/codex-trajectory/scene", mime_type="application/json"
        )
        def codex_trajectory_scene() -> str:
            return self.service.read_resource(
                "embodichain://demos/codex-trajectory/scene"
            )

        @server.resource(
            "embodichain://robots/{robot_id}/config", mime_type="application/json"
        )
        def robot_config(robot_id: str) -> str:
            return self.service.read_resource(f"embodichain://robots/{robot_id}/config")

        @server.resource(
            "embodichain://worlds/{world_id}/manifest", mime_type="application/json"
        )
        def world_manifest(world_id: str) -> str:
            return self.service.read_resource(
                f"embodichain://worlds/{world_id}/manifest"
            )

        @server.resource(
            "embodichain://runs/{run_id}/status", mime_type="application/json"
        )
        def run_status(run_id: str) -> str:
            return self.service.read_resource(f"embodichain://runs/{run_id}/status")

        @server.resource(
            "embodichain://runs/{run_id}/metrics", mime_type="application/json"
        )
        def run_metrics(run_id: str) -> str:
            return self.service.read_resource(f"embodichain://runs/{run_id}/metrics")

        @server.resource(
            "embodichain://trajectories/{trajectory_id}", mime_type="application/json"
        )
        def trajectory(trajectory_id: str) -> str:
            return self.service.read_resource(
                f"embodichain://trajectories/{trajectory_id}"
            )

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

        @server.prompt(name="generate_robot_trajectory")
        def generate_robot_trajectory(world_id: str, robot_id: str) -> str:
            return (
                f"Generate a joint trajectory for robot {robot_id} in world {world_id}. "
                "Read the world state, choose safe waypoints, call "
                "generate_robot_trajectory, inspect its validation report, and "
                "return the trajectory resource URI. Ask before calling "
                "execute_trajectory if the user did not explicitly request execution."
            )
