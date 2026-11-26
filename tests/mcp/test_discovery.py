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

"""Verify catalog discovery through the stdio command."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest


def test_stdio_catalog_without_dexsim() -> None:
    pytest.importorskip("mcp.server.mcpserver")
    import mcp

    task_id = "embodichain.extensions.tasks:push_cube"
    startup = (
        "import runpy, sys; sys.modules['dexsim'] = None; "
        "runpy.run_module('embodichain', run_name='__main__')"
    )
    parameters = mcp.StdioServerParameters(
        command=sys.executable, args=["-c", startup, "mcp", "--catalog"]
    )

    async def inspect_catalog() -> None:
        async with mcp.Client(parameters) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            capabilities = await client.call_tool(
                "embodichain_list_service_capabilities"
            )
            assert not capabilities.is_error
            task_resource = "embodichain://tasks/catalog"
            robot_template = "embodichain://robots/{robot_id}/config"
            assert capabilities.structured_content == {
                "schema_version": 1,
                "tools": sorted(tools),
                "resources": [task_resource],
                "resource_templates": [robot_template],
                "prompts": ["embodichain_inspect_setup"],
            }
            assert "embodichain_get_robot_state" not in tools
            for tool in tools.values():
                assert tool.annotations.read_only_hint is True
                assert "schema_version" in tool.output_schema["required"]
                assert tool.output_schema["properties"]["schema_version"]["const"] == 1

            server = await client.call_tool("embodichain_server_info")
            assert server.structured_content["name"] == "EmbodiChain"
            assert server.structured_content["transport"] == "stdio"
            assert server.structured_content["session_id"] is None
            health = await client.call_tool("embodichain_health")
            assert health.structured_content == {
                "schema_version": 1,
                "status": "ready",
                "session_id": None,
                "host_state": None,
            }
            robots = await client.call_tool("embodichain_list_robots")
            assert "URRobot" in robots.structured_content["robot_ids"]
            robot = await client.call_tool(
                "embodichain_get_robot_info", {"robot_id": "URRobot", "variant": "ur5"}
            )
            assert robot.structured_content["variant"] == "ur5"
            assert robot.structured_content["control_parts"] == {"arm": 6}
            assert robot.structured_content["dof"] == 6
            unknown = await client.call_tool(
                "embodichain_get_robot_info",
                {"robot_id": "URRobot", "variant": "unknown"},
            )
            assert unknown.is_error

            robot_resource = await client.read_resource(
                "embodichain://robots/URRobot/config"
            )
            default_robot = await client.call_tool(
                "embodichain_get_robot_info", {"robot_id": "URRobot"}
            )
            assert (
                json.loads(robot_resource.contents[0].text)
                == default_robot.structured_content
            )
            assert default_robot.structured_content["variant"] == "ur10"

            catalog = await client.read_resource(task_resource)
            tasks = await client.call_tool("embodichain_list_tasks")
            assert json.loads(catalog.contents[0].text) == tasks.structured_content
            assert task_id in {
                entry["task_id"] for entry in tasks.structured_content["tasks"]
            }
            task = await client.call_tool(
                "embodichain_get_task_info", {"task_id": task_id}
            )
            assert task.structured_content["task_id"] == task_id
            assert task.structured_content["default_deployment"] in {
                deployment["name"]
                for deployment in task.structured_content["deployments"]
            }
            assert any(
                deployment["physics"] == "default"
                for deployment in task.structured_content["deployments"]
            )

            backends = await client.call_tool("embodichain_list_physics_backends")
            assert backends.structured_content == {
                "schema_version": 1,
                "backends": ["default", "newton"],
            }
            prompt = await client.get_prompt("embodichain_inspect_setup")
            assert (
                "embodichain_list_service_capabilities"
                in prompt.messages[0].content.text
            )

    asyncio.run(inspect_catalog())
