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

"""Run the Codex trajectory workflow through an MCP stdio client."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters


def _root() -> Path:
    """Return the repository root containing the EmbodiChain package."""
    return Path(__file__).resolve().parents[3]


def _structured(result: Any) -> dict[str, Any]:
    """Return structured MCP tool output or raise with the tool error."""
    if result.is_error:
        raise RuntimeError(str(result.content))
    value = result.structured_content
    if not isinstance(value, dict):
        raise TypeError(f"Expected structured mapping, got {type(value).__name__}")
    return value


async def run(*, execute: bool) -> None:
    """Run the demo with an MCP client and the local stdio server."""
    root = _root()
    demo_dir = root / "examples/mcp/codex_trajectory_demo"
    scene = json.loads((demo_dir / "scene.json").read_text(encoding="utf-8"))
    trajectory_cfg = json.loads(
        (demo_dir / "trajectory.json").read_text(encoding="utf-8")
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "embodichain", "mcp"],
        cwd=root,
    )

    async with Client(params) as client:
        resource = await client.read_resource(
            "embodichain://demos/codex-trajectory/scene"
        )
        resource_scene = json.loads(resource.contents[0].text)
        if resource_scene != scene:
            raise RuntimeError("Built-in demo resource differs from scene.json")

        world = _structured(
            await client.call_tool("create_world", {"backend": "default", "seed": 7})
        )
        world_id = world["world_id"]
        try:
            await client.call_tool(
                "load_task_or_scene",
                {"world_id": world_id, "scene": resource_scene},
            )
            state = _structured(
                await client.call_tool("get_world_state", {"world_id": world_id})
            )
            robot_id = trajectory_cfg["robot_id"]
            current_qpos = state["result"]["robots"][robot_id]["qpos"][0]
            waypoints = trajectory_cfg["waypoints"]
            if waypoints[0] != current_qpos:
                waypoints = [current_qpos, *waypoints[1:]]

            trajectory = _structured(
                await client.call_tool(
                    "generate_robot_trajectory",
                    {
                        "world_id": world_id,
                        "robot_id": robot_id,
                        "waypoints": waypoints,
                        "samples_per_segment": trajectory_cfg["samples_per_segment"],
                    },
                )
            )
            record = trajectory["result"]
            print(json.dumps(record, indent=2, ensure_ascii=False))
            if execute:
                execution = _structured(
                    await client.call_tool(
                        "execute_trajectory",
                        {"trajectory_id": record["trajectory_id"]},
                    )
                )
                print(json.dumps(execution, indent=2, ensure_ascii=False))
        finally:
            await client.call_tool("destroy_world", {"world_id": world_id})


def main() -> None:
    """Parse command-line options and run the demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated trajectory after validation.",
    )
    args = parser.parse_args()
    asyncio.run(run(execute=args.execute))


if __name__ == "__main__":
    main()
