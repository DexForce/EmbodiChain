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

"""Compose a UR5 arm and gripper through the EmbodiChain MCP server."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

from mcp import Client, StdioServerParameters

__all__ = ["main", "run"]


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


async def run(*, verify_simulation: bool) -> None:
    """Run the arm-and-end-effector assembly workflow through MCP."""
    root = _root()
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "embodichain", "mcp"],
        cwd=root,
    )
    components = [
        {
            "component_type": "arm",
            "asset": "UniversalRobots/UR5/UR5.urdf",
        },
        {
            "component_type": "hand",
            "asset": "DH_PGC_140_50_M/DH_PGC_140_50_M.urdf",
        },
    ]

    async with Client(params) as client:
        catalog = _structured(await client.call_tool("urdf_list_assets", {}))
        print(json.dumps(catalog, indent=2, ensure_ascii=False))

        assembly = _structured(
            await client.call_tool(
                "urdf_compose",
                {
                    "assembly_name": "ur5_with_gripper",
                    "components": components,
                    "name_case": {"joint": "upper", "link": "lower"},
                },
            )
        )
        assembly_id = str(assembly["assembly_id"])
        print(json.dumps(assembly, indent=2, ensure_ascii=False))

        manifest = await client.read_resource(
            f"embodichain://urdf/assemblies/{assembly_id}/manifest"
        )
        print(manifest.contents[0].text)

        validation = _structured(
            await client.call_tool("urdf_validate", {"assembly_id": assembly_id})
        )
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        if not validation.get("valid"):
            raise RuntimeError(f"Generated URDF is invalid: {validation}")

        if verify_simulation:
            verification = _structured(
                await client.call_tool(
                    "urdf_verify_in_simulation",
                    {"assembly_id": assembly_id, "backend": "default", "seed": 7},
                )
            )
            print(json.dumps(verification, indent=2, ensure_ascii=False))
            if verification.get("status") != "succeeded":
                raise RuntimeError(f"Simulation verification failed: {verification}")


def main() -> None:
    """Parse showcase options and run the MCP client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Stop after pure URDF validation without starting a simulator.",
    )
    args = parser.parse_args()
    asyncio.run(run(verify_simulation=not args.no_verify))


if __name__ == "__main__":
    main()
