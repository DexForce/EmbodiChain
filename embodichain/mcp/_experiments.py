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

"""Bounded local GenSim experiments exposed before an execution World exists."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import asyncio

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from embodichain.gen_sim.agent_experiment import ExperimentStore

__all__: list[str] = []


def serve(root: Path, agent: str | None = None) -> None:
    store = ExperimentStore(root)
    server = MCPServer("GenSim experimental authoring")

    def scoped(run_id: str) -> str:
        if agent and run_id != store.run_id_for(agent, "initial"):
            raise ValueError("This server is bound to a different experimental run")
        return run_id

    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False)

    @server.tool()
    def gensim_create_run(agent_session: str, command_id: str) -> dict[str, Any]:
        """Freeze a DexUni UR5 placement goal, budget and initial candidate."""
        if agent and (agent_session != agent or command_id != "initial"):
            raise ValueError(
                "This server admits only its operator-selected agent and initial command"
            )
        return store.create(agent_session, command_id)

    @server.tool(annotations=read)
    async def gensim_inspect_run(
        run_id: str, wait_seconds: float = 0
    ) -> dict[str, Any]:
        """Inspect attempts, measured feedback and immutable revisions. Wait up to 20 seconds."""
        if not 0 <= wait_seconds <= 20:
            raise ValueError("Wait must be between 0 and 20 seconds")
        deadline = asyncio.get_running_loop().time() + wait_seconds
        while True:
            r = store.inspect(scoped(run_id))
            if (
                not r["attempts"]
                or r["attempts"][-1]["state"] not in ("queued", "running")
                or asyncio.get_running_loop().time() >= deadline
            ):
                return r
            await asyncio.sleep(0.5)

    @server.tool()
    def gensim_revise_program(
        run_id: str, expected_revision: str, offset_x_m: float, command_id: str
    ) -> dict[str, Any]:
        """Revise only the candidate release target's horizontal offset from the goal."""
        return store.revise(scoped(run_id), expected_revision, offset_x_m, command_id)

    @server.tool(annotations=read)
    def gensim_validate_program(run_id: str, expected_revision: str) -> dict[str, Any]:
        """Validate sealed inputs and the authoritative deployment schema."""
        return store.validate(scoped(run_id), expected_revision)

    @server.tool()
    def gensim_execute_attempt(
        run_id: str, expected_revision: str, command_id: str
    ) -> dict[str, Any]:
        """Submit a separate Gym worker; retries return the original attempt."""
        return store.submit(scoped(run_id), expected_revision, command_id)

    @server.tool()
    def gensim_cancel_attempt(run_id: str, job_id: str) -> dict[str, Any]:
        """Request a stop; inspect the job for actual completion and measured progress."""
        return store.cancel(scoped(run_id), job_id)

    server.run(transport="stdio")
