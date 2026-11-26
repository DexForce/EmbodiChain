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

from pathlib import Path
from dexsim.runtime import Session
from mcp.server import MCPServer

from .tools.robot import register_robot_tools

__all__ = ["create_server", "serve"]


def create_server(
    session: Session,
    *,
    writer_key_file: Path | None = None,
    handoff_key_file: Path | None = None,
    motion_output: Path | None = None,
) -> MCPServer:
    """Register the available domain tools using a caller-owned session."""
    server = MCPServer("EmbodiChain")
    register_robot_tools(server, session)
    if handoff_key_file is not None:
        from .tools.handoff import register_handoff_tools

        register_handoff_tools(server, session, handoff_key_file)
    if handoff_key_file is not None and motion_output is not None:
        from ._motion import register_motion_tools

        register_motion_tools(
            server,
            session.session_id,
            session,
            handoff_key_file.read_bytes(),
            motion_output.resolve(),
            include_robot=False,
            include_policy=True,
        )
    elif writer_key_file is not None or handoff_key_file is not None:
        from embodichain.lab.sim.dexsim_session.writer import WriterClient
        from .tools.control import register_control_tools

        writer = WriterClient(session, writer_key_file or handoff_key_file)
        if writer_key_file is not None:
            writer.call("status", {})
        register_control_tools(server, writer)
    return server


def serve(
    session_id: str,
    *,
    directory: str | None = None,
    writer_key_file: Path | None = None,
    handoff_key_file: Path | None = None,
    motion_output: Path | None = None,
    run_id: str | None = None,
    turn: int | None = None,
) -> None:
    """Connect to an existing session and release the connection when stdio closes."""
    labels = {"kind": "embodichain-mcp"}
    if run_id is not None or turn is not None:
        if not run_id or turn is None or turn < 1:
            raise ValueError("Supply both the Run ID and a positive turn number")
        labels.update(run_id=run_id, turn=turn)
    with Session.connect(session_id, directory=directory, client=labels) as session:
        create_server(
            session,
            writer_key_file=writer_key_file,
            handoff_key_file=handoff_key_file,
            motion_output=motion_output,
        ).run(transport="stdio")
