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

"""Exercise B on a fresh dedicated host and write the native readback evidence."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from dexsim.runtime import Session

from .fixture import ARM, native
from embodichain.lab.sim.dexsim_session.writer import WriterClient


async def verify(
    directory: Path, interface: str, labels: dict[str, Any]
) -> dict[str, Any]:
    (record,) = directory.glob("dsws_*.json")
    sid = record.stem
    key_file = directory / "writer.key"
    session = Session.connect(
        sid, directory=directory, client={"kind": "acceptance", **labels}
    )

    @asynccontextmanager
    async def connection():
        if interface == "python":

            with Session.connect(
                sid, directory=directory, client={"kind": "python-writer", **labels}
            ) as writer_session:
                client = WriterClient(writer_session, key_file)

                async def direct(operation, arguments):
                    return client.call(operation, arguments)

                yield direct
        else:
            from mcp import Client, StdioServerParameters

            args = [
                "-m",
                "embodichain",
                "mcp",
                "--session",
                sid,
                "--session-directory",
                str(directory),
                "--writer-key-file",
                str(key_file),
            ]
            if labels:
                args.extend(
                    ["--run-id", labels["run_id"], "--turn", str(labels["turn"])]
                )
            async with Client(
                StdioServerParameters(command=sys.executable, args=args)
            ) as mcp:

                async def remote(operation, arguments):
                    result = await mcp.call_tool(
                        "embodichain_control_" + operation, arguments
                    )
                    if result.is_error:
                        raise RuntimeError(str(result.content))
                    return result.structured_content

                yield remote

    def observe(name):
        return native(
            session, "articulation.get_state", object_ref=name, joint_names=ARM
        )

    try:
        a0, b0 = observe("robot_a"), observe("robot_b")
        async with connection() as call:
            st = await call("status", {})
            target = [0.15, -0.15, 0.12, -0.15, 0.12, -0.15]
            request = dict(
                command_id="target-b-once",
                session_id=sid,
                topology_revision=st["topology_revision"],
                revision=st["revision"],
                robot_uid="UR5",
                control_part="arm",
                target=target,
                units=["rad"] * 6,
                control_interval_s=0.04,
            )
            receipt = await call("target_and_step", request)
        # The first client disconnected. Retry with identical intent via a new connection.
        async with connection() as call:
            replay = await call("target_and_step", request)
            recovered = await call("receipt", {"command_id": request["command_id"]})
            stopped = await call("stop", {})
        a1, b1 = observe("robot_a"), observe("robot_b")
        assert receipt == replay == recovered
        assert (
            receipt["state"] == "ready"
            and receipt["target_accepted"]
            and receipt["control_step"] == 1
            and receipt["physics_steps"] == 4
        )
        assert stopped["state"] == "stopped" and stopped["control_step"] == 1
        np.testing.assert_allclose(
            a1["joint_target_position"], a0["joint_target_position"], atol=1e-6
        )
        np.testing.assert_allclose(b1["joint_target_position"], target, atol=1e-6)
        np.testing.assert_allclose(
            receipt["measured_after"]["qpos"], b1["joint_position"], atol=1e-6
        )
        assert not np.allclose(b0["joint_position"], b1["joint_position"], atol=1e-6)
        if b0["step_index"] is not None:
            assert b1["step_index"] - b0["step_index"] == 4
            assert abs(b1["simulation_time"] - b0["simulation_time"] - 0.04) < 1e-6
        return dict(
            interface=interface,
            session_id=sid,
            labels=labels,
            runtime=session.info["runtime"],
            before=dict(a=a0, b=b0),
            after=dict(a=a1, b=b1),
            receipt=receipt,
            replay_matches=True,
            reconciled=True,
            stop=stopped,
            journal=session.info["journal"],
        )
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--interface", choices=["python", "mcp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dexpilot-record", type=Path)
    args = parser.parse_args()
    labels = {}
    if args.dexpilot_record:
        record = json.loads(args.dexpilot_record.read_text())
        (target,) = args.directory.glob("dsws_*.json")
        if target.stem != record["session_id"]:
            raise ValueError("DexPilot and acceptance host differ")
        labels = {k: record["client"][k] for k in ("run_id", "turn")}
    result = asyncio.run(verify(args.directory, args.interface, labels))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "interface",
                    "session_id",
                    "labels",
                    "replay_matches",
                    "reconciled",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
