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

"""Run the UR5 experiment with six explicitly authorized tools in a read-only Codex session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

TOOLS = (
    "gensim_create_run",
    "gensim_inspect_run",
    "gensim_revise_program",
    "gensim_validate_program",
    "gensim_execute_attempt",
    "gensim_cancel_attempt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--approve-experiment-tools",
        action="store_true",
        help="Approve exactly the six local experiment tools for this invocation",
    )
    args = parser.parse_args()
    if not args.approve_experiment_tools:
        parser.error("Explicit authorization for the six experiment tools is required")
    folder = args.output.resolve()
    folder.mkdir(exist_ok=False, parents=True, mode=0o700)
    agent = "codex-" + uuid4().hex
    prefix = "mcp_servers.gensim."
    flags = []
    config = {
        prefix + "command": "/usr/bin/xvfb-run",
        prefix
        + "args": [
            "-a",
            sys.executable,
            "-m",
            "embodichain",
            "mcp",
            "--experiment-output",
            str(folder / "runs"),
            "--experiment-agent",
            agent,
        ],
        prefix + "enabled_tools": list(TOOLS),
        prefix + "startup_timeout_sec": 90,
        prefix + "tool_timeout_sec": 90,
    }
    for name in TOOLS:
        config[prefix + "tools." + name + ".approval_mode"] = "approve"
    for key, value in config.items():
        flags.extend(["-c", key + "=" + json.dumps(value)])
    prompt = f"Use only GenSim MCP. Create a run with agent_session={agent} and command_id=initial. Inspect and validate the prepared DexUni UR5 program, execute it and inspect measured results, using wait_seconds=20 until terminal. Correct the existing candidate offset_x_m using the observed physical feedback. Keep goal, evaluator and budget fixed. Validate the new revision and run the second attempt. Report both revisions, verdicts, jobs, trajectories, costs and timing. Do not use the shell or other tools."
    with (folder / "events.jsonl").open("w") as events:
        subprocess.run(
            [
                "codex",
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "-s",
                "read-only",
                *flags,
                "--json",
                "-o",
                str(folder / "summary.md"),
                prompt,
            ],
            cwd=folder,
            stdout=events,
            check=True,
        )


if __name__ == "__main__":
    main()
