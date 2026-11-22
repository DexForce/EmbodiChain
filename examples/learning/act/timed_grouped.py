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

"""Start the experiment wall clock before Xvfb, simulator and training imports."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

__all__ = ["main"]


def main() -> None:
    """Start one X server and optionally launch a single-node DDP job."""
    origin = time.monotonic()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nproc-per-node", type=int, choices=(1, 2), default=1)
    args, rest = parser.parse_known_args()
    if any(
        arg == "--clock-origin" or arg.startswith("--clock-origin=") for arg in rest
    ):
        parser.error("The cold launcher supplies the clock origin")
    env = os.environ.copy()
    # Resolve example imports from this installation while retaining the caller's cwd.
    env["PYTHONSAFEPATH"] = "1"
    examples_parent = Path(__file__).resolve().parents[3]
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{examples_parent}{os.pathsep}{pythonpath}"
        if pythonpath
        else str(examples_parent)
    )
    for key, value in (
        ("EMBODICHAIN_SIM_EXIT_PROCESS", "0"),
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        ("OMP_NUM_THREADS", "2"),
        ("MKL_NUM_THREADS", "2"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("GLOO_SOCKET_IFNAME", "lo"),
    ):
        env.setdefault(key, value)
    if args.nproc_per_node == 2:
        runner = [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--local-addr=127.0.0.1",
            "--nproc-per-node=2",
            "--max-restarts=0",
            "--module",
        ]
    else:
        runner = ["-m"]
    command = [
        "xvfb-run",
        "-a",
        sys.executable,
        "-Bu",
        *runner,
        "examples.learning.act.grouped",
        *rest,
        "--clock-origin",
        str(origin),
    ]
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
