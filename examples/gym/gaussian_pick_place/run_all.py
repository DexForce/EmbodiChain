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

"""Run all three physical tasks in isolated processes, then audit/export them."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

__all__ = ["main"]


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gaussian_pick_place")
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spp", type=int, default=8)
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    root = args.output.resolve() / f"franka-gs-{stamp}-{uuid4().hex[:8]}"
    root.mkdir(parents=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONDONTWRITEBYTECODE="1",
        SDL_VIDEODRIVER="dummy",
        EMBODICHAIN_SIM_EXIT_PROCESS="0",
    )
    attempts, runs = [], []
    for scene in ["discoverse", "office", "robosimgs"]:
        logfile = root / f"{scene}.log"
        print(f"Recording {scene}; log: {logfile}", flush=True)
        command = [
            sys.executable,
            "-B",
            "-u",
            str(here / "run.py"),
            "--scene",
            scene,
            "--output",
            str(root / "runs"),
            "--seed",
            str(args.seed),
            "--spp",
            str(args.spp),
        ]
        with logfile.open("w") as stream:
            result = subprocess.run(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        with logfile.open() as stream:
            first = next(
                (line.strip() for line in stream if line.startswith("RUN ")), ""
            )
        directory = (
            Path(first.removeprefix("RUN ")) if first.startswith("RUN ") else None
        )
        attempts.append(
            {
                "scene": scene,
                "returncode": result.returncode,
                "run": str(directory),
                "log": logfile.name,
            }
        )
        (root / "attempts.json").write_text(json.dumps(attempts, indent=2))
        if result.returncode == 0 and directory is not None:
            runs.append(directory)
    if len(runs) != 3:
        raise RuntimeError(
            f"Some scene recordings failed; inspect {root / 'attempts.json'}"
        )
    # Export uses recorded pixels without a native World or re-rendering.
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(here / "_bundle.py"),
            "--runs",
            *map(str, runs),
            "--output",
            str(root / "delivery"),
        ],
        env=env,
        check=True,
    )
    print(f"Complete task videos: {root / 'delivery/complete_tasks.mp4'}", flush=True)


if __name__ == "__main__":
    main()
