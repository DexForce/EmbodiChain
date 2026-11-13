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

"""Run small L-03 asset probes serially and stop at the first failed process.

Run: python -m scripts.benchmark.tech_reports.loading.experiments.l03_preflight_batch
    --manifest SHARED_MANIFEST --output NEW_DIRECTORY
No timing from these functional preflights enters the performance comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

__all__ = ["main"]


def main() -> None:
    """Preserve commands, logs and failures for each independent process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-mode", choices=("joints", "pose", "full"), default="full"
    )
    parser.add_argument("--warmup-joint-state", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--explicit-cleanup", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument(
        "--robots", nargs="+", default=["g1", "cartpole", "franka", "go2"]
    )
    parser.add_argument(
        "--num-envs", nargs="+", type=int, choices=(1, 4), default=[1, 4]
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=("isaaclab", "embodichain"),
        default=["isaaclab", "embodichain"],
    )
    args = parser.parse_args()
    root = Path.cwd()
    profiles = json.loads((root / "configs/environments.json").read_text())["profiles"]
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "batch_source.py")
    records = []
    for robot in args.robots:
        for count in args.num_envs:
            for platform in args.platforms:
                name = f"{robot}_{count:04d}_{platform}"
                output = args.output.resolve() / name
                paths = profiles[platform]["launch_pythonpath"].copy()
                if platform == "embodichain":
                    paths.insert(
                        0, str(root / "outputs/benchmarks/P-03/runtime/p03-frozen-v1")
                    )
                overrides = {
                    "PYTHONPATH": ":".join(paths),
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                    "OMP_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                }
                command = [
                    profiles[platform]["python"],
                    "-u",
                    "-m",
                    "scripts.benchmark.tech_reports.loading.experiments.l03_asset_probe",
                    "--platform",
                    platform,
                    "--manifest",
                    str(args.manifest.resolve()),
                    "--robot",
                    robot,
                    "--num-envs",
                    str(count),
                    "--output",
                    str(output),
                    "--init-mode",
                    args.init_mode,
                    "--steps",
                    str(args.steps),
                ]
                if args.warmup_joint_state:
                    command.append("--warmup-joint-state")
                if args.explicit_cleanup:
                    command.append("--explicit-cleanup")
                if args.overlap:
                    command.append("--overlap")
                record = {
                    "name": name,
                    "command": command,
                    "cwd": str(root),
                    "environment_overrides": overrides,
                    "status": "running",
                    "started_at": datetime.now().astimezone().isoformat(),
                    "performance_eligible": False,
                    "commit": subprocess.check_output(
                        [
                            "git",
                            "-C",
                            profiles[platform]["repository"],
                            "rev-parse",
                            "HEAD",
                        ],
                        text=True,
                    ).strip(),
                }
                (args.output / f"{name}-gpu.txt").write_text(
                    subprocess.check_output(["nvidia-smi"], text=True)
                )
                records.append(record)
                print("START", name, flush=True)
                start = time.monotonic()
                with (args.output / f"{name}.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=root,
                        env=dict(os.environ, **overrides),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    record["pid"] = process.pid
                    (args.output / "execution.json").write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    try:
                        process.wait(timeout=180)
                        record["status"] = (
                            "completed" if process.returncode == 0 else "failed"
                        )
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        record["status"] = "timeout"
                record.update(
                    returncode=process.returncode,
                    wall_s_diagnostic_only=time.monotonic() - start,
                    ended_at=datetime.now().astimezone().isoformat(),
                )
                (args.output / "execution.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                print("END", name, record["status"], flush=True)
                if record["status"] != "completed":
                    print(
                        "\n".join(
                            (args.output / f"{name}.log")
                            .read_text(errors="replace")
                            .splitlines()[-24:]
                        ),
                        flush=True,
                    )
                    return


if __name__ == "__main__":
    main()
