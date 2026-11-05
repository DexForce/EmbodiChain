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

"""Sequential P-03 launcher retaining every command, exit and simulator log."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import psutil


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--platforms", nargs="+", default=["embodichain", "isaaclab"])
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--config", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--episode-start", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--record", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--latency", action="store_true")
    p.add_argument("--mode", choices=["physics", "env"], default="physics")
    p.add_argument(
        "--require-idle",
        action="store_true",
        help="Stop before a new run if another GPU compute process is present.",
    )
    p.add_argument("--tag", default="pilot")
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    root = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
    ).resolve()
    output = root / "outputs/benchmarks/P-03"
    runtime = output / "runtime/p03-frozen-v1"
    config = (
        args.config
        or root / "scripts/benchmark/tech_reports/assets/physics/P-03/protocol-v4.json"
    ).resolve()
    inputs = (args.inputs or output / "prepared/grasp-v1").resolve()
    for shape in args.shapes:
        for repeat in range(args.repeat):
            for platform in args.platforms:
                if args.require_idle:
                    occupied = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,process_name",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    ).strip()
                    if occupied:
                        raise SystemExit(
                            "Formal timing deferred: GPU has other compute processes:\n"
                            + occupied
                        )
                run_id = (
                    datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
                    + f"_{args.tag}_{platform}_{shape}_{args.mode}_n{args.num_envs:04d}_r{repeat:02d}"
                )
                folder = output / "runs" / run_id
                folder.mkdir()
                env = os.environ.copy()
                paths = [str(root)]
                if platform == "embodichain":
                    paths = (
                        [str(runtime)]
                        + paths
                        + [
                            "/home/oem/code/EmbodiChain",
                            "/home/oem/code/EmbodiChain/extensions",
                            "/home/oem/code/EmbodiChain/extensions/embodichain_tasks",
                        ]
                    )
                else:
                    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
                env["PYTHONPATH"] = ":".join(paths)
                cmd = [
                    f"/home/oem/miniconda3/envs/{platform}/bin/python",
                    "-u",
                    "-m",
                    "scripts.benchmark.tech_reports.physics.experiments.p03_run",
                    "--platform",
                    platform,
                    "--config",
                    str(config),
                    "--inputs",
                    str(inputs),
                    "--output",
                    str(folder),
                    "--shape",
                    shape,
                    "--num-envs",
                    str(args.num_envs),
                    "--count",
                    str(args.count),
                    "--episode-start",
                    str(args.episode_start),
                    "--mode",
                    args.mode,
                ]
                for flag in ["record", "calibrate", "latency"]:
                    if getattr(args, flag):
                        cmd.append("--" + flag)
                launch = dict(
                    command=cmd,
                    PYTHONPATH=env["PYTHONPATH"],
                    OMNI_KIT_ACCEPT_EULA=env.get("OMNI_KIT_ACCEPT_EULA"),
                    runtime_manifest=(
                        str(runtime / "manifest.json")
                        if platform == "embodichain"
                        else None
                    ),
                )
                (folder / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
                print(folder, flush=True)
                with (
                    (folder / "console.log").open("w") as log,
                    (folder / "gpu_memory.csv").open("w") as gpu_log,
                    (folder / "process_memory.csv").open("w") as ram_log,
                ):
                    ram_log.write("epoch_s,rss_bytes\n")
                    gpu_monitor = subprocess.Popen(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=timestamp,pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                            "--loop-ms=20",
                        ],
                        stdout=gpu_log,
                        stderr=subprocess.DEVNULL,
                    )
                    try:
                        r = subprocess.Popen(
                            cmd,
                            env=env,
                            cwd=root,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        launch["pid"] = r.pid
                        (folder / "launch.json").write_text(
                            json.dumps(launch, indent=2) + "\n"
                        )
                        process = psutil.Process(r.pid)
                        start = time.monotonic()
                        while r.poll() is None:
                            if time.monotonic() - start > args.timeout:
                                r.kill()
                                r.wait()
                                raise subprocess.TimeoutExpired(cmd, args.timeout)
                            try:
                                ram_log.write(
                                    f"{time.time():.6f},{process.memory_info().rss}\n"
                                )
                            except psutil.NoSuchProcess:
                                break
                            time.sleep(0.02)
                        r.wait()
                        result = dict(returncode=r.returncode)
                    except subprocess.TimeoutExpired:
                        result = dict(
                            returncode=None, failure="timeout", timeout_s=args.timeout
                        )
                    finally:
                        gpu_monitor.terminate()
                        gpu_monitor.wait()
                result["ended_at"] = datetime.now().astimezone().isoformat()
                (folder / "exit.json").write_text(json.dumps(result, indent=2) + "\n")
                state = (
                    json.loads((folder / "run.json").read_text())
                    if (folder / "run.json").exists()
                    else {}
                )
                print(
                    result,
                    "run status",
                    state.get("status"),
                    state.get("failure"),
                    flush=True,
                )
                if state.get("status") != "completed":
                    print((folder / "console.log").read_text()[-1800:], flush=True)


if __name__ == "__main__":
    main()
