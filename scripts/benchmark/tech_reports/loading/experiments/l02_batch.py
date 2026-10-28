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

"""Sequential L-02 fresh-process runs, matched inputs, raw memory and exit records."""

from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[5] / "outputs/benchmarks/L-02/runtime")
)
import pynvml as nv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite", choices=["pilot", "main", "capacity", "media"], default="pilot"
    )
    p.add_argument("--shapes", nargs="+", default=["cube"])
    p.add_argument("--counts", nargs="+", type=int, default=[1])
    p.add_argument("--modes", nargs="+", default=["S"])
    p.add_argument("--platforms", nargs="+", default=["embodichain", "isaaclab"])
    p.add_argument("--cache", nargs="+", default=["H"])
    p.add_argument("--repeat", type=int)
    p.add_argument("--tag", required=True)
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--wait-idle", action="store_true")
    args = p.parse_args()
    root = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
    ).resolve()
    base = root / "outputs/benchmarks/L-02"
    protocol = json.loads(
        (
            root / "scripts/benchmark/tech_reports/assets/loading/L-02/protocol-v2.json"
        ).read_text()
    )
    asset_dir = root / protocol["asset_dir"]
    manifest = json.loads((asset_dir / "manifest.json").read_text())
    byname = {x["file"]: x for x in manifest["files"]}
    import numpy as np

    layouts = np.load(asset_dir / "layouts.npz")
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    hardware = dict(
        gpu_name=nv.nvmlDeviceGetName(handle),
        gpu_uuid=nv.nvmlDeviceGetUUID(handle),
        driver=nv.nvmlSystemGetDriverVersion(),
        gpu_total_bytes=nv.nvmlDeviceGetMemoryInfo(handle).total,
        cpu=Path("/proc/cpuinfo").read_text().split("model name")[1].splitlines()[0],
        cpu_count=psutil.cpu_count(),
        ram_total_bytes=psutil.virtual_memory().total,
    )
    hardware = {
        k: v.decode() if isinstance(v, bytes) else v for k, v in hardware.items()
    }
    commits = {
        str(repo): subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        for repo in [
            Path("/home/oem/code/EmbodiChain"),
            Path("/home/oem/code/IsaacLab"),
            Path("/home/oem/code/dexsim"),
        ]
    }

    def gpu_processes() -> dict:
        return {
            v.pid: int(v.usedGpuMemory)
            for query in [
                nv.nvmlDeviceGetComputeRunningProcesses,
                nv.nvmlDeviceGetGraphicsRunningProcesses,
            ]
            for v in query(handle)
            if v.usedGpuMemory is not None
        }

    # Graphics desktop is recorded but is not a compute competitor.
    def compute_pids() -> list:
        return [v.pid for v in nv.nvmlDeviceGetComputeRunningProcesses(handle)]

    plan = base / "plans" / (args.tag + ".json")
    if plan.exists():
        jobs = json.loads(plan.read_text())
    else:
        jobs = []
        for shape in args.shapes:
            for count in args.counts:
                for mode in args.modes:
                    if count == 1 and mode == "U":
                        continue
                    for cache in args.cache:
                        repeats = args.repeat or (
                            10
                            if args.suite == "main"
                            and shape == "cube"
                            and mode == "S"
                            and count == 1000
                            else 3 if args.suite in ["main", "capacity"] else 1
                        )
                        purposes = (
                            [("prepare", -1)]
                            if cache == "H" and args.suite not in ["pilot", "media"]
                            else []
                        )
                        purposes += [(args.suite, i) for i in range(repeats)]
                        for purpose, repeat in purposes:
                            platforms = (
                                args.platforms
                                if repeat % 2 == 0
                                else list(reversed(args.platforms))
                            )
                            for platform in platforms:
                                jobs.append(
                                    dict(
                                        shape=shape,
                                        object_count=count,
                                        asset_mode=mode,
                                        cache_mode=cache,
                                        purpose=purpose,
                                        repeat=repeat,
                                        platform=platform,
                                        status="pending",
                                    )
                                )
        plan.write_text(json.dumps(jobs, indent=2) + "\n")
    for job in jobs:
        if job["status"] != "pending":
            continue
        busy = compute_pids()
        if busy and not args.allow_busy:
            if not args.wait_idle:
                raise SystemExit(f"GPU busy {busy}; pending jobs remain in {plan}")
            print("Waiting for compute GPU", busy, flush=True)
            while busy:
                time.sleep(5)
                busy = compute_pids()
        if args.suite == "capacity":
            previous_failures = [
                x
                for x in jobs
                if x["platform"] == job["platform"]
                and x["asset_mode"] == job["asset_mode"]
                and x["shape"] == job["shape"]
                and x["object_count"] < job["object_count"]
                and x["status"] == "failed"
            ]
            if previous_failures:
                job.update(
                    status="not_run_after_failure",
                    reason="Lower scale has a recorded failed attempt",
                    boundary_run=previous_failures[-1].get("run_id"),
                )
                plan.write_text(json.dumps(jobs, indent=2) + "\n")
                continue
        count = job["object_count"]
        shape = job["shape"]
        mode = job["asset_mode"]
        cfg = dict(protocol, **{k: v for k, v in job.items() if k != "status"})
        cfg["positions"] = layouts["capacity" if count > 1000 else "main"][
            :count
        ].tolist()
        names = [
            f"{shape}_S.usda" if mode == "S" else f"{shape}_U_{i:05d}.usda"
            for i in range(count)
        ]
        for name in set(names):
            assert (
                hashlib.sha256((asset_dir / name).read_bytes()).hexdigest()
                == byname[name]["sha256"]
            ), name
        cfg["objects"] = [
            dict(byname[name], path=str(asset_dir / name)) for name in names
        ]
        c = protocol["renderer"]
        cfg["camera"] = {
            k: c[k] for k in ["width", "height", "intrinsics", "eye", "target"]
        }
        if count > 1000:
            cfg["camera"].update(eye=c["capacity_eye"], target=c["capacity_target"])
        run_id = (
            datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            + f"_{args.tag}_{job['platform']}_{shape}_{mode}_{count:05d}_{job['cache_mode']}_{job['purpose']}_{job['repeat']:02d}"
        )
        out = base / "runs" / run_id
        out.mkdir(parents=True)
        cfg["output"] = str(out)
        cfg["record"] = args.suite == "media"
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        source = out / "source"
        source.mkdir()
        for path in [
            Path(__file__),
            Path(__file__).with_name("l02_run.py"),
            root
            / f"scripts/benchmark/tech_reports/loading/platforms/{job['platform']}/l02_loading.py",
            root
            / "scripts/benchmark/tech_reports/assets/loading/L-02/protocol-v2.json",
        ]:
            shutil.copy2(path, source / path.name)
        env = os.environ.copy()
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["OMP_NUM_THREADS"] = "4"
        env["OPENBLAS_NUM_THREADS"] = "4"
        paths = [str(root)]
        if job["platform"] == "embodichain":
            paths = [
                str(root / protocol["dexsim_runtime"]),
                str(root),
                "/home/oem/code/EmbodiChain",
                "/home/oem/code/EmbodiChain/extensions",
                "/home/oem/code/EmbodiChain/extensions/embodichain_tasks",
            ]
        env["PYTHONPATH"] = ":".join(paths)
        command = [
            f"/home/oem/miniconda3/envs/{job['platform']}/bin/python",
            "-u",
            "-m",
            "scripts.benchmark.tech_reports.loading.experiments.l02_run",
            "--config",
            str(out / "config.json"),
        ]
        launch = dict(
            command=command,
            env={
                k: env[k]
                for k in [
                    "PYTHONPATH",
                    "OMNI_KIT_ACCEPT_EULA",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                ]
            },
            hardware=hardware,
            commits=commits,
            gpu_before=gpu_processes(),
            gpu_total_before_bytes=nv.nvmlDeviceGetMemoryInfo(handle).used,
            start_epoch=time.time(),
        )
        print("START", run_id, flush=True)
        with (
            (out / "console.log").open("w") as log,
            (out / "memory.jsonl").open("w") as mem,
        ):
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            launch["pid"] = process.pid
            (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            deadline = time.monotonic() + protocol["timeout_s"]
            timeout = False
            while process.poll() is None:
                try:
                    proc = psutil.Process(process.pid)
                    children = proc.children(recursive=True)
                    ids = [process.pid] + [x.pid for x in children]
                    rss = sum(
                        x.memory_info().rss for x in [proc] + children if x.is_running()
                    )
                except psutil.NoSuchProcess:
                    ids = [process.pid]
                    rss = None
                mem.write(
                    json.dumps(
                        dict(
                            epoch=time.time(),
                            rss_bytes=rss,
                            tree_pids=ids,
                            gpu_process_bytes=gpu_processes(),
                            gpu_total_bytes=nv.nvmlDeviceGetMemoryInfo(handle).used,
                            compute_pids=compute_pids(),
                        )
                    )
                    + "\n"
                )
                if time.monotonic() > deadline:
                    import signal

                    os.killpg(process.pid, signal.SIGTERM)
                    timeout = True
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                time.sleep(0.02)
        code = process.wait()
        (out / "exit.json").write_text(
            json.dumps(
                dict(code=code, timeout=timeout, end_epoch=time.time()), indent=2
            )
            + "\n"
        )
        job.update(
            status="completed" if code == 0 and not timeout else "failed",
            run_id=run_id,
        )
        plan.write_text(json.dumps(jobs, indent=2) + "\n")
        print("END", run_id, job["status"], flush=True)
    nv.nvmlShutdown()


if __name__ == "__main__":
    main()
