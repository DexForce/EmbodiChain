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

"""L-03 sequential fresh-process matrix with 20 ms memory samples.

Run: python -m scripts.benchmark.tech_reports.loading.experiments.l03_batch --suite main --tag NEW_TAG --wait-idle
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import psutil

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
).resolve()
sys.path.insert(0, str(ROOT / "outputs/benchmarks/L-02/runtime"))
import pynvml as nv

__all__ = ["main"]


def main() -> None:
    """Run the selected three-robot matrix, preserving failures and stopping on errors."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=("pilot", "main", "media"), default="pilot")
    p.add_argument(
        "--robots",
        nargs="+",
        choices=("cartpole", "franka", "go2"),
        default=["cartpole", "franka", "go2"],
    )
    p.add_argument("--num-envs", nargs="+", type=int, default=[1, 100, 1000])
    p.add_argument(
        "--platforms",
        nargs="+",
        choices=("embodichain", "isaaclab"),
        default=["embodichain", "isaaclab"],
    )
    p.add_argument("--cache", nargs="+", choices=("C", "H"), default=["C", "H"])
    p.add_argument("--repeat", type=int)
    p.add_argument("--tag", required=True)
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/benchmarks/L-03/three-robots-v1",
    )
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--wait-idle", action="store_true")
    args = p.parse_args()
    assert not (
        args.suite == "main" and args.allow_busy
    ), "Formal samples require an idle compute GPU"
    if args.suite == "media":
        assert args.num_envs == [
            4
        ], "Media uses four arenas and separately accounted cameras"
    base = args.output_root.resolve()
    (base / "plans").mkdir(parents=True, exist_ok=True)
    manifest_path = (
        ROOT / "outputs/benchmarks/L-03/prepared/shared-usd-v1/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    profiles = json.loads((ROOT / "configs/environments.json").read_text())["profiles"]
    for robot in args.robots:
        for item in manifest["robots"][robot]["files"]:
            assert (
                hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
                == item["sha256"]
            )
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
        platform: subprocess.check_output(
            ["git", "-C", profile["repository"], "rev-parse", "HEAD"], text=True
        ).strip()
        for platform, profile in profiles.items()
    }

    def compute_pids() -> list[int]:
        return [v.pid for v in nv.nvmlDeviceGetComputeRunningProcesses(handle)]

    def gpu_processes() -> dict[int, int]:
        return {
            v.pid: int(v.usedGpuMemory)
            for query in (
                nv.nvmlDeviceGetComputeRunningProcesses,
                nv.nvmlDeviceGetGraphicsRunningProcesses,
            )
            for v in query(handle)
            if v.usedGpuMemory is not None
        }

    plan = base / "plans" / (args.tag + ".json")
    if plan.exists():
        jobs = json.loads(plan.read_text())
        assert not any(
            j["status"] == "running" for j in jobs
        ), "Reconcile interrupted job before resuming"
    else:
        jobs = []
        for n in args.num_envs:
            for robot in args.robots:
                for cache in args.cache:
                    purposes = (
                        [("preload", -1)]
                        if cache == "H" and args.suite == "main"
                        else []
                    )
                    purposes += [
                        (args.suite, i)
                        for i in range(
                            args.repeat or (3 if args.suite == "main" else 1)
                        )
                    ]
                    for purpose, repeat in purposes:
                        for platform in (
                            args.platforms
                            if repeat % 2 == 0
                            else list(reversed(args.platforms))
                        ):
                            jobs.append(
                                dict(
                                    robot=robot,
                                    num_envs=n,
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
                raise SystemExit(f"GPU busy {busy}; pending jobs preserved in {plan}")
            print("WAIT_GPU", busy, flush=True)
            while busy:
                time.sleep(5)
                busy = compute_pids()
        run_id = (
            datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            + f"_{args.tag}_{job['platform']}_{job['robot']}_{job['num_envs']:05d}_{job['cache_mode']}_{job['purpose']}_{job['repeat']:02d}"
        )
        out = base / "runs" / run_id
        out.mkdir(parents=True, exist_ok=False)
        cfg = dict(
            manifest["robots"][job["robot"]],
            **{k: v for k, v in job.items() if k != "status"},
        )
        cfg.update(
            output=str(out),
            record=job["purpose"] == "media",
            dt=0.005,
            colliders_per_env={"cartpole": 3, "franka": 12, "go2": 27}[job["robot"]],
            cache_policy="C: new process; H: matching preload then new process. OS page cache and shader/kernel caches retained in both. No claim of disk-cold starts.",
            dexsim_runtime="outputs/benchmarks/P-03/runtime/p03-frozen-v1",
            camera=dict(
                width=960,
                height=540,
                intrinsics=[650, 650, 480, 270],
                eye=[7, 7, 6],
                target=[0, 0, 2.2],
            ),
        )
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        source = out / "source"
        source.mkdir()
        source_files = [
            Path(__file__),
            Path(__file__).with_name("l03_run.py"),
            ROOT
            / f"scripts/benchmark/tech_reports/loading/platforms/{job['platform']}/l03_loading.py",
        ]
        hashes = {}
        for path in source_files:
            shutil.copy2(path, source / path.name)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        paths = profiles[job["platform"]]["launch_pythonpath"].copy()
        if job["platform"] == "embodichain":
            paths.insert(0, str(ROOT / cfg["dexsim_runtime"]))
        overrides = dict(
            PYTHONPATH=":".join(paths),
            OMNI_KIT_ACCEPT_EULA="YES",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
        )
        command = [
            profiles[job["platform"]]["python"],
            "-u",
            "-m",
            "scripts.benchmark.tech_reports.loading.experiments.l03_run",
            "--config",
            str(out / "config.json"),
        ]
        launch = dict(
            command=command,
            cwd=str(ROOT),
            environment_overrides=overrides,
            hardware=hardware,
            commits=commits,
            source_sha256=hashes,
            gpu_before=gpu_processes(),
            compute_before=compute_pids(),
            start_epoch=time.time(),
        )
        print("START", run_id, flush=True)
        with (
            (out / "console.log").open("w") as log,
            (out / "memory.jsonl").open("w") as memory,
        ):
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=dict(os.environ, **overrides),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            launch["pid"] = process.pid
            (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
            job.update(status="running", run_id=run_id, pid=process.pid)
            plan.write_text(json.dumps(jobs, indent=2) + "\n")
            deadline = time.monotonic() + (240 if cfg["record"] else 180)
            timed_out = False
            competitors = set(launch["compute_before"])
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    ids = [process.pid] + [c.pid for c in children]
                    rss = sum(
                        c.memory_info().rss
                        for c in [parent] + children
                        if c.is_running()
                    )
                except psutil.NoSuchProcess:
                    ids, rss = [process.pid], None
                compute = compute_pids()
                competitors.update(set(compute) - set(ids))
                memory.write(
                    json.dumps(
                        dict(
                            epoch=time.time(),
                            rss_bytes=rss,
                            tree_pids=ids,
                            gpu_process_bytes=gpu_processes(),
                            gpu_total_bytes=nv.nvmlDeviceGetMemoryInfo(handle).used,
                            compute_pids=compute,
                        )
                    )
                    + "\n"
                )
                if time.monotonic() > deadline:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.02)
            code = process.wait()
        result = (
            json.loads((out / "run.json").read_text())
            if (out / "run.json").exists()
            else {}
        )
        completed = code == 0 and not timed_out and result.get("status") == "completed"
        eligible = completed and job["purpose"] == "main" and not competitors
        (out / "exit.json").write_text(
            json.dumps(
                dict(
                    code=code,
                    timeout=timed_out,
                    end_epoch=time.time(),
                    external_compute_pids=sorted(competitors),
                    performance_eligible=eligible,
                ),
                indent=2,
            )
            + "\n"
        )
        job.update(
            status="completed" if completed else "failed", performance_eligible=eligible
        )
        plan.write_text(json.dumps(jobs, indent=2) + "\n")
        print("END", run_id, job["status"], "eligible", eligible, flush=True)
        if not completed or (job["purpose"] == "main" and competitors):
            print(
                "STOP: failure or competing GPU process; remaining jobs retained.",
                flush=True,
            )
            print(
                "\n".join(
                    (out / "console.log").read_text(errors="replace").splitlines()[-25:]
                ),
                flush=True,
            )
            break
    nv.nvmlShutdown()


if __name__ == "__main__":
    main()
