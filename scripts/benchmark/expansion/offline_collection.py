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

"""Measure accepted and confirmed native collection throughput in isolated processes.

Run: python scripts/benchmark/expansion/offline_collection.py --job JOB.yaml --output OUTPUT
Each fresh job owns one World. Startup-inclusive throughput is primary; native
profiler phases and the collection-only interval are reported separately. RGB-D
requires a camera embodiment in JOB. No repeated runs overwrite prior evidence.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import time
import json
import platform
import os
import psutil
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", default="1,8,32")
    parser.add_argument("--modes", default="state,rgbd")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from embodichain.lab.expansion.configured import load_expansion_job

    cfg = load_expansion_job(args.job)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "seed": cfg.seed,
        "threads": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "scope": "fresh-process wall time through verified commit; no warmup excluded",
        "runs": [],
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        report["gpus"] = [str(pynvml.nvmlDeviceGetName(h)) for h in handles]
    except Exception:
        pynvml = None
        handles = []
    for mode in args.modes.split(","):
        if mode not in {"state", "rgbd"}:
            raise ValueError("modes must be state,rgbd")
        for size in map(int, args.sizes.split(",")):
            data = cfg.to_dict()
            name = f"{mode}-b{size}"
            root = (args.output / name).resolve()
            data.update(
                batch_size=size,
                target_episodes=size,
                max_attempts=size * 2,
                capture_media=mode == "rgbd",
                export_lerobot=False,
                render_profiles=(),
                output_dir=str(root),
            )
            job = args.output / (name + ".yaml")
            job.write_text(yaml.safe_dump(data))
            peak_rss = peak_gpu = 0
            start = time.perf_counter()
            with (args.output / (name + ".log")).open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "embodichain",
                        "expansion",
                        "run",
                        "--job",
                        str(job),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        peak_rss = max(
                            peak_rss,
                            sum(
                                p.memory_info().rss
                                for p in [parent, *parent.children(recursive=True)]
                            ),
                        )
                    except psutil.Error:
                        pass
                    if pynvml is not None:
                        used = 0
                        for handle in handles:
                            try:
                                entries = pynvml.nvmlDeviceGetComputeRunningProcesses(
                                    handle
                                ) + pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                                used += max(
                                    (
                                        entry.usedGpuMemory
                                        for entry in entries
                                        if entry.pid == process.pid
                                    ),
                                    default=0,
                                )
                            except pynvml.NVMLError:
                                pass
                        peak_gpu = max(peak_gpu, used)
                    time.sleep(0.2)
            elapsed = time.perf_counter() - start
            runtime = (
                json.loads((root / "runtime.json").read_text())
                if (root / "runtime.json").exists()
                else {}
            )
            from embodichain.lab.expansion.workflow import inspect_expansion_job

            outcome = (
                inspect_expansion_job(root)
                if (root / "job.json").exists()
                else {
                    "committed": 0,
                    "attempts": 0,
                    "complete": False,
                    "failure": "preflight; see job log",
                }
            )
            row = {
                "mode": mode,
                "batch_size": size,
                "exit_code": process.returncode,
                "elapsed_seconds": elapsed,
                "peak_process_rss_bytes": peak_rss,
                "peak_process_gpu_bytes": peak_gpu,
                "committed_episodes_per_second": outcome["committed"] / elapsed,
                "success_rate": outcome["committed"] / max(outcome["attempts"], 1),
                "outcome": outcome,
                "runtime": runtime,
            }
            report["runs"].append(row)
            (args.output / "results.json").write_text(json.dumps(report, indent=2))
            print(
                name,
                outcome["committed"],
                "committed",
                round(elapsed, 2),
                "seconds",
                flush=True,
            )
    if pynvml is not None:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
