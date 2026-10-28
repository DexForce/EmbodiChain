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

"""Run an ACT example with a Torch cap and sample NVML process memory.

Example::

    CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark.learning.monitor_act_runtime \
      --module examples.learning.act.play --report /tmp/act-memory.json -- \
      --checkpoint /tmp/act-dev/best.pt --output /tmp/act-play

The cap applies to Torch's allocator, not Vulkan or other native allocations.
NVML samples are observations rather than proof of an instantaneous peak.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from typing import Any


def _child(module: str, arguments: list[str], cap_gib: float) -> None:
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(cap_gib * 1024**3)
    if cap > total:
        raise ValueError("Torch cap exceeds the selected GPU's total memory")
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    print(f"Torch allocator cap: {cap} bytes; GPU total: {total} bytes", flush=True)
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__")


def _measure(
    module: str, arguments: list[str], cap_gib: float, interval: float, report: Path
) -> dict[str, Any]:
    import pynvml

    pynvml.nvmlInit()
    handles = [
        pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
    ]
    uuids = [pynvml.nvmlDeviceGetUUID(handle) for handle in handles]
    baseline = [pynvml.nvmlDeviceGetMemoryInfo(handle).used for handle in handles]
    peak_device = baseline.copy()
    peak_compute = [0] * len(handles)
    peak_graphics = [0] * len(handles)
    samples = 0
    started = time.monotonic()
    log_path = report.with_suffix(report.suffix + ".log")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--module",
        module,
        "--torch-cap-gib",
        str(cap_gib),
        "--",
        *arguments,
    ]
    child: subprocess.Popen[Any] | None = None
    try:
        with log_path.open("x") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy()
            )
            while child.poll() is None:
                for index, handle in enumerate(handles):
                    peak_device[index] = max(
                        peak_device[index], pynvml.nvmlDeviceGetMemoryInfo(handle).used
                    )
                    for get_processes, peaks in (
                        (pynvml.nvmlDeviceGetComputeRunningProcesses, peak_compute),
                        (pynvml.nvmlDeviceGetGraphicsRunningProcesses, peak_graphics),
                    ):
                        try:
                            processes = get_processes(handle)
                        except pynvml.NVMLError:
                            continue
                        for process in processes:
                            memory = process.usedGpuMemory
                            if (
                                process.pid == child.pid
                                and memory is not None
                                and 0 <= memory < 2**63
                            ):
                                peaks[index] = max(peaks[index], memory)
                samples += 1
                time.sleep(interval)
        result = {
            "module": module,
            "arguments": arguments,
            "torch_allocator_cap_bytes": int(cap_gib * 1024**3),
            "returncode": child.returncode,
            "pid": child.pid,
            "elapsed_seconds": time.monotonic() - started,
            "sample_interval_seconds": interval,
            "samples": samples,
            "devices": [
                {
                    "index": i,
                    "uuid": uuids[i],
                    "baseline_used_bytes": baseline[i],
                    "peak_used_bytes": peak_device[i],
                    "compute_process_peak_bytes": peak_compute[i],
                    "graphics_process_peak_bytes": peak_graphics[i],
                }
                for i in range(len(handles))
                if peak_compute[i] or peak_graphics[i]
            ],
            "measurement_note": (
                "Device totals may include other processes; compute and graphics "
                "process counters may overlap and must not be added. Sampling "
                "can miss short-lived peaks and does not cap native allocations."
            ),
            "log": str(log_path.resolve()),
        }
        report.write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        pynvml.nvmlShutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--torch-cap-gib", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.torch_cap_gib) or args.torch_cap_gib <= 0:
        parser.error("--torch-cap-gib must be finite and positive")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if args.child:
        _child(args.module, arguments, args.torch_cap_gib)
        return
    if args.report is None:
        parser.error("--report is required")
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = _measure(
        args.module, arguments, args.torch_cap_gib, args.interval, args.report
    )
    print(json.dumps(result, indent=2))
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
