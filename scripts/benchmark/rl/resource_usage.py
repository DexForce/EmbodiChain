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
"""Sample process RAM and device-scoped process VRAM outside benchmark timing."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

__all__ = ["ProcessResources"]


def _process_gpu_mib(output: str, pids: set[int], gpu_uuid: str) -> float | None:
    """Use an exact GPU/PID match; missing or unavailable values stay unknown."""
    values = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        pid, memory, uuid = parts
        if uuid.lower().removeprefix("gpu-") != gpu_uuid.lower().removeprefix("gpu-"):
            continue
        try:
            if int(pid) in pids:
                value = float(memory)
                if value >= 0 and value < float("inf"):
                    values.append(value)
        except ValueError:
            continue
    return max(values) if values else None


class ProcessResources:
    """Record phase-boundary samples for this Linux CUDA worker, in MiB."""

    def __init__(self, gpu_uuid: str) -> None:
        self.gpu_uuid = gpu_uuid
        self.pids = {os.getpid()}
        status = Path("/proc/self/status")
        if status.is_file():
            for line in status.read_text().splitlines():
                if line.startswith("NSpid:"):
                    self.pids.update(int(value) for value in line.split()[1:])
        self.samples: list[dict[str, Any]] = []

    def sample(self, label: str) -> dict[str, Any]:
        """Synchronize, then sample process and allocator memory outside timing."""
        import psutil
        import resource
        import torch

        torch.cuda.synchronize()
        item: dict[str, Any] = {
            "label": label,
            "cpu_rss_mib": psutil.Process().memory_info().rss / 2**20,
            # Linux ru_maxrss is KiB and covers the worker lifetime so far.
            "cpu_rss_lifetime_peak_mib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            / 1024,
            "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "gpu_process_mib": None,
            "gpu_error": None,
        }
        try:
            pss = getattr(psutil.Process().memory_full_info(), "pss", None)
            item["cpu_pss_mib"] = None if pss is None else pss / 2**20
        except (psutil.AccessDenied, OSError):
            item["cpu_pss_mib"] = None
        try:
            query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            item["gpu_process_mib"] = _process_gpu_mib(
                query.stdout, self.pids, self.gpu_uuid
            )
            if item["gpu_process_mib"] is None:
                item["gpu_error"] = (
                    "No exact GPU/PID memory sample; not interpreted as zero"
                )
        except (OSError, subprocess.SubprocessError) as error:
            item["gpu_error"] = str(error)
        self.samples.append(item)
        return item

    def result(self) -> dict[str, Any]:
        """Return provenance and all observed values, including missing samples."""
        return {
            "schema": "process-memory",
            "unit": "MiB",
            "gpu_method": "nvidia-smi compute-process memory, exact UUID/PID",
            "sampling": "phase boundaries outside timed regions",
            "scope": "scene creation and training, sampled between phases",
            "samples": self.samples,
        }
