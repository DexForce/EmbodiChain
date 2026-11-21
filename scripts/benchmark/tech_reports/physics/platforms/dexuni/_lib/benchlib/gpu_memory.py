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
"""Process-scoped GPU memory sampling for benchmark scenarios."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

MIB = 1024.0 * 1024.0


class GpuProcessIdentityError(RuntimeError):
    """Raised when a GPU process cannot be attributed without ambiguity."""


def _normalized_process_names(value: Any) -> set[str]:
    """Return full and basename forms for one process-name value."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip().removesuffix(" (deleted)")
    if not text:
        return set()
    return {text, os.path.basename(text)}


def _process_identity_names(pid: int) -> tuple[str, ...]:
    """Return executable-name evidence visible in the process namespace."""
    values = []
    try:
        values.append(os.readlink(f"/proc/{pid}/exe"))
    except OSError:
        pass
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0", 1)[0]
        values.append(command)
    except OSError:
        pass
    try:
        values.append(
            Path(f"/proc/{pid}/comm").read_text(encoding="utf-8", errors="replace")
        )
    except OSError:
        pass
    if pid == os.getpid():
        values.append(sys.executable)

    names = set()
    for value in values:
        names.update(_normalized_process_names(value))
    return tuple(sorted(names))


def _pid_namespace_candidates(pid: int) -> tuple[int, ...]:
    """Return host-to-container PID aliases reported by Linux ``NSpid``."""
    candidates = [pid]
    try:
        status = Path(f"/proc/{pid}/status").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return tuple(candidates)

    for line in status.splitlines():
        if not line.startswith("NSpid:"):
            continue
        for value in line.split()[1:]:
            try:
                candidate = int(value)
            except ValueError:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        break
    return tuple(candidates)


@dataclass(frozen=True)
class GpuProcessMemorySample:
    """One process-scoped framebuffer-memory sample."""

    label: str
    process_mib: float


class GpuProcessMemoryTracker:
    """Sample GPU framebuffer memory attributed to the current process.

    NVML is preferred because it can query both compute and graphics process
    lists.  ``nvidia-smi pmon`` is used as a dependency-free fallback.  A PID
    reported by both lists is counted once by taking the larger value.
    """

    def __init__(self, gpu_id: int = 0, pid: Optional[int] = None):
        self.gpu_id = gpu_id
        self.pid = os.getpid() if pid is None else pid
        self.pid_candidates = _pid_namespace_candidates(self.pid)
        self.process_identity_names = _process_identity_names(self.pid)
        self.pid_resolution = "unresolved"
        self.samples: List[GpuProcessMemorySample] = []
        self.tracking_available = False
        self.backend = "unavailable"
        self.error = ""
        self._pynvml = None
        self._nvml_handle = None
        self._initial_gpu_pids: set[int] = set()
        self._gpu_pid_baseline_recorded = False

    def start(self) -> bool:
        """Initialize the sampler and record the process baseline."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            self.backend = "nvml"
        except Exception as error:
            self._shutdown_nvml()
            self.error = str(error)
            if self._nvidia_smi_process_mib() is None:
                return False
            self.backend = "nvidia-smi"

        self.tracking_available = True
        if self.sample("initial") is None:
            self.tracking_available = False
            return False
        return True

    def sample(self, label: str) -> Optional[float]:
        """Record and return the current process GPU memory in MiB."""
        if self.backend == "nvml":
            try:
                value = self._nvml_process_mib(
                    allow_process_name_resolution=label != "initial"
                )
            except GpuProcessIdentityError as error:
                self.error = str(error)
                value = None
            except Exception as error:
                self.error = str(error)
                value = self._nvidia_smi_process_mib()
                if value is not None:
                    self.backend = "nvidia-smi"
        elif self.backend == "nvidia-smi":
            value = self._nvidia_smi_process_mib()
        else:
            value = None

        if value is None:
            self.tracking_available = False
            return None
        self.samples.append(GpuProcessMemorySample(label, value))
        return value

    def close(self) -> None:
        """Release NVML after the owning World has been destroyed."""
        self._shutdown_nvml()

    @property
    def peak_mib(self) -> float:
        return max((sample.process_mib for sample in self.samples), default=0.0)

    def metrics(self) -> Dict[str, Any]:
        values = [sample.process_mib for sample in self.samples]
        by_label = {sample.label: sample.process_mib for sample in self.samples}
        initial = values[0] if values else 0.0
        final = values[-1] if values else 0.0
        world_mib = by_label.get("after_world_create", initial)
        return {
            "gpu_tracking_available": self.tracking_available,
            "gpu_tracking_backend": self.backend,
            "gpu_tracking_error": self.error,
            "gpu_process_pid_candidates": list(self.pid_candidates),
            "gpu_process_pid_resolution": self.pid_resolution,
            "gpu_memory_sample_count": len(values),
            "gpu_process_initial_mib": initial,
            "gpu_process_after_world_create_mib": world_mib,
            "gpu_process_after_finalize_mib": by_label.get("after_finalize", 0.0),
            "gpu_process_after_first_step_mib": by_label.get(
                "after_first_measure_step", 0.0
            ),
            "gpu_process_peak_mib": self.peak_mib,
            "gpu_process_scene_delta_mib": max(0.0, self.peak_mib - world_mib),
            "gpu_process_final_mib": final,
            "gpu_process_samples_mib": by_label,
        }

    def _nvml_process_mib(self, allow_process_name_resolution: bool = True) -> float:
        if self._pynvml is None or self._nvml_handle is None:
            raise RuntimeError("NVML is not initialized")

        process_mib: Dict[int, float] = {}
        successful_queries = 0
        unavailable = getattr(self._pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        for function_name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            function = getattr(self._pynvml, function_name, None)
            if not callable(function):
                continue
            try:
                processes = function(self._nvml_handle)
                successful_queries += 1
            except Exception:
                continue
            for process in processes:
                used = getattr(process, "usedGpuMemory", None)
                if used is None:
                    continue
                used_bytes = int(used)
                if unavailable is not None and used_bytes == int(unavailable):
                    continue
                if used_bytes < 0 or used_bytes >= (1 << 63):
                    continue
                process_mib[int(process.pid)] = max(
                    process_mib.get(int(process.pid), 0.0), used_bytes / MIB
                )

        if successful_queries == 0:
            raise RuntimeError("NVML process-memory queries are unavailable")
        return self._resolve_nvml_process_mib(
            process_mib, allow_process_name_resolution=allow_process_name_resolution
        )

    def _resolve_nvml_process_mib(
        self, process_mib: Dict[int, float], allow_process_name_resolution: bool
    ) -> float:
        """Resolve an NVML host PID without claiming an unrelated process."""
        if not self._gpu_pid_baseline_recorded:
            self._initial_gpu_pids = set(process_mib)
            self._gpu_pid_baseline_recorded = True

        direct_values = [
            value for pid, value in process_mib.items() if pid in self.pid_candidates
        ]
        if direct_values:
            self.pid_resolution = "pid"
            return max(direct_values)
        if not allow_process_name_resolution:
            return 0.0

        new_pids = sorted(set(process_mib) - self._initial_gpu_pids)
        get_process_name = getattr(self._pynvml, "nvmlSystemGetProcessName", None)
        identity_names = set(self.process_identity_names)
        matching_pids = []
        named_pids = []
        for pid in new_pids:
            if not callable(get_process_name) or not identity_names:
                continue
            try:
                reported_names = _normalized_process_names(get_process_name(pid))
            except Exception:
                continue
            if reported_names:
                named_pids.append(pid)
            if identity_names.intersection(reported_names):
                matching_pids.append(pid)

        if len(matching_pids) > 1:
            raise GpuProcessIdentityError(
                "NVML host PID resolution is ambiguous for matching GPU "
                f"processes: {matching_pids}"
            )
        if matching_pids:
            resolved_pid = matching_pids[0]
            resolution = "nvml_process_name"
        elif len(new_pids) == 1 and not named_pids:
            resolved_pid = new_pids[0]
            resolution = "nvml_new_process"
        else:
            if len(new_pids) > 1:
                raise GpuProcessIdentityError(
                    "NVML host PID resolution is ambiguous for newly "
                    f"appeared GPU processes: {new_pids}"
                )
            if process_mib:
                self.error = (
                    "NVML reported GPU processes, but none could be safely "
                    "matched to the benchmark process"
                )
            return 0.0

        self.pid_candidates = (*self.pid_candidates, resolved_pid)
        self.pid_resolution = resolution
        self.error = ""
        return process_mib[resolved_pid]

    def _nvidia_smi_process_mib(self) -> Optional[float]:
        command = [
            "nvidia-smi",
            "pmon",
            "--id",
            str(self.gpu_id),
            "--select",
            "m",
            "--count",
            "1",
        ]
        try:
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=2.0
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None

        values = []
        for line in result.stdout.splitlines():
            if not line or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                if int(fields[1]) in self.pid_candidates:
                    values.append(float(fields[3]))
            except ValueError:
                continue
        return max(values, default=0.0)

    def _shutdown_nvml(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._pynvml = None
        self._nvml_handle = None
