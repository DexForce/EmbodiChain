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

"""Read-only process resource sampling with a persistent selected-device NVML handle.

This module never imports a simulator or initializes CUDA. Sampling is optional;
missing counters remain null with reasons. Torch allocator statistics belong to
the worker and are deliberately separate from NVML process memory.
"""

from __future__ import annotations

from collections import Counter
import copy
import importlib
import math
import os
from pathlib import Path
import statistics
import threading
import time
from types import TracebackType
from typing import Any

__all__ = ["ResourceSampler"]

_SIGNALS = (
    "gpu_process_bytes",
    "gpu_device_bytes",
    "cpu_rss_bytes",
    "cpu_high_water_bytes",
)


def _error(exc: Exception) -> str:
    """Retain the error category as well as the driver or OS explanation."""
    return f"{type(exc).__name__}: {exc}"


def _bytes(value: object) -> int | None:
    """Reject NVML sentinels, nulls and booleans instead of reporting zero."""
    value = getattr(value, "value", value)
    if type(value) is int and 0 <= value < 2**64 - 1:
        return value
    return None


def _read_cpu_memory(pid: int) -> dict[str, Any]:
    """Read current RSS and kernel high-water RSS, falling back to statm."""
    reasons = []
    try:
        fields = {}
        for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
            name, _, raw = line.partition(":")
            if name in {"VmRSS", "VmHWM"}:
                amount, unit = raw.split()
                if unit != "kB" or int(amount) < 0:
                    raise ValueError(f"Unexpected {name} unit/value")
                fields[name] = int(amount) * 1024
        if "VmRSS" not in fields:
            raise ValueError("VmRSS absent from procfs status")
        return {
            "cpu_rss_bytes": fields["VmRSS"],
            "cpu_high_water_bytes": fields.get("VmHWM"),
            "method": "procfs_status",
            "reasons": (
                {"cpu_high_water_bytes": "VmHWM absent"}
                if "VmHWM" not in fields
                else {}
            ),
        }
    except (OSError, ValueError) as exc:
        reasons.append(_error(exc))
    try:
        fields = (Path("/proc") / str(pid) / "statm").read_text().split()
        rss = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        if rss < 0:
            raise ValueError("Negative procfs RSS")
        return {
            "cpu_rss_bytes": rss,
            "cpu_high_water_bytes": None,
            "method": "procfs_statm",
            "reasons": {"cpu_high_water_bytes": "; ".join(reasons)},
        }
    except (OSError, ValueError, IndexError) as exc:
        reasons.append(_error(exc))
    reason = "cpu_memory_unavailable: " + "; ".join(reasons)
    return {
        "cpu_rss_bytes": None,
        "cpu_high_water_bytes": None,
        "method": "procfs_unavailable",
        "reasons": {key: reason for key in ("cpu_rss_bytes", "cpu_high_water_bytes")},
    }


def _signal_summary(samples: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    """Describe sampled values and gaps without hiding partial read failures."""
    values = [sample[signal] for sample in samples if sample.get(signal) is not None]
    return {
        "median_bytes": statistics.median(values) if values else None,
        "peak_bytes": max(values, default=None),
        "valid_samples": len(values),
        "missing_samples": len(samples) - len(values),
        "missing_reasons": dict(
            Counter(
                sample["reasons"].get(signal, "missing")
                for sample in samples
                if sample.get(signal) is None
            )
        ),
    }


class ResourceSampler:
    """Sample one process on one physical NVML device, without a CUDA context.

    ``gpu_id`` is an NVML physical-device index, not a CUDA-visible ordinal.
    Graphics and compute process lists are both queried. Duplicate entries for
    the selected PID use their maximum reported process total, never their sum.
    An absent PID, unsupported list or unavailable memory counter remains null.

    Peaks are sampled over the context's lifetime and can miss shorter spikes.
    CPU ``VmHWM`` additionally provides the kernel process-lifetime high water.
    The measurement phase's median is the steady GPU process-memory metric.

    Args:
        gpu_id: Physical NVML device index selected by the launcher.
        pid: Process to measure; defaults to the current worker.
        interval_s: Sampling interval between read operations, in seconds.
        enabled: Disable all reads and optional imports when false.
    """

    def __init__(
        self,
        gpu_id: int,
        pid: int | None = None,
        interval_s: float = 0.1,
        enabled: bool = True,
    ) -> None:
        if type(gpu_id) is not int or gpu_id < 0:
            raise ValueError("gpu_id must be a nonnegative NVML device index")
        if pid is not None and (type(pid) is not int or pid < 1):
            raise ValueError("pid must be a positive integer")
        if (
            isinstance(interval_s, bool)
            or not math.isfinite(interval_s)
            or interval_s <= 0
        ):
            raise ValueError("interval_s must be positive and finite")
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        self.gpu_id, self.pid = gpu_id, os.getpid() if pid is None else pid
        self.interval_s, self.enabled = float(interval_s), enabled
        self._nvml: Any = None
        self._handle: Any = None
        self._nvml_initialized = False
        self._nvml_reason = "sampler_not_started"
        self._gpu: dict[str, Any] = {
            "index": gpu_id,
            "index_space": "nvml_physical_device_index",
            "uuid": None,
            "name": None,
            "driver": None,
        }
        self._samples: list[dict[str, Any]] = []
        self._phase = "setup"
        self._phase_events: list[dict[str, Any]] = []
        self._started_at: float | None = None
        self._entered = False
        self._active = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process_functions: dict[str, Any] = {}
        self._lifecycle_errors: list[str] = []

    def __enter__(self) -> ResourceSampler:
        """Start optional readers and the sampling thread once."""
        if self._entered:
            raise RuntimeError("ResourceSampler instances cannot be restarted")
        self._entered = self._active = True
        self._started_at = time.monotonic()
        self._phase_events.append({"phase": "setup", "time_s": 0.0})
        if not self.enabled:
            self._nvml_reason = "sampler_disabled"
            return self
        self._initialize_nvml()
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run, name=f"render-resources-{self.pid}", daemon=True
        )
        try:
            self._thread.start()
        except Exception:
            self._active = False
            self._shutdown_nvml()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop and join sampling before closing its NVML handle."""
        if not self._active:
            return
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join()
            if self.enabled:
                self._sample_once()
        finally:
            self._active = False
            self._shutdown_nvml()

    def mark_phase(self, name: str) -> None:
        """Mark a lifecycle boundary and sample immediately outside the timer.

        Args:
            name: Phase name, notably ``setup``, ``warmup`` or ``measurement``.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("phase name must be nonempty")
        if not self._active:
            raise RuntimeError("mark_phase requires an active ResourceSampler context")
        with self._lock:
            self._phase = name
            self._phase_events.append(
                {"phase": name, "time_s": time.monotonic() - self._started_at}
            )
        if self.enabled:
            self._sample_once()

    def _initialize_nvml(self) -> None:
        """Resolve one persistent device handle and independent provenance fields."""
        try:
            self._nvml = importlib.import_module("pynvml")
            self._nvml.nvmlInit()
            self._nvml_initialized = True
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except Exception as exc:
            self._nvml_reason = "nvml_unavailable: " + _error(exc)
            return
        self._nvml_reason = ""
        for key, method, args in (
            ("uuid", "nvmlDeviceGetUUID", (self._handle,)),
            ("name", "nvmlDeviceGetName", (self._handle,)),
            ("driver", "nvmlSystemGetDriverVersion", ()),
        ):
            try:
                value = getattr(self._nvml, method)(*args)
                self._gpu[key] = (
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else str(value)
                )
            except Exception as exc:
                self._gpu[f"{key}_reason"] = _error(exc)

    def _shutdown_nvml(self) -> None:
        """Balance only this instance's successful NVML initialization."""
        if self._nvml_initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:
                self._lifecycle_errors.append("nvml_shutdown: " + _error(exc))
            finally:
                self._nvml_initialized = False

    def _processes(self, kind: str) -> list[Any]:
        """Cache a supported process-list API, falling back on missing symbols."""
        if kind in self._process_functions:
            return self._process_functions[kind](self._handle)
        base = f"nvmlDeviceGet{kind}RunningProcesses"
        error: Exception = AttributeError(f"No {kind} process-list API")
        for name in (base, base + "_v3", base + "_v2"):
            function = getattr(self._nvml, name, None)
            if function is None:
                continue
            try:
                result = function(self._handle)
                self._process_functions[kind] = function
                return result
            except Exception as exc:
                error = exc
                if type(exc).__name__ not in {
                    "NVMLError_FunctionNotFound",
                    "AttributeError",
                }:
                    raise
        raise error

    def _read_gpu_memory(self) -> dict[str, Any]:
        """Keep device-wide usage separate from deduplicated per-process usage."""
        sample: dict[str, Any] = {
            "gpu_process_bytes": None,
            "gpu_device_bytes": None,
            "reasons": {},
        }
        if self._handle is None:
            sample["reasons"] = {
                key: self._nvml_reason
                for key in ("gpu_process_bytes", "gpu_device_bytes")
            }
            return sample
        try:
            sample["gpu_device_bytes"] = _bytes(
                self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used
            )
            if sample["gpu_device_bytes"] is None:
                sample["reasons"]["gpu_device_bytes"] = "device_memory_not_available"
        except Exception as exc:
            sample["reasons"]["gpu_device_bytes"] = _error(exc)
        values = []
        query_errors = []
        invalid = False
        counts = {}
        for kind in ("Compute", "Graphics"):
            try:
                entries = [
                    entry for entry in self._processes(kind) if entry.pid == self.pid
                ]
                counts[kind.lower()] = len(entries)
                for entry in entries:
                    value = _bytes(entry.usedGpuMemory)
                    if value is None:
                        invalid = True
                    else:
                        values.append(value)
            except Exception as exc:
                counts[kind.lower()] = None
                query_errors.append(f"{kind.lower()}_process_query: {_error(exc)}")
        sample["process_entries"] = counts
        if query_errors:
            sample["reasons"]["gpu_process_bytes"] = "; ".join(query_errors)
        elif invalid:
            sample["reasons"]["gpu_process_bytes"] = "process_memory_not_available"
        elif not values:
            sample["reasons"]["gpu_process_bytes"] = "process_not_reported_by_nvml"
        else:
            sample["gpu_process_bytes"] = max(values)
        return sample

    def _sample_once(self) -> None:
        """Record one bounded-size observation, including read errors and phase."""
        with self._sample_lock:
            start = time.monotonic()
            with self._lock:
                phase = self._phase
            try:
                cpu = _read_cpu_memory(self.pid)
            except Exception as exc:
                cpu = {
                    "cpu_rss_bytes": None,
                    "cpu_high_water_bytes": None,
                    "method": "reader_error",
                    "reasons": {
                        key: _error(exc)
                        for key in ("cpu_rss_bytes", "cpu_high_water_bytes")
                    },
                }
            gpu = self._read_gpu_memory()
            sample = {
                **cpu,
                **gpu,
                "phase": phase,
                "time_s": start - self._started_at,
                "read_duration_s": time.monotonic() - start,
                "cpu_method": cpu["method"],
                "reasons": {**cpu["reasons"], **gpu["reasons"]},
            }
            sample.pop("method")
            with self._lock:
                self._samples.append(sample)

    def _run(self) -> None:
        """Wait interruptibly between samples; stop does not wait a full interval."""
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe metrics, phase populations, provenance and read gaps.

        Returns:
            Separate metric values and missing reasons, all raw samples, selected
            GPU provenance and setup/measurement phase statistics. Finite sampled
            peaks may underestimate true peaks; partial failures remain visible.
        """
        with self._lock:
            samples = copy.deepcopy(self._samples)
            events = copy.deepcopy(self._phase_events)
        phases = {}
        for phase in dict.fromkeys(
            ("setup", "measurement", *(event["phase"] for event in events))
        ):
            members = [sample for sample in samples if sample["phase"] == phase]
            phases[phase] = {
                "sample_count": len(members),
                **{signal: _signal_summary(members, signal) for signal in _SIGNALS},
            }
        all_signals = {signal: _signal_summary(samples, signal) for signal in _SIGNALS}
        rss_peaks = [
            sample[key]
            for sample in samples
            for key in ("cpu_rss_bytes", "cpu_high_water_bytes")
            if sample.get(key) is not None
        ]
        metrics = {
            "gpu_process_peak_bytes": all_signals["gpu_process_bytes"]["peak_bytes"],
            "gpu_process_steady_bytes": phases["measurement"]["gpu_process_bytes"][
                "median_bytes"
            ],
            "gpu_device_peak_bytes": all_signals["gpu_device_bytes"]["peak_bytes"],
            "cpu_peak_rss_bytes": max(rss_peaks, default=None),
        }
        populations = {
            "gpu_process_peak_bytes": all_signals["gpu_process_bytes"],
            "gpu_process_steady_bytes": phases["measurement"]["gpu_process_bytes"],
            "gpu_device_peak_bytes": all_signals["gpu_device_bytes"],
            "cpu_peak_rss_bytes": all_signals["cpu_rss_bytes"],
        }
        empty_reason = (
            "sampler_disabled"
            if not self.enabled
            else (
                "sampler_not_started"
                if not self._entered
                else "no_samples_in_requested_phase"
            )
        )
        reasons = {
            key: "; ".join(populations[key]["missing_reasons"]) or empty_reason
            for key, value in metrics.items()
            if value is None
        }
        return {
            "schema_version": 1,
            "method": "persistent_nvml_graphics_and_compute_process_lists_and_procfs",
            "enabled": self.enabled,
            "running": self._active,
            "pid": self.pid,
            "interval_s": self.interval_s,
            "sample_count": len(samples),
            "gpu": copy.deepcopy(self._gpu),
            "metrics": metrics,
            "metric_reasons": reasons,
            "metric_warnings": {
                key: population["missing_reasons"]
                for key, population in populations.items()
                if population["missing_reasons"]
            },
            "phases": phases,
            "phase_events": events,
            "samples": samples,
            "whole_window": all_signals,
            "lifecycle_errors": list(self._lifecycle_errors),
            "scope": {
                "gpu_process": "selected PID on selected device; max of duplicate graphics/compute entries",
                "gpu_device": "selected whole device, includes other processes",
                "cpu_peak": "process lifetime VmHWM when available; otherwise sampled RSS peak",
                "gpu_peak": "sampled sampler-lifetime peak; sub-interval spikes may be missed",
                "gpu_steady": "median of measurement-phase samples; not a Torch allocator metric",
            },
        }
