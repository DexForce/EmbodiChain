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

"""CPU-only NVML/procfs fixtures for the optional rendering resource sampler."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmark.rendering import resources

_PID = 1234
_GPU = 3
_LONG_INTERVAL = 3600.0  # Tests request boundary samples; stop must wake immediately.


class _Nvml:
    """Selected-device fixture with independent graphics and compute records."""

    def __init__(self) -> None:
        self.compute: list[Any] = []
        self.graphics: list[Any] = []
        self.used = 4096
        self.initialized = self.closed = 0
        self.handles: list[int] = []
        self.compute_error: Exception | None = None
        self.device_error: Exception | None = None

    def nvmlInit(self) -> None:
        self.initialized += 1

    def nvmlShutdown(self) -> None:
        self.closed += 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        self.handles.append(index)
        return f"selected-{index}"

    def nvmlDeviceGetUUID(self, handle: str) -> bytes:
        return b"GPU-selected-uuid"

    def nvmlDeviceGetName(self, handle: str) -> str:
        return "Selected GPU"

    def nvmlSystemGetDriverVersion(self) -> bytes:
        return b"580.42"

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> SimpleNamespace:
        assert handle == f"selected-{_GPU}"
        if self.device_error:
            raise self.device_error
        return SimpleNamespace(used=self.used)

    def nvmlDeviceGetComputeRunningProcesses(self, handle: str) -> list[Any]:
        assert handle == f"selected-{_GPU}"
        if self.compute_error:
            raise self.compute_error
        return self.compute

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle: str) -> list[Any]:
        assert handle == f"selected-{_GPU}"
        return self.graphics


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> _Nvml:
    module = _Nvml()

    def load(name: str) -> _Nvml:
        assert name == "pynvml"
        return module

    monkeypatch.setattr(resources.importlib, "import_module", load)
    monkeypatch.setattr(
        resources,
        "_read_cpu_memory",
        lambda pid: {
            "cpu_rss_bytes": 1024,
            "cpu_high_water_bytes": 2048,
            "method": "fixture",
            "reasons": {},
        },
    )
    return module


def test_graphics_only_process_memory_and_selected_device_are_measured(
    nvml: _Nvml,
) -> None:
    nvml.graphics = [
        SimpleNamespace(pid=_PID, usedGpuMemory=512),
        SimpleNamespace(pid=9999, usedGpuMemory=8192),
    ]
    with resources.ResourceSampler(
        _GPU, pid=_PID, interval_s=_LONG_INTERVAL
    ) as sampler:
        sampler.mark_phase("measurement")
    result = sampler.summary()

    assert result["metrics"] == {
        "gpu_process_peak_bytes": 512,
        "gpu_process_steady_bytes": 512,
        "gpu_device_peak_bytes": 4096,
        "cpu_peak_rss_bytes": 2048,
    }
    assert nvml.handles == [_GPU]
    assert nvml.initialized == nvml.closed == 1
    assert result["gpu"]["uuid"] == "GPU-selected-uuid"
    assert result["gpu"]["name"] == "Selected GPU"
    assert result["gpu"]["driver"] == "580.42"
    assert result["sample_count"] == 3
    assert result["samples"][0]["process_entries"] == {"compute": 0, "graphics": 1}
    assert not result["running"]
    json.dumps(result, allow_nan=False)


def test_compute_graphics_duplicate_pid_is_not_double_counted(nvml: _Nvml) -> None:
    nvml.compute = [SimpleNamespace(pid=_PID, usedGpuMemory=500)]
    nvml.graphics = [
        SimpleNamespace(pid=_PID, usedGpuMemory=600),
        SimpleNamespace(pid=_PID, usedGpuMemory=600),
    ]
    with resources.ResourceSampler(
        _GPU, pid=_PID, interval_s=_LONG_INTERVAL
    ) as sampler:
        sampler.mark_phase("measurement")
    assert sampler.summary()["metrics"]["gpu_process_peak_bytes"] == 600


@pytest.mark.parametrize(
    "value", [None, -1, 2**64 - 1, ctypes.c_ulonglong(-1), "N/A", True]
)
def test_unavailable_nvml_memory_is_null_with_reason(
    nvml: _Nvml, value: object
) -> None:
    nvml.graphics = [SimpleNamespace(pid=_PID, usedGpuMemory=value)]
    with resources.ResourceSampler(
        _GPU, pid=_PID, interval_s=_LONG_INTERVAL
    ) as sampler:
        sampler.mark_phase("measurement")
    result = sampler.summary()
    assert result["metrics"]["gpu_process_peak_bytes"] is None
    assert (
        result["metric_reasons"]["gpu_process_peak_bytes"]
        == "process_memory_not_available"
    )
    assert result["metrics"]["gpu_device_peak_bytes"] == 4096


def test_absent_process_and_permission_failure_do_not_become_zero(nvml: _Nvml) -> None:
    with resources.ResourceSampler(
        _GPU, pid=_PID, interval_s=_LONG_INTERVAL
    ) as sampler:
        sampler.mark_phase("measurement")
    assert (
        sampler.summary()["metric_reasons"]["gpu_process_peak_bytes"]
        == "process_not_reported_by_nvml"
    )
    nvml.compute_error = PermissionError("process query denied")
    nvml.graphics = [SimpleNamespace(pid=_PID, usedGpuMemory=700)]
    nvml.device_error = PermissionError("device query denied")
    with resources.ResourceSampler(_GPU, pid=_PID, interval_s=_LONG_INTERVAL) as failed:
        failed.mark_phase("measurement")
    result = failed.summary()
    assert result["metrics"]["gpu_process_peak_bytes"] is None
    assert "PermissionError" in result["metric_reasons"]["gpu_process_peak_bytes"]
    assert result["metrics"]["gpu_device_peak_bytes"] is None
    assert result["metrics"]["cpu_peak_rss_bytes"] == 2048


def test_setup_peak_and_measurement_steady_population_are_separate(nvml: _Nvml) -> None:
    entry = SimpleNamespace(pid=_PID, usedGpuMemory=900)
    nvml.graphics = [entry]
    with resources.ResourceSampler(
        _GPU, pid=_PID, interval_s=_LONG_INTERVAL
    ) as sampler:
        assert sampler.summary()["phases"]["measurement"]["sample_count"] == 0
        assert sampler.summary()["metrics"]["gpu_process_steady_bytes"] is None
        entry.usedGpuMemory = 300
        sampler.mark_phase("measurement")
        entry.usedGpuMemory = 500
    result = sampler.summary()
    assert result["phases"]["setup"]["gpu_process_bytes"]["peak_bytes"] == 900
    assert result["phases"]["measurement"]["gpu_process_bytes"]["peak_bytes"] == 500
    assert result["metrics"]["gpu_process_peak_bytes"] == 900
    assert result["metrics"]["gpu_process_steady_bytes"] == 400


def test_partial_sampling_failure_remains_visible_alongside_observed_peak(
    nvml: _Nvml,
) -> None:
    nvml.graphics = [SimpleNamespace(pid=_PID, usedGpuMemory=700)]
    with resources.ResourceSampler(
        _GPU, pid=_PID, interval_s=_LONG_INTERVAL
    ) as sampler:
        nvml.compute_error = PermissionError("became unavailable")
        sampler.mark_phase("measurement")
    result = sampler.summary()
    assert result["metrics"]["gpu_process_peak_bytes"] == 700
    assert result["metrics"]["gpu_process_steady_bytes"] is None
    assert result["metric_warnings"]["gpu_process_peak_bytes"]
    assert result["whole_window"]["gpu_process_bytes"]["missing_samples"] == 2


def test_disabled_sampler_performs_no_reads_and_retains_empty_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any) -> None:
        raise AssertionError("disabled sampler performed a read")

    monkeypatch.setattr(resources.importlib, "import_module", forbidden)
    monkeypatch.setattr(resources, "_read_cpu_memory", forbidden)
    with resources.ResourceSampler(_GPU, enabled=False) as sampler:
        sampler.mark_phase("measurement")
    result = sampler.summary()
    assert result["sample_count"] == 0
    assert result["phases"]["measurement"]["sample_count"] == 0
    assert all(value is None for value in result["metrics"].values())
    assert set(result["metric_reasons"].values()) == {"sampler_disabled"}


def test_nvml_import_failure_keeps_cpu_evidence(
    monkeypatch: pytest.MonkeyPatch, nvml: _Nvml
) -> None:
    def missing(name: str) -> None:
        raise ModuleNotFoundError("pynvml is not installed")

    monkeypatch.setattr(resources.importlib, "import_module", missing)
    with resources.ResourceSampler(
        _GPU, pid=_PID, interval_s=_LONG_INTERVAL
    ) as sampler:
        sampler.mark_phase("measurement")
    result = sampler.summary()
    assert result["metrics"]["cpu_peak_rss_bytes"] == 2048
    assert result["metrics"]["gpu_process_peak_bytes"] is None
    assert "ModuleNotFoundError" in result["metric_reasons"]["gpu_process_peak_bytes"]
    assert nvml.initialized == nvml.closed == 0


def test_sampling_thread_stops_immediately_and_cleanup_preserves_worker_exception(
    nvml: _Nvml,
) -> None:
    sampler = resources.ResourceSampler(_GPU, pid=_PID, interval_s=_LONG_INTERVAL)
    before = time.monotonic()
    with pytest.raises(RuntimeError, match="worker failure"):
        with sampler:
            assert sampler._thread.is_alive()
            raise RuntimeError("worker failure")
    assert time.monotonic() - before < 2.0
    assert not sampler._thread.is_alive()
    assert nvml.closed == 1
    sampler.__exit__(None, None, None)
    assert nvml.closed == 1
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        sampler.__enter__()
    with pytest.raises(RuntimeError, match="active"):
        sampler.mark_phase("measurement")


def test_periodic_sampling_occurs_before_join(
    nvml: _Nvml, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = threading.Event()
    original = resources.ResourceSampler._sample_once

    def sample(sampler: resources.ResourceSampler) -> None:
        original(sampler)
        if threading.current_thread().name.startswith("render-resources-"):
            observed.set()

    monkeypatch.setattr(resources.ResourceSampler, "_sample_once", sample)
    with resources.ResourceSampler(_GPU, pid=_PID, interval_s=0.01) as sampler:
        assert observed.wait(2.0)
    assert sampler.summary()["sample_count"] >= 3
    assert not sampler._thread.is_alive()


def test_procfs_rss_highwater_and_statm_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def status(path: Path) -> str:
        assert path.name == "status"
        return "Name:\tworker\nVmRSS:\t12 kB\nVmHWM:\t24 kB\n"

    monkeypatch.setattr(Path, "read_text", status)
    assert resources._read_cpu_memory(_PID)["cpu_high_water_bytes"] == 24 * 1024

    def fallback(path: Path) -> str:
        if path.name == "status":
            raise PermissionError("status unavailable")
        return "100 7 0 0 0 0 0"

    monkeypatch.setattr(Path, "read_text", fallback)
    monkeypatch.setattr(resources.os, "sysconf", lambda key: 4096)
    result = resources._read_cpu_memory(_PID)
    assert result["cpu_rss_bytes"] == 7 * 4096
    assert result["cpu_high_water_bytes"] is None
    assert "PermissionError" in result["reasons"]["cpu_high_water_bytes"]

    def absent(path: Path) -> str:
        raise FileNotFoundError("process exited")

    monkeypatch.setattr(Path, "read_text", absent)
    result = resources._read_cpu_memory(_PID)
    assert result["cpu_rss_bytes"] is None
    assert "FileNotFoundError" in result["reasons"]["cpu_rss_bytes"]


def test_module_import_does_not_load_optional_or_cuda_runtime_packages() -> None:
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pynvml', 'torch', 'numpy', 'dexsim', 'isaaclab', 'isaacsim', 'embodichain'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from scripts.benchmark.rendering.resources import ResourceSampler
with ResourceSampler(0, enabled=False) as sampler:
    sampler.mark_phase('measurement')
assert sampler.summary()['sample_count'] == 0
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpu_id": -1},
        {"gpu_id": True},
        {"pid": 0},
        {"interval_s": 0},
        {"interval_s": float("nan")},
        {"enabled": "yes"},
    ],
)
def test_invalid_sampler_arguments_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        resources.ResourceSampler(**{"gpu_id": _GPU, **kwargs})
