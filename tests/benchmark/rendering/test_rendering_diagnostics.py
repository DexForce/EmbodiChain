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

"""CPU fixture proofs for partial trace evidence and matched overhead windows."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import weakref

import numpy as np
import pytest

from scripts.benchmark.rendering import diagnostics


class Clock:
    def __init__(self, read_cost: float = 0.0) -> None:
        self.now, self.read_cost = 0.0, read_cost

    def __call__(self) -> float:
        self.now += self.read_cost
        return self.now


class HostArray(np.ndarray):
    def __new__(cls, adapter: Adapter) -> HostArray:
        result = np.zeros((1, 3, 4, 3), dtype=np.uint8).view(cls)
        result.adapter = adapter
        adapter.host_arrays.append(weakref.ref(result))
        return result

    def __array_finalize__(self, source: Any) -> None:
        self.adapter = getattr(source, "adapter", None)

    def copy(self, order: str = "C") -> HostArray:
        self.adapter.events.append("host_copy")
        self.adapter.clock.now += 0.005
        result = super().copy(order=order)
        self.adapter.host_arrays.append(weakref.ref(result))
        return result


class Adapter:
    def __init__(self, read_cost: float = 0.0) -> None:
        self.clock = Clock(read_cost)
        self.events: list[str] = []
        self.host_arrays: list[Any] = []
        self.original = {
            "rgb": SimpleNamespace(is_cuda=True, shape=(1, 3, 4, 3), dtype="uint8")
        }
        self.transfer_sources = []
        self.metadata = {
            "resolution_evidence": {"native_resolution_status": "not_verified"}
        }

    def render_device(self) -> dict[str, Any]:
        assert all(reference() is None for reference in self.host_arrays)
        self.events.append("render")
        self.clock.now += 0.02
        return self.original

    def synchronize(self) -> None:
        self.events.append("sync")
        self.clock.now += 0.01

    def to_host(self, arrays: dict[str, Any]) -> dict[str, HostArray]:
        assert arrays is self.original
        self.transfer_sources.append(arrays)
        self.events.append("d2h")
        self.clock.now += 0.03
        return {"rgb": HostArray(self)}


class Profiler:
    ProfilerActivity = SimpleNamespace(CPU="CPU", CUDA="CUDA")

    def __init__(self, adapter: Adapter, payload: dict[str, Any] | None = None) -> None:
        self.adapter = adapter
        self.payload = payload or {"traceEvents": []}
        self.cuda_available = True
        self.fail_cuda = self.fail_stop = self.fail_export = self.fail_range = False
        self.activities: list[list[str]] = []
        self.ranges: list[str] = []
        self.started = self.stopped = 0

    def supported_activities(self) -> set[str]:
        return {"CPU", "CUDA"} if self.cuda_available else {"CPU"}

    def profile(self, *, activities: list[str], **kwargs: Any) -> Any:
        self.activities.append(activities)
        owner = self

        class Session:
            def start(self) -> None:
                owner.started += 1
                owner.adapter.clock.now += 100.0  # Must remain outside window.
                if owner.fail_cuda and "CUDA" in activities:
                    raise RuntimeError("CUPTI unavailable")

            def stop(self) -> None:
                owner.stopped += 1
                owner.adapter.clock.now += 100.0
                if owner.fail_stop:
                    raise RuntimeError("profiler stop failure")

            def export_chrome_trace(self, path: str) -> None:
                owner.adapter.clock.now += 100.0
                if owner.fail_export:
                    raise RuntimeError("profiler export failure")
                Path(path).write_text(json.dumps(owner.payload))

        return Session()

    @contextmanager
    def record_function(self, name: str) -> Any:
        if self.fail_range:
            raise RuntimeError("record function unavailable")
        self.ranges.append(name)
        yield


def _options(adapter: Adapter, delivery: str = "host_ready") -> dict[str, Any]:
    return {
        "num_envs": 1,
        "cameras_per_env": 1,
        "width": 4,
        "height": 3,
        "delivery": delivery,
        "clock": adapter.clock,
    }


def _event(
    name: str,
    category: str,
    ts: float,
    *,
    duration: float = 1,
    args: dict[str, Any] | None = None,
    pid: int = 1,
    tid: int = 2,
) -> dict[str, Any]:
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": ts,
        "dur": duration,
        "args": args or {},
        "pid": pid,
        "tid": tid,
    }


def _native_trace() -> dict[str, Any]:
    return {
        "traceEvents": [
            _event("r09/to_host", "user_annotation", 0, duration=10),
            _event("cudaMemcpyAsync", "cuda_runtime", 2, args={"correlation": 1}),
            _event(
                "Memcpy DtoH (Device -> Host)",
                "gpu_memcpy",
                1000,
                args={"correlation_id": 1, "bytes": 36},
                pid=0,
                tid=7,
            ),
            _event("r09/state_publish", "user_annotation", 20, duration=10),
            _event("cudaMemcpyAsync", "cuda_runtime", 22, args={"correlation": 2}),
            _event(
                "Memcpy HtoD",
                "gpu_memcpy",
                1020,
                args={"correlation": 2, "Bytes": "12"},
                pid=0,
                tid=7,
            ),
            _event("r09/render_device", "user_annotation", 40, duration=10),
            _event("aten::copy", "cpu_op", 42, args={"External id": 3}),
            _event(
                "Memcpy DtoD",
                "gpu_memcpy",
                1040,
                args={"external_id": 3, "size_bytes": 64},
                pid=0,
                tid=7,
            ),
            _event("r09/synchronize", "user_annotation", 60, duration=10),
            _event("cudaMemcpy", "cuda_runtime", 62, args={"correlation": 4}),
            _event(
                "Memcpy DtoH", "gpu_memcpy", 1060, args={"correlation": 4}, pid=0, tid=7
            ),
            # Clock overlap alone cannot assign this event to observation delivery.
            _event("Memcpy unknown kind", "gpu_memcpy", 8, args={"bytes": 5}),
            # Explicit CPU scopes/byte arguments never become GPU memcpy evidence.
            _event(
                "r09/host_copy",
                "user_annotation",
                80,
                duration=5,
                args={"bytes": 90000},
            ),
            _event("memcpy", "cpu_op", 82, args={"bytes": 90000}),
        ]
    }


@pytest.mark.parametrize(
    "delivery,readbacks,copies",
    [
        ("device_ready", 0, 0),
        ("host_ready", 1, 0),
        ("duplicate_d2h", 2, 0),
        ("host_copy", 1, 1),
    ],
)
def test_trace_routes_genuine_readbacks_and_cpu_copies_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delivery: str,
    readbacks: int,
    copies: int,
) -> None:
    adapter = Adapter()
    profiler = Profiler(adapter)
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)
    result = diagnostics.trace_delivery(
        adapter, output=tmp_path, warmup=2, iterations=3, **_options(adapter, delivery)
    )

    assert adapter.events.count("render") == 5
    assert adapter.events.count("d2h") == 5 * readbacks
    assert adapter.events.count("host_copy") == 5 * copies
    counts = result["declared_observation_transfers"]
    assert counts["d2h_bytes"] == 3 * readbacks * 36
    assert counts["host_copy_bytes"] == 3 * copies * 36
    assert result["cpu_range_counts"].get("r09/host_copy", 0) == 3 * copies
    assert profiler.ranges.count("r09/to_host") == 3 * readbacks
    assert result["cpu_range_counts"]["r09/render_device"] == 3
    assert result["measurement"]["metrics"]["window_s"] == pytest.approx(
        3 * (0.03 + 0.03 * readbacks + 0.005 * copies)
    )
    assert result["publication_eligible"] is False
    assert result["resolution_evidence"]["native_resolution_status"] == "not_verified"
    assert (tmp_path / "cpu_ranges.json").is_file()
    assert (tmp_path / "torch_trace.json").is_file()
    assert profiler.started == profiler.stopped == 1


def test_trace_attributes_native_pose_and_observation_transfers_by_launch_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Adapter()
    profiler = Profiler(adapter, _native_trace())
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)

    result = diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))

    observed = result["observed_transfers"]
    assert observed["directions"]["D2H"]["observed_bytes"] == 36
    assert observed["directions"]["D2H"]["observed_event_count"] == 2
    assert observed["directions"]["D2H"]["events_without_byte_count"] == 1
    assert observed["directions"]["H2D"]["observed_bytes"] == 12
    assert observed["directions"]["D2D"]["observed_bytes"] == 64
    assert observed["observed_copy_api_calls"] == 3
    assert observed["attribution_counts"] == {
        "observation_delivery": 1,
        "state_publication": 1,
        "renderer_submission": 1,
        "completion": 1,
        "unattributed": 1,
    }
    assert observed["directions"]["D2H"]["actual_native_transfer_bytes"] is None
    assert observed["native_graphics_visibility"] == "not_established"
    assert result["declared_observation_transfers"]["d2h_bytes"] == 108


def test_cpu_profiler_api_calls_never_become_gpu_kernel_or_transfer_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Adapter()
    profiler = Profiler(
        adapter,
        {
            "traceEvents": [
                _event("cudaMemcpyAsync", "cuda_runtime", 0, args={"bytes": 999})
            ]
        },
    )
    profiler.cuda_available = False
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)

    result = diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))

    assert result["profiler"]["activities"] == ["CPU"]
    assert "cuda_profiler_activity_unavailable" in result["profiler"]["reasons"]
    observed = result["observed_transfers"]
    assert observed["observed_copy_api_calls"] == 1
    assert observed["observed_cuda_kernel_events"] == 0
    assert observed["directions"]["D2H"]["observed_bytes"] is None
    assert observed["absence_means_zero"] is False
    assert result["measurement"]["metrics"]["native_render_s"] is None


def test_cuda_profiler_start_failure_falls_back_without_repeating_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Adapter()
    profiler = Profiler(adapter)
    profiler.fail_cuda = True
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)
    result = diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))
    assert profiler.activities == [["CPU", "CUDA"], ["CPU"]]
    assert result["profiler"]["activities"] == ["CPU"]
    assert adapter.events.count("render") == 3
    assert any(
        "CUPTI unavailable" in reason for reason in result["profiler"]["reasons"]
    )


@pytest.mark.parametrize("delivery", ["device_ready", "duplicate_d2h"])
def test_requested_cuda_with_only_cpu_api_events_is_partial_not_healthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delivery: str
) -> None:
    adapter = Adapter()
    profiler = Profiler(
        adapter,
        {
            "traceEvents": [
                _event("cudaMemcpyAsync", "cuda_runtime", 0, args={"bytes": 36})
            ]
        },
    )
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)
    result = diagnostics.trace_delivery(
        adapter, output=tmp_path, **_options(adapter, delivery)
    )
    assert result["status"] == "completed"
    assert result["measurement"]["counts"]["render_calls"] == 3
    assert result["profiler"]["activities"] == ["CPU", "CUDA"]
    assert result["profiler"]["cuda_trace_status"] == "partial"
    assert "cuda_activity_requested_but_not_observed" in result["profiler"]["reasons"]
    assert result["trace_health"]["observed_cuda_activity_events"] == 0
    assert result["trace_health"]["absence_means_zero"] is False
    if delivery == "duplicate_d2h":
        assert (
            "declared_d2h_without_cuda_transfer_events" in result["profiler"]["reasons"]
        )
        assert result["declared_observation_transfers"]["d2h_calls"] == 6
    assert (
        result["observed_transfers"]["directions"]["D2H"][
            "actual_native_transfer_count"
        ]
        is None
    )
    assert result["cpu_range_counts"]["r09/render_device"] == 3
    assert (tmp_path / "cpu_ranges.json").is_file()


def test_observed_kernel_without_declared_readback_events_is_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Adapter()
    profiler = Profiler(adapter, {"traceEvents": [_event("kernel", "kernel", 0)]})
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)
    result = diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))
    assert result["profiler"]["cuda_trace_status"] == "partial"
    assert (
        "cuda_activity_requested_but_not_observed" not in result["profiler"]["reasons"]
    )
    assert "declared_d2h_without_cuda_transfer_events" in result["profiler"]["reasons"]
    assert result["trace_health"]["observed_cuda_activity_events"] == 1


def test_observed_cuda_activity_does_not_claim_complete_native_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Adapter()
    profiler = Profiler(adapter, _native_trace())
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)
    result = diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))
    assert result["profiler"]["cuda_trace_status"] == "observed"
    assert result["trace_health"]["reasons"] == []
    assert (
        result["observed_transfers"]["native_graphics_visibility"] == "not_established"
    )


@pytest.mark.parametrize(
    "failure", ["fail_stop", "fail_export", "fail_range", "malformed_trace"]
)
def test_optional_profiler_failures_preserve_completed_capture_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    adapter = Adapter()
    profiler = Profiler(adapter)
    if failure == "malformed_trace":
        profiler.payload = {"traceEvents": "bad events"}
    else:
        setattr(profiler, failure, True)
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)

    result = diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))

    assert result["status"] == "completed"
    assert result["measurement"]["counts"]["render_calls"] == 3
    assert result["profiler"]["reasons"]
    assert result["cpu_range_counts"]["r09/render_device"] == 3
    assert (tmp_path / "delivery_diagnostics.json").is_file()


def test_missing_torch_retains_cpu_trace_and_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def absent() -> None:
        raise ModuleNotFoundError("Torch profiler missing")

    monkeypatch.setattr(diagnostics, "_load_profiler", absent)
    adapter = Adapter()
    result = diagnostics.trace_delivery(
        adapter, output=tmp_path, **_options(adapter, "host_copy")
    )
    assert result["profiler"]["activities"] == []
    assert result["cpu_range_counts"]["r09/host_copy"] == 3
    assert result["observed_transfers"]["source"] == "no_native_profiler_trace"
    assert not (tmp_path / "torch_trace.json").exists()


def test_capture_failure_is_preserved_and_not_relabelled_as_profiler_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Adapter()
    profiler = Profiler(adapter)
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)

    def fail(arrays: Any) -> None:
        raise RuntimeError("readback failed")

    adapter.to_host = fail
    with pytest.raises(RuntimeError, match="readback failed"):
        diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))
    result = json.loads((tmp_path / "delivery_diagnostics.json").read_text())
    assert result["status"] == "failed" and result["measurement"] is None
    assert profiler.stopped == 1


@pytest.mark.parametrize(
    "delivery", ["device_ready", "host_ready", "duplicate_d2h", "host_copy"]
)
@pytest.mark.parametrize("diagnostic", [False, True])
def test_overhead_windows_use_identical_delivery_validation_and_sync_policy(
    delivery: str, diagnostic: bool
) -> None:
    adapter = Adapter()
    indices = []

    def prepare(index: int) -> None:
        indices.append(index)
        adapter.clock.now += 0.005

    result = diagnostics.framework_overhead(
        adapter,
        warmup=2,
        iterations=3,
        repeats=3,
        prepare=prepare,
        diagnostic=diagnostic,
        **_options(adapter, delivery),
    )
    assert result["equivalent_policy"]
    assert result["overhead_ratio"] == pytest.approx(1.0)
    assert result["common_loop_overhead_ratio"] == pytest.approx(1.0)
    assert len(result["windows"]) == 9
    assert [window["treatment"] for window in result["windows"][:6]] == [
        "minimal",
        "direct",
        "common",
        "common",
        "direct",
        "minimal",
    ]
    assert indices == [0, 1, 0, 1, 2] * 9
    assert all(
        window["counts"] == result["windows"][0]["counts"]
        for window in result["windows"]
    )
    assert result["target"]["maximum_overhead_fraction"] == 0.03
    assert result["target"]["status"] == "not_assessed"
    assert result["publication_eligible"] is False


def test_minimal_and_shared_capture_baselines_measure_different_harness_costs(
    tmp_path: Path,
) -> None:
    adapter = Adapter(read_cost=0.001)
    result = diagnostics.framework_overhead(
        adapter,
        warmup=0,
        iterations=10,
        repeats=3,
        output=tmp_path / "overhead.json",
        **_options(adapter, "device_ready"),
    )
    assert result["overhead_ratio"] > result["common_loop_overhead_ratio"] > 1
    assert result["overhead_fraction"] > 0.03
    # Even a deterministic test fixture never manufactures a target pass verdict.
    assert result["target"]["status"] == "not_assessed"
    assert (
        result["uncertainty"]["method"]
        == "median_absolute_deviation_of_paired_window_ratios"
    )
    assert (tmp_path / "overhead.json").is_file()


def test_counter_mismatch_suppresses_framework_overhead_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = diagnostics.measure_rendering

    def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        result["counts"]["d2h_group_calls"] += 1
        return result

    monkeypatch.setattr(diagnostics, "measure_rendering", changed)
    adapter = Adapter()
    result = diagnostics.framework_overhead(
        adapter, warmup=0, iterations=1, repeats=3, **_options(adapter)
    )
    assert result["equivalent_policy"] is False
    assert result["overhead_ratio"] is None
    assert result["common_loop_overhead_ratio"] is None


def test_state_ranges_exclude_warmup_and_cover_diagnostic_syncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Adapter()
    profiler = Profiler(adapter)
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: profiler)

    def prepare(index: int) -> None:
        adapter.clock.now += 0.005

    result = diagnostics.trace_delivery(
        adapter,
        output=tmp_path,
        warmup=2,
        iterations=3,
        prepare=prepare,
        diagnostic=True,
        **_options(adapter),
    )
    assert result["cpu_range_counts"]["r09/state_publish"] == 3
    assert result["cpu_range_counts"]["r09/synchronize"] == 9
    assert result["measurement"]["counts"]["synchronize_calls"] == 9


def test_diagnostic_import_and_cpu_trace_work_without_site_packages(
    tmp_path: Path,
) -> None:
    script = """
import importlib.abc
from pathlib import Path
from types import SimpleNamespace
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'embodichain', 'dexsim', 'isaaclab', 'isaacsim'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
from scripts.benchmark.rendering.diagnostics import trace_delivery
class Adapter:
    def render_device(self):
        return {'rgb': SimpleNamespace(shape=(1,1,1,3), dtype='uint8', is_cuda=True)}
    def synchronize(self):
        pass
result = trace_delivery(Adapter(), output=Path(sys.argv[1]), num_envs=1, cameras_per_env=1, width=1, height=1, delivery='device_ready', iterations=1)
assert result['status'] == 'completed'
assert result['observed_transfers']['directions']['D2H']['observed_bytes'] is None
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_diagnostics_require_positive_windows_and_never_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = Adapter()
    with pytest.raises(ValueError, match="three"):
        diagnostics.framework_overhead(
            adapter, warmup=0, iterations=1, repeats=2, **_options(adapter)
        )
    monkeypatch.setattr(diagnostics, "_load_profiler", lambda: Profiler(adapter))
    diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))
    count = adapter.events.count("render")
    with pytest.raises(FileExistsError):
        diagnostics.trace_delivery(adapter, output=tmp_path, **_options(adapter))
    assert adapter.events.count("render") == count
