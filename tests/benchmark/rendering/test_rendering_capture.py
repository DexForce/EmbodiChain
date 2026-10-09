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

"""CPU-only proofs of completed capture boundaries and actual copy routing."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any
import weakref

import numpy as np
import pytest

from scripts.benchmark.rendering.capture import measure_rendering


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class HostArray(np.ndarray):
    """Track host memcpy separately from the fake device's D2H operation."""

    adapter: FakeAdapter

    def __new__(
        cls, shape: tuple[int, ...], dtype: str, adapter: FakeAdapter
    ) -> HostArray:
        result = np.zeros(shape, dtype=dtype).view(cls)
        result.adapter = adapter
        adapter.host_references.append(weakref.ref(result))
        return result

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        self.adapter = getattr(source, "adapter", None)

    def copy(self, order: str = "C") -> HostArray:
        self.adapter.host_copies += 1
        self.adapter.events.append(("host_copy", self.adapter.clock.now))
        self.adapter.clock.now += 0.01
        return super().copy(order=order)


class FakeDeviceArray:
    is_cuda = True

    def __init__(
        self, shape: tuple[int, ...], dtype: str, adapter: FakeAdapter
    ) -> None:
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.adapter = adapter

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise AssertionError("Device buffers must not be implicitly read or converted")

    def to_host(self) -> HostArray:
        self.adapter.d2h_calls += 1
        self.adapter.events.append(("d2h", self.adapter.clock.now))
        self.adapter.clock.now += 0.02
        return HostArray(self.shape, str(self.dtype), self.adapter)


class FakeAdapter:
    def __init__(self, *, batch: int = 6, channels: bool = False) -> None:
        self.clock = FakeClock()
        self.events: list[tuple[str, float]] = []
        self.render_calls = 0
        self.d2h_calls = 0
        self.host_copies = 0
        self.sync_calls = 0
        self.to_host_inputs = []
        self.host_references = []
        image_shape = (batch, 4, 5)
        scalar_shape = image_shape + (1,) if channels else image_shape
        self.device_arrays = {
            "rgb": FakeDeviceArray(image_shape + (3,), "uint8", self),
            "depth": FakeDeviceArray(scalar_shape, "float32", self),
            "seg": FakeDeviceArray(scalar_shape, "int32", self),
        }

    def render_device(self) -> dict[str, FakeDeviceArray]:
        self.render_calls += 1
        self.events.append(("render", self.clock.now))
        self.clock.now += 0.04
        # Identical buffers and pixels are valid for a freshly rendered static scene.
        return self.device_arrays

    def synchronize(self) -> None:
        self.sync_calls += 1
        self.events.append(("sync", self.clock.now))
        self.clock.now += 0.03

    def to_host(self, arrays: dict[str, FakeDeviceArray]) -> dict[str, HostArray]:
        self.to_host_inputs.append(arrays)
        return {name: array.to_host() for name, array in arrays.items()}


def measure(adapter: FakeAdapter, **kwargs: Any) -> dict[str, Any]:
    options = {
        "warmup": 0,
        "iterations": 3,
        "num_envs": 2,
        "cameras_per_env": 3,
        "width": 5,
        "height": 4,
        "delivery": "host_ready",
        "clock": adapter.clock,
    }
    options.update(kwargs)
    return measure_rendering(adapter, **options)


@pytest.mark.parametrize(
    "delivery, d2h_groups, host_groups",
    [
        ("device_ready", 0, 0),
        ("host_ready", 1, 0),
        ("duplicate_d2h", 2, 0),
        ("host_copy", 1, 1),
    ],
)
def test_real_d2h_and_cpu_copy_routes_have_separate_counts(
    delivery: str, d2h_groups: int, host_groups: int
) -> None:
    adapter = FakeAdapter()
    result = measure(adapter, delivery=delivery)
    counts = result["counts"]
    # Six exposures, each carrying RGB uint8 + depth float32 + seg int32.
    bytes_per_batch = 6 * 4 * 5 * (3 + 4 + 4)
    assert counts["render_calls"] == adapter.render_calls == 3
    assert counts["exposure_count"] == 3 * 6
    assert counts["d2h_group_calls"] == 3 * d2h_groups
    assert counts["d2h_calls"] == adapter.d2h_calls == 3 * d2h_groups * 3
    assert counts["host_copy_group_calls"] == 3 * host_groups
    assert counts["host_copy_calls"] == adapter.host_copies == 3 * host_groups * 3
    assert counts["synchronize_calls"] == adapter.sync_calls == 3
    assert counts["d2h_calls_by_modality"] == {
        name: 3 * d2h_groups for name in ("rgb", "depth", "seg")
    }
    assert result["metrics"]["d2h_bytes_per_capture"] == bytes_per_batch * d2h_groups
    assert (
        result["metrics"]["host_copy_bytes_per_capture"]
        == bytes_per_batch * host_groups
    )
    assert all(source is adapter.device_arrays for source in adapter.to_host_inputs)


def test_duplicate_d2h_never_uses_the_first_host_delivery_as_its_source() -> None:
    adapter = FakeAdapter()
    result = measure(adapter, delivery="duplicate_d2h", iterations=1)
    assert len(adapter.to_host_inputs) == 2
    assert (
        adapter.to_host_inputs[0] is adapter.to_host_inputs[1] is adapter.device_arrays
    )
    assert adapter.d2h_calls == 6
    assert adapter.host_copies == 0
    assert result["raw"][0]["d2h_group_calls"] == 2


def test_rate_denominator_counts_exposures_without_modality_multiplier() -> None:
    adapter = FakeAdapter()
    result = measure(adapter)
    metrics = result["metrics"]
    # Each call: 40 ms renderer callback + three 20 ms D2Hs + 30 ms completion.
    expected_latency = 0.13
    assert metrics["window_s"] == pytest.approx(expected_latency * 3)
    assert metrics["batch_steps_per_s"] == pytest.approx(1 / expected_latency)
    assert metrics["camera_frames_per_s"] == pytest.approx(6 / expected_latency)
    assert metrics["gpixel_per_s"] == pytest.approx(6 * 4 * 5 / expected_latency / 1e9)
    assert metrics["latency_p50_s"] == pytest.approx(expected_latency)
    assert metrics["latency_p95_s"] == pytest.approx(expected_latency)
    assert metrics["boundary"] == "render_to_host_ready"
    assert metrics["completion"] == "blocking_host_delivery_and_device_sync"
    assert metrics["native_render_s"] is None
    assert (
        result["metric_reasons"]["native_render_s"]
        == "native_backend_timer_unavailable"
    )


def test_warmup_counts_and_memory_start_hook_are_outside_measured_window() -> None:
    adapter = FakeAdapter()
    hook_render_counts = []

    def start() -> None:
        hook_render_counts.append(adapter.render_calls)
        adapter.clock.now += 10  # Sampler/peak-reset overhead must stay excluded.

    result = measure(adapter, warmup=2, on_measurement_start=start)
    assert hook_render_counts == [2]
    assert adapter.render_calls == 5
    assert result["counts"]["render_calls"] == 3
    assert result["warmup_counts"]["render_calls"] == 2
    assert result["counts"]["d2h_calls"] == 9
    assert result["warmup_counts"]["d2h_calls"] == 6
    assert len(result["raw"]) == 3
    assert len(result["warmup_raw"]) == 2
    assert result["metrics"]["window_s"] == pytest.approx(0.39)
    assert result["raw"][0]["started_s"] >= 10


def test_temporal_publication_precedes_each_render_and_is_in_the_boundary() -> None:
    adapter = FakeAdapter()
    indices = []

    def prepare(index: int) -> None:
        indices.append(index)
        adapter.events.append(("prepare", adapter.clock.now))
        adapter.clock.now += 0.05

    result = measure(adapter, warmup=2, prepare=prepare, delivery="device_ready")
    assert indices == [0, 1, 0, 1, 2]
    assert [name for name, _ in adapter.events] == ["prepare", "render", "sync"] * 5
    assert result["metrics"]["boundary"] == "state_publish_to_device_ready"
    assert result["metrics"]["completion"] == "selected_device_synchronized"
    assert result["metrics"]["latency_p50_s"] == pytest.approx(0.12)
    assert result["metrics"]["window_s"] == pytest.approx(0.36)
    assert result["counts"]["state_publish_calls"] == 3
    for row in result["raw"]:
        assert row["timestamps_s"]["state_publish_started_s"] == row["started_s"]
        assert row["completed_s"] - row["started_s"] == pytest.approx(0.12)


def test_completed_buffers_are_released_before_memory_hook_and_next_render() -> None:
    adapter = FakeAdapter()
    original = adapter.render_device

    def check_released() -> None:
        assert all(reference() is None for reference in adapter.host_references)

    def render() -> dict[str, FakeDeviceArray]:
        check_released()
        return original()

    adapter.render_device = render
    measure(adapter, warmup=2, on_measurement_start=check_released)
    check_released()


def test_diagnostic_phases_are_separate_and_extra_synchronization_is_visible() -> None:
    adapter = FakeAdapter()

    def prepare(index: int) -> None:
        adapter.clock.now += 0.05

    result = measure(adapter, prepare=prepare, diagnostic=True, iterations=1)
    metrics = result["metrics"]
    assert metrics["diagnostic"] is True
    assert result["counts"]["synchronize_calls"] == 3
    assert metrics["state_publish_p50_s"] == pytest.approx(0.08)
    assert metrics["device_ready_p50_s"] == pytest.approx(0.07)
    assert metrics["host_ready_p50_s"] == pytest.approx(0.09)
    assert metrics["latency_p50_s"] == pytest.approx(0.24)
    assert metrics["native_render_s"] is None
    row = result["raw"][0]
    assert (
        row["timestamps_s"]["state_publish_completed_s"]
        <= row["timestamps_s"]["render_started_s"]
    )
    assert (
        row["timestamps_s"]["device_ready_s"]
        <= row["timestamps_s"]["host_delivery_started_s"]
    )


def test_performance_mode_does_not_invent_diagnostic_phase_times() -> None:
    adapter = FakeAdapter()
    result = measure(adapter, iterations=1)
    for name in ("state_publish_p50_s", "device_ready_p50_s", "host_ready_p50_s"):
        assert result["metrics"][name] is None
        assert result["metric_reasons"][name] == "diagnostic_disabled"
    assert adapter.sync_calls == 1


@pytest.mark.parametrize("channels", [False, True])
def test_canonical_scalar_shapes_and_identical_static_pixels_are_valid(
    channels: bool,
) -> None:
    result = measure(FakeAdapter(channels=channels))
    assert result["counts"]["exposure_count"] == 18
    assert result["array_specs"]["depth"]["shape"] == (
        [6, 4, 5, 1] if channels else [6, 4, 5]
    )


@pytest.mark.parametrize(
    "modality, shape, dtype",
    [
        ("rgb", (5, 4, 5, 3), "uint8"),
        ("rgb", (6, 4, 5, 4), "uint8"),
        ("rgb", (6, 4, 5, 3), "float32"),
        ("depth", (6, 4, 5, 3), "float32"),
        ("depth", (6, 4, 5), "int32"),
        ("seg", (6, 5, 4), "int32"),
        ("seg", (6, 4, 5), "float32"),
    ],
)
def test_every_modality_requires_canonical_batch_shape_and_dtype(
    modality: str, shape: tuple[int, ...], dtype: str
) -> None:
    adapter = FakeAdapter()
    adapter.device_arrays[modality] = FakeDeviceArray(shape, dtype, adapter)
    with pytest.raises(ValueError, match="batch shape/dtype"):
        measure(adapter, delivery="device_ready")


def test_host_arrays_cannot_be_mislabeled_as_device_ready() -> None:
    adapter = FakeAdapter()
    adapter.device_arrays["rgb"] = np.zeros((6, 4, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="original CUDA device"):
        measure(adapter, delivery="device_ready")


@pytest.mark.parametrize("mutation", ["drop", "dtype"])
def test_to_host_cannot_drop_a_modality_or_change_dtype(mutation: str) -> None:
    adapter = FakeAdapter()
    original = adapter.to_host

    def missing(arrays: dict[str, Any]) -> dict[str, Any]:
        result = original(arrays)
        if mutation == "drop":
            result.pop("seg")
        else:
            result["seg"] = result["seg"].astype(np.float32)
        return result

    adapter.to_host = missing
    with pytest.raises(ValueError, match="modality set|canonical array shape/dtype"):
        measure(adapter)


def test_invalid_depth_infinity_is_valid_canonical_delivery() -> None:
    adapter = FakeAdapter()
    original = adapter.to_host

    def with_invalid_depth(arrays: dict[str, Any]) -> dict[str, Any]:
        result = original(arrays)
        result["depth"].fill(np.inf)
        return result

    adapter.to_host = with_invalid_depth
    result = measure(adapter)
    assert result["counts"]["batch_steps"] == 3


def test_cached_host_delivery_is_rejected_for_duplicate_d2h() -> None:
    adapter = FakeAdapter()
    cached = adapter.to_host(adapter.device_arrays)
    adapter.to_host = lambda arrays: cached
    with pytest.raises(ValueError, match="independent host buffers"):
        measure(adapter, delivery="duplicate_d2h")


def test_modality_set_cannot_change_between_completed_captures() -> None:
    adapter = FakeAdapter()
    original = adapter.render_device

    def render() -> dict[str, FakeDeviceArray]:
        result = original()
        return result if adapter.render_calls == 1 else {"rgb": result["rgb"]}

    adapter.render_device = render
    with pytest.raises(ValueError, match="changed within the run"):
        measure(adapter, delivery="device_ready")


def test_module_import_is_standard_library_only() -> None:
    command = [
        sys.executable,
        "-S",
        "-c",
        "import sys; from scripts.benchmark.rendering.capture import measure_rendering; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules",
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[3],
    )
