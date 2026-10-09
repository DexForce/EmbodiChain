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

"""Completed pure-render capture boundaries with explicit transfer accounting.

Adapters own render submission, renderer/CUDA interoperability, and one blocking
D2H per modality in ``to_host``. This module never initializes CUDA or interprets
identical pixels as stale. Callback latency and whole-window throughput describe
the selected application boundary, never native GPU kernel time. Diagnostic
phase measurements add synchronization and are a distinct timing treatment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import statistics
from time import perf_counter
from typing import Any, Protocol

from scripts.benchmark.core.measurement import measure_loop

__all__ = [
    "CaptureAdapter",
    "CapturePacket",
    "capture_packet",
    "measure_rendering",
    "validate_capture_packet",
]

_DELIVERIES = {"device_ready", "host_ready", "duplicate_d2h", "host_copy"}
_DTYPES = {"rgb": "uint8", "depth": "float32", "seg": "int32", "normals": "float32"}
_ITEM_SIZES = {"uint8": 1, "float32": 4, "int32": 4}


class CaptureAdapter(Protocol):
    """Backend callbacks; each ``to_host`` invocation performs actual D2H.

    ``synchronize`` completes renderer interoperability and the selected CUDA
    device, including work initiated by state publication and image delivery.
    """

    def render_device(self) -> Mapping[str, Any]:
        """Submit a fresh exposure.

        Returns:
            Original device arrays whose submission may still be asynchronous.
        """
        ...

    def synchronize(self) -> None:
        """Wait for the selected device and renderer's interop completion."""
        ...

    def to_host(self, arrays: Mapping[str, Any]) -> Mapping[str, Any]:
        """Copy every original device array once into completed host storage.

        Args:
            arrays: Original device arrays returned by ``render_device``.

        Returns:
            One completed NumPy array per modality, without a second CPU copy.
        """
        ...


def _array_spec(array: Any) -> dict[str, Any]:
    shape = getattr(array, "shape", None)
    dtype = str(getattr(array, "dtype", "")).removeprefix("torch.")
    if shape is None or dtype not in _ITEM_SIZES:
        raise ValueError("Capture array needs a supported shape and canonical dtype")
    shape = tuple(int(size) for size in shape)
    return {
        "shape": list(shape),
        "dtype": dtype,
        "nbytes": math.prod(shape) * _ITEM_SIZES[dtype],
    }


@dataclass(frozen=True)
class CapturePacket:
    """One completed batch with original buffers and observed callback counts.

    ``d2h_batches`` retains each distinct host delivery, preventing the second
    GPU transfer from being silently replaced by a copy of the first host array.
    Array payloads are excluded from :meth:`to_metadata` and benchmark JSON.
    """

    frame_index: int
    delivery: str
    device_arrays: Mapping[str, Any]
    arrays: Mapping[str, Any]
    d2h_batches: tuple[Mapping[str, Any], ...]
    host_copy_arrays: Mapping[str, Any]
    exposure_count: int
    synchronize_calls: int
    state_publish_calls: int
    started_s: float
    completed_s: float
    timestamps_s: Mapping[str, float | None]
    phases_s: Mapping[str, float | None]

    def to_metadata(self) -> dict[str, Any]:
        """Return timing and transfer evidence without reading array contents.

        Returns:
            JSON-compatible counters, timestamps, and incremental phase times.
        """
        d2h_by_modality = {
            name: sum(name in batch for batch in self.d2h_batches)
            for name in self.device_arrays
        }
        copies_by_modality = {
            name: int(name in self.host_copy_arrays) for name in self.device_arrays
        }
        return {
            "frame_index": self.frame_index,
            "delivery": self.delivery,
            "started_s": self.started_s,
            "completed_s": self.completed_s,
            "timestamps_s": dict(self.timestamps_s),
            "phases_s": dict(self.phases_s),
            "batch_steps": 1,
            "render_calls": 1,
            "exposure_count": self.exposure_count,
            "state_publish_calls": self.state_publish_calls,
            "synchronize_calls": self.synchronize_calls,
            "d2h_group_calls": len(self.d2h_batches),
            "d2h_calls": sum(d2h_by_modality.values()),
            "d2h_calls_by_modality": d2h_by_modality,
            "d2h_bytes": sum(
                _array_spec(array)["nbytes"]
                for batch in self.d2h_batches
                for array in batch.values()
            ),
            "host_copy_group_calls": int(bool(self.host_copy_arrays)),
            "host_copy_calls": len(self.host_copy_arrays),
            "host_copy_calls_by_modality": copies_by_modality,
            "host_copy_bytes": sum(
                _array_spec(array)["nbytes"] for array in self.host_copy_arrays.values()
            ),
        }


def capture_packet(
    adapter: CaptureAdapter,
    *,
    frame_index: int,
    exposure_count: int,
    delivery: str,
    prepare: Callable[[int], None] | None = None,
    diagnostic: bool = False,
    clock: Callable[[], float] = perf_counter,
) -> CapturePacket:
    """Capture one batch and complete the requested delivery boundary.

    Args:
        adapter: Backend implementing the renderer and transfer callbacks.
        frame_index: Frame passed to optional frozen-state publication.
        exposure_count: Environments multiplied by cameras per environment.
        delivery: ``device_ready``, ``host_ready``, ``duplicate_d2h``, or ``host_copy``.
        prepare: State-publication callback, executed before render submission.
        diagnostic: Synchronize each phase for separate diagnostic durations.
        clock: Monotonic seconds source, injectable for CPU-only contract tests.

    Returns:
        Completed packet. Call :func:`validate_capture_packet` outside its timed
        operation to check batch shape, dtype, device origin, and host storage.
    """
    if delivery not in _DELIVERIES:
        raise ValueError(f"Unsupported capture delivery: {delivery}")
    started = clock()
    sync_calls = 0
    timestamps: dict[str, float | None] = {
        "state_publish_started_s": None,
        "state_publish_completed_s": None,
        "render_started_s": None,
        "device_ready_s": None,
        "host_delivery_started_s": None,
        "host_delivery_completed_s": None,
    }
    phases: dict[str, float | None] = {
        "state_publish_s": None,
        "device_ready_s": None,
        "host_ready_s": None,
    }
    if prepare is not None:
        timestamps["state_publish_started_s"] = started
        prepare(frame_index)
        if diagnostic:
            adapter.synchronize()
            sync_calls += 1
            timestamps["state_publish_completed_s"] = clock()
            phases["state_publish_s"] = (
                timestamps["state_publish_completed_s"] - started
            )
    render_started = clock() if diagnostic else None
    timestamps["render_started_s"] = render_started
    device = adapter.render_device()
    if not isinstance(device, Mapping) or not device:
        raise ValueError("render_device must return a nonempty modality mapping")
    if diagnostic:
        adapter.synchronize()
        sync_calls += 1
        timestamps["device_ready_s"] = clock()
        phases["device_ready_s"] = timestamps["device_ready_s"] - render_started
    host_batches = []
    host_copies: Mapping[str, Any] = {}
    delivered = device
    if delivery != "device_ready":
        host_started = clock() if diagnostic else None
        timestamps["host_delivery_started_s"] = host_started
        for _ in range(2 if delivery == "duplicate_d2h" else 1):
            # Every readback gets the ORIGINAL device mapping. Shallowly retain
            # mappings so a backend cannot overwrite the first delivery record.
            host = adapter.to_host(device)
            if not isinstance(host, Mapping):
                raise ValueError("to_host must return a modality mapping")
            host_batches.append(dict(host))
        delivered = host_batches[-1]
        if delivery == "host_copy":
            host_copies = {name: array.copy() for name, array in delivered.items()}
            delivered = host_copies
    # Performance mode has one explicit completion boundary. A diagnostic
    # device-only capture has already completed at its render-phase boundary.
    if not diagnostic or delivery != "device_ready":
        adapter.synchronize()
        sync_calls += 1
    completed = clock()
    if diagnostic and delivery != "device_ready":
        timestamps["host_delivery_completed_s"] = completed
        phases["host_ready_s"] = completed - timestamps["host_delivery_started_s"]
    for elapsed in [completed - started, *phases.values()]:
        if elapsed is not None and (not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("Capture durations must be finite and nonnegative")
    return CapturePacket(
        frame_index=frame_index,
        delivery=delivery,
        device_arrays=device,
        arrays=delivered,
        d2h_batches=tuple(host_batches),
        host_copy_arrays=host_copies,
        exposure_count=exposure_count,
        synchronize_calls=sync_calls,
        state_publish_calls=int(prepare is not None),
        started_s=started,
        completed_s=completed,
        timestamps_s=timestamps,
        phases_s=phases,
    )


def validate_capture_packet(
    packet: CapturePacket,
    *,
    num_envs: int,
    cameras_per_env: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Validate every modality without reading device pixels or initializing CUDA.

    Args:
        packet: Completed packet from :func:`capture_packet`.
        num_envs: Expected replicated environment count.
        cameras_per_env: Expected cameras in every environment.
        width: Expected image width.
        height: Expected image height.

    Returns:
        Canonical modality specifications, suitable for JSON provenance.
    """
    if not isinstance(packet, CapturePacket) or packet.delivery not in _DELIVERIES:
        raise ValueError("Expected a supported CapturePacket")
    batch_size = num_envs * cameras_per_env
    if packet.exposure_count != batch_size:
        raise ValueError("Capture exposure count differs from the camera batch")
    specs = {}
    for name, array in packet.device_arrays.items():
        if name not in _DTYPES:
            raise ValueError(f"Unsupported capture modality: {name}")
        spec = _array_spec(array)
        expected = (batch_size, height, width)
        valid_shapes = (
            (expected + (3,),)
            if name in {"rgb", "normals"}
            else (expected, expected + (1,))
        )
        if tuple(spec["shape"]) not in valid_shapes or spec["dtype"] != _DTYPES[name]:
            raise ValueError(
                f"Invalid {name} batch shape/dtype: {spec}; expected {valid_shapes} {_DTYPES[name]}"
            )
        is_device = (
            bool(getattr(array, "is_cuda", False))
            or str(getattr(array, "device", "")).startswith("cuda")
            or hasattr(array, "__cuda_array_interface__")
        )
        if not is_device:
            raise ValueError("render_device must expose original CUDA device arrays")
        specs[name] = spec
    if not specs:
        raise ValueError("Capture requires at least one modality")
    expected_groups = {
        "device_ready": 0,
        "host_ready": 1,
        "duplicate_d2h": 2,
        "host_copy": 1,
    }[packet.delivery]
    if len(packet.d2h_batches) != expected_groups:
        raise ValueError("Capture D2H group count does not match its delivery")
    if packet.delivery == "device_ready":
        if packet.arrays is not packet.device_arrays or packet.host_copy_arrays:
            raise ValueError("Device-ready delivery cannot substitute host arrays")
        return specs
    import numpy as np

    for batch in (*packet.d2h_batches, packet.arrays):
        if set(batch) != set(specs):
            raise ValueError("Host delivery changed the modality set")
        for name, array in batch.items():
            if not isinstance(array, np.ndarray) or _array_spec(array) != specs[name]:
                raise ValueError(
                    f"Host {name} must preserve the canonical array shape/dtype"
                )
    if packet.delivery in {"duplicate_d2h", "host_copy"}:
        for name, first in packet.d2h_batches[0].items():
            if np.shares_memory(first, packet.arrays[name]):
                raise ValueError(
                    "Repeated delivery must provide independent host buffers"
                )
    if bool(packet.host_copy_arrays) != (packet.delivery == "host_copy"):
        raise ValueError("Host-copy evidence does not match the delivery")
    return specs


def _sum_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "batch_steps",
        "render_calls",
        "exposure_count",
        "state_publish_calls",
        "synchronize_calls",
        "d2h_group_calls",
        "d2h_calls",
        "d2h_bytes",
        "host_copy_group_calls",
        "host_copy_calls",
        "host_copy_bytes",
    )
    result = {name: sum(row[name] for row in rows) for name in names}
    for field in ("d2h_calls_by_modality", "host_copy_calls_by_modality"):
        modalities = sorted({name for row in rows for name in row[field]})
        result[field] = {
            name: sum(row[field].get(name, 0) for row in rows) for name in modalities
        }
    return result


def measure_rendering(
    adapter: CaptureAdapter,
    *,
    warmup: int,
    iterations: int,
    num_envs: int,
    cameras_per_env: int,
    width: int,
    height: int,
    delivery: str,
    prepare: Callable[[int], None] | None = None,
    diagnostic: bool = False,
    clock: Callable[[], float] = perf_counter,
    on_measurement_start: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Measure completed render batches using the shared whole-window timer.

    Args:
        adapter: Backend with fresh render, synchronization, and D2H callbacks.
        warmup: Untimed captures, excluded from measured counters and throughput.
        iterations: Number of measured captures.
        num_envs: Environments represented in each complete batch.
        cameras_per_env: Cameras in each environment, independent of modalities.
        width: Image width in pixels.
        height: Image height in pixels.
        delivery: Requested device/host/duplicate-D2H/host-copy boundary.
        prepare: Publish frozen state before each capture. Frame indices start
            at zero independently for warmup and measurement.
        diagnostic: Enable synchronized phase timing; separate from performance.
        clock: Monotonic seconds source.
        on_measurement_start: Called after warmup, before the throughput clock;
            useful for resetting GPU peak counters and resource-sampler phases.

    Returns:
        Metrics, raw measured records, excluded warmup records, transfer counters,
        canonical array specifications, and explicit missing-metric reasons.
        Validation is outside batch latency and inside the throughput window.
    """
    for name, value, minimum in (
        ("warmup", warmup, 0),
        ("iterations", iterations, 1),
        ("num_envs", num_envs, 1),
        ("cameras_per_env", cameras_per_env, 1),
        ("width", width, 1),
        ("height", height, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if delivery not in _DELIVERIES:
        raise ValueError(f"Unsupported capture delivery: {delivery}")
    exposure_count = num_envs * cameras_per_env
    specs: dict[str, Any] | None = None
    raw: list[dict[str, Any]] = []
    warmup_raw = []

    def capture(index: int) -> CapturePacket:
        return capture_packet(
            adapter,
            frame_index=index,
            exposure_count=exposure_count,
            delivery=delivery,
            prepare=prepare,
            diagnostic=diagnostic,
            clock=clock,
        )

    def validate(packet: CapturePacket) -> None:
        nonlocal specs
        current = validate_capture_packet(
            packet,
            num_envs=num_envs,
            cameras_per_env=cameras_per_env,
            width=width,
            height=height,
        )
        if specs is not None and current != specs:
            raise ValueError(
                "Capture modalities or array specifications changed within the run"
            )
        specs = current

    for index in range(warmup):
        packet = capture(index)
        validate(packet)
        warmup_raw.append(packet.to_metadata())
        del packet
    if on_measurement_start is not None:
        on_measurement_start()
    frame_index = 0
    pending_packet: CapturePacket | None = None

    def operation() -> None:
        nonlocal frame_index, pending_packet
        pending_packet = capture(frame_index)
        frame_index += 1

    def record(_: None) -> None:
        nonlocal pending_packet
        if pending_packet is None:
            raise RuntimeError("Capture operation did not provide a completed packet")
        validate(pending_packet)
        raw.append(pending_packet.to_metadata())
        # The shared timer retains its operation return until the next call.
        # Return None and release this separate holder so previous GPU/host
        # arrays do not inflate the next capture's memory footprint.
        pending_packet = None

    timing = measure_loop(
        operation, warmup=0, iterations=iterations, validate=record, clock=clock
    )
    for row, latency in zip(raw, timing.latencies_s, strict=True):
        row["latency_s"] = latency
    counts = _sum_counts(raw)
    metric_reasons = {"native_render_s": "native_backend_timer_unavailable"}
    phase_metrics = {}
    for phase in ("state_publish", "device_ready", "host_ready"):
        name = phase + "_p50_s"
        values = [
            row["phases_s"][phase + "_s"]
            for row in raw
            if row["phases_s"][phase + "_s"] is not None
        ]
        phase_metrics[name] = statistics.median(values) if values else None
        if not values:
            metric_reasons[name] = (
                "diagnostic_disabled" if not diagnostic else "phase_not_applicable"
            )
    batch_rate = timing.operations_per_s
    camera_rate = batch_rate * exposure_count
    metrics = {
        "boundary": ("state_publish_to_" if prepare is not None else "render_to_")
        + delivery,
        "completion": (
            "selected_device_synchronized"
            if delivery == "device_ready"
            else "blocking_host_delivery_and_device_sync"
        ),
        "diagnostic": diagnostic,
        "num_envs": num_envs,
        "cameras_per_env": cameras_per_env,
        "width": width,
        "height": height,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "window_s": timing.window_s,
        "batch_steps_per_s": batch_rate,
        "camera_frames_per_s": camera_rate,
        "gpixel_per_s": camera_rate * width * height / 1e9,
        "latency_p50_s": timing.p50_s,
        "latency_p95_s": timing.p95_s,
        **phase_metrics,
        "d2h_bytes_per_capture": counts["d2h_bytes"] / iterations,
        "host_copy_bytes_per_capture": counts["host_copy_bytes"] / iterations,
        "native_render_s": None,
        "phase_semantics": "incremental synchronized stages; diagnostic treatment only",
        "throughput_window": "all measured captures plus validation and record bookkeeping",
        "copy_count_basis": "one blocking D2H per modality per adapter.to_host invocation",
        "freshness_evidence": "one render_device invocation per batch; identical pixels are valid for a static scene",
    }
    return {
        "metrics": metrics,
        "metric_reasons": metric_reasons,
        "raw": raw,
        "counts": counts,
        "warmup_raw": warmup_raw,
        "warmup_counts": _sum_counts(warmup_raw),
        "array_specs": specs,
    }
