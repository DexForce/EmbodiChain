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

"""R-09 trace evidence and matched measurement-loop overhead diagnostics.

Diagnostic timings never enter publication throughput. Torch profiling is lazy
and optional; CPU wall ranges are always retained. CUDA activities are a partial
view of native graphics rendering, and absent copy events are not zero transfers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import hashlib
import importlib
import math
import os
from pathlib import Path
import statistics
import threading
from time import perf_counter
from typing import Any

from scripts.benchmark.core.artifacts import read_json, write_json

from .capture import (
    CaptureAdapter,
    CapturePacket,
    capture_packet,
    measure_rendering,
    validate_capture_packet,
)

__all__ = ["trace_delivery", "framework_overhead"]

_SCOPES = {
    "r09/render_device": "renderer_submission",
    "r09/synchronize": "completion",
    "r09/to_host": "observation_delivery",
    "r09/host_copy": "cpu_host_copy_scope",
    "r09/state_publish": "state_publication",
}
_COUNT_FIELDS = (
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


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _policy(
    warmup: int,
    iterations: int,
    num_envs: int,
    cameras_per_env: int,
    width: int,
    height: int,
    delivery: str,
    diagnostic: bool,
) -> dict[str, Any]:
    """Validate the common policy before profiler creation or output writes."""
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
    if delivery not in {"device_ready", "host_ready", "duplicate_d2h", "host_copy"}:
        raise ValueError("Unknown delivery boundary")
    if type(diagnostic) is not bool:
        raise ValueError("diagnostic must be boolean")
    return {
        "warmup": warmup,
        "iterations": iterations,
        "num_envs": num_envs,
        "cameras_per_env": cameras_per_env,
        "width": width,
        "height": height,
        "delivery": delivery,
        "diagnostic": diagnostic,
    }


def _load_profiler() -> Any:
    """Load optional Torch profiling only for an explicitly requested trace."""
    return importlib.import_module("torch.profiler")


def _start_profiler() -> tuple[Any, Any, dict[str, Any]]:
    """Fall back from CUDA profiling to CPU profiling, without replaying captures."""
    evidence: dict[str, Any] = {
        "activities": [],
        "reasons": [],
        "native_graphics_visibility": "not_established",
    }
    try:
        module = _load_profiler()
        cpu, cuda = module.ProfilerActivity.CPU, module.ProfilerActivity.CUDA
    except Exception as error:
        evidence["reasons"].append("torch_profiler_unavailable: " + _error(error))
        return None, None, evidence
    try:
        supported = set(module.supported_activities())
    except Exception as error:
        supported = {cpu}
        evidence["reasons"].append("cuda_activity_probe_failed: " + _error(error))
    choices = [[cpu, cuda], [cpu]] if cuda in supported else [[cpu]]
    if cuda not in supported:
        evidence["reasons"].append("cuda_profiler_activity_unavailable")
    for activities in choices:
        session = None
        try:
            session = module.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            session.start()
            evidence["activities"] = ["CPU", "CUDA"] if cuda in activities else ["CPU"]
            return session, module.record_function, evidence
        except Exception as error:
            evidence["reasons"].append("profiler_start_failed: " + _error(error))
            if session is not None:
                try:
                    session.stop()
                except Exception as cleanup_error:
                    evidence["reasons"].append(
                        "profiler_cleanup_failed: " + _error(cleanup_error)
                    )
    return None, None, evidence


class _Ranges:
    """Pair optional record_function annotations with a standalone CPU trace."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.active = False
        self.record_function: Any = None
        self.events: list[dict[str, Any]] = []
        self.errors: list[str] = []

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        if not self.active:
            yield
            return
        start = self.clock()
        context = None
        if self.record_function:
            try:
                context = self.record_function(name)
                context.__enter__()
            except Exception as error:
                self.errors.append("record_function_start_failed: " + _error(error))
                context = self.record_function = None
        try:
            yield
        finally:
            if context is not None:
                try:
                    context.__exit__(None, None, None)
                except Exception as error:
                    self.errors.append("record_function_stop_failed: " + _error(error))
            self.events.append(
                {
                    "name": name,
                    "cat": "cpu_wall_range",
                    "ph": "X",
                    "ts": start * 1e6,
                    "dur": (self.clock() - start) * 1e6,
                    "pid": os.getpid(),
                    "tid": threading.get_ident(),
                    "args": {"scope": "CPU callback wall time; not GPU kernel time"},
                }
            )


class _TraceAdapter:
    """Annotate callbacks and CPU copies without substituting a host copy for D2H."""

    def __init__(self, adapter: CaptureAdapter, ranges: _Ranges, delivery: str) -> None:
        self.adapter, self.ranges = adapter, ranges
        self.host_type: Any = None
        if delivery == "host_copy":
            import numpy as np

            class HostBuffer(np.ndarray):
                def __array_finalize__(self, source: Any) -> None:
                    self.source = getattr(source, "source", None)

                def copy(self, order: str = "C") -> Any:
                    with ranges.scope("r09/host_copy"):
                        return self.source.copy(order=order)

            self.host_type = HostBuffer

    def render_device(self) -> Mapping[str, Any]:
        with self.ranges.scope("r09/render_device"):
            return self.adapter.render_device()

    def synchronize(self) -> None:
        with self.ranges.scope("r09/synchronize"):
            self.adapter.synchronize()

    def to_host(self, arrays: Mapping[str, Any]) -> Mapping[str, Any]:
        with self.ranges.scope("r09/to_host"):
            host = self.adapter.to_host(arrays)
        if self.host_type is None:
            return host
        wrapped = {}
        for name, array in host.items():
            wrapped[name] = array.view(self.host_type)
            wrapped[name].source = array
        return wrapped


def _key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _arguments(event: dict[str, Any]) -> dict[str, Any]:
    return event["args"] if isinstance(event.get("args"), dict) else {}


def _identities(event: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        ("externalid" if _key(name) == "externalid" else "correlation", str(value))
        for name, value in _arguments(event).items()
        if _key(name) in {"correlation", "correlationid", "externalid"}
        and type(value) in (int, str)
    }


def _direction(event: dict[str, Any]) -> str:
    args = _arguments(event)
    value = _key(
        " ".join(
            [
                str(event.get("name", "")),
                *(
                    str(value)
                    for name, value in args.items()
                    if _key(name) in {"copykind", "memcpykind", "kind", "direction"}
                ),
            ]
        )
    )
    for direction, tokens in (
        ("D2H", ("dtoh", "d2h", "devicetohost")),
        ("H2D", ("htod", "h2d", "hosttodevice")),
        ("D2D", ("dtod", "d2d", "devicetodevice")),
    ):
        if any(token in value for token in tokens):
            return direction
    return "unknown"


def _event_bytes(event: dict[str, Any]) -> int | None:
    for key, value in _arguments(event).items():
        if _key(key) not in {
            "bytes",
            "sizebytes",
            "transferbytes",
            "numbytes",
            "bytecount",
        }:
            continue
        if isinstance(value, str) and value.isdecimal():
            value = int(value)
        if type(value) is int and value >= 0:
            return value
    return None


def _attribution(
    event: dict[str, Any], launches: list[dict[str, Any]], scopes: list[dict[str, Any]]
) -> str:
    """Use matched CPU launch identities and enclosing ranges, never GPU overlap."""
    identities = _identities(event)
    matched = [launch for launch in launches if identities & _identities(launch)]
    owners = set()
    for launch in matched:
        contained = [
            scope
            for scope in scopes
            if scope.get("pid") == launch.get("pid")
            and scope.get("tid") == launch.get("tid")
            and _number(scope.get("ts"))
            and _number(scope.get("dur"))
            and _number(launch.get("ts"))
            and scope["ts"] <= launch["ts"] <= scope["ts"] + scope["dur"]
        ]
        if contained:
            owner = min(contained, key=lambda scope: scope["dur"])
            owners.add(_SCOPES[owner["name"]])
    return next(iter(owners)) if len(owners) == 1 else "unattributed"


def _parse_transfers(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Distinguish observed CUDA copies from API calls and unknown native work."""
    if payload is not None and not isinstance(payload, dict):
        raise ValueError("Chrome trace must be a JSON object")
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else []
    if not isinstance(events, list):
        raise ValueError("Chrome traceEvents must be a list")
    events = [
        event for event in events if isinstance(event, dict) and event.get("ph") == "X"
    ]
    scopes = [event for event in events if event.get("name") in _SCOPES]
    gpu = [
        event
        for event in events
        if str(event.get("cat", "")).lower() in {"gpu_memcpy", "cuda_memcpy"}
    ]
    launches = [
        event
        for event in events
        if event not in gpu
        and str(event.get("cat", "")).lower()
        in {"cuda_runtime", "cuda_driver", "cpu_op"}
    ]
    copies = [
        {
            "name": str(event.get("name", "")),
            "direction": _direction(event),
            "bytes": _event_bytes(event),
            "attribution": _attribution(event, launches, scopes),
            "trace_timestamp_us": event.get("ts"),
            "trace_duration_us": event.get("dur"),
            "args": event.get("args", {}),
        }
        for event in gpu
    ]
    directions = {}
    for direction in ("D2H", "H2D", "D2D", "unknown"):
        selected = [event for event in copies if event["direction"] == direction]
        values = [event["bytes"] for event in selected if event["bytes"] is not None]
        directions[direction] = {
            "observed_event_count": len(selected),
            "observed_bytes": sum(values) if values else None,
            "events_without_byte_count": len(selected) - len(values),
            "byte_scope": "sum over explicitly emitted event byte fields; lower bound when fields are missing",
            "actual_native_transfer_count": None,
            "actual_native_transfer_bytes": None,
        }
    api_calls = [
        event
        for event in launches
        if str(event.get("cat", "")).lower() in {"cuda_runtime", "cuda_driver"}
        and "memcpy" in str(event.get("name", "")).lower()
    ]
    return {
        "source": (
            "torch_profiler_chrome_trace"
            if payload is not None
            else "no_native_profiler_trace"
        ),
        "directions": directions,
        "events": copies,
        "attribution_counts": dict(Counter(event["attribution"] for event in copies)),
        "observed_copy_api_calls": len(api_calls),
        "copy_api_scope": "CUDA runtime/driver API records; not evidence of completed GPU copies",
        "observed_cuda_kernel_events": sum(
            str(event.get("cat", "")).lower() in {"kernel", "gpu_kernel"}
            for event in events
        ),
        "observed_cuda_activity_events": sum(
            str(event.get("cat", "")).lower()
            in {
                "kernel",
                "gpu_kernel",
                "gpu_memcpy",
                "cuda_memcpy",
                "gpu_memset",
                "cuda_memset",
            }
            for event in events
        ),
        "native_graphics_visibility": "not_established",
        "absence_means_zero": False,
        "limitations": [
            "CUDA activity collection does not establish full Vulkan/RTX/native graphics visibility.",
            "CPU callback and CUDA API durations are not GPU kernel durations.",
            "Missing copy events or byte fields remain unknown; they are never native transfer totals.",
            "Only correlated CPU launches inside named scopes are attributed; temporal overlap alone is insufficient.",
        ],
    }


def _trace_health(
    profiler: dict[str, Any],
    observed: dict[str, Any],
    measurement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep successful profiler startup separate from observed CUDA activity."""
    requested = "CUDA" in profiler["activities"]
    activities = observed["observed_cuda_activity_events"]
    declared_d2h = (measurement or {}).get("counts", {}).get("d2h_calls", 0)
    observed_d2h = observed["directions"]["D2H"]["observed_event_count"]
    reasons = []
    if requested and not activities:
        reasons.append("cuda_activity_requested_but_not_observed")
    if requested and declared_d2h and not observed_d2h:
        reasons.append("declared_d2h_without_cuda_transfer_events")
    status = (
        "partial"
        if reasons
        else "observed" if requested and activities else "unavailable"
    )
    profiler["cuda_trace_status"] = status
    profiler["reasons"].extend(reasons)
    return {
        "status": status,
        "cuda_requested": requested,
        "observed_cuda_activity_events": activities,
        "declared_observation_d2h_calls": declared_d2h,
        "observed_cuda_d2h_events": observed_d2h,
        "reasons": reasons,
        "scope": "activity-presence check, not proof of complete native graphics visibility",
        "absence_means_zero": False,
    }


def _artifact(path: Path, kind: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": kind,
    }


def trace_delivery(
    adapter: CaptureAdapter,
    *,
    output: Path,
    num_envs: int,
    cameras_per_env: int,
    width: int,
    height: int,
    delivery: str,
    warmup: int = 0,
    iterations: int = 3,
    prepare: Callable[[int], None] | None = None,
    diagnostic: bool = False,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Trace the declared delivery boundary separately from publication timing.

    Args:
        adapter: Already initialized backend; this function does not create it.
        output: New evidence directory, or a directory without diagnostic files.
        num_envs: Environments in the batch.
        cameras_per_env: Cameras in each environment.
        width: Output width in pixels.
        height: Output height in pixels.
        delivery: The common capture delivery policy.
        warmup: Untimed, unprofiled warmup captures.
        iterations: Small measured diagnostic window; defaults to three.
        prepare: Optional state publication using the same frame-index contract.
        diagnostic: Add the same explicit phase synchronizations as capture.
        clock: Monotonic seconds source, injectable for CPU-only tests.

    Returns:
        Measurement and CPU/profiler evidence, declared observation transfers,
        observed native/pose copies and visibility limits. Absence never proves
        zero native transfers. Trace setup and export are outside the timer.
    """
    policy = _policy(
        warmup,
        iterations,
        num_envs,
        cameras_per_env,
        width,
        height,
        delivery,
        diagnostic,
    )
    output = Path(output).resolve()
    if any(
        (output / name).exists()
        for name in ("delivery_diagnostics.json", "cpu_ranges.json", "torch_trace.json")
    ):
        raise FileExistsError("Refusing to overwrite diagnostic evidence")
    output.mkdir(parents=True, exist_ok=True)
    ranges = _Ranges(clock)
    traced = _TraceAdapter(adapter, ranges, delivery)
    session = None
    profiler: dict[str, Any] = {
        "activities": [],
        "reasons": ["measurement_not_started"],
        "native_graphics_visibility": "not_established",
    }

    def start() -> None:
        nonlocal session, profiler
        session, ranges.record_function, profiler = _start_profiler()
        ranges.active = True

    def publish(index: int) -> None:
        with ranges.scope("r09/state_publish"):
            prepare(index)

    measurement = None
    error: BaseException | None = None
    artifacts = []
    trace = None
    try:
        measurement = measure_rendering(
            traced,
            **policy,
            prepare=publish if prepare else None,
            clock=clock,
            on_measurement_start=start,
        )
    except BaseException as caught:
        error = caught
    finally:
        ranges.active = False
        if session is not None:
            try:
                session.stop()
            except Exception as caught:
                profiler["reasons"].append("profiler_stop_failed: " + _error(caught))
            try:
                path = output / "torch_trace.json"
                session.export_chrome_trace(str(path))
                artifacts.append(_artifact(path, "torch_profiler_trace"))
                trace = read_json(path)
            except Exception as caught:
                profiler["reasons"].append("profiler_export_failed: " + _error(caught))
        path = write_json(
            output / "cpu_ranges.json",
            {
                "traceEvents": ranges.events,
                "displayTimeUnit": "ms",
                "clock": "perf_counter",
                "scope": "CPU callback wall ranges only; no GPU kernels",
            },
        )
        artifacts.append(_artifact(path, "cpu_wall_ranges"))
    try:
        observed = _parse_transfers(trace)
    except Exception as caught:
        profiler["reasons"].append("profiler_parse_failed: " + _error(caught))
        observed = _parse_transfers(None)
    profiler["reasons"].extend(ranges.errors)
    trace_health = _trace_health(profiler, observed, measurement)
    metadata = getattr(adapter, "metadata", {})
    result = {
        "schema_version": 1,
        "kind": "R-09_delivery_trace",
        "status": "failed" if error else "completed",
        "error": _error(error) if error else None,
        "publication_eligible": False,
        "treatment": "instrumented diagnostic; do not pool with publication throughput",
        "policy": {**policy, "state_publication": prepare is not None},
        "measurement": measurement,
        "profiler": profiler,
        "trace_health": trace_health,
        "declared_observation_transfers": (
            measurement["counts"] if measurement else None
        ),
        "observed_transfers": observed,
        "cpu_range_counts": dict(Counter(event["name"] for event in ranges.events)),
        "resolution_evidence": (
            metadata.get("resolution_evidence")
            if isinstance(metadata, Mapping)
            else None
        ),
        "artifacts": artifacts,
        "reconciliation": "Declared counters cover observation to_host calls and explicit CPU copies. Observed trace events can additionally include native renderer, state/pose, completion and unattributed transfers; these populations are not interchangeable.",
    }
    write_json(output / "delivery_diagnostics.json", result)
    if error is not None:
        raise error
    return result


def _counts(raw: list[dict[str, Any]]) -> dict[str, Any]:
    result = {field: sum(row[field] for row in raw) for field in _COUNT_FIELDS}
    for field in ("d2h_calls_by_modality", "host_copy_calls_by_modality"):
        names = sorted({name for row in raw for name in row[field]})
        result[field] = {
            name: sum(row[field].get(name, 0) for row in raw) for name in names
        }
    return result


def _direct_window(
    adapter: CaptureAdapter,
    policy: dict[str, Any],
    prepare: Callable[[int], None] | None,
    clock: Callable[[], float],
) -> dict[str, Any]:
    """Direct loop with the same capture, validation and metadata work."""
    specs = None

    def capture(index: int) -> Any:
        return capture_packet(
            adapter,
            frame_index=index,
            exposure_count=policy["num_envs"] * policy["cameras_per_env"],
            delivery=policy["delivery"],
            prepare=prepare,
            diagnostic=policy["diagnostic"],
            clock=clock,
        )

    def validate(packet: Any) -> None:
        nonlocal specs
        current = validate_capture_packet(
            packet,
            **{
                name: policy[name]
                for name in ("num_envs", "cameras_per_env", "width", "height")
            },
        )
        if specs is not None and current != specs:
            raise ValueError(
                "Capture modalities or array specifications changed within the direct window"
            )
        specs = current

    for index in range(policy["warmup"]):
        packet = capture(index)
        validate(packet)
        packet.to_metadata()
        del packet
    raw = []
    start = clock()
    for index in range(policy["iterations"]):
        packet = capture(index)
        validate(packet)
        raw.append(packet.to_metadata())
        del packet
    window = clock() - start
    if not math.isfinite(window) or window <= 0:
        raise ValueError("Direct measurement window must be positive and finite")
    return {"window_s": window, "counts": _counts(raw), "specs": specs, "raw": raw}


def _minimal_capture(
    adapter: CaptureAdapter,
    policy: dict[str, Any],
    prepare: Callable[[int], None] | None,
    index: int,
) -> CapturePacket:
    """Perform equivalent adapter calls, with only the envelope validation needs."""
    syncs = 0
    if prepare is not None:
        prepare(index)
        if policy["diagnostic"]:
            adapter.synchronize()
            syncs += 1
    device = adapter.render_device()
    if not isinstance(device, Mapping) or not device:
        raise ValueError("render_device must return a nonempty modality mapping")
    if policy["diagnostic"]:
        adapter.synchronize()
        syncs += 1
    batches = []
    copies = {}
    delivered = device
    delivery = policy["delivery"]
    if delivery != "device_ready":
        for _ in range(2 if delivery == "duplicate_d2h" else 1):
            host = adapter.to_host(device)
            if not isinstance(host, Mapping):
                raise ValueError("to_host must return a modality mapping")
            batches.append(dict(host))
        delivered = batches[-1]
        if delivery == "host_copy":
            copies = {name: array.copy() for name, array in delivered.items()}
            delivered = copies
    if not policy["diagnostic"] or delivery != "device_ready":
        adapter.synchronize()
        syncs += 1
    # These unused timestamp fields are placeholders for the validation envelope,
    # not measurements. Minimal windows never publish per-capture timing records.
    return CapturePacket(
        frame_index=index,
        delivery=delivery,
        device_arrays=device,
        arrays=delivered,
        d2h_batches=tuple(batches),
        host_copy_arrays=copies,
        exposure_count=policy["num_envs"] * policy["cameras_per_env"],
        synchronize_calls=syncs,
        state_publish_calls=int(prepare is not None),
        started_s=0.0,
        completed_s=0.0,
        timestamps_s={},
        phases_s={},
    )


def _minimal_window(
    adapter: CaptureAdapter,
    policy: dict[str, Any],
    prepare: Callable[[int], None] | None,
    clock: Callable[[], float],
) -> dict[str, Any]:
    """Time direct calls and mandatory validation, without optional telemetry."""
    specs = None

    def capture_and_validate(index: int) -> None:
        nonlocal specs
        packet = _minimal_capture(adapter, policy, prepare, index)
        current = validate_capture_packet(
            packet,
            **{
                name: policy[name]
                for name in ("num_envs", "cameras_per_env", "width", "height")
            },
        )
        if specs is not None and current != specs:
            raise ValueError(
                "Capture modalities or array specifications changed within the minimal window"
            )
        specs = current
        # Do not retain arrays across captures: memory lifetime matches common.
        del packet

    for index in range(policy["warmup"]):
        capture_and_validate(index)
    start = clock()
    for index in range(policy["iterations"]):
        capture_and_validate(index)
    window = clock() - start
    if not math.isfinite(window) or window <= 0:
        raise ValueError("Minimal measurement window must be positive and finite")
    iterations, delivery = policy["iterations"], policy["delivery"]
    groups = {"device_ready": 0, "host_ready": 1, "duplicate_d2h": 2, "host_copy": 1}[
        delivery
    ]
    copies = int(delivery == "host_copy")
    syncs = (
        1
        + int(policy["diagnostic"] and prepare is not None)
        + int(policy["diagnostic"] and delivery != "device_ready")
    )
    batch_bytes = sum(spec["nbytes"] for spec in specs.values())
    counts = {
        "batch_steps": iterations,
        "render_calls": iterations,
        "exposure_count": iterations * policy["num_envs"] * policy["cameras_per_env"],
        "state_publish_calls": iterations * int(prepare is not None),
        "synchronize_calls": iterations * syncs,
        "d2h_group_calls": iterations * groups,
        "d2h_calls": iterations * groups * len(specs),
        "d2h_bytes": iterations * groups * batch_bytes,
        "host_copy_group_calls": iterations * copies,
        "host_copy_calls": iterations * copies * len(specs),
        "host_copy_bytes": iterations * copies * batch_bytes,
        "d2h_calls_by_modality": {name: iterations * groups for name in specs},
        "host_copy_calls_by_modality": {name: iterations * copies for name in specs},
    }
    return {
        "window_s": window,
        "counts": counts,
        "specs": specs,
        "raw": [],
        "count_basis": "direct loop invocation contract and validated byte shapes; no per-capture telemetry in this baseline",
    }


def framework_overhead(
    adapter: CaptureAdapter,
    *,
    warmup: int,
    iterations: int,
    num_envs: int,
    cameras_per_env: int,
    width: int,
    height: int,
    delivery: str,
    repeats: int = 3,
    prepare: Callable[[int], None] | None = None,
    diagnostic: bool = False,
    output: Path | None = None,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Compare direct and common loops with identical capture and validation work.

    Args:
        adapter: Already initialized backend, reused by both loop treatments.
        warmup: Untimed captures before every independently timed window.
        iterations: Captures in each window; individual frames are not repeats.
        num_envs: Environments in the batch.
        cameras_per_env: Cameras per environment.
        width: Image width.
        height: Image height.
        delivery: Identical capture delivery policy for both treatments.
        repeats: At least three window triples, alternating treatment order.
        prepare: Identical state-publication callback; indices reset each window.
        diagnostic: Identical phase-synchronization policy in both loops.
        output: Optional new JSON artifact, written after all measurements.
        clock: Monotonic seconds source, injectable for CPU-only tests.

    Returns:
        Raw windows and paired common/minimal ratios/MAD estimate harness cost
        above direct adapter calls with identical mandatory validation. A second
        common/shared-capture ratio isolates loop dispatch/latency bookkeeping.
        The 3% target is recorded without an automatic pass verdict. Renderer
        initialization and the shared mandatory validation are excluded costs.
    """
    policy = _policy(
        warmup,
        iterations,
        num_envs,
        cameras_per_env,
        width,
        height,
        delivery,
        diagnostic,
    )
    if type(repeats) is not int or repeats < 3:
        raise ValueError("At least three paired window repeats are required")
    if output is not None and Path(output).exists():
        raise FileExistsError(output)
    windows, pairs = [], []
    for repeat in range(repeats):
        measured = {}
        order = (
            ("minimal", "direct", "common")
            if repeat % 2 == 0
            else ("common", "direct", "minimal")
        )
        for index, treatment in enumerate(order):
            if treatment == "minimal":
                result = _minimal_window(adapter, policy, prepare, clock)
            elif treatment == "direct":
                result = _direct_window(adapter, policy, prepare, clock)
            else:
                result = measure_rendering(
                    adapter, **policy, prepare=prepare, clock=clock
                )
                result = {
                    "window_s": result["metrics"]["window_s"],
                    "counts": result["counts"],
                    "specs": result["array_specs"],
                    "raw": result["raw"],
                }
            measured[treatment] = result
            windows.append(
                {"repeat": repeat, "order": index, "treatment": treatment, **result}
            )
        equivalent = all(
            measured[name]["counts"] == measured["common"]["counts"]
            and measured[name]["specs"] == measured["common"]["specs"]
            for name in ("minimal", "direct")
        )
        pairs.append(
            {
                "repeat": repeat,
                "equivalent_counts_and_specs": equivalent,
                "common_over_direct": (
                    measured["common"]["window_s"] / measured["direct"]["window_s"]
                    if equivalent
                    else None
                ),
                "common_over_minimal": (
                    measured["common"]["window_s"] / measured["minimal"]["window_s"]
                    if equivalent
                    else None
                ),
            }
        )
    values = [
        pair["common_over_direct"]
        for pair in pairs
        if pair["common_over_direct"] is not None
    ]
    eligible = len(values) == repeats
    loop_ratio = statistics.median(values) if eligible else None
    harness_values = [
        pair["common_over_minimal"]
        for pair in pairs
        if pair["common_over_minimal"] is not None
    ]
    ratio = statistics.median(harness_values) if eligible else None
    result = {
        "schema_version": 1,
        "kind": "framework_overhead",
        "publication_eligible": False,
        "policy": {
            **policy,
            "state_publication": prepare is not None,
            "validation": "same validate_capture_packet and cross-capture spec checks; inside every window",
            "metadata": "common/direct record packet metadata; minimal omits optional per-capture telemetry",
        },
        "windows": windows,
        "pairs": pairs,
        "paired_windows": repeats,
        "equivalent_policy": eligible,
        "comparison_reasons": (
            [] if eligible else ["capture_counts_or_array_specs_mismatch"]
        ),
        "overhead_ratio": ratio,
        "overhead_fraction": ratio - 1 if ratio is not None else None,
        "ratio_definition": "median of paired common_window_s / minimal_window_s",
        "uncertainty": {
            "method": "median_absolute_deviation_of_paired_window_ratios",
            "value": (
                statistics.median(abs(value - ratio) for value in harness_values)
                if ratio is not None
                else None
            ),
        },
        "common_loop_overhead_ratio": loop_ratio,
        "common_loop_ratio_definition": "common / direct shared capture+validation+metadata loop",
        "common_loop_ratio_mad": (
            statistics.median(abs(value - loop_ratio) for value in values)
            if loop_ratio is not None
            else None
        ),
        "target": {
            "maximum_overhead_fraction": 0.03,
            "applies_to": "overhead_fraction",
            "status": "not_assessed",
            "reason": "A preset target, not a hardcoded pass or independent-process performance claim",
        },
        "population": "separately timed in-process windows; not frames or fresh-process repeats",
        "excluded": [
            "adapter/renderer initialization",
            "each window warmup",
            "profiler instrumentation",
            "JSON writeout",
        ],
        "scope": "harness timing/telemetry above direct adapter calls plus identical mandatory validation; minimal retains a small validation envelope but no per-frame timing/metadata",
    }
    if output is not None:
        write_json(Path(output), result)
    return result
