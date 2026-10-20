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

"""Source-identity accounting and bounded instrumentation for data delivery."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from .payload import payload_bytes

__all__ = ["StreamMetrics", "ResourceMonitor", "write_json", "validate_batch"]


def write_json(path: Path, value: Any) -> None:
    """Write a complete finite JSON artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_batch(
    batch: dict[str, Any],
    *,
    batch_size: int,
    history: int,
    future: int,
    cameras: tuple[str, ...],
    full_images: bool = False,
) -> None:
    """Check causal/source boundaries and bounded image samples on owned data.

    Args:
        batch: Common CPU payload returned by either data path.
        batch_size: Requested context count.
        history: Causal history extent including now.
        future: Prediction extent including now.
        cameras: Required camera UIDs.
        full_images: Inspect every pixel in a separate engineering run. Normal
            instrumented runs check a fixed spatial grid instead.
    """
    length = history + future - 1
    valid, ids = batch["valid"], batch["frame_ids"]
    if valid.shape != (batch_size, length) or valid.dtype != torch.bool:
        raise ValueError("Context validity shape/dtype mismatch")
    if ids.shape != (batch_size, length, 3) or ids.dtype != torch.int64:
        raise ValueError("Context source ID shape/dtype mismatch")
    if not valid[:, history - 1].all():
        raise ValueError("Every sampled current anchor must be valid")
    anchor = ids[:, history - 1]
    offsets = torch.arange(1 - history, future)
    if (ids[~valid] != -1).any():
        raise ValueError("Invalid source padding must be -1")
    if ((ids[..., :2] != anchor[:, None, :2]) & valid[..., None]).any():
        raise ValueError("A context crosses episode/run identity")
    if ((ids[..., 2] != anchor[:, None, 2] + offsets) & valid).any():
        raise ValueError("Context frame indices are not causal and contiguous")
    loss = batch["loss_mask"]
    if not torch.equal(loss, valid & (offsets >= 0)):
        raise ValueError("Loss mask includes history or invalid targets")
    continuity = batch["continuity_id"]
    if ((continuity < 0) & valid).any() or (
        (continuity != continuity[:, history - 1, None]) & valid
    ).any():
        raise ValueError("Context crosses a continuity boundary")
    for name in ("state", "actions"):
        value = batch[name]
        if (
            value.shape[:2] != valid.shape
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
            or (value[~valid] != 0).any()
        ):
            raise ValueError(f"Invalid numeric payload: {name}")
    if set(batch["images"]) != set(cameras):
        raise ValueError("Missing or unexpected camera data")
    for name, image in batch["images"].items():
        if (
            image.shape != (batch_size, length, 3, 256, 256)
            or image.dtype != torch.float32
            or image.device.type != "cpu"
        ):
            raise ValueError(f"Image contract mismatch: {name}")
        sample = image if full_images else image[..., ::32, ::32]
        if (
            not torch.isfinite(sample).all()
            or (sample < 0).any()
            or (sample > 1).any()
            or (sample[~valid] != 0).any()
        ):
            raise ValueError(f"Image range/padding mismatch: {name}")


class StreamMetrics:
    """Count immutable source deliveries, retaining evictions and failed runs."""

    def __init__(self, output: Path, *, history: int, quota: int | None) -> None:
        self.output, self.history, self.quota = output, history, quota
        launch = os.environ.get("ODS_PROCESS_LAUNCH_MONOTONIC_NS")
        self.origin_ns = (
            int(launch)
            if launch is not None
            else int(
                (time.monotonic() - (time.time() - psutil.Process().create_time()))
                * 1e9
            )
        )
        self.origin = self.origin_ns / 1e9
        self.episodes: dict[str, dict[str, Any]] = {}
        self.delivered = 0
        self.window_delivered = 0
        self.post_window_chunks = 0
        self.generations: Counter[int] = Counter()
        self.window_generations: Counter[int] = Counter()
        self.first_sample_by_generation: dict[int, float] = {}
        self.publications: dict[int, dict[str, float]] = {}
        self.latencies: list[float] = []
        self.services: list[float] = []
        self.worker_services: list[float] = []
        self.waits: Counter[str] = Counter()
        self.first_batch: float | None = None
        self.reader_started: float | None = None
        self.measurement_started: float | None = None
        self.last_batch: float | None = None
        self.overlap_generations: set[int] = set()
        self.publication_completed: dict[int, float] = {}
        self.image_samples = 0
        self.bytes = 0
        self._lifecycle = (output / "lifecycle.jsonl").open("w", buffering=1)
        self._batches = (output / "batches.jsonl").open("w", buffering=1)

    def elapsed(self) -> float:
        """Return launch-relative host monotonic seconds."""
        return time.monotonic() - self.origin

    def events(self, events: list[dict[str, Any]]) -> None:
        """Record publications/evictions without keeping their frame arrays."""
        for event in events:
            source = f"{event['run_id']}:{event['episode_id']}"
            at = (event["monotonic_ns"] - self.origin_ns) / 1e9
            record = {**event, "source_id": source, "time_seconds": at}
            self._lifecycle.write(json.dumps(record) + "\n")
            row = self.episodes.setdefault(
                source,
                {
                    "source_id": source,
                    "generation": int(event["generation"]),
                    "delivered_chunks": 0,
                    "first_consumed_seconds": None,
                    "evicted": False,
                },
            )
            if event["kind"] == "published":
                row["published_seconds"] = at
                row["slot"] = event["slot"]
                self.publications.setdefault(int(event["generation"]), {})[source] = at
            else:
                row["evicted"] = True
                row["evicted_seconds"] = at

    def interval(self, kind: str, started: float, ended: float) -> None:
        """Record cooperative flow-control waits using absolute host times."""
        self.waits[kind] += ended - started

    def generation_complete(self, generation: int) -> None:
        """Observe a completed fill without equating it with first publication."""
        self.publication_completed.setdefault(generation, self.elapsed())

    def batch(
        self,
        batch: dict[str, Any],
        *,
        requested_at: float,
        returned_at: float,
        service_seconds: float,
        sampled_at: float | None = None,
        within_window: bool = True,
    ) -> None:
        """Record one successfully returned batch, counting each anchor once."""
        at = returned_at - self.origin
        if self.first_batch is None:
            self.first_batch = at
        self.last_batch = at
        ids = batch["frame_ids"][:, self.history - 1]
        versions = batch["generations"][:, self.history - 1]
        sources = Counter(f"{int(row[0])}:{int(row[1])}" for row in ids)
        for source, count in sources.items():
            index = next(
                i
                for i, row in enumerate(ids)
                if f"{int(row[0])}:{int(row[1])}" == source
            )
            version = int(versions[index])
            row = self.episodes.setdefault(
                source,
                {
                    "source_id": source,
                    "generation": version,
                    "delivered_chunks": 0,
                    "first_consumed_seconds": None,
                    "evicted": False,
                },
            )
            row["delivered_chunks"] += count
            row["generation"] = version
            if row["first_consumed_seconds"] is None:
                row["first_consumed_seconds"] = at
            if self.quota is not None and row["delivered_chunks"] > self.quota:
                raise ValueError(f"Source delivery exceeded K: {source}")
        counts = Counter(map(int, versions.tolist()))
        self.generations.update(counts)
        chunks = len(ids)
        latency = returned_at - requested_at
        valid_images = int(batch["valid"].sum()) * len(batch["images"])
        byte_count = payload_bytes(batch)
        self.delivered += chunks
        if within_window:
            self.window_delivered += chunks
            self.window_generations.update(counts)
            self.image_samples += valid_images
            self.bytes += byte_count
            self.latencies.append(latency)
            self.services.append(service_seconds)
            for generation in counts:
                self.first_sample_by_generation.setdefault(
                    generation, (sampled_at or returned_at) - self.origin
                )
            if "_service" in batch:
                values = batch["_service"]["worker_item_seconds"]
                self.worker_services.append(sum(values) / len(values))
        else:
            self.post_window_chunks += chunks
        self._batches.write(
            json.dumps(
                {
                    "time_seconds": at,
                    "chunks": chunks,
                    "generations": counts,
                    "episode_counts": sources,
                    "request_latency_seconds": latency,
                    "service_seconds": service_seconds,
                    "valid_image_samples": valid_images,
                    "payload_bytes": byte_count,
                    "within_window": within_window,
                    "sampled_seconds": (sampled_at or returned_at) - self.origin,
                }
            )
            + "\n"
        )

    def summary(self, duration: float, *, cache_episodes: int = 0) -> dict[str, Any]:
        """Return measured counters; empty timing populations remain unknown."""

        def quantile(values: list[float], q: float) -> float | None:
            return float(np.quantile(values, q)) if values else None

        self.overlap_generations = {
            generation
            for generation, sampled in self.first_sample_by_generation.items()
            if generation > 1
            and len(self.publications.get(generation, {})) == cache_episodes
            and sampled < max(self.publications[generation].values())
        }
        return {
            "first_batch_from_launch_seconds": self.first_batch,
            "first_batch_from_reader_seconds": (
                self.first_batch - self.reader_started
                if self.first_batch is not None and self.reader_started is not None
                else None
            ),
            "source_ready_seconds": self.reader_started,
            "delivered_chunks": self.delivered,
            "window_delivered_chunks": self.window_delivered,
            "post_window_chunks": self.post_window_chunks,
            "delivered_chunks_per_second": (
                self.window_delivered / duration if duration else None
            ),
            "request_latency_p50_seconds": quantile(self.latencies, 0.5),
            "request_latency_p95_seconds": quantile(self.latencies, 0.95),
            "service_p50_seconds": quantile(self.services, 0.5),
            "service_p95_seconds": quantile(self.services, 0.95),
            "worker_item_service_p50_seconds": quantile(self.worker_services, 0.5),
            "consumed_generations": sorted(self.window_generations),
            "generation_chunk_counts": dict(self.generations),
            "overlap_generations": sorted(self.overlap_generations),
            "data_wait_seconds": self.waits["data_wait"],
            "throttle_seconds": self.waits["usage_throttle"],
            "valid_image_samples": self.image_samples,
            "payload_bytes": self.bytes,
            "logical_payload_bytes_per_second": (
                self.bytes / duration if duration else None
            ),
            "cap_violations": sum(
                self.quota is not None and row["delivered_chunks"] > self.quota
                for row in self.episodes.values()
            ),
            "observation_scope": "in-window complete-batch CPU delivery; late completions retained in lifetime ledger only; no optimizer",
        }

    def close(self) -> None:
        """Close event logs after final worker draining."""
        self._lifecycle.close()
        self._batches.close()


class ResourceMonitor:
    """Sample owned-process PSS and GPU contexts across devices, not exact peaks."""

    def __init__(self, output: Path, interval: float = 2.0) -> None:
        self._output, self._interval = output, interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_pss: int | None = None
        self.peak_vram: int | None = None
        self.primary_device_peak: int | None = None
        self.context_peaks: dict[str, int] = {}
        self.samples = 0
        self.errors: list[str] = []

    def start(self) -> None:
        """Start one bounded-lifetime CPU observer thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        handle = None
        device_handles = []
        try:
            import pynvml

            pynvml.nvmlInit()
            device_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            visible = (
                os.environ.get("ODS_GPU_UUID")
                or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
            )
            if visible.startswith("GPU-"):
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible)
            elif visible.isdigit():
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible))
        except Exception as error:
            self.errors.append(f"NVML unavailable: {error}")
        owner = psutil.Process()
        with (self._output / "resources.jsonl").open("w", buffering=1) as stream:
            while not self._stop.is_set():
                pss = 0
                available = False
                processes = [owner, *owner.children(recursive=True)]
                owned_pids = {process.pid for process in processes}
                for process in processes:
                    try:
                        value = getattr(process.memory_full_info(), "pss", None)
                        if value is not None:
                            available = True
                            pss += value
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                vram = None
                if handle is not None:
                    try:
                        vram = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                    except Exception as error:
                        self.errors.append(str(error))
                        handle = None
                contexts = {}
                context_supported = bool(device_handles)
                for device_handle in device_handles:
                    try:
                        uuid = pynvml.nvmlDeviceGetUUID(device_handle)
                        per_pid = {}
                        total = pynvml.nvmlDeviceGetMemoryInfo(device_handle).total
                        for query in (
                            pynvml.nvmlDeviceGetComputeRunningProcesses,
                            pynvml.nvmlDeviceGetGraphicsRunningProcesses,
                        ):
                            for item in query(device_handle):
                                value = item.usedGpuMemory
                                if (
                                    item.pid in owned_pids
                                    and isinstance(value, int)
                                    and 0 <= value <= total
                                ):
                                    per_pid[item.pid] = max(
                                        per_pid.get(item.pid, 0), value
                                    )
                        if per_pid:
                            contexts[uuid] = sum(per_pid.values())
                    except Exception:
                        context_supported = False
                if context_supported:
                    self.peak_vram = max(self.peak_vram or 0, sum(contexts.values()))
                    for uuid, value in contexts.items():
                        self.context_peaks[uuid] = max(
                            self.context_peaks.get(uuid, 0), value
                        )
                if available:
                    self.peak_pss = max(self.peak_pss or 0, pss)
                if vram is not None:
                    self.primary_device_peak = max(self.primary_device_peak or 0, vram)
                self.samples += 1
                stream.write(
                    json.dumps(
                        {
                            "monotonic": time.monotonic(),
                            "pss_bytes": pss if available else None,
                            "device_used_bytes": vram,
                            "owned_context_bytes": (
                                contexts if context_supported else None
                            ),
                        }
                    )
                    + "\n"
                )
                self._stop.wait(self._interval)

    def close(self) -> dict[str, Any]:
        """Join instrumentation and return explicitly sampled resource maxima."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("Resource observer did not join")
        return {
            "peak_pss_bytes": self.peak_pss,
            "peak_vram_bytes": self.peak_vram,
            "primary_device_sampled_peak_bytes": self.primary_device_peak,
            "owned_device_context_sampled_peaks": self.context_peaks,
            "resource_sample_count": self.samples,
            "resource_errors": self.errors,
            "memory_scope": "sampled owned-process PSS and NVML context memory across devices (max per PID/device before sum); primary device total separately diagnostic; not guaranteed peaks",
        }
