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

"""Compact source-frame accounting and optional PickPlace usage pacing.

Counts use one int64 array per source episode, not one Python tuple per target.
A source identity is (engine run ID, episode ID); its frame index never aliases
another publication. Accounting is CPU-side and consumes the copied provenance.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch

__all__ = [
    "AnchorBudget",
    "CompactFrameSet",
    "UsageTracker",
    "SamplingRNG",
    "retained_deferred_checkpoints",
]


def _arrays(frame_ids: torch.Tensor) -> Iterable[tuple[tuple[int, int], np.ndarray]]:
    """Group provenance using vector operations and at most one loop per episode."""
    if frame_ids.dtype != torch.int64 or frame_ids.shape[-1:] != (3,):
        raise ValueError("Expected int64 frame identities [..., 3]")
    ids = frame_ids.detach().reshape(-1, 3).cpu().numpy()
    if not len(ids):
        return
    if np.any(ids < 0):
        raise ValueError("Padding must be masked before accounting")
    runs = [ids[0, 0]] if np.all(ids[:, 0] == ids[0, 0]) else np.unique(ids[:, 0])
    for run in runs:
        rows = ids if len(runs) == 1 else ids[ids[:, 0] == run]
        width = int(rows[:, 2].max()) + 1
        first, last = int(rows[:, 1].min()), int(rows[:, 1].max())
        # Common sequential ODS episode IDs avoid sorting every target. Cap the
        # dense workspace at 2 MiB before falling back for sparse external IDs.
        if (last - first + 1) * width <= 262_144:
            counts = np.bincount(
                (rows[:, 1] - first) * width + rows[:, 2],
                minlength=(last - first + 1) * width,
            ).reshape(-1, width)
            for offset in np.flatnonzero(np.any(counts, axis=1)):
                yield (int(run), first + int(offset)), counts[offset]
        else:
            episodes, inverse = np.unique(rows[:, 1], return_inverse=True)
            counts = np.bincount(
                inverse * width + rows[:, 2], minlength=len(episodes) * width
            )
            for episode, values in zip(episodes, counts.reshape(len(episodes), width)):
                yield (int(run), int(episode)), values


def _add(storage: dict[tuple[int, int], np.ndarray], ids: torch.Tensor) -> None:
    for key, values in _arrays(ids):
        previous = storage.get(key)
        if previous is None:
            storage[key] = values.copy()
        else:
            if len(previous) < len(values):
                previous = np.pad(previous, (0, len(values) - len(previous)))
                storage[key] = previous
            previous[: len(values)] += values


class CompactFrameSet:
    """Set-compatible cardinality and tensor insertion without target tuples."""

    def __init__(
        self,
        counts: dict[tuple[int, int], np.ndarray] | None = None,
        *,
        accumulate: bool = True,
    ) -> None:
        if not accumulate and counts is None:
            raise ValueError("A nonaccumulating view needs externally owned counts")
        self.counts = {} if counts is None else counts
        self.accumulate = accumulate

    def add_tensor(self, ids: torch.Tensor) -> None:
        # A usage runner has already counted validated targets at completion.
        # Its tracker can share that cardinality view without scanning twice.
        if self.accumulate:
            _add(self.counts, ids)

    def update(self, values: Iterable[tuple[int, ...]]) -> None:
        rows = list(values)
        if rows:
            self.add_tensor(torch.tensor(rows, dtype=torch.int64))

    def __len__(self) -> int:
        return sum(int(np.count_nonzero(values)) for values in self.counts.values())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, set):
            return {
                (*key, int(frame))
                for key, values in self.counts.items()
                for frame in np.flatnonzero(values)
            } == other
        return NotImplemented


class SamplingRNG:
    """Optional CPU sampler stream isolated from model/augmentation randomness.

    The showcase samples only on its main thread. This state scope is not a
    thread-safe replacement for generator-aware sampler APIs.
    """

    def __init__(self, seed: int | None) -> None:
        self.state = (
            None
            if seed is None
            else torch.Generator(device="cpu").manual_seed(seed).get_state()
        )

    @contextmanager
    def scope(self) -> Iterator[None]:
        if self.state is None:
            yield
            return
        previous = torch.get_rng_state()
        torch.set_rng_state(self.state)
        try:
            yield
        finally:
            self.state = torch.get_rng_state()
            torch.set_rng_state(previous)


class AnchorBudget:
    """Global chunk credits earned once per published source, never per sample.

    The cold-start lead-time prior is 30 seconds until an observed request to
    first-publication latency is available. It is a conservative scheduling
    prior, not a measurement of production latency.
    """

    def __init__(
        self,
        reuse: float,
        batch_size: int,
        initial_anchors: int,
        *,
        prior_seconds: float = 30.0,
    ) -> None:
        if (
            not math.isfinite(reuse)
            or reuse <= 0
            or min(batch_size, initial_anchors) < 1
        ):
            raise ValueError("Reuse, batch size and initial anchors must be positive")
        if not math.isfinite(prior_seconds) or prior_seconds <= 0:
            raise ValueError("Lead-time prior must be finite and positive")
        self.reuse = float(reuse)
        self.batch_size = int(batch_size)
        self.capacity = max(float(batch_size), reuse * initial_anchors)
        self.credits = 0.0
        self.completed_chunks = 0
        self.discarded_credits = 0.0
        self._seen: set[tuple[int, int]] = set()
        self._requests: dict[int, float] = {}
        self._latencies: list[float] = []
        self.prior_seconds = prior_seconds
        self.chunks_per_second: float | None = None

    def publish(self, event: dict[str, Any]) -> None:
        if event["kind"] != "published":
            return
        key = (int(event["run_id"]), int(event["episode_id"]))
        if key in self._seen:
            return
        self._seen.add(key)
        addition = self.reuse * int(event["eligible_anchors"])
        if addition < 0:
            raise ValueError("Eligible anchor count cannot be negative")
        self.discarded_credits += max(0.0, self.credits + addition - self.capacity)
        self.credits = min(self.capacity, self.credits + addition)
        generation = int(event["generation"])
        if generation in self._requests:
            self._latencies.append(
                max(0.0, event["monotonic_ns"] / 1e9 - self._requests.pop(generation))
            )

    def request(self, generation: int, monotonic_seconds: float) -> None:
        self._requests.setdefault(generation, monotonic_seconds)

    @property
    def can_update(self) -> bool:
        return self.credits + 1e-9 >= self.batch_size

    def complete(self, chunks: int, compute_seconds: float) -> None:
        if chunks < 1 or chunks > self.credits + 1e-9:
            raise ValueError("A completed update must fit its granted credits")
        self.credits = max(0.0, self.credits - chunks)
        self.completed_chunks += chunks
        self.observe_rate(chunks, compute_seconds)

    def observe_rate(self, chunks: int, active_seconds: float) -> None:
        """Observe service/update/accounting time, excluding intentional waits."""
        if active_seconds > 0:
            rate = chunks / active_seconds
            self.chunks_per_second = (
                rate
                if self.chunks_per_second is None
                else 0.9 * self.chunks_per_second + 0.1 * rate
            )

    @property
    def lead_seconds(self) -> float:
        return 1.25 * (
            float(np.percentile(self._latencies, 95))
            if self._latencies
            else self.prior_seconds
        )

    def should_refill(self) -> bool:
        if not self.can_update:
            return True
        return (
            self.chunks_per_second is not None
            and self.credits / self.chunks_per_second <= self.lead_seconds
        )

    def summary(self) -> dict[str, Any]:
        return {
            "target_anchor_reuse": self.reuse,
            "capacity_chunks": self.capacity,
            "remaining_chunks": self.credits,
            "completed_chunks": self.completed_chunks,
            "discarded_credits": self.discarded_credits,
            "active_chunks_per_second_ema": self.chunks_per_second,
            "first_publication_latencies_seconds": self._latencies,
            "cold_start_prior_seconds": self.prior_seconds,
        }


class UsageTracker:
    """Exact sampled/completed counts with explicit publication denominators.

    Evicted-unused counts describe whether an evicted episode was ever used by
    run end, including already-copied batches completed after eviction. They do
    not claim to reconstruct instantaneous optimizer counts at eviction time.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.episodes: dict[tuple[int, int], dict[str, Any]] = {}
        self.sampled: dict[tuple[int, int], np.ndarray] = {}
        self.anchors: dict[tuple[int, int], np.ndarray] = {}
        self.targets: dict[tuple[int, int], np.ndarray] = {}
        self.history: dict[tuple[int, int], np.ndarray] = {}
        self.first_sample_ns: dict[tuple[int, int], int] = {}
        self.age_histogram: dict[int, int] = {}
        self._pending_ages: list[tuple[tuple[int, int], int, int]] = []

    def lifecycle(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.events.append(event)
            key = (int(event["run_id"]), int(event["episode_id"]))
            if event["kind"] == "published":
                if key in self.episodes:
                    raise ValueError("Repeated source publication identity")
                self.episodes[key] = {**event, "evicted": False}
            elif event["kind"] == "evicted":
                if key not in self.episodes:
                    raise ValueError("Eviction lacks publication event")
                self.episodes[key]["evicted"] = True
                self.episodes[key]["evicted_at_monotonic_ns"] = event["monotonic_ns"]
            else:
                raise ValueError("Unknown lifecycle event")
        pending, self._pending_ages = self._pending_ages, []
        for key, sampled_ns, count in pending:
            self._record_age(key, sampled_ns, count)

    def _record_age(self, key: tuple[int, int], sampled_ns: int, count: int) -> None:
        if key not in self.episodes:
            self._pending_ages.append((key, sampled_ns, count))
            return
        age_ns = sampled_ns - int(self.episodes[key]["monotonic_ns"])
        if age_ns < 0:
            raise ValueError("Sampling precedes source publication")
        bucket = age_ns // 100_000_000
        self.age_histogram[bucket] = self.age_histogram.get(bucket, 0) + count

    @staticmethod
    def anchor_ids(
        batch: dict[str, torch.Tensor], anchor_index: int = 1
    ) -> torch.Tensor:
        ids = batch["frame_ids"]
        return ids if ids.ndim == 2 else ids[:, anchor_index]

    def sample(self, batch: dict[str, torch.Tensor], anchor_index: int = 1) -> None:
        ids = self.anchor_ids(batch, anchor_index)
        _add(self.sampled, ids)
        sampled_ns = time.monotonic_ns()
        # Per-episode operations; never materialize a Python object per target.
        rows = ids.reshape(-1, 3).cpu().numpy()
        for run in np.unique(rows[:, 0]):
            episodes, counts = np.unique(rows[rows[:, 0] == run, 1], return_counts=True)
            for episode, count in zip(episodes, counts):
                key = (int(run), int(episode))
                self.first_sample_ns.setdefault(key, sampled_ns)
                self._record_age(key, sampled_ns, int(count))

    def complete(
        self, batch: dict[str, torch.Tensor], anchor_index: int = 1
    ) -> torch.Tensor:
        ids = batch["frame_ids"]
        _add(self.anchors, self.anchor_ids(batch, anchor_index))
        mask = batch.get("loss_mask")
        supervised_ids = ids if mask is None else ids[mask]
        _add(self.targets, supervised_ids)
        if ids.ndim == 3:
            history = ids[:, : anchor_index + 1]
            valid = batch["valid"][:, : anchor_index + 1]
            _add(self.history, history[valid])
        return supervised_ids

    def summary(self) -> dict[str, Any]:
        observed = (
            set(self.sampled)
            | set(self.anchors)
            | set(self.targets)
            | set(self.history)
        )
        if observed - self.episodes.keys():
            raise ValueError("Observed source identity lacks a lifecycle publication")
        records = []
        for key, event in self.episodes.items():
            frames = np.asarray(event["valid_frame_indices"], dtype=np.int64)
            row = {
                "run_id": key[0],
                "episode_id": key[1],
                "generation": event["generation"],
                "eligible_anchors": int(event["eligible_anchors"]),
                "valid_frames": len(frames),
                "evicted": event["evicted"],
                "published_at_monotonic_ns": event["monotonic_ns"],
                "valid_frame_indices": frames.tolist(),
                "eligible_anchor_indices": event["eligible_anchor_indices"],
            }
            for name, storage in (
                ("sampled_anchor", self.sampled),
                ("completed_anchor", self.anchors),
                ("supervised_frame", self.targets),
                ("history_frame", self.history),
            ):
                values = storage.get(key, np.zeros(0, dtype=np.int64))
                indices = (
                    np.asarray(event["eligible_anchor_indices"], dtype=np.int64)
                    if name.endswith("anchor")
                    else frames
                )
                counts = np.zeros(len(indices), dtype=np.int64)
                in_range = indices < len(values)
                counts[in_range] = values[indices[in_range]]
                row[name + "_counts"] = counts.tolist()
                row[name + "_total"] = int(counts.sum())
                row[name + "_coverage"] = (
                    float(np.count_nonzero(counts) / len(indices))
                    if len(indices)
                    else None
                )
                row[name + "_p50_p95_max"] = (
                    [
                        float(np.percentile(counts, 50)),
                        float(np.percentile(counts, 95)),
                        int(counts.max()),
                    ]
                    if len(counts)
                    else None
                )
            records.append(row)
        denominator = sum(row["eligible_anchors"] for row in records)
        total_chunks = sum(row["completed_anchor_total"] for row in records)
        total_targets = sum(row["supervised_frame_total"] for row in records)
        valid_denominator = sum(row["valid_frames"] for row in records)
        evicted_frames = sum(row["valid_frames"] for row in records if row["evicted"])
        evicted_unused = sum(
            sum(count == 0 for count in row["supervised_frame_counts"])
            for row in records
            if row["evicted"]
        )
        age_quantiles = None
        if self.age_histogram:
            buckets = sorted(self.age_histogram)
            cumulative = np.cumsum([self.age_histogram[bucket] for bucket in buckets])
            age_quantiles = [
                (
                    buckets[
                        int(
                            np.searchsorted(
                                cumulative, quantile * cumulative[-1], side="left"
                            )
                        )
                    ]
                    + 1
                )
                / 10
                for quantile in (0.5, 0.95)
            ]
        first_use = [
            (sampled - self.episodes[key]["monotonic_ns"]) / 1e9
            for key, sampled in self.first_sample_ns.items()
        ]
        return {
            "sample_age_seconds_p50_p95_upper_bins": age_quantiles,
            "sample_age_histogram_bin_seconds": 0.1,
            "first_use_latency_seconds_p50_p95": (
                np.percentile(first_use, [50, 95]).tolist() if first_use else None
            ),
            "episodes": records,
            "lifecycle_events": self.events,
            "eligible_anchors_published": denominator,
            "completed_chunks": total_chunks,
            "supervised_targets": total_targets,
            "anchor_reuse": total_chunks / denominator if denominator else None,
            "supervised_frame_reuse": (
                total_targets / valid_denominator if valid_denominator else None
            ),
            "evicted_unused_at_run_end_fraction": (
                evicted_unused / evicted_frames if evicted_frames else None
            ),
        }


def retained_deferred_checkpoints(
    checkpoints: list[tuple[Path, str]], generation_paths: list[Path]
) -> list[tuple[Path, str]]:
    """Evaluate only the last three generation snapshots, preserving other panels.

    All immutable checkpoint files remain on disk. Fixed-update, initial and
    latest panels are never dropped by this generation-specific cap.
    """
    excluded = set(generation_paths[:-3])
    return [(path, name) for path, name in checkpoints if path not in excluded]
