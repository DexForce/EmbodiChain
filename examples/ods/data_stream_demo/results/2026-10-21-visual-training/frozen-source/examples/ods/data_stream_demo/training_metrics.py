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

"""Exact completed-training diversity and timing for the visual ODS experiment.

Publication and delivery alone never increase the training coverage counters.
The owner calls :meth:`TrainingMetrics.complete_update` only after its device
has completed a real optimizer update. Images are neither inspected nor kept
by this recorder; compact per-source frame counters retain exact lifetimes.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .metrics import write_json

__all__ = ["TrainingMetrics"]

_INITIAL_XY_LOW = np.array([-0.46, -0.12], dtype=np.float64)
_INITIAL_XY_HIGH = np.array([-0.38, -0.04], dtype=np.float64)
_INITIAL_BIN_METERS = 0.01
_VOXEL_METERS = 0.02


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cpu_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    return value.detach().numpy()


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    quantiles = np.percentile(values, [50, 95])
    return {"p50": float(quantiles[0]), "p95": float(quantiles[1])}


@dataclass
class _Episode:
    source_id: str
    ordinal: int
    generation: int
    initial_xy: tuple[float, float]
    initial_bin: tuple[int, int] | None
    published_ns: int
    frame_count: int
    accepted_ns: int | None
    metadata: dict[str, Any]
    anchors: np.ndarray = field(init=False)
    history: np.ndarray = field(init=False)
    targets: np.ndarray = field(init=False)
    trained_chunks: int = 0
    first_completed_ns: int | None = None
    retired_ns: int | None = None

    def __post_init__(self) -> None:
        self.anchors = np.zeros(self.frame_count, dtype=np.int64)
        self.history = np.zeros(self.frame_count, dtype=np.int64)
        self.targets = np.zeros(self.frame_count, dtype=np.int64)


class TrainingMetrics:
    """Record completed work, exact sample reuse and task-state coverage.

    Args:
        output: Run directory for launch-relative JSONL and summary sidecars.
        origin_ns: Supervisor's host-monotonic timestamp before launch.
        history: Number of causal observations, including the anchor.
        future: Number of supervised actions, including the anchor.
        batch_size: Exact number of chunks in every optimizer update.
        reuse_cap: Per-source lifetime training upper bound, or ``None`` for
            the reusable offline baseline. Delivery is checked separately
            against the sampler's ledger by the experiment owner.

    A source is identified by the first two columns of ``frame_ids`` as the
    string ``"run_id:episode_id"``. Coverage grids use arena coordinates;
    these privileged measurements never become policy inputs.
    """

    def __init__(
        self,
        output: Path,
        *,
        origin_ns: int,
        history: int,
        future: int,
        batch_size: int,
        reuse_cap: int | None,
    ) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.origin_ns = _positive_int(origin_ns, "origin_ns")
        self.history = _positive_int(history, "history")
        self.future = _positive_int(future, "future")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.reuse_cap = (
            None if reuse_cap is None else _positive_int(reuse_cap, "reuse_cap")
        )
        self.episodes: dict[str, _Episode] = {}
        self.updates = 0
        self.chunks = 0
        self._initial_bins: Counter[tuple[int, int]] = Counter()
        self._voxels: set[tuple[int, int, int]] = set()
        self._curve: list[dict[str, Any]] = []
        self._intervals: list[dict[str, Any]] = []
        self._last_losses: dict[str, float] = {}
        self._last_completed_ns = origin_ns
        self._events = (self.output / "training-events.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )
        self._closed = False

    def _seconds(self, at_ns: int) -> float:
        if type(at_ns) is not int or at_ns < self.origin_ns:
            raise ValueError("Timestamp must use the supervisor monotonic clock")
        return (at_ns - self.origin_ns) / 1e9

    def event(self, kind: str, at_ns: int | None = None, **metadata: Any) -> None:
        """Append a finite JSON event using the common host monotonic clock."""
        if self._closed:
            raise RuntimeError("Training metrics are closed")
        if not kind:
            raise ValueError("An event needs a name")
        timestamp = time.monotonic_ns() if at_ns is None else at_ns
        row = {
            **metadata,
            "kind": kind,
            "monotonic_ns": timestamp,
            "time_seconds": self._seconds(timestamp),
        }
        self._events.write(json.dumps(row, allow_nan=False) + "\n")

    def register_episode(
        self,
        source_id: str,
        *,
        ordinal: int,
        generation: int,
        initial_xy: Sequence[float],
        published_ns: int,
        frame_count: int,
        accepted_ns: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register one accepted publication without counting it as training."""
        pieces = source_id.split(":")
        if len(pieces) != 2 or any(int(part) < 0 for part in pieces):
            raise ValueError("source_id must be a nonnegative run:episode identity")
        if source_id in self.episodes:
            raise ValueError(f"Duplicate immutable source identity: {source_id}")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        if type(generation) is not int or generation < 1:
            raise ValueError("generation must be a positive integer")
        count = _positive_int(frame_count, "frame_count")
        self._seconds(published_ns)
        if accepted_ns is not None:
            self._seconds(accepted_ns)
            if accepted_ns > published_ns:
                raise ValueError("Acceptance cannot follow source publication")
        xy = np.asarray(initial_xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("initial_xy must contain two finite arena coordinates")
        in_bounds = bool(
            ((xy >= _INITIAL_XY_LOW - 1e-7) & (xy <= _INITIAL_XY_HIGH + 1e-7)).all()
        )
        coordinates = (xy - _INITIAL_XY_LOW) / _INITIAL_BIN_METERS
        nearest = np.rint(coordinates)
        # Float32 task states can straddle an exact centimeter boundary by a
        # few nanometers. Snap only that numerical error, never physical bins.
        coordinates = np.where(
            np.abs(coordinates - nearest) <= 1e-5, nearest, coordinates
        )
        bins = np.minimum(7, np.maximum(0, np.floor(coordinates))).astype(np.int64)
        initial_bin = tuple(int(v) for v in bins) if in_bounds else None
        copied_metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        episode = _Episode(
            source_id=source_id,
            ordinal=ordinal,
            generation=generation,
            initial_xy=(float(xy[0]), float(xy[1])),
            initial_bin=initial_bin,
            published_ns=published_ns,
            frame_count=count,
            accepted_ns=accepted_ns,
            metadata=copied_metadata,
        )
        self.event(
            "episode_published",
            published_ns,
            source_id=source_id,
            ordinal=ordinal,
            generation=generation,
            initial_xy=episode.initial_xy,
            initial_xy_bin=initial_bin,
            frame_count=count,
            accepted_ns=accepted_ns,
            metadata=copied_metadata,
        )
        self.episodes[source_id] = episode

    def retire_episode(self, source_id: str, *, retired_ns: int) -> None:
        """Record retirement while preserving a source's cumulative counters."""
        episode = self.episodes[source_id]
        if retired_ns < episode.published_ns:
            raise ValueError("Retirement precedes publication")
        if episode.retired_ns is not None:
            raise ValueError("Source was already retired")
        self.event("episode_retired", retired_ns, source_id=source_id)
        episode.retired_ns = retired_ns

    def interval(self, name: str, start_ns: int, end_ns: int, **metadata: Any) -> None:
        """Record a stage interval without adding overlapping durations together."""
        start = self._seconds(start_ns)
        end = self._seconds(end_ns)
        if end < start:
            raise ValueError("An interval cannot end before it starts")
        row = {
            **metadata,
            "name": name,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start,
        }
        self.event("interval", end_ns, **row)
        self._intervals.append(row)

    def _batch(
        self, batch: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = self.history + self.future - 1
        ids = _cpu_array(batch["frame_ids"], "frame_ids")
        valid = _cpu_array(batch["valid"], "valid")
        mask = _cpu_array(batch["loss_mask"], "loss_mask")
        generations = _cpu_array(batch["generations"], "generations")
        geometry = _cpu_array(batch["geometry"], "geometry")
        if ids.shape != (self.batch_size, length, 3) or ids.dtype != np.int64:
            raise ValueError("frame_ids shape/dtype differs from the training protocol")
        if (
            valid.shape != (self.batch_size, length)
            or valid.dtype != np.bool_
            or not valid.all()
        ):
            raise ValueError("Training requires complete, unpadded contexts")
        expected_mask = np.broadcast_to(
            np.arange(length) >= self.history - 1, valid.shape
        )
        if mask.dtype != np.bool_ or not np.array_equal(mask, expected_mask):
            raise ValueError("loss_mask must contain exactly the F future targets")
        if generations.shape != valid.shape or generations.dtype != np.int64:
            raise ValueError("Generation shape/dtype mismatch")
        anchor = ids[:, self.history - 1]
        offsets = np.arange(1 - self.history, self.future, dtype=np.int64)
        if (
            (ids < 0).any()
            or (ids[..., :2] != anchor[:, None, :2]).any()
            or (ids[..., 2] != anchor[:, None, 2] + offsets).any()
            or (generations != generations[:, self.history - 1, None]).any()
        ):
            raise ValueError("A training context crosses a source/frame boundary")
        if "continuity_id" in batch:
            continuity = _cpu_array(batch["continuity_id"], "continuity_id")
            if (
                continuity.shape != valid.shape
                or (continuity != continuity[:, self.history - 1, None]).any()
            ):
                raise ValueError("A training context crosses continuity boundaries")
        if (
            geometry.shape != (self.batch_size, length, 51)
            or not np.isfinite(geometry).all()
        ):
            raise ValueError("geometry must contain the 51 finite task-state features")
        return ids, generations, geometry[:, self.history - 1, 27:30]

    def complete_update(
        self,
        batch: Mapping[str, Any],
        *,
        update: int,
        losses: Mapping[str, float],
        requested_ns: int,
        train_started_ns: int,
        completed_ns: int,
    ) -> None:
        """Count a synchronized, completed optimizer update exactly once.

        All validation happens before counters are changed. The input may keep
        GPU image/model data, but the frame IDs and geometry must be owned CPU
        tensors supplied by the data adapter.
        """
        if update != self.updates + 1:
            raise ValueError(
                "Completed update numbers must start at one and be consecutive"
            )
        for timestamp in (requested_ns, train_started_ns, completed_ns):
            self._seconds(timestamp)
        if not (requested_ns <= train_started_ns <= completed_ns):
            raise ValueError("Training update timestamps are not causal")
        if completed_ns < self._last_completed_ns:
            raise ValueError(
                "Completed optimizer updates cannot move backwards in time"
            )
        finite_losses = {name: _finite(value, name) for name, value in losses.items()}
        if not finite_losses:
            raise ValueError("A completed update must record real finite losses")
        ids, generations, xyz = self._batch(batch)
        anchor_rows = ids[:, self.history - 1]
        source_ids = [f"{row[0]}:{row[1]}" for row in anchor_rows]
        counts = Counter(source_ids)
        for source_id, count in counts.items():
            if source_id not in self.episodes:
                raise ValueError(f"Training source was not registered: {source_id}")
            episode = self.episodes[source_id]
            rows = np.asarray(
                [i for i, key in enumerate(source_ids) if key == source_id]
            )
            if (
                train_started_ns < episode.published_ns
                or (generations[rows] != episode.generation).any()
                or (ids[rows, :, 2] >= episode.frame_count).any()
            ):
                raise ValueError(
                    "Training references unpublished or mismatched source data"
                )
            if (
                self.reuse_cap is not None
                and episode.trained_chunks + count > self.reuse_cap
            ):
                raise ValueError(f"Completed training exceeds source cap: {source_id}")
        pairs = batch.get("_pairs")
        if pairs is not None:
            expected_pairs = [
                (self.episodes[source].ordinal, int(row[2]))
                for source, row in zip(source_ids, anchor_rows)
            ]
            if [tuple(int(v) for v in pair) for pair in pairs] != expected_pairs:
                raise ValueError(
                    "Batch source/anchor manifest does not match frame IDs"
                )
        voxels = np.floor(np.asarray(xyz, dtype=np.float64) / _VOXEL_METERS).astype(
            np.int64
        )
        chunks = [
            {
                "source_id": source,
                "ordinal": self.episodes[source].ordinal,
                "generation": self.episodes[source].generation,
                "anchor": int(row[2]),
                "anchor_cube_xyz": [float(v) for v in position],
            }
            for source, row, position in zip(source_ids, anchor_rows, xyz)
        ]
        # The log is append-only. It records sufficient identities to recompute
        # every count without retaining image tensors or per-frame Python sets.
        for index, source in enumerate(source_ids):
            episode = self.episodes[source]
            anchor_frame = int(anchor_rows[index, 2])
            episode.trained_chunks += 1
            episode.anchors[anchor_frame] += 1
            episode.history[ids[index, : self.history, 2]] += 1
            episode.targets[ids[index, self.history - 1 :, 2]] += 1
            if episode.first_completed_ns is None:
                episode.first_completed_ns = completed_ns
            if episode.initial_bin is not None:
                self._initial_bins[episode.initial_bin] += 1
            self._voxels.add(tuple(int(v) for v in voxels[index]))
        self.updates = update
        self.chunks += self.batch_size
        self._last_completed_ns = completed_ns
        self._last_losses = finite_losses
        curve = {
            "update": update,
            "time_seconds": self._seconds(completed_ns),
            **self._coverage(),
        }
        self.event(
            "update_completed",
            completed_ns,
            update=update,
            requested_seconds=self._seconds(requested_ns),
            train_started_seconds=self._seconds(train_started_ns),
            losses=finite_losses,
            chunks=chunks,
            coverage=curve,
        )
        self._curve.append(curve)

    def _coverage(self) -> dict[str, Any]:
        trained = [row for row in self.episodes.values() if row.trained_chunks]
        proportions = [row.trained_chunks / self.chunks for row in trained]
        unique_targets = sum(int(np.count_nonzero(row.targets)) for row in trained)
        target_occurrences = self.chunks * self.future
        return {
            "completed_updates": self.updates,
            "trained_chunks": self.chunks,
            "valid_target_occurrences": target_occurrences,
            "history_frame_occurrences": self.chunks * self.history,
            "trained_sources": len(trained),
            "effective_sources": (
                math.exp(-sum(p * math.log(p) for p in proportions)) if trained else 0.0
            ),
            "initial_xy_occupied_cells": len(self._initial_bins),
            "initial_xy_coverage_fraction": len(self._initial_bins) / 64,
            "anchor_xyz_occupied_voxels": len(self._voxels),
            "unique_anchors": sum(
                int(np.count_nonzero(row.anchors)) for row in trained
            ),
            "unique_history_frames": sum(
                int(np.count_nonzero(row.history)) for row in trained
            ),
            "unique_target_frames": unique_targets,
            "target_duplicate_fraction": (
                1 - unique_targets / target_occurrences if target_occurrences else 0.0
            ),
        }

    def _source_rows(self) -> list[dict[str, Any]]:
        rows = []
        for episode in sorted(self.episodes.values(), key=lambda row: row.ordinal):
            first = episode.first_completed_ns
            rows.append(
                {
                    "source_id": episode.source_id,
                    "ordinal": episode.ordinal,
                    "generation": episode.generation,
                    "frame_count": episode.frame_count,
                    "initial_xy": episode.initial_xy,
                    "initial_xy_bin": episode.initial_bin,
                    "published_seconds": self._seconds(episode.published_ns),
                    "accepted_seconds": (
                        self._seconds(episode.accepted_ns)
                        if episode.accepted_ns is not None
                        else None
                    ),
                    "first_completed_training_seconds": (
                        self._seconds(first) if first is not None else None
                    ),
                    "publication_to_first_training_seconds": (
                        (first - episode.published_ns) / 1e9
                        if first is not None
                        else None
                    ),
                    "trained_chunks": episode.trained_chunks,
                    "unique_anchors": int(np.count_nonzero(episode.anchors)),
                    "unique_history_frames": int(np.count_nonzero(episode.history)),
                    "unique_target_frames": int(np.count_nonzero(episode.targets)),
                    "target_occurrences": int(episode.targets.sum()),
                    "anchor_counts": episode.anchors.tolist(),
                    "history_frame_counts": episode.history.tolist(),
                    "target_frame_counts": episode.targets.tolist(),
                    "retired_seconds": (
                        self._seconds(episode.retired_ns)
                        if episode.retired_ns is not None
                        else None
                    ),
                    "metadata": episode.metadata,
                }
            )
        return rows

    def summary(
        self,
        *,
        total_seconds: float,
        training_complete_seconds: float,
        ledger: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an auditable snapshot; failures may also snapshot partial work.

        ``ledger`` is copied verbatim as the sampler's separate reservation and
        delivery evidence. The caller owns delivery-cap and exact-budget
        qualification; this class enforces the completed-training cap itself.
        """
        total = _finite(total_seconds, "total_seconds")
        train_end = _finite(training_complete_seconds, "training_complete_seconds")
        if total < train_end or train_end < 0:
            raise ValueError("Completion times must satisfy 0 <= training <= total")
        if self.updates and train_end + 1e-7 < self._seconds(self._last_completed_ns):
            raise ValueError("Reported training completion precedes the final update")
        sources = self._source_rows()
        latencies = [
            row["publication_to_first_training_seconds"]
            for row in sources
            if row["publication_to_first_training_seconds"] is not None
        ]
        initial_weights = np.zeros((8, 8), dtype=np.int64)
        for (x, y), count in self._initial_bins.items():
            initial_weights[x, y] = count
        metrics = {
            "schema": "ods-training-metrics/v1",
            "history": self.history,
            "future": self.future,
            "batch_size": self.batch_size,
            "reuse_cap": self.reuse_cap,
            "origin_monotonic_ns": self.origin_ns,
            "total_seconds": total,
            "training_complete_seconds": train_end,
            "post_training_drain_seconds": total - train_end,
            "first_update_seconds": (
                self._curve[0]["time_seconds"] if self._curve else None
            ),
            **self._coverage(),
            "published_sources": len(sources),
            "untrained_sources": sum(row["trained_chunks"] == 0 for row in sources),
            "initial_xy_grid": {
                "low_meters": _INITIAL_XY_LOW.tolist(),
                "high_meters": _INITIAL_XY_HIGH.tolist(),
                "shape": [8, 8],
                "bin_meters": _INITIAL_BIN_METERS,
                "trained_chunk_weights_xy": initial_weights.tolist(),
                "out_of_bounds_sources": [
                    row["source_id"] for row in sources if row["initial_xy_bin"] is None
                ],
            },
            "anchor_xyz_voxel_meters": _VOXEL_METERS,
            "anchor_xyz_voxels": [list(row) for row in sorted(self._voxels)],
            "publication_to_first_training_seconds": _quantiles(latencies),
            "max_source_trained_chunks": max(
                (row["trained_chunks"] for row in sources), default=0
            ),
            "training_cap_violations": sum(
                self.reuse_cap is not None and row["trained_chunks"] > self.reuse_cap
                for row in sources
            ),
            "episodes": sources,
            "coverage_curve": self._curve.copy(),
            "intervals": self._intervals.copy(),
            "last_losses": self._last_losses.copy(),
            "ledger": json.loads(json.dumps(ledger or {}, allow_nan=False)),
        }
        write_json(self.output / "training-sources.json", {"episodes": sources})
        return metrics

    def close(self) -> None:
        """Close the append-only event stream exactly once."""
        if not self._closed:
            self._events.close()
            self._closed = True
