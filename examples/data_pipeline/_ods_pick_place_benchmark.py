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

"""Private CPU corpus and accounting helpers for the PickPlace experiment.

This is intentionally independent of the simulator and the live ODS engine.
Corpus observations and actions are the policy's numeric request schema, not
images, simulator state, or reconstructed expert controller commands.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Iterable
from typing import Any

import torch

from embodichain.data_pipeline.engine._sampling import _draw_budgeted_rows

__all__ = [
    "FrozenCorpus",
    "EfficiencyTracker",
    "resource_snapshot",
]

_FORMAT = "ods-pick-place-cpu-corpus-v1"
_TENSOR_KEYS = {"observations", "actions", "valid", "frame_ids"}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _validate_tensors(tensors: dict[str, torch.Tensor], schema: dict[str, Any]) -> None:
    if not _TENSOR_KEYS <= set(tensors) <= _TENSOR_KEYS | {"generations"} or not all(
        isinstance(t, torch.Tensor) for t in tensors.values()
    ):
        raise ValueError(
            "Corpus must contain the four numeric tensor fields and optional generations"
        )
    obs, act, valid, ids = (
        tensors[k] for k in ("observations", "actions", "valid", "frame_ids")
    )
    if any(t.device.type != "cpu" for t in tensors.values()):
        raise ValueError("Corpus tensors must be on CPU")
    if obs.ndim != 3 or act.ndim != 3 or obs.shape[:2] != act.shape[:2]:
        raise ValueError(
            "Observations/actions must have matching [episode, time, feature] shapes"
        )
    if obs.dtype != torch.float32 or act.dtype != torch.float32:
        raise ValueError("Policy observations and actions must be float32")
    if valid.dtype != torch.bool or valid.shape != obs.shape[:2] or not valid.any():
        raise ValueError("valid must be a nonempty bool [episode, time] mask")
    if ids.dtype != torch.int64 or ids.shape != (*valid.shape, 3):
        raise ValueError(
            "frame_ids must be int64 [episode, time, 3] (run, episode, step)"
        )
    generations = tensors.get("generations")
    if generations is not None and (
        generations.dtype != torch.int64
        or generations.shape != valid.shape
        or (generations[valid] < 1).any()
    ):
        raise ValueError(
            "generations must be positive int64 [episode, time] on valid frames"
        )
    for name, width in (("observation", obs.shape[-1]), ("action", act.shape[-1])):
        names = schema.get(name, {}).get("names")
        if (
            not isinstance(names, list)
            or len(names) != width
            or not all(isinstance(n, str) and n for n in names)
            or len(set(names)) != width
        ):
            raise ValueError(
                f"Schema {name}.names must name all {width} ordered features"
            )
    if not all(torch.isfinite(t[valid]).all() for t in (obs, act)):
        raise ValueError("Valid observations/actions must be finite")
    identities = set()
    for row in range(valid.shape[0]):
        selected = ids[row, valid[row]]
        if not len(selected):
            continue
        identity = tuple(selected[0, :2].tolist())
        steps = valid[row].nonzero().flatten()
        if (
            identity in identities
            or (selected < 0).any()
            or not (selected[:, :2] == selected[0, :2]).all()
            or not torch.equal(selected[:, 2], steps)
        ):
            raise ValueError(
                "Frame identities must preserve unique episode rows and within-episode steps"
            )
        if generations is not None and len(generations[row, valid[row]].unique()) != 1:
            raise ValueError("One corpus episode must belong to one generation")
        identities.add(identity)


def _content_manifest(
    tensors: dict[str, torch.Tensor], schema: dict[str, Any]
) -> dict[str, Any]:
    schema_hash = hashlib.sha256(_json_bytes(schema)).hexdigest()
    episodes = []
    for row, mask in enumerate(tensors["valid"]):
        episodes.append(
            {
                "valid_steps": mask.nonzero().flatten().tolist(),
                "observations_sha256": _tensor_hash(tensors["observations"][row, mask]),
                "actions_sha256": _tensor_hash(tensors["actions"][row, mask]),
            }
        )
    # Engine-local run IDs are provenance, not content identity.
    content_id = hashlib.sha256(
        _json_bytes({"schema_sha256": schema_hash, "episodes": episodes})
    ).hexdigest()
    return {
        "schema_sha256": schema_hash,
        "content_id": content_id,
        "episodes": episodes,
        "tensors": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_hash(value),
            }
            for key, value in sorted(tensors.items())
        },
    }


class FrozenCorpus:
    """Read-only-by-convention corpus with independent sampled tensor copies.

    Optional quotas count each sampled row once, regardless of context length.
    They apply to all sampling entry points and cannot be replenished. Configure
    them before sampling; the stateful sampler then requires its creating
    process and thread. Tensor metadata must remain unchanged after assignment.
    """

    def __init__(
        self, tensors: dict[str, torch.Tensor], manifest: dict[str, Any]
    ) -> None:
        self.observations = tensors["observations"]
        self.actions = tensors["actions"]
        self.valid = tensors["valid"]
        self.frame_ids = tensors["frame_ids"]
        self.generations = tensors.get("generations")
        self.manifest = manifest
        self._rows = self.valid.any(1).nonzero().flatten()
        self._frames = {
            int(row): self.valid[row].nonzero().flatten() for row in self._rows
        }
        self._sampling_owner = (os.getpid(), threading.get_ident())
        self._sampling_started = False
        self._quota: dict[str, Any] | None = None

    def configure_sampling_budget(self, budget: int, hard_limit: int) -> None:
        """Assign a permanent per-source chunk allowance before first sampling.

        The stateful sampler belongs to its creating process and thread. Copying
        it into DataLoader workers cannot duplicate allowances. Valid rows must
        each identify one distinct immutable ``(run, episode)`` source. Empty
        padded rows receive no allowance. There is no reset or refill operation.
        """
        self._require_sampling_owner()
        if self._quota is not None or self._sampling_started:
            raise RuntimeError("Configure the sampling budget once, before sampling")
        if (
            any(
                type(n) is not int or not 0 < n <= (1 << 63) - 1
                for n in (budget, hard_limit)
            )
            or budget > hard_limit
        ):
            raise ValueError("Require positive int64 budget <= hard_limit")
        if any(
            t.device.type != "cpu"
            for t in (self.observations, self.actions, self.valid, self.frame_ids)
        ):
            raise ValueError("Offline sampling budgets require CPU corpus tensors")
        identities = torch.full(
            (len(self.valid), 2), -1, dtype=torch.int64, device="cpu"
        )
        seen = set()
        generations = {}
        for row in self._rows.tolist():
            ids = self.frame_ids[row, self.valid[row]]
            identity = tuple(ids[0, :2].tolist())
            if (ids[:, :2] < 0).any() or not (ids[:, :2] == ids[0, :2]).all():
                raise ValueError(
                    "Each valid row must contain one immutable source identity"
                )
            if identity in seen:
                raise ValueError("Duplicate source identity across corpus rows")
            seen.add(identity)
            identities[row] = ids[0, :2]
            values = (
                self.generations[row, self.valid[row]].unique()
                if self.generations is not None
                else None
            )
            generations[row] = (
                int(values[0]) if values is not None and len(values) == 1 else None
            )
        self._quota = {
            "budget": budget,
            "hard_limit": hard_limit,
            "identities": identities,
            "generations": generations,
            "delivered": torch.zeros(len(self.valid), dtype=torch.int64, device="cpu"),
            "clock_origin_monotonic_ns": time.monotonic_ns(),
        }

    def _require_sampling_owner(self) -> None:
        if self._sampling_owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError(
                "Budgeted FrozenCorpus sampling is owner-process/thread only"
            )

    def get_sampling_budget_snapshot(self) -> dict[str, Any]:
        """Return detached per-source delivery counts; sampling is one row/chunk."""
        quota = self._quota
        if quota is None:
            return {
                "enabled": False,
                "rows": [],
                "remaining_chunks": None,
                "unlocked_remaining_chunks": None,
                "total_delivered_chunks": None,
                "clock_origin_monotonic_ns": None,
            }
        self._require_sampling_owner()
        rows = []
        for row in self._rows.tolist():
            run_id, episode_id = quota["identities"][row].tolist()
            delivered = int(quota["delivered"][row])
            rows.append(
                {
                    "slot": row,
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "generation": quota["generations"][row],
                    "assigned": quota["budget"],
                    "delivered": delivered,
                    "remaining": quota["budget"] - delivered,
                    "locked": False,
                }
            )
        remaining = sum(row["remaining"] for row in rows)
        return {
            "enabled": True,
            "rows": rows,
            "remaining_chunks": remaining,
            "unlocked_remaining_chunks": remaining,
            "total_delivered_chunks": sum(row["delivered"] for row in rows),
            "episode_sample_budget": quota["budget"],
            "episode_sample_hard_limit": quota["hard_limit"],
            "clock_origin_monotonic_ns": quota["clock_origin_monotonic_ns"],
        }

    @staticmethod
    def _budget_exhausted() -> None:
        from embodichain.data_pipeline.engine.data import OnlineDataBudgetExhaustedError

        raise OnlineDataBudgetExhaustedError(
            "Insufficient offline source quota for a full batch"
        )

    def _draw_rows(
        self, eligible: torch.Tensor, batch_size: int, generator: torch.Generator | None
    ) -> torch.Tensor:
        if self._quota is None:
            return eligible[
                torch.randint(
                    len(eligible), (batch_size,), generator=generator, device="cpu"
                )
            ]
        self._require_sampling_owner()
        remaining = self._quota["budget"] - self._quota["delivered"][eligible]
        if sum(remaining.tolist()) < batch_size:
            self._budget_exhausted()
        return _draw_budgeted_rows(eligible, remaining, batch_size, generator=generator)

    def _check_quota_rows(self, rows: torch.Tensor) -> torch.Tensor | None:
        if self._quota is None:
            return None
        self._require_sampling_owner()
        counts = torch.bincount(rows.reshape(-1), minlength=len(self.valid))
        if bool((counts > self._quota["budget"] - self._quota["delivered"]).any()):
            self._budget_exhausted()
        return counts

    def _finish_sample(
        self, rows: torch.Tensor, result: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        counts = self._check_quota_rows(rows)
        if counts is not None:
            self._quota["delivered"] += counts
        self._sampling_started = True
        return result

    def _filtered_valid(self, generations: Iterable[int] | None) -> torch.Tensor:
        if generations is None:
            return self.valid
        if self.generations is None:
            raise ValueError("Corpus has no generation metadata")
        return self.valid & torch.isin(
            self.generations,
            torch.tensor(list(generations), dtype=torch.int64, device="cpu"),
        )

    def sample_context(
        self,
        batch_size: int,
        history_length: int,
        prediction_horizon: int,
        *,
        generations: Iterable[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample episode-uniform anchors with masked causal/future context.

        Matches the live engine's context convention. Invalid gaps split spans;
        this corpus has no additional continuity changes within valid spans.
        """
        if any(
            type(n) is not int or n < 1
            for n in (batch_size, history_length, prediction_horizon)
        ):
            raise ValueError("Context lengths and batch size must be positive integers")
        valid = self._filtered_valid(generations)
        eligible = valid.any(1).nonzero().flatten()
        if not len(eligible):
            raise RuntimeError("No valid offline context anchor")
        rows = self._draw_rows(eligible, batch_size, generator)
        anchors = torch.multinomial(
            valid[rows].float(), 1, replacement=True, generator=generator
        ).flatten()
        steps = torch.arange(valid.shape[1], device="cpu")
        lower = torch.where(
            (~valid[rows]) & (steps[None] < anchors[:, None]), steps[None] + 1, 0
        ).amax(-1)
        upper = torch.where(
            (~valid[rows]) & (steps[None] > anchors[:, None]),
            steps[None] - 1,
            valid.shape[1] - 1,
        ).amin(-1)
        offsets = torch.arange(1 - history_length, prediction_horizon, device="cpu")
        times = anchors[:, None] + offsets[None]
        mask = (times >= lower[:, None]) & (times <= upper[:, None])
        result = self._select(rows[:, None], times.clamp(0, valid.shape[1] - 1))
        for key in ("observations", "actions"):
            result[key][~mask] = 0
        result["frame_ids"][~mask] = -1
        if "generations" in result:
            result["generations"][~mask] = -1
        result["valid"] = mask
        result["loss_mask"] = mask & (offsets[None] >= 0)
        return self._finish_sample(rows, result)

    def _select(
        self, rows: torch.Tensor, times: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self._quota is not None:
            self._require_sampling_owner()
        result = {
            "observations": self.observations[rows, times],
            "actions": self.actions[rows, times],
            "valid": self.valid[rows, times],
            "frame_ids": self.frame_ids[rows, times],
        }
        if self.generations is not None:
            result["generations"] = self.generations[rows, times]
        if self._quota is not None:
            expected = self._quota["identities"][rows]
            if not torch.all(
                (result["frame_ids"][..., :2] == expected).all(-1) | ~result["valid"]
            ):
                raise RuntimeError(
                    "Corpus source identity changed after quota assignment"
                )
        return result


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _union(intervals))


def _intersection(
    a: Iterable[tuple[float, float]], b: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intersect normalized intervals in linear time after the two sorts.

    Normalize each iterable once, including single-pass generators. Empty
    intervals contribute no duration and are omitted from the intersection.
    """
    left, right = _union(a), _union(b)
    result: list[tuple[float, float]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start_a, end_a = left[i]
        start_b, end_b = right[j]
        start, end = max(start_a, start_b), min(end_a, end_b)
        if end > start:
            result.append((start, end))
        if end_a <= end_b:
            i += 1
        if end_b <= end_a:
            j += 1
    return result


class EfficiencyTracker:
    """Account for observed events; absent measurement paths remain null.

    Timestamps are elapsed seconds from one monotonic launch clock. Evaluation
    timestamps are result arrival times. Record a checkpoint when its weights
    are saved so an asynchronous evaluation retains that model's target counts.
    Data-wait intervals must mean blocked solely on unavailable data, excluding
    sampling/indexing/copying. ``training_loop`` intervals include those waits.
    """

    def __init__(
        self,
        horizon_seconds: float,
        *,
        allocated_gpu_ids: Iterable[str | int] = (),
        start_monotonic: float | None = None,
    ) -> None:
        if not math.isfinite(horizon_seconds) or horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be finite and positive")
        self.horizon_seconds = float(horizon_seconds)
        self.start_monotonic = (
            time.monotonic() if start_monotonic is None else start_monotonic
        )
        self._start_wall = time.time() - self.elapsed_seconds()
        self.allocated_gpu_ids = sorted({str(value) for value in allocated_gpu_ids})
        self.targets = 0
        from examples.data_pipeline._ods_pick_place_usage import CompactFrameSet

        self.unique_frames = CompactFrameSet()
        self.timeline: list[dict[str, Any]] = []
        self.intervals: dict[str, list[tuple[float, float]]] = {}
        self._checkpoints: dict[str, tuple[int, int]] = {}
        self._evaluations: list[dict[str, Any]] = []
        self._resources: list[dict[str, Any]] = []
        self._first_update: float | None = None
        self._last_event = 0.0
        self._ended = False

    def elapsed_seconds(self) -> float:
        """Read the acquisition-inclusive monotonic launch clock."""
        return time.monotonic() - self.start_monotonic

    def _event(self, kind: str, elapsed: float, **values: Any) -> None:
        if self._ended or not math.isfinite(elapsed) or elapsed < self._last_event:
            raise ValueError(
                "Events must use nondecreasing finite monotonic elapsed times"
            )
        self._last_event = elapsed
        self.timeline.append({"kind": kind, "seconds": elapsed, **values})

    def record_update(
        self,
        elapsed: float,
        frame_ids: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Count only targets belonging to a completed optimizer update."""
        if frame_ids.dtype != torch.int64 or frame_ids.shape[-1:] != (3,):
            raise ValueError("frame_ids must be int64 [..., 3]")
        if valid_mask is not None and (
            valid_mask.dtype != torch.bool or valid_mask.shape != frame_ids.shape[:-1]
        ):
            raise ValueError("Target mask must be bool and match frame identities")
        selected = (
            frame_ids.reshape(-1, 3) if valid_mask is None else frame_ids[valid_mask]
        )
        if not len(selected) or (selected < 0).any():
            raise ValueError("An update must contain nonnegative valid target IDs")
        self._event("update", elapsed, targets=len(selected))
        self.targets += len(selected)
        self.unique_frames.add_tensor(selected)
        if self._first_update is None:
            self._first_update = elapsed

    def record_checkpoint(
        self,
        checkpoint: str,
        *,
        policy_step: int | None = None,
        elapsed: float | None = None,
    ) -> None:
        """Freeze supervision counters after saving immutable model weights.

        Args:
            checkpoint: Unique identity of the already-saved checkpoint.
            policy_step: Completed optimizer step represented by the model.
                Supplying this also records its save-completion event.
            elapsed: Optional elapsed save-completion timestamp. With a policy
                step, defaults to the current launch clock. Legacy calls that
                omit the step retain counter-only accounting.
        """
        if checkpoint in self._checkpoints or self._ended:
            raise ValueError(
                "Checkpoint identities must be unique and recorded before finish"
            )
        if policy_step is not None:
            if type(policy_step) is not int or policy_step < 0:
                raise ValueError("Checkpoint policy step must be a nonnegative integer")
            self._event(
                "checkpoint_saved",
                self.elapsed_seconds() if elapsed is None else elapsed,
                checkpoint=checkpoint,
                policy_step=policy_step,
            )
        elif elapsed is not None:
            raise ValueError("A checkpoint timestamp requires its policy step")
        self._checkpoints[checkpoint] = (self.targets, len(self.unique_frames))

    def record_evaluation(
        self,
        elapsed: float,
        success_rate: float | None,
        *,
        checkpoint: str,
        scheduled_index: int | None = None,
    ) -> None:
        """Publish a scheduled dev result; a failed result breaks the streak."""
        if success_rate is not None and (
            not math.isfinite(success_rate) or not 0 <= success_rate <= 1
        ):
            raise ValueError("success_rate must be a fraction or None for failure")
        index = (
            (self._evaluations[-1]["scheduled_index"] + 1 if self._evaluations else 0)
            if scheduled_index is None
            else scheduled_index
        )
        if (
            type(index) is not int
            or index < 0
            or (self._evaluations and index <= self._evaluations[-1]["scheduled_index"])
        ):
            raise ValueError("Scheduled evaluation indices must increase")
        if any(value["checkpoint"] == checkpoint for value in self._evaluations):
            raise ValueError("Scheduled development checkpoints must be distinct")
        counts = self._checkpoints.get(checkpoint, (None, None))
        self._event(
            "evaluation", elapsed, checkpoint=checkpoint, success_rate=success_rate
        )
        self._evaluations.append(
            {
                "seconds": elapsed,
                "success_rate": success_rate,
                "checkpoint": checkpoint,
                "scheduled_index": index,
                "targets": counts[0],
                "unique_frames": counts[1],
            }
        )

    def record_interval(self, kind: str, start: float, end: float) -> None:
        """Record elapsed host intervals; union accounting avoids double counts.

        Recognized measurement kinds are ``training_loop``, ``evaluation``,
        ``data_wait`` and ``batch_service``. Other kinds remain in the timeline.
        Host production/learner intervals do not imply completed device overlap.
        """
        if (
            self._ended
            or not all(math.isfinite(t) for t in (start, end))
            or start < 0
            or end < start
        ):
            raise ValueError("Intervals must be finite, nonnegative and ordered")
        self.intervals.setdefault(kind, []).append((start, end))

    def record_resources(self, snapshot: dict[str, Any]) -> None:
        """Retain sampled PSS peaks and per-PID CPU counters without summing RSS."""
        if self._ended:
            raise ValueError("Cannot record resources after finish")
        self._resources.append(snapshot)

    def finish(self, elapsed: float, end_reason: str) -> dict[str, Any]:
        """Finalize reached/censored endpoints, holding available quality to H."""
        self._event("finish", elapsed, end_reason=end_reason)
        self._ended = True
        endpoint = min(elapsed, self.horizon_seconds)
        evaluations = [e for e in self._evaluations if e["seconds"] <= endpoint]
        attained = None
        for previous, current in zip(evaluations, evaluations[1:]):
            if (
                current["scheduled_index"] == previous["scheduled_index"] + 1
                and previous["success_rate"] is not None
                and current["success_rate"] is not None
                and min(previous["success_rate"], current["success_rate"]) >= 0.8
            ):
                attained = current
                break
        area, best, last = 0.0, 0.0, 0.0
        for result in evaluations:
            area += best * (result["seconds"] - last)
            if result["success_rate"] is not None:
                best = max(best, result["success_rate"])
            last = result["seconds"]
        area += best * (self.horizon_seconds - last)
        clipped = {
            kind: [
                (start, min(end, elapsed)) for start, end in values if start < elapsed
            ]
            for kind, values in self.intervals.items()
        }
        training = clipped.get("training_loop", [])
        evaluation = clipped.get("evaluation", [])
        training_seconds = _duration(training) - _duration(
            _intersection(training, evaluation)
        )
        waiting = _intersection(clipped.get("data_wait", []), training)
        wait_seconds = _duration(waiting) - _duration(
            _intersection(waiting, evaluation)
        )
        pss = [
            r["host_pss_bytes"]
            for r in self._resources
            if r.get("host_pss_bytes") is not None
        ]
        cpu_by_pid: dict[tuple[str, float], list[float]] = {}
        cuda_peak: dict[str, int] = {}
        for resource in self._resources:
            for pid, counters in resource.get("processes", {}).items():
                if counters.get("create_time") is None:
                    continue
                identity = (pid, counters["create_time"])
                value = counters["cpu_core_seconds"]
                if identity not in cpu_by_pid:
                    cpu_by_pid[identity] = [
                        0.0 if counters["create_time"] >= self._start_wall else value,
                        value,
                    ]
                cpu_by_pid[identity][1] = max(cpu_by_pid[identity][1], value)
            for device, counters in resource.get("torch_cuda_this_process", {}).items():
                if counters is not None:
                    cuda_peak[device] = max(
                        cuda_peak.get(device, 0), counters["max_reserved_bytes"]
                    )
        latencies = [
            1000 * (end - start) for start, end in clipped.get("batch_service", [])
        ]
        t80 = None if attained is None else attained["seconds"]
        return {
            "end_reason": end_reason,
            "observed_seconds": elapsed,
            "horizon_seconds": self.horizon_seconds,
            "efficiency/reached_80": attained is not None,
            "efficiency/t80_dev_seconds": t80,
            "efficiency/t80_dev_right_censored": attained is None,
            "efficiency/censor_seconds": endpoint if attained is None else None,
            "efficiency/t80_dev_checkpoint": (
                None if attained is None else attained["checkpoint"]
            ),
            "efficiency/auc_success_time": area / self.horizon_seconds,
            "pipeline/first_update_seconds": self._first_update,
            "pipeline/targets_per_second": (
                self.targets / training_seconds if training_seconds > 0 else None
            ),
            "pipeline/acquisition_inclusive_targets_per_second": (
                self.targets / elapsed if elapsed > 0 else None
            ),
            "pipeline/training_loop_seconds": training_seconds if training else None,
            "pipeline/exposed_data_wait_seconds": (
                wait_seconds if training and "data_wait" in self.intervals else None
            ),
            "pipeline/data_wait_fraction": (
                wait_seconds / training_seconds
                if training_seconds > 0 and "data_wait" in self.intervals
                else None
            ),
            "pipeline/batch_latency_ms_p50_p95": (
                torch.quantile(
                    torch.tensor(latencies, dtype=torch.float64),
                    torch.tensor([0.5, 0.95], dtype=torch.float64),
                ).tolist()
                if latencies
                else None
            ),
            "learning/valid_targets": self.targets,
            "learning/unique_frames": len(self.unique_frames),
            "learning/targets_to_80": None if attained is None else attained["targets"],
            "learning/unique_frames_to_80": (
                None if attained is None else attained["unique_frames"]
            ),
            "data/targets_per_unique_frame": (
                self.targets / len(self.unique_frames) if self.unique_frames else None
            ),
            "cost/gpu_hours_to_80": (
                len(self.allocated_gpu_ids) * t80 / 3600 if t80 is not None else None
            ),
            "cost/allocated_gpu_hours": len(self.allocated_gpu_ids) * elapsed / 3600,
            "cost/allocated_physical_gpu_ids": self.allocated_gpu_ids,
            "cost/host_pss_sampled_peak_bytes": max(pss) if pss else None,
            "cost/cpu_core_seconds_observed": (
                sum(last - first for first, last in cpu_by_pid.values())
                if cpu_by_pid
                else None
            ),
            "cost/torch_cuda_this_process_max_reserved_bytes_by_device": cuda_peak
            or None,
            "cost/gpu_peak_bytes": None,
            "pipeline/producer_learner_overlap_fraction": None,
        }


def resource_snapshot(
    pids: Iterable[int] | None = None, *, cuda_devices: Iterable[int] = ()
) -> dict[str, Any]:
    """Measure explicit owned PIDs and this process's PyTorch CUDA allocator.

    PIDs are deduplicated and user+system CPU excludes child CPU counters. PSS
    avoids double-counting shared pages; a missing process or unsupported PSS
    makes the total unknown. CUDA counters describe only this process, never
    simulator/driver device totals. Device allocation-hours are tracked separately.
    """
    import psutil

    processes = {}
    missing = []
    for pid in sorted(set([os.getpid()] if pids is None else pids)):
        try:
            process = psutil.Process(pid)
            cpu = process.cpu_times()
            try:
                pss = getattr(process.memory_full_info(), "pss", None)
            except (psutil.AccessDenied, NotImplementedError):
                pss = None
            processes[str(pid)] = {
                "cpu_core_seconds": cpu.user + cpu.system,
                "host_pss_bytes": pss,
                "create_time": process.create_time(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            missing.append(pid)
    pss_values = [value["host_pss_bytes"] for value in processes.values()]
    cuda = {}
    for index in sorted(set(cuda_devices)):
        key = f"cuda:{index}"
        if torch.cuda.is_initialized():
            cuda[key] = {
                "allocated_bytes": torch.cuda.memory_allocated(index),
                "reserved_bytes": torch.cuda.memory_reserved(index),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(index),
            }
        else:
            cuda[key] = None
    return {
        "monotonic_seconds": time.monotonic(),
        "processes": processes,
        "missing_pids": missing,
        "host_pss_bytes": (
            sum(pss_values)
            if pss_values and not missing and all(v is not None for v in pss_values)
            else None
        ),
        "torch_cuda_this_process": cuda,
    }
