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

"""Compare visual BC diversity and complete-pipeline time at a fixed update budget.

Run ``python -m examples.ods.data_stream_demo.train --help``. The supervisor
selects physical GPU UUIDs before launching the real acquisition/learner worker.
Every arm records standard LeRobot data; offline training really decodes it.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from typing import Any

__all__ = ["build_parser", "run_one", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Expose explicit resource roles and real, fixed training-work budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("single", "smoke", "calibrate", "pilot", "report"),
        default="single",
    )
    parser.add_argument(
        "--arm", choices=("offline", "ods_overlap", "ods_serial"), default="ods_overlap"
    )
    parser.add_argument("--vision", choices=("single", "dual"), default="single")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int)
    parser.add_argument("--aux-graphics-gpu-id", type=int)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--train-device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=19001)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=8)
    parser.add_argument("--producer-envs", type=int, default=2)
    parser.add_argument("--episode-budget", type=int, default=128)
    parser.add_argument(
        "--handoff-mode", choices=("generation", "publication"), default="publication"
    )
    parser.add_argument("--encoder-threads", type=int, default=1)
    parser.add_argument(
        "--prefetch-policy", choices=("budget", "eager"), default="budget"
    )
    parser.add_argument("--calibration-updates", type=int)
    parser.add_argument("--offline-episodes", type=int, default=32)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--prediction-horizon", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--evaluation-scenes", type=int, default=20)
    parser.add_argument("--initialization-timeout", type=float, default=1800)
    parser.add_argument("--generation-timeout", type=float, default=1800)
    parser.add_argument("--data-timeout", type=float, default=120)
    parser.add_argument("--run-timeout", type=float, default=3600)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", default="single", help=argparse.SUPPRESS)
    return parser


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _digest(values: dict[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(memoryview(value.numpy()))
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def _validate(args: argparse.Namespace) -> None:
    for name in (
        "updates",
        "batch_size",
        "buffer_size",
        "producer_envs",
        "episode_budget",
        "offline_episodes",
        "prediction_horizon",
        "max_episode_steps",
        "encoder_threads",
    ):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        args.history_length != 2
        or not 0 < args.action_horizon <= args.prediction_horizon
    ):
        raise ValueError("Visual BC requires H=2 and a positive action horizon <=F")
    if args.buffer_size % args.producer_envs or args.batch_size % args.buffer_size:
        raise ValueError("C must divide B and be divisible by P")
    if args.buffer_size * args.episode_budget < args.batch_size:
        raise ValueError("A complete retained block must fund at least one batch")
    if args.offline_episodes % args.buffer_size:
        raise ValueError("Offline episode count must be a multiple of C")
    if args.prefetch_policy not in ("budget", "eager"):
        raise ValueError("prefetch_policy must be budget or eager")
    if args.handoff_mode not in ("generation", "publication"):
        raise ValueError("handoff_mode must be generation or publication")
    if (
        args.arm == "ods_overlap"
        and args.handoff_mode == "publication"
        and args.producer_envs * args.episode_budget < args.batch_size
    ):
        raise ValueError(
            "Publication handoff requires P*K >= B to fund a complete batch"
        )
    if args.calibration_updates is not None and (
        type(args.calibration_updates) is not int or args.calibration_updates < 1
    ):
        raise ValueError("calibration_updates must be a positive integer or None")
    if args.arm == "ods_serial" and args.trace is None:
        raise ValueError("Serial attribution requires the corresponding overlap trace")
    if (
        not args.sim_device.startswith("cuda:")
        or not args.train_device.startswith("cuda:")
        or args.sim_device == args.train_device
    ):
        raise ValueError(
            "The real experiment requires separate CUDA generation and training devices"
        )


def _prefetch_decision(
    *,
    remaining_updates: int,
    batch_size: int,
    remaining_credits: int,
    handoff_seconds: float | None,
    update_seconds: float | None,
    policy: str = "budget",
) -> dict[str, Any]:
    """Request necessary supply, or a refresh likely to reach a future update.

    K remains a maximum reuse budget. Sufficient current credits do not prevent
    timely refreshes when production is faster than the remaining learner work.
    Unknown timing is conservative; quota shortages always request supply.
    """
    if (
        type(remaining_updates) is not int
        or remaining_updates < 0
        or type(batch_size) is not int
        or batch_size < 1
        or type(remaining_credits) is not int
        or remaining_credits < 0
        or policy not in ("budget", "eager")
    ):
        raise ValueError(
            "Prefetch requires nonnegative work/credits and a valid policy"
        )
    for value in (handoff_seconds, update_seconds):
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("Measured prefetch timings must be finite and positive")
    remaining_seconds = (
        None
        if update_seconds is None
        else max(0, remaining_updates - 1) * update_seconds
    )
    if remaining_updates == 0:
        request, reason = False, "training_complete"
    elif remaining_credits < remaining_updates * batch_size:
        request, reason = True, "quota_required"
    elif remaining_updates == 1:
        request, reason = False, "last_update_funded"
    elif policy == "eager":
        request, reason = True, "eager_control"
    elif handoff_seconds is None or update_seconds is None:
        request, reason = False, "timing_not_observed"
    elif handoff_seconds < remaining_seconds:
        request, reason = True, "timely_refresh"
    else:
        request, reason = False, "arrival_after_remaining_work"
    return {
        "request": request,
        "reason": reason,
        "policy": policy,
        "remaining_updates": remaining_updates,
        "remaining_credits": remaining_credits,
        "handoff_seconds": handoff_seconds,
        "ready_update_seconds": update_seconds,
        "remaining_refresh_window_seconds": remaining_seconds,
    }


class _Experiment:
    """Own one finite pipeline; retained data and recording staging are bounded."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        from .metrics import ResourceMonitor
        from .training_metrics import TrainingMetrics

        self.args, self.output = args, args.output
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "config.json").exists():
            raise FileExistsError(
                "Each timed experiment requires a fresh output directory"
            )
        self.origin_ns = int(
            os.environ.get("ODS_PROCESS_LAUNCH_MONOTONIC_NS", time.monotonic_ns())
        )
        self.deadline_ns = self.origin_ns + int(args.run_timeout * 1e9)
        self.metrics = TrainingMetrics(
            self.output,
            origin_ns=self.origin_ns,
            history=2,
            future=args.prediction_horizon,
            batch_size=args.batch_size,
            reuse_cap=None if args.arm == "offline" else args.episode_budget,
        )
        self.monitor = ResourceMonitor(self.output)
        self.engine = self.flow = self.writer = self.reader = None
        self.model = self.optimizer = self.block = None
        self.metadata: dict[str, Any] = {}
        self.statistics: dict[str, torch.Tensor] = {}
        self.registry: list[dict[str, Any]] = []
        self.trace_updates: list[dict[str, Any]] = []
        self.retired_ledgers: list[dict[str, Any]] = []
        self.staging: list[Any] | None = None
        self.staging_index = 0
        self.packet = None
        self.training_completed_ns = self.origin_ns
        self.generation = self.completed_updates = 0
        self.generation_starts: dict[int, int] = {}
        self.writer_stats: dict[str, Any] = {}
        self.recording_log_sha256: str | None = None
        self.resources: dict[str, Any] = {}
        self.cleanup: dict[str, Any] = {}
        self.gpu_step_ms: list[float] = []
        self.ready_update_seconds: list[float] = []
        self.generation_seconds: list[float] = []
        self.snapshot_seconds: list[float] = []
        self.last_prefetch_decision: tuple[int, str, bool] | None = None
        self.closed = False
        self.retained_probe = None
        self.retained_probe_hash: str | None = None
        self.initial_episodes = None
        self.source_run_id: int | None = None
        self.cache_bytes = 0
        self.serial_trace = json.loads(args.trace.read_text()) if args.trace else None
        self.retained_blocks: list[dict[str, Any]] = []
        self.peak_owned_episodes = 0
        self.recorded_generations: set[int] = set()
        self.ready_episodes: dict[int, Any] = {}
        self.update_completion_ns: list[int] = []

    def now(self) -> int:
        return time.monotonic_ns()

    def poll(self) -> None:
        if self.now() >= self.deadline_ns:
            raise TimeoutError("End-to-end pipeline exceeded its declared deadline")
        if self.writer is not None:
            self.writer.check_health()
            self.pump_recording()

    def pump_recording(self) -> None:
        from .payload import pack_episode

        if self.staging is None or self.writer is None:
            return
        if self.packet is None and self.staging_index < len(self.staging):
            started = self.now()
            self.packet = pack_episode(self.staging[self.staging_index], self.metadata)
            self.staging[self.staging_index] = None
            self.metrics.interval("recording_copy", started, self.now())
        if self.packet is not None and self.writer.try_submit(self.packet):
            self.packet = None
            self.staging_index += 1
        if self.staging_index == len(self.staging):
            self.staging = None

    def flush_staging(self) -> None:
        started = self.now()
        while self.staging is not None:
            self.poll()
            if self.flow is not None:
                self.flow.drain_events()
                self.engine.check_health()
            if self.staging is not None:
                time.sleep(0.005)
        self.metrics.interval("recording_staging_wait", started, self.now())

    def stage(self, episodes: list[Any]) -> None:
        if self.staging is None:
            if self.packet is not None:
                raise RuntimeError("A submission packet lost its staging owner")
            self.staging, self.staging_index = list(episodes), 0
        else:
            self.staging = self.staging[self.staging_index :] + list(episodes)
            self.staging_index = 0
        self.check_retention()
        self.pump_recording()

    def check_retention(self) -> int:
        """Count the union of raw episode owners; archive packets are separate."""
        owned = {id(row) for row in self.staging or [] if row is not None}
        owned.update(id(row) for row in self.ready_episodes.values())
        if self.block is not None:
            owned.update(id(row["raw"]) for row in self.block.episodes.values())
        if len(owned) > self.args.buffer_size:
            raise RuntimeError("Owned training/recording episodes exceeded C")
        self.peak_owned_episodes = max(self.peak_owned_episodes, len(owned))
        return len(owned)

    def staging_slots(self) -> int:
        return sum(row is not None for row in self.staging or [])

    def events(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            details = dict(row)
            stamp = int(details.pop("monotonic_ns"))
            details["lifecycle_kind"] = details.pop("kind")
            self.metrics.event("producer_lifecycle", stamp, **details)

    def requested(self, generation: int, started: float) -> None:
        stamp = int(started * 1e9)
        self.generation_starts[generation] = stamp
        self.metrics.event("generation_requested", stamp, generation=generation)

    def take(self, generation: int, *, drain: bool = False) -> list[Any]:
        from . import acquisition

        started = self.now()
        self.flow.wait_for_generation(generation)
        self.metrics.interval(
            "generation_drain_wait" if drain else "data_wait",
            started,
            self.now(),
            generation=generation,
        )
        copied = self.now()
        episodes = list(acquisition.iter_generation_episodes(self.engine, generation))
        return self.register_episodes(generation, episodes, copied)

    def register_episodes(
        self, generation: int, episodes: list[Any], copied: int
    ) -> list[Any]:
        from .payload import _numeric, numeric_episode_digest
        from .training_data import geometry_from_raw

        published = 0
        for episode in episodes:
            ordinal = int(episode["ods_episode_id"][0])
            if ordinal != len(self.registry):
                raise ValueError(
                    "Accepted episode ordinals differ from recording order"
                )
            source = [int(episode["ods_run_id"][0]), ordinal]
            timestamp = int(episode["ods_published_at_monotonic_ns"][0])
            geometry = geometry_from_raw(episode)
            xy = geometry[0, 27:29].tolist()
            digest = numeric_episode_digest(
                {
                    "state": _numeric(episode, self.metadata),
                    "actions": episode["actions"],
                    "continuity_id": episode["continuity_id"],
                }
            )
            record = {
                "ordinal": ordinal,
                "source_id": source,
                "generation": generation,
                "length": len(episode["valid"]),
                "initial_xy": xy,
                "published_ns": timestamp,
                "numeric_sha256": digest,
            }
            self.registry.append(record)
            self.metrics.register_episode(
                f"{source[0]}:{source[1]}",
                ordinal=ordinal,
                generation=generation,
                initial_xy=xy,
                published_ns=timestamp,
                frame_count=record["length"],
                metadata={"numeric_sha256": digest},
            )
            published = max(published, timestamp)
        copied_end = self.now()
        self.metrics.interval(
            "snapshot_copy", copied, copied_end, generation=generation
        )
        if len(self.registry) == generation * self.args.buffer_size:
            if generation in self.recorded_generations:
                raise RuntimeError("Generation service was recorded twice")
            begin = self.generation_starts.get(generation, self.origin_ns)
            self.metrics.interval("generation", begin, published, generation=generation)
            self.generation_seconds.append((published - begin) / 1e9)
            self.recorded_generations.add(generation)
        self.snapshot_seconds.append((copied_end - copied) / 1e9)
        self.generation = generation
        return episodes

    def bootstrap(self, episodes: list[Any]) -> None:
        import torch

        from .lerobot_io import StandardLeRobotWriter
        from .metrics import write_json
        from .training_data import make_statistics
        from .visual_policy import VisualChunkBC

        started = self.now()
        self.statistics = make_statistics(episodes, self.metadata)
        torch.manual_seed(self.args.seed + 1234)
        self.model = VisualChunkBC(
            **self.statistics,
            cameras=tuple(self.metadata["cameras"]),
            prediction_horizon=self.args.prediction_horizon,
        ).to(self.args.train_device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=1e-6, foreach=False
        )
        torch.cuda.synchronize(self.args.train_device)
        self.initial_model_hash = _digest(self.model.state_dict())
        self.normalization_hash = _digest(self.statistics)
        self.save_checkpoint("initial.pt", 0)
        self.metrics.interval("model_initialization", started, self.now())
        self.writer = StandardLeRobotWriter(
            self.output / "lerobot_dataset",
            self.metadata,
            episode_count=(
                self.args.offline_episodes if self.args.arm == "offline" else None
            ),
            queue_size=2,
            events_path=self.output / "writer-sdk-events.jsonl",
            encoder_threads=self.args.encoder_threads,
        )
        self.stage(episodes)
        write_json(
            self.output / "initial-contract.json",
            {
                "model_sha256": self.initial_model_hash,
                "normalization_sha256": self.normalization_hash,
                "initial_episodes": self.registry.copy(),
                "camera_order": list(self.metadata["cameras"]),
                "training_budget": {
                    "updates": self.args.updates,
                    "chunks": self.args.updates * self.args.batch_size,
                    "targets": self.args.updates
                    * self.args.batch_size
                    * self.args.prediction_horizon,
                },
            },
        )
        if self.serial_trace:
            self.validate_trace()

    def record_writer_intervals(self) -> None:
        """Import the joined SDK thread's real service spans, without callbacks."""
        if self.recording_log_sha256 is not None:
            raise RuntimeError("Recording service intervals were already imported")
        path = self.output / "writer-sdk-events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        phases = Counter(row.get("name") for row in rows)
        if phases != {
            "sdk_create": 1,
            "sdk_episode_write_encode": len(self.registry),
            "sdk_finalize": 1,
        }:
            raise ValueError("Recording service log is incomplete")
        previous_end, ordinal, threads = self.origin_ns, 0, set()
        for row in rows:
            if (
                row.get("schema") != "ods-sdk-writer-service/v1"
                or row.get("kind") != "interval"
                or row.get("status") != "complete"
                or row.get("worker_pid") != os.getpid()
                or type(row.get("worker_thread_id")) is not int
                or row["worker_thread_id"] <= 0
                or type(row.get("start_ns")) is not int
                or type(row.get("end_ns")) is not int
                or not previous_end <= row["start_ns"] <= row["end_ns"] <= self.now()
            ):
                raise ValueError("Recording service clock/owner/status is invalid")
            if row["name"] == "sdk_episode_write_encode":
                source = self.registry[ordinal]
                if (
                    row.get("source_id") != source["source_id"]
                    or row.get("episode_ordinal") != ordinal
                ):
                    raise ValueError("Recording service source order differs")
                ordinal += 1
            elif (
                row.get("source_id") is not None
                or row.get("episode_ordinal") is not None
            ):
                raise ValueError("Dataset-wide recording service has an episode ID")
            self.metrics.interval(
                "sdk_recording",
                row["start_ns"],
                row["end_ns"],
                sdk_phase=row["name"],
                worker_pid=row["worker_pid"],
                worker_thread_id=row["worker_thread_id"],
                source_id=row.get("source_id"),
                episode_ordinal=row.get("episode_ordinal"),
                scope=row["scope"],
                stage_intervals=row.get("stage_intervals"),
            )
            previous_end = row["end_ns"]
            threads.add(row["worker_thread_id"])
        if len(threads) != 1:
            raise ValueError("Recording service did not have one SDK-owner thread")
        self.recording_log_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def validate_trace(self) -> None:
        trace = self.serial_trace
        expected = {
            key: getattr(self.args, key)
            for key in (
                "vision",
                "seed",
                "updates",
                "batch_size",
                "buffer_size",
                "producer_envs",
                "episode_budget",
                "prediction_horizon",
            )
        }
        if trace.get("format") == "ods-visual-training-trace-v2":
            expected.update(
                handoff_mode=self.args.handoff_mode,
                encoder_threads=self.args.encoder_threads,
            )
            blocks = trace.get("retained_blocks", [])
            if not blocks or blocks[0].get("first_update") != 1:
                raise ValueError("Serial v2 trace requires retained source boundaries")
            previous_update, next_ordinal = 0, 0
            for block in blocks:
                ordinals, first = block.get("ordinals", []), block.get("first_update")
                if (
                    not ordinals
                    or ordinals
                    != list(range(next_ordinal, next_ordinal + len(ordinals)))
                    or type(first) is not int
                    or not previous_update < first <= self.args.updates
                    or self.args.batch_size % len(ordinals)
                    or len(ordinals) * self.args.episode_budget < self.args.batch_size
                    or any(
                        o // self.args.buffer_size + 1 != block.get("generation")
                        for o in ordinals
                    )
                ):
                    raise ValueError("Serial retained-block contract is invalid")
                previous_update, next_ordinal = first, next_ordinal + len(ordinals)
        if (
            trace.get("format")
            not in {"ods-visual-training-trace-v1", "ods-visual-training-trace-v2"}
            or trace["recipe"] != expected
        ):
            raise ValueError("Serial replay recipe differs from overlap")
        if (
            len(trace["updates"]) != self.args.updates
            or trace["normalization_sha256"] != self.normalization_hash
            or trace["initial_model_sha256"] != self.initial_model_hash
        ):
            raise ValueError("Serial replay normalization/initial model/budget differs")

    def save_checkpoint(self, name: str, step: int) -> None:
        import numpy as np
        import torch

        target = self.output / name
        temporary = target.with_suffix(".tmp")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "normalization": self.statistics,
                "config": _config(self.args),
                "cameras": tuple(self.metadata["cameras"]),
                "step": step,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "training_cuda": torch.cuda.get_rng_state(self.args.train_device),
                },
                "source_run_id": self.source_run_id,
            },
            temporary,
        )
        temporary.replace(target)

    def update(
        self,
        batch: dict[str, Any],
        plan: list[tuple[int, int]],
        *,
        requested_ns: int,
        generation: int,
    ) -> None:
        import torch

        if self.retained_probe is None:
            self.retained_probe = {
                key: value
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            self.retained_probe.update(
                {f"image/{key}": value for key, value in batch["images"].items()}
            )
            self.retained_probe_hash = _digest(self.retained_probe)
        transfer_start = self.now()
        geometry = batch["geometry"].to(self.args.train_device)
        actions = batch["actions"].to(self.args.train_device)
        images = {
            key: value.to(self.args.train_device)
            for key, value in batch["images"].items()
        }
        valid = batch["valid"].to(self.args.train_device)
        loss_mask = batch["loss_mask"].to(self.args.train_device)
        torch.cuda.synchronize(self.args.train_device)
        self.metrics.interval("device_transfer", transfer_start, self.now())
        training_start = self.now()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        begin.record(torch.cuda.current_stream(self.args.train_device))
        self.optimizer.zero_grad(set_to_none=True)
        arm, hand = self.model.loss(
            geometry, actions, images, loss_mask=loss_mask, valid_mask=valid
        )
        loss = (9 * arm + hand) / 10
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite real BC loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1, error_if_nonfinite=True
        )
        self.optimizer.step()
        end.record(torch.cuda.current_stream(self.args.train_device))
        end.synchronize()
        completed = self.now()
        gpu_ms = begin.elapsed_time(end)
        self.gpu_step_ms.append(gpu_ms)
        self.metrics.interval(
            "training", training_start, completed, gpu_milliseconds=gpu_ms
        )
        self.completed_updates += 1
        self.training_completed_ns = completed
        self.update_completion_ns.append(completed)
        self.ready_update_seconds.append((completed - requested_ns) / 1e9)
        if self.block is not None:
            self.block.mark_trained(plan)
        self.metrics.complete_update(
            batch,
            update=self.completed_updates,
            losses={"arm": float(arm), "gripper": float(hand), "total": float(loss)},
            requested_ns=requested_ns,
            train_started_ns=training_start,
            completed_ns=completed,
        )
        self.trace_updates.append(
            {
                "update": self.completed_updates,
                "generation": generation,
                "pairs": [list(pair) for pair in plan],
            }
        )
        if (
            self.completed_updates % 16 == 0
            or self.completed_updates == self.args.updates
        ):
            print(
                json.dumps(
                    {
                        "update": self.completed_updates,
                        "budget": self.args.updates,
                        "generation": generation,
                        "generated_episodes": len(self.registry),
                        "loss": float(loss),
                    }
                ),
                flush=True,
            )

    def retire(self) -> None:
        if self.block is None:
            return
        snapshot = self.block.snapshot()
        if (
            snapshot["reserved_chunks"]
            or snapshot["delivered_chunks"] != snapshot["trained_chunks"]
        ):
            raise ValueError("Retirement would discard an unfinished training batch")
        self.retired_ledgers.append(snapshot)
        for row in self.block.records:
            self.metrics.retire_episode(
                ":".join(map(str, row["source_id"])), retired_ns=self.now()
            )
        self.block = None

    def install(self, episodes: list[Any], *, record: bool = True) -> None:
        from .training_data import RetainedBlock

        self.block = RetainedBlock(
            episodes,
            self.metadata,
            seed=self.args.seed + 10000 + self.generation,
            episode_budget=self.args.episode_budget,
            history_length=2,
            prediction_horizon=self.args.prediction_horizon,
        )
        if record:
            self.stage(episodes)
        self.check_retention()
        self.retained_blocks.append(
            {
                "generation": self.generation,
                "ordinals": [row["ordinal"] for row in self.block.records],
                "first_update": self.completed_updates + 1,
            }
        )

    def receive_publications(self, generation: int, count: int) -> list[Any]:
        """Copy a complete producer subbatch only when owned slots are free."""
        from .acquisition import poll_generation_episodes

        if (
            self.block is not None
            or self.check_retention() + count > self.args.buffer_size
        ):
            return []
        copied = self.now()
        episodes = poll_generation_episodes(
            self.engine, generation, len(self.registry), max_count=count
        )
        if not episodes:
            return []
        if len(episodes) != count:
            raise RuntimeError(
                "Producer publication was not an aligned complete subbatch"
            )
        self.metrics.event(
            "publication_handoff",
            self.now(),
            generation=generation,
            count=count,
            generation_complete=self.engine.get_flow_status()["completed_generation"]
            >= generation,
            owned_slots_before=self.staging_slots(),
        )
        return self.register_episodes(generation, episodes, copied)

    def wait_publications(self, generation: int, count: int) -> list[Any]:
        """Wait for free retention slots and publication, propagating failures."""
        started = self.now()
        waiting_kind = "publication_wait"
        deadline = min(
            self.deadline_ns, started + int(self.args.generation_timeout * 1e9)
        )
        while self.now() < deadline:
            self.flow.poll()
            kind = (
                "recording_capacity_wait"
                if self.check_retention() + count > self.args.buffer_size
                else "publication_wait"
            )
            if kind != waiting_kind:
                self.metrics.interval(
                    waiting_kind, started, self.now(), generation=generation
                )
                started, waiting_kind = self.now(), kind
            episodes = self.receive_publications(generation, count)
            if episodes:
                self.metrics.interval(
                    waiting_kind, started, self.now(), generation=generation
                )
                return episodes
            time.sleep(0.005)
        raise TimeoutError(
            "Published episode handoff or bounded recording staging timed out"
        )

    def drain_publications(self) -> None:
        """Persist every accepted source of the final already-started generation."""
        target = self.engine.get_flow_status()["requested_generation"]
        while len(self.registry) < target * self.args.buffer_size:
            episodes = self.wait_publications(target, self.args.producer_envs)
            self.stage(episodes)
            del episodes
        self.flow.wait_for_generation(target)
        self.flush_staging()

    def collect_publications(self) -> None:
        """Stream archive submission during the cold offline acquisition phase."""
        self.initial_episodes = None
        while len(self.registry) < self.args.offline_episodes:
            status = self.engine.get_flow_status()
            target = status["requested_generation"]
            if len(self.registry) == target * self.args.buffer_size:
                self.flow.wait_for_generation(target)
                target = self.flow.request_refill()
            episodes = self.wait_publications(target, self.args.producer_envs)
            self.stage(episodes)
            del episodes
        self.drain_publications()

    def offline(self) -> None:
        from .lerobot_io import TrainingLeRobotReader, read_lerobot_manifest
        from .training_data import build_offline_plan, unpack_geometry

        episodes, self.initial_episodes = self.initial_episodes, None
        if self.args.handoff_mode == "publication":
            del episodes
            self.collect_publications()
        while self.args.handoff_mode == "generation":
            # Old staging owns its snapshot, so recording can overlap the next
            # shared-cache fill even in the collect-then-train baseline.
            target = (
                self.flow.request_refill()
                if len(self.registry) < self.args.offline_episodes
                else None
            )
            self.flush_staging()
            del episodes
            if target is None:
                break
            episodes = self.take(target)
            self.stage(episodes)
        started = self.now()
        self.writer_stats = self.writer.finalize(
            expected_episode_count=len(self.registry), progress=self.flow.poll
        )
        self.writer = None
        self.metrics.interval("recording_finalize", started, self.now())
        self.record_writer_intervals()
        self.stop_engine()
        manifest = read_lerobot_manifest(self.output / "lerobot_dataset")
        plans = build_offline_plan(
            manifest,
            self.args.updates,
            self.args.batch_size,
            self.args.seed + 10000,
            history_length=2,
            prediction_horizon=self.args.prediction_horizon,
        )
        self.reader = TrainingLeRobotReader(
            self.output / "lerobot_dataset",
            plans,
            history_length=2,
            prediction_horizon=self.args.prediction_horizon,
            num_workers=4,
            prefetch_factor=1,
            timeout=self.args.data_timeout,
            seed=self.args.seed + 10000,
        )
        for plan in plans:
            self.poll()
            started = self.now()
            batch = self.reader.next_batch()
            batch["geometry"] = unpack_geometry(batch["state"], self.metadata)
            self.metrics.interval("batch_service", started, self.now())
            worker_spans = batch["_service"]["worker_fetch_intervals"]
            if sum(row["item_count"] for row in worker_spans) != self.args.batch_size:
                raise ValueError("SDK worker service does not cover the whole batch")
            for row in worker_spans:
                if (
                    type(row["worker_pid"]) is not int
                    or row["worker_pid"] <= 0
                    or type(row["item_count"]) is not int
                    or row["item_count"] <= 0
                    or not self.origin_ns
                    <= row["start_ns"]
                    <= row["end_ns"]
                    <= self.now()
                ):
                    raise ValueError("Invalid prefetched SDK worker service interval")
                self.metrics.interval(
                    "sdk_decode",
                    row["start_ns"],
                    row["end_ns"],
                    worker_pid=row["worker_pid"],
                    item_count=row["item_count"],
                    batch_update=self.completed_updates + 1,
                    scope=row["scope"],
                )
            self.update(batch, plan, requested_ns=started, generation=0)
            del batch
        delivered = self.reader.snapshot_budget()
        trained = Counter(
            pair[0] for row in self.trace_updates for pair in row["pairs"]
        )
        self.retired_ledgers.append(
            {
                "episode_budget": None,
                "episodes": [
                    {
                        **row,
                        "ordinal": int(row["source_id"][1]),
                        "trained": trained[int(row["source_id"][1])],
                    }
                    for row in delivered["episodes"]
                ],
            }
        )
        self.reader_stats = self.reader.stats
        self.cleanup["reader"] = self.reader.close()
        self.reader = None

    def prefetch(self) -> int | None:
        """Decide at a batch boundary, before reserving the next training batch."""
        sampler = self.block.sampler
        credits = sum(
            self.args.episode_budget - sampler.delivered[key] - sampler.reserved[key]
            for key in sampler.delivered
        )
        recording_wait = 0.0
        if self.staging is not None:
            stats = self.writer.stats
            backlog = len(self.registry) - stats["written_episodes"]
            must_finish = max(0, backlog - stats["queue_capacity"] - 1)
            if must_finish:
                recording_wait = (
                    must_finish
                    * stats["sdk_write_encode_seconds"]
                    / stats["written_episodes"]
                    if stats["written_episodes"]
                    else None
                )
        handoff = (
            max(max(self.generation_seconds[-3:]), recording_wait)
            + max(self.snapshot_seconds[-3:])
            if self.generation_seconds
            and self.snapshot_seconds
            and recording_wait is not None
            else None
        )
        decision = _prefetch_decision(
            remaining_updates=self.args.updates - self.completed_updates,
            batch_size=self.args.batch_size,
            remaining_credits=credits,
            handoff_seconds=handoff,
            update_seconds=(
                min(self.ready_update_seconds[-8:])
                if len(self.ready_update_seconds) >= 3
                else None
            ),
            policy=self.args.prefetch_policy,
        )
        signature = self.generation, decision["reason"], decision["request"]
        if signature != self.last_prefetch_decision:
            self.metrics.event(
                "prefetch_decision", self.now(), generation=self.generation, **decision
            )
            self.last_prefetch_decision = signature
        return self.flow.request_refill() if decision["request"] else None

    def overlap_publications(self) -> None:
        """Train as producer subbatches arrive, retaining at most C raw sources."""
        episodes, self.initial_episodes = self.initial_episodes, None
        self.install(episodes, record=False)
        del episodes
        in_block = 0
        waiting_since: int | None = None
        waiting_origin: int | None = None
        waiting_kind = "publication_wait"
        while self.completed_updates < self.args.updates:
            self.flow.poll()
            status = self.engine.get_flow_status()
            target = status["requested_generation"]
            unread = target * self.args.buffer_size - len(self.registry)
            if not unread and status["completed_generation"] >= target:
                requested = self.prefetch()
                if requested is not None:
                    target = requested
                    unread = target * self.args.buffer_size - len(self.registry)
            count = self.args.producer_envs
            available = status["successful_episodes"] >= len(self.registry) + count
            if (
                unread
                and (self.block is None or in_block and available)
                and self.staging_slots() + count <= self.args.buffer_size
            ):
                # Accepted count is only a readiness hint. The acquisition
                # function validates actual publication and clones under lock.
                self.retire()
                episodes = self.receive_publications(target, count)
                if episodes:
                    self.install(episodes)
                    del episodes
                    in_block = 0
            if self.block is not None:
                requested_ns = self.now()
                try:
                    plan = self.block.next_plan(self.args.batch_size)
                except StopIteration:
                    plan = None
                if plan is not None:
                    if waiting_since is not None:
                        self.metrics.interval(
                            waiting_kind,
                            waiting_since,
                            requested_ns,
                            generation=target,
                        )
                        waiting_since = None
                        waiting_origin = None
                    batch = self.block.materialize(plan)
                    self.metrics.interval("batch_service", requested_ns, self.now())
                    self.update(
                        batch,
                        plan,
                        requested_ns=requested_ns,
                        generation=self.generation,
                    )
                    in_block += 1
                    del batch
                    continue
            kind = (
                "recording_capacity_wait"
                if self.staging_slots() + count > self.args.buffer_size
                else "publication_wait"
            )
            if waiting_since is None:
                waiting_since = waiting_origin = self.now()
                waiting_kind = kind
            elif kind != waiting_kind:
                self.metrics.interval(
                    waiting_kind, waiting_since, self.now(), generation=target
                )
                waiting_since, waiting_kind = self.now(), kind
            if self.now() - waiting_origin > self.args.generation_timeout * 1e9:
                raise TimeoutError(
                    "Publication or recorder backpressure exceeded its bound"
                )
            time.sleep(0.005)
        self.retire()
        self.drain_publications()

    def overlap(self) -> None:
        if self.args.handoff_mode == "publication":
            return self.overlap_publications()
        episodes, self.initial_episodes = self.initial_episodes, None
        self.install(episodes, record=False)
        del episodes
        in_block = 0
        pending = self.prefetch()
        while self.completed_updates < self.args.updates:
            self.flow.poll()
            if pending is None:
                pending = self.prefetch()
            ready = (
                pending is not None
                and self.engine.get_flow_status()["completed_generation"] >= pending
            )
            if in_block and ready:
                self.flush_staging()
                self.retire()
                self.install(self.take(pending))
                in_block = 0
                pending = self.prefetch()
            requested = self.now()
            try:
                plan = self.block.next_plan(self.args.batch_size)
            except StopIteration:
                if pending is None:
                    pending = self.flow.request_refill()
                wait_started = self.now()
                self.flow.wait_for_generation(pending)
                self.metrics.interval(
                    "quota_wait", wait_started, self.now(), generation=pending
                )
                continue
            batch = self.block.materialize(plan)
            self.metrics.interval("batch_service", requested, self.now())
            self.update(batch, plan, requested_ns=requested, generation=self.generation)
            in_block += 1
            del batch
        self.flush_staging()
        self.retire()
        if pending is not None:
            # A whole already-requested generation is drained and recorded,
            # even when none of its frames entered an optimizer update.
            tail = self.take(pending, drain=True)
            self.stage(tail)
            self.flush_staging()
            del tail

    def serial(self) -> None:
        if self.serial_trace.get("retained_blocks"):
            return self.serial_blocks()
        episodes, self.initial_episodes = self.initial_episodes, None
        self.install(episodes, record=False)
        del episodes
        for row in self.serial_trace["updates"]:
            target = row["generation"]
            if target != self.generation:
                if target != self.generation + 1:
                    raise ValueError("Serial trace skips a production generation")
                # The previous optimizer work has synchronized. Block only
                # generation/training overlap, not generation/recording overlap.
                self.flow.request_refill()
                self.flush_staging()
                self.retire()
                self.install(self.take(target))
            self.flow.poll()
            requested = self.now()
            plan = [tuple(pair) for pair in row["pairs"]]
            batch = self.block.materialize(plan, reserve=True)
            self.metrics.interval("batch_service", requested, self.now())
            self.update(batch, plan, requested_ns=requested, generation=target)
            del batch
        while self.generation < self.serial_trace["generated_generations"]:
            target = self.flow.request_refill()
            self.flush_staging()
            self.retire()
            tail = self.take(target, drain=True)
            self.stage(tail)
            del tail
        self.flush_staging()
        self.retire()
        expected = [
            (row["length"], row["numeric_sha256"])
            for row in self.serial_trace["episodes"]
        ]
        actual = [(row["length"], row["numeric_sha256"]) for row in self.registry]
        if expected != actual or self.serial_trace["updates"] != self.trace_updates:
            raise ValueError(
                "Serial attribution did not reproduce the overlap data/work trace"
            )

    def serial_blocks(self) -> None:
        """Replay exact retained-block boundaries with no generation during updates."""
        episodes, self.initial_episodes = self.initial_episodes, None
        self.install(episodes, record=False)
        del episodes
        blocks = {
            row["first_update"]: row for row in self.serial_trace["retained_blocks"]
        }
        for row in self.serial_trace["updates"]:
            spec = blocks.get(row["update"])
            if spec is not None and row["update"] != 1:
                self.retire()
                target = spec["generation"]
                current = self.engine.get_flow_status()["requested_generation"]
                if target != current:
                    if (
                        target != current + 1
                        or len(self.registry) != current * self.args.buffer_size
                    ):
                        raise ValueError("Serial blocks skip unread source data")
                    if self.flow.request_refill() != target:
                        raise ValueError("Serial refill target differs from trace")
                if self.args.handoff_mode == "publication":
                    # Archiving remains concurrent with generation. Hold only
                    # the bounded raw sources needed for later traced updates.
                    while len(self.registry) < target * self.args.buffer_size:
                        accepted = self.wait_publications(
                            target, self.args.producer_envs
                        )
                        for raw in accepted:
                            ordinal = int(raw["ods_episode_id"][0])
                            if ordinal in self.ready_episodes:
                                raise ValueError("Duplicate serial retained source")
                            self.ready_episodes[ordinal] = raw
                        del raw
                        self.stage(accepted)
                        del accepted
                    self.flow.wait_for_generation(target)
                    episodes = [
                        self.ready_episodes.pop(ordinal) for ordinal in spec["ordinals"]
                    ]
                    self.install(episodes, record=False)
                else:
                    self.flow.wait_for_generation(target)
                    self.flush_staging()
                    episodes = self.wait_publications(target, len(spec["ordinals"]))
                    self.install(episodes)
                del episodes
            if spec is not None and self.retained_blocks[-1] != spec:
                raise ValueError("Serial retained source boundaries differ")
            self.flow.poll()
            requested = self.now()
            plan = [tuple(pair) for pair in row["pairs"]]
            batch = self.block.materialize(plan, reserve=True)
            self.metrics.interval("batch_service", requested, self.now())
            self.update(
                batch, plan, requested_ns=requested, generation=row["generation"]
            )
            del batch
        self.retire()
        self.ready_episodes.clear()
        self.drain_publications()
        while self.generation < self.serial_trace["generated_generations"]:
            self.flow.request_refill()
            self.drain_publications()
        expected = [
            (r["length"], r["numeric_sha256"]) for r in self.serial_trace["episodes"]
        ]
        actual = [(r["length"], r["numeric_sha256"]) for r in self.registry]
        if expected != actual or self.serial_trace["updates"] != self.trace_updates:
            raise ValueError("Serial data or update trace differs from overlap")

    def stop_engine(self) -> None:
        if self.engine is None:
            return
        started = self.now()
        self.engine_stats = self.engine.get_stats()
        process = self.engine._sim_process
        # Engine.stop() closes the multiprocessing.Process handle. Retain its
        # joined Popen status so native aborts are still distinguishable from 0.
        popen = None if process is None else process._popen
        self.engine.stop()
        self.flow.drain_events()
        exitcode = None if popen is None else popen.returncode
        self.cleanup["producer"] = {
            "joined": self.engine._sim_process is None and exitcode is not None,
            "exitcode": exitcode,
        }
        if exitcode != 0:
            raise RuntimeError(
                f"Producer shutdown was not clean: {self.cleanup['producer']}"
            )
        self.engine = self.flow = None
        self.metrics.interval("producer_shutdown", started, self.now())

    def execute(self) -> dict[str, Any]:
        import numpy as np
        import torch

        from embodichain.data_pipeline.runtime import (
            ODSFlowController,
            ODSFlowControllerCfg,
        )
        from . import acquisition
        from .metrics import write_json
        from .payload import describe_raw_buffer
        from .run import _environment, _schema

        args = self.args
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(args.train_device)
        write_json(self.output / "config.json", _config(args))
        write_json(self.output / "environment.json", _environment())
        self.monitor.start()
        self.engine = acquisition.make_engine(
            args.vision,
            args.sim_device,
            producer_envs=args.producer_envs,
            seed=args.seed,
            buffer_size=args.buffer_size,
            budget=None,
            max_episode_steps=args.max_episode_steps,
            initialization_timeout=args.initialization_timeout,
            camera_profile=getattr(args, "camera_profile", "benchmark"),
        )
        self.source_run_id = self.engine.run_id
        self.cache_bytes = sum(
            value.numel() * value.element_size()
            for value in self.engine.shared_buffer.values(
                include_nested=True, leaves_only=True
            )
        )
        self.metadata = describe_raw_buffer(self.engine.shared_buffer)
        self.metadata.update(action_schema=_schema()["action"], step_dt=1 / 30)
        self.metadata["action_parameters"] = self.engine.cfg.gym_config["env"][
            "actions"
        ]
        write_json(self.output / "payload_schema.json", self.metadata)
        self.flow = ODSFlowController(
            self.engine,
            ODSFlowControllerCfg(generation_timeout=args.generation_timeout),
            progress=self.poll,
            on_events=self.events,
            on_refill=self.requested,
        )
        self.generation_starts[1] = self.now()
        self.engine.start()
        write_json(
            self.output / "roles.json",
            {"learner_pid": os.getpid(), "producer_pid": self.engine._sim_process.pid},
        )
        self.flow.drain_events()
        self.initial_episodes = self.take(1)
        self.bootstrap(self.initial_episodes)
        if args.arm == "offline":
            self.offline()
        elif args.arm == "ods_overlap":
            self.overlap()
        else:
            self.serial()
        if self.retained_probe_hash != _digest(self.retained_probe):
            raise ValueError(
                "A retained delivered batch changed after source replacement"
            )
        self.retained_probe = None
        if self.writer is not None:
            started = self.now()
            self.writer_stats = self.writer.finalize(
                expected_episode_count=len(self.registry), progress=self.flow.poll
            )
            self.writer = None
            self.metrics.interval("recording_finalize", started, self.now())
            self.record_writer_intervals()
        self.stop_engine()
        if self.recording_log_sha256 is None:
            raise ValueError("Recording SDK service intervals were not verified")
        if self.completed_updates != args.updates:
            raise ValueError("The actual optimizer budget is incomplete")
        final_hash = _digest(self.model.state_dict())
        if final_hash == self.initial_model_hash:
            raise ValueError("Real optimizer work did not update model parameters")
        self.save_checkpoint("final.pt", self.completed_updates)
        recipe = {
            key: getattr(args, key)
            for key in (
                "vision",
                "seed",
                "updates",
                "batch_size",
                "buffer_size",
                "producer_envs",
                "episode_budget",
                "prediction_horizon",
            )
        }
        recipe.update(
            handoff_mode=args.handoff_mode, encoder_threads=args.encoder_threads
        )
        write_json(
            self.output / "training-trace.json",
            {
                "format": "ods-visual-training-trace-v2",
                "recipe": recipe,
                "initial_model_sha256": self.initial_model_hash,
                "normalization_sha256": self.normalization_hash,
                "generated_generations": self.generation,
                "episodes": self.registry,
                "updates": self.trace_updates,
                "retained_blocks": self.retained_blocks,
            },
        )
        ledger_rows = [
            row for snapshot in self.retired_ledgers for row in snapshot["episodes"]
        ]
        observed = {tuple(row["source_id"]) for row in ledger_rows}
        for source in self.registry:
            if tuple(source["source_id"]) not in observed:
                ledger_rows.append(
                    {
                        "source_id": source["source_id"],
                        "ordinal": source["ordinal"],
                        "reserved": 0,
                        "delivered": 0,
                        "trained": 0,
                    }
                )
        if sum(
            row["trained"] for row in ledger_rows
        ) != args.updates * args.batch_size or any(
            row["reserved"]
            or row["delivered"] != row["trained"]
            or (args.arm != "offline" and row["delivered"] > args.episode_budget)
            for row in ledger_rows
        ):
            raise ValueError("Lifetime source ledger differs from completed training")
        self.resources = self.monitor.close()
        gpu_activity = self.resources.get("primary_gpu_utilization", {}).values()
        sampled_roles = {
            role
            for device in gpu_activity
            if device["sample_count"] > 0
            for role in device["roles"]
        }
        if (
            self.resources.get("observer_complete") is not True
            or self.resources.get("cpu_utilization", {}).get("valid_interval_count", 0)
            < 1
            or not {"simulation", "training"}.issubset(sampled_roles)
        ):
            raise RuntimeError("Timed CPU/GPU resource observations are incomplete")
        dataset_bytes = sum(
            path.stat().st_size
            for path in (self.output / "lerobot_dataset").rglob("*")
            if path.is_file()
        )
        checkpoint_bytes = sum(
            (self.output / name).stat().st_size for name in ("initial.pt", "final.pt")
        )
        total_seconds = (self.now() - self.origin_ns) / 1e9
        result = self.metrics.summary(
            total_seconds=total_seconds,
            training_complete_seconds=(self.training_completed_ns - self.origin_ns)
            / 1e9,
            ledger={"episodes": ledger_rows},
        )
        publication_end = {
            generation: max(
                row["published_ns"]
                for row in self.registry
                if row["generation"] == generation
            )
            for generation in self.recorded_generations
        }
        early_updates = sum(
            row["generation"] > 1
            and completed < publication_end.get(row["generation"], 0)
            for row, completed in zip(self.trace_updates, self.update_completion_ns)
        )
        result.update(
            timeline_contract="sdk-worker-service/v1",
            writer_sdk_events_sha256=self.recording_log_sha256,
            dataset_bytes=dataset_bytes,
            checkpoint_bytes=checkpoint_bytes,
            storage_scope="logical finalized dataset and initial/final checkpoint file sizes; excludes logs, temporary-file peaks and post-timing audit/evaluation artifacts",
            generated_episodes=len(self.registry),
            gpu_step_milliseconds=self.gpu_step_ms,
            writer=self.writer_stats,
            resources=self.resources,
            initial_model_sha256=self.initial_model_hash,
            final_model_sha256=final_hash,
            normalization_sha256=self.normalization_hash,
            cache_bytes=self.cache_bytes,
            peak_owned_raw_episodes=self.peak_owned_episodes,
            owned_raw_episode_limit=args.buffer_size,
            updates_before_generation_fully_published=early_updates,
            engine_stats=self.engine_stats,
        )
        self.metrics.close()
        self.closed = True
        return {
            "status": "complete",
            "qualification_passed": True,
            "config": _config(args),
            "training_metrics": result,
            "cleanup": self.cleanup,
        }

    def failure(self, error: BaseException) -> dict[str, Any]:
        errors = []
        for label, operation in (
            ("reader", lambda: self.reader.close() if self.reader else None),
            ("writer", lambda: self.writer.abort() if self.writer else None),
            ("producer", lambda: self.engine.stop() if self.engine else None),
        ):
            try:
                operation()
            except BaseException as cleanup_error:
                errors.append(f"{label}: {cleanup_error!r}")
        resources = self.monitor.close()
        now = self.now()
        partial = self.metrics.summary(
            total_seconds=(now - self.origin_ns) / 1e9,
            training_complete_seconds=(self.training_completed_ns - self.origin_ns)
            / 1e9,
        )
        partial["resources"] = resources
        self.metrics.close()
        self.closed = True
        return {
            "status": "failed",
            "qualification_passed": False,
            "config": _config(self.args),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "cleanup_errors": errors,
            "training_metrics": partial,
        }


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    """Run one real finite learner pipeline and preserve failures as evidence."""
    from .metrics import write_json

    _validate(args)
    experiment = _Experiment(args)
    try:
        result = experiment.execute()
    except BaseException as error:
        result = experiment.failure(error)
    write_json(args.output / "summary.json", result)
    return result


def main() -> None:
    """Launch a qualified two-device suite or its private finite worker."""
    args = build_parser().parse_args()
    if args.worker:
        result = run_one(args)
    else:
        from .training_suite import run_suite

        result = run_suite(args)
    if result.get("status") not in ("complete", "passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
