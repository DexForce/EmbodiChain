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

"""Run real ODS and standard-decoded LeRobot delivery without a learner.

Run ``python -m examples.ods.data_stream_demo.run --help`` for the isolated
single-run interface and supervised smoke/calibration/report suites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from embodichain.data_pipeline.datasets import (
    DataBudgetExhaustedError,
    OnlineDataSource,
    SequenceSpecCfg,
)
from embodichain.data_pipeline.runtime import ODSFlowController, ODSFlowControllerCfg

from . import acquisition
from .lerobot_io import (
    StandardLeRobotReader,
    StandardLeRobotWriter,
    audit_video_quality,
)
from .metrics import ResourceMonitor, StreamMetrics, validate_batch, write_json
from .payload import (
    describe_raw_buffer,
    normalize_online_batch,
    numeric_episode_digest,
    pack_episode,
)

__all__ = ["build_parser", "run_one", "main"]

ARMS = (
    "lerobot",
    "streaming",
    "protected",
    "frozen",
    "lerobot_unbounded",
    "lerobot_same_cache",
)


class _ObservationComplete(Exception):
    pass


class _SupplyExhausted(Exception):
    pass


class _MeasuredOnlineSource(OnlineDataSource):
    """Time successful immediate sampling separately from controller retries."""

    last_service_seconds: float = 0.0

    def sample_context(
        self,
        batch_size: int,
        spec: SequenceSpecCfg,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        started = time.monotonic()
        result = super().sample_context(batch_size, spec, generator=generator)
        self.last_service_seconds = time.monotonic() - started
        return result


def build_parser() -> argparse.ArgumentParser:
    """Build explicit acquisition, decoding and finite-run settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("single", "smoke", "calibrate", "report", "appendix"),
        default="single",
    )
    parser.add_argument("--arm", choices=ARMS, default="protected")
    parser.add_argument(
        "--vision", choices=("state", "single", "dual"), default="state"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--gpu-ids", nargs="+", type=int)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=19001)
    parser.add_argument("--buffer-size", type=int, default=8)
    parser.add_argument("--producer-envs", type=int, default=2)
    parser.add_argument("--accepted-episodes", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--prediction-horizon", type=int, default=32)
    parser.add_argument("--episode-budget", type=int, default=128)
    parser.add_argument("--request-rate", type=float, default=64.0)
    parser.add_argument("--duration-seconds", type=float, default=180.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--data-timeout", type=float, default=120.0)
    parser.add_argument("--generation-timeout", type=float, default=1800.0)
    parser.add_argument("--initialization-timeout", type=float, default=1800.0)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--full-image-check", action="store_true")
    parser.add_argument(
        "--calibration-run", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--suite-tag",
        choices=("single", "smoke", "calibration", "report", "appendix"),
        default="single",
        help=argparse.SUPPRESS,
    )
    return parser


def _environment() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    sources = [
        *Path(__file__).parent.glob("*.py"),
        root / "examples/ods/simple_bc_demo/environment.py",
        root / "embodichain/data_pipeline/engine/data.py",
        root / "embodichain/data_pipeline/runtime.py",
    ]
    packages = {}
    for package in ("torch", "lerobot", "torchcodec", "av", "dexsim"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "packages": packages,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "cache_policy": "fresh processes; warm software/filesystem caches; no global cache purge",
    }


def _schema() -> dict[str, Any]:
    return {
        "observation": {"version": "data-stream-raw-v1", "timing": "pre_action"},
        "action": {
            "semantics": "policy_request",
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
            "step_dt": 1 / 30,
            "terms": [
                {"term": {"contract": "eef_pose.relative@1"}},
                {"term": {"contract": "gripper.parallel@1"}},
            ],
        },
    }


def _disk_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _initial_manifest(engine: Any, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Hash initial numeric episodes under the copy lock without cloning images."""
    records = []
    with engine._lock_index.get_lock():
        if tuple(engine._lock_index[:]) != (-1, -1):
            raise RuntimeError("Initial manifest requires an idle completed fill")
        for slot in range(engine.buffer_size):
            row = engine.shared_buffer[slot]
            valid = row["valid"].bool()
            state = torch.cat(
                [
                    row[("obs", *feature["path"])][valid].reshape(-1, feature["width"])
                    for feature in metadata["state_features"]
                ],
                -1,
            )
            numeric = {
                "state": state,
                "actions": row["actions"][valid].clone(),
                "continuity_id": row["continuity_id"][valid].clone(),
            }
            records.append(
                {
                    "source_id": [
                        int(row["ods_run_id"][valid][0]),
                        int(row["ods_episode_id"][valid][0]),
                    ],
                    "generation": int(row["ods_generation"][valid][0]),
                    "steps": int(valid.sum()),
                    "numeric_sha256": numeric_episode_digest(numeric),
                }
            )
    return records


def _validate_args(args: argparse.Namespace) -> None:
    for key in (
        "buffer_size",
        "producer_envs",
        "accepted_episodes",
        "batch_size",
        "episode_budget",
        "history_length",
        "prediction_horizon",
        "max_episode_steps",
        "num_workers",
        "prefetch_factor",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if (
        args.buffer_size % args.producer_envs
        or args.accepted_episodes % args.buffer_size
    ):
        raise ValueError("C must be divisible by P and N by C")
    if args.accepted_episodes < args.buffer_size:
        raise ValueError("N cannot be smaller than C")
    for key in (
        "request_rate",
        "duration_seconds",
        "data_timeout",
        "generation_timeout",
        "initialization_timeout",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max_batches must be positive")
    if args.dataset_root is not None and not args.arm.startswith("lerobot"):
        raise ValueError("An existing corpus requires a LeRobot arm")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory")


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    """Run and audit one finite real-physics or standard-reader experiment.

    Args:
        args: Parsed, explicit single-run settings.

    Returns:
        Complete or failed summary, also persisted after every cleanup attempt.
    """
    _validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    quota = (
        None if args.arm in ("streaming", "lerobot_unbounded") else args.episode_budget
    )
    offline = args.arm.startswith("lerobot")
    cfg = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    cfg.update(
        suite="calibration" if args.calibration_run else args.suite_tag,
        cache_episodes=args.buffer_size,
        quota=quota,
        requested_chunks_per_second=args.request_rate,
        measurement_seconds=args.duration_seconds,
        history=args.history_length,
        future=args.prediction_horizon,
        camera_profile_version=getattr(acquisition, "CAMERA_PROFILE_VERSION", "v1"),
    )
    write_json(args.output / "config.json", cfg)
    write_json(args.output / "environment.json", _environment())
    tracker = StreamMetrics(args.output, history=args.history_length, quota=quota)
    resources = ResourceMonitor(args.output)
    resources.start()
    engine = flow = reader = writer = None
    stats: dict[str, Any] = {}
    budget: dict[str, Any] = {}
    reader_stats: dict[str, Any] = {}
    cleanup: dict[str, Any] = {"errors": []}
    metrics: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    status, error, end_reason = "complete", None, "observation_window"
    corpus = args.dataset_root
    measurement_start = None
    measured_duration = 0.0
    deadline = None
    in_sampling = False

    def sampling_progress() -> None:
        if not in_sampling:
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise _ObservationComplete
        if engine is not None and quota is not None:
            state = engine.get_flow_status()
            pending = state["requested_generation"] > state["completed_generation"]
            final = (
                args.arm == "frozen"
                or state["successful_episodes"] >= args.accepted_episodes
            )
            if (
                not pending
                and final
                and state["unlocked_remaining_chunks"] < args.batch_size
            ):
                raise _SupplyExhausted

    try:
        if corpus is None:
            engine = acquisition.make_engine(
                args.vision,
                args.sim_device,
                producer_envs=args.producer_envs,
                seed=args.seed,
                buffer_size=args.buffer_size,
                budget=None if offline else quota,
                max_episode_steps=args.max_episode_steps,
                initialization_timeout=args.initialization_timeout,
            )
            write_json(args.output / "acquisition_config.json", engine.cfg.gym_config)
            flow = ODSFlowController(
                engine,
                ODSFlowControllerCfg(
                    data_wait_timeout=args.data_timeout,
                    generation_timeout=args.generation_timeout,
                    max_generated_episodes=(
                        args.buffer_size
                        if args.arm == "frozen"
                        else args.accepted_episodes
                    ),
                ),
                on_events=tracker.events,
                on_interval=tracker.interval,
                progress=sampling_progress,
            )
            engine.start()
            flow.drain_events()
            tracker.generation_complete(1)
            metadata = describe_raw_buffer(engine.shared_buffer)
            metadata.update(action_schema=_schema()["action"], step_dt=1 / 30)
            # Explicit recording configuration supplements vector order; these
            # are policy requests, never relabeled joint/controller commands.
            metadata["action_parameters"] = engine.cfg.gym_config["env"]["actions"]
            write_json(args.output / "payload_schema.json", metadata)
            write_json(
                args.output / "initial_numeric_manifest.json",
                _initial_manifest(engine, metadata),
            )
            if offline:
                corpus = args.output / "lerobot_dataset"
                writer = StandardLeRobotWriter(
                    corpus, metadata, episode_count=args.accepted_episodes
                )
                for generation in range(
                    1, args.accepted_episodes // args.buffer_size + 1
                ):
                    flow.wait_for_generation(generation)
                    tracker.generation_complete(generation)
                    for rawrow in acquisition.iter_generation_episodes(
                        engine, generation
                    ):
                        episode = pack_episode(rawrow, metadata)
                        manifest.append(
                            {
                                "source_id": episode["frame_ids"][0, :2].tolist(),
                                "steps": len(episode["valid"]),
                                "generation": generation,
                                "numeric_sha256": numeric_episode_digest(episode),
                            }
                        )
                        writer.submit(episode, progress=flow.poll)
                        del episode, rawrow
                    if generation < args.accepted_episodes // args.buffer_size:
                        requested = flow.request_refill()
                        if requested != generation + 1:
                            raise RuntimeError("Offline collection generation mismatch")
                metrics["writer"] = writer.finalize(progress=flow.poll)
                writer = None
                stats = engine.get_stats()
                engine.stop()
                flow.drain_events()
                engine = flow = None
                write_json(args.output / "numeric_manifest.json", manifest)
        tracker.reader_started = tracker.elapsed()
        measurement_start = tracker.origin + tracker.reader_started
        tracker.measurement_started = tracker.reader_started
        minimum_finish = measurement_start + args.duration_seconds
        deadline = measurement_start + (
            max(args.duration_seconds, 900.0)
            if args.calibration_run and not offline
            else args.duration_seconds
        )
        if offline:
            reader = StandardLeRobotReader(
                corpus,
                batch_size=args.batch_size,
                history_length=args.history_length,
                prediction_horizon=args.prediction_horizon,
                episode_budget=quota,
                seed=args.seed + 10000,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                timeout=args.data_timeout,
                episodes=(
                    list(range(args.buffer_size))
                    if args.arm == "lerobot_same_cache"
                    else None
                ),
            )
            metadata = reader.metadata
            if args.suite_tag in ("report", "appendix"):
                expected = (
                    args.buffer_size
                    if args.arm == "lerobot_same_cache"
                    else args.accepted_episodes
                )
                if len(reader.snapshot_budget()["episodes"]) != expected:
                    raise ValueError(
                        "Formal reader corpus count differs from the accepted-data budget"
                    )
                if (
                    metadata.get("action_schema") != _schema()["action"]
                    or metadata.get("step_dt") != 1 / 30
                ):
                    raise ValueError(
                        "Formal corpus action semantics/control period differ"
                    )
            # Keep never-selected offline identities visible in the audit too.
            for row in reader.snapshot_budget()["episodes"]:
                source = ":".join(map(str, row["source_id"]))
                tracker.episodes.setdefault(
                    source,
                    {
                        "source_id": source,
                        "delivered_chunks": 0,
                        "first_consumed_seconds": None,
                        "evicted": False,
                    },
                )
        else:
            source = _MeasuredOnlineSource(engine, schema=_schema())
            sequence = SequenceSpecCfg(
                history_length=args.history_length,
                prediction_horizon=args.prediction_horizon,
            )
            rng = torch.Generator().manual_seed(args.seed + 10000)
        cameras = acquisition.camera_uids(args.vision)
        if set(metadata["cameras"]) != set(cameras):
            raise ValueError(
                "Corpus camera profile does not match the requested visual condition"
            )
        next_request = measurement_start
        exhausted = False
        batches = 0
        retained = None
        retained_digest = None
        while time.monotonic() < deadline:
            if flow is not None:
                flow.poll()
                state = engine.get_flow_status()
                completed = int(state["completed_generation"])
                tracker.generation_complete(completed)
                pending = state["requested_generation"] > completed
                if (
                    not pending
                    and args.arm != "frozen"
                    and state["successful_episodes"] < args.accepted_episodes
                ):
                    pending = flow.request_refill() is not None
                if (
                    quota is not None
                    and not pending
                    and state["unlocked_remaining_chunks"] < args.batch_size
                ):
                    exhausted = True
                    end_reason = "usage_budget"
                if (
                    args.calibration_run
                    and time.monotonic() >= minimum_finish
                    and completed >= args.accepted_episodes // args.buffer_size
                ):
                    end_reason = "calibration_complete"
                    break
            if exhausted:
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
                continue
            while time.monotonic() < min(next_request, deadline):
                if flow is not None:
                    flow.poll()
                time.sleep(
                    min(0.02, max(0.0, min(next_request, deadline) - time.monotonic()))
                )
            if time.monotonic() >= deadline:
                break
            requested_at = time.monotonic()
            next_request = requested_at + args.batch_size / args.request_rate
            try:
                in_sampling = True
                if offline:
                    batch = reader.next_batch()
                    sampled_at = time.monotonic()
                else:
                    raw = flow.sample_context(
                        source, args.batch_size, sequence, generator=rng
                    )
                    sampled_at = time.monotonic()
                    batch = normalize_online_batch(raw, metadata)
                    del raw
            except (StopIteration, _SupplyExhausted):
                exhausted = True
                end_reason = "usage_budget"
                continue
            except _ObservationComplete:
                break
            finally:
                in_sampling = False
            returned_at = time.monotonic()
            service_seconds = (
                float(batch["_service"]["worker_fetch_seconds"])
                if offline
                else source.last_service_seconds + returned_at - sampled_at
            )
            validate_batch(
                batch,
                batch_size=args.batch_size,
                history=args.history_length,
                future=args.prediction_horizon,
                cameras=cameras,
                full_images=args.full_image_check,
            )
            tracker.batch(
                batch,
                requested_at=requested_at,
                returned_at=returned_at,
                service_seconds=service_seconds,
                sampled_at=sampled_at,
                within_window=returned_at <= deadline,
            )
            if retained is None:
                # Views intentionally retain the original returned allocation.
                # This is a real ownership audit, with at most one extra batch
                # (1.55 GiB dual RGB backing storage) included in memory samples.
                retained = {
                    "state": batch["state"][0],
                    "images": {
                        name: image[0, args.history_length - 1]
                        for name, image in batch["images"].items()
                    },
                }
                retained_digest = _retained_digest(retained)
            batches += 1
            del batch
            if returned_at > deadline:
                end_reason = "observation_window"
                break
            if args.max_batches is not None and batches >= args.max_batches:
                end_reason = "batch_budget"
                break
        measured_duration = min(time.monotonic(), deadline) - measurement_start
        if retained is not None and _retained_digest(retained) != retained_digest:
            raise ValueError("An earlier owned output changed after source replacement")
        metrics["retained_output_check"] = retained is not None
        if engine is not None:
            stats = engine.get_stats()
            budget = engine.get_sampling_budget_snapshot()
            flow.drain_events()
        if reader is not None:
            reader_stats = reader.stats
            budget = reader.snapshot_budget()
        metrics.update(
            tracker.summary(measured_duration, cache_episodes=args.buffer_size)
        )
        metrics.update(
            cache_bytes=(
                sum(
                    value.numel() * value.element_size()
                    for value in engine.shared_buffer.values(
                        include_nested=True, leaves_only=True
                    )
                )
                if engine is not None
                else None
            ),
            accepted_episodes=stats.get(
                "successful_episodes",
                len(getattr(reader, "manifest", {}).get("episodes", [])),
            ),
            accepted_episodes_per_second=(
                stats.get("successful_episodes", 0) / stats["production_seconds"]
                if stats.get("production_seconds", 0) > 0
                else None
            ),
            actual_measurement_seconds=measured_duration,
            final_disk_bytes=_disk_bytes(corpus) if corpus is not None else 0,
            peak_temporary_disk_bytes=None,
            unconsumed_prefetch_chunks=(
                budget.get("reserved_chunks", 0) if offline else 0
            ),
            corpus_episodes=(
                len(reader.manifest["episodes"])
                if offline
                else stats.get("successful_episodes")
            ),
            visible_source_episodes=(
                len(budget.get("episodes", [])) if offline else None
            ),
            retained_output_backing_scope="one original final batch retained for ownership audit, included in PSS samples",
            component_service_scope=(
                "aggregate worker SDK item fetch/decode/padding before DataLoader collation; not consumer next() latency"
                if offline
                else "successful immediate Source sampling/copy plus RGB conversion; controller retry waits excluded"
            ),
        )
        if quota is not None:
            delivered = (
                budget.get("delivered_chunks")
                if offline
                else budget.get("total_delivered_chunks")
            )
            if delivered != tracker.delivered:
                raise ValueError("Source ledger and consumer deliveries disagree")
        required_generations = min(3, args.accepted_episodes // args.buffer_size)
        require_streaming = args.suite_tag == "report" and not offline
        actual_count = (
            metrics["visible_source_episodes"]
            if offline
            else stats.get("successful_episodes")
        )
        expected_count = (
            args.buffer_size
            if args.arm in ("frozen", "lerobot_same_cache")
            else args.accepted_episodes
        )
        expected_generations = expected_count // args.buffer_size
        # Acceptance is counted before the final row is published/unlocked.
        # Formal runs must prove the complete finite supply, not only N accepts.
        if stats:
            complete_supply = (
                stats.get("completed_generation") == expected_generations
                and stats.get("requested_generation") == expected_generations
                and set(tracker.publications) == set(range(1, expected_generations + 1))
                and all(
                    len(sources) == args.buffer_size
                    for sources in tracker.publications.values()
                )
                and len(
                    {
                        source
                        for sources in tracker.publications.values()
                        for source in sources
                    }
                )
                == expected_count
            )
        else:
            visible_ids = {
                tuple(row["source_id"]) for row in budget.get("episodes", [])
            }
            corpus_rows = [
                row
                for row in reader.manifest["episodes"]
                if tuple(row["source_id"]) in visible_ids
            ]
            complete_supply = (
                len(corpus_rows) == expected_count
                and len({tuple(row["source_id"]) for row in corpus_rows})
                == expected_count
                and all(
                    sum(row["generation"] == generation for row in corpus_rows)
                    == args.buffer_size
                    for generation in range(1, expected_generations + 1)
                )
            )
        checks = {
            "valid_batches": bool(tracker.window_delivered),
            "cap": metrics["cap_violations"] == 0,
            "retained_output": metrics["retained_output_check"],
            "generation_coverage": (
                len(tracker.window_generations) >= required_generations
                if require_streaming
                else True
            ),
            "partial_publication_delivery": (
                bool(tracker.overlap_generations) if require_streaming else True
            ),
            "standard_reader": (
                reader_stats.get("decoded_corpus_preloaded") is False
                if offline
                else True
            ),
            "camera_delivery": (
                (
                    reader_stats.get("requested_camera_timestamps", 0) > 0
                    if cameras
                    else True
                )
                if offline
                else True
            ),
            "accepted_data_budget": (
                actual_count == expected_count
                if args.suite_tag in ("report", "appendix")
                else True
            ),
            "complete_publication": (
                complete_supply if args.suite_tag in ("report", "appendix") else True
            ),
        }
        metrics["qualification_checks"] = checks
        metrics["qualification_passed"] = all(checks.values())
    except BaseException as failure:
        status, error = "failed", {
            "type": type(failure).__name__,
            "message": str(failure),
            "traceback": traceback.format_exc(),
        }
        elapsed = (
            min(time.monotonic(), deadline) - measurement_start
            if measurement_start is not None and deadline is not None
            else 0
        )
        metrics.update(
            tracker.summary(
                max(measured_duration, elapsed), cache_episodes=args.buffer_size
            )
        )
        metrics["qualification_passed"] = False
    finally:
        # Failure audits precede shutdown and preserve already committed chunks.
        # They must not replace the original exception or prevent other cleanup.
        for name, action in (
            (
                "engine_stats",
                lambda: engine.get_stats() if engine is not None else None,
            ),
            (
                "sampling_budget",
                lambda: (
                    reader.snapshot_budget()
                    if reader is not None
                    else (
                        engine.get_sampling_budget_snapshot()
                        if engine is not None
                        else None
                    )
                ),
            ),
        ):
            try:
                value = action()
                if value is not None:
                    if name == "engine_stats":
                        stats = value
                    else:
                        budget = value
            except BaseException as failure:
                cleanup.setdefault("audit_errors", []).append(
                    {"component": name, "error": repr(failure)}
                )
        for name, action in (
            ("reader", lambda: reader.close() if reader is not None else None),
            ("writer", lambda: writer.abort() if writer is not None else None),
            ("engine", lambda: engine.stop() if engine is not None else None),
            ("events", lambda: flow.drain_events() if flow is not None else None),
        ):
            try:
                result = action()
                if result is not None:
                    cleanup[name] = result
            except BaseException as failure:
                cleanup["errors"].append({"component": name, "error": repr(failure)})
        try:
            metrics.update(resources.close())
        except BaseException as failure:
            cleanup["errors"].append({"component": "resources", "error": repr(failure)})
        tracker.close()
        if cleanup["errors"]:
            status = "failed"
            metrics["qualification_passed"] = False
        if reader is not None:
            reader_stats = reader.stats
        charged = budget.get("delivered_chunks", budget.get("total_delivered_chunks"))
        if charged is not None:
            metrics["unvalidated_delivered_chunks"] = max(
                0, charged - tracker.delivered
            )
        metrics.update(
            tracker.summary(
                measured_duration
                or (
                    min(time.monotonic(), deadline) - measurement_start
                    if deadline is not None and measurement_start is not None
                    else 0
                ),
                cache_episodes=args.buffer_size,
            )
        )
        if status == "complete" and offline and corpus is not None:
            try:
                quality = audit_video_quality(corpus)
                write_json(args.output / "video_quality.json", quality)
                metrics["video_quality"] = quality
                samples = quality.get("samples", [])
                psnr = [row["psnr_db"] for row in samples if row["psnr_db"] is not None]
                metrics["quality_psnr_db"] = sum(psnr) / len(psnr) if psnr else None
                metrics["quality_mae"] = (
                    sum(row["mae_uint8"] for row in samples) / len(samples)
                    if samples
                    else None
                )
            except BaseException as failure:
                status = "failed"
                error = {
                    "type": type(failure).__name__,
                    "message": str(failure),
                    "phase": "posthoc_quality",
                }
                metrics["qualification_passed"] = False
        summary = {
            "config": cfg,
            "status": status,
            "error": error,
            "end_reason": end_reason,
            "metrics": metrics,
            "episode_usage": list(tracker.episodes.values()),
            "engine_stats": stats,
            "sampling_budget": budget,
            "reader_stats": reader_stats,
            "cleanup": cleanup,
        }
        write_json(args.output / "summary.json", summary)
    return summary


def _retained_digest(value: dict[str, Any]) -> str:
    digest = hashlib.sha256(value["state"].contiguous().numpy().tobytes())
    for name, image in sorted(value["images"].items()):
        digest.update(name.encode())
        digest.update(image.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    """Run one experiment or delegate an isolated finite suite."""
    args = build_parser().parse_args()
    if args.suite != "single":
        from .suite import run_suite

        run_suite(args)
        return
    result = run_one(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "qualified": result["metrics"].get("qualification_passed"),
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    if result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
