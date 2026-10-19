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

"""Private experiment orchestration for the PickPlace showcase entry point."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.datasets import (
    BatchSource,
    DataBudgetExhaustedError,
    DataUnavailableError,
    OnlineDataSource,
    SequenceSpecCfg,
)

from embodichain.data_pipeline.engine.data import (
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
)
from examples.ods.simple_bc_demo.data import (
    _content_manifest,
    EfficiencyTracker,
    resource_snapshot,
)
from examples.ods.simple_bc_demo.evaluation import _evaluate_worker, _receive
from examples.ods.simple_bc_demo.observations import (
    _geometry_statistics,
    policy_observations,
)
from examples.ods.simple_bc_demo.config import load_task_config
from examples.ods.simple_bc_demo.checkpoint import load_policy, save_checkpoint
from examples.ods.simple_bc_demo.models import ChunkBC, pose_targets
from examples.ods.simple_bc_demo.usage import (
    CompactFrameSet,
    SamplingRNG,
    UsageTracker,
    retained_deferred_checkpoints,
)


class _ResourceSampler:
    """Best-effort wall-clock resource samples, never an exact peak or CPU total.

    A background thread covers owner-side waits and short runs independently of
    evaluation frequency. Collection cost is measured; this instrumentation
    requires its own performance validation before use in a timed comparison.
    """

    def __init__(
        self, cuda_devices: tuple[int, ...], interval_seconds: float = 5.0
    ) -> None:
        if not np.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Resource sampling interval must be finite and positive")
        self.cuda_devices = cuda_devices
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._records: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._collection_seconds = 0.0
        self._closed = False

    def start(self) -> None:
        if self._thread is not None or self._closed:
            raise RuntimeError("Resource sampler cannot be restarted")
        self.capture("before_worker_startup")
        self._thread = threading.Thread(
            target=self._run, name="ods-resource-sampler", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.capture("periodic")

    def capture(self, boundary: str) -> None:
        with self._lock:
            if self._closed:
                return
        started = time.monotonic()
        snapshot = None
        error_record = None
        try:
            process = psutil.Process()
            pids = [process.pid] + [
                child.pid for child in process.children(recursive=True)
            ]
            # resource_snapshot checks is_initialized before reading CUDA
            # allocator counters; it never creates a CUDA context itself.
            snapshot = resource_snapshot(pids, cuda_devices=self.cuda_devices)
            snapshot["boundary"] = boundary
            snapshot["collection_started_monotonic_seconds"] = started
            snapshot["process_tree_scope"] = "best_effort_current_descendants"
        except Exception as error:
            error_record = {
                "boundary": boundary,
                "monotonic_seconds": started,
                "error": f"{type(error).__name__}: {error}",
            }
        elapsed = time.monotonic() - started
        with self._lock:
            self._collection_seconds += elapsed
            if snapshot is not None:
                snapshot["collection_seconds"] = elapsed
                self._records.append(snapshot)
            if error_record is not None:
                error_record["collection_seconds"] = elapsed
                self._errors.append(error_record)

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                with self._lock:
                    self._closed = True
                raise RuntimeError("Resource sampler did not stop within 10 seconds")
        with self._lock:
            self._closed = True
            return list(self._records)

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started": self._thread is not None,
                "stopped": self._closed
                and (self._thread is None or not self._thread.is_alive()),
                "interval_seconds": self.interval_seconds,
                "samples": len(self._records),
                "collection_wall_seconds": self._collection_seconds,
                "errors": list(self._errors),
                "missing_pid_observations": sum(
                    len(row.get("missing_pids", [])) for row in self._records
                ),
                "scope": "sampled process-tree PSS and observed CPU counters; Torch allocator counters belong to the learner only",
                "limitations": [
                    "Memory values are sampled peaks, not exact peaks",
                    "CPU counters omit work after a process's last observation and may miss short-lived descendants",
                    "Missing processes or collection errors remain unknown, not zero",
                    "Collection wall time sums sample service durations, including any overlapping forced/periodic captures; it is not end-to-end overhead",
                    "End-to-end overhead and scheduling perturbation need workload-specific measurement",
                ],
            }


def _write_evaluation_scalars(writer: Any, report: dict[str, Any]) -> None:
    """Index posthoc evaluation by the evaluated policy, not the live learner."""
    policy_step = int(report["policy_step"])
    if policy_step < 0:
        raise ValueError("Evaluation policy_step must be nonnegative")
    for key in ("success_rate", "lift_rate", "goal_error", "episode_length"):
        writer.add_scalar(f"eval/{key}", report[key], policy_step)


class _BudgetExpired(Exception):
    pass


def _timeout(args: argparse.Namespace, tracker: EfficiencyTracker) -> float:
    if args.wall_time_budget is None:
        return 1800
    remaining = args.wall_time_budget - tracker.elapsed_seconds()
    if remaining <= 0:
        raise _BudgetExpired
    return min(1800, remaining)


def _result(
    process: mp.Process,
    results: Any,
    args: argparse.Namespace,
    tracker: EfficiencyTracker,
) -> dict[str, Any]:
    try:
        return _receive(process, results, timeout=_timeout(args, tracker))
    except TimeoutError:
        if args.wall_time_budget and tracker.elapsed_seconds() >= args.wall_time_budget:
            raise _BudgetExpired from None
        raise


def _wait(
    engine: OnlineDataEngine,
    generation: int,
    args: argparse.Namespace,
    tracker: EfficiencyTracker,
) -> None:
    try:
        engine.wait_for_generation(generation, timeout=_timeout(args, tracker))
    except TimeoutError:
        if args.wall_time_budget and tracker.elapsed_seconds() >= args.wall_time_budget:
            raise _BudgetExpired from None
        raise


def _selection(report: dict[str, Any]) -> tuple[float, float, float]:
    return report["success_rate"], report["lift_rate"], -report["goal_error"]


def _state_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(f"{name}:{cpu.dtype}:{tuple(cpu.shape)}".encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _normalization_values(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Include every input and output scale that changes policy semantics."""
    names = (
        ("mean", "std", "target_mean", "target_std")
        if isinstance(model, ChunkBC)
        else ("mean", "std")
    )
    return {name: getattr(model, name) for name in names}


def _reuse_qualification(
    path: Path, config: dict[str, Any], source_hashes: dict[str, str]
) -> dict[str, Any]:
    report = json.loads(path.read_text())
    original_config = json.loads((path.parent / "task_config.json").read_text())
    environment = json.loads((path.parent / "environment.json").read_text())

    def physical(value: dict[str, Any]) -> dict[str, Any]:
        return {
            key: entry
            for key, entry in value.items()
            if key not in {"device", "gpu_id", "seed"}
        }

    if physical(original_config) != physical(config):
        raise ValueError(
            "Expert qualification physical/controller configuration differs"
        )
    for key in (
        "embodichain/lab/gym/envs/managers/actions.py",
        "examples/ods/simple_bc_demo/environment.py",
    ):
        if environment.get("source_sha256", {}).get(key) != source_hashes[key]:
            raise ValueError("Expert qualification task/controller source differs")
    records = report["records"]
    rate = sum(bool(row["success"]) for row in records) / max(len(records), 1)
    if (
        len(records) < 40
        or len({row["seed"] for row in records}) != len(records)
        or rate < 0.95
        or rate != report["success_rate"]
    ):
        raise ValueError(
            "Reused qualification must contain at least 40 independent scenes and >=95% measured success"
        )
    return {**report, "reused_from": str(path.resolve())}


def _snapshot(engine: OnlineDataEngine) -> dict[str, torch.Tensor]:
    # A completed generation has no writer. Copy with the same publication lock
    # nevertheless, rather than exposing a view that can be overwritten later.
    with engine._lock_index.get_lock():
        batch = engine.shared_buffer.clone()
    continuity = batch.get("continuity_id", None)
    if continuity is not None and (
        (batch["valid"] & (continuity < 0)).any()
        or (
            batch["valid"][:, 1:]
            & batch["valid"][:, :-1]
            & (continuity[:, 1:] != continuity[:, :-1])
        ).any()
    ):
        raise ValueError(
            "Offline episodes must be continuous within each valid span "
            "with nonnegative continuity IDs"
        )
    return {
        "observations": policy_observations(batch["obs"], valid_mask=batch["valid"]),
        "actions": batch["actions"],
        "valid": batch["valid"],
        "frame_ids": torch.stack(
            (batch["ods_run_id"], batch["ods_episode_id"], batch["ods_frame_index"]),
            dim=-1,
        ),
        "generations": batch["ods_generation"],
        "initial_positions": batch["obs", "cube_pose"][:, 0, :3],
    }


def _engine(config: dict[str, Any], args: argparse.Namespace) -> OnlineDataEngine:
    config = copy.deepcopy(config)
    config["num_envs"] = args.producer_envs
    config["seed"] = args.producer_seed
    auto_plan = getattr(args, "auto_configuration", {}).get("configuration", {})
    engine = OnlineDataEngine(
        OnlineDataEngineCfg(
            gym_config=config,
            state_dim=9,
            action_dim=7,
            action_semantics="policy_request",
            action_contracts=("eef_pose.relative@1", "gripper.parallel@1"),
            buffer_size=args.buffer_size,
            max_episode_steps=600,
            buffer_device="cpu",
            refill_policy="manual",
            include_provenance=True,
            episode_sample_budget=auto_plan.get("episode_sample_budget"),
            episode_sample_hard_limit=auto_plan.get("episode_sample_hard_limit"),
            lifecycle_event_capacity=(
                2 * args.buffer_size if args.usage_statistics else 0
            ),
            initialization_timeout=1800,
            shutdown_timeout=30.0,
        )
    )
    if auto_plan:
        actual_bytes = sum(
            value.numel() * value.element_size()
            for value in engine.shared_buffer.values(
                include_nested=True, leaves_only=True
            )
        )
        runtime = args.auto_configuration["input"]["runtime"]
        memory = runtime.get("cache_memory_bytes")
        if memory is not None and actual_bytes > memory:
            engine.stop()
            raise ValueError(
                f"Measured ODS payload {actual_bytes} bytes exceeds cache allowance {memory}; recalibrate bytes_per_episode"
            )
        args.auto_configuration["measured_buffer_payload_bytes"] = actual_bytes
    return engine


def _schema(ready: dict[str, Any]) -> dict[str, Any]:
    names = (
        [f"qpos_{i}" for i in range(9)]
        + [f"qvel_{i}" for i in range(9)]
        + [f"tcp_{i}" for i in range(9)]
        + [f"cube_{i}" for i in range(9)]
        + [f"goal_{i}" for i in range(3)]
        + [f"relative_{i}" for i in range(6)]
        + [f"cube_velocity_{i}" for i in range(6)]
    )
    return {
        "observation": {
            "names": names,
            "version": "pick_place.geometry@2",
            "frame": "arena",
            "rotation": "matrix_first_two_columns",
        },
        "action": {
            "names": (["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]),
            "semantics": "policy_request",
            "terms": ready["action_terms"],
            "step_dt": 1 / 30,
        },
    }


def _offline_reset_positions(corpus: Any, count: int) -> list[list[float]]:
    """Recover training-reset positions from causal frame-zero observations."""
    observation = corpus.metadata["schema"]["observation"]
    if (
        observation.get("version") != "pick_place.geometry@2"
        or observation["names"][27:30] != ["cube_0", "cube_1", "cube_2"]
        or count != len(corpus.data["valid"])
        or not corpus.data["valid"][:count, 0].all()
        or not (corpus.data["frame_ids"][:count, 0, 2] == 0).all()
    ):
        raise ValueError(
            "Offline reset diagnostics require exactly the requested episodes and geometry frame zero"
        )
    return corpus.data["obs"][:count, 0, 27:30].tolist()


def _sample(
    source: BatchSource,
    count: int,
    tracker: EfficiencyTracker,
    sequence: SequenceSpecCfg,
) -> TensorDict:
    """Bound live-data waits while propagating offline quota/read failures."""
    blocked = None
    engine = getattr(source, "engine", None)
    while True:
        try:
            batch = source.sample_context(count, sequence)
            break
        except (DataUnavailableError, DataBudgetExhaustedError) as error:
            if engine is None or engine.state is not OnlineDataEngineState.READY:
                raise
            if blocked is None:
                blocked = tracker.elapsed_seconds()
            if tracker.elapsed_seconds() - blocked >= 30:
                raise TimeoutError("No eligible ODS context for 30 seconds") from error
            time.sleep(0.02)
    ended = tracker.elapsed_seconds()
    tracker.record_interval("data_wait", ended if blocked is None else blocked, ended)
    return batch


def run(args: argparse.Namespace) -> None:
    """Execute one explicitly labeled diagnostic or efficiency experiment."""
    started = time.monotonic() - (time.time() - psutil.Process().create_time())
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    torch.set_num_threads(1)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory, including when resuming")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str((args.output / "dataset_cache").resolve())
    auto_report = getattr(args, "auto_configuration", None)
    auto_plan = None if auto_report is None else auto_report["configuration"]
    if auto_report is not None:
        (args.output / "auto_configuration.json").write_text(
            json.dumps(auto_report, indent=2)
        )
    for seed_fn in (random.seed, np.random.seed, torch.manual_seed):
        seed_fn(args.seed)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = visible.split(",") if visible else None
    physical_gpus = [
        visible_ids[int(d.split(":")[-1])] if visible_ids else d.split(":")[-1]
        for d in (args.sim_device, args.train_device)
        if d.startswith("cuda:")
    ]
    tracker = EfficiencyTracker(
        args.wall_time_budget or 86400,
        allocated_gpu_ids=physical_gpus,
        start_monotonic=started,
    )
    config = load_task_config(args.sim_device, args.pose_ik)
    config["seed"] = args.seed
    (args.output / "task_config.json").write_text(json.dumps(config, indent=2))
    (args.output / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    from importlib.metadata import version

    (args.output / "splits.json").write_text(
        json.dumps(
            {
                "training_seed": args.seed,
                "producer_seed": args.producer_seed,
                "qualification_seeds": list(
                    range(10000, 10000 + args.qualification_scenes)
                ),
                "development_seeds": list(
                    range(args.dev_seed_base, args.dev_seed_base + args.dev_scenes)
                ),
                "test_seeds": list(
                    range(
                        args.test_seed_base,
                        args.test_seed_base + args.final_test_scenes,
                    )
                ),
                "test_role": "engineering_only" if args.smoke else "reserved_final",
            },
            indent=2,
        )
    )
    if args.stage != "showcase":
        split_path = args.output / "splits.json"
        splits = json.loads(split_path.read_text())
        splits["development_seeds"] = []
        splits["training_reset_diagnostic_seeds"] = list(
            range(21000, 21000 + args.buffer_size)
        )
        splits["training_reset_verification_seeds"] = list(
            range(
                22000,
                22000 + args.buffer_size * (10 if args.stage == "overfit-one" else 5),
            )
        )
        split_path.write_text(json.dumps(splits, indent=2))
    metadata = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "dexsim": version("dexsim-engine"),
        "timezone": "Asia/Shanghai",
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_visible_devices": visible,
    }
    try:
        metadata["allocated_gpu_hardware"] = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total",
                    "--format=csv,noheader",
                    "--id=" + ",".join(sorted(set(physical_gpus))),
                ],
                text=True,
                timeout=5,
            ).strip()
            if physical_gpus
            else None
        )
    except (OSError, subprocess.SubprocessError):
        metadata["allocated_gpu_hardware"] = None
    metadata["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    root = Path(__file__).resolve().parents[3]
    source_paths = [
        "examples/ods/simple_bc_demo/train.py",
        "examples/ods/simple_bc_demo/trainer.py",
        "examples/ods/simple_bc_demo/data.py",
        "examples/ods/simple_bc_demo/lerobot.py",
        "examples/ods/simple_bc_demo/models.py",
        "examples/ods/simple_bc_demo/usage.py",
        "examples/ods/simple_bc_demo/autoconfiguration.py",
        "examples/ods/simple_bc_demo/collection.py",
        "examples/ods/simple_bc_demo/flow.py",
        "examples/ods/simple_bc_demo/checkpoint.py",
        "examples/ods/simple_bc_demo/config.py",
        "examples/ods/simple_bc_demo/evaluation.py",
        "examples/ods/simple_bc_demo/observations.py",
        "examples/ods/simple_bc_demo/configs/task.yaml",
        "examples/ods/simple_bc_demo/configs/env.yaml",
        "examples/ods/simple_bc_demo/configs/franka.yaml",
        "embodichain/data_pipeline/datasets/schema.py",
        "embodichain/data_pipeline/datasets/sequence.py",
        "embodichain/data_pipeline/datasets/sequence_data.py",
        "embodichain/data_pipeline/datasets/sources.py",
        "embodichain/data_pipeline/datasets/lerobot.py",
        "embodichain/data_pipeline/datasets/_sampling.py",
        "embodichain/data_pipeline/configuration.py",
        "embodichain/data_pipeline/engine/data.py",
        "embodichain/lab/gym/envs/embodied_env.py",
        "embodichain/lab/gym/envs/base_env.py",
        "embodichain/lab/sim/objects/rigid_object.py",
        "embodichain/lab/gym/envs/managers/actions.py",
        "examples/ods/simple_bc_demo/environment.py",
    ]
    metadata["source_sha256"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in source_paths
    }
    metadata["working_tree_dirty"] = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
    metadata["hf_datasets_cache"] = os.environ.get("HF_DATASETS_CACHE")
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    context = mp.get_context("spawn")
    requests, results = context.Queue(), context.Queue()
    evaluator = context.Process(
        target=_evaluate_worker,
        args=(
            config,
            requests,
            results,
            str(args.output),
            not args.no_viewer,
            args.viser_port,
            not args.no_video,
        ),
    )
    closed_evaluators: list[dict[str, int | None]] = []

    def restart_evaluator() -> dict[str, Any]:
        """Start each deferred panel from an independent simulator process."""
        nonlocal evaluator, requests, results
        previous = evaluator
        resource_sampler.capture("before_evaluator_shutdown")
        requests.put(None)
        previous.join(30)
        if previous.is_alive():
            previous.terminate()
            previous.join(10)
        if previous.is_alive():
            previous.kill()
            previous.join(10)
        if previous.is_alive() or previous.exitcode != 0:
            raise RuntimeError("Deferred evaluator did not close cleanly")
        closed_evaluators.append({"pid": previous.pid, "exitcode": previous.exitcode})
        previous_requests, previous_results = requests, results
        requests, results = context.Queue(), context.Queue()
        evaluator = context.Process(
            target=_evaluate_worker,
            args=(
                config,
                requests,
                results,
                str(args.output),
                not args.no_viewer,
                args.viser_port,
                not args.no_video,
            ),
        )
        previous_requests.close()
        previous_results.close()
        previous_requests.join_thread()
        previous_results.join_thread()
        previous.close()
        before = tracker.elapsed_seconds()
        evaluator.start()
        panel_ready = _result(evaluator, results, args, tracker)
        resource_sampler.capture("fresh_evaluator_ready")
        tracker.record_interval("evaluation_startup", before, tracker.elapsed_seconds())
        if panel_ready != ready:
            raise ValueError("Fresh evaluator schema differs from the qualified task")
        return {
            "pid": evaluator.pid,
            "startup_seconds": tracker.elapsed_seconds() - before,
        }

    engine = None
    corpus = None
    writer = None
    tensorboard = None
    stats = None
    usage = (
        UsageTracker(episode_capacity_hint=args.buffer_size)
        if args.usage_statistics
        else None
    )
    if usage is not None:
        tracker.unique_frames = CompactFrameSet(usage.targets, accumulate=False)
    sampling_rng = SamplingRNG(args.sampling_seed)

    def save_training_checkpoint(
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        config_args: argparse.Namespace,
        run_id: int,
    ) -> None:
        save_checkpoint(
            path,
            model,
            optimizer,
            step,
            config_args,
            run_id,
            sampling_rng_state=sampling_rng.state,
        )

    def drain_usage() -> list[dict[str, Any]]:
        if engine is None or usage is None:
            return []
        events = engine.drain_lifecycle_events()
        usage.lifecycle(events)
        return events

    def save_usage(report: dict[str, Any] | None = None) -> None:
        if usage is None:
            return
        (args.output / "usage.json").write_text(
            json.dumps(
                {
                    **(usage.summary() if report is None else report),
                    "counter_backend": usage.counter_backend,
                    "availability_scope": (
                        "resident_corpus_load"
                        if corpus is not None
                        else "live_publication"
                    ),
                },
                indent=2,
            )
        )

    recorded_refills: set[int] = set()

    def request_refill() -> int:
        before = tracker.elapsed_seconds()
        generation = engine.request_refill()
        if generation not in recorded_refills:
            tracker._event("refill_requested", before, generation=generation)
            recorded_refills.add(generation)
        return generation

    end_reason = "update_budget"
    generation_counts: dict[int, int] = {}
    resource_sampler = _ResourceSampler(
        (int(args.train_device.split(":")[-1]),)
        if args.train_device.startswith("cuda:")
        else ()
    )
    resources_finalized = False

    def finish_resource_sampling() -> None:
        nonlocal resources_finalized
        if resources_finalized:
            return
        samples = resource_sampler.stop()
        for snapshot in samples:
            snapshot["elapsed_seconds"] = (
                snapshot["monotonic_seconds"] - tracker.start_monotonic
            )
            snapshot["collection_started_elapsed_seconds"] = (
                snapshot["collection_started_monotonic_seconds"]
                - tracker.start_monotonic
            )
            tracker.record_resources(snapshot)
        resources_finalized = True
        (args.output / "resources.json").write_text(
            json.dumps(
                {
                    "sampling": resource_sampler.report(),
                    "snapshots": samples,
                    "excluded_phases": [
                        "post-report final verification",
                        "final cleanup",
                    ],
                },
                indent=2,
            )
        )

    try:
        _timeout(args, tracker)
        resource_sampler.start()
        evaluator.start()
        ready = _result(evaluator, results, args, tracker)
        resource_sampler.capture("initial_evaluator_ready")
        if args.qualification_report:
            qualification = _reuse_qualification(
                args.qualification_report, config, metadata["source_sha256"]
            )
            (args.output / "expert_qualification.json").write_text(
                json.dumps(qualification, indent=2)
            )
            split_path = args.output / "splits.json"
            split_manifest = json.loads(split_path.read_text())
            split_manifest["qualification_seeds"] = [
                row["seed"] for row in qualification["records"]
            ]
            split_manifest["qualification_reused_from"] = str(args.qualification_report)
            split_path.write_text(json.dumps(split_manifest, indent=2))
        else:
            requests.put(
                {
                    "kind": "qualify",
                    "name": "expert_qualification",
                    "seeds": list(range(10000, 10000 + args.qualification_scenes)),
                }
            )
            qualification = _result(evaluator, results, args, tracker)
        if qualification["success_rate"] < (
            1.0 if len(qualification["records"]) < 40 else 0.95
        ):
            raise RuntimeError(
                "Expert physical qualification failed; see expert_qualification.json"
            )
        if args.qualification_only:
            return
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(args.output / "events"), flush_secs=5)
        if not args.no_viewer:
            with (args.output / "tensorboard.log").open("w") as stream:
                tensorboard = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tensorboard.main",
                        "--logdir",
                        str(args.output / "events"),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(args.tensorboard_port),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            print(
                f"Viser: http://127.0.0.1:{args.viser_port}\nTensorBoard: http://127.0.0.1:{args.tensorboard_port}",
                flush=True,
            )
        schema = _schema(ready)
        corpus = None
        if args.offline_data is not None:
            if args.mode not in {
                "offline-full",
                "offline-32",
                "frozen",
            }:
                raise ValueError("--offline-data requires an offline or frozen mode")
            load_started = tracker.elapsed_seconds()
            from examples.ods.simple_bc_demo.lerobot import (
                load_lerobot_corpus,
            )

            corpus = load_lerobot_corpus(
                args.offline_data,
                expected_schema=schema,
                memory_limit_bytes=args.offline_memory_limit_mib * 1024 * 1024,
            )
            tracker.record_interval(
                "offline_load", load_started, tracker.elapsed_seconds()
            )
            initial_values = corpus.data["obs"][: args.buffer_size][
                corpus.data["valid"][: args.buffer_size]
            ]
            initial_actions = corpus.data["actions"][: args.buffer_size][
                corpus.data["valid"][: args.buffer_size]
            ]
            initial_positions = (
                _offline_reset_positions(corpus, args.buffer_size)
                if args.stage != "showcase"
                else None
            )
            source_run = 0
            initial_data_content_id = _content_manifest(
                {
                    "observations": corpus.data["obs"][: args.buffer_size],
                    "actions": corpus.data["actions"][: args.buffer_size],
                    "valid": corpus.data["valid"][: args.buffer_size],
                    "frame_ids": corpus.data["frame_ids"][: args.buffer_size],
                },
                schema,
            )["content_id"]
        else:
            engine = _engine(config, args)
            if auto_report is not None:
                (args.output / "auto_configuration.json").write_text(
                    json.dumps(auto_report, indent=2)
                )
            engine.cfg.initialization_timeout = _timeout(args, tracker)
            initial_fill_started = tracker.elapsed_seconds()
            try:
                engine.start()
            except TimeoutError:
                if (
                    args.wall_time_budget
                    and tracker.elapsed_seconds() >= args.wall_time_budget
                ):
                    raise _BudgetExpired from None
                raise
            tracker.record_interval(
                "initial_fill", initial_fill_started, tracker.elapsed_seconds()
            )
            resource_sampler.capture("producer_ready")
            drain_usage()
            initial = _snapshot(engine)
            initial_data_content_id = _content_manifest(
                {
                    key: initial[key]
                    for key in ("observations", "actions", "valid", "frame_ids")
                },
                schema,
            )["content_id"]
            initial_values = initial["observations"][initial["valid"]]
            initial_actions = initial["actions"][initial["valid"]]
            initial_positions = initial["initial_positions"].tolist()
            source_run = engine.run_id
            if args.mode in {"offline-full", "offline-32"}:
                offline_episodes = getattr(args, "offline_episodes", None) or (
                    args.buffer_size * (1 if args.mode == "offline-32" else args.rounds)
                )
                from examples.ods.simple_bc_demo.collection import (
                    collect_lerobot,
                )
                from examples.ods.simple_bc_demo.lerobot import (
                    load_lerobot_corpus,
                )

                def collection_progress() -> None:
                    drain_usage()
                    _timeout(args, tracker)
                    # Also checks process liveness when a hard exit has not
                    # yet been reflected in the shared lifecycle state.
                    engine.wait_for_generation(1, timeout=0.1)

                path = args.output / "lerobot_dataset"
                collection_report = collect_lerobot(
                    engine,
                    initial,
                    path,
                    schema=schema,
                    episodes=offline_episodes,
                    snapshot=_snapshot,
                    request_refill=request_refill,
                    progress=collection_progress,
                    clock=tracker.elapsed_seconds,
                    record_interval=tracker.record_interval,
                )
                (args.output / "offline_collection.json").write_text(
                    json.dumps(collection_report, indent=2)
                )
                stats = engine.get_stats()
                resource_sampler.capture("before_offline_producer_shutdown")
                engine.stop()
                drain_usage()
                engine = None
                resource_sampler.capture("after_offline_producer_shutdown")
                before = tracker.elapsed_seconds()
                corpus = load_lerobot_corpus(
                    path,
                    expected_schema=schema,
                    memory_limit_bytes=args.offline_memory_limit_mib * 1024 * 1024,
                )
                tracker.record_interval(
                    "offline_load", before, tracker.elapsed_seconds()
                )
                if usage is not None:
                    (args.output / "producer_lifecycle.json").write_text(
                        json.dumps(
                            {
                                "events": usage.events,
                                "scope": "production cache, not resident offline corpus eviction",
                            },
                            indent=2,
                        )
                    )
                    usage = UsageTracker()
                    tracker.unique_frames = CompactFrameSet(
                        usage.targets, accumulate=False
                    )
        if (
            corpus is not None
            and getattr(args, "offline_sample_budget", None) is not None
        ):
            corpus.configure_sampling_budget(
                args.offline_sample_budget, args.offline_sample_hard_limit
            )
        if corpus is not None and usage is not None and not usage.episodes:
            eligible = corpus.data["valid"]
            if (
                args.mode in {"frozen", "offline-32"}
                and corpus.data.get("generations") is not None
            ):
                eligible = eligible & (corpus.data.get("generations") == 1)
            usage = UsageTracker(
                episode_capacity_hint=max(1, int(eligible.any(dim=1).sum()))
            )
            tracker.unique_frames = CompactFrameSet(usage.targets, accumulate=False)
            for row in range(len(corpus.data["valid"])):
                frames = torch.where(corpus.data["valid"][row])[0]
                if not len(frames):
                    continue
                identity = corpus.data["frame_ids"][row, frames[0]]
                usage.lifecycle(
                    [
                        {
                            "kind": "published",
                            "sequence": row,
                            "run_id": int(identity[0]),
                            "episode_id": int(identity[1]),
                            "generation": (
                                int(corpus.data.get("generations")[row, frames[0]])
                                if corpus.data.get("generations") is not None
                                else 1
                            ),
                            "slot": row,
                            "monotonic_ns": time.monotonic_ns(),
                            "valid_frames": len(frames),
                            "eligible_anchors": len(frames),
                            "valid_frame_indices": corpus.data["frame_ids"][
                                row, frames, 2
                            ].tolist(),
                            "eligible_anchor_indices": corpus.data["frame_ids"][
                                row, frames, 2
                            ].tolist(),
                        }
                    ]
                )
        if initial_values.numel() == 0:
            raise RuntimeError("Initial data has no valid training observations")
        mean, std = _geometry_statistics(initial_values)
        targets = pose_targets(initial_values, initial_actions)
        target_mean = targets.mean(0)
        target_std = targets.std(0, correction=0).clamp_min(0.05)
        target_mean[3:], target_std[3:] = 0, 1
        del targets
        del initial_values
        del initial_actions
        if "initial" in locals():
            del initial
        # Data generation is a child process; reset the learner RNG so offline
        # and ODS runs start from the same parameter state.
        torch.manual_seed(args.seed)
        state = None
        if args.resume:
            model, state = load_policy(str(args.resume), args.train_device)
            if getattr(model, "schema", None) != schema:
                raise ValueError(
                    "Resume checkpoint lacks a matching observation/controller schema"
                )
            if (
                model.prediction_horizon != args.prediction_horizon
                or model.action_horizon != args.action_horizon
                or model.observation_noise != args.observation_noise
            ):
                raise ValueError("Resume chunk policy configuration differs")
        else:
            model = ChunkBC(
                mean,
                std,
                target_mean,
                target_std,
                args.prediction_horizon,
                args.action_horizon,
                args.observation_noise,
            ).to(args.train_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-6
        )
        step = 0
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            step = state["step"]
            torch.set_rng_state(state["torch_rng"])
            if state.get("config", {}).get("sampling_seed") != args.sampling_seed:
                raise ValueError("Resume sampling RNG configuration differs")
            if args.sampling_seed is not None:
                if state.get("sampling_rng") is None:
                    raise ValueError(
                        "Resume checkpoint lacks its independent sampler RNG"
                    )
                sampling_rng.state = state["sampling_rng"].cpu()
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
            if state["cuda_rng"] is not None and args.train_device.startswith("cuda:"):
                if isinstance(state["cuda_rng"], dict):
                    torch.cuda.set_rng_state(
                        state["cuda_rng"]["train"], args.train_device
                    )
                else:
                    source_device = state.get("config", {}).get(
                        "train_device", args.train_device
                    )
                    torch.cuda.set_rng_state(
                        state["cuda_rng"][int(source_device.split(":")[-1])],
                        args.train_device,
                    )
        model.schema = schema
        initial_parameters = {
            n: p.detach().cpu().clone() for n, p in model.named_parameters()
        }
        (args.output / "data_manifest.json").write_text(
            json.dumps(
                {
                    "schema": schema,
                    "initial_model_sha256": _state_digest(initial_parameters),
                    "normalization_sha256": _state_digest(_normalization_values(model)),
                    "normalization_fields": sorted(_normalization_values(model)),
                    "initial_data_content_id": initial_data_content_id,
                    "run_id": source_run,
                    "source": "data-ready" if args.offline_data else "cold",
                    "offline_manifest": corpus.metadata if corpus else None,
                    "resume_data": (
                        "fresh ODS cache; source run ID changed"
                        if args.resume
                        else None
                    ),
                },
                indent=2,
            )
        )
        total_updates = args.rounds * args.updates_per_round
        dev_request = {
            "seeds": list(
                range(args.dev_seed_base, args.dev_seed_base + args.dev_scenes)
            )
        }
        if args.stage != "showcase":
            if initial_positions is None:
                raise ValueError("Small-data gates require initial reset observations")
            # Periodic evaluation diagnoses fitting on each reset once. The
            # selected policy is then verified on the full repeatability panel.
            dev_request = {
                "seeds": list(range(21000, 21000 + len(initial_positions))),
                "initial_positions": initial_positions,
            }
            (args.output / "training_reset_positions.json").write_text(
                json.dumps(initial_positions, indent=2)
            )

        def evaluate(
            path: Path, name: str, *, scheduled: bool = True
        ) -> dict[str, Any]:
            before = tracker.elapsed_seconds()
            requests.put(
                {
                    "kind": "student",
                    "name": name,
                    "checkpoint": str(path),
                    "trace": args.stage != "showcase",
                    **dev_request,
                }
            )
            try:
                report = _result(evaluator, results, args, tracker)
            finally:
                tracker.record_interval("evaluation", before, tracker.elapsed_seconds())
            if (
                scheduled
                and args.stage == "showcase"
                and args.dev_scenes >= 20
                and not args.smoke
            ):
                tracker.record_evaluation(
                    tracker.elapsed_seconds(),
                    report["success_rate"],
                    checkpoint=str(path),
                )
            elif scheduled:
                tracker._event(
                    "diagnostic_evaluation",
                    tracker.elapsed_seconds(),
                    checkpoint=str(path),
                    success_rate=report["success_rate"],
                    panel=args.stage,
                )
            _write_evaluation_scalars(writer, report)
            resource_sampler.capture("evaluation_completed")
            return report

        baseline = args.output / f"initial_step_{step}.pt"
        save_training_checkpoint(baseline, model, optimizer, step, args, source_run)
        tracker.record_checkpoint(str(baseline), policy_step=step)
        deferred_checkpoints: list[tuple[Path, str]] = []
        generation_checkpoint_paths: list[Path] = []
        if args.defer_evaluation:
            deferred_checkpoints.append((baseline, "initial_dev"))
            initial_report = {
                "success_rate": 0.0,
                "lift_rate": 0.0,
                "goal_error": float("inf"),
            }
        else:
            initial_report = evaluate(baseline, "initial_dev")
        best_report, best_path = initial_report, baseline
        initial_published = tracker.elapsed_seconds()
        if args.wall_time_budget and initial_published > args.wall_time_budget:
            best_path = None
        train_steps = 0
        pending_ids: list[torch.Tensor] = []
        recent_target_counts: deque[int] = deque(maxlen=50)
        last_saved_generation = 1
        resource_sampler.capture("training_start")
        source = (
            corpus
            if corpus is not None
            else OnlineDataSource(
                engine, schema=schema, observation_transform=policy_observations
            )
        )
        sequence = SequenceSpecCfg(
            history_length=2,
            prediction_horizon=args.prediction_horizon,
            generations=(
                (1,)
                if corpus is not None and args.mode in {"frozen", "offline-32"}
                else None
            ),
        )
        loop_started = tracker.elapsed_seconds()
        next_quota_update = time.monotonic()
        for round_index in range(args.rounds):
            model.train()
            generation = 1
            barrier = None
            if (
                round_index
                and engine is not None
                and args.mode == "online"
                and auto_plan is None
                and (
                    args.max_generated_episodes is None
                    or engine.get_stats()["successful_episodes"]
                    < args.max_generated_episodes
                )
            ):
                generation = request_refill()
                barrier = args.updates_per_round // 4 if args.mode == "online" else 0
            for update_index in range(args.updates_per_round):
                if (
                    args.wall_time_budget
                    and tracker.elapsed_seconds() >= args.wall_time_budget
                ):
                    end_reason = "wall_time_budget"
                    break
                update_started = tracker.elapsed_seconds()
                drain_usage()
                if (
                    args.defer_evaluation
                    and engine is not None
                    and getattr(
                        args, "deferred_evaluation_schedule", "updates_and_generations"
                    )
                    == "updates_and_generations"
                ):
                    completed_generation = engine.get_stats()["completed_generation"]
                    if completed_generation > last_saved_generation:
                        if pending_ids:
                            if args.train_device.startswith("cuda:"):
                                torch.cuda.synchronize(args.train_device)
                            for ids in pending_ids:
                                tracker.record_update(tracker.elapsed_seconds(), ids)
                            pending_ids.clear()
                        last_saved_generation = completed_generation
                        generation_path = (
                            args.output
                            / f"generation_{completed_generation}_step_{step}.pt"
                        )
                        save_training_checkpoint(
                            generation_path, model, optimizer, step, args, source_run
                        )
                        tracker.record_checkpoint(
                            str(generation_path), policy_step=step
                        )
                        generation_checkpoint_paths.append(generation_path)
                        deferred_checkpoints.append(
                            (
                                generation_path,
                                f"generation_{completed_generation}_publication_dev",
                            )
                        )
                if auto_plan is not None:
                    quota_wait_started = None
                    while True:
                        drain_usage()
                        publication = engine.get_stats()
                        quota = engine.get_sampling_budget_snapshot()
                        pending = (
                            publication["requested_generation"]
                            > publication["completed_generation"]
                        )
                        room = (
                            publication["successful_episodes"]
                            < args.max_generated_episodes
                        )
                        if (
                            not pending
                            and room
                            and quota["unlocked_remaining_chunks"]
                            <= auto_plan["refill_remaining_chunks"]
                        ):
                            request_refill()
                            pending = True
                        if quota["unlocked_remaining_chunks"] >= args.batch_size:
                            break
                        if not pending and not room:
                            end_reason = "usage_budget"
                            break
                        if quota_wait_started is None:
                            quota_wait_started = tracker.elapsed_seconds()
                        if tracker.elapsed_seconds() - quota_wait_started >= 30:
                            raise TimeoutError(
                                "No unlocked trajectory quota for 30 seconds"
                            )
                        _timeout(args, tracker)
                        time.sleep(0.02)
                    quota_wait_ended = tracker.elapsed_seconds()
                    tracker.record_interval(
                        "quota_wait",
                        (
                            quota_wait_ended
                            if quota_wait_started is None
                            else quota_wait_started
                        ),
                        quota_wait_ended,
                    )
                    if end_reason == "usage_budget":
                        break
                    before_pacing = tracker.elapsed_seconds()
                    while time.monotonic() < next_quota_update:
                        drain_usage()
                        _timeout(args, tracker)
                        engine.get_sampling_budget_snapshot()
                        time.sleep(
                            min(0.02, max(0.0, next_quota_update - time.monotonic()))
                        )
                    tracker.record_interval(
                        "usage_throttle", before_pacing, tracker.elapsed_seconds()
                    )
                    next_quota_update = time.monotonic() + args.batch_size / min(
                        auto_plan["training_chunks_per_second_target"],
                        auto_plan["active_chunks_per_second_cap"],
                    )
                if barrier is not None and update_index == barrier:
                    before = tracker.elapsed_seconds()
                    try:
                        if usage is None:
                            _wait(engine, generation, args, tracker)
                        else:
                            while (
                                engine.get_stats()["completed_generation"] < generation
                            ):
                                drain_usage()
                                _timeout(args, tracker)
                                if engine.state is not OnlineDataEngineState.READY:
                                    engine.wait_for_generation(generation, timeout=0.1)
                                time.sleep(0.02)
                            drain_usage()
                    finally:
                        tracker.record_interval(
                            "data_wait", before, tracker.elapsed_seconds()
                        )
                if (
                    args.wall_time_budget
                    and tracker.elapsed_seconds() >= args.wall_time_budget
                ):
                    end_reason = "wall_time_budget"
                    break
                service_started = tracker.elapsed_seconds()
                try:
                    with sampling_rng.scope():
                        batch = _sample(source, args.batch_size, tracker, sequence)
                except DataBudgetExhaustedError:
                    if corpus is None:
                        raise
                    end_reason = "usage_budget"
                    break
                if usage is not None:
                    usage.sample(batch)
                obs = batch["obs"].to(args.train_device)
                actions = batch["actions"].to(args.train_device)
                tracker.record_interval(
                    "batch_service", service_started, tracker.elapsed_seconds()
                )
                arm, hand = model.loss(
                    obs,
                    actions,
                    loss_mask=batch["loss_mask"].to(args.train_device),
                    valid_mask=batch["valid"].to(args.train_device),
                )
                optimizer.zero_grad(set_to_none=True)
                arm_dims = 9
                ((arm * arm_dims + hand) / (arm_dims + 1)).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1, error_if_nonfinite=True
                )
                optimizer.step()
                if usage is not None:
                    # Publication may happen concurrently; only count finished GPU work.
                    if args.train_device.startswith("cuda:"):
                        torch.cuda.synchronize(args.train_device)
                    completed_target_ids = usage.complete(batch)
                if usage is not None:
                    tracker.record_interval(
                        "training_service", service_started, tracker.elapsed_seconds()
                    )
                step += 1
                train_steps += 1
                target_mask = batch.get("loss_mask", None)
                supervised_ids = (
                    completed_target_ids
                    if usage is not None
                    else (
                        batch["frame_ids"]
                        if target_mask is None
                        else batch["frame_ids"][target_mask]
                    )
                )
                if usage is None:
                    pending_ids.append(supervised_ids)
                else:
                    # Usage runs synchronize above, so timestamp actual completion
                    # instead of delaying the first update until a 50-step flush.
                    tracker.record_update(tracker.elapsed_seconds(), supervised_ids)
                recent_target_counts.append(len(supervised_ids))
                if (
                    train_steps % 50 == 0
                    or train_steps % args.eval_every == 0
                    or train_steps == total_updates
                ):
                    if args.train_device.startswith("cuda:"):
                        torch.cuda.synchronize(args.train_device)
                    for ids in pending_ids:
                        tracker.record_update(tracker.elapsed_seconds(), ids)
                    pending_ids.clear()
                tracker.record_interval(
                    "training_loop", update_started, tracker.elapsed_seconds()
                )
                versions, counts = (
                    batch["generations"]
                    if target_mask is None
                    else batch["generations"][target_mask]
                ).unique(return_counts=True)
                for version, count in zip(versions.tolist(), counts.tolist()):
                    generation_counts[version] = (
                        generation_counts.get(version, 0) + count
                    )
                if train_steps % 50 == 0 or train_steps == total_updates:
                    writer.add_scalar("train/arm_loss", arm.item(), step)
                    writer.add_scalar("train/gripper_loss", hand.item(), step)
                    for version, count in generation_counts.items():
                        writer.add_scalar(
                            f"data/supervised_generation_{version}", count, step
                        )
                    writer.add_scalar(
                        "data/unique_supervised_frames",
                        len(tracker.unique_frames),
                        step,
                    )
                    window = tracker.intervals.get("training_loop", [])[-50:]
                    window_seconds = sum(end - begin for begin, end in window)
                    writer.add_scalar(
                        "data/targets_per_second",
                        sum(recent_target_counts) / max(window_seconds, 1e-9),
                        step,
                    )
                    producer_stats = engine.get_stats() if engine is not None else stats
                    if producer_stats is not None:
                        for key, value in producer_stats.items():
                            if key != "run_id":
                                writer.add_scalar(f"ods/{key}", value, step)
                        writer.add_scalar(
                            "ods/accepted_frames_per_second",
                            producer_stats["generated_frames"]
                            / max(producer_stats["production_seconds"], 1e-9),
                            step,
                        )
                if train_steps % args.eval_every == 0 or train_steps == total_updates:
                    checkpoint = args.output / f"step_{step}.pt"
                    save_training_checkpoint(
                        checkpoint, model, optimizer, step, args, source_run
                    )
                    tracker.record_checkpoint(str(checkpoint), policy_step=step)
                    if args.defer_evaluation:
                        deferred_checkpoints.append((checkpoint, f"step_{step}_dev"))
                        report = initial_report
                    else:
                        report = evaluate(checkpoint, f"step_{step}_dev")
                    in_budget = (
                        not args.wall_time_budget
                        or tracker.elapsed_seconds() <= args.wall_time_budget
                    )
                    if in_budget and (
                        best_path is None
                        or _selection(report) > _selection(best_report)
                    ):
                        best_report, best_path = report, checkpoint
                    print(
                        (
                            f"step={step} evaluation=deferred elapsed={tracker.elapsed_seconds():.1f}s"
                            if args.defer_evaluation
                            else f"step={step} success={report['success_rate']:.1%} lift={report['lift_rate']:.1%} elapsed={tracker.elapsed_seconds():.1f}s"
                        ),
                        flush=True,
                    )
            if end_reason in {"wall_time_budget", "usage_budget"}:
                break
        if args.defer_evaluation:
            if pending_ids:
                if args.train_device.startswith("cuda:"):
                    torch.cuda.synchronize(args.train_device)
                for ids in pending_ids:
                    tracker.record_update(tracker.elapsed_seconds(), ids)
                pending_ids.clear()
            latest_path = args.output / f"step_{step}.pt"
            if latest_path != baseline and not latest_path.exists():
                save_training_checkpoint(
                    latest_path, model, optimizer, step, args, source_run
                )
                tracker.record_checkpoint(str(latest_path), policy_step=step)
                deferred_checkpoints.append((latest_path, f"step_{step}_dev"))
            if latest_path != baseline:
                deferred_checkpoints = [
                    (path, name)
                    for path, name in deferred_checkpoints
                    if path != latest_path
                ]
                deferred_checkpoints.append((latest_path, f"step_{step}_dev"))
        resource_sampler.capture("training_end")
        # Cover loop bookkeeping, logging and checkpoint I/O too. Interval union
        # avoids counting per-update measurements twice; evaluation is removed
        # by EfficiencyTracker rather than omitted from the wall-clock window.
        tracker.record_interval(
            "training_loop", loop_started, tracker.elapsed_seconds()
        )
        if pending_ids:
            if args.train_device.startswith("cuda:"):
                torch.cuda.synchronize(args.train_device)
            for ids in pending_ids:
                tracker.record_update(tracker.elapsed_seconds(), ids)
        if engine is not None:
            stats = engine.get_stats()
            if auto_plan is not None:
                (args.output / "sampling_budget.json").write_text(
                    json.dumps(engine.get_sampling_budget_snapshot(), indent=2)
                )
        if corpus is not None:
            (args.output / "sampling_budget.json").write_text(
                json.dumps(corpus.get_sampling_budget_snapshot(), indent=2)
            )
        drain_usage()
        if usage is not None:
            usage_report = usage.summary()
            if auto_plan is not None:
                from examples.ods.simple_bc_demo.flow import summarize_flow

                flow = summarize_flow(
                    monotonic_origin_seconds=tracker.start_monotonic,
                    intervals=tracker.intervals,
                    refill_requests=[
                        e for e in tracker.timeline if e["kind"] == "refill_requested"
                    ],
                    episode_records=usage_report["episodes"],
                    batch_size=args.batch_size,
                    updates=train_steps,
                    loop_start=loop_started,
                    loop_end=tracker.elapsed_seconds(),
                    buffer_size=args.buffer_size,
                )
                (args.output / "flow_measurements.json").write_text(
                    json.dumps(flow, indent=2)
                )
            save_usage(usage_report)
        if args.defer_evaluation:
            if engine is not None:
                # Snapshot live availability before env.close clears its final
                # rollout slice. Keep final producer counters after shutdown.
                live_valid_frames = engine.get_stats()["valid_frames"]
                resource_sampler.capture("before_producer_shutdown")
                engine.stop()
                drain_usage()
                stats = engine.get_stats()
                stats["valid_frames"] = live_valid_frames
                engine = None
                resource_sampler.capture("after_producer_shutdown")
            training_finished_seconds = tracker.elapsed_seconds()
            original_budget = args.wall_time_budget
            args.wall_time_budget = None
            all_deferred_checkpoints = deferred_checkpoints
            deferred_checkpoints = retained_deferred_checkpoints(
                deferred_checkpoints, generation_checkpoint_paths
            )
            panel_manifest: list[dict[str, Any]] = []
            try:
                for path, name in deferred_checkpoints:
                    panel_process = restart_evaluator()
                    report = evaluate(path, name, scheduled=False)
                    panel_manifest.append(
                        {
                            "checkpoint": str(path),
                            "name": name,
                            "seeds": dev_request["seeds"],
                            **panel_process,
                        }
                    )
                    if path == baseline:
                        initial_report = report
                        best_report, best_path = report, path
                    elif _selection(report) > _selection(best_report):
                        best_report, best_path = report, path
                (args.output / "deferred_evaluation.json").write_text(
                    json.dumps(
                        {
                            "training_finished_seconds": training_finished_seconds,
                            "selection_is_posthoc": True,
                            "fresh_process_per_panel": True,
                            "stats_valid_frames_snapshot": "immediately_before_producer_shutdown",
                            "panels": panel_manifest,
                            "generation_panels_retained": (
                                3
                                if getattr(
                                    args,
                                    "deferred_evaluation_schedule",
                                    "updates_and_generations",
                                )
                                == "updates_and_generations"
                                else 0
                            ),
                            "saved_but_not_evaluated": [
                                str(path)
                                for path, _ in all_deferred_checkpoints
                                if path not in {p for p, _ in deferred_checkpoints}
                            ],
                            "checkpoints": [
                                str(path) for path, _ in deferred_checkpoints
                            ],
                            "last_checkpoint": str(deferred_checkpoints[-1][0]),
                            "last_report": report,
                        },
                        indent=2,
                    )
                )
            finally:
                args.wall_time_budget = original_budget
        small_data_report = None
        if args.stage != "showcase" and best_path is not None:
            repeats = 10 if args.stage == "overfit-one" else 5
            dev_request = {
                "seeds": list(range(22000, 22000 + len(initial_positions) * repeats)),
                "initial_positions": [
                    p for p in initial_positions for _ in range(repeats)
                ],
            }
            small_data_report = evaluate(
                best_path, "small_data_verification", scheduled=False
            )
        resource_sampler.capture("before_report_finalize")
        finish_resource_sampling()
        elapsed = tracker.elapsed_seconds()
        if args.wall_time_budget is None:
            tracker.horizon_seconds = max(elapsed, 1e-6)
        efficiency = tracker.finish(elapsed, end_reason)
        efficiency["resource_sampling"] = resource_sampler.report()
        if args.defer_evaluation:
            efficiency["training_finished_seconds"] = training_finished_seconds
        efficiency["horizon_prespecified"] = args.wall_time_budget is not None
        efficiency["data_ready"] = args.offline_data is not None
        data_paths = [
            path
            for path in (
                args.offline_data,
                args.output / "lerobot_dataset",
                args.output / "dataset_cache",
            )
            if path is not None and path.exists()
        ]
        if corpus is not None:
            data_paths.extend(
                Path(path)
                for path in (
                    corpus.metadata.get("storage", {}).get("sdk_cache_files") or []
                )
                if Path(path).exists()
            )
        data_files = {
            child.resolve()
            for path in data_paths
            for child in (path.rglob("*") if path.is_dir() else [path])
            if child.is_file()
        }
        efficiency["cost/data_disk_final_bytes"] = sum(
            path.stat().st_size for path in data_files
        )
        efficiency["cost/data_disk_peak_bytes"] = None
        efficiency["cost/data_disk_scope"] = (
            "Final explicit dataset and run-local SDK cache files; temporary write peak is unmeasured; model/evaluation logs excluded"
        )
        efficiency["data/accepted_frames"] = (
            stats["generated_frames"]
            if stats
            else (int(corpus.data["valid"].sum()) if corpus else None)
        )
        accepted = efficiency["data/accepted_frames"]
        efficiency["data/consumed_unique_fraction"] = (
            len(tracker.unique_frames) / accepted if accepted else None
        )
        efficiency["data/evicted_unused_fraction"] = None
        if usage is not None:
            efficiency["data/evicted_unused_at_run_end_fraction"] = usage.summary()[
                "evicted_unused_at_run_end_fraction"
            ]
        if args.defer_evaluation:
            efficiency["evaluation_is_posthoc"] = True
            efficiency["training_finished_seconds"] = training_finished_seconds
        efficiency["data/generation_targets"] = generation_counts
        efficiency["data/age_seconds_p50_p95"] = None
        efficiency["pipeline/usage_throttle_seconds"] = sum(
            end - begin for begin, end in tracker.intervals.get("usage_throttle", [])
        )
        efficiency["cost/data_write_read_bytes"] = None
        efficiency["cost/data_memory_peak_bytes"] = None
        for key, value in efficiency.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(key, value, step)
        writer.flush()
        (args.output / "efficiency.json").write_text(json.dumps(efficiency, indent=2))
        (args.output / "timeline.json").write_text(
            json.dumps(
                {
                    "monotonic_origin_seconds": tracker.start_monotonic,
                    "events": tracker.timeline,
                    "intervals": tracker.intervals,
                },
                indent=2,
            )
        )
        test_report = None
        threshold_test = None
        initial_test_rate = None
        test_started = time.monotonic()
        if args.final_test_scenes and best_path is not None:
            test_base = args.test_seed_base
            test_seeds = list(range(test_base, test_base + args.final_test_scenes))
            paths = [("initial_test", baseline), ("final_test", best_path)]
            target_path = efficiency["efficiency/t80_dev_checkpoint"]
            if target_path is not None:
                paths.append(("threshold_test", Path(target_path)))
            for name, path in paths:
                requests.put(
                    {
                        "kind": "student",
                        "name": name,
                        "checkpoint": str(path),
                        "seeds": test_seeds,
                        "record": True,
                    }
                )
                report = _receive(evaluator, results)
                if name == "final_test":
                    test_report = report
                elif name == "initial_test":
                    initial_test_rate = report["success_rate"]
                elif name == "threshold_test":
                    threshold_test = report
        updated = any(
            not torch.equal(initial_parameters[n], p.detach().cpu())
            for n, p in model.named_parameters()
        )
        qualified = (
            len(qualification["records"]) >= 40
            and args.dev_scenes >= 20
            and args.final_test_scenes >= 100
            and not args.smoke
            and args.stage == "showcase"
        )
        summary = {
            "policy": args.policy,
            "mode": args.mode,
            "stage": args.stage,
            "smoke": args.smoke,
            "training_seed": args.seed,
            "parameters_updated": updated,
            "training_step": step,
            "best_checkpoint": None if best_path is None else str(best_path),
            "best_dev_success": best_report["success_rate"],
            "initial_dev_success": initial_report["success_rate"],
            "final_success_rate": (
                None if test_report is None else test_report["success_rate"]
            ),
            "threshold_test_success": (
                None if threshold_test is None else threshold_test["success_rate"]
            ),
            "initial_success_rate": initial_test_rate,
            "learning_target_met": qualified
            and test_report is not None
            and test_report["success_rate"] >= 0.8
            and test_report["success_rate"] > initial_test_rate,
            "small_data_gate_met": small_data_report is not None
            and small_data_report["success_rate"] >= 0.9,
            "supervised_generations": generation_counts,
            "stats": stats,
            "efficiency": efficiency,
            "final_verification_seconds": time.monotonic() - test_started,
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    except _BudgetExpired:
        # No post-deadline result can choose a checkpoint or become equal-budget
        # quality. Cleanup is still required and is reported outside this cap.
        resource_sampler.capture("budget_expired")
        finish_resource_sampling()
        elapsed = tracker.elapsed_seconds()
        if "loop_started" in locals():
            tracker.record_interval("training_loop", loop_started, elapsed)
        efficiency = tracker.finish(elapsed, "wall_time_budget")
        efficiency["resource_sampling"] = resource_sampler.report()
        efficiency["horizon_prespecified"] = True
        efficiency["data_ready"] = args.offline_data is not None
        (args.output / "efficiency.json").write_text(json.dumps(efficiency, indent=2))
        (args.output / "timeline.json").write_text(
            json.dumps(
                {
                    "monotonic_origin_seconds": tracker.start_monotonic,
                    "events": tracker.timeline,
                    "intervals": tracker.intervals,
                },
                indent=2,
            )
        )
        (args.output / "summary.json").write_text(
            json.dumps(
                {
                    "policy": args.policy,
                    "mode": args.mode,
                    "stage": args.stage,
                    "training_seed": args.seed,
                    "smoke": args.smoke,
                    "end_reason": "wall_time_budget",
                    "final_success_rate": None,
                    "threshold_test_success": None,
                    "learning_target_met": False,
                    "best_checkpoint": (
                        str(best_path)
                        if "best_path" in locals() and best_path is not None
                        else None
                    ),
                    "efficiency": efficiency,
                },
                indent=2,
            )
        )
    except BaseException:
        import traceback

        (args.output / "failure.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": tracker.elapsed_seconds(),
                    "error": traceback.format_exc(),
                },
                indent=2,
            )
        )
        raise
    finally:
        cleanup_started = time.monotonic()
        cleanup_error = None
        resource_cleanup_error = None
        if not resources_finalized:
            try:
                resource_sampler.capture("before_failure_or_early_return_cleanup")
                finish_resource_sampling()
            except BaseException as error:
                resource_cleanup_error = error
        if engine is not None:
            try:
                engine.stop()
            except BaseException as error:
                cleanup_error = error
        if usage is not None:
            drain_usage()
            save_usage()
        if writer is not None:
            writer.close()
        if evaluator.pid is not None:
            requests.put(None)
            evaluator.join(30)
            if evaluator.is_alive():
                evaluator.terminate()
                evaluator.join(10)
            if evaluator.is_alive():
                evaluator.kill()
                evaluator.join()
        if tensorboard is not None:
            tensorboard.terminate()
            try:
                tensorboard.wait(10)
            except subprocess.TimeoutExpired:
                tensorboard.kill()
                tensorboard.wait()
        requests.close()
        results.close()
        requests.join_thread()
        results.join_thread()
        (args.output / "cleanup.json").write_text(
            json.dumps(
                {
                    "seconds": time.monotonic() - cleanup_started,
                    "owned_worker_pids": [entry["pid"] for entry in closed_evaluators]
                    + [evaluator.pid],
                    "closed_evaluators": closed_evaluators,
                    "evaluator_exitcode": evaluator.exitcode,
                    "engine_cleanup_error": (
                        None if cleanup_error is None else str(cleanup_error)
                    ),
                    "resource_sampler_cleanup_error": (
                        None
                        if resource_cleanup_error is None
                        else str(resource_cleanup_error)
                    ),
                },
                indent=2,
            )
        )
        if sys.exc_info()[0] is None:
            if cleanup_error is not None:
                raise cleanup_error
            if resource_cleanup_error is not None:
                raise resource_cleanup_error
