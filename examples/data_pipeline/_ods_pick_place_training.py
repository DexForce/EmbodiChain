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
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from embodichain.data_pipeline.engine.data import (
    OnlineDataEngine,
    OnlineDataEngineCfg,
    OnlineDataEngineState,
)
from examples.data_pipeline._ods_pick_place_benchmark import (
    EfficiencyTracker,
    export_corpus,
    load_corpus,
    resource_snapshot,
)
from examples.data_pipeline.ods_pick_place_showcase import (
    GeometryRecurrentBC,
    MLPBC,
    _evaluate_worker,
    _geometry_statistics,
    _receive,
    load_policy,
    load_task_config,
    policy_observations,
    save_checkpoint,
)
from examples.data_pipeline._ods_pick_place_policy import ChunkBC, pose_targets
from examples.data_pipeline._ods_pick_place_usage import (
    AnchorBudget,
    CompactFrameSet,
    SamplingRNG,
    UsageTracker,
    retained_deferred_checkpoints,
)


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
        "extensions/embodichain_tasks/embodichain_tasks/manipulation/pick_place.py",
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
    if (
        continuity is not None
        and (
            batch["valid"][:, 1:]
            & batch["valid"][:, :-1]
            & (continuity[:, 1:] != continuity[:, :-1])
        ).any()
    ):
        raise ValueError("Offline episodes must be continuous within each valid span")
    return {
        "observations": policy_observations(batch["obs"], "geometry"),
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
    pose = args.policy.endswith("pose")
    config = copy.deepcopy(config)
    config["num_envs"] = args.producer_envs
    config["seed"] = args.producer_seed
    return OnlineDataEngine(
        OnlineDataEngineCfg(
            gym_config=config,
            state_dim=9,
            action_dim=7 if pose else 8,
            action_semantics="policy_request" if pose else "expert_controller_command",
            action_contracts=(
                ("eef_pose.relative@1", "gripper.parallel@1") if pose else None
            ),
            buffer_size=args.buffer_size,
            max_episode_steps=600,
            buffer_device="cpu",
            refill_policy="manual",
            include_provenance=True,
            lifecycle_event_capacity=(
                2 * args.buffer_size if args.usage_statistics else 0
            ),
            initialization_timeout=1800,
            shutdown_timeout=30.0,
        )
    )


def _schema(ready: dict[str, Any], pose: bool) -> dict[str, Any]:
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
            "names": (
                ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
                if pose
                else ready["joints"]
            ),
            "semantics": "policy_request" if pose else "expert_controller_command",
            "terms": ready["action_terms"],
            "step_dt": 1 / 30,
        },
    }


def _offline_reset_positions(corpus: Any, count: int) -> list[list[float]]:
    """Recover training-reset positions from causal frame-zero observations."""
    observation = corpus.manifest["schema"]["observation"]
    if (
        observation.get("version") != "pick_place.geometry@2"
        or observation["names"][27:30] != ["cube_0", "cube_1", "cube_2"]
        or count != len(corpus.valid)
        or not corpus.valid[:count, 0].all()
        or not (corpus.frame_ids[:count, 0, 2] == 0).all()
    ):
        raise ValueError(
            "Offline reset diagnostics require exactly the requested episodes and geometry frame zero"
        )
    return corpus.observations[:count, 0, 27:30].tolist()


def _sample(
    engine: OnlineDataEngine,
    count: int,
    tracker: EfficiencyTracker,
    history: bool = False,
    chunk_horizon: int | None = None,
) -> dict[str, torch.Tensor]:
    started = tracker.elapsed_seconds()
    blocked = None
    while True:
        try:
            batch = (
                engine.sample_batch_with_context(count, 2, chunk_horizon)
                if chunk_horizon is not None
                else (
                    engine.sample_batch_with_history(count, 24, start_probability=0.25)
                    if history
                    else engine.sample_batch(count, 1)
                )
            )
            break
        except RuntimeError as error:
            if (
                engine.state is not OnlineDataEngineState.READY
                or "No unlocked valid chunk" not in str(error)
            ):
                raise
            if blocked is None:
                blocked = tracker.elapsed_seconds()
            if tracker.elapsed_seconds() - blocked >= 30:
                raise TimeoutError("No eligible ODS frame for 30 seconds") from error
            time.sleep(0.02)
    ended = tracker.elapsed_seconds()
    tracker.record_interval("data_wait", ended if blocked is None else blocked, ended)
    sequence = history or chunk_horizon is not None
    selection = (slice(None), slice(None)) if sequence else (slice(None), 0)
    return {
        "observations": policy_observations(batch["obs"], "geometry")[selection],
        "actions": batch["actions"][selection],
        "frame_ids": torch.stack(
            (batch["ods_run_id"], batch["ods_episode_id"], batch["ods_frame_index"]),
            dim=-1,
        )[selection],
        "generations": batch["ods_generation"][selection],
        **(
            {"valid": batch["valid"], "loss_mask": batch["loss_mask"]}
            if sequence
            else {}
        ),
    }


def run(args: argparse.Namespace) -> None:
    """Execute one explicitly labeled diagnostic or efficiency experiment."""
    started = time.monotonic() - (time.time() - psutil.Process().create_time())
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    torch.set_num_threads(1)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory, including when resuming")
    if args.mode == "offline-matched" and (
        args.offline_data is None or args.batch_manifest is None
    ):
        raise ValueError("offline-matched requires --offline-data and --batch-manifest")
    if args.batch_manifest and args.mode != "offline-matched":
        raise ValueError("Batch manifests are consumed only by offline-matched")
    if args.policy in {"gru-pose", "chunk-pose"} and (
        args.record_batches or args.mode == "offline-matched"
    ):
        raise ValueError(
            "Exact batch-manifest replay currently supports MLP frames; recurrent offline-full/frozen sampling is supported"
        )
    args.output.mkdir(parents=True, exist_ok=True)
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
    pose = args.policy.endswith("pose")
    recurrent = args.policy == "gru-pose"
    chunked = args.policy == "chunk-pose"
    config = load_task_config(
        args.sim_device, "pose" if pose else "joint", args.pose_ik
    )
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
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        "examples/data_pipeline/ods_pick_place_showcase.py",
        "examples/data_pipeline/_ods_pick_place_training.py",
        "examples/data_pipeline/_ods_pick_place_benchmark.py",
        "examples/data_pipeline/_ods_pick_place_lerobot.py",
        "examples/data_pipeline/_ods_pick_place_policy.py",
        "examples/data_pipeline/_ods_pick_place_usage.py",
        "embodichain/data_pipeline/engine/data.py",
        "embodichain/lab/gym/envs/embodied_env.py",
        "embodichain/lab/gym/envs/base_env.py",
        "embodichain/lab/sim/objects/rigid_object.py",
        "embodichain/lab/gym/envs/managers/actions.py",
        "extensions/embodichain_tasks/embodichain_tasks/manipulation/pick_place.py",
    ]
    metadata["source_sha256"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in source_paths
    }
    metadata["working_tree_dirty"] = bool(
        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    )
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
        tracker.record_interval("evaluation_startup", before, tracker.elapsed_seconds())
        if panel_ready != ready:
            raise ValueError("Fresh evaluator schema differs from the qualified task")
        return {
            "pid": evaluator.pid,
            "startup_seconds": tracker.elapsed_seconds() - before,
        }

    engine = None
    writer = None
    tensorboard = None
    stats = None
    usage = UsageTracker() if args.usage_statistics else None
    if usage is not None:
        tracker.unique_frames = CompactFrameSet(usage.targets, accumulate=False)
    pacing = None
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
        if pacing is not None:
            for event in events:
                pacing.publish(event)
        return events

    snapshots = []
    batch_ids = []
    end_reason = "update_budget"
    generation_counts: dict[int, int] = {}
    try:
        _timeout(args, tracker)
        evaluator.start()
        ready = _result(evaluator, results, args, tracker)
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
        schema = _schema(ready, pose)
        corpus = None
        if args.offline_data is not None:
            if args.mode not in {
                "offline-full",
                "offline-32",
                "offline-matched",
                "frozen",
            }:
                raise ValueError("--offline-data requires an offline or frozen mode")
            load_started = tracker.elapsed_seconds()
            if args.offline_data.is_dir():
                from examples.data_pipeline._ods_pick_place_lerobot import (
                    load_lerobot_corpus,
                )

                corpus = load_lerobot_corpus(args.offline_data, expected_schema=schema)
            else:
                corpus = load_corpus(args.offline_data, expected_schema=schema)
            tracker.record_interval(
                "offline_load", load_started, tracker.elapsed_seconds()
            )
            initial_values = corpus.observations[: args.buffer_size][
                corpus.valid[: args.buffer_size]
            ]
            initial_actions = corpus.actions[: args.buffer_size][
                corpus.valid[: args.buffer_size]
            ]
            initial_positions = (
                _offline_reset_positions(corpus, args.buffer_size)
                if args.stage != "showcase"
                else None
            )
            source_run = 0
        else:
            engine = _engine(config, args)
            engine.cfg.initialization_timeout = _timeout(args, tracker)
            try:
                engine.start()
            except TimeoutError:
                if (
                    args.wall_time_budget
                    and tracker.elapsed_seconds() >= args.wall_time_budget
                ):
                    raise _BudgetExpired from None
                raise
            events = drain_usage()
            if args.usage_schedule == "anchor_budget":
                pacing = AnchorBudget(
                    args.anchor_reuse,
                    args.batch_size,
                    sum(
                        event["eligible_anchors"]
                        for event in events
                        if event["kind"] == "published"
                    ),
                    prior_seconds=args.refill_prior_seconds,
                )
                for event in events:
                    pacing.publish(event)
            initial = _snapshot(engine)
            initial_values = initial["observations"][initial["valid"]]
            initial_actions = initial["actions"][initial["valid"]]
            initial_positions = initial["initial_positions"].tolist()
            source_run = engine.run_id
            if (
                args.mode in {"offline-full", "offline-32", "offline-matched"}
                or args.export_data
            ):
                snapshots.append(initial)
            if args.mode in {"offline-full", "offline-32"}:
                for _ in range(1, 1 if args.mode == "offline-32" else args.rounds):
                    generation = engine.request_refill()
                    _wait(engine, generation, args, tracker)
                    drain_usage()
                    snapshots.append(_snapshot(engine))
                path = args.export_data or args.output / "corpus.pt"
                export_corpus(
                    path,
                    **{
                        key: torch.cat([s[key] for s in snapshots])
                        for key in (
                            "observations",
                            "actions",
                            "valid",
                            "frame_ids",
                            "generations",
                        )
                    },
                    schema=schema,
                    provenance={
                        "seed": args.seed,
                        "producer_stats": engine.get_stats(),
                    },
                )
                stats = engine.get_stats()
                engine.stop()
                engine = None
                snapshots.clear()
                corpus = load_corpus(path, expected_schema=schema)
        if corpus is not None and usage is not None and not usage.episodes:
            for row in range(len(corpus.valid)):
                frames = torch.where(corpus.valid[row])[0]
                if not len(frames):
                    continue
                identity = corpus.frame_ids[row, frames[0]]
                usage.lifecycle(
                    [
                        {
                            "kind": "published",
                            "sequence": row,
                            "run_id": int(identity[0]),
                            "episode_id": int(identity[1]),
                            "generation": (
                                int(corpus.generations[row, frames[0]])
                                if corpus.generations is not None
                                else 1
                            ),
                            "slot": row,
                            "monotonic_ns": time.monotonic_ns(),
                            "valid_frames": len(frames),
                            "eligible_anchors": len(frames),
                            "valid_frame_indices": corpus.frame_ids[
                                row, frames, 2
                            ].tolist(),
                            "eligible_anchor_indices": corpus.frame_ids[
                                row, frames, 2
                            ].tolist(),
                        }
                    ]
                )
        if initial_values.numel() == 0:
            raise RuntimeError("Initial data has no valid training observations")
        mean, std = _geometry_statistics(initial_values)
        if chunked:
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
        if chunked:
            model = ChunkBC(
                mean,
                std,
                target_mean,
                target_std,
                args.prediction_horizon,
                args.action_horizon,
                args.observation_noise,
            ).to(args.train_device)
        elif recurrent:
            model = GeometryRecurrentBC(
                mean, std, torch.tensor(ready["low"]), torch.tensor(ready["high"])
            ).to(args.train_device)
        else:
            model = MLPBC(
                mean,
                std,
                torch.tensor(ready["low"]),
                torch.tensor(ready["high"]),
                arm_command_mode="pose" if pose else "residual",
            ).to(args.train_device)
        optimizer_type = torch.optim.AdamW if chunked else torch.optim.Adam
        optimizer_options = {"weight_decay": 1e-6} if chunked else {}
        optimizer = optimizer_type(
            model.parameters(), lr=args.learning_rate, **optimizer_options
        )
        step = 0
        if args.resume:
            model, state = load_policy(str(args.resume), args.train_device)
            if (
                not isinstance(model, (MLPBC, ChunkBC))
                or isinstance(model, ChunkBC) != chunked
                or isinstance(model, GeometryRecurrentBC) != recurrent
                or model.arm_command_mode != ("pose" if pose else "residual")
            ):
                raise ValueError(
                    "Resume policy/action schema does not match the requested task"
                )
            if getattr(model, "schema", None) != schema:
                raise ValueError(
                    "Resume checkpoint lacks a matching observation/controller schema"
                )
            if chunked and (
                model.prediction_horizon != args.prediction_horizon
                or model.action_horizon != args.action_horizon
                or model.observation_noise != args.observation_noise
            ):
                raise ValueError("Resume chunk policy configuration differs")
            optimizer = optimizer_type(
                model.parameters(), lr=args.learning_rate, **optimizer_options
            )
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
                    "run_id": source_run,
                    "source": "data-ready" if args.offline_data else "cold",
                    "offline_manifest": corpus.manifest if corpus else None,
                    "resume_data": (
                        "fresh ODS cache; source run ID changed"
                        if args.resume
                        else None
                    ),
                },
                indent=2,
            )
        )
        manifest = (
            torch.load(args.batch_manifest, map_location="cpu", weights_only=True)
            if args.batch_manifest
            else None
        )
        total_updates = args.rounds * args.updates_per_round
        if manifest is not None and (
            manifest.dtype != torch.int64
            or manifest.shape != (total_updates, args.batch_size, 3)
        ):
            raise ValueError(
                "Batch replay manifest must match update and batch budgets"
            )
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
            for key in ("success_rate", "lift_rate", "goal_error", "episode_length"):
                writer.add_scalar(f"eval/{key}", report[key], step)
            return report

        baseline = args.output / f"initial_step_{step}.pt"
        save_training_checkpoint(baseline, model, optimizer, step, args, source_run)
        tracker.record_checkpoint(str(baseline))
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
        loop_started = tracker.elapsed_seconds()
        for round_index in range(args.rounds):
            model.train()
            generation = 1
            barrier = None
            if (
                round_index
                and engine is not None
                and args.mode in {"online", "serial"}
                and pacing is None
                and (
                    args.max_generated_episodes is None
                    or engine.get_stats()["successful_episodes"]
                    < args.max_generated_episodes
                )
            ):
                generation = engine.request_refill()
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
                if args.defer_evaluation and engine is not None:
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
                        tracker.record_checkpoint(str(generation_path))
                        generation_checkpoint_paths.append(generation_path)
                        deferred_checkpoints.append(
                            (
                                generation_path,
                                f"generation_{completed_generation}_publication_dev",
                            )
                        )
                if pacing is not None:
                    blocked_started = None
                    while True:
                        drain_usage()
                        publication = engine.get_stats()
                        pending = (
                            publication["requested_generation"]
                            > publication["completed_generation"]
                        )
                        room = (
                            publication["successful_episodes"]
                            < args.max_generated_episodes
                        )
                        if not pending and room and pacing.should_refill():
                            requested_at = time.monotonic()
                            pacing.request(engine.request_refill(), requested_at)
                            pending = True
                        if pacing.can_update:
                            break
                        if not pending and not room:
                            # Generation completion is published after its final
                            # events. Drain again after observing terminal state
                            # so a concurrent final publication cannot lose credits.
                            drain_usage()
                            if pacing.can_update:
                                break
                            end_reason = "usage_budget"
                            break
                        if (
                            args.wall_time_budget
                            and tracker.elapsed_seconds() >= args.wall_time_budget
                        ):
                            end_reason = "wall_time_budget"
                            break
                        if engine.state is not OnlineDataEngineState.READY:
                            # Surface the original worker failure immediately.
                            engine.sample_batch_with_context(
                                1, 2, args.prediction_horizon
                            )
                        if blocked_started is None:
                            blocked_started = tracker.elapsed_seconds()
                        time.sleep(0.02)
                    throttle_ended = tracker.elapsed_seconds()
                    tracker.record_interval(
                        "usage_throttle",
                        throttle_ended if blocked_started is None else blocked_started,
                        throttle_ended,
                    )
                    if end_reason in {"usage_budget", "wall_time_budget"}:
                        break
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
                    if args.export_data:
                        snapshots.append(_snapshot(engine))
                if (
                    args.wall_time_budget
                    and tracker.elapsed_seconds() >= args.wall_time_budget
                ):
                    end_reason = "wall_time_budget"
                    break
                service_started = tracker.elapsed_seconds()
                with sampling_rng.scope():
                    if corpus is not None:
                        before = tracker.elapsed_seconds()
                        if chunked:
                            batch = corpus.sample_context(
                                args.batch_size,
                                2,
                                args.prediction_horizon,
                                generations=(
                                    [1]
                                    if args.mode in {"frozen", "offline-32"}
                                    else None
                                ),
                            )
                        elif recurrent:
                            batch = corpus.sample_histories(
                                args.batch_size,
                                generations=(
                                    [1]
                                    if args.mode in {"frozen", "offline-32"}
                                    else None
                                ),
                            )
                        else:
                            batch = (
                                corpus.sample_frame_ids(manifest[train_steps])
                                if args.mode == "offline-matched"
                                else corpus.sample_frames(
                                    args.batch_size,
                                    generations=(
                                        [1]
                                        if args.mode in {"frozen", "offline-32"}
                                        else None
                                    ),
                                )
                            )
                        # Resident tensor indexing/copy is batch service, as in ODS.
                        # Actual storage loading is timed separately above.
                        now = tracker.elapsed_seconds()
                        tracker.record_interval("data_wait", now, now)
                    else:
                        batch = _sample(
                            engine,
                            args.batch_size,
                            tracker,
                            history=recurrent,
                            chunk_horizon=args.prediction_horizon if chunked else None,
                        )
                if usage is not None:
                    usage.sample(batch)
                obs = batch["observations"].to(args.train_device)
                actions = batch["actions"].to(args.train_device)
                tracker.record_interval(
                    "batch_service", service_started, tracker.elapsed_seconds()
                )
                arm, hand = model.loss(
                    obs,
                    actions,
                    **(
                        {
                            "loss_mask": batch["loss_mask"].to(args.train_device),
                            "valid_mask": batch["valid"].to(args.train_device),
                        }
                        if chunked
                        else (
                            {"loss_mask": batch["loss_mask"].to(args.train_device)}
                            if recurrent
                            else {}
                        )
                    ),
                )
                optimizer.zero_grad(set_to_none=True)
                arm_dims = 9 if chunked else actions.shape[-1] - 1
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
                    if pacing is not None:
                        pacing.complete(args.batch_size, 0.0)
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
                if args.record_batches:
                    batch_ids.append(batch["frame_ids"].clone())
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
                if pacing is not None:
                    data_wait = tracker.intervals.get("data_wait", [])[-1:]
                    wait_seconds = sum(end - begin for begin, end in data_wait)
                    pacing.observe_rate(
                        args.batch_size,
                        tracker.elapsed_seconds() - service_started - wait_seconds,
                    )
                if train_steps % args.eval_every == 0 or train_steps == total_updates:
                    checkpoint = args.output / f"step_{step}.pt"
                    save_training_checkpoint(
                        checkpoint, model, optimizer, step, args, source_run
                    )
                    tracker.record_checkpoint(str(checkpoint))
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
                    pids = [os.getpid()] + [
                        p.pid for p in psutil.Process().children(recursive=True)
                    ]
                    tracker.record_resources(
                        resource_snapshot(
                            pids,
                            cuda_devices=(
                                [int(args.train_device.split(":")[-1])]
                                if args.train_device.startswith("cuda:")
                                else []
                            ),
                        )
                    )
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
                tracker.record_checkpoint(str(latest_path))
                deferred_checkpoints.append((latest_path, f"step_{step}_dev"))
            if latest_path != baseline:
                deferred_checkpoints = [
                    (path, name)
                    for path, name in deferred_checkpoints
                    if path != latest_path
                ]
                deferred_checkpoints.append((latest_path, f"step_{step}_dev"))
        # Cover loop bookkeeping, logging and checkpoint I/O too. Interval union
        # avoids counting per-update measurements twice; evaluation is removed
        # by EfficiencyTracker rather than omitted from the wall-clock window.
        tracker.record_interval(
            "training_loop", loop_started, tracker.elapsed_seconds()
        )
        export_complete = None
        if args.export_data and engine is not None:
            publication = engine.get_stats()
            export_complete = (
                publication["requested_generation"]
                == publication["completed_generation"]
            )
            if not export_complete:
                (args.output / "export_status.json").write_text(
                    json.dumps(
                        {
                            "complete": False,
                            "reason": "Stopped before refill publication; no replay corpus was exported",
                            "publication": publication,
                        },
                        indent=2,
                    )
                )
        if args.export_data and engine is not None and export_complete:
            export_corpus(
                args.export_data,
                **{
                    key: torch.cat([s[key] for s in snapshots])
                    for key in (
                        "observations",
                        "actions",
                        "valid",
                        "frame_ids",
                        "generations",
                    )
                },
                schema=schema,
                provenance={
                    "seed": args.seed,
                    "archival_variant": True,
                    "producer_stats": engine.get_stats(),
                },
            )
        if pending_ids:
            if args.train_device.startswith("cuda:"):
                torch.cuda.synchronize(args.train_device)
            for ids in pending_ids:
                tracker.record_update(tracker.elapsed_seconds(), ids)
        if args.record_batches:
            torch.save(
                (
                    torch.stack(batch_ids)
                    if batch_ids
                    else torch.empty((0, args.batch_size, 3), dtype=torch.int64)
                ),
                args.output / "batch_ids.pt",
            )
        if engine is not None:
            stats = engine.get_stats()
        drain_usage()
        if usage is not None:
            (args.output / "usage.json").write_text(
                json.dumps(
                    {
                        **usage.summary(),
                        "schedule": None if pacing is None else pacing.summary(),
                    },
                    indent=2,
                )
            )
        if args.defer_evaluation:
            if engine is not None:
                # Snapshot live availability before env.close clears its final
                # rollout slice. Keep final producer counters after shutdown.
                live_valid_frames = engine.get_stats()["valid_frames"]
                engine.stop()
                drain_usage()
                stats = engine.get_stats()
                stats["valid_frames"] = live_valid_frames
                engine = None
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
                            "generation_panels_retained": 3,
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
        elapsed = tracker.elapsed_seconds()
        if args.wall_time_budget is None:
            tracker.horizon_seconds = max(elapsed, 1e-6)
        efficiency = tracker.finish(elapsed, end_reason)
        efficiency["horizon_prespecified"] = args.wall_time_budget is not None
        efficiency["data_ready"] = args.offline_data is not None
        data_paths = [
            path
            for path in (args.offline_data, args.export_data, args.output / "corpus.pt")
            if path is not None and path.exists()
        ]
        data_files = {
            child.resolve()
            for path in data_paths
            for child in (path.rglob("*") if path.is_dir() else [path])
            if child.is_file()
        }
        data_files |= {
            Path(str(path) + ".manifest.json")
            for path in list(data_files)
            if Path(str(path) + ".manifest.json").exists()
        }
        efficiency["cost/data_disk_peak_bytes"] = sum(
            path.stat().st_size for path in data_files
        )
        efficiency["data/accepted_frames"] = (
            stats["generated_frames"]
            if stats
            else (int(corpus.valid.sum()) if corpus else None)
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
                {"events": tracker.timeline, "intervals": tracker.intervals}, indent=2
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
            "export_complete": export_complete,
            "batch_manifest_complete": (
                train_steps == total_updates if args.record_batches else None
            ),
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
        summary["matching"] = (
            "exact_corpus_batch_manifest"
            if args.mode == "offline-matched"
            else "unmatched_refresh_schedule" if args.mode == "serial" else None
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    except _BudgetExpired:
        # No post-deadline result can choose a checkpoint or become equal-budget
        # quality. Cleanup is still required and is reported outside this cap.
        elapsed = tracker.elapsed_seconds()
        if "loop_started" in locals():
            tracker.record_interval("training_loop", loop_started, elapsed)
        efficiency = tracker.finish(elapsed, "wall_time_budget")
        efficiency["horizon_prespecified"] = True
        efficiency["data_ready"] = args.offline_data is not None
        (args.output / "efficiency.json").write_text(json.dumps(efficiency, indent=2))
        (args.output / "timeline.json").write_text(
            json.dumps(
                {"events": tracker.timeline, "intervals": tracker.intervals}, indent=2
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
                    "export_complete": False,
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
        if engine is not None:
            try:
                engine.stop()
            except BaseException as error:
                cleanup_error = error
        if usage is not None:
            drain_usage()
            (args.output / "usage.json").write_text(
                json.dumps(
                    {
                        **usage.summary(),
                        "schedule": None if pacing is None else pacing.summary(),
                    },
                    indent=2,
                )
            )
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
                },
                indent=2,
            )
        )
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise cleanup_error
