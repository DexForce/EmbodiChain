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
"""Run native peg PPO with strict native-capacity guards in both environments.

Run: python -m scripts.benchmark.assembly.train_peg --config TRAIN.yaml --output HEALTH.json

This wrapper calls the ordinary ``train_from_config`` entry point and its
optional trainer-ready hook. It does not implement learning, choose actions,
or modify observations, rewards, physical parameters or task guards. Private
native inspection stays in the benchmark health observer. The observer runs
after each completed control interval, before the environment can auto-reset;
MuJoCo Warp's sticky flags retain failures from every physical substep.

Capture complete raw native/Python logs with the calling shell, for example::

    python -m scripts.benchmark.assembly.train_peg --config TRAIN.yaml \
      --output HEALTH.json --raw-log TRAIN.log > TRAIN.log 2>&1
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

from scripts.benchmark.assembly.peg_reachability import _PhysicsHealth, _versions

__all__: list[str] = []


def _file_metadata(
    path: str | Path | None, *, digest: bool = False
) -> dict[str, Any] | None:
    """Describe an actual file without inventing missing log/checkpoint data."""
    if path is None:
        return None
    selected = Path(path).resolve()
    value: dict[str, Any] = {"path": str(selected), "exists": selected.is_file()}
    if selected.is_file():
        value["bytes"] = selected.stat().st_size
        if digest:
            value["sha256"] = hashlib.sha256(selected.read_bytes()).hexdigest()
    return value


def _write(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace one bounded health summary at checkpoint boundaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _epa_preflight() -> dict[str, Any]:
    """Fail closed unless the loaded MuJoCo Warp 3.12 EPA capacity is prepared."""
    from mujoco_warp._src import collision_convex, types

    installed = version("mujoco-warp")
    constants = {
        "types.MJ_MAX_EPAHORIZON": int(types.MJ_MAX_EPAHORIZON),
        "collision_convex.MJ_MAX_EPAHORIZON": int(collision_convex.MJ_MAX_EPAHORIZON),
    }
    return {
        "passed": installed.startswith("3.12.") and min(constants.values()) >= 128,
        "mujoco_warp_version": installed,
        "required": "MuJoCo Warp 3.12 with loaded EPA horizon capacity at least 128",
        "loaded_constants": constants,
        "loaded_sources": {
            name: _file_metadata(inspect.getfile(module), digest=True)
            for name, module in (
                ("types", types),
                ("collision_convex", collision_convex),
            )
        },
        "scope": "loaded Python constants and source identities before any native training scene; runtime capacity flags remain guarded throughout training",
    }


def _write_markdown(path: Path, document: dict[str, Any]) -> None:
    """Present final validation and health in exactly three benchmark tables."""
    training = document.get("training", {})
    success = training.get("last_eval_metrics", {}).get(
        "eval/success_rate", "not evaluated"
    )
    lines = [
        "# Native peg PPO health report",
        "",
        document["qualification"],
        "",
        "## Time & Memory",
        "",
        "| implementation | elapsed_wall_seconds | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---:|---|---|---|---|",
        f"| native peg PPO with health guard | {document.get('elapsed_wall_seconds', 'unmeasured')} | not measured | not measured | not measured | not measured |",
        "",
        "## Success & Other Metrics",
        "",
        "| environment | success_rate | control_boundaries_observed | capacity_loss | convergence_budget | unclassified_bits |",
        "|---|---:|---:|---|---|---|",
    ]
    for name, health in document.get("health", {}).items():
        lines.append(
            f"| {name} | {success if name == 'eval' else 'not evaluated'} | {health['control_boundaries_observed']} | {health['capacity_overflow_seen']} | {health['convergence_budget_exhausted_seen']} | {health['unclassified_overflow_seen']} |"
        )
    if not document.get("health"):
        lines.append(
            "| unavailable | not evaluated | 0 | unverified | unverified | unverified |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | qualification |",
        "|---:|---|---:|---|",
        f"| 1 | native proprioceptive PPO | {success} | latest native validation only; no paper or held-out qualification |",
        "",
        f"Status: {document['status']}. GPU memory and standalone latency were not benchmarked by this training wrapper. Native training logs retain their own throughput measurements; synchronous health readback is included in those measurements.",
        "",
        "Native capacity/collision loss and unclassified flags fail before auto-reset; convergence-budget events remain distinct. Control-boundary snapshots cover all physical steps through sticky flags, while contact-count maxima sample only the final substep. JSON stores aggregate counters at checkpoints, without a per-control trace or per-world zero arrays.",
        "",
        "The wrapper neither selects actions nor changes PPO, observations, rewards, reset distributions, physical parameters or task guards. Its validation success is an observed training diagnostic, not a reproduction claim.",
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(lines))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Attach diagnostic observers through the native trainer-ready hook."""
    from embodichain.learning.rl.train import train_from_config
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.utils.utility import load_config

    config_path = args.config.resolve()
    source = load_config(str(config_path))
    if source.get("algorithm", {}).get("name") != "ppo":
        raise ValueError(
            "This diagnostic wrapper requires the native PPO configuration."
        )
    if source.get("trainer", {}).get("distributed", False):
        raise ValueError("Use one process for this diagnostic health report.")
    if source.get("policy", {}).get("obs_groups", {}).get("actor") != [
        "proprioception"
    ]:
        raise ValueError(
            "Peg reproduction actor must read only the proprioception group."
        )
    if "on_trainer_ready" not in inspect.signature(train_from_config).parameters:
        raise RuntimeError(
            "Native train_from_config lacks the optional on_trainer_ready hook."
        )
    document: dict[str, Any] = {
        "schema": "embodichain-peg-native-training-health/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "qualification": "Native proprioception-only PPO attempt; this health report does not establish paper reproduction or learned success.",
        "config": _file_metadata(config_path, digest=True),
        "source_config": source,
        "resume": {
            "checkpoint": args.resume,
            "run": str(args.run) if args.run else None,
            "scheduler": args.resume_scheduler,
        },
        "software": _versions(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "wrapper_source": _file_metadata(Path(__file__), digest=True),
        "health_observer_source": _file_metadata(
            inspect.getfile(_PhysicsHealth), digest=True
        ),
        "raw_logs": {
            "capture": "caller-owned stdout/stderr capture; native file descriptors are not redirected by this wrapper",
            "file": _file_metadata(args.raw_log),
        },
        "observation_scope": "after completed sim.update returns and before BaseEnv auto-reset; sticky native flags cover all substeps, sampled contact/constraint counts cover the last substep only",
        "failure_policy": "capacity/collision-loss or unclassified overflow bits raise before auto-reset; ITERATIONS/LS_ITERATIONS are recorded convergence-budget events, not dropped contacts",
        "checkpoint_health": [],
    }
    try:
        difference = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
        patch = args.output.with_suffix(".source.patch")
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_bytes(difference)
        document["working_tree_patch"] = _file_metadata(patch, digest=True)
    except (OSError, subprocess.SubprocessError):
        document["working_tree_patch"] = None
    observers: dict[str, tuple[Any, _PhysicsHealth]] = {}
    trainer_ref: Any = None
    started = time.perf_counter()

    def snapshot(reason: str, checkpoint: str | Path | None = None) -> None:
        """Save accumulated counters without per-control traces or zero arrays."""
        document["health"] = {
            name: deepcopy(observer.values) for name, (_, observer) in observers.items()
        }
        if trainer_ref is not None:
            document["training"] = {
                "global_step": int(trainer_ref.global_step),
                "num_updates": int(trainer_ref.num_updates),
                "run_path": str(Path(trainer_ref.checkpoint_dir).resolve().parent),
                "latest_checkpoint": _file_metadata(trainer_ref.latest_checkpoint_path),
                "best_checkpoint": _file_metadata(trainer_ref.best_checkpoint_path),
                "last_train_metrics": dict(trainer_ref.last_train_metrics),
                "last_eval_metrics": dict(trainer_ref.last_eval_metrics),
            }
        if checkpoint is not None:
            document["checkpoint_health"].append(
                {
                    "checkpoint": _file_metadata(checkpoint),
                    "global_step": int(trainer_ref.global_step),
                    "num_updates": int(trainer_ref.num_updates),
                    "health": deepcopy(document["health"]),
                }
            )
        document["last_snapshot_reason"] = reason
        document["elapsed_wall_seconds"] = time.perf_counter() - started
        document["raw_logs"]["file"] = _file_metadata(args.raw_log)
        _write(args.output, document)

    def ready(trainer: Any) -> None:
        """Attach train/evaluation guards without changing the native trainer."""
        nonlocal trainer_ref
        trainer_ref = trainer
        environments = {}
        for name, selected in (("train", trainer.env), ("eval", trainer.eval_env)):
            if selected is None:
                continue
            env = getattr(selected, "unwrapped", selected)
            if getattr(env, "assembly_kind", None) != "peg":
                raise ValueError(
                    f"Expected a native peg assembly environment for {name}."
                )
            observer = _PhysicsHealth(
                env, scope="control", strict=True, include_world_flags=False
            )
            observers[name] = (env, observer)
            environments[name] = {
                "num_envs": int(env.num_envs),
                "device": str(env.device),
                "physics_dt": float(env.physics_dt),
                "control_dt": float(env.step_dt),
                "configured_physics_frequency_hz": 1.0 / float(env.physics_dt),
                "configured_policy_frequency_hz": 1.0 / float(env.step_dt),
                "cadence_scope": "resolved task/manager configuration; effective timestep and control-clock evidence are recorded separately by the health observer after live stepping",
                "asset_sha256": env.assembly_asset.content_hash,
                "asset_generator_revision": env.assembly_asset.generator_revision,
                "fidelity": deepcopy(env.assembly_fidelity),
            }
        document["environments"] = environments
        document["status"] = "running"
        original_save = trainer.save_checkpoint

        def save_checkpoint(path: str | Path | None = None) -> str:
            result = original_save(path)
            snapshot("checkpoint", result)
            return result

        trainer.save_checkpoint = save_checkpoint
        snapshot("trainer_ready")

    snapshot("before_native_entry")
    try:
        document["epa_preflight"] = _epa_preflight()
        snapshot("native_dependency_preflight")
        if not document["epa_preflight"]["passed"]:
            raise RuntimeError(
                "Loaded MuJoCo Warp lacks the required 3.12 EPA-horizon >= 128 capacity profile."
            )
        discover_task_packages()
        execute_init_hooks()
        result = train_from_config(
            str(config_path),
            distributed=False,
            resume=args.resume,
            run=args.run,
            resume_scheduler=args.resume_scheduler,
            on_trainer_ready=ready,
        )
        document["native_return"] = result
        if trainer_ref is None:
            raise RuntimeError("The native trainer-ready hook did not run.")
        planned = (
            int(source["trainer"]["iterations"])
            * int(trainer_ref.buffer_size)
            * int(trainer_ref.env.num_envs)
        )
        document["planned_total_transitions"] = planned
        document["status"] = (
            "completed"
            if trainer_ref.global_step >= planned
            else "stopped_before_budget"
        )
    except BaseException as error:
        document["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        document["error"] = traceback.format_exc()
        raise
    finally:
        # Native train_from_config owns teardown, including exceptional exits.
        # These Python-only restorations do not inspect destroyed native state.
        for env, observer in observers.values():
            observer.restore(env)
        snapshot("final")
        if trainer_ref is not None:
            root = Path(trainer_ref.checkpoint_dir).resolve().parent
            document["run_manifest"] = _file_metadata(
                root / "run-manifest.json", digest=True
            )
            document["tensorboard_logs"] = [
                _file_metadata(path)
                for path in sorted(root.glob("logs/**/events.out.tfevents.*"))
            ]
        document["raw_logs"]["file"] = _file_metadata(args.raw_log)
        document["raw_logs"][
            "size_scope"
        ] = "finalization-time size snapshot; caller-owned open stdout/stderr file may still receive final messages"
        _write(args.output, document)
        _write_markdown(args.output, document)
        print(f"Native training health report: {args.output}", flush=True)
    return document


def main() -> None:
    """Select native training inputs without overriding the PPO experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path)
    parser.add_argument("--resume")
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--resume-scheduler", choices=("restore", "restart"), default="restore"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "--output already exists; preserve previous experiment evidence with a new path."
        )
    if args.resume in {"latest", "best"} and args.run is None:
        parser.error("--resume latest/best requires --run.")
    if args.run is not None and args.resume not in {"latest", "best"}:
        parser.error("--run requires --resume latest/best.")
    if args.resume is None and args.resume_scheduler != "restore":
        parser.error("--resume-scheduler requires --resume.")
    run_training(args)


if __name__ == "__main__":
    main()
