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
"""Evaluate a native peg checkpoint with strict capacity and raw-state guards.

Run: python -m scripts.benchmark.assembly.eval_peg RUN --checkpoint best \
    --episodes 8192 --num-envs 1024 --seed 20000 --output outputs/peg_final

Checkpoint resolution, runtime reconstruction, weight loading, deterministic
episode evaluation and evaluation.json all use the production evaluation
framework. This wrapper adds read-only benchmark health evidence before
auto-reset; it does not change the policy, reset support, physical parameters,
task guards or episode accounting. Runtime-construction shape-probe resets
are unscored. Only the production evaluator's completed episodes are scored.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

from scripts.benchmark.assembly.peg_reachability import _PhysicsHealth, _versions
from scripts.benchmark.assembly.train_peg import _epa_preflight, _file_metadata, _write

__all__: list[str] = []


def _write_markdown(path: Path, document: dict[str, Any]) -> None:
    """Keep fresh evaluation health separate from historical training scope."""
    metrics = document.get("metrics", {})
    health = document.get("native_health", {})
    state = health.get("native_state", {})
    success = metrics.get("eval/success_rate", "not evaluated")
    lines = [
        "# Strict native peg policy evaluation",
        "",
        document["qualification"],
        "",
        "## Time & Memory",
        "",
        "| implementation | elapsed_wall_seconds | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "|---|---:|---|---|---|---|",
        f"| native deterministic evaluation with health guard | {document.get('elapsed_wall_seconds', 'unmeasured')} | not measured | not measured | not measured | not measured |",
        "",
        "## Success & Other Metrics",
        "",
        "| success_rate | scored_episodes | CI95_low | CI95_high | capacity_loss | convergence_budget | native_state_all_samples_finite |",
        "|---:|---:|---:|---:|---|---|---|",
        f"| {success} | {metrics.get('eval/completed_episodes', 0)} | {metrics.get('eval/success_ci95_low', 'not evaluated')} | {metrics.get('eval/success_ci95_high', 'not evaluated')} | {health.get('capacity_overflow_seen', 'unverified')} | {health.get('convergence_budget_exhausted_seen', 'unverified')} | {state.get('all_samples_finite', 'unverified')} |",
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | qualification |",
        "|---:|---|---:|---|",
        f"| 1 | native proprioceptive PPO checkpoint | {success} | fresh nominal evaluation; training coverage and paper equivalence remain separate |",
        "",
        f"Status: {document['status']}. Standard evaluation report: {document.get('evaluation_report', 'not produced')}. No task success report is produced when native health or evaluation fails.",
        "",
        "Capacity/collision loss, unknown overflow bits or nonfinite complete native qpos/qvel/qacc matrices fail before auto-reset. States between control boundaries are not sampled. Sticky capacity flags cover all physical substeps. Prepared solver options are distinguished from the first effective timestep and first two observed control-clock intervals.",
        "",
        "Only exact production-evaluator episode accounting is scored. The runtime factory's shape-probe reset and post-build health snapshot do not count as episodes. This fresh evaluation evidence does not retroactively qualify training. End-effector path length, action norms, actual joint torque and memory were not measured.",
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> Path:
    """Reuse the native evaluation framework and add bounded health metadata."""
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.learning.rl.evaluation import evaluate_episodes
    from embodichain.learning.rl.policy_evaluation.cli import (
        _build_native_runtime,
        _flush_simulator,
        _headless_report,
        _resolve_input,
        _validate_native_options,
        parse_args as parse_native_args,
    )
    from embodichain.learning.rl.policy_evaluation.report import (
        _json_value,
        write_evaluation_report,
    )

    parent = args.output.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = parent / f"{stamp}-strict-peg-health"
    directory.mkdir(parents=True)
    health_path = directory / "health.json"
    native_args = parse_native_args(
        [
            str(args.run),
            "--checkpoint",
            args.checkpoint,
            "--episodes",
            str(args.episodes),
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--renderer",
            "no-render",
            "--output",
            str(parent),
        ]
    )
    document: dict[str, Any] = {
        "schema": "embodichain-peg-native-evaluation-health/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "qualification": "Fresh strict evaluation of one nominal simulation checkpoint; no retroactive training-health or full paper-reproduction claim.",
        "run": str(args.run.expanduser().resolve()),
        "episodes_requested": args.episodes,
        "evaluation_reset_seed": args.seed,
        "num_envs_requested": args.num_envs,
        "software": _versions(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "wrapper_source": _file_metadata(Path(__file__), digest=True),
        "health_observer_source": _file_metadata(
            inspect.getfile(_PhysicsHealth), digest=True
        ),
        "raw_log": {
            "capture": "caller-owned stdout/stderr; pass --raw-log when redirecting native and Python logs",
            "file": _file_metadata(args.raw_log),
        },
        "episode_accounting": "runtime construction resets to infer policy shape before guard attachment; that reset and the build snapshot are unscored. evaluate_episodes resets with evaluation_reset_seed and scores exactly the requested completed episodes.",
        "not_measured": [
            "end_effector_path_length",
            "raw_action_norm",
            "actual_joint_torque",
            "GPU_memory",
        ],
    }
    native = resolved = observer = env = None
    metrics = None
    failure: BaseException | None = None
    started = time.perf_counter()
    _write(health_path, document)
    try:
        document["epa_preflight"] = _epa_preflight()
        if not document["epa_preflight"]["passed"]:
            raise RuntimeError(
                "Strict evaluation requires the prepared MuJoCo Warp 3.12 EPA-horizon >=128 profile."
            )
        resolved = _resolve_input(native_args)
        if args.checkpoint == "best" and resolved.selected_checkpoint != "best":
            raise RuntimeError(
                "No validation-selected best checkpoint exists; strict evaluation refuses the general CLI's latest fallback."
            )
        document["checkpoint"] = {
            **_file_metadata(resolved.checkpoint, digest=True),
            "requested": resolved.requested_checkpoint,
            "selected": resolved.selected_checkpoint,
        }
        document["configs"] = {
            name: _file_metadata(path, digest=True)
            for name, path in resolved.configs.items()
        }
        discover_task_packages()
        execute_init_hooks()
        _validate_native_options(native_args)
        native = _build_native_runtime(native_args, resolved, viewer=False)
        if not native.uses_simulator:
            raise ValueError("Strict peg evaluation requires a native simulator task.")
        env = getattr(native.runtime.env, "unwrapped", native.runtime.env)
        if getattr(env, "assembly_kind", None) != "peg":
            raise ValueError("Strict peg evaluation requires the peg assembly task.")
        policy = getattr(native.runtime.policy, "module", native.runtime.policy)
        if tuple(getattr(policy, "actor_obs_groups", ())) != ("proprioception",):
            raise ValueError(
                "Peg actor must read only the proprioception observation group."
            )
        observer = _PhysicsHealth(
            env, scope="control", strict=True, include_world_flags=False
        )
        document["runtime"] = {
            "build_seed": native.seed,
            "evaluation_reset_seed": native.seed,
            "num_envs": int(env.num_envs),
            "task_id": native.runtime.env_id,
            "device": str(native.device),
            "simulation_device": str(native.simulation_device),
            "configured_physics_dt": float(env.physics_dt),
            "configured_control_dt": float(env.step_dt),
            "configured_physics_frequency_hz": 1.0 / float(env.physics_dt),
            "configured_policy_frequency_hz": 1.0 / float(env.step_dt),
            "cadence_scope": "resolved configuration; actual first effective timestep and two control clocks are reported by native_health",
            "asset_sha256": env.assembly_asset.content_hash,
            "asset_generator_revision": env.assembly_asset.generator_revision,
            "asset_bottom_thickness": env.assembly_asset.config.get(
                "bottom_thickness", 0
            ),
            "fidelity": deepcopy(env.assembly_fidelity),
        }
        if native.device.type == "cuda":
            import torch

            document["gpu"] = {
                "name": torch.cuda.get_device_name(native.device),
                "torch_cuda_version": torch.version.cuda,
            }
        document["postbuild_health_unscored"] = deepcopy(observer.values)
        document["status"] = "evaluating"
        _write(health_path, _json_value(document))
        metrics = evaluate_episodes(
            policy=native.runtime.policy,
            env=native.runtime.env,
            num_episodes=args.episodes,
            device=native.device,
            seed=native.seed,
        )
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise RuntimeError("Nonfinite production evaluation metrics.")
        if (
            metrics.get("eval/completed_episodes") != args.episodes
            or metrics.get("eval/success_samples") != args.episodes
        ):
            raise RuntimeError(
                "Production evaluation did not score exactly the requested peg episodes with success labels."
            )
        health = observer.values
        state = health.get("native_state", {})
        if (
            not health["overflow_available"]
            or health["control_boundaries_observed"] < 1
        ):
            raise RuntimeError(
                "Missing native capacity observations during evaluation."
            )
        if (
            state.get("available") is not True
            or state.get("all_samples_finite") is not True
        ):
            raise RuntimeError(
                "Missing or nonfinite complete native-state observations."
            )
        document["metrics"] = metrics
        document["status"] = "evaluated_awaiting_cleanup"
    except BaseException as error:
        failure = error
        document["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        document["error"] = traceback.format_exc()
    finally:
        if observer is not None:
            document["native_health"] = deepcopy(observer.values)
            observer.restore(env)
        cleanup_errors = []
        try:
            if native is not None:
                native.runtime.close()
        except Exception:
            cleanup_errors.append(traceback.format_exc())
        try:
            _flush_simulator(native.uses_simulator if native is not None else True)
        except Exception:
            cleanup_errors.append(traceback.format_exc())
        if cleanup_errors:
            document["cleanup_errors"] = cleanup_errors
            document["status"] = "failed"
            if failure is None:
                failure = RuntimeError(
                    "Native runtime cleanup failed; see health.json."
                )
        document["elapsed_wall_seconds"] = time.perf_counter() - started
        document["raw_log"]["file"] = _file_metadata(args.raw_log)
        document["raw_log"][
            "size_scope"
        ] = "finalization-time snapshot of a caller-owned open log; final messages can append"
        _write(health_path, _json_value(document))
        _write_markdown(health_path, document)
        print(f"Native evaluation health: {health_path}", flush=True)
    if failure is not None:
        raise failure
    report = write_evaluation_report(
        parent, _headless_report(native, resolved, args.episodes, metrics)
    )
    document["evaluation_report"] = str(report)
    document["status"] = "completed"
    _write(health_path, _json_value(document))
    _write_markdown(health_path, document)
    return report


def main() -> None:
    """Evaluate a fixed checkpoint without changing its saved task resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--episodes", type=int, default=8192)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("--episodes and --num-envs must be positive.")
    print(f"Evaluation report: {run(args)}", flush=True)


if __name__ == "__main__":
    main()
