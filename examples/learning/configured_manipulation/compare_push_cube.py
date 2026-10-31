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

"""Run the fixed-budget legacy/configured PushCube PPO comparison.

The parent schedules at most two independent simulator processes. Each worker
uses the production trainer, records the initial policy, reloads the final
checkpoint, and evaluates both policies on the same single-environment cases.
"""

from __future__ import annotations

import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any
from threading import Event, RLock

import numpy as np
import torch
import yaml

from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.learning.rl.evaluation import infer_policy_action
from embodichain.learning.rl.runtime import build_gym_policy_runtime

__all__: list[str] = []

_ROOT = Path(__file__).resolve().parents[3]
_TASK = _ROOT / "extensions/embodichain_tasks/configs/tasks/manipulation/push_cube"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(child) for child in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(data), indent=2, allow_nan=False))
    temporary.replace(path)


def _software_versions() -> dict[str, str | None]:
    native = importlib.import_module("dexsim")
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "dexsim": str(getattr(native, "__version__", "unknown")),
        "dexsim_commit": str(getattr(native, "__commit_id__", "unknown")),
    }


def _fingerprint(variant: str, seed: int, iterations: int, cases: int) -> str:
    source = (
        _TASK
        / "agents"
        / ("ppo.json" if variant == "legacy" else "ppo.configured_rl.yaml")
    )
    gym = _TASK / ("env.json" if variant == "legacy" else "env.configured_rl.yaml")
    implementation = [
        "examples/learning/configured_manipulation/compare_push_cube.py",
        "embodichain/lab/gym/envs/configured_rl_env.py",
        "embodichain/lab/gym/envs/managers/outcome_manager.py",
        "embodichain/lab/gym/envs/managers/conditions.py",
        "embodichain/lab/gym/envs/objectives/conditions.py",
        "embodichain/learning/rl/train.py",
        "embodichain/learning/rl/runtime.py",
        "embodichain/learning/rl/utils/trainer.py",
        "extensions/embodichain_tasks/embodichain_tasks/manipulation/push_cube.py",
    ]
    declaration = {
        "variant": variant,
        "seed": seed,
        "iterations": iterations,
        "cases": cases,
        "train": yaml.safe_load(source.read_text()),
        "gym": yaml.safe_load(gym.read_text()),
        "software": _software_versions(),
        "implementation": {
            name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest()
            for name in implementation
        },
    }
    return hashlib.sha256(json.dumps(declaration, sort_keys=True).encode()).hexdigest()


def _validate_result(
    row: dict[str, Any], variant: str, seed: int, iterations: int, cases: int
) -> None:
    if (
        row.get("variant"),
        row.get("seed"),
        row.get("iterations"),
        row.get("transitions"),
    ) != (variant, seed, iterations, 64 * 1024 * iterations):
        raise ValueError(
            "Existing experiment has a different identity or training budget; use a new output directory."
        )
    if row.get("fingerprint") != _fingerprint(variant, seed, iterations, cases):
        raise ValueError(
            "Existing experiment has a different configuration or implementation; use a new output directory."
        )
    expected = list(range(20000, 20000 + cases))
    for key in ("untrained_cases", "final_cases"):
        if [case.get("case_seed") for case in row.get(key, [])] != expected:
            raise ValueError(
                "Experiment does not contain the complete fixed evaluation cases."
            )


def _evaluate(
    config: dict[str, Any],
    device: torch.device,
    gpu: int,
    weights: dict[str, torch.Tensor],
    cases: int,
) -> list[dict[str, Any]]:
    runtime = build_gym_policy_runtime(
        config,
        device=device,
        num_envs=1,
        headless=True,
        renderer="no-render",
        gpu_id=gpu,
        seed=20000,
    )
    env = runtime.env
    results = []
    try:
        runtime.policy.load_state_dict(weights)
        runtime.policy.eval()
        for case in range(cases):
            obs, _ = env.reset(seed=20000 + case, options={"save_data": False})
            initial = {
                "cube_pose": env.sim.get_rigid_object("cube")
                .get_local_pose(to_matrix=True)[0]
                .cpu()
                .tolist(),
                "goal_pose": env.goal_pose[0].cpu().tolist(),
                "robot_qpos": env.robot.get_qpos()[0].cpu().tolist(),
            }
            total_reward = 0.0
            for step in range(env.max_episode_steps):
                with torch.no_grad():
                    action = infer_policy_action(
                        runtime.policy, obs, device=device, num_envs=1
                    )
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward[0])
                if bool(terminated[0] | truncated[0]):
                    break
            failure = bool(info.get("fail", torch.zeros(1, dtype=torch.bool))[0])
            success = (
                bool(info["success"][0]) and not failure and not bool(truncated[0])
            )
            metrics = info.get("metrics", {})
            distance = metrics.get(
                "distance_to_goal", metrics.get("success.goal.distance")
            )
            reason = (
                "truncated"
                if bool(truncated[0])
                else "success" if success else "failure"
            )
            results.append(
                {
                    "case_seed": 20000 + case,
                    "success": success,
                    "return": total_reward,
                    "length": step + 1,
                    "distance": float(distance[0]) if distance is not None else None,
                    "reason": reason,
                    "initial": initial,
                }
            )
    finally:
        runtime.close()
    return results


def _link_parent_lifetime() -> None:
    # Native engines may install their own SIGTERM handler. On Linux, kill an
    # orphaned worker even if its controller disappears during daemon recovery.
    if sys.platform == "linux":
        parent = os.getppid()
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(
                ctypes.get_errno(), "Could not link worker lifetime to its parent"
            )
        if os.getppid() != parent:
            raise RuntimeError("Experiment controller exited during worker startup.")


def _worker(args: argparse.Namespace) -> None:
    _link_parent_lifetime()
    output = args.output / f"{args.variant}_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    discover_task_packages()
    execute_init_hooks()
    train = importlib.import_module("embodichain.learning.rl.train")
    source = (
        _TASK
        / "agents"
        / ("ppo.json" if args.variant == "legacy" else "ppo.configured_rl.yaml")
    )
    config = yaml.safe_load(source.read_text())
    trainer = config["trainer"]
    logical_gpu = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else args.gpu
    trainer.update(
        exp_name=f"push_cube_{args.variant}_s{args.seed}",
        gym_config=str(
            _TASK
            / ("env.json" if args.variant == "legacy" else "env.configured_rl.yaml")
        ),
        seed=args.seed,
        device=f"cuda:{logical_gpu}",
        gpu_id=logical_gpu,
        num_envs=64,
        buffer_size=1024,
        iterations=args.iterations,
        enable_eval=True,
        num_eval_envs=16,
        num_eval_episodes=100,
        eval_freq=64 * 1024 * min(50, args.iterations),
        eval_seed=args.seed + 10000,
        save_freq=0,
        save_frequency_updates=100,
        use_wandb=False,
        events={},
        renderer="no-render",
        headless=True,
    )
    config_path = output / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    fingerprint = _fingerprint(args.variant, args.seed, args.iterations, args.cases)
    captured = []
    original = train.Trainer
    started = time.time()

    class RecordingTrainer(original):
        def __init__(self, *positional: Any, **kwargs: Any) -> None:
            super().__init__(*positional, **kwargs)
            captured.append(self)
            torch.save(
                {
                    key: value.detach().cpu().clone()
                    for key, value in self.policy.state_dict().items()
                },
                output / "initial_policy.pt",
            )

        def _log_train(self, losses: dict[str, float]) -> None:
            super()._log_train(losses)
            _write(
                output / "progress.json",
                {
                    "phase": "training",
                    "updates": self.num_updates,
                    "target_updates": args.iterations,
                    "global_step": self.global_step,
                    "elapsed_seconds": time.time() - started,
                    "last_train": self.last_train_metrics,
                },
            )
            with (output / "learning_curve.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(_plain(self.last_train_metrics), allow_nan=False) + "\n"
                )

        def _eval_once(self, num_episodes: int) -> dict[str, float]:
            result = super()._eval_once(num_episodes)
            _write(output / "eval_history.json", self.eval_history)
            return result

    train.Trainer = RecordingTrainer
    try:
        train.train_from_config(config_path)
    finally:
        train.Trainer = original
    trained = captured[0]
    checkpoint = Path(trained.latest_checkpoint_path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["num_updates"] != args.iterations:
        raise RuntimeError("Final checkpoint does not have the requested update count.")
    _write(
        output / "progress.json",
        {
            "phase": "fixed_case_evaluation",
            "updates": trained.num_updates,
            "checkpoint": str(checkpoint),
        },
    )
    device = torch.device(f"cuda:{logical_gpu}")
    initial_weights = torch.load(output / "initial_policy.pt", weights_only=True)
    initial = _evaluate(config, device, logical_gpu, initial_weights, args.cases)
    final = _evaluate(config, device, logical_gpu, payload["policy"], args.cases)
    if any(one["initial"] != two["initial"] for one, two in zip(initial, final)):
        raise RuntimeError(
            "Untrained and trained policy cases have different initial states."
        )
    result = {
        "fingerprint": fingerprint,
        "software": _software_versions(),
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu)),
            "cpu_threads": torch.get_num_threads(),
        },
        "variant": args.variant,
        "seed": args.seed,
        "iterations": args.iterations,
        "transitions": payload["global_step"],
        "checkpoint": str(checkpoint),
        "run": str(checkpoint.parent.parent),
        "elapsed_seconds": time.time() - started,
        "untrained_success_rate": float(np.mean([case["success"] for case in initial])),
        "success_rate": float(np.mean([case["success"] for case in final])),
        "untrained_cases": initial,
        "final_cases": final,
    }
    _write(output / "result.json", result)
    _write(
        output / "progress.json",
        {
            "phase": "complete",
            "success_rate": result["success_rate"],
            "checkpoint": str(checkpoint),
        },
    )


def _summarize(output: Path, iterations: int = 1000, cases: int = 100) -> None:
    rows = [
        json.loads(path.read_text()) for path in sorted(output.glob("*/result.json"))
    ]
    if len(rows) != 6:
        raise RuntimeError(f"Expected six completed experiments, got {len(rows)}.")
    identities = {(row.get("variant"), row.get("seed")) for row in rows}
    if identities != {
        (variant, seed) for variant in ("legacy", "configured") for seed in (42, 43, 44)
    }:
        raise ValueError("Experiments must contain each exact variant/seed pair once.")
    for row in rows:
        _validate_result(row, row["variant"], row["seed"], iterations, cases)
    if len({json.dumps(row.get("software"), sort_keys=True) for row in rows}) != 1:
        raise ValueError("Experiments used different software versions.")
    legacy = [row["success_rate"] for row in rows if row["variant"] == "legacy"]
    configured = [row["success_rate"] for row in rows if row["variant"] == "configured"]
    untrained = [
        row["untrained_success_rate"] for row in rows if row["variant"] == "configured"
    ]
    for seed in (42, 43, 44):
        pair = [row for row in rows if row["seed"] == seed]
        if any(
            one["initial"] != two["initial"]
            for one, two in zip(pair[0]["final_cases"], pair[1]["final_cases"])
        ):
            raise RuntimeError(
                "Legacy and configured evaluation initial states differ."
            )
    summary = {
        "legacy_mean": float(np.mean(legacy)),
        "legacy_std": float(np.std(legacy)),
        "configured_mean": float(np.mean(configured)),
        "configured_std": float(np.std(configured)),
        "untrained_mean": float(np.mean(untrained)),
        "not_worse_than_legacy": bool(np.mean(configured) >= np.mean(legacy) - 0.05),
        "improves_over_untrained": bool(
            np.mean(configured) >= np.mean(untrained) + 0.20
        ),
        "experiments": [
            {key: value for key, value in row.items() if not key.endswith("cases")}
            for row in rows
        ],
    }
    summary["iterations"] = iterations
    summary["cases_per_checkpoint"] = cases
    summary["formal_protocol"] = iterations == 1000 and cases == 100
    summary["accepted"] = (
        summary["formal_protocol"]
        and summary["not_worse_than_legacy"]
        and summary["improves_over_untrained"]
    )
    _write(output / "summary.json", summary)
    lines = [
        "# PushCube PPO comparison",
        "",
        "| Variant | Mean success | Standard deviation |",
        "| --- | ---: | ---: |",
        f"| Legacy | {summary['legacy_mean']:.1%} | {summary['legacy_std']:.1%} |",
        f"| Configured | {summary['configured_mean']:.1%} | {summary['configured_std']:.1%} |",
        f"| Untrained configured | {summary['untrained_mean']:.1%} | — |",
        "",
        f"Acceptance: **{'PASS' if summary['accepted'] else 'FAIL'}**.",
        "",
        f"Validated budgets: 64 environments × 1024 rollout steps × {iterations} updates; seeds 42/43/44. Final checkpoints, {cases} case seeds 20000–{20000+cases-1}, deterministic actions. Formal protocol: {summary['formal_protocol']}. Fallen objects are failures in the configured task; the legacy task retains its truncation behavior.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=("legacy", "configured"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.cases <= 0:
        parser.error("iterations and cases must be positive.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        _summarize(args.output, args.iterations, args.cases)
    elif args.worker:
        try:
            _worker(args)
        except BaseException:
            directory = args.output / f"{args.variant}_{args.seed}"
            directory.mkdir(parents=True, exist_ok=True)
            progress = directory / "progress.json"
            previous = json.loads(progress.read_text()) if progress.exists() else {}
            _write(
                progress,
                {**previous, "phase": "failed", "error": traceback.format_exc()},
            )
            raise
    else:
        if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus):
            parser.error("Select one or two distinct GPUs.")

        cancelled = Event()
        active_lock = RLock()
        active: dict[int, tuple[subprocess.Popen, Path]] = {}

        def stop_workers() -> None:
            cancelled.set()
            with active_lock:
                children = list(active.values())
            for process, directory in children:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "progress.json"
                    previous = json.loads(path.read_text()) if path.exists() else {}
                    _write(
                        path,
                        {
                            **previous,
                            "phase": "interrupted",
                            "error": "Experiment controller stopped before completion.",
                        },
                    )

        def interrupted(signum: int, frame: Any) -> None:
            stop_workers()
            raise SystemExit(128 + signum)

        previous_handlers = {
            kind: signal.signal(kind, interrupted)
            for kind in (signal.SIGINT, signal.SIGTERM)
        }

        def lane(gpu: int, jobs: list[tuple[str, int]]) -> None:
            for variant, seed in jobs:
                if cancelled.is_set():
                    return
                result = args.output / f"{variant}_{seed}" / "result.json"
                if result.exists():
                    _validate_result(
                        json.loads(result.read_text()),
                        variant,
                        seed,
                        args.iterations,
                        args.cases,
                    )
                    continue
                if (result.parent / "progress.json").exists():
                    raise ValueError(
                        "An incomplete experiment is already present; use a new output directory to preserve failure evidence."
                    )
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--variant",
                    variant,
                    "--seed",
                    str(seed),
                    "--gpu",
                    str(gpu),
                    "--output",
                    str(args.output),
                    "--iterations",
                    str(args.iterations),
                    "--cases",
                    str(args.cases),
                ]
                with (args.output / f"{variant}_{seed}.log").open("w") as log:
                    with active_lock:
                        if cancelled.is_set():
                            return
                        process = subprocess.Popen(
                            command,
                            cwd=_ROOT,
                            env={
                                **os.environ,
                                "OMP_NUM_THREADS": "2",
                                "MKL_NUM_THREADS": "2",
                                "CUDA_VISIBLE_DEVICES": str(gpu),
                            },
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        active[process.pid] = (process, result.parent)
                    try:
                        code = process.wait()
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                    finally:
                        with active_lock:
                            active.pop(process.pid, None)

        jobs = [
            (variant, seed)
            for seed in (42, 43, 44)
            for variant in ("legacy", "configured")
        ]
        try:
            with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
                futures = [
                    pool.submit(lane, gpu, jobs[index :: len(args.gpus)])
                    for index, gpu in enumerate(args.gpus)
                ]
                try:
                    for future in as_completed(futures):
                        future.result()
                except BaseException:
                    stop_workers()
                    raise
        finally:
            stop_workers()
            for kind, handler in previous_handlers.items():
                signal.signal(kind, handler)
        _summarize(args.output, args.iterations, args.cases)


if __name__ == "__main__":
    main()
