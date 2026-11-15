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

"""Train/evaluate APG with native EmbodiChain Sharpa physical collection.

Run with the native DexSim Python and the pinned upstream source on PYTHONPATH.
Only the optimizer/world-model/policy reference implementation is reused; the
upstream environment is never constructed by this entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _apg_learner import _APGLearner
from _full_native_task import _FullNativeTask as _NativeTask
from _native_checkpoint import (
    _ADAPTER_FILES,
    _restore_learner,
    _runtime_source,
    _save_replay,
    _verify_resume,
)
from omegaconf import OmegaConf
from reference import _verify_imported_learner
from sharpa_rl_unilab.training.configuration import configure_threads
from sharpa_rl_unilab.training.policy import (
    inference_distribution,
    make_actor,
    make_models,
)

__all__ = ["main"]


def _wait_memory(device: str, minimum_gib: float) -> None:
    """Wait at a batch boundary for shared-device headroom without changing state."""
    waiting = False
    while torch.cuda.mem_get_info(device)[0] < minimum_gib * 2**30:
        torch.cuda.empty_cache()
        if not waiting:
            print(f"Waiting for {minimum_gib:g}GiB free CUDA memory", flush=True)
            waiting = True
        time.sleep(2)
    if waiting:
        print("CUDA memory available; continuing", flush=True)


class _MemoryGuardedTask(_NativeTask):
    """Check shared-device headroom before each control period."""

    def __init__(self, *args: Any, minimum_free: float, **kwargs: Any) -> None:
        self.minimum_free = minimum_free
        super().__init__(*args, **kwargs)

    def step(self, action: torch.Tensor) -> Any:
        _wait_memory(str(self.device), self.minimum_free)
        return super().step(action)


def _save(
    output: Path, cfg: Any, learner: _APGLearner, iteration: int, elapsed: float
) -> Path:
    path = output / f"native_teacher_{iteration:05d}.pt"
    torch.save(
        {
            "algorithm": "embodichain_apg",
            "stage": "teacher",
            "native_environment": True,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "actor": learner.checkpoint_actor.state_dict(),
            "training_actor": learner.actor.state_dict(),
            "actor_average": (
                learner.actor_avg.state_dict()
                if learner.actor_avg is not None
                else None
            ),
            "critic": learner.critic.state_dict(),
            "world_model": learner.wm.state_dict(),
            "actor_optimizer": learner.opt_pi.state_dict(),
            "critic_optimizer": learner.opt_v.state_dict(),
            "world_model_optimizer": learner.opt_wm.state_dict(),
            "iteration": iteration,
            "collected": learner.collected,
            "received": learner.received,
            "collection_credit": getattr(learner, "collection_credit", 0.0),
            "wall_seconds": elapsed,
            "native_identity": learner.env.identity,
        },
        path,
    )
    _save_replay(output, learner, path, iteration)
    return path


@torch.no_grad()
def _evaluate(
    task: _NativeTask, actor: torch.nn.Module, seeds: list[int], output: Path
) -> dict:
    records = []
    traces = []
    actor.eval()
    for seed in seeds:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        obs, _ = task.reset(seed=seed)
        alive = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
        total = torch.zeros(task.num_envs, device=task.device)
        length = torch.zeros_like(total)
        tail = torch.zeros_like(total)
        positive = torch.zeros_like(total)
        tail_positive = torch.zeros_like(total)
        trace = []
        for step in range(400):
            action = inference_distribution(actor, obs, task.device).deterministic()
            state = task.step(action)
            delta = task.signed_angle_delta.clone()
            trace.append(torch.stack((delta, alive.float()), -1).cpu())
            total += alive * delta
            length += alive * task.step_dt
            positive += alive * (delta / task.step_dt > 0.1)
            if step >= 300:
                tail += alive * delta
                tail_positive += alive * (delta / task.step_dt > 0.1)
            alive &= ~state.terminated
            obs = state.obs
        stable = alive & (total >= 2 * math.pi) & (tail >= 0.5)
        for row in range(task.num_envs):
            records.append(
                {
                    "seed": seed,
                    "row": row,
                    "scale": task.identity["scales"][row // task.width],
                    "angle_rad": float(total[row]),
                    "survival_seconds": float(length[row]),
                    "dropped": not bool(alive[row]),
                    "stable": bool(stable[row]),
                    "tail_angle_rad": float(tail[row]),
                    "positive_fraction": float(positive[row] / 400),
                    "tail_positive_fraction": float(tail_positive[row] / 100),
                }
            )
        traces.append(torch.stack(trace))
    summary = {
        key: float(np.mean([r[key] for r in records]))
        for key in (
            "angle_rad",
            "survival_seconds",
            "dropped",
            "stable",
            "tail_angle_rad",
            "positive_fraction",
            "tail_positive_fraction",
        )
    }
    summary["median_angle_rad"] = float(np.median([r["angle_rad"] for r in records]))
    summary["passed"] = (
        summary["dropped"] <= 0.05
        and summary["stable"] >= 0.9
        and summary["median_angle_rad"] >= 2 * math.pi
        and summary["positive_fraction"] >= 0.8
        and summary["tail_positive_fraction"] >= 0.8
    )
    by_scale = {}
    for scale in task.identity["scales"]:
        selected = [r for r in records if r["scale"] == scale]
        value = {
            key: float(np.mean([r[key] for r in selected]))
            for key in (
                "angle_rad",
                "survival_seconds",
                "dropped",
                "stable",
                "tail_angle_rad",
                "positive_fraction",
                "tail_positive_fraction",
            )
        }
        value["median_angle_rad"] = float(np.median([r["angle_rad"] for r in selected]))
        value["episodes"] = len(selected)
        value["passed"] = (
            value["dropped"] <= 0.05
            and value["stable"] >= 0.9
            and value["median_angle_rad"] >= 2 * math.pi
            and value["positive_fraction"] >= 0.8
            and value["tail_positive_fraction"] >= 0.8
        )
        by_scale[str(scale)] = value
    summary["overall_passed"] = summary["passed"]
    summary["all_scales_passed"] = all(v["passed"] for v in by_scale.values())
    summary["passed"] = summary["overall_passed"] and summary["all_scales_passed"]
    result = {
        "by_scale": by_scale,
        "native_environment": True,
        "scope": task.identity,
        "episodes": len(records),
        "summary": summary,
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(traces, output / "physical_rotation_traces.pt")
    print(json.dumps(summary), flush=True)
    return result


def main() -> None:
    """Own a native eight-scale training experiment and its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument("--resume-native", type=Path)
    parser.add_argument("--resume-replay", type=Path)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resource-profile",
        choices=["reference", "reference_eager", "reference_host", "compact", "shared"],
        default="reference_host",
    )
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=[10001, 10002, 10003]
    )
    parser.add_argument(
        "--min-free-gpu-gib",
        type=float,
        default=None,
        help="Wait before training/control steps; defaults to 6 GiB for reference_eager, 4 for reference_host, otherwise 1.5.",
    )
    args = parser.parse_args()
    if args.resume_replay and not args.resume_native:
        raise ValueError("--resume-replay requires --resume-native")
    if args.resume_native and args.smoke:
        raise ValueError("Resume uses the saved configuration; omit --smoke")
    if args.min_free_gpu_gib is not None and args.min_free_gpu_gib < 0:
        raise ValueError("Minimum free GPU memory must be nonnegative")
    minimum_free = (
        args.min_free_gpu_gib
        if args.min_free_gpu_gib is not None
        else (
            6.0
            if args.resource_profile == "reference_eager"
            else (4.0 if args.resource_profile == "reference_host" else 1.5)
        )
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.reference_config)
    snapshot = None
    start_iteration = 0
    if args.resume_native:
        snapshot = torch.load(
            args.resume_native, map_location="cpu", weights_only=False
        )
        cfg = OmegaConf.create(snapshot["config"])
        if int(cfg.training.num_envs) != args.num_envs:
            raise ValueError(
                "Resume environment count must match the saved configuration"
            )
        start_iteration = int(snapshot["iteration"])
        if args.iterations <= start_iteration:
            raise ValueError(
                "--iterations must exceed the completed checkpoint iteration"
            )
    args.seed = (
        args.seed if args.seed is not None else (int(cfg.algo.seed) if snapshot else 1)
    )
    if snapshot and args.seed != int(cfg.algo.seed):
        raise ValueError("Resume seed must match the saved configuration")
    cfg.training.device = "cuda:0"
    cfg.training.num_envs = args.num_envs
    cfg.algo.seed = args.seed
    cfg.algo.max_iterations = args.iterations
    multiplier = 2048 / args.num_envs
    cfg.algo.buffer_steps = round(1000 * multiplier)
    cfg.algo.warmup_steps = round(64 * multiplier)
    cfg.algo.algo = "embodichain_apg"
    if args.resource_profile == "compact":
        cfg.algo.buffer_steps = max(64, 524288 // args.num_envs)
        cfg.algo.wm_hidden = [1024, 1024, 1024]
        cfg.algo.wm_batch = 512
        cfg.algo.pi_batch = 256
        cfg.algo.compile = False
    if args.resource_profile == "shared":
        cfg.algo.buffer_steps = max(64, 262144 // args.num_envs)
        cfg.algo.wm_hidden = [1024, 1024, 1024]
        cfg.algo.wm_batch = 256
        cfg.algo.pi_batch = 128
        cfg.algo.compile = False
    if args.resource_profile in {"reference_eager", "reference_host"}:
        cfg.algo.compile = False
    if args.resource_profile == "reference_host":
        cfg.algo.replay_device = "cpu"
        cfg.algo.activation_checkpoint = "world_model_mlp_nonreentrant"
    cfg.algo.save_interval = 100
    cfg.algo.num_envs = args.num_envs
    if snapshot is not None:
        # Restore the saved learning configuration after fresh-run profile assembly.
        cfg = OmegaConf.create(snapshot["config"])
        cfg.training.device = "cuda:0"
        cfg.algo.max_iterations = args.iterations
        cfg.algo.save_interval = 100
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True
    ).strip()
    if revision != "9816885a3b1fbdb947db71e72c27ed75effcbf9f":
        raise ValueError("Expected the qualified upstream reference revision")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=args.upstream, text=True
    ).strip():
        raise ValueError("The upstream reference sources must remain clean")
    _verify_imported_learner(args.upstream.resolve())
    if (output / "metrics.jsonl").exists():
        raise FileExistsError("Use a fresh native training output directory")
    snapshot_dir = output / "sources"
    snapshot_dir.mkdir(exist_ok=True)
    hashes = {}
    for name in [
        "native_full.py",
        "_full_native_task.py",
        "_resource_learner.py",
        "_native_checkpoint.py",
        "_native_task.py",
        "_simulation.py",
        "_native_asset.py",
        "_apg_learner.py",
        "reference.py",
    ]:
        path = Path(__file__).parent / name
        shutil.copy2(path, snapshot_dir / name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "experiment.json").write_text(
        json.dumps(
            {
                "upstream_revision": revision,
                "resource_profile": args.resource_profile,
                "source_hashes": hashes,
                "baseline_algorithm_comparison": False,
                "native_fixed_scale_calibration": False,
                "native_eight_scale_randomized_task": True,
                "memory_guard_per_control_step": args.checkpoint is None,
                "minimum_free_gpu_gib": minimum_free,
                "initialized_from_native_checkpoint": args.resume_native is not None,
            },
            indent=2,
        )
        + "\n"
    )
    if snapshot is not None:
        _verify_resume(args.resume_native, snapshot, hashes, args.resource_profile)
    if args.smoke:
        cfg.algo.buffer_steps = 64
        cfg.algo.warmup_steps = 16
        cfg.algo.wm_hidden = [128, 128]
        cfg.algo.wm_batch = 32
        cfg.algo.pi_batch = 16
        cfg.algo.wm_batches = 1
        cfg.algo.wm_only_iters = 2
        cfg.algo.critic_epochs = 1
        cfg.algo.critic_minibatches = 1
        cfg.algo.horizon = 8
        cfg.algo.compile = False
        cfg.algo.use_amp = False
    configure_threads(cfg)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    task = None
    try:
        if args.checkpoint:
            snapshot = torch.load(
                args.checkpoint, map_location="cpu", weights_only=False
            )
            if not snapshot.get("native_environment", False):
                raise ValueError(
                    "Use native_full_evaluate.py --allow-reference-diagnostic for source-policy transfer checks"
                )
            saved = OmegaConf.create(snapshot["config"])
            actor = make_actor(saved, "cuda:0", stage="teacher")
            actor.load_state_dict(snapshot["actor"], strict=True)
            task = _NativeTask(
                cfg,
                args.upstream.resolve(),
                output,
                num_envs=args.num_envs,
                seed=args.seed,
            )
            _evaluate(task, actor, args.eval_seeds, output)
            return
        actor, critic, _ = make_models(cfg, cfg.training.device)
        initial = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        task = _MemoryGuardedTask(
            cfg,
            args.upstream.resolve(),
            output,
            num_envs=args.num_envs,
            seed=args.seed,
            minimum_free=minimum_free,
        )
        learner_type = _APGLearner
        if args.resource_profile == "reference_host":
            from _resource_learner import _ResourceAPGLearner

            learner_type = _ResourceAPGLearner
        learner = learner_type(
            task, cfg, actor, critic, task.mdp, task.capture, value_bootstrap=True
        )
        replay_restored = False
        if snapshot is not None:
            replay_restored = _restore_learner(
                learner, snapshot, args.resume_native, args.resume_replay, output
            )
            initial = {
                k: v.detach().cpu().clone()
                for k, v in learner.actor.state_dict().items()
            }
        if args.resource_profile == "reference_host":
            resource_path = _runtime_source(
                "embodichain/learning/rl/experimental/_host_replay.py"
            )
            shutil.copy2(resource_path, snapshot_dir / "host_replay_adapter.py")
        OmegaConf.save(cfg, output / "config.yaml")
        torch.save(
            initial, output / ("starting_actor.pt" if snapshot else "initial_actor.pt")
        )
        started = time.monotonic()
        for name in _ADAPTER_FILES:
            path = _runtime_source(f"embodichain/lab/sim/{name}")
            target = snapshot_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        if not replay_restored:
            for offset in range(0, int(cfg.algo.warmup_steps), 16):
                _wait_memory(cfg.training.device, minimum_free)
                learner.collect(min(16, int(cfg.algo.warmup_steps) - offset))
        if snapshot is not None:
            learner.collected += int(snapshot["collected"])
            learner.received += int(snapshot.get("received", snapshot["collected"]))
        accumulator = float(snapshot.get("collection_credit", 0.0)) if snapshot else 0.0
        for iteration in range(start_iteration, args.iterations):
            accumulator += multiplier / float(cfg.algo.utd)
            count = int(accumulator)
            accumulator -= count
            learner.collection_credit = accumulator
            torch.cuda.empty_cache()
            _wait_memory(cfg.training.device, minimum_free)
            metrics = learner.collect(count)
            _wait_memory(cfg.training.device, minimum_free)
            for _ in range(int(cfg.algo.wm_batches)):
                loss, error = learner.wm_update()
            metrics.update(
                {"loss/world_model": float(loss), "loss/world_model_last": float(error)}
            )
            if iteration >= int(cfg.algo.wm_only_iters):
                _wait_memory(cfg.training.device, minimum_free)
                metrics.update(learner.policy_update())
            metrics.update(learner.episodes.metrics())
            metrics.update(
                iteration=iteration + 1,
                collected=learner.collected,
                wall_seconds=time.monotonic() - started,
            )
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if (iteration + 1) % 20 == 0 or iteration == 0:
                print(json.dumps(metrics), flush=True)
            if (
                (iteration + 1) % 100 == 0
                or iteration + 1 == args.iterations
                or (snapshot is not None and iteration == start_iteration)
            ):
                path = _save(
                    output, cfg, learner, iteration + 1, time.monotonic() - started
                )
                print(f"CHECKPOINT {path}", flush=True)
        (output / "completed.json").write_text(
            json.dumps(
                {
                    "iterations": args.iterations,
                    "collected": learner.collected,
                    "native_environment": True,
                    "scope": task.identity,
                    "wall_seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()
