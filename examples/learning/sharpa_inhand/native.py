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
from _aligned_apg import _APGLearner
from _native_task import _NativeTask
from omegaconf import OmegaConf
from sharpa_rl_unilab.training.configuration import configure_threads
from sharpa_rl_unilab.training.policy import (
    inference_distribution,
    make_actor,
    make_models,
)

__all__ = ["main"]


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
            "critic": learner.critic.state_dict(),
            "world_model": learner.wm.state_dict(),
            "actor_optimizer": learner.opt_pi.state_dict(),
            "critic_optimizer": learner.opt_v.state_dict(),
            "world_model_optimizer": learner.opt_wm.state_dict(),
            "iteration": iteration,
            "collected": learner.collected,
            "wall_seconds": elapsed,
            "native_identity": learner.env.identity,
        },
        path,
    )
    return path


@torch.no_grad()
def _evaluate(
    task: _NativeTask, actor: torch.nn.Module, seeds: list[int], output: Path
) -> dict:
    records = []
    traces = []
    actor.eval()
    for seed in seeds:
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
    result = {
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
    """Own the explicit native calibration experiment and its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resource-profile", choices=["reference", "compact"], default="reference"
    )
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=[10001, 10002, 10003]
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.reference_config)
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
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True
    ).strip()
    if revision != "9816885a3b1fbdb947db71e72c27ed75effcbf9f":
        raise ValueError("Expected the qualified upstream reference revision")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=args.upstream, text=True
    ).strip():
        raise ValueError("The upstream reference sources must remain clean")
    if (output / "metrics.jsonl").exists():
        raise FileExistsError("Use a fresh native training output directory")
    snapshot_dir = output / "sources"
    snapshot_dir.mkdir(exist_ok=True)
    hashes = {}
    for name in [
        "native.py",
        "_native_task.py",
        "_native_asset.py",
        "_simulation.py",
        "_aligned_apg.py",
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
                "native_fixed_scale_calibration": True,
            },
            indent=2,
        )
        + "\n"
    )
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
        task = _NativeTask(
            cfg, args.upstream.resolve(), output, num_envs=args.num_envs, seed=args.seed
        )
        learner = _APGLearner(
            task, cfg, actor, critic, task.mdp, task.capture, value_bootstrap=True
        )
        OmegaConf.save(cfg, output / "config.yaml")
        torch.save(initial, output / "initial_actor.pt")
        started = time.monotonic()
        learner.collect(int(cfg.algo.warmup_steps))
        accumulator = 0.0
        for iteration in range(args.iterations):
            accumulator += multiplier / float(cfg.algo.utd)
            count = int(accumulator)
            accumulator -= count
            torch.cuda.empty_cache()
            metrics = learner.collect(count)
            for _ in range(int(cfg.algo.wm_batches)):
                loss, error = learner.wm_update()
            metrics.update(
                {"loss/world_model": float(loss), "loss/world_model_last": float(error)}
            )
            if iteration >= int(cfg.algo.wm_only_iters):
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
            if (iteration + 1) % 250 == 0 or iteration + 1 == args.iterations:
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
