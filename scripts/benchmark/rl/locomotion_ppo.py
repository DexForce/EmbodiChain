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
"""Measure G1 EmbodiChain PPO rollout and optimizer throughput on CUDA."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

__all__ = ["main"]

BACKENDS = {
    "default": "Dexsim Default CUDA",
    "newton": "Dexsim Newton/MJWarp CUDA",
}
TASK_DIR = (
    Path(__file__).resolve().parents[3]
    / "embodichain_tasks/configs/tasks/locomotion/velocity/g1_flat"
)


def _load_config(task_dir: Path, backend: str, envs: int, steps: int) -> dict:
    import yaml

    suffix = ".newton" if backend == "newton" else ""
    config = yaml.safe_load((task_dir / f"agents/ppo{suffix}.yaml").read_text())
    config["trainer"].update(
        gym_config=str((task_dir / f"env{suffix}.yaml").resolve()),
        num_envs=envs,
        buffer_size=steps,
    )
    mini_batches = config["algorithm"]["cfg"]["num_mini_batches"]
    if envs * steps < mini_batches or envs * steps % mini_batches:
        raise ValueError("Rollout size must be divisible by num_mini_batches")
    config["algorithm"]["cfg"]["batch_size"] = envs * steps // mini_batches
    return config


def _build_trainer(runtime: Any, config: dict, output: Path) -> Any:
    from embodichain.learning.rl.algo import build_algo
    from embodichain.learning.rl.utils.trainer import Trainer

    algorithm = build_algo(
        "ppo", config["algorithm"]["cfg"], runtime.policy, runtime.device
    )
    return Trainer(
        policy=runtime.policy,
        env=runtime.env,
        algorithm=algorithm,
        buffer_size=config["trainer"]["buffer_size"],
        batch_size=algorithm.cfg.batch_size,
        writer=None,
        eval_freq=0,
        save_freq=0,
        checkpoint_dir=str(output),
        exp_name="g1-ppo-benchmark",
        use_wandb=False,
    )


def _check_backend(env: Any, name: str) -> Any:
    import torch

    sim = env.unwrapped.sim
    if sim.physics.name != name or torch.device(sim.device).type != "cuda":
        raise RuntimeError("Requested CUDA physics backend is not active")
    if name == "newton":
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(sim._world)
        if (
            backend.solver.use_mujoco_cpu
            or not backend.solver.mjw_data.qpos.device.is_cuda
        ):
            raise RuntimeError("Newton/MJWarp is not using CUDA")
        return backend
    if not (sim._world_config.enable_gpu_sim and sim._world_config.direct_gpu_api):
        raise RuntimeError("Default backend is not using direct GPU simulation")
    return None


def _measure(operation: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    import psutil
    import torch

    torch.cuda.synchronize()
    process = psutil.Process()
    cpu_before = process.memory_info().rss
    gpu_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return value, {
        "wall_s": elapsed,
        "cpu_delta_mb": (process.memory_info().rss - cpu_before) / 2**20,
        "gpu_delta_mb": (torch.cuda.memory_allocated() - gpu_before) / 2**20,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def _summarize(result: dict) -> dict:
    measured = [row for row in result["updates"] if not row["warmup"]]
    if not measured or result["status"] != "passed":
        return {}
    rollout = sum(row["rollout"]["wall_s"] for row in measured)
    update = sum(row["ppo_update"]["wall_s"] for row in measured)
    transitions = result["num_envs"] * result["steps"] * len(measured)
    rates = [
        result["num_envs"]
        * result["steps"]
        / (row["rollout"]["wall_s"] + row["ppo_update"]["wall_s"])
        for row in measured
    ]
    samples = result.get("resources", {}).get("samples", [])
    gpu_samples = [row["gpu_process_mib"] for row in samples]
    return {
        "rollout_sps": transitions / rollout,
        "ppo_sps": transitions / (rollout + update),
        "rollout_ms": rollout * 1000 / len(measured),
        "update_ms": update * 1000 / len(measured),
        "ppo_sps_cv_pct": statistics.pstdev(rates) / statistics.mean(rates) * 100,
        "cpu_rss_peak_mib": max(
            (
                max(row["cpu_rss_mib"], row["cpu_rss_lifetime_peak_mib"])
                for row in samples
            ),
            default=None,
        ),
        "gpu_process_sampled_peak_mib": (
            max(gpu_samples) if gpu_samples and None not in gpu_samples else None
        ),
    }


def _write_report(output: Path, result: dict) -> None:
    summary = _summarize(result)
    label = BACKENDS[result["backend"]]
    runtime = result.get("runtime", {})

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        "# G1 EmbodiChain PPO",
        "",
        f"{label}; {result['num_envs']} environments × {result['steps']} steps; "
        f"{result['warmup']} warmup + {result['measured_updates']} measured updates.",
        "",
        f"GPU: {runtime.get('gpu', 'n/a')}; CUDA: {runtime.get('cuda', 'n/a')}; "
        f"DexSim: {runtime.get('dexsim_commit', 'n/a')}; seed: {result.get('seed', 'n/a')}.",
        "",
        "Memory is in MiB. GPU delta/peak use the PyTorch allocator; "
        "resources in result.json also record process VRAM at phase boundaries.",
        "",
        f"Process RAM (RSS peak): {fmt(summary.get('cpu_rss_peak_mib'))} MiB; "
        f"process VRAM (sampled peak): {fmt(summary.get('gpu_process_sampled_peak_mib'))} MiB.",
        "",
        "## Time & Memory",
        "",
        "| Phase | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- | --- |",
    ]
    for phase in ("create_runtime", "rollout", "ppo_update"):
        if phase == "create_runtime":
            rows = [result[phase]] if phase in result else []
        else:
            rows = [row[phase] for row in result["updates"] if not row["warmup"]]
        if summary and rows:
            values = [
                statistics.mean(row["wall_s"] for row in rows) * 1000,
                statistics.mean(row["cpu_delta_mb"] for row in rows),
                statistics.mean(row["gpu_delta_mb"] for row in rows),
                max(row["peak_gpu_mb"] for row in rows),
            ]
            lines.append(
                f"| {phase} | " + " | ".join(f"{v:.3f}" for v in values) + " |"
            )
        else:
            lines.append(f"| {phase} | n/a | n/a | n/a | n/a |")
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Backend | Status | Success rate | Rollout SPS | PPO SPS | PPO SPS CV (%) |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {label} | {result['status']} | n/a | "
        + (
            f"{summary['rollout_sps']:.1f} | {summary['ppo_sps']:.1f} | {summary['ppo_sps_cv_pct']:.2f}"
            if summary
            else "n/a | n/a | n/a"
        )
        + " |",
        "",
        "## Leaderboard",
        "",
        "| Algorithm | Backend | Success rank |",
        "| --- | --- | --- |",
        f"| EmbodiChain PPO | {label} | n/a |",
        "",
        "This workload measures short training throughput; episode success is not evaluated.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace, result: dict) -> None:
    import torch
    import yaml
    import dexsim
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.learning.rl.runtime import build_gym_policy_runtime
    from scripts.benchmark.rl.resource_usage import ProcessResources
    from scripts.benchmark.rl.runtime import set_random_seed

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    set_random_seed(args.seed, device)
    discover_task_packages()
    execute_init_hooks()
    config = _load_config(args.task_dir, args.backend, args.envs, args.steps)
    env_config = yaml.safe_load(Path(config["trainer"]["gym_config"]).read_text())
    env_config["num_envs"] = args.envs
    if args.backend == "newton":
        env_config["physics_config"]["solver_cfg"][
            "iterations"
        ] = args.solver_iterations
    env_path = args.output / "env.yaml"
    env_path.write_text(yaml.safe_dump(env_config, sort_keys=False))
    config["trainer"]["gym_config"] = str(env_path.resolve())
    result["configuration"] = {
        "policy": config["policy"],
        "algorithm": config["algorithm"],
        "physics": env_config["physics_config"],
    }
    result["runtime"] = {
        "dexsim_commit": getattr(dexsim, "__commit_id__", None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "cpu_threads": torch.get_num_threads(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("newton", "warp-lang", "mujoco-warp")
        },
    }
    resources = ProcessResources(str(torch.cuda.get_device_properties(device).uuid))
    runtime = None
    try:
        resources.sample("before_create")
        runtime, result["create_runtime"] = _measure(
            lambda: build_gym_policy_runtime(
                config,
                device=device,
                num_envs=args.envs,
                headless=True,
                renderer=config["trainer"]["renderer"],
                gpu_id=0,
                seed=args.seed,
                config_dir=args.task_dir,
            )
        )
        backend = _check_backend(runtime.env, args.backend)
        trainer = _build_trainer(runtime, config, args.output)
        trainer.algorithm.bind_schedule(total_updates=args.warmup + args.updates)
        resources.sample("after_create")
        for index in range(args.warmup + args.updates):
            warming = index < args.warmup
            if index == args.warmup:
                before = [p.detach().clone() for p in runtime.policy.parameters()]
                resources.sample("after_warmup")
                if backend is not None and backend.cuda_graph_status != "captured":
                    raise RuntimeError(
                        "Newton CUDA graph was not captured during warmup"
                    )
            _, rollout = _measure(trainer._collect_rollout)
            batch = trainer.buffer.get(flatten=False)
            if not all(
                bool(torch.isfinite(batch[key]).all())
                for key in ("obs", "critic_obs", "action", "reward")
            ):
                raise RuntimeError("Nonfinite rollout")
            losses, update = _measure(lambda: trainer.algorithm.update(batch))
            trainer.num_updates += 1
            if not all(math.isfinite(float(value)) for value in losses.values()):
                raise RuntimeError("Nonfinite PPO loss")
            if backend is not None and bool(
                backend.solver.mjw_data.overflow.numpy().any()
            ):
                raise RuntimeError("MJWarp capacity or solver iteration limit reached")
            result["updates"].append(
                {
                    "warmup": warming,
                    "rollout": rollout,
                    "ppo_update": update,
                    "losses": {k: float(v) for k, v in losses.items()},
                }
            )
            resources.sample("warmup" if warming else f"measured_{index}")
        delta = max(
            float((old - new.detach()).abs().max())
            for old, new in zip(before, runtime.policy.parameters())
        )
        if (
            not math.isfinite(delta)
            or delta <= 0
            or not all(
                bool(torch.isfinite(p).all()) for p in runtime.policy.parameters()
            )
        ):
            raise RuntimeError("PPO did not produce finite parameter updates")
        result["parameter_max_abs_delta"] = delta
        trainer.save_checkpoint(str(args.output / "model.pt"))
    finally:
        result["resources"] = resources.result()
        if runtime is not None:
            runtime.close()


def main() -> None:
    """Run one backend and save timings, a checkpoint, and a Markdown report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, default=TASK_DIR)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--solver-iterations",
        type=int,
        default=20,
        help="Newton solver iteration limit",
    )
    args = parser.parse_args()
    if (
        min(
            args.envs,
            args.steps,
            args.warmup,
            args.updates,
            args.cpu_threads,
            args.solver_iterations,
        )
        < 1
    ):
        parser.error("Workload sizes and CPU thread count must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "protocol": "g1-embodichain-ppo",
        "backend": args.backend,
        "num_envs": args.envs,
        "steps": args.steps,
        "warmup": args.warmup,
        "measured_updates": args.updates,
        "seed": args.seed,
        "status": "failed",
        "updates": [],
    }
    try:
        _run(args, result)
        result["status"] = "passed"
    finally:
        result["summary"] = _summarize(result)
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        _write_report(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
