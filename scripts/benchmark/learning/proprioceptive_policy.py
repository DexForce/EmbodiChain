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

"""Measure temporal policy inference and raw-history rollout capacity.

Run: python -m scripts.benchmark.learning.proprioceptive_policy --device cuda:0

This measures GPU/CPU-resident policy work, not simulation or training throughput.
It writes one Markdown report and a JSON sidecar with raw latency samples.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

import psutil
import torch
from tensordict import TensorDict

from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.algo import PPO, PPOCfg
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.policies import build_policy
from embodichain.learning.rl.utils import OptimizerCfg

__all__ = ["run_all_benchmarks"]


class _CapacityEnvironment:
    """Synthetic tensor workload; it does not model an assembly task."""

    def __init__(self, num_envs: int, policy) -> None:
        self.num_envs = num_envs
        self.device = policy.device
        self.obs_dim = policy.obs_dim
        self.critic_dim = policy.critic_obs_dim
        self.points = policy.cfg.contact_points
        self.step_index = 0
        self._obs = torch.randn(num_envs, self.obs_dim, device=self.device)
        self._critic = torch.randn(num_envs, self.critic_dim, device=self.device)

    def _observation(self) -> dict[str, torch.Tensor]:
        return {
            "proprioception": self._obs,
            "privileged": self._critic,
            "contact_labels": torch.full(
                (self.num_envs, self.points),
                self.step_index % 2,
                dtype=torch.long,
                device=self.device,
            ),
        }

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        self.step_index = 0
        return self._observation(), {}

    def step(self, action: torch.Tensor) -> tuple:
        self.step_index += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._observation(), -action.square().mean(-1), done, done, {}


def _training_capacity(policy, buffer: RolloutBuffer, args: argparse.Namespace) -> dict:
    """Measure a native PPO update on a labeled synthetic tensor rollout."""
    env = _CapacityEnvironment(buffer.num_envs, policy)
    collector = SyncCollector(env, policy, policy.device)
    algorithm = PPO(
        PPOCfg(
            device=str(policy.device),
            n_epochs=1,
            batch_size=4096,
            auxiliary_coef=1.0 if policy.cfg.contact_points else 0.0,
            optimizer=OptimizerCfg(learning_rate=1e-3),
        ),
        policy,
    )
    rollout = collector.collect(args.rollout_length, rollout=buffer.start_rollout())
    encoder_before = next(policy.actor.encoder.parameters()).detach().clone()
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    start = time.perf_counter()
    metrics = algorithm.update(rollout)
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    encoder_changed = not torch.equal(
        encoder_before, next(policy.actor.encoder.parameters())
    )
    # A separate warmed update measures allocator memory independently of time.
    rollout = collector.collect(args.rollout_length, rollout=buffer.start_rollout())
    peak = None
    if policy.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(policy.device)
    algorithm.update(rollout)
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
        peak = torch.cuda.max_memory_allocated(policy.device) / 1024**2
    obs = rollout["obs"][:, 0]
    first = TensorDict(
        {"obs": obs, "critic_obs": rollout["critic_obs"][:, 0]},
        batch_size=[buffer.num_envs],
        device=policy.device,
    )
    alternative = first.clone()
    alternative["critic_obs"].add_(100)
    alternative["contact_labels"] = torch.ones(
        buffer.num_envs,
        policy.cfg.contact_points,
        dtype=torch.long,
        device=policy.device,
    )
    with torch.inference_mode():
        original_action = policy.get_action(first, deterministic=True)["action"]
        alternative_action = policy.get_action(alternative, deterministic=True)[
            "action"
        ]
    return {
        "scope": "synthetic resident tensor rollout; not assembly training or simulator throughput",
        "num_envs": buffer.num_envs,
        "rollout_length": args.rollout_length,
        "epochs": 1,
        "batch_size": 4096,
        "first_update_ms": elapsed_ms,
        "warmed_update_peak_torch_mib": peak,
        "encoder_parameters_changed": encoder_changed,
        "privileged_action_max_error": float(
            (original_action - alternative_action).abs().max()
        ),
        "finite_gradients": all(
            p.grad is None or bool(p.grad.isfinite().all()) for p in policy.parameters()
        ),
        "metrics": metrics,
    }


def _table(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return ["No measurements."]
    keys = list(rows[0])
    return [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
        *("| " + " | ".join(str(row[key]) for key in keys) + " |" for row in rows),
    ]


def run_all_benchmarks(args: argparse.Namespace) -> Path:
    """Benchmark deterministic exported actor and one largest rollout buffer."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; use --device cpu.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    policy = build_policy(
        {
            "name": "proprioceptive_actor_critic",
            "obs_groups": {"actor": ["proprioception"], "critic": ["privileged"]},
            "cfg": {
                "frame_dim": args.frame_dim,
                "contact_points": args.contact_points,
                "contact_classes": args.contact_classes,
            },
        },
        args.frame_dim * 31,
        7,
        device,
        critic_obs_space=args.critic_dim,
    ).eval()
    actor = policy.export_actor()

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    performance: list[dict[str, object]] = []
    accuracy: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    for count in args.env_counts:
        obs = torch.randn(count, policy.obs_dim, device=device)
        critic_obs = torch.randn(count, args.critic_dim, device=device)
        with torch.inference_mode():
            reference = policy.get_action(
                TensorDict(
                    {"obs": obs, "critic_obs": critic_obs},
                    batch_size=[count],
                    device=device,
                ),
                deterministic=True,
            )["action"]
            for _ in range(args.warmup):
                actor(obs)
            synchronize()
            samples = []
            for _ in range(args.repeats):
                synchronize()
                start = time.perf_counter()
                result = actor(obs)
                synchronize()
                samples.append(1000 * (time.perf_counter() - start))
                del result
            process = psutil.Process()
            cpu_before = process.memory_info().rss
            gpu_before = None
            if device.type == "cuda":
                gpu_before = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            result = actor(obs)
            synchronize()
            cpu_delta = (process.memory_info().rss - cpu_before) / 1024**2
            gpu_delta = peak = None
            if gpu_before is not None:
                gpu_delta = (torch.cuda.memory_allocated(device) - gpu_before) / 1024**2
                peak = torch.cuda.max_memory_allocated(device) / 1024**2
            finite = result.isfinite().all(-1)
            err = float((result - reference).abs().max())
            success = float(
                (finite & ((result - reference).abs().amax(-1) <= 1e-5)).float().mean()
            )
        performance.append(
            {
                "sample_size": count,
                "impl": "exported_actor_fp32",
                "cost_time_ms": statistics.median(samples),
                "cpu_delta_mb": cpu_delta,
                "gpu_delta_mb": gpu_delta,
                "peak_gpu_mb": peak,
            }
        )
        accuracy.append(
            {
                "sample_size": count,
                "impl": "exported_actor_fp32",
                "success_rate": success,
                "maximum_export_error": err,
            }
        )
        raw.append({"num_envs": count, "samples_ms": samples})
        del obs, critic_obs, result, reference
    largest = max(args.env_counts)
    buffer = RolloutBuffer(
        largest,
        args.rollout_length,
        policy.obs_dim,
        7,
        device,
        critic_obs_dim=args.critic_dim,
        distribution_param_dim=7,
        training_field_specs=policy.training_field_specs,
    )
    payload_mib = (
        sum(t.numel() * t.element_size() for t in buffer.buffer.values()) / 1024**2
    )
    metadata = {
        "hardware": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": str(device),
        "dtype": "float32",
        "seed": args.seed,
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_scope": "resident exported actor; completed-call wall, explicit device synchronization",
        "memory_scope": "separate warmed call retaining one output; CPU RSS retained delta, Torch allocator peak only",
        "rollout_payload_mib": payload_mib,
        "rollout_envs": largest,
        "rollout_length": args.rollout_length,
        "frame_dim": args.frame_dim,
        "critic_dim": args.critic_dim,
        "contact_points": args.contact_points,
        "contact_classes": args.contact_classes,
    }
    if args.training_update:
        metadata["training_capacity"] = _training_capacity(policy, buffer, args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output / f"proprioceptive_policy_{stamp}.md"
    leaderboard = [
        {
            "rank": 1,
            "algorithm": "exported_actor_fp32",
            "overall_success_rate": statistics.mean(
                row["success_rate"] for row in accuracy
            ),
        }
    ]
    lines = [
        "# Proprioceptive policy benchmark",
        "",
        "## Time & Memory",
        "",
        *_table(performance),
        "",
        "## Success & Other Metrics",
        "",
        *_table(accuracy),
        "",
        "## Leaderboard",
        "",
        *_table(leaderboard),
        "",
        "## Measurement",
        "",
        *(f"- {key}: {value}" for key, value in metadata.items()),
    ]
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "performance": performance,
                "accuracy": accuracy,
                "raw": raw,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Markdown report saved: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--env-counts", nargs="+", type=int, default=[1, 64, 256, 1024, 12288]
    )
    parser.add_argument("--frame-dim", type=int, default=36)
    parser.add_argument("--critic-dim", type=int, default=64)
    parser.add_argument("--contact-points", type=int, default=128)
    parser.add_argument("--contact-classes", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/benchmarks")
    parser.add_argument(
        "--training-update",
        action="store_true",
        help="Run one timed and one separate memory-profiled synthetic native PPO update.",
    )
    parsed = parser.parse_args()
    if any(
        value <= 0
        for value in [*parsed.env_counts, parsed.warmup, parsed.repeats, parsed.threads]
    ):
        parser.error(
            "Environment counts, warmup, repeats and threads must be positive."
        )
    run_all_benchmarks(parsed)
