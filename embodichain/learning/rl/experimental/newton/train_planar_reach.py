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

"""Train APG on random Newton planar reach problems."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from embodichain.learning.rl.algo import APG, APGCfg
from embodichain.learning.rl.collector import (
    DifferentiableCollector,
    DifferentiableRollout,
)
from embodichain.learning.rl.differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from embodichain.learning.rl.models import ActorOnly, MLP
from embodichain.utils import configclass

from .planar_reach import NewtonPlanarReachEnv, NewtonPlanarReachEnvCfg

__all__ = ["NewtonPlanarReachTrainingCfg", "train_planar_reach"]


@configclass
class NewtonPlanarReachTrainingCfg:
    """Configuration for random-target Newton APG training."""

    device: str = "cpu"
    seed: int = 29
    eval_seed: int = 10029
    eval_batches: int = 4
    num_envs: int = 128
    num_updates: int = 600
    horizon: int = 32
    learning_rate: float = 3e-3
    action_scale: float = 0.25
    success_threshold: float = 0.05
    initial_joint_scale: float = 0.5
    target_joint_scale: float = 1.0


def train_planar_reach(
    cfg: NewtonPlanarReachTrainingCfg | None = None,
) -> dict[str, Any]:
    """Train on random reaches and evaluate on held-out samples.

    Args:
        cfg: Demonstration configuration.

    Returns:
        Metrics before and after training on held-out evaluation batches.
    """
    cfg = cfg or NewtonPlanarReachTrainingCfg()
    if cfg.eval_batches <= 0:
        raise ValueError("eval_batches must be positive.")
    if cfg.num_updates < 0:
        raise ValueError("num_updates cannot be negative.")
    if cfg.horizon <= 0:
        raise ValueError("horizon must be positive.")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=cfg.num_envs,
            device=cfg.device,
            action_scale=cfg.action_scale,
            success_threshold=0.0,
            max_episode_steps=cfg.horizon,
            initial_joint_scale=cfg.initial_joint_scale,
            target_joint_scale=cfg.target_joint_scale,
        )
    )
    eval_env = NewtonPlanarReachEnv(
        NewtonPlanarReachEnvCfg(
            num_envs=cfg.num_envs,
            device=cfg.device,
            action_scale=cfg.action_scale,
            success_threshold=0.0,
            max_episode_steps=cfg.horizon + 1,
            initial_joint_scale=cfg.initial_joint_scale,
            target_joint_scale=cfg.target_joint_scale,
        )
    )
    actor = MLP(
        input_dim=8,
        output_dim=2,
        hidden_dims=[64, 64],
        activation="tanh",
        last_activation="tanh",
    ).to(device)
    actor.init_orthogonal(scales=[2**0.5, 2**0.5, 0.01])
    policy = ActorOnly(8, 2, device, actor=actor)
    algorithm = APG(
        APGCfg(
            device=cfg.device,
            learning_rate=cfg.learning_rate,
            gamma=0.99,
            max_grad_norm=10.0,
        ),
        policy,
    )
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(
            segment_length=cfg.horizon,
            deterministic_actions=True,
        ),
        train_env,
        policy,
        algorithm,
    )
    eval_collector = DifferentiableCollector(eval_env, policy, device)

    initial_metrics = _evaluate(eval_collector, cfg)
    trainer.collector.reset(seed=cfg.seed)
    training_summary = trainer.train(
        total_timesteps=cfg.num_updates * cfg.horizon * cfg.num_envs
    )
    skipped_updates = sum(
        int(entry["train/skipped_update"])
        for entry in training_summary["train_history"]
    )

    final_metrics = _evaluate(eval_collector, cfg)
    return {
        "seed": cfg.seed,
        "eval_seed": cfg.eval_seed,
        "eval_samples": cfg.eval_batches * cfg.num_envs,
        "num_envs": cfg.num_envs,
        "num_updates": training_summary["num_updates"],
        "global_step": training_summary["global_step"],
        "horizon": cfg.horizon,
        "initial_return": initial_metrics["return"],
        "final_return": final_metrics["return"],
        "initial_mean_min_distance": initial_metrics["mean_min_distance"],
        "final_mean_min_distance": final_metrics["mean_min_distance"],
        "final_distance": final_metrics["final_distance"],
        "initial_success_rate": initial_metrics["success_rate"],
        "final_success_rate": final_metrics["success_rate"],
        "success_threshold": cfg.success_threshold,
        "skipped_updates": skipped_updates,
    }


def _evaluate(
    collector: DifferentiableCollector,
    cfg: NewtonPlanarReachTrainingCfg,
) -> dict[str, float]:
    batch_metrics = []
    for batch in range(cfg.eval_batches):
        collector.reset(seed=cfg.eval_seed + batch)
        rollout = collector.collect(cfg.horizon, deterministic=True)
        batch_metrics.append(_trajectory_metrics(rollout, cfg.success_threshold))
    return {
        key: sum(metrics[key] for metrics in batch_metrics) / cfg.eval_batches
        for key in batch_metrics[0]
    }


def _trajectory_metrics(
    rollout: DifferentiableRollout,
    success_threshold: float,
) -> dict[str, float]:
    distances = torch.stack(
        [transition.info["distance"] for transition in rollout.transitions]
    )
    min_distances = distances.min(dim=0).values
    return {
        "return": rollout.rewards.sum(dim=0).mean().detach().item(),
        "mean_min_distance": min_distances.mean().detach().item(),
        "final_distance": distances[-1].mean().detach().item(),
        "success_rate": (min_distances < success_threshold).float().mean().item(),
    }


def _parse_args() -> NewtonPlanarReachTrainingCfg:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-updates", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()
    return NewtonPlanarReachTrainingCfg(
        device=args.device,
        seed=args.seed,
        eval_batches=args.eval_batches,
        num_envs=args.num_envs,
        num_updates=args.num_updates,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    print(json.dumps(train_planar_reach(_parse_args()), indent=2))
