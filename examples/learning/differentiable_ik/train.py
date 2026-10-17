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

"""Train through Franka IK and a frozen learned joint-tracking model.

Run: python -m examples.learning.differentiable_ik.train --device cuda
The state model learns synthetic q_next = q + 0.2 * (q_des - q) transitions.
It is an integration experiment, not a model of physical Franka dynamics.
No SimulationManager or physics engine is constructed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from embodichain.compute.kinematics import (
    DifferentiableIKCfg,
    SerialChainKinematics,
    pose_error,
)
from embodichain.data import get_data_path
from embodichain.learning.layers import DifferentiableIKLayer, KinematicsLayer
from embodichain.learning.models import MLP
from embodichain.learning.rl.algo import APG, APGCfg
from embodichain.learning.rl.differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from embodichain.learning.rl.models import ActorOnly
from embodichain.learning.rl.utils import OptimizerCfg


class JointTrackingModel(nn.Module):
    """Example-only learned residual transition, with fixed geometry context."""

    def __init__(self, center: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("center", center.clone())
        self.net = MLP(14, 7, [64, 64], activation="tanh").to(center.device)

    def forward(self, q: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return q + 0.2 * self.net(torch.cat((q - self.center, command - q), -1))


def pretrain_model(
    model: JointTrackingModel, steps: int, batch: int
) -> dict[str, float]:
    """Fit synthetic transitions and report held-out model error."""
    device = model.center.device
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    validation_generator = torch.Generator(device=device).manual_seed(7103)
    validation_q = model.center + 0.3 * torch.randn(
        256, 7, generator=validation_generator, device=device
    )
    validation_command = validation_q + 0.2 * torch.randn(
        256, 7, generator=validation_generator, device=device
    )
    with torch.no_grad():
        truth = validation_q + 0.2 * (validation_command - validation_q)
        initial = (
            (model(validation_q, validation_command) - truth)
            .square()
            .mean()
            .sqrt()
            .item()
        )
    for _ in range(steps):
        q = model.center + 0.3 * torch.randn(batch, 7, device=device)
        command = q + 0.2 * torch.randn(batch, 7, device=device)
        loss = (model(q, command) - (q + 0.2 * (command - q))).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = (
            (model(validation_q, validation_command) - truth)
            .square()
            .mean()
            .sqrt()
            .item()
        )
    model.requires_grad_(False).eval()
    # Free pretraining adjoints; frozen parameters still permit input gradients.
    for parameter in model.parameters():
        parameter.grad = None
    return {"model_initial_rmse": initial, "model_final_rmse": final}


def _correct_goal(goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    # An SO(3) exponential gives valid target rotations, including at zero.
    x, y, z = (0.3 * action[:, 3:]).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    rotation = torch.matrix_exp(skew) @ goal[:, :3, :3]
    position = goal[:, :3, 3] + 0.1 * action[:, :3]
    return torch.cat((torch.cat((rotation, position[..., None]), -1), goal[:, 3:]), -2)


class FrankaIKEnv:
    """Example-only DifferentiableVecEnv combining IK and learned transitions."""

    def __init__(
        self,
        ik: DifferentiableIKLayer,
        dynamics: JointTrackingModel,
        batch: int,
        horizon: int,
        seed: int,
    ) -> None:
        self.ik, self.dynamics = ik, dynamics
        self.num_envs, self.horizon = batch, horizon
        self.device = ik.kinematics.tcp.device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (13,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        self.q: torch.Tensor
        self.goal: torch.Tensor
        self.elapsed = 0

    def _observation(self) -> torch.Tensor:
        p, r = pose_error(self.ik.kinematics(self.q), self.goal)
        return torch.cat((self.q - self.dynamics.center, p, r), -1)

    def reset(
        self, *, seed: int | None = None, options: Any = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None:
            self.generator.manual_seed(seed)
        k = self.ik.kinematics
        with torch.no_grad():
            self.q = (
                self.dynamics.center
                + 0.12
                * torch.randn(
                    self.num_envs, 7, generator=self.generator, device=self.device
                )
            ).clamp(min=k.lower, max=k.upper)
            goal_q = (
                self.q
                + 0.15
                * torch.randn(
                    self.num_envs, 7, generator=self.generator, device=self.device
                )
            ).clamp(min=k.lower, max=k.upper)
            self.goal = k(goal_q)
        self.elapsed = 0
        return self._observation(), {}

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        target = _correct_goal(self.goal, action)
        result = self.ik.solve(target, self.q)
        self.q = self.dynamics(self.q, result.qpos)
        p, r = pose_error(self.ik.kinematics(self.q), self.goal)
        loss = p.square().sum(-1) + 0.1 * r.square().sum(-1)
        reward = -100 * loss
        self.elapsed += 1
        done = self.elapsed >= self.horizon
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.full_like(terminated, done)
        info = {
            "loss": loss.detach(),
            "success": ((p.norm(dim=-1) < 0.01) & (r.norm(dim=-1) < 0.1)).detach(),
            "ik_success": result.success.detach(),
        }
        # Keep the terminal reward graph alive; only the next observation resets.
        observation = self.reset()[0] if done else self._observation()
        return observation, reward, terminated, truncated, info

    def detach_state(self) -> torch.Tensor:
        self.q = self.q.detach()
        return self._observation().detach()

    def close(self) -> None:
        pass


def evaluate(actor: nn.Module, env: FrankaIKEnv, seed: int) -> dict[str, float]:
    """Evaluate the same held-out initial states before and after training."""
    with torch.no_grad():
        obs, _ = env.reset(seed=seed)
        losses = []
        for _ in range(env.horizon):
            obs, _, _, _, info = env.step(actor(obs))
            losses.append(info["loss"].mean())
        return {
            "loss": torch.stack(losses).mean().item(),
            "success_rate": info["success"].float().mean().item(),
            "ik_success_rate": info["ik_success"].float().mean().item(),
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Pretrain transitions, then train an actor using existing complete-rollout APG."""
    if (
        min(
            args.batch_size,
            args.horizon,
            args.ik_iterations,
            args.updates,
            args.model_steps,
        )
        < 1
    ):
        raise ValueError(
            "Batch, horizon, iterations and training budgets must be positive"
        )
    torch.manual_seed(args.seed)
    device = torch.empty(0, device=args.device).device
    # Matches the official FrankaPandaCfg arm: base -> fr3_hand_tcp, identity TCP.
    path = get_data_path("Franka/Panda/PandaWithHand.urdf")
    model = SerialChainKinematics.from_urdf(
        Path(path).read_text(), "fr3_hand_tcp", "base", device=device
    )
    kinematics = KinematicsLayer(model)
    ik = DifferentiableIKLayer(
        kinematics, DifferentiableIKCfg(max_iterations=args.ik_iterations)
    )
    center = kinematics.default_seed().clone()
    center[1] = -0.4  # Avoid the midpoint shoulder singularity.
    dynamics = JointTrackingModel(center)
    model_metrics = pretrain_model(
        dynamics, args.model_steps, max(256, args.batch_size)
    )
    env = FrankaIKEnv(ik, dynamics, args.batch_size, args.horizon, args.seed)
    eval_env = FrankaIKEnv(
        ik, dynamics, args.batch_size, args.horizon, args.seed + 10000
    )
    actor = MLP(13, 6, [64, 64], activation="tanh", last_activation="tanh").to(device)
    actor.init_orthogonal(scales=[2**0.5, 2**0.5, 0.01])
    policy = ActorOnly(13, 6, device, actor)
    algorithm = APG(
        APGCfg(
            device=str(device),
            gamma=1.0,
            max_grad_norm=10.0,
            optimizer=OptimizerCfg(learning_rate=1e-3),
        ),
        policy,
    )
    trainer = DifferentiableTrainer(
        DifferentiableTrainerCfg(
            rollout_mode="complete",
            segment_length=args.horizon,
            update_horizon=args.horizon,
            deterministic_actions=True,
            rollout_seed=args.seed,
        ),
        env,
        policy,
        algorithm,
    )
    initial = evaluate(actor, eval_env, args.seed + 10000)
    summary = trainer.train(total_updates=args.updates)
    final = evaluate(actor, eval_env, args.seed + 10000)
    finite = all(
        p.grad is None or torch.isfinite(p.grad).all().item()
        for p in policy.parameters()
    )
    metrics = {
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "ik_iterations": args.ik_iterations,
        "updates": args.updates,
        **model_metrics,
        "initial": initial,
        "final": final,
        "policy_gradients_finite": finite,
        "skipped_updates": sum(
            int(row.get("train/skipped_update", 0)) for row in summary["train_history"]
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    trainer.save_checkpoint(args.output / "policy.pt")
    torch.save(
        {
            "dynamics": dynamics.state_dict(),
            "ik": ik.state_dict(),
            "robot_asset": "Franka/Panda/PandaWithHand.urdf",
        },
        args.output / "models.pt",
    )
    env.close()
    eval_env.close()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--ik-iterations", type=int, default=16)
    parser.add_argument("--model-steps", type=int, default=500)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/differentiable_ik")
    )
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
