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

"""Visualize a trained PointMass policy as 2D trajectory plots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from tensordict import TensorDict

from embodichain.learning.rl.models import Policy, build_mlp_from_cfg, build_policy
from embodichain.utils.logger import log_info
from embodichain.utils.utility import load_config
from embodichain_tasks.classic_control.point_mass import PointMassEnv


@dataclass(frozen=True)
class EpisodeTrajectory:
    """One deterministic PointMass rollout used for plotting."""

    positions: np.ndarray
    goal: np.ndarray
    obstacles: np.ndarray
    radii: np.ndarray
    success: bool
    final_distance: float

    @property
    def num_steps(self) -> int:
        return max(int(self.positions.shape[0]) - 1, 0)


def build_policy_from_config(
    policy_block: dict[str, Any],
    env: PointMassEnv,
    device: torch.device,
) -> Policy:
    """Build a PointMass policy matching the training config."""
    obs_dim = int(env.single_observation_space.shape[-1])
    action_dim = int(env.single_action_space.shape[-1])
    actor_cfg = policy_block.get("actor")
    critic_cfg = policy_block.get("critic")
    actor = (
        build_mlp_from_cfg(actor_cfg, obs_dim, action_dim)
        if actor_cfg is not None
        else None
    )
    critic = (
        build_mlp_from_cfg(critic_cfg, obs_dim, 1) if critic_cfg is not None else None
    )
    return build_policy(
        policy_block,
        env.single_observation_space,
        env.single_action_space,
        device,
        actor=actor,
        critic=critic,
    )


def load_policy_checkpoint(
    policy: Policy,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """Load policy weights from a Trainer / DifferentiableTrainer checkpoint."""
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if "policy" not in checkpoint:
        raise KeyError(f"Checkpoint {path} does not contain a 'policy' state dict.")
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()


@torch.no_grad()
def rollout_episode(
    env: PointMassEnv,
    policy: Policy,
    *,
    seed: int,
    env_index: int = 0,
) -> EpisodeTrajectory:
    """Roll out one deterministic episode and record the 2D trajectory."""
    if not 0 <= env_index < env.num_envs:
        raise ValueError(f"env_index out of range for num_envs={env.num_envs}.")

    observation, _ = env.reset(seed=seed)
    positions = [env.position[env_index].detach().cpu().numpy().copy()]
    goal = env.goal_position[env_index].detach().cpu().numpy().copy()
    obstacles = env.obstacle_position[env_index].detach().cpu().numpy().copy()
    radii = env.obstacle_radius[env_index].detach().cpu().numpy().copy()

    success = False
    final_distance = float("nan")
    for _ in range(env.max_episode_steps):
        policy_input = TensorDict(
            {"obs": observation},
            batch_size=[env.num_envs],
            device=env.device,
        )
        action = policy.get_action(policy_input, deterministic=True)["action"]
        observation, _, terminated, truncated, info = env.step(action)
        if bool(terminated[env_index] or truncated[env_index]):
            # Auto-reset replaces env.position; use the terminal snapshot in info.
            positions.append(
                info["metrics"]["final_position"][env_index]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            success = bool(info["success"][env_index])
            final_distance = float(info["metrics"]["final_distance"][env_index])
            break
        positions.append(env.position[env_index].detach().cpu().numpy().copy())

    return EpisodeTrajectory(
        positions=np.asarray(positions, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        obstacles=np.asarray(obstacles, dtype=np.float32),
        radii=np.asarray(radii, dtype=np.float32),
        success=success,
        final_distance=final_distance,
    )


def plot_episode(
    episode: EpisodeTrajectory,
    *,
    title: str,
    output_path: Path,
    workspace_half: float,
) -> None:
    """Save a top-down trajectory plot for one episode."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=160)
    ax.set_aspect("equal")
    ax.set_xlim(-workspace_half, workspace_half)
    ax.set_ylim(-workspace_half, workspace_half)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for center, radius in zip(episode.obstacles, episode.radii):
        ax.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                float(radius),
                facecolor="#c44e52",
                edgecolor="#8c2f32",
                alpha=0.35,
                zorder=1,
            )
        )

    ax.plot(
        episode.positions[:, 0],
        episode.positions[:, 1],
        color="#4c72b0",
        linewidth=2.0,
        zorder=2,
        label="trajectory",
    )
    ax.scatter(
        episode.positions[0, 0],
        episode.positions[0, 1],
        c="#4c72b0",
        s=60,
        marker="o",
        zorder=3,
        label="start",
    )
    ax.scatter(
        episode.positions[-1, 0],
        episode.positions[-1, 1],
        c="#55a868",
        s=70,
        marker="*",
        zorder=3,
        label="end",
    )
    ax.scatter(
        episode.goal[0],
        episode.goal[1],
        c="#dd8452",
        s=80,
        marker="x",
        linewidths=2.0,
        zorder=3,
        label="goal",
    )

    status = "success" if episode.success else "fail"
    ax.set_title(
        f"{title}\n{status}, steps={episode.num_steps}, "
        f"final_distance={episode.final_distance:.3f}"
    )
    ax.legend(loc="upper right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained PointMass policy as 2D trajectory plots."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config used to build the env/policy (.yaml/.yml/.json).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint containing a 'policy' state dict.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/point_mass_viz",
        help="Directory for saved PNG plots.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=4,
        help="Number of deterministic episodes to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10042,
        help="Base seed; episode i uses seed + i.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for rollout (cpu or cuda:0).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="policy",
        help="Filename prefix for saved plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")

    cfg = load_config(args.config)
    if "policy" not in cfg:
        raise KeyError(f"Config {args.config} is missing a 'policy' block.")
    trainer_cfg = cfg.get("trainer", {})
    env_block = trainer_cfg.get("learning_env", {})
    env_cfg = {} if isinstance(env_block, str) else dict(env_block.get("cfg", {}))

    device = torch.device(args.device)
    env = PointMassEnv(num_envs=1, device=device, **env_cfg)
    policy = build_policy_from_config(cfg["policy"], env, device)
    load_policy_checkpoint(policy, args.checkpoint, device)

    output_dir = Path(args.output_dir)
    successes = 0
    try:
        for episode_idx in range(args.num_episodes):
            episode = rollout_episode(
                env,
                policy,
                seed=args.seed + episode_idx,
            )
            path = output_dir / f"{args.tag}_ep{episode_idx}.png"
            plot_episode(
                episode,
                title=f"{args.tag} episode {episode_idx}",
                output_path=path,
                workspace_half=env.workspace_half,
            )
            successes += int(episode.success)
            log_info(
                f"saved {path} success={episode.success} "
                f"distance={episode.final_distance:.4f} steps={episode.num_steps}"
            )
    finally:
        env.close()

    log_info(
        f"finished {successes}/{args.num_episodes} successful episodes -> {output_dir}"
    )


if __name__ == "__main__":
    main()
