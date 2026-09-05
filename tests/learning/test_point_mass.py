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

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from embodichain.learning.rl.algo import PPO, PPOCfg
from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.env import (
    DifferentiableVecEnv,
    build_learning_env,
)
from embodichain.learning.rl.models import ActorCritic
from embodichain.learning.rl.train import train_from_config
from embodichain.learning.rl.utils import OptimizerCfg
from embodichain.learning.rl.utils.trainer import Trainer
from embodichain_tasks.classic_control.point_mass import PointMassEnv


def test_point_mass_is_registered_differentiable_environment() -> None:
    env = build_learning_env("PointMassRL", num_envs=4, device="cpu")

    assert isinstance(env, PointMassEnv)
    assert isinstance(env, DifferentiableVecEnv)
    assert env.single_observation_space.shape == (14,)
    assert env.single_action_space.shape == (2,)


def test_point_mass_reset_seed_is_reproducible() -> None:
    env = PointMassEnv(num_envs=8)

    first, _ = env.reset(seed=17)
    env.step(torch.ones(8, 2))
    second, _ = env.reset(seed=17)

    torch.testing.assert_close(first, second)


def test_point_mass_step_preserves_action_gradient() -> None:
    env = PointMassEnv(num_envs=4)
    env.reset(seed=3)
    action = torch.randn(4, 2, requires_grad=True)

    next_observation, reward, _, _, _ = env.step(action)
    reward.sum().backward()

    assert next_observation.grad_fn is not None
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    assert float(action.grad.norm()) > 0.0


def test_point_mass_auto_reset_keeps_terminal_metrics() -> None:
    env = PointMassEnv(num_envs=2)
    env.reset(seed=5)
    env.velocity.zero_()
    env.goal_position[0] = env.position[0]
    terminal_goal = env.goal_position[0].clone()

    next_observation, _, terminated, truncated, info = env.step(torch.zeros(2, 2))

    assert bool(terminated[0])
    assert not bool(truncated[0])
    assert float(info["metrics"]["final_distance"][0]) < env.success_threshold
    torch.testing.assert_close(
        info["metrics"]["final_position"][0],
        terminal_goal,
        atol=env.success_threshold,
        rtol=0.0,
    )
    # Auto-reset replaced the live state; terminal metrics stay pre-reset.
    assert not torch.allclose(
        env.position[0], info["metrics"]["final_position"][0], atol=1e-6
    )
    assert env.episode_step[0] == 0
    assert next_observation.shape == (2, 14)


def test_point_mass_success_bonus_applies_only_on_success() -> None:
    env = PointMassEnv(num_envs=2, success_bonus=5.0)
    env.reset(seed=5)
    env.velocity.zero_()
    env.goal_position[0] = env.position[0]
    env.goal_position[1] = env.position[1] + 0.5

    _, reward, terminated, _, _ = env.step(torch.zeros(2, 2))

    assert bool(terminated[0])
    assert not bool(terminated[1])
    assert float(reward[0]) > float(reward[1]) + 4.0


def test_point_mass_detach_state_keeps_values() -> None:
    env = PointMassEnv(num_envs=3)
    env.reset(seed=9)
    action = torch.randn(3, 2, requires_grad=True)
    observation, *_ = env.step(action)

    detached = env.detach_state()

    torch.testing.assert_close(observation, detached)
    assert detached.grad_fn is None


@pytest.mark.parametrize(
    ("algorithm_name", "policy_name"),
    [("apg", "actor_only"), ("ppo", "actor_critic")],
)
def test_unified_train_entry_runs_apg_and_ppo(
    tmp_path, monkeypatch, algorithm_name: str, policy_name: str
) -> None:
    monkeypatch.chdir(tmp_path)
    policy = {
        "name": policy_name,
        "actor": {
            "type": "mlp",
            "network_cfg": {"hidden_sizes": [8], "activation": "tanh"},
        },
    }
    algorithm_cfg = {
        "optimizer": {"name": "adam", "learning_rate": 1e-3},
        "batch_size": 8,
        "gamma": 0.99,
        "max_grad_norm": 1.0,
    }
    if policy_name == "actor_critic":
        policy["critic"] = {
            "type": "mlp",
            "network_cfg": {"hidden_sizes": [8], "activation": "tanh"},
        }
        algorithm_cfg.update(
            {
                "n_epochs": 1,
                "gae_lambda": 0.95,
                "clip_coef": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
            }
        )
    config = {
        "trainer": {
            "exp_name": f"smoke_{algorithm_name}",
            "learning_env": {
                "name": "PointMassRL",
                "cfg": {"max_episode_steps": 4},
            },
            "device": "cpu",
            "num_envs": 2,
            "iterations": 1,
            "buffer_size": 4,
            "segment_length": 2,
            "update_horizon": 4,
            "enable_eval": False,
            "use_wandb": False,
        },
        "policy": policy,
        "algorithm": {"name": algorithm_name, "cfg": algorithm_cfg},
    }
    config_path = tmp_path / f"{algorithm_name}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = train_from_config(str(config_path))

    assert summary["global_step"] == 8
    assert summary["latest_checkpoint_path"] is not None


def test_sync_collector_accepts_tensor_point_mass_observations() -> None:
    env = PointMassEnv(num_envs=2, max_episode_steps=4)
    actor = nn.Linear(14, 2)
    critic = nn.Linear(14, 1)
    policy = ActorCritic(14, 2, env.device, actor=actor, critic=critic)
    buffer = RolloutBuffer(
        num_envs=2,
        rollout_len=3,
        obs_dim=14,
        action_dim=2,
        device=env.device,
    )
    collector = SyncCollector(env=env, policy=policy, device=env.device)
    rollout = collector.collect(num_steps=3, rollout=buffer.start_rollout())

    assert rollout["obs"].shape == (2, 4, 14)
    assert torch.isfinite(rollout["reward"][:, :3]).all()
    assert torch.isfinite(rollout["action"][:, :3]).all()


def test_trainer_saves_best_checkpoint_from_eval(tmp_path: Path) -> None:
    env = PointMassEnv(num_envs=2, max_episode_steps=4)
    eval_env = PointMassEnv(num_envs=2, max_episode_steps=4)
    actor = nn.Linear(14, 2)
    critic = nn.Linear(14, 1)
    policy = ActorCritic(14, 2, env.device, actor=actor, critic=critic)
    algorithm = PPO(
        PPOCfg(
            device="cpu",
            optimizer=OptimizerCfg(learning_rate=1e-3),
            n_epochs=1,
            batch_size=8,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
        ),
        policy,
    )
    trainer = Trainer(
        policy=policy,
        env=env,
        algorithm=algorithm,
        buffer_size=4,
        batch_size=8,
        writer=None,
        eval_freq=8,
        save_freq=0,
        checkpoint_dir=str(tmp_path),
        exp_name="point_mass_best",
        use_wandb=False,
        eval_env=eval_env,
        num_eval_episodes=2,
        eval_seed=7,
        best_eval_metric="eval/avg_reward",
        best_eval_mode="max",
    )

    summary = trainer.train(total_timesteps=8)

    assert summary["last_eval_metrics"]
    assert "eval/avg_reward" in summary["last_eval_metrics"]
    assert summary["best_checkpoint_path"] is not None
    assert Path(summary["best_checkpoint_path"]).is_file()
