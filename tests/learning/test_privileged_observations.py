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

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from embodichain.learning.rl.algo import PPO, PPOCfg
from embodichain.learning.rl.buffer import RolloutBuffer, transition_view
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.models import ActorCritic, build_policy
from embodichain.learning.rl.utils import (
    OptimizerCfg,
    flatten_observation_groups,
)

NUM_ENVS = 2
ACTOR_OBS_DIM = 3
CRITIC_OBS_DIM = 5
ACTION_DIM = 2
ROLLOUT_LEN = 3


class _PrivilegedObservationEnv:
    """Small vector environment with different actor and critic inputs."""

    def __init__(self) -> None:
        self.num_envs = NUM_ENVS
        self.device = torch.device("cpu")
        self.action_manager = None
        self.step_count = 0

    def _observation(self) -> dict[str, torch.Tensor]:
        return {
            "policy": torch.full(
                (NUM_ENVS, ACTOR_OBS_DIM),
                float(self.step_count),
            ),
            "critic": torch.full(
                (NUM_ENVS, CRITIC_OBS_DIM),
                float(10 + self.step_count),
            ),
        }

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        self.step_count = 0
        return self._observation(), {}

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        self.step_count += 1
        reward = action.sum(dim=-1)
        terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)
        truncated = torch.zeros(NUM_ENVS, dtype=torch.bool)
        return self._observation(), reward, terminated, truncated, {}


def _build_policy() -> ActorCritic:
    actor = nn.Linear(ACTOR_OBS_DIM, ACTION_DIM, bias=False)
    critic = nn.Linear(CRITIC_OBS_DIM, 1, bias=False)
    with torch.no_grad():
        actor.weight.zero_()
        critic.weight.fill_(1.0)
    return ActorCritic(
        obs_dim=ACTOR_OBS_DIM,
        action_dim=ACTION_DIM,
        device=torch.device("cpu"),
        actor=actor,
        critic=critic,
        critic_obs_dim=CRITIC_OBS_DIM,
        actor_obs_groups=("policy",),
        critic_obs_groups=("critic",),
    )


def test_flatten_observation_groups_preserves_configured_order() -> None:
    observation = TensorDict(
        {
            "first": torch.ones(NUM_ENVS, 2),
            "second": torch.full((NUM_ENVS, 1), 2.0),
        },
        batch_size=[NUM_ENVS],
    )

    flattened = flatten_observation_groups(observation, ("second", "first"))

    assert flattened.shape == (NUM_ENVS, 3)
    torch.testing.assert_close(flattened[:, 0], torch.full((NUM_ENVS,), 2.0))
    torch.testing.assert_close(flattened[:, 1:], torch.ones(NUM_ENVS, 2))


def test_actor_critic_routes_privileged_observation_to_critic() -> None:
    policy = _build_policy()
    policy_input = TensorDict(
        {
            "obs": torch.zeros(NUM_ENVS, ACTOR_OBS_DIM),
            "critic_obs": torch.full((NUM_ENVS, CRITIC_OBS_DIM), 2.0),
        },
        batch_size=[NUM_ENVS],
    )

    output = policy.get_action(policy_input, deterministic=True)

    torch.testing.assert_close(output["action"], torch.zeros(NUM_ENVS, ACTION_DIM))
    torch.testing.assert_close(
        output["value"],
        torch.full((NUM_ENVS,), float(CRITIC_OBS_DIM * 2)),
    )


def test_actor_critic_reports_missing_privileged_observation() -> None:
    policy = _build_policy()
    policy_input = TensorDict(
        {"obs": torch.zeros(NUM_ENVS, ACTOR_OBS_DIM)},
        batch_size=[NUM_ENVS],
    )

    with pytest.raises(KeyError, match="critic_obs"):
        policy.get_action(policy_input)


def test_collector_and_ppo_keep_privileged_observations() -> None:
    env = _PrivilegedObservationEnv()
    policy = _build_policy()
    buffer = RolloutBuffer(
        num_envs=NUM_ENVS,
        rollout_len=ROLLOUT_LEN,
        obs_dim=ACTOR_OBS_DIM,
        action_dim=ACTION_DIM,
        device=torch.device("cpu"),
        critic_obs_dim=CRITIC_OBS_DIM,
    )
    collector = SyncCollector(env=env, policy=policy, device=torch.device("cpu"))
    rollout = collector.collect(
        num_steps=ROLLOUT_LEN,
        rollout=buffer.start_rollout(),
    )

    assert rollout["obs"].shape == (
        NUM_ENVS,
        ROLLOUT_LEN + 1,
        ACTOR_OBS_DIM,
    )
    assert rollout["critic_obs"].shape == (
        NUM_ENVS,
        ROLLOUT_LEN + 1,
        CRITIC_OBS_DIM,
    )
    torch.testing.assert_close(
        rollout["value"][:, -1],
        torch.full(
            (NUM_ENVS,),
            float((10 + ROLLOUT_LEN) * CRITIC_OBS_DIM),
        ),
    )
    assert "critic_obs" in transition_view(rollout).keys()

    algorithm = PPO(
        PPOCfg(
            device="cpu",
            optimizer=OptimizerCfg(learning_rate=1e-3),
            n_epochs=1,
            batch_size=NUM_ENVS * ROLLOUT_LEN,
            gamma=0.99,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
        ),
        policy,
    )
    losses = algorithm.update(rollout)

    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_build_policy_configures_distinct_observation_sets() -> None:
    policy = build_policy(
        {
            "name": "actor_critic",
            "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        },
        ACTOR_OBS_DIM,
        ACTION_DIM,
        torch.device("cpu"),
        actor=nn.Linear(ACTOR_OBS_DIM, ACTION_DIM),
        critic=nn.Linear(CRITIC_OBS_DIM, 1),
        critic_obs_space=CRITIC_OBS_DIM,
    )

    assert policy.actor_obs_dim == ACTOR_OBS_DIM
    assert policy.critic_obs_dim == CRITIC_OBS_DIM
    assert policy.actor_obs_groups == ("policy",)
    assert policy.critic_obs_groups == ("critic",)
