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

import gymnasium as gym
import torch

from embodichain.learning.rl.evaluation import evaluate_episodes


class _DeterministicPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_action(self, tensordict, deterministic: bool = False):
        assert deterministic
        tensordict["action"] = torch.zeros(
            tensordict.batch_size[0], 1, device=tensordict.device
        )
        return tensordict


class _AsyncAutoResetEnv:
    num_envs = 3
    device = torch.device("cpu")
    single_observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
    single_action_space = gym.spaces.Box(-1.0, 1.0, (1,))

    def __init__(self) -> None:
        self.steps = torch.zeros(3, dtype=torch.long)
        self.horizons = torch.tensor([1, 2, 3])
        self.last_seed = None

    def reset(self, *, seed=None, options=None):
        self.last_seed = seed
        self.steps.zero_()
        return self.steps[:, None].float(), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.horizons
        terminal_steps = self.steps.clone()
        info = {
            "success": done.clone(),
            "metrics": {
                "terminal_step": terminal_steps.float(),
                # Non-scalar per-env metric must not be flattened into eval logs.
                "final_position": torch.stack(
                    (terminal_steps.float(), terminal_steps.float() + 10.0),
                    dim=-1,
                ),
            },
        }
        self.steps[done] = 0
        return (
            self.steps[:, None].float(),
            torch.ones(3),
            done,
            torch.zeros(3, dtype=torch.bool),
            info,
        )


def test_evaluate_episodes_counts_actual_completions_and_restores_mode() -> None:
    policy = _DeterministicPolicy()
    policy.train()
    env = _AsyncAutoResetEnv()

    result = evaluate_episodes(
        policy=policy,
        env=env,
        num_episodes=4,
        device="cpu",
        seed=123,
    )

    assert policy.training
    assert env.last_seed == 123
    assert result["eval/avg_reward"] == 1.25
    assert result["eval/avg_length"] == 1.25
    assert result["eval/success_rate"] == 1.0
    assert result["eval/metrics/terminal_step"] == 1.25
    assert "eval/metrics/final_position" not in result
