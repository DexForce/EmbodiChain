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

from types import SimpleNamespace

import gymnasium as gym
import torch

from embodichain.learning.rl.runtime import (
    _GymEnvironmentRuntime,
    _build_gym_environment,
    build_gym_policy_runtime,
)


def _policy_config() -> dict:
    network = {
        "type": "mlp",
        "network_cfg": {"hidden_sizes": [8], "activation": "relu"},
    }
    return {
        "name": "actor_critic",
        "actor": network,
        "critic": network,
    }


class GymEnvironment:
    flattened_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,))
    observation_space = gym.spaces.Dict(
        {"policy": gym.spaces.Box(-1.0, 1.0, shape=(1, 3))}
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1, 2))

    def __init__(self) -> None:
        self.closed = 0
        self.action_manager = SimpleNamespace(total_action_dim=2)

    def reset(self):
        return {"policy": torch.zeros(1, 3)}, {}

    def get_wrapper_attr(self, name):
        return getattr(self, name)

    def close(self) -> None:
        self.closed += 1


def test_gym_environment_applies_runtime_overrides(tmp_path, monkeypatch):
    gym_config = tmp_path / "gym.yaml"
    gym_config.write_text("id: Example\n", encoding="utf-8")
    env_cfg = SimpleNamespace(
        num_envs=8,
        sim_cfg=None,
        profiler=None,
    )
    built = GymEnvironment()
    monkeypatch.setattr(
        "embodichain.learning.rl.runtime.config_to_cfg",
        lambda value, manager_modules: env_cfg,
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.runtime.build_env",
        lambda env_id, base_env_cfg: built,
    )

    runtime = _build_gym_environment(
        {"trainer": {"gym_config": "gym.yaml"}},
        simulation_device=torch.device("cpu"),
        num_envs=1,
        headless=True,
        renderer="hybrid",
        gpu_id=0,
        config_dir=tmp_path,
    )

    assert runtime.env is built
    assert runtime.env_id == "Example"
    assert runtime.env_cfg.num_envs == 1
    assert runtime.env_cfg.sim_cfg.sim_device == torch.device("cpu")
    assert runtime.env_cfg.sim_cfg.headless is True
    assert runtime.env_cfg.sim_cfg.render_cfg.renderer == "hybrid"


def test_gym_runtime_uses_the_same_task_spaces_for_policy_build(monkeypatch):
    env = GymEnvironment()
    task = _GymEnvironmentRuntime(
        env=env,
        env_id="Example",
        env_cfg=SimpleNamespace(),
        gym_config={"id": "Example"},
        gym_config_path=SimpleNamespace(resolve=lambda: None),
    )
    monkeypatch.setattr(
        "embodichain.learning.rl.runtime._build_gym_environment",
        lambda *args, **kwargs: task,
    )

    runtime = build_gym_policy_runtime(
        {"trainer": {"gym_config": "gym.yaml"}, "policy": _policy_config()},
        device=torch.device("cpu"),
        num_envs=1,
        headless=True,
        renderer="hybrid",
        gpu_id=0,
    )

    assert runtime.env is env
    assert runtime.policy.actor[0].in_features == 3
    assert runtime.policy.actor[-1].out_features == 2
