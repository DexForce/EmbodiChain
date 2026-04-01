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

from copy import deepcopy

import torch
from tensordict import TensorDict

from embodichain.agents.rl.buffer import RolloutBuffer
from embodichain.agents.rl.collector import SyncCollector
from embodichain.agents.rl.utils import flatten_dict_observation
from embodichain.lab.gym.envs.tasks.rl import build_env
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, DEFAULT_MANAGER_MODULES
from embodichain.lab.sim import SimulationManagerCfg, SimulationManager
from embodichain.utils.utility import load_json


class _FakePolicy:
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    def train(self) -> None:
        pass

    def forward(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict["obs"]
        tensordict["action"] = obs[:, : self.action_dim] * 0.25
        tensordict["sample_log_prob"] = obs.sum(dim=-1) * 0.1
        tensordict["value"] = obs.mean(dim=-1)
        return tensordict

    def get_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        return self.forward(tensordict)

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        tensordict["value"] = tensordict["obs"].mean(dim=-1)
        return tensordict


class _FakeEnv:
    def __init__(
        self, num_envs: int, obs_dim: int, action_dim: int, device: torch.device
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.rollout_buffer: TensorDict | None = None
        self.current_rollout_step = 0
        self._obs = self._make_obs(step=0)

    def reset(self, **kwargs):
        self.current_rollout_step = 0
        self._obs = self._make_obs(step=0)
        return self._obs, {}

    def set_rollout_buffer(self, rollout_buffer: TensorDict) -> None:
        self.rollout_buffer = rollout_buffer
        self.current_rollout_step = 0

    def step(self, action):
        step_idx = self.current_rollout_step + 1
        next_obs = self._make_obs(step=step_idx)
        reward = action.sum(dim=-1)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.rollout_buffer is not None:
            self.rollout_buffer["reward"][:, self.current_rollout_step] = reward
            self.rollout_buffer["done"][:, self.current_rollout_step] = (
                terminated | truncated
            )
            self.rollout_buffer["terminated"][:, self.current_rollout_step] = terminated
            self.rollout_buffer["truncated"][:, self.current_rollout_step] = truncated
            self.current_rollout_step += 1

        self._obs = next_obs
        return next_obs, reward, terminated, truncated, {}

    def _make_obs(self, step: int) -> TensorDict:
        base = torch.full(
            (self.num_envs, self.obs_dim),
            fill_value=float(step),
            dtype=torch.float32,
            device=self.device,
        )
        return TensorDict(
            {
                "agent": TensorDict(
                    {"state": base},
                    batch_size=[self.num_envs],
                    device=self.device,
                )
            },
            batch_size=[self.num_envs],
            device=self.device,
        )


def test_shared_rollout_collects_policy_and_env_fields():
    device = torch.device("cpu")
    num_envs = 3
    rollout_len = 4
    obs_dim = 5
    action_dim = 2

    env = _FakeEnv(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    policy = _FakePolicy(obs_dim=obs_dim, action_dim=action_dim, device=device)
    collector = SyncCollector(env=env, policy=policy, device=device)
    buffer = RolloutBuffer(
        num_envs=num_envs,
        rollout_len=rollout_len,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    rollout = collector.collect(
        num_steps=rollout_len,
        rollout=buffer.start_rollout(),
    )
    buffer.add(rollout)
    stored = buffer.get(flatten=False)

    assert stored.batch_size == torch.Size([num_envs, rollout_len + 1])
    assert stored["obs"].shape == torch.Size([num_envs, rollout_len + 1, obs_dim])
    assert torch.allclose(stored["obs"][:, 0], torch.zeros(num_envs, obs_dim))
    assert torch.allclose(
        stored["obs"][:, -1],
        torch.full((num_envs, obs_dim), float(rollout_len), dtype=torch.float32),
    )
    assert torch.allclose(
        stored["value"][:, 1], torch.ones(num_envs, dtype=torch.float32)
    )
    assert torch.allclose(
        stored["action"][:, 0],
        torch.zeros(num_envs, action_dim),
    )
    assert torch.allclose(
        stored["sample_log_prob"][:, 1],
        torch.full((num_envs,), 0.5, dtype=torch.float32),
    )
    assert torch.allclose(
        stored["reward"][:, 2],
        torch.full((num_envs,), 1.0, dtype=torch.float32),
    )
    assert torch.allclose(
        stored["value"][:, -1],
        torch.full((num_envs,), 4.0, dtype=torch.float32),
    )
    assert torch.allclose(
        stored["action"][:, -1],
        torch.zeros(num_envs, action_dim, dtype=torch.float32),
    )


def test_embodied_env_writes_next_fields_into_external_rollout():
    gym_config = load_json("configs/agents/rl/basic/cart_pole/gym_config.json")
    env_cfg = config_to_cfg(gym_config, manager_modules=DEFAULT_MANAGER_MODULES)
    env_cfg = deepcopy(env_cfg)
    env_cfg.num_envs = 2
    env_cfg.sim_cfg = SimulationManagerCfg(
        headless=True,
        sim_device=torch.device("cpu"),
        enable_rt=False,
        gpu_id=0,
    )

    env = build_env(gym_config["id"], base_env_cfg=env_cfg)
    try:
        obs, _ = env.reset()
        obs_dim = flatten_dict_observation(obs).shape[-1]
        action_dim = env.action_space.shape[-1]
        buffer = RolloutBuffer(
            num_envs=env.num_envs,
            rollout_len=4,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=torch.device("cpu"),
        )
        rollout = buffer.start_rollout()
        env.set_rollout_buffer(rollout)

        action = torch.zeros(
            env.num_envs,
            action_dim,
            dtype=torch.float32,
            device=env.device,
        )
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = (terminated | truncated).cpu()

        assert env.current_rollout_step == 1
        assert torch.allclose(rollout["reward"][:, 0].cpu(), reward.cpu())
        assert torch.equal(rollout["done"][:, 0].cpu(), done)
        assert torch.equal(rollout["terminated"][:, 0].cpu(), terminated.cpu())
        assert torch.equal(rollout["truncated"][:, 0].cpu(), truncated.cpu())
    finally:
        env.close()
        if SimulationManager.is_instantiated():
            SimulationManager.get_instance().destroy()


class _FakePolicyRawObs:
    """Policy that reads nested env observations (collector passes raw_obs slices)."""

    use_raw_obs = True

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.obs_dim = 0
        self.action_dim = action_dim
        self.device = device

    def train(self) -> None:
        pass

    def get_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        obs = tensordict["obs"]
        flat = obs["agent"]["state"]
        tensordict["action"] = flat[:, : self.action_dim] * 0.25
        tensordict["sample_log_prob"] = flat.sum(dim=-1) * 0.1
        tensordict["value"] = flat.mean(dim=-1)
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict["obs"]
        flat = obs["agent"]["state"]
        tensordict["value"] = flat.mean(dim=-1)
        return tensordict


class _FakePolicyActionChunk:
    """Policy that emits a fixed-length action chunk (sequential env steps per chunk)."""

    use_action_chunk = True
    action_chunk_size = 2
    execute_full_chunk = False

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    def train(self) -> None:
        pass

    def get_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        obs = tensordict["obs"]
        row0 = obs[:, : self.action_dim] * 0.1
        row1 = obs[:, : self.action_dim] * 0.2
        chunk = torch.stack([row0, row1], dim=1)
        tensordict["action_chunk"] = chunk
        tensordict["action"] = chunk[:, 0]
        tensordict["sample_log_prob"] = torch.zeros(
            obs.shape[0], device=obs.device, dtype=torch.float32
        )
        tensordict["value"] = torch.zeros(
            obs.shape[0], device=obs.device, dtype=torch.float32
        )
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        tensordict["value"] = tensordict["obs"].mean(dim=-1)
        return tensordict


class _FakePolicyExecuteFullChunk:
    """Policy that runs an entire chunk inside one logical rollout step."""

    use_action_chunk = True
    action_chunk_size = 3
    execute_full_chunk = True

    def __init__(self, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    def train(self) -> None:
        pass

    def get_action(
        self, tensordict: TensorDict, deterministic: bool = False
    ) -> TensorDict:
        obs = tensordict["obs"]
        n = obs.shape[0]
        t = self.action_chunk_size
        base = obs[:, : self.action_dim] * 0.1
        chunk = base.unsqueeze(1).expand(n, t, -1).clone()
        tensordict["action_chunk"] = chunk
        tensordict["action"] = chunk[:, 0]
        tensordict["sample_log_prob"] = torch.zeros(
            n, device=obs.device, dtype=torch.float32
        )
        tensordict["value"] = torch.zeros(n, device=obs.device, dtype=torch.float32)
        return tensordict

    def get_value(self, tensordict: TensorDict) -> TensorDict:
        tensordict["value"] = tensordict["obs"].mean(dim=-1)
        return tensordict


def test_collector_populates_raw_obs_buffer():
    device = torch.device("cpu")
    num_envs = 2
    rollout_len = 3
    obs_dim = 5
    action_dim = 2

    env = _FakeEnv(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    policy = _FakePolicyRawObs(obs_dim=obs_dim, action_dim=action_dim, device=device)
    collector = SyncCollector(env=env, policy=policy, device=device)
    buffer = RolloutBuffer(
        num_envs=num_envs,
        rollout_len=rollout_len,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        use_raw_obs=True,
        store_flat_obs=False,
    )

    rollout = collector.collect(
        num_steps=rollout_len,
        rollout=buffer.start_rollout(),
    )
    buffer.add(rollout)
    stored = buffer.get(flatten=False)

    assert "obs" not in stored.keys()
    assert hasattr(stored, "raw_obs")
    assert len(stored.raw_obs) == rollout_len + 1
    for t in range(rollout_len + 1):
        assert stored.raw_obs[t] is not None
        assert stored.raw_obs[t].batch_size == torch.Size([num_envs])
        assert torch.allclose(
            stored.raw_obs[t]["agent"]["state"],
            torch.full((num_envs, obs_dim), float(t), dtype=torch.float32),
        )
    assert torch.allclose(
        stored["value"][:, -1],
        torch.full((num_envs,), float(rollout_len), dtype=torch.float32),
    )


def test_collector_chunk_step_alternates_for_sequential_action_chunk():
    device = torch.device("cpu")
    num_envs = 2
    rollout_len = 4
    obs_dim = 5
    action_dim = 2

    env = _FakeEnv(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    policy = _FakePolicyActionChunk(
        obs_dim=obs_dim, action_dim=action_dim, device=device
    )
    collector = SyncCollector(env=env, policy=policy, device=device)
    buffer = RolloutBuffer(
        num_envs=num_envs,
        rollout_len=rollout_len,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        action_chunk_size=policy.action_chunk_size,
    )

    rollout = collector.collect(
        num_steps=rollout_len,
        rollout=buffer.start_rollout(),
    )

    assert hasattr(rollout, "chunk_step")
    expected = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=device).expand(
        num_envs, -1
    )
    assert torch.equal(rollout.chunk_step, expected)
    assert torch.all(rollout["step_repeat"][:, :-1] == 1.0)
    assert torch.allclose(rollout["action_chunk"][:, 1], rollout["action_chunk"][:, 0])
    assert torch.allclose(rollout["action_chunk"][:, 3], rollout["action_chunk"][:, 2])


def test_collector_execute_full_chunk_sets_step_repeat_and_action_chunk():
    device = torch.device("cpu")
    num_envs = 2
    rollout_len = 2
    obs_dim = 5
    action_dim = 2
    chunk_t = 3

    env = _FakeEnv(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    policy = _FakePolicyExecuteFullChunk(
        obs_dim=obs_dim, action_dim=action_dim, device=device
    )
    assert policy.action_chunk_size == chunk_t
    collector = SyncCollector(env=env, policy=policy, device=device)
    buffer = RolloutBuffer(
        num_envs=num_envs,
        rollout_len=rollout_len,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        action_chunk_size=chunk_t,
    )

    rollout = collector.collect(
        num_steps=rollout_len,
        rollout=buffer.start_rollout(),
    )

    assert "action_chunk" in rollout.keys()
    assert rollout["action_chunk"].shape == (
        num_envs,
        rollout_len + 1,
        chunk_t,
        action_dim,
    )
    assert torch.all(rollout.chunk_step[:, :rollout_len] == 0)
    assert torch.all(rollout["step_repeat"][:, :rollout_len] == float(chunk_t))
    assert torch.all(rollout["step_repeat"][:, -1] == 0.0)
