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

import pytest
import torch
from tensordict import TensorDict

from embodichain.learning.rl.buffer import RolloutBuffer
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.policies import ActorCritic
from embodichain.learning.rl.utils import (
    flatten_dict_observation,
    flatten_observation_groups,
)
from embodichain.learning.rl.utils.helper import _pack_observation_groups


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
        self,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        nested_observation: bool = False,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.nested_observation = nested_observation
        # The flat action manager is intentionally present without the removed
        # legacy conversion method; SyncCollector must pass raw actions to step.
        self.action_manager = object()
        self.rollout_buffer: TensorDict | None = None
        self.current_rollout_step = 0
        self.steps = 0
        self._obs = self._make_obs(step=0)

    def reset(self, **kwargs):
        self.current_rollout_step = 0
        self.steps = 0
        self._obs = self._make_obs(step=0)
        return self._obs, {}

    def set_rollout_buffer(self, rollout_buffer: TensorDict) -> None:
        self.rollout_buffer = rollout_buffer
        self.current_rollout_step = 0

    def step(self, action):
        self.steps += 1
        step_idx = self.steps
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
                    (
                        {"z": base[:, -1], "a": {"matrix": base[:, :-1].unsqueeze(1)}}
                        if self.nested_observation
                        else {"state": base}
                    ),
                    batch_size=[self.num_envs],
                    device=self.device,
                )
            },
            batch_size=[self.num_envs],
            device=self.device,
        )


class _HistoryEnv(_FakeEnv):
    """Exercise the native history scope without constructing a simulator."""

    def __init__(self, *args, **kwargs):
        from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

        super().__init__(*args, **kwargs)
        self.rollout_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self.set_rollout_buffer = EmbodiedEnv.set_rollout_buffer.__get__(self)
        self._infer_rollout_buffer_mode = (
            EmbodiedEnv._infer_rollout_buffer_mode.__get__(self)
        )

    def _action_history_scope(self):
        from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

        return EmbodiedEnv._action_history_scope(self)


@pytest.mark.no_sim
@pytest.mark.parametrize(
    "reuse_observation,nested_observation,history_mode",
    [
        (False, False, "legacy"),
        (True, False, "legacy"),
        (False, True, "legacy"),
        (True, True, "legacy"),
        (True, True, "history"),
        (True, True, "bound_history"),
    ],
)
def test_shared_rollout_collects_policy_and_env_fields(
    reuse_observation, nested_observation, history_mode
):
    device = torch.device("cpu")
    num_envs = 3
    rollout_len = 4
    obs_dim = 5
    action_dim = 2

    env_type = _FakeEnv if history_mode == "legacy" else _HistoryEnv
    env = env_type(
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        nested_observation=nested_observation,
    )
    if reuse_observation:

        def make_obs(step):
            for leaf in env._obs.values(include_nested=True, leaves_only=True):
                leaf.fill_(step)
            return env._obs

        env._make_obs = make_obs
    policy = _FakePolicy(obs_dim=obs_dim, action_dim=action_dim, device=device)
    collector = SyncCollector(env=env, policy=policy, device=device)
    buffer = RolloutBuffer(
        num_envs=num_envs,
        rollout_len=rollout_len,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    rollout = buffer.start_rollout()
    if history_mode == "bound_history":
        env.set_rollout_buffer(rollout)

    rollout = collector.collect(
        num_steps=rollout_len,
        rollout=rollout,
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
    if history_mode == "history":
        assert env.rollout_buffer is None
    else:
        assert env.rollout_buffer is rollout
        assert env.current_rollout_step == rollout_len


@pytest.mark.no_sim
@pytest.mark.parametrize("override_style", ["subclass", "instance"])
def test_collection_preserves_custom_rollout_binding(override_style):
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    class CustomBindingEnv(_FakeEnv):
        def set_rollout_buffer(self, rollout):
            bindings.append(rollout)
            super().set_rollout_buffer(rollout)

    device = torch.device("cpu")
    bindings = []
    env_type = CustomBindingEnv if override_style == "subclass" else _FakeEnv
    env = env_type(2, 1, 1, device)
    env._action_history_scope = EmbodiedEnv._action_history_scope.__get__(env)
    if override_style == "instance":

        def set_rollout_buffer(rollout):
            bindings.append(rollout)
            _FakeEnv.set_rollout_buffer(env, rollout)

        env.set_rollout_buffer = set_rollout_buffer
    rollout = RolloutBuffer(2, 2, 1, 1, device).start_rollout()
    SyncCollector(env, _FakePolicy(1, 1, device), device).collect(2, rollout)

    assert len(bindings) == 1 and bindings[0] is rollout
    assert env.current_rollout_step == 2
    torch.testing.assert_close(
        rollout["reward"][:, :-1], torch.tensor([[0.0, 0.25]] * 2)
    )


@pytest.mark.no_sim
def test_collection_history_preserves_rollout_and_episode_reward_boundaries():
    from embodichain.lab.gym.envs.managers.rewards import action_smoothness_penalty

    class RewardEnv(_HistoryEnv):
        def step(self, action):
            obs, _, terminated, truncated, info = super().step(action)
            reward = action_smoothness_penalty(self, obs, action, info)
            terminated[0] = self.steps == 1
            truncated[1] = self.steps == 2
            return obs, reward, terminated, truncated, info

    device = torch.device("cpu")
    env = RewardEnv(3, 1, 1, device)
    collector = SyncCollector(env, _FakePolicy(1, 1, device), device)
    expected_rewards = (
        [[0.0, 0.0, -0.25], [0.0, -0.25, 0.0], [0.0, -0.25, -0.25]],
        [[0.0, -0.25, -0.25]] * 3,
    )
    for expected in expected_rewards:
        rollout = collector.collect(
            3, RolloutBuffer(3, 3, 1, 1, device).start_rollout()
        )
        # Actions rise by 0.25 per step; preceding termination/truncation masks
        # clear only that row. A new rollout clears history without resetting obs.
        torch.testing.assert_close(rollout["reward"][:, :-1], torch.tensor(expected))
        assert env.rollout_buffer is None
        torch.testing.assert_close(
            action_smoothness_penalty(env, {}, torch.full((3, 1), 5.0), {}),
            torch.zeros(3),
        )


@pytest.mark.no_sim
@pytest.mark.parametrize("failure_stage", ["reset", "step", "callback", "bootstrap"])
def test_collection_failure_releases_borrowed_history(failure_stage, monkeypatch):
    from embodichain.lab.gym.envs.managers.rewards import action_smoothness_penalty

    device = torch.device("cpu")
    env = _HistoryEnv(2, 1, 1, device)
    policy = _FakePolicy(1, 1, device)
    collector = SyncCollector(env, policy, device, reset_every_rollout=True)

    def fail(*args, **kwargs):
        raise RuntimeError(f"{failure_stage} failed")

    callback = None
    with monkeypatch.context() as patch:
        if failure_stage == "reset":
            patch.setattr(env, "reset", fail)
        elif failure_stage == "step":
            step = env.step

            def fail_after_one_step(action):
                if env.steps:
                    fail()
                return step(action)

            patch.setattr(env, "step", fail_after_one_step)
        elif failure_stage == "bootstrap":
            patch.setattr(policy, "get_value", fail)
        else:
            callback = fail
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            collector.collect(
                2,
                RolloutBuffer(2, 2, 1, 1, device).start_rollout(),
                on_step_callback=callback,
            )

    assert env.rollout_buffer is None
    assert env._previous_transition is None
    torch.testing.assert_close(
        action_smoothness_penalty(env, {}, torch.full((2, 1), 5.0), {}),
        torch.zeros(2),
    )
    # A subsequent collection must be allowed and start with fresh history.
    recovered = collector.collect(2, RolloutBuffer(2, 2, 1, 1, device).start_rollout())
    torch.testing.assert_close(recovered["obs"][:, 0], torch.zeros(2, 1))


@pytest.mark.no_sim
@pytest.mark.parametrize("groups", [None, ("second", "first", "first.scalar")])
@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_observation_packing_preserves_nested_order_and_promotion(groups, mixed_dtype):
    # The integer rounds inside its float32 group before the float64 outer cat.
    scalar = torch.tensor([2**24 + 1, 2**24 + 3], dtype=torch.int64)
    matrix = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2).transpose(1, 2)
    second = torch.tensor([[11.0], [13.0]], dtype=torch.float64)
    if not mixed_dtype:
        scalar, second = scalar.float(), second.float()
    observation = TensorDict(
        {"second": second, "first": {"scalar": scalar, "matrix": matrix}},
        batch_size=[2],
    )
    dtype = torch.float64 if mixed_dtype else torch.float32
    rounded = [float(2**24), float(2**24 + 4)]
    if groups is None:
        # No selected subgroups: all leaves participate in one promotion.
        scalar_values = [2**24 + 1, 2**24 + 3] if mixed_dtype else rounded
        expected = torch.tensor(
            [[0, 2, 1, 3, scalar_values[0], 11], [4, 6, 5, 7, scalar_values[1], 13]],
            dtype=dtype,
        )
    else:
        repeated = [2**24 + 1, 2**24 + 3] if mixed_dtype else rounded
        expected = torch.tensor(
            [
                [11, 0, 2, 1, 3, rounded[0], repeated[0]],
                [13, 4, 6, 5, 7, rounded[1], repeated[1]],
            ],
            dtype=dtype,
        )
    storage = torch.full((2, 3, expected.shape[1]), -1.0, dtype=dtype)
    target = storage[:, 1]
    _pack_observation_groups(observation, groups, out=target)

    torch.testing.assert_close(target, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        target, flatten_observation_groups(observation, groups), rtol=0, atol=0
    )
    assert torch.all(storage[:, 0] == -1)
    assert torch.all(storage[:, 2] == -1)
    matrix.fill_(-2)
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


@pytest.mark.no_sim
def test_observation_packing_snapshots_overlapping_source_views():
    target = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    observation = TensorDict(
        {"first": target[:, :2], "second": target[:, 2:]}, batch_size=[3]
    )
    expected = torch.tensor([[2, 3, 0, 1], [6, 7, 4, 5], [10, 11, 8, 9]]).float()
    _pack_observation_groups(observation, ("second", "first"), out=target)
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


@pytest.mark.no_sim
def test_collected_observations_remain_raw_with_custom_policy_transforms():
    device = torch.device("cpu")
    env = _FakeEnv(3, 5, 2, device, nested_observation=True)
    actor, critic = torch.nn.Linear(5, 2, bias=False), torch.nn.Linear(4, 1, bias=False)
    actor_transform = torch.nn.Linear(5, 5, bias=False)
    critic_transform = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        actor.weight.fill_(1)
        critic.weight.fill_(1)
        actor_transform.weight.copy_(2 * torch.eye(5))
        critic_transform.weight.copy_(3 * torch.eye(4))
    policy = ActorCritic(
        5,
        2,
        device,
        actor,
        critic,
        actor_obs_groups=("agent.z", "agent.a"),
        critic_obs_dim=4,
        critic_obs_groups=("agent.a",),
        actor_normalizer=actor_transform,
        critic_normalizer=critic_transform,
    )
    buffer = RolloutBuffer(
        3, 2, 5, 2, device, critic_obs_dim=4, distribution_param_dim=2
    )
    rollout = SyncCollector(env, policy, device).collect(
        2, rollout=buffer.start_rollout()
    )
    steps = torch.arange(3, dtype=torch.float32)
    torch.testing.assert_close(rollout["obs"], steps[None, :, None].expand(3, 3, 5))
    torch.testing.assert_close(
        rollout["critic_obs"], steps[None, :, None].expand(3, 3, 4)
    )
    torch.testing.assert_close(rollout["value"], (12 * steps).expand(3, 3))
    torch.testing.assert_close(
        rollout["action_mean"][:, :-1],
        (10 * steps[:-1])[None, :, None].expand(3, 2, 2),
    )


@pytest.mark.no_sim
@pytest.mark.parametrize(
    ("groups", "error", "message"),
    [
        ((), ValueError, "Observation groups cannot be empty"),
        (("absent",), KeyError, "Observation group 'absent' was not found"),
        (("empty",), ValueError, "No tensors found"),
    ],
)
def test_observation_packing_preserves_group_validation(groups, error, message):
    observation = TensorDict(
        {"state": torch.ones(2, 3), "empty": TensorDict({}, batch_size=[2])},
        batch_size=[2],
    )
    target = torch.full((2, 3), -1.0)
    for flatten in (
        lambda: flatten_observation_groups(observation, groups),
        lambda: _pack_observation_groups(observation, groups, out=target),
    ):
        with pytest.raises(error, match=message):
            flatten()
    assert torch.all(target == -1)


@pytest.mark.requires_sim
@pytest.mark.requires_tasks
def test_embodied_env_writes_next_fields_into_external_rollout():
    from embodichain.lab.sim.cfg import RenderCfg
    from embodichain.lab.gym.utils.registration import build_env
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
    from embodichain.lab.sim import SimulationManagerCfg, SimulationManager
    from embodichain.utils.utility import load_json

    gym_config = load_json(
        "embodichain/extensions/tasks/configs/tasks/classic_control/cart_pole/env.json"
    )
    env_cfg = config_to_cfg(gym_config, manager_modules=get_manager_modules())
    env_cfg = deepcopy(env_cfg)
    env_cfg.num_envs = 2
    env_cfg.sim_cfg = SimulationManagerCfg(
        headless=True,
        device=torch.device("cpu"),
        render_cfg=RenderCfg(renderer="hybrid"),
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
