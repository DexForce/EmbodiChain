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
import math

import pytest
import torch
from tensordict import TensorDict

from embodichain.learning.rl.algo import PPO, PPOCfg
from embodichain.learning.rl.buffer import RolloutBuffer, transition_view
from embodichain.learning.rl.collector import SyncCollector
from embodichain.learning.rl.policies import (
    ObservationHistory,
    ProprioceptiveActorCriticCfg,
    build_policy,
)
from embodichain.learning.rl.utils import OptimizerCfg


def test_config_retains_legacy_positional_arguments() -> None:
    cfg = ProprioceptiveActorCriticCfg(33, 30)
    assert (cfg.frame_dim, cfg.history_length, cfg.architecture_version) == (
        33,
        30,
        "legacy_v1",
    )


def _policy(
    contact_points: int = 2,
    history_length: int = 30,
    use_history: bool = True,
    normalize: bool = False,
    architecture_version: str = "legacy_v1",
):
    return build_policy(
        {
            "name": "proprioceptive_actor_critic",
            "obs_groups": {"actor": ["proprioception"], "critic": ["privileged"]},
            "actor_obs_normalization": normalize,
            "critic_obs_normalization": normalize,
            "cfg": {
                "architecture_version": architecture_version,
                "frame_dim": 3,
                "history_length": history_length,
                "use_history": use_history,
                "contact_points": contact_points,
                "contact_class_weights": [1.0, 3.0],
                "actor_hidden_sizes": [8],
                "critic_hidden_sizes": [8],
                "decoder_hidden_sizes": [8],
            },
        },
        3 * (history_length + 1),
        2,
        torch.device("cpu"),
        critic_obs_space=4,
    )


def _batch(policy, batch_size: int = 2):
    return TensorDict(
        {
            "obs": torch.randn(batch_size, policy.obs_dim),
            "critic_obs": torch.randn(batch_size, 4),
        },
        batch_size=[batch_size],
    )


def test_history_selective_reset_and_snapshot_ownership() -> None:
    history = ObservationHistory(2, 3, history_length=2, action_slice=(2, 3))
    history.reset(torch.tensor([[1.0, 2.0, 9.0], [3.0, 4.0, 8.0]]))
    before = history.observation
    history.append(torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]))
    after = history.append(
        torch.tensor([[11.0, 12.0, 13.0], [14.0, 15.0, 16.0]]),
        reset_mask=torch.tensor([True, False]),
    ).reshape(2, 3, 3)
    torch.testing.assert_close(before[0], torch.tensor([1.0, 2.0, 0.0] * 3))
    torch.testing.assert_close(after[0], torch.tensor([[11.0, 12.0, 0.0]] * 3))
    torch.testing.assert_close(
        after[1], torch.tensor([[3.0, 4.0, 0.0], [8.0, 9.0, 10.0], [14.0, 15.0, 16.0]])
    )


@pytest.mark.parametrize("architecture_version", ["legacy_v1", "official_v1"])
def test_actor_does_not_read_privileged_values_and_export_matches(
    architecture_version: str,
) -> None:
    policy = _policy(normalize=True, architecture_version=architecture_version)
    td = _batch(policy)
    policy.update_normalization(td)
    original = policy.get_action(td.clone(), deterministic=True)
    changed = td.clone()
    changed["critic_obs"] = torch.full((2, 4), 1e4)
    alternate = policy.get_action(changed, deterministic=True)
    torch.testing.assert_close(original["action"], alternate["action"])
    assert not torch.equal(original["value"], alternate["value"])
    exported = policy.export_actor()
    torch.testing.assert_close(exported(td["obs"]), original["action"])
    assert not any(
        "critic" in name or "decoder" in name for name in exported.state_dict()
    )
    restored = _policy(normalize=True, architecture_version=architecture_version)
    restored.load_state_dict(deepcopy(policy.state_dict()))
    assert int(restored.actor_obs_normalizer.count) == 2
    torch.testing.assert_close(restored.export_actor()(td["obs"]), original["action"])


@pytest.mark.parametrize("architecture_version", ["legacy_v1", "official_v1"])
def test_weighted_contact_loss_matches_independent_logsumexp_and_backpropagates(
    architecture_version: str,
) -> None:
    policy = _policy(architecture_version=architecture_version)
    batch = _batch(policy)
    policy.get_action(batch)
    labels = torch.tensor([[0, 1], [1, 1]])
    batch["contact_labels"] = labels
    evaluated = policy.evaluate_actions(batch)
    logits = evaluated["contact_logits"]
    selected = logits.gather(-1, labels[..., None]).squeeze(-1)
    nll = logits.logsumexp(-1) - selected
    weights = torch.tensor([[1.0, 3.0], [3.0, 3.0]])
    expected = (nll * weights).sum() / weights.sum()
    loss = policy.auxiliary_loss(batch, evaluated)
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.actor.encoder.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.contact_decoder.parameters()
    )
    assert all(p.grad is None for p in policy.critic.parameters())


def test_official_history_includes_current_and_resets_selected_rows() -> None:
    history = ObservationHistory(2, 1, history_length=2, layout="current_history")
    history.reset(torch.tensor([[1.0], [10.0]]))
    history.append(torch.tensor([[2.0], [20.0]]))
    before = history.observation
    history.append(torch.tensor([[3.0], [30.0]]), torch.tensor([True, False]))
    # Author export duplicates the current frame at the history's newest slot.
    torch.testing.assert_close(
        before, torch.tensor([[2.0, 1.0, 2.0], [20.0, 10.0, 20.0]])
    )
    torch.testing.assert_close(
        history.observation, torch.tensor([[3.0, 3.0, 3.0], [30.0, 20.0, 30.0]])
    )


def test_official_normalizer_uses_variance_epsilon_clipping_and_finite_input() -> None:
    policy = _policy(normalize=True, architecture_version="official_v1")
    normalizer = policy.actor_obs_normalizer
    with torch.no_grad():
        normalizer.variance.zero_()
    inputs = torch.zeros(1, policy.obs_dim)
    inputs[0, :5] = torch.tensor([1e-4, 1e3, -1e3, float("nan"), float("inf")])
    torch.testing.assert_close(
        normalizer(inputs)[0, :5], torch.tensor([1.0, 10.0, -10.0, 0.0, 0.0])
    )


def test_official_network_matches_independent_browser_reference() -> None:
    """Fixed weights/inputs and expected outputs from author's PolicyNet JS.

    The controlled tensors use the published peg.json ordering; no author
    checkpoint weights or network dependency are needed during the test.
    Reference: younghyopark.me/blind-proprioceptive-assembly/static/js/policy_net.js
    (retrieved 2026-10-26). The browser reference uses scalar double accumulation
    with FP32 layer outputs, whereas this implementation uses native FP32 GEMMs.
    """
    policy = build_policy(
        {
            "name": "proprioceptive_actor_critic",
            "obs_groups": {"actor": ["proprioception"], "critic": ["privileged"]},
            "actor_obs_normalization": True,
            "cfg": {
                "architecture_version": "official_v1",
                "contact_points": 64,
                "contact_classes": 3,
            },
        },
        33 * 31,
        7,
        torch.device("cpu"),
        critic_obs_space=235,
    )
    offset = 0
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name.startswith(("actor.", "contact_decoder.")):
                values = (torch.arange(parameter.numel()) + offset) % 29 - 14
                parameter.copy_(values.reshape_as(parameter) * 0.005)
                offset += parameter.numel()
        feature = torch.arange(policy.obs_dim)
        policy.actor_obs_normalizer.mean.copy_((feature % 17 - 8) * 0.01)
        policy.actor_obs_normalizer.variance.copy_(0.5 + (feature % 13) * 0.1)
    observation = ((torch.arange(policy.obs_dim) % 37 - 18) * 0.05)[None]
    batch = TensorDict(
        {"obs": observation, "critic_obs": torch.zeros(1, 235)}, batch_size=[1]
    )
    policy.get_action(batch, deterministic=True)
    evaluated = policy.evaluate_actions(batch)
    expected_action = torch.tensor(
        [
            [
                0.67808330,
                -0.68325704,
                0.13631289,
                -0.14759670,
                -0.10906429,
                0.10355318,
                -0.29522273,
            ]
        ]
    )
    expected_logits = torch.tensor(
        [
            -0.03768333,
            0.11337375,
            0.02615663,
            -0.03177710,
            0.14484464,
            -0.10274863,
            0.12700826,
            -0.05767719,
            0.00503581,
            0.08803601,
            -0.04914308,
            0.24929811,
        ]
    )
    torch.testing.assert_close(batch["action"], expected_action, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        evaluated["contact_logits"].flatten()[:12],
        expected_logits,
        atol=1e-5,
        rtol=1e-4,
    )


class _LabelEnv:
    """Reuse label storage so copying after step would produce incorrect labels."""

    num_envs = 2
    device = torch.device("cpu")

    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim
        self.labels = torch.zeros(2, 2, dtype=torch.long)
        self.step_index = 0

    def _observation(self):
        return {
            "proprioception": torch.full((2, self.obs_dim), float(self.step_index)),
            "privileged": torch.ones(2, 4),
            "contact_labels": self.labels,
        }

    def reset(self):
        self.step_index = 0
        self.labels.zero_()
        return self._observation(), {}

    def step(self, action):
        self.step_index += 1
        self.labels.fill_(self.step_index % 2)
        return (
            self._observation(),
            action.sum(-1),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            {},
        )


def test_labels_are_pre_action_aligned_across_rollouts_and_ppo_updates() -> None:
    policy = _policy()
    env = _LabelEnv(policy.obs_dim)
    collector = SyncCollector(env, policy, env.device)
    buffer = RolloutBuffer(
        2,
        3,
        policy.obs_dim,
        2,
        env.device,
        critic_obs_dim=4,
        distribution_param_dim=2,
        training_field_specs=policy.training_field_specs,
    )
    rollout = collector.collect(3, rollout=buffer.start_rollout())
    buffer.add(rollout)
    stored = buffer.get(flatten=False)
    torch.testing.assert_close(
        stored["contact_labels"][0, :, 0], torch.tensor([0, 1, 0, 0])
    )
    flat = transition_view(stored, flatten=True)
    assert flat["contact_labels"].dtype == torch.long
    assert flat["contact_labels"].shape == (6, 2)
    expected = torch.tensor([0, 1, 0, 0, 1, 0])
    torch.testing.assert_close(flat["contact_labels"][:, 0], expected)
    before = next(policy.contact_decoder.parameters()).detach().clone()
    algorithm = PPO(
        PPOCfg(
            device="cpu",
            n_epochs=1,
            batch_size=6,
            auxiliary_coef=1.0,
            optimizer=OptimizerCfg(learning_rate=1e-3),
        ),
        policy,
    )
    metrics = algorithm.update(stored)
    assert metrics["auxiliary_loss"] > 0
    assert all(math.isfinite(v) for v in metrics.values())
    assert not torch.equal(next(policy.contact_decoder.parameters()), before)
    second = collector.collect(3, rollout=buffer.start_rollout())
    torch.testing.assert_close(
        second["contact_labels"][0, :3, 0], torch.tensor([1, 0, 1])
    )


@pytest.mark.parametrize("name", ["next_value", "advantage"])
def test_training_fields_cannot_silently_replace_derived_transition_data(
    name: str,
) -> None:
    with pytest.raises(ValueError, match="collides"):
        RolloutBuffer(
            1,
            1,
            2,
            1,
            torch.device("cpu"),
            training_field_specs={name: ((), torch.float32)},
        )


@pytest.mark.parametrize(("history_length", "use_history"), [(0, True), (30, False)])
def test_no_history_no_decoder_ablation_has_current_only_actor(
    history_length: int, use_history: bool
) -> None:
    policy = _policy(
        contact_points=0, history_length=history_length, use_history=use_history
    )
    td = _batch(policy)
    policy.get_action(td)
    evaluated = policy.evaluate_actions(td)
    assert evaluated["action_mean"].shape == (2, 2)
    assert "contact_logits" not in evaluated.keys()
    assert policy.training_field_specs == {}
    assert policy.actor.encoder is None


@pytest.mark.parametrize(
    ("actor_group", "label_group"),
    [
        ("contact_labels", "contact_labels"),
        ("robot", "robot.contact_labels"),
    ],
)
def test_contact_labels_cannot_be_selected_as_actor_observations(
    actor_group: str,
    label_group: str,
) -> None:
    with pytest.raises(ValueError, match="Contact labels"):
        build_policy(
            {
                "name": "proprioceptive_actor_critic",
                "obs_groups": {"actor": [actor_group], "critic": ["privileged"]},
                "cfg": {"frame_dim": 3, "contact_observation_group": label_group},
            },
            93,
            2,
            torch.device("cpu"),
            critic_obs_space=4,
        )


def test_timeout_bootstraps_terminal_state_instead_of_current_or_reset_state() -> None:
    from embodichain.learning.rl.algo import compute_gae
    from embodichain.learning.rl.policies import ActorCritic

    class TimeoutEnv:
        num_envs = 1
        device = torch.device("cpu")

        def reset(self):
            return {"state": torch.tensor([[2.0]])}, {}

        def step(self, action):
            return (
                {"state": torch.tensor([[99.0]])},
                torch.tensor([1.0]),
                torch.tensor([False]),
                torch.tensor([True]),
                {
                    "terminal_observation": {"state": torch.tensor([[7.0]])},
                    "terminal_env_ids": torch.tensor([0]),
                    "terminal_observation_mask": torch.tensor([True]),
                },
            )

    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1.0)
    policy = ActorCritic(1, 1, torch.device("cpu"), torch.nn.Linear(1, 1), critic)
    env = TimeoutEnv()
    buffer = RolloutBuffer(1, 1, 1, 1, env.device)
    rollout = SyncCollector(env, policy, env.device).collect(
        1, rollout=buffer.start_rollout()
    )
    _, returns = compute_gae(rollout, gamma=0.9, gae_lambda=1.0)
    torch.testing.assert_close(returns, torch.tensor([[7.3]]))


@pytest.mark.parametrize("tensor_dict_info", [False, True])
def test_base_env_preserves_terminal_rows_before_reused_reset_storage(
    tensor_dict_info: bool,
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace
    from embodichain.lab.gym.envs import BaseEnv

    state = torch.tensor([[3.0], [7.0], [11.0]])
    observation = {"state": state}
    info = {"success": torch.tensor([True, False, False])}
    if tensor_dict_info:
        info = TensorDict(info, batch_size=[3])

    def reset(**kwargs):
        state[kwargs["options"]["reset_ids"]] = 99.0
        return observation, {}

    env = SimpleNamespace(
        _profiler=SimpleNamespace(section=lambda *args, **kwargs: nullcontext()),
        sim=SimpleNamespace(render_frame=lambda: nullcontext()),
        _preprocess_action=lambda **kwargs: kwargs["action"],
        _step_action=lambda **kwargs: kwargs["action"],
        _postprocess_action=lambda **kwargs: kwargs["action"],
        _advance_physics=lambda: None,
        _update_sim_state=lambda **kwargs: None,
        _update_physical_objective=lambda: None,
        get_obs=lambda **kwargs: observation,
        get_info=lambda **kwargs: info,
        get_reward=lambda **kwargs: torch.ones(3),
        _extend_reward=lambda **kwargs: kwargs["rewards"],
        check_truncated=lambda **kwargs: torch.tensor([False, True, False]),
        _hook_after_sim_step=lambda **kwargs: None,
        _elapsed_steps=torch.zeros(3, dtype=torch.long),
        max_episode_steps=10,
        cfg=SimpleNamespace(ignore_terminations=False),
        num_envs=3,
        device=torch.device("cpu"),
        reset=reset,
    )
    _, _, _, _, info = BaseEnv.step(env, torch.zeros(3, 1))
    torch.testing.assert_close(
        info["terminal_observation"]["state"], torch.tensor([[3.0], [7.0]])
    )
    torch.testing.assert_close(state, torch.tensor([[99.0], [99.0], [11.0]]))
    torch.testing.assert_close(
        info["terminal_observation_mask"], torch.tensor([False, True, False])
    )
