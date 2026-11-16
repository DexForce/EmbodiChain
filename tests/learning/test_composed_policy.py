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
    ActorCritic,
    AuxiliaryTask,
    ContactPrediction,
    ObservationHistory,
    build_policy,
)
from embodichain.learning.rl.utils import OptimizerCfg


def _policy(contact_points=2, history_length=30, use_history=True, normalize=False):
    block = {
        "name": "actor_critic",
        "obs_groups": {"actor": ["proprioception"], "critic": ["privileged"]},
        "actor_obs_normalization": normalize,
        "critic_obs_normalization": normalize,
        "actor": {
            "type": "temporal",
            "network_cfg": {
                "frame_dim": 3,
                "history_length": history_length,
                "use_history": use_history,
                "hidden_sizes": [8],
            },
        },
        "critic": {"type": "mlp", "network_cfg": {"hidden_sizes": [8]}},
    }
    if contact_points:
        block["auxiliary_tasks"] = {
            "contact": {
                "type": "contact_prediction",
                "contact_points": contact_points,
                "contact_classes": 2,
                "class_weights": [1.0, 3.0],
                "hidden_sizes": [8],
            }
        }
    return build_policy(
        block, 3 * (history_length + 1), 2, torch.device("cpu"), critic_obs_space=4
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
        after[1],
        torch.tensor([[14.0, 15.0, 16.0], [8.0, 9.0, 10.0], [14.0, 15.0, 16.0]]),
    )


def test_actor_does_not_read_privileged_values_and_export_matches() -> None:
    policy = _policy(normalize=True)
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
    restored = _policy(normalize=True)
    restored.load_state_dict(deepcopy(policy.state_dict()))
    assert int(restored.actor_obs_normalizer.count) == 2
    torch.testing.assert_close(restored.export_actor()(td["obs"]), original["action"])


def test_weighted_contact_loss_matches_independent_logsumexp_and_backpropagates() -> (
    None
):
    policy = _policy()
    batch = _batch(policy)
    policy.get_action(batch)
    labels = torch.tensor([[0, 1], [1, 1]])
    batch["contact_labels"] = labels
    encoder_calls = []
    policy.actor.encoder.register_forward_hook(lambda *args: encoder_calls.append(1))
    evaluated = policy.evaluate_actions(batch)
    assert encoder_calls == [1]
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
    history = ObservationHistory(2, 1, history_length=2)
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
    policy = _policy(normalize=True)
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
            "name": "actor_critic",
            "obs_groups": {"actor": ["proprioception"], "critic": ["privileged"]},
            "actor_obs_normalization": True,
            "actor": {"type": "temporal", "network_cfg": {}},
            "critic": {"type": "mlp", "network_cfg": {"hidden_sizes": [256, 256, 128]}},
            "auxiliary_tasks": {
                "contact": {
                    "type": "contact_prediction",
                    "contact_points": 64,
                    "contact_classes": 3,
                }
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
    history_length: int,
    use_history: bool,
) -> None:
    policy = _policy(
        contact_points=0,
        history_length=history_length,
        use_history=use_history,
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
                "name": "actor_critic",
                "obs_groups": {"actor": [actor_group], "critic": ["privileged"]},
                "actor": {"type": "temporal", "network_cfg": {"frame_dim": 3}},
                "critic": {"type": "mlp", "network_cfg": {"hidden_sizes": [8]}},
                "auxiliary_tasks": {
                    "contact": {
                        "type": "contact_prediction",
                        "contact_points": 2,
                        "observation_group": label_group,
                    }
                },
            },
            93,
            2,
            torch.device("cpu"),
            critic_obs_space=4,
        )


def test_actor_critic_composes_generic_objectives_with_one_feature_pass() -> None:
    """Shared observations, non-temporal actors and non-contact tasks remain valid."""

    class FeatureActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 2)
            self.input_dim = (3,)  # Custom actors need no scalar dimension convention.
            self.feature_calls = 0

        def forward(self, value):
            return self.head(value)

        def forward_with_features(self, value):
            self.feature_calls += 1
            return self.head(value), {"latent": value}

    class FeatureRegression(AuxiliaryTask):
        modules = {}
        training_observation_groups = {"target": ("feature_target",)}
        training_field_specs = {"target": ((3,), torch.float32)}
        output_names = ("prediction",)

        def evaluate(self, features):
            return {"prediction": features["latent"]}

        def loss(self, labels, outputs, buffers):
            return (outputs["prediction"] - labels["target"]).square().mean()

    actor = FeatureActor()
    shared_normalizer = torch.nn.LayerNorm(3)
    policy = ActorCritic(
        3,
        2,
        torch.device("cpu"),
        actor,
        torch.nn.Linear(3, 1),
        actor_obs_groups=("state",),
        actor_normalizer=shared_normalizer,
        critic_normalizer=shared_normalizer,
        auxiliary_tasks={
            "contact": ContactPrediction(
                3, 2, class_weights=(1.0, 3.0), hidden_sizes=(8,)
            ),
            "regression": FeatureRegression(),
        },
    ).to(dtype=torch.float64)
    assert policy.actor_obs_groups == policy.critic_obs_groups
    assert policy.contact_class_weights.dtype == torch.float64
    calls = []
    policy.contact_decoder.register_forward_hook(lambda *args: calls.append(1))
    batch = TensorDict(
        {
            "obs": torch.tensor(
                [[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]],
                dtype=torch.float64,
                requires_grad=True,
            ),
            "contact_labels": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "regression_target": torch.zeros(2, 3),
        },
        batch_size=[2],
    )
    policy.get_action(batch)
    assert calls == []
    evaluated = policy.evaluate_actions(batch)
    assert actor.feature_calls == 1
    assert calls == [1]
    loss = policy.auxiliary_loss(batch, evaluated)
    logits = evaluated["contact_logits"]
    target = batch["contact_labels"]
    nll = logits.logsumexp(-1) - logits.gather(-1, target[..., None]).squeeze(-1)
    weights = torch.tensor([[1.0, 3.0], [3.0, 1.0]], dtype=torch.float64)
    obs = batch["obs"]
    normalized = (obs - obs.mean(-1, keepdim=True)) / (
        obs.var(-1, unbiased=False, keepdim=True) + policy.actor_obs_normalizer.eps
    ).sqrt()
    expected = (nll * weights).sum() / weights.sum() + normalized.square().mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert batch["obs"].grad.abs().sum() > 0
    assert policy.contact_decoder[0].weight.grad.abs().sum() > 0
    actor_params, critic_params = policy.optimization_parameter_groups()
    ids = [id(parameter) for parameter in (*actor_params, *critic_params)]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(parameter) for parameter in policy.parameters()}
    copied = deepcopy(policy)
    assert copied.auxiliary_tasks["contact"].decoder is copied.contact_decoder
    exported = policy.export_actor()
    expected_action = policy.get_action(batch.clone(), deterministic=True)["action"]
    torch.testing.assert_close(exported(batch["obs"]), expected_action)
    assert not any(
        "contact" in name or "critic" in name for name in exported.state_dict()
    )


def test_auxiliary_tasks_require_an_actor_feature_interface_and_isolated_labels():
    with pytest.raises(ValueError, match="forward_with_features"):
        ActorCritic(
            3,
            2,
            torch.device("cpu"),
            torch.nn.Linear(3, 2),
            torch.nn.Linear(3, 1),
            actor_obs_groups=("state",),
            auxiliary_tasks={"contact": ContactPrediction(3, 2)},
        )
    with pytest.raises(ValueError, match="Contact labels"):
        policy = _policy()
        ActorCritic(
            policy.obs_dim,
            2,
            torch.device("cpu"),
            policy.actor,
            policy.critic,
            actor_obs_groups=("robot",),
            auxiliary_tasks={
                "contact": ContactPrediction(
                    32, 2, observation_group="robot.contact_labels"
                )
            },
        )


def test_composed_checkpoint_continues_optimizer_state(tmp_path):
    from embodichain.learning.rl.utils.trainer import Trainer

    def trainer(directory):
        policy = _policy(normalize=True)
        env = _LabelEnv(policy.obs_dim)
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
        return Trainer(
            policy,
            env,
            algorithm,
            buffer_size=3,
            batch_size=6,
            writer=None,
            eval_freq=0,
            save_freq=0,
            checkpoint_dir=str(directory),
            exp_name="composed",
            use_wandb=False,
        )

    original = trainer(tmp_path / "original")
    original.train(6)
    fixed = original.collector.collect(3, original.buffer.start_rollout()).clone()
    path = original.save_checkpoint()
    original.load_checkpoint(path)
    expected_metrics = original.algorithm.update(fixed.clone())
    expected_state = deepcopy(original.policy.state_dict())
    expected_optimizer = deepcopy(original.algorithm.optimizer.state_dict())
    resumed = trainer(tmp_path / "resumed")
    resumed.load_checkpoint(path)
    actual_metrics = resumed.algorithm.update(fixed.clone())
    assert actual_metrics == expected_metrics
    for name, value in resumed.policy.state_dict().items():
        torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
    actual_optimizer = resumed.algorithm.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for key, state in actual_optimizer["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, expected_optimizer["state"][key][field], rtol=0, atol=0
            )


def test_composed_actor_is_built_through_the_learning_runtime():
    from gymnasium import spaces
    from types import SimpleNamespace
    from embodichain.learning.rl.runtime import _build_learning_policy

    block = {
        "name": "actor_critic",
        "obs_groups": {"actor": ["proprioception"], "critic": ["privileged"]},
        "actor": {
            "type": "temporal",
            "network_cfg": {"frame_dim": 3, "history_length": 30, "hidden_sizes": [8]},
        },
        "critic": {"type": "mlp", "network_cfg": {"hidden_sizes": [8]}},
        "auxiliary_tasks": {
            "contact": {
                "type": "contact_prediction",
                "contact_points": 2,
                "hidden_sizes": [8],
            }
        },
    }
    env = SimpleNamespace(
        single_observation_space=spaces.Dict(
            {
                "proprioception": spaces.Box(-1.0, 1.0, shape=(93,)),
                "privileged": spaces.Box(-1.0, 1.0, shape=(4,)),
                "contact_labels": spaces.Box(0, 1, shape=(2,), dtype="int64"),
            }
        ),
        single_action_space=spaces.Box(-1.0, 1.0, shape=(2,)),
    )
    before = deepcopy(block)
    policy = _build_learning_policy(block, env, torch.device("cpu"))
    assert type(policy) is ActorCritic
    assert policy.obs_dim == 93 and policy.critic_obs_dim == 4
    assert policy.training_field_specs == {"contact_labels": ((2,), torch.long)}
    assert block == before


@pytest.mark.parametrize("override_style", ["subclass", "instance"])
def test_custom_auxiliary_override_remains_supported_without_registered_tasks(
    override_style,
):
    class CustomObjective(ActorCritic):
        def auxiliary_loss(self, batch, evaluated):
            return evaluated["action_mean"].square().mean()

    def policy(cls):
        return cls(
            3, 2, torch.device("cpu"), torch.nn.Linear(3, 2), torch.nn.Linear(3, 1)
        )

    custom = policy(CustomObjective if override_style == "subclass" else ActorCritic)
    if override_style == "instance":
        custom.auxiliary_loss = (
            lambda batch, evaluated: evaluated["action_mean"].square().mean()
        )
    PPO(PPOCfg(device="cpu", auxiliary_coef=1.0), custom)
    td = TensorDict({"obs": torch.ones(2, 3)}, batch_size=[2])
    custom.get_action(td)
    evaluated = custom.evaluate_actions(td)
    custom.auxiliary_loss(td, evaluated).backward()
    assert custom.actor.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="configured tasks"):
        PPO(PPOCfg(device="cpu", auxiliary_coef=1.0), policy(ActorCritic))


def test_component_checkpoint_contract_rejects_shape_preserving_objective_changes(
    tmp_path,
):
    """Headless loss settings cannot hide behind equal neural tensors."""
    from embodichain.learning.rl.policies import TemporalActor, TemporalActorCfg
    from embodichain.learning.rl.utils.trainer import Trainer

    class FeaturePenalty(AuxiliaryTask):
        modules = {}
        training_observation_groups = {}
        training_field_specs = {}
        output_names = ("prediction",)

        def __init__(self, scale):
            self.scale = scale

        def evaluate(self, features):
            return {"prediction": features["latent"]}

        def loss(self, labels, outputs, buffers):
            return outputs["prediction"].square().mean() * self.scale

        def checkpoint_contract(self):
            return {**super().checkpoint_contract(), "scale": self.scale}

    def trainer(changed):
        actor = TemporalActor(
            TemporalActorCfg(
                frame_dim=3,
                history_length=30,
                use_history=False,
                hidden_sizes=(8,),
            ),
            2,
        )
        tasks = {"penalty": FeaturePenalty(2 if changed else 1)}
        policy = ActorCritic(
            93,
            2,
            torch.device("cpu"),
            actor,
            torch.nn.Linear(4, 1),
            critic_obs_dim=4,
            actor_obs_groups=("proprioception",),
            critic_obs_groups=("privileged",),
            auxiliary_tasks=tasks,
        )
        algorithm = PPO(
            PPOCfg(device="cpu", auxiliary_coef=1.0 if tasks else 0.0), policy
        )
        return Trainer(
            policy,
            _LabelEnv(93),
            algorithm,
            buffer_size=3,
            batch_size=6,
            writer=None,
            eval_freq=0,
            save_freq=0,
            checkpoint_dir=str(tmp_path),
            exp_name="contract",
            use_wandb=False,
        )

    original, changed = trainer(False), trainer(True)
    assert [
        (key, value.shape) for key, value in original.policy.state_dict().items()
    ] == [(key, value.shape) for key, value in changed.policy.state_dict().items()]
    path = original.save_checkpoint()
    before = deepcopy(changed.policy.state_dict())
    with pytest.raises(ValueError, match="policy_components"):
        changed.load_checkpoint(path)
    for key, value in changed.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_removed_policy_and_history_versions_are_rejected():
    from embodichain.learning.rl.policies import TemporalActor, TemporalActorCfg

    with pytest.raises(ValueError, match="not registered"):
        build_policy({"name": "proprioceptive_actor_critic"}, 3, 2, torch.device("cpu"))
    with pytest.raises(ValueError, match="architecture_version"):
        TemporalActor(TemporalActorCfg(architecture_version="legacy_v1"), 2)
