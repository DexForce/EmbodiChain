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

"""RWM fitting, graph lifecycle and simulator-independent checkpoint behavior."""

from __future__ import annotations

from copy import deepcopy
import subprocess
import sys

import pytest
import torch

from embodichain_models.dynamics.rwm.robot_state import (
    integrate_robot_state,
    robot_state_error,
)
from embodichain_models.dynamics.rwm.model import (
    RobotWorldModel,
    RobotWorldModelCfg,
    frozen_world_model,
)
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner
from embodichain_models.dynamics.rwm import (
    RobotWorldModelObjective,
    RWMPredictor,
    build_rwm_trainer,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _state(batch: int, joints: int = 1) -> torch.Tensor:
    result = torch.zeros(batch, 18 + 2 * joints)
    result[:, 2] = 1
    result[:, 3:12] = torch.eye(3).flatten()
    return result


def _controlled_batch(batch: int = 64, horizon: int = 4) -> SupervisedBatch:
    initial = _state(batch)
    initial[:, 18:] = torch.randn(batch, 2) * 0.3
    actions = torch.randn(batch, horizon, 1)
    states = []
    state = initial
    for index in range(horizon):
        increment = torch.zeros(batch, 14)
        increment[:, 12] = 0.1 * state[:, 19] + 0.005 * actions[:, index, 0]
        increment[:, 13] = 0.1 * actions[:, index, 0]
        state = integrate_robot_state(state, increment)
        states.append(state)
    return SupervisedBatch(
        {"initial_state": initial, "actions": actions},
        torch.stack(states, 1),
        {"valid": torch.ones(batch, horizon, dtype=torch.bool)},
    )


def _nontrivial_model(joints: int = 1) -> RobotWorldModel:
    model = RobotWorldModel(
        RobotWorldModelCfg(num_joints=joints, action_dim=joints, hidden_dims=(24, 24))
    )
    with torch.no_grad():
        torch.nn.init.normal_(model.net[-1].weight, std=0.03)
        torch.nn.init.normal_(model.net[-1].bias, std=0.01)
    return model


def test_multistep_fitting_uses_shared_learner_and_decreases_holdout_error():
    torch.manual_seed(11)
    batch = _controlled_batch(256)
    holdout = _controlled_batch(64)
    model = RobotWorldModel(RobotWorldModelCfg(hidden_dims=(48, 48)))
    model.update_normalization(
        batch.inputs["initial_state"],
        batch.inputs["actions"][:, 0],
        batch.targets[:, 0],
    )
    objective = RobotWorldModelObjective()
    before = float(objective.compute_loss(model, holdout).loss)
    learner = SupervisedLearner(
        model, objective, torch.optim.Adam(model.parameters(), lr=2e-3), max_grad_norm=1
    )
    for _ in range(250):
        learner.update(batch)
    after = float(objective.compute_loss(model, holdout).loss)
    assert learner.num_updates == 250
    assert after < before * 0.15, (before, after)


def test_frozen_world_model_preserves_multistep_state_and_action_gradients():
    torch.manual_seed(4)
    model = _nontrivial_model()
    state = _state(2).requires_grad_()
    actions = torch.randn(2, 3, 1, requires_grad=True)

    def predict(initial, controls):
        current = initial
        for index in range(3):
            current = model(current, controls[:, index])
        return current[:, 18:].sum() + current[:, :3].sum()

    with frozen_world_model(model):
        value = predict(state, actions)
        state_grad, action_grad = torch.autograd.grad(value, (state, actions))
        assert torch.isfinite(state_grad).all() and torch.isfinite(action_grad).all()
        assert action_grad[:, 0].abs().sum() > 0
        epsilon = 1e-3
        plus, minus = actions.detach().clone(), actions.detach().clone()
        plus[0, 0, 0] += epsilon
        minus[0, 0, 0] -= epsilon
        numerical = (predict(state.detach(), plus) - predict(state.detach(), minus)) / (
            2 * epsilon
        )
        torch.testing.assert_close(
            action_grad[0, 0, 0], numerical, atol=2e-4, rtol=0.03
        )
        state_plus, state_minus = state.detach().clone(), state.detach().clone()
        state_plus[0, 18] += epsilon
        state_minus[0, 18] -= epsilon
        numerical_state = (
            predict(state_plus, actions.detach())
            - predict(state_minus, actions.detach())
        ) / (2 * epsilon)
        torch.testing.assert_close(
            state_grad[0, 18], numerical_state, atol=2e-4, rtol=0.01
        )
        with pytest.raises(RuntimeError, match="frozen"):
            model.update_normalization(
                state.detach(), actions[:, 0].detach(), state.detach()
            )
    assert all(
        parameter.requires_grad and parameter.grad is None
        for parameter in model.parameters()
    )


def test_predictor_selective_reset_preserves_other_graphs_and_detach_values():
    model = _nontrivial_model()
    predictor = RWMPredictor(model, _state(2))
    action = torch.ones(2, 1, requires_grad=True)
    original = predictor.step(action)
    original_copy = original.detach().clone()
    replacement = _state(1)
    replacement[:, 18] = 4
    predictor.reset(replacement, torch.tensor([0]))
    gradient = torch.autograd.grad(predictor.state[:, 18:].sum(), action)[0]
    assert gradient[0].eq(0).all() and gradient[1].abs().sum() > 0
    torch.testing.assert_close(original, original_copy)
    before = predictor.state.detach().clone()
    detached = predictor.detach_state()
    assert not detached.requires_grad
    torch.testing.assert_close(detached, before, atol=0, rtol=0)
    snapshot = predictor.state_dict()
    predictor.step(torch.ones(2, 1))
    predictor.load_state_dict(snapshot)
    torch.testing.assert_close(predictor.state, before, atol=0, rtol=0)


def test_target_mask_excludes_invalid_suffix_and_rejects_episode_reentry():
    batch = _controlled_batch(2, 3)
    model = _nontrivial_model()
    objective = RobotWorldModelObjective()
    valid = torch.tensor([[True, True, False], [True, False, False]])
    original = SupervisedBatch(batch.inputs, batch.targets.clone(), {"valid": valid})
    changed = batch.targets.clone()
    changed[~valid, :3] = 100
    actual = objective.compute_loss(
        model, SupervisedBatch(batch.inputs, changed, {"valid": valid})
    )
    torch.testing.assert_close(
        actual.loss, objective.compute_loss(model, original).loss
    )
    assert actual.weight == 3
    with pytest.raises(ValueError, match="prefix"):
        objective.compute_loss(
            model,
            SupervisedBatch(
                batch.inputs,
                changed,
                {"valid": torch.tensor([[True, False, True], [True, True, True]])},
            ),
        )


def test_nan_padding_preserves_masked_loss_diagnostics_and_backward():
    torch.manual_seed(19)
    corpus = _controlled_batch(3, 4)
    contexts = torch.randn(3, 4, 2)
    valid = torch.tensor(
        [
            [True, True, True, False],
            [True, False, False, False],
            [True, True, False, False],
        ]
    )
    model = RobotWorldModel(RobotWorldModelCfg(context_dim=2, hidden_dims=(24, 24)))
    with torch.no_grad():
        torch.nn.init.normal_(model.net[-1].weight, std=0.03)
        torch.nn.init.normal_(model.net[-1].bias, std=0.01)
    objective = RobotWorldModelObjective()
    results = []
    gradients = []
    for poison_padding in (False, True):
        current = deepcopy(model)
        state = corpus.inputs["initial_state"].clone().requires_grad_()
        actions = corpus.inputs["actions"].clone()
        context = contexts.clone()
        targets = corpus.targets.clone()
        if poison_padding:
            actions[~valid] = float("nan")
            context[~valid] = float("nan")
            targets[~valid] = float("nan")
        actions.requires_grad_()
        context.requires_grad_()
        targets.requires_grad_()
        result = objective.compute_loss(
            current,
            SupervisedBatch(
                {"initial_state": state, "actions": actions, "contexts": context},
                targets,
                {"valid": valid},
            ),
        )
        result.loss.backward()
        result_gradients = [
            state.grad,
            actions.grad,
            context.grad,
            targets.grad,
            *(parameter.grad for parameter in current.parameters()),
        ]
        assert all(torch.isfinite(value).all() for value in result_gradients)
        assert actions.grad[~valid].eq(0).all()
        assert context.grad[~valid].eq(0).all()
        assert targets.grad[~valid].eq(0).all()
        assert torch.isfinite(result.loss)
        assert all(torch.isfinite(value) for value in result.metrics.values())
        results.append(result)
        gradients.append(result_gradients)
    assert results[0].weight == results[1].weight == 6
    torch.testing.assert_close(results[0].loss, results[1].loss, atol=0, rtol=0)
    for name in results[0].metrics:
        torch.testing.assert_close(
            results[0].metrics[name], results[1].metrics[name], atol=0, rtol=0
        )
    for clean, padded in zip(*gradients):
        torch.testing.assert_close(clean, padded, atol=0, rtol=0)


def test_all_valid_objective_preserves_direct_rollout_numerics_and_gradients():
    torch.manual_seed(23)
    corpus = _controlled_batch(3, 4)
    model = _nontrivial_model()
    objective = RobotWorldModelObjective()
    reference = deepcopy(model)
    state = corpus.inputs["initial_state"]
    predictions = []
    for index in range(corpus.inputs["actions"].shape[1]):
        state = reference(state, corpus.inputs["actions"][:, index])
        predictions.append(state)
    predicted = torch.stack(predictions, 1)
    error = robot_state_error(predicted, corpus.targets)
    scale = reference.output_normalizer.standard_deviation.clamp_min(
        objective.error_floor
    )
    squared = (error / scale).square().mean(-1)
    expected_loss = squared.mean()
    expected_loss.backward()
    actual = objective.compute_loss(model, corpus)
    actual.loss.backward()
    torch.testing.assert_close(actual.loss, expected_loss, atol=0, rtol=0)
    torch.testing.assert_close(
        actual.metrics["last_step_mse"], squared[:, -1].mean(), atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual.metrics["position_error_m"],
        (predicted[..., :3] - corpus.targets[..., :3]).norm(dim=-1).mean(),
        atol=0,
        rtol=0,
    )
    for current, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(current.grad, expected.grad, atol=0, rtol=0)


def test_robot_layout_context_checkpoint_and_fp32_under_autocast():
    torch.manual_seed(7)
    model = RobotWorldModel(
        RobotWorldModelCfg(
            num_joints=29, action_dim=29, context_dim=3, hidden_dims=(32,)
        )
    )
    state, action, context = _state(2, 29), torch.randn(2, 29), torch.randn(2, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(state, action, context)
    assert result.dtype == torch.float32 and result.shape == state.shape
    torch.testing.assert_close(result, state, atol=0, rtol=0)
    clone = RobotWorldModel(deepcopy(model.cfg))
    clone.load_state_dict(deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(state, action, context), result)
    with pytest.raises(ValueError, match="context"):
        model(state, action)
    incompatible = RobotWorldModel(
        RobotWorldModelCfg(
            num_joints=29,
            action_dim=29,
            context_dim=3,
            hidden_dims=(32,),
            action_semantics="joint_position",
        )
    )
    with pytest.raises(ValueError, match="contract"):
        incompatible.load_state_dict(model.state_dict())


def test_rwm_imports_do_not_initialize_simulation():
    code = """
import sys
import embodichain.learning.sl
assert not any(name == 'embodichain_models' or name.startswith('embodichain_models.') for name in sys.modules)
from embodichain_models.dynamics.rwm import RobotWorldModel, RobotWorldModelObjective, RWMPredictor, build_rwm_trainer
import embodichain_models.dynamics.rwm.replay
assert not any(name == 'dexsim' or name.startswith('dexsim.') or name.startswith('embodichain.lab.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_offline_recipe_preserves_committed_updates_on_resume(tmp_path):
    corpus = _controlled_batch(16, 2)
    path = tmp_path / "training.pt"
    torch.save(
        {
            "inputs": corpus.inputs,
            "targets": corpus.targets,
            "masks": corpus.masks,
            "metadata": {
                "split": "train",
                "action_semantics": "generalized_effort",
                "step_dt": 0.02,
            },
        },
        path,
    )
    config = {
        "model": {"hidden_dims": [16]},
        "data": {"path": str(path)},
        "trainer": {
            "total_updates": 4,
            "batch_size": 8,
            "output_dir": str(tmp_path / "run"),
            "seed": 17,
        },
    }
    full = build_rwm_trainer(config)
    full.fit(stop_after=2)
    resumed = build_rwm_trainer(
        {
            **config,
            "trainer": {**config["trainer"], "output_dir": str(tmp_path / "resumed")},
        }
    )
    resumed.resume(tmp_path / "run" / "last.pt")
    assert resumed.learner.num_updates == 2
    resumed.fit()
    full.fit()
    for name, expected in full.learner.model.state_dict().items():
        actual = resumed.learner.model.state_dict()[name]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        else:
            assert actual == expected


@pytest.mark.gpu
def test_cuda_replay_and_bfloat16_network_preserve_fp32_state():
    from embodichain_models.dynamics.rwm.replay import TransitionReplay

    model = RobotWorldModel(
        RobotWorldModelCfg(hidden_dims=(16,), amp_bfloat16=True)
    ).cuda()
    state = _state(2).cuda()
    action = torch.ones(2, 1, device="cuda", requires_grad=True)
    prediction = model(state, action)
    assert prediction.dtype == torch.float32
    prediction.sum().backward()
    assert torch.isfinite(action.grad).all()
    replay = TransitionReplay(4, 2, 20, 1, device="cuda:0")
    replay.append(state, action, prediction)
    batch = replay.sample_windows(2, 1)
    assert batch["state"].is_cuda and not batch["state"].requires_grad


@pytest.mark.parametrize("declared_names", [None, ["other"], ["hinge"]])
def test_recipe_binds_recorded_joint_order(tmp_path, declared_names):
    corpus = _controlled_batch(8, 2)
    metadata = {
        "split": "train",
        "action_semantics": "generalized_effort",
        "step_dt": 0.02,
    }
    if declared_names is not None:
        metadata["joint_names"] = declared_names
    path = tmp_path / "named.pt"
    torch.save(
        {
            "inputs": corpus.inputs,
            "targets": corpus.targets,
            "masks": corpus.masks,
            "metadata": metadata,
        },
        path,
    )
    config = {
        "model": {"hidden_dims": [16], "joint_names": ["hinge"]},
        "data": {"path": str(path)},
        "trainer": {
            "total_updates": 1,
            "batch_size": 4,
            "output_dir": str(tmp_path / "run"),
        },
    }
    if declared_names == ["hinge"]:
        assert build_rwm_trainer(config).fit()["num_updates"] == 1
    else:
        with pytest.raises(ValueError, match="joint_names"):
            build_rwm_trainer(config)
        assert not (tmp_path / "run").exists()
