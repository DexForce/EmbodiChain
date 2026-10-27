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

"""Configurable BC targets, masked gradients and per-environment execution."""

from __future__ import annotations

from copy import deepcopy
import subprocess
import sys

import pytest
import torch

from embodichain_models.policy.bc.model import (
    ChunkBCModel,
    ChunkBCModelCfg,
    ChunkBCExecutor,
)
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner
from embodichain_models.policy.bc.supervised import ChunkBCObjective


def test_single_step_masked_objective_has_independent_loss_and_gradient():
    model = ChunkBCModel(
        ChunkBCModelCfg(observation_dim=3, target_dim=2, hidden_dims=(4,))
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.set_normalization(
        torch.zeros(3),
        torch.ones(3),
        torch.tensor([2.0, -1.0]),
        torch.tensor([2.0, 4.0]),
    )
    targets = torch.tensor([[[4.0, 3.0]], [[100.0, 100.0]]])
    batch = SupervisedBatch(
        {"history": torch.ones(2, 1, 3)},
        targets,
        {"target_valid": torch.tensor([[True], [False]])},
    )
    result = ChunkBCObjective().compute_loss(model, batch)
    # Valid target normalizes to [1,1], prediction is [0,0]. Invalid labels
    # contribute neither loss nor the independently known final-bias gradient.
    torch.testing.assert_close(result.loss, torch.tensor(1.0))
    result.loss.backward()
    torch.testing.assert_close(model.head[-1].bias.grad, torch.tensor([-1.0, -1.0]))
    assert result.weight == 1


def test_observation_noise_is_training_only_and_uses_normalization_scale():
    model = ChunkBCModel(
        ChunkBCModelCfg(observation_dim=2, target_dim=1, hidden_dims=(2,))
    )
    with torch.no_grad():
        model.head[0].weight.copy_(torch.eye(2))
        model.head[0].bias.zero_()
        model.head[2].weight.copy_(torch.tensor([[1.0, 2.0]]))
        model.head[2].bias.zero_()
    model.set_normalization(
        torch.zeros(2), torch.tensor([2.0, 4.0]), torch.zeros(1), torch.ones(1)
    )
    history = torch.tensor([[[2.0, 8.0]]])
    batch = SupervisedBatch(
        {"history": history},
        torch.zeros(1, 1, 1),
        {"target_valid": torch.ones(1, 1, dtype=torch.bool)},
    )
    original = history.clone()
    torch.manual_seed(27)
    expected_noise = torch.randn_like(history) * 0.01
    # The model sees the independent normalized input [1,2] + noise.
    expected = (1 + expected_noise[0, 0, 0] + 2 * (2 + expected_noise[0, 0, 1])) ** 2
    torch.manual_seed(27)
    actual = ChunkBCObjective(observation_noise=0.01).compute_loss(model.train(), batch)
    torch.testing.assert_close(actual.loss, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(history, original, atol=0, rtol=0)
    for objective, train in (
        (ChunkBCObjective(), True),
        (ChunkBCObjective(0.01), False),
    ):
        model.train(train)
        before = torch.get_rng_state().clone()
        torch.testing.assert_close(
            objective.compute_loss(model, batch).loss, torch.tensor(25.0)
        )
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    for invalid in (-0.01, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="noise"):
            ChunkBCObjective(invalid)


def test_state_fit_and_checkpoint_preserve_configured_target_semantics():
    torch.manual_seed(6)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = ChunkBCModel(
            ChunkBCModelCfg(
                observation_dim=2,
                target_dim=1,
                hidden_dims=(16,),
                target_semantics="joint_position",
            )
        )
        history = torch.randn(64, 1, 2)
        target = history[..., :1] * 0.4 - history[..., 1:] * 0.2
        batch = SupervisedBatch(
            {"history": history},
            target,
            {"target_valid": torch.ones(64, 1, dtype=torch.bool)},
        )
        objective = ChunkBCObjective()
        initial = float(objective.compute_loss(model, batch).loss)
        learner = SupervisedLearner(
            model, objective, torch.optim.Adam(model.parameters(), lr=0.01)
        )
        for _ in range(100):
            learner.update(batch)
        assert float(objective.compute_loss(model, batch).loss) < initial * 0.05
        restored = ChunkBCModel(deepcopy(model.cfg))
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(history), model(history), atol=0, rtol=0)
        incompatible = ChunkBCModel(
            ChunkBCModelCfg(
                observation_dim=2,
                target_dim=1,
                hidden_dims=(16,),
                target_semantics="generalized_effort",
            )
        )
        other = SupervisedLearner(
            incompatible,
            objective,
            torch.optim.Adam(incompatible.parameters(), lr=0.01),
        )
        before = deepcopy(incompatible.state_dict())
        with pytest.raises(ValueError, match="metadata"):
            other.load_state_dict(learner.state_dict())
        for name, value in before.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    incompatible.state_dict()[name], value, atol=0, rtol=0
                )
    finally:
        torch.set_num_threads(previous)


def test_executor_updates_measurements_and_resets_only_selected_queues():
    model = ChunkBCModel(
        ChunkBCModelCfg(
            observation_dim=1,
            target_dim=1,
            history_length=2,
            prediction_horizon=3,
            hidden_dims=(1,),
        )
    )
    with torch.no_grad():
        model.head[0].weight.copy_(torch.tensor([[1.0, 1.0]]))
        model.head[0].bias.zero_()
        model.head[2].weight.fill_(1)
        model.head[2].bias.copy_(torch.tensor([0.0, 10.0, 20.0]))
    executor = ChunkBCExecutor(model, action_horizon=2)
    first = executor.action(torch.tensor([[1.0], [2.0]]))
    torch.testing.assert_close(first, torch.tensor([[2.0], [4.0]]))
    second = executor.action(torch.tensor([[3.0], [4.0]]))
    torch.testing.assert_close(second, torch.tensor([[12.0], [14.0]]))
    executor.reset(torch.tensor([0]))
    third = executor.action(torch.tensor([[5.0], [6.0]]))
    # Row0 starts from two current measurements; row1 replans from its actual
    # preceding observation4 and current6, including measurements while queued.
    torch.testing.assert_close(third, torch.tensor([[10.0], [10.0]]))
    executor.reset(torch.tensor([0]))
    fourth = executor.action(torch.tensor([[7.0], [8.0]]))
    torch.testing.assert_close(fourth, torch.tensor([[14.0], [20.0]]))
    torch.testing.assert_close(first, torch.tensor([[2.0], [4.0]]))


def test_nonvisual_bc_import_does_not_import_vision_or_simulation():
    code = """
import sys
from embodichain_models.policy.bc import ChunkBCModel, ChunkBCModelCfg, ChunkBCObjective
model = ChunkBCModel(ChunkBCModelCfg())
assert 'torchvision' not in sys.modules
assert not any(name == 'dexsim' or name.startswith('dexsim.') or name.startswith('embodichain.lab.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.gpu
def test_visual_bc_cuda_backward_and_target_execution():
    torch.manual_seed(31)
    model = ChunkBCModel(
        ChunkBCModelCfg(
            observation_dim=27,
            target_dim=10,
            history_length=2,
            prediction_horizon=32,
            cameras=("high", "wrist"),
        )
    ).cuda()
    history = torch.randn(1, 2, 27, device="cuda")
    images = {
        name: torch.rand(1, 2, 3, 256, 256, device="cuda") for name in model.cfg.cameras
    }
    batch = SupervisedBatch(
        {"history": history, "images": images},
        torch.randn(1, 32, 10, device="cuda"),
        {"target_valid": torch.ones(1, 32, dtype=torch.bool, device="cuda")},
    )
    learner = SupervisedLearner(
        model,
        ChunkBCObjective(),
        torch.optim.AdamW(model.parameters(), lr=3e-4),
        max_grad_norm=1,
    )
    result = learner.update(batch)
    assert result["num_updates"] == 1 and result["grad_norm"] > 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    executor = ChunkBCExecutor(model.eval(), action_horizon=8)
    targets = executor.action(
        history[:, -1], {name: value[:, -1] for name, value in images.items()}
    )
    assert (
        targets.shape == (1, 10) and targets.is_cuda and torch.isfinite(targets).all()
    )
