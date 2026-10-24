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

"""Optimizer and graph contracts shared by BC and transition fitting."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import torch
import pytest

from embodichain.learning.sl import LossResult, SupervisedBatch, SupervisedLearner


class Regression:
    def compute_loss(self, model, batch):
        error = (model(batch.inputs) - batch.targets).square()
        mask = batch.masks.get("valid", torch.ones_like(error, dtype=torch.bool))
        return LossResult(error[mask].mean(), weight=int(mask.sum()))


def test_update_matches_independent_masked_sgd_and_retains_context():
    torch.manual_seed(11)
    model = torch.nn.Linear(2, 1)
    reference = copy.deepcopy(model)
    x = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    y = torch.tensor([[0.5], [1000.0]])
    mask = torch.tensor([[True], [False]])
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    expected_loss = (reference(x[:1]) - y[:1]).square().mean()
    expected_loss.backward()
    expected_optimizer.step()
    inside = []

    @contextmanager
    def numerical_context():
        inside.append(True)
        try:
            yield
        finally:
            inside.pop()

    model.weight.register_hook(
        lambda grad: grad if inside else pytest.fail("backward escaped context")
    )
    learner = SupervisedLearner(
        model,
        Regression(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        numerical_context=numerical_context,
    )
    report = learner.update(SupervisedBatch(x, y, {"valid": mask}))
    assert report["loss"] == expected_loss.item()
    assert report["num_updates"] == 1
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_multistep_objective_preserves_action_graph_and_batch_transfer():
    class Dynamics:
        def compute_loss(self, model, batch):
            state = batch.inputs["state"]
            for action in batch.inputs["actions"].unbind(1):
                state = state + model(action)
            return LossResult((state - batch.targets).square().mean())

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.5)
    actions = torch.ones(2, 3, 1, requires_grad=True)
    batch = SupervisedBatch(
        {"state": torch.zeros(2, 1), "actions": actions}, torch.zeros(2, 1)
    ).to("cpu")
    learner = SupervisedLearner(
        model, Dynamics(), torch.optim.SGD(model.parameters(), lr=0.01)
    )
    learner.update(batch)
    torch.testing.assert_close(actions.grad, torch.full_like(actions, 0.75))


def test_nonfinite_loss_does_not_commit_update():
    class Invalid:
        def compute_loss(self, model, batch):
            return LossResult(model(batch.inputs).sum() * float("inf"))

    model = torch.nn.Linear(1, 1)
    before = copy.deepcopy(model.state_dict())
    learner = SupervisedLearner(model, Invalid(), torch.optim.Adam(model.parameters()))
    with pytest.raises((RuntimeError, FloatingPointError)):
        learner.update(SupervisedBatch(torch.ones(1, 1), None))
    assert learner.num_updates == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not learner.optimizer.state
