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

"""Heading increments, weighted autoregression and frozen-model gradients."""

from __future__ import annotations

import math

import pytest
import torch

from embodichain_models.dynamics.rwm.kinematic import (
    KinematicWorldModel,
    KinematicWorldModelCfg,
    heading_state_increment,
    integrate_heading_state,
)
from embodichain_models.dynamics.rwm.model import frozen_world_model
from embodichain_models.dynamics.rwm.robot_state import rotation_exp
from embodichain_models.dynamics.rwm.supervised import (
    RobotWorldModelObjective,
    replay_supervised_batch,
)
from embodichain.learning.models.normalizer import EmpiricalNormalizer
from embodichain.learning.sl import SupervisedBatch, SupervisedLearner

URDF = """<robot name="one"><link name="base"/><link name="arm"/>
<joint name="joint" type="revolute"><parent link="base"/><child link="arm"/>
<origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint></robot>"""


def model_and_state():
    model = KinematicWorldModel(
        URDF,
        KinematicWorldModelCfg(
            joint_names=("joint",),
            resting_position=(0.0,),
            action_dim=1,
            context_dim=0,
            hidden_dims=(16, 16),
        ),
    )
    state = torch.zeros(2, 20)
    state[:, 2] = 1
    state[:, 3:12] = torch.eye(3).flatten()
    return model, state


def test_heading_increment_moves_horizontal_and_transports_body_angular_velocity():
    _, state = model_and_state()
    # A pitched robot still translates along the ground heading axes.
    rotation = rotation_exp(torch.tensor([[0.0, math.pi / 3, 0.0]]))
    state[:, 3:12] = rotation.flatten()
    state[:, 15:18] = torch.tensor([0.0, 0.0, 1.0])
    delta = torch.zeros(2, 14)
    delta[:, 0] = 0.4
    delta[:, 5] = 0.2
    predicted = integrate_heading_state(state, delta)
    torch.testing.assert_close(
        predicted[:, :3], state[:, :3] + torch.tensor([0.4, 0.0, 0.0])
    )
    expected_rotation = rotation @ rotation_exp(delta[:, 3:6])
    expected_angular = (
        expected_rotation @ rotation.transpose(-1, -2) @ state[:, 15:18, None]
    ).squeeze(-1)
    torch.testing.assert_close(predicted[:, 15:18], expected_angular)
    torch.testing.assert_close(
        heading_state_increment(state, predicted), delta, atol=1e-6, rtol=1e-5
    )


def test_variance_epsilon_normalizer_uses_measured_moments_and_clips():
    normalizer = EmpiricalNormalizer(2, epsilon=0.5, variance_epsilon=True, clip=2)
    normalizer.update(torch.tensor([[1.0, 3.0], [3.0, 3.0]]))
    torch.testing.assert_close(normalizer.scale, torch.tensor([[math.sqrt(1.25), 0.5]]))
    torch.testing.assert_close(
        normalizer(torch.tensor([[2.0, 5.0]])), torch.tensor([[0.0, 2.0]])
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_weighted_time_sum_matches_explicit_objective_and_backward_keeps_first_action(
    compiled, monkeypatch
):
    if compiled:
        compile_fn = torch.compile
        monkeypatch.setattr(
            torch,
            "compile",
            lambda fn, **kwargs: compile_fn(fn, backend="aot_eager", fullgraph=True),
        )
    torch.manual_seed(4)
    model, state = model_and_state()
    actions = torch.tensor(
        [[[0.2], [0.3], [0.1]], [[-0.1], [0.1], [0.4]]], requires_grad=True
    )
    target = state[:, None].expand(-1, 3, -1).clone()
    target[..., 18] = 0.1
    batch = SupervisedBatch(
        {"initial_state": state, "actions": actions},
        target,
        {"valid": torch.ones(2, 3, dtype=torch.bool)},
    )
    weights = (1.0,) * 12 + (2.0, 3.0)
    objective = RobotWorldModelObjective(
        None,
        temporal_reduction="sum",
        error_weights=weights,
        compile_mode="default" if compiled else None,
    )
    with torch.device(actions.device):
        result = objective.compute_loss(model, batch)
    current, expected = state, state.new_zeros(())
    for step in range(3):
        current = model(current, actions[:, step])
        error = (
            model.state_error(current, target[:, step]) / model.output_normalizer.scale
        )
        expected = expected + (error.square() * error.new_tensor(weights)).mean()
    torch.testing.assert_close(result.loss, expected)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    learner = SupervisedLearner(model, objective, optimizer, max_grad_norm=1)
    before = [p.detach().clone() for p in model.parameters()]
    learner.update(batch)
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters())
    )
    optimizer.zero_grad(set_to_none=True)
    with frozen_world_model(model):
        current = state
        for step in range(3):
            current = model(current, actions[:, step])
        gradient = torch.autograd.grad(current[:, 18:].sum(), actions)[0]
        assert torch.isfinite(gradient).all() and gradient[:, 0].abs().sum() > 0
        with pytest.raises(RuntimeError, match="frozen"):
            model.update_normalization(state, actions[:, 0].detach(), state)
    assert all(p.grad is None for p in model.parameters())


def test_complete_window_adapter_rejects_unusable_physical_transitions():
    _, state = model_and_state()
    windows = {
        "state": state[:, None].repeat(1, 2, 1),
        "next_state": state[:, None].repeat(1, 2, 1),
        "action": torch.zeros(2, 2, 1),
        "context": torch.zeros(2, 2, 0),
        "valid": torch.ones(2, 2, dtype=torch.bool),
        "usable": torch.tensor([[True, False], [True, True]]),
    }
    batch = replay_supervised_batch(windows, complete_windows=True)
    assert batch.targets.shape[0] == 1
    windows["usable"][:] = False
    with pytest.raises(ValueError, match="No complete"):
        replay_supervised_batch(windows, complete_windows=True)


def test_checkpoint_rejects_different_kinematic_geometry(tmp_path):
    model, _ = model_and_state()
    path = tmp_path / "world-model.pt"
    torch.save({"learner": {"model": model.state_dict()}}, path)
    state = torch.load(path, weights_only=True)["learner"]["model"]
    restored, _ = model_and_state()
    restored.load_state_dict(state)
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    other = KinematicWorldModel(URDF.replace('xyz="1 0 0"', 'xyz="2 0 0"'), model.cfg)
    with pytest.raises(ValueError, match="contract"):
        other.load_state_dict(state)
