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

"""Independent geometry, multistep gradients and immutable rollout ownership."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from embodichain.extensions.models.dynamics.nerd import (
    NeRDGeometry,
    NeRDModel,
    NeRDModelCfg,
    NeRDStateCodec,
    NeRDTransition,
)
from embodichain.extensions.models.dynamics.nerd.codec import reset_history


def _transition(robot: str, batch: int = 2):
    torch.manual_seed(822)
    dtype = torch.float64
    codec = NeRDStateCodec(robot)
    gravity = torch.tensor([[0.0, 0.0, -9.81]], dtype=dtype).expand(batch, -1)
    calls = []

    def fk(q, qd):
        calls.append(None)
        shift = torch.stack((q[:, 7].sin() * 0.1, q[:, 7] * 0, q[:, 8] * 0.02), -1)
        return torch.cat((q[:, :3] + shift, q[:, 3:7]), -1), qd[:, :6]

    kwargs = {}
    dims = {"states_embedding": 4, "joint_f": 2}
    if robot == "ant":
        kwargs = dict(
            fk=fk,
            contact_bodies=torch.arange(batch)[:, None].expand(-1, 25),
            local_points=torch.zeros(batch, 25, 3, dtype=dtype),
            thickness=torch.full((batch, 25), 0.05, dtype=dtype),
        )
        dims = {
            "states_embedding": 29,
            "contact_normals": 75,
            "contact_points_1": 75,
            "contact_depths": 25,
            "joint_f": 8,
            "gravity_dir": 3,
        }
    geometry = NeRDGeometry(gravity.requires_grad_(), **kwargs)
    model = NeRDModel(
        NeRDModelCfg(
            input_dims=dims,
            output_dim=codec.q_dim + codec.qd_dim,
            n_embd=12,
            n_head=3,
            n_layer=1,
            history_length=3,
            block_size=3,
            head_hidden_dims=(8,),
            attention_backend="math",
        )
    ).double()
    transition = NeRDTransition(model, codec, geometry, 1 / 60)
    q = torch.zeros(batch, codec.q_dim, dtype=dtype)
    if robot == "ant":
        q[:, 2] = 0.05
        q[:, 6] = 1.0
    return (
        transition,
        q.requires_grad_(),
        torch.zeros(batch, codec.qd_dim, dtype=dtype, requires_grad=True),
        calls,
    )


def test_contact_projection_matches_known_rotated_slot_and_owns_fixed_inputs():
    gravity = torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float64)
    local = torch.tensor([[[0.1, 0.0, 0.0]]], dtype=torch.float64)
    geometry = NeRDGeometry(
        gravity,
        fk=lambda q, qd: (q, qd),
        contact_bodies=torch.tensor([[0]]),
        local_points=local,
        thickness=torch.tensor([[0.05]], dtype=torch.float64),
    )
    local.fill_(10)
    angle = torch.pi / 4
    pose = torch.tensor([[0.2, 0.3, 0.03, 0, 0, 0, 0]], dtype=torch.float64)
    pose[:, 5] = torch.sin(torch.tensor(angle))
    pose[:, 6] = torch.cos(torch.tensor(angle))
    result = geometry.contacts_from_poses(pose)
    torch.testing.assert_close(
        result["contact_points_1"],
        torch.tensor([[0.2, 0.4, 0.0]], dtype=torch.float64),
        atol=1e-8,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        result["contact_depths"], torch.tensor([[0.03]], dtype=torch.float64)
    )
    assert result["contact_masks"].tolist() == [[True]]
    assert geometry.local_points[0, 0, 0] == 0.1
    assert not geometry.gravity_dir.requires_grad


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_multistep_rollout_keeps_state_history_and_control_gradients(robot):
    transition, q, qd, calls = _transition(robot)
    state = initial = transition.initial_state(q, qd)
    controls = 2 if robot == "cartpole" else 8
    actions = [
        torch.full((2, controls), 0.02, dtype=q.dtype, requires_grad=True)
        for _ in range(4)
    ]
    for action in actions:
        state = transition.step(state, action)
    loss = state.q.square().sum() + state.qd.square().sum()
    gradients = torch.autograd.grad(loss, (q, qd, *actions))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert len(calls) == (4 if robot == "ant" else 0)
    assert initial.valid_lengths.tolist() == [0, 0]
    assert all(torch.count_nonzero(x) == 0 for x in initial.history.values())
    assert state.valid_lengths.tolist() == [3, 3]


def test_initial_state_contact_path_matches_directional_finite_difference():
    transition, q, qd, _ = _transition("ant")
    effort = torch.full((2, 8), 0.02, dtype=q.dtype)
    direction = torch.zeros_like(q)
    direction[:, 2] = 0.4
    direction[:, 7] = 0.3
    direction[:, 8] = 0.2

    def evaluate(position):
        result = transition.step(transition.initial_state(position, qd), effort)
        return result.q.square().sum() + result.qd.square().sum()

    derivative = (torch.autograd.grad(evaluate(q), q)[0] * direction).sum()
    epsilon = 1e-5
    expected = (
        evaluate(q.detach() + epsilon * direction)
        - evaluate(q.detach() - epsilon * direction)
    ) / (2 * epsilon)
    torch.testing.assert_close(derivative, expected, atol=1e-6, rtol=2e-5)


def test_detach_is_the_only_recurrent_gradient_boundary():
    transition, q, qd, _ = _transition("ant")
    first_action = torch.full((2, 8), 0.02, dtype=q.dtype, requires_grad=True)
    second_action = first_action.detach().clone().requires_grad_()
    first = transition.step(transition.initial_state(q, qd), first_action)
    saved = first.q.detach().clone()
    second = transition.step(first.detach(), second_action)
    gradients = torch.autograd.grad(
        second.q.sum(), (first_action, second_action), allow_unused=True
    )
    assert gradients[0] is None
    assert gradients[1] is not None and gradients[1].abs().sum() > 0
    torch.testing.assert_close(first.q, saved)
    first.q.sum().backward()
    assert first_action.grad is not None


def test_fixed_geometry_never_reuses_a_consumed_autograd_graph():
    transition, q, qd, _ = _transition("ant")
    for _ in range(2):
        action = torch.full((2, 8), 0.02, dtype=q.dtype, requires_grad=True)
        state = transition.initial_state(q.detach(), qd.detach())
        result = transition.step(state, action)
        result.q.sum().backward()
        assert action.grad is not None and torch.isfinite(action.grad).all()


def test_full_history_reset_does_not_reenter_consumed_episode_graph():
    transition, q, qd, _ = _transition("cartpole")
    state = transition.initial_state(q.detach(), qd.detach())
    for _ in range(2):
        action = torch.full((2, 2), 0.02, dtype=q.dtype, requires_grad=True)
        result = transition.step(state, action)
        result.q.square().sum().backward()
        history, lengths = reset_history(
            result.history, result.valid_lengths, torch.ones(2, dtype=torch.bool)
        )
        assert all(not value.requires_grad for value in history.values())
        state = type(result)(q.detach(), qd.detach(), history, lengths)


def test_frozen_network_accepts_explicit_parameter_gradients():
    transition, q, qd, _ = _transition("cartpole")
    transition.network.requires_grad_(False)
    parameters = {
        name: value.detach().clone().requires_grad_()
        for name, value in transition.network.named_parameters()
    }
    result = transition.step(
        transition.initial_state(q, qd),
        torch.ones(2, 2, dtype=q.dtype),
        parameters=parameters,
    )
    result.q.sum().backward()
    assert any(
        value.grad is not None and value.grad.abs().sum() > 0
        for value in parameters.values()
    )
    assert all(value.grad is None for value in transition.network.parameters())


def test_wrong_timestep_fails_before_geometry_execution():
    transition, q, qd, calls = _transition("ant")
    state = transition.initial_state(q, qd)
    with pytest.raises(ValueError, match="trained dt"):
        transition.step(state, torch.zeros(2, 8, dtype=q.dtype), dt=1 / 120)
    assert not calls
    assert state.valid_lengths.tolist() == [0, 0]


def test_model_layer_imports_without_native_backends():
    root = Path(__file__).resolve().parents[4]
    script = f"""
import importlib.abc,sys
sys.path[:0] = {[str(root), str(root / 'embodichain/extensions/models')]!r}
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'newton','dexsim','warp'}} or fullname.startswith('embodichain.lab'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockNative())
from embodichain.extensions.models.dynamics.nerd import NeRDGeometry,NeRDState,NeRDTransition
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
