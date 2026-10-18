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

"""Training module lifecycle and full-rollout gradient contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from embodichain.compute.kinematics import DifferentiableIKCfg, SerialChainKinematics
from embodichain.learning.layers import DifferentiableIKLayer, KinematicsLayer


def make_layer(dtype=torch.float64, device="cpu", iterations=4, enforce=True):
    xml = (
        Path(__file__).resolve().parents[2] / "fixtures/kinematics/seven_dof.urdf"
    ).read_text()
    model = SerialChainKinematics.from_urdf(xml, "link6", dtype=dtype, device=device)
    tcp = torch.eye(4, device=device, dtype=dtype)
    tcp[:3, 3] = tcp.new_tensor([0.1, 0.02, -0.03])
    return DifferentiableIKLayer(
        KinematicsLayer(model, tcp),
        DifferentiableIKCfg(max_iterations=iterations, enforce_joint_limits=enforce),
    )


def test_import_boundary():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if any(fullname == n or fullname.startswith(n + '.') for n in ('embodichain.lab', 'embodichain.learning.rl', 'dexsim', 'warp', 'wandb', 'tensordict')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from embodichain.learning.layers import KinematicsLayer, DifferentiableIKLayer
from embodichain.learning.models import MLP, EmpiricalNormalizer, build_mlp_from_cfg
network = build_mlp_from_cfg({"type": "mlp", "network_cfg": {"hidden_sizes": [4]}}, 3, 2)
assert isinstance(network, MLP)
assert network.__class__.__module__ == "embodichain.learning.models.mlp"
assert not hasattr(__import__("embodichain.learning.models", fromlist=["Policy"]), "Policy")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_arbitrary_batches_and_unreachable(dtype):
    layer = make_layer(dtype=dtype, iterations=24)
    seed = torch.zeros(2, 3, 7, dtype=dtype)
    target = layer.kinematics(seed + 0.01)
    result = layer.solve(target, seed)
    assert result.qpos.shape == (2, 3, 7)
    assert result.success.shape == (2, 3)
    assert result.success.all()
    target[..., :3, 3] += 100
    result = layer.solve(target, seed)
    assert not result.success.any()
    assert torch.isfinite(result.qpos).all()
    assert torch.isfinite(result.position_error).all()
    assert result.position_error.norm(dim=-1).min() > 1


def test_repeated_calls_inputs_and_batch_independence():
    layer = make_layer()
    seed = torch.full((2, 7), 0.1, dtype=torch.float64, requires_grad=True)
    target = layer.kinematics(seed.detach() + 0.05).requires_grad_()
    original_seed, original_target = seed.clone(), target.clone()
    first = layer(target, seed)
    second = layer(target, seed + 0.02)
    other_grad = torch.autograd.grad(first[0].sum(), seed, retain_graph=True)[0]
    torch.testing.assert_close(other_grad[1], torch.zeros(7, dtype=seed.dtype))
    grads = torch.autograd.grad(first.sum() + second.sum(), (seed, target))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    torch.testing.assert_close(seed, original_seed)
    torch.testing.assert_close(target, original_target)
    assert first.data_ptr() != second.data_ptr()


def test_short_rollout_has_state_and_policy_gradients():
    layer = make_layer(iterations=2)
    initial = torch.full((1, 7), 0.1, dtype=torch.float64, requires_grad=True)
    action = torch.tensor(
        [[0.04, -0.03, 0.02]], dtype=torch.float64, requires_grad=True
    )
    base = layer.kinematics(initial.detach() + 0.1).detach()
    # A frozen state model still transmits derivatives to its inputs.
    dynamics = torch.nn.Linear(14, 7, bias=False, dtype=torch.float64)
    with torch.no_grad():
        dynamics.weight.copy_(torch.cat((0.8 * torch.eye(7), 0.2 * torch.eye(7)), 1))
    dynamics.requires_grad_(False)

    def rollout(seed, correction):
        top = torch.cat(
            (base[..., :3, :3], base[..., :3, 3:] + correction[..., None]), -1
        )
        target = torch.cat((top, base[..., 3:, :]), -2)
        q = seed
        for _ in range(3):
            command = layer(target, q)
            q = dynamics(torch.cat((q, command), -1))
        return layer.kinematics(q)[..., :3, 3]

    assert torch.autograd.gradcheck(
        rollout, (initial, action), eps=1e-6, atol=1e-4, rtol=1e-3, fast_mode=True
    )
    grads = torch.autograd.grad(rollout(initial, action).sum(), (initial, action))
    assert all(g.norm() > 0 and torch.isfinite(g).all() for g in grads)


def test_limits_projection_can_be_disabled():
    layer = make_layer(iterations=2)
    seed = torch.full((1, 7), 2.2, dtype=torch.float64, requires_grad=True)
    target = layer.kinematics(seed.detach())
    limited = layer.solve(target, seed)
    assert (limited.qpos <= 2).all()
    gradient = torch.autograd.grad(limited.qpos.sum(), seed)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(seed))
    layer.cfg.enforce_joint_limits = False
    free = layer.solve(target, seed)
    torch.testing.assert_close(free.qpos, seed)
    assert not free.success.any()
    assert torch.isfinite(torch.autograd.grad(free.qpos.sum(), seed)[0]).all()


def test_checkpoint_dtype_and_eval_lifecycle():
    layer = make_layer(dtype=torch.float32).double().eval()
    q = torch.full((1, 7), 0.1, dtype=torch.float64, requires_grad=True)
    target = layer.kinematics(q.detach() + 0.05)
    expected = layer(target, q)
    state = deepcopy(layer.state_dict())
    rebuilt = make_layer(dtype=torch.float64, iterations=1)
    rebuilt.load_state_dict(state)
    torch.testing.assert_close(expected, rebuilt(target, q))
    assert rebuilt.cfg.max_iterations == layer.cfg.max_iterations
    assert torch.isfinite(torch.autograd.grad(expected.sum(), q)[0]).all()
    with pytest.raises(ValueError, match="float32 and float64"):
        rebuilt.half()
    with pytest.raises(ValueError, match="geometry"):
        rebuilt.kinematics.set_extra_state({"model_fingerprint": "wrong"})


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_parity_and_no_device_synchronization(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cpu = make_layer(dtype=dtype)
    gpu = deepcopy(cpu).to("cuda")
    seed = torch.full((2, 7), 0.1, dtype=dtype, requires_grad=True)
    target = cpu.kinematics(seed.detach() + 0.05).requires_grad_()
    expected = cpu(target, seed)
    expected_grad = torch.autograd.grad(expected.sum(), (seed, target))
    q = seed.detach().cuda().requires_grad_()
    goal = target.detach().cuda().requires_grad_()
    # Warm backend handles before recording the hot path.
    gpu(goal, q).sum().backward()
    q.grad = goal.grad = None
    # PK fills homogeneous matrix constants using CPU scalar tensors. Those
    # aten::item events are harmless; reject synchronization of CUDA data.
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = gpu(goal, q)
        grads = torch.autograd.grad(actual.sum(), (q, goal))
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.testing.assert_close(expected, actual.cpu(), atol=2e-5, rtol=2e-4)
    for a, b in zip(expected_grad, grads):
        torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=2e-4)


def test_apg_import_boundary_and_model_registry():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'embodichain.lab' or fullname.startswith('embodichain.lab.') or fullname == 'dexsim':
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from embodichain.learning.rl import DifferentiableTrainer, DifferentiableTrainerCfg
from embodichain.learning.rl.algo import APG
from embodichain.learning.models import MLP
from embodichain.learning.rl.models import ActorOnly, ActorCritic, get_policy_class
from embodichain.learning.rl.models.registry import get_policy_class as registry_lookup
assert get_policy_class is registry_lookup
assert get_policy_class("actor_only") is ActorOnly
assert get_policy_class("actor_critic") is ActorCritic
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_restores_tcp_and_limits_without_geometry_mutation():
    layer = make_layer()
    before = layer.kinematics.model.chain.low.clone()
    layer.kinematics.tcp[:3, 3] = torch.tensor([0.2, 0.1, -0.15], dtype=torch.float64)
    layer.kinematics.lower.fill_(-1.0)
    layer.kinematics.upper.fill_(1.0)
    rebuilt = make_layer()
    rebuilt.load_state_dict(deepcopy(layer.state_dict()))
    q = torch.zeros(1, 7, dtype=torch.float64)
    torch.testing.assert_close(rebuilt.kinematics(q), layer.kinematics(q))
    torch.testing.assert_close(rebuilt.kinematics.lower, layer.kinematics.lower)
    torch.testing.assert_close(layer.kinematics.model.chain.low, before)
    assert not list(layer.parameters())


def test_wrong_device_precision_and_batch_are_rejected():
    layer = make_layer()
    seed = torch.zeros(2, 7, dtype=torch.float64)
    target = layer.kinematics(seed)
    with pytest.raises(ValueError, match="dtype"):
        layer(target.float(), seed)
    with pytest.raises(ValueError, match="Seed"):
        layer(target, seed[:1])
    result = layer.solve(target[0])
    assert result.qpos.shape == (7,) and result.success.shape == ()
