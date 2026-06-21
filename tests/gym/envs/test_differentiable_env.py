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
"""Tests for DifferentiableEmbodiedEnv."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.gym.envs.differentiable_env import (
    DifferentiableEmbodiedEnv,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.sim.sim_manager import SimulationManagerCfg


def _diff_env_cfg(
    requires_grad: bool = True, backend: str = "newton"
) -> EmbodiedEnvCfg:
    if backend == "newton":
        physics_cfg = NewtonPhysicsCfg(
            requires_grad=requires_grad,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        )
    else:
        physics_cfg = DefaultPhysicsCfg()
    sim_cfg = SimulationManagerCfg(
        physics_cfg=physics_cfg,
        num_envs=2,
        headless=True,
    )
    return EmbodiedEnvCfg(sim_cfg=sim_cfg)


def test_construct_without_requires_grad_raises():
    with pytest.raises(Exception, match=r"requires_grad"):
        DifferentiableEmbodiedEnv(_diff_env_cfg(requires_grad=False))


def test_construct_on_default_backend_raises():
    with pytest.raises(Exception, match=r"Newton"):
        DifferentiableEmbodiedEnv(_diff_env_cfg(backend="default"))


def _import_franka_env():
    """Import the Franka APG env, skipping if the URDF is unavailable.

    The URDF resolves through ``newton.utils.download_asset`` which
    requires network access on first run. Tests skip cleanly when the
    asset cannot be fetched.
    """
    from embodichain.lab.gym.envs.tasks.special.franka_reach_apg import (
        FrankaReachApgEnv,
    )

    return FrankaReachApgEnv


def test_franka_apg_smoke_backward():
    """Verify reward is autograd-tracked and action.grad flows back."""
    try:
        FrankaReachApgEnv = _import_franka_env()
    except FileNotFoundError as e:
        pytest.skip(f"Franka URDF not available: {e}")

    env = FrankaReachApgEnv(num_envs=2)
    env.reset(seed=0)
    action = torch.zeros(2, 7, requires_grad=True, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward.requires_grad, "Reward must be autograd-tracked."
    loss = reward.sum()
    loss.backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()


def test_franka_apg_one_iter_loss_reduces():
    """Verify a single SGD step reduces the APG loss."""
    try:
        FrankaReachApgEnv = _import_franka_env()
    except FileNotFoundError as e:
        pytest.skip(f"Franka URDF not available: {e}")

    env = FrankaReachApgEnv(num_envs=2)
    env.reset(seed=0)
    action = torch.zeros(2, 7, requires_grad=True, device=env.device)
    opt = torch.optim.SGD([action], lr=0.01)

    losses = []
    for _ in range(3):
        env.reset(seed=0)
        opt.zero_grad()
        _, reward, _, _, _ = env.step(action)
        loss = (-reward).sum()
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    assert losses[-1] < losses[0], f"APG did not reduce loss: {losses}"
