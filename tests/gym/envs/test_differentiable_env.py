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
