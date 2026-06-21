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
"""Tests for the differentiable-stepper delegators on SimulationManager."""

from __future__ import annotations

import pytest

from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg


def test_default_backend_rejects_differentiable_stepper():
    sim = SimulationManager(
        SimulationManagerCfg(
            physics_cfg=DefaultPhysicsCfg(),
            num_envs=1,
            headless=True,
        )
    )
    with pytest.raises(Exception, match=r"Newton"):
        sim.create_differentiable_stepper()
    SimulationManager.reset()


def test_newton_without_grad_rejects_differentiable_stepper():
    sim = SimulationManager(
        SimulationManagerCfg(
            physics_cfg=NewtonPhysicsCfg(requires_grad=False, use_cuda_graph=False),
            num_envs=1,
            headless=True,
        )
    )
    sim.finalize_newton_physics()
    with pytest.raises(Exception, match=r"grad"):
        sim.create_differentiable_stepper()
    SimulationManager.reset()


def test_newton_with_grad_creates_stepper():
    sim = SimulationManager(
        SimulationManagerCfg(
            physics_cfg=NewtonPhysicsCfg(
                requires_grad=True,
                solver_cfg={"solver_type": "semi_implicit"},
                use_cuda_graph=False,
            ),
            num_envs=1,
            headless=True,
        )
    )
    sim.finalize_newton_physics()
    stepper = sim.create_differentiable_stepper()
    from dexsim.engine.newton_physics.differentiable_stepper import (
        DifferentiableStepper,
    )

    assert isinstance(stepper, DifferentiableStepper)
    SimulationManager.reset()
