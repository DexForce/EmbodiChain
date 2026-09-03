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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.sim.diff.runtime import NewtonDifferentiableTrajectory
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
            physics_cfg=NewtonPhysicsCfg(
                requires_grad=False,
                solver_cfg={"solver_type": "semi_implicit"},
                use_cuda_graph=False,
            ),
            num_envs=1,
            headless=True,
        )
    )
    sim.prepare()
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
    sim.prepare()
    stepper = sim.create_differentiable_stepper()
    from dexsim.engine.newton_physics.differentiable_stepper import (
        DifferentiableStepper,
    )

    assert isinstance(stepper, DifferentiableStepper)
    SimulationManager.reset()


def test_tape_context_records_step():
    import warp as wp

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
    sim.prepare()
    from embodichain.lab.sim.diff import tape_context

    with tape_context(sim) as tape:
        pass  # empty tape is valid; tape.backward() on empty is a no-op

    assert isinstance(tape, wp.Tape)
    SimulationManager.reset()


@pytest.mark.parametrize(
    (
        "update_interval",
        "num_substeps",
        "physics_steps",
        "collision_steps",
        "solver_contact_indices",
    ),
    [
        pytest.param(
            None,
            3,
            2,
            (0, 3),
            (0, 0, 0, 1, 1, 1),
            id="default-first-substep",
        ),
        pytest.param(
            2,
            5,
            2,
            (0, 2, 4, 5, 7, 9),
            (0, 0, 1, 1, 2, 3, 3, 4, 4, 5),
            id="every-two-substeps",
        ),
        pytest.param(
            1,
            3,
            1,
            (0, 1, 2),
            (0, 1, 2),
            id="every-substep",
        ),
    ],
)
def test_differentiable_trajectory_respects_collision_pipeline_update_interval(
    update_interval: int | None,
    num_substeps: int,
    physics_steps: int,
    collision_steps: tuple[int, ...],
    solver_contact_indices: tuple[int, ...],
) -> None:
    initial_state = object()
    states = [Mock() for _ in range(physics_steps * num_substeps + 1)]
    control = object()
    contacts = [object() for _ in collision_steps]
    collision_pipeline = Mock()
    collision_pipeline.contacts.side_effect = contacts
    solver = Mock()
    backend = SimpleNamespace(
        model=SimpleNamespace(
            state=Mock(side_effect=states),
            control=Mock(return_value=control),
        ),
        runtime=SimpleNamespace(
            current_state=initial_state,
            has_external_wrenches=False,
        ),
        collision_pipeline=collision_pipeline,
        solver=solver,
        cfg=SimpleNamespace(
            collision_pipeline_cfg=SimpleNamespace(update_interval=update_interval)
        ),
    )
    runtime = SimpleNamespace(
        num_substeps=num_substeps,
        _validated_backend=lambda: backend,
        _backend=lambda: backend,
    )

    trajectory = NewtonDifferentiableTrajectory(
        runtime,
        physics_steps=physics_steps,
        physics_dt=0.1,
    )
    final_state = trajectory.step()

    states[0].assign.assert_called_once_with(initial_state)
    assert [call.args[0] for call in collision_pipeline.collide.call_args_list] == [
        states[index] for index in collision_steps
    ]
    assert [
        call.args[1] for call in collision_pipeline.collide.call_args_list
    ] == contacts
    assert [call.args[3] for call in solver.step.call_args_list] == [
        contacts[index] for index in solver_contact_indices
    ]
    assert final_state is states[-1]
