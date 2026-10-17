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

"""Robot adapters must preserve the shared differentiable IK contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from embodichain.compute.kinematics import DifferentiableIKCfg, SerialChainKinematics
from embodichain.lab.sim.motion.solvers import DifferentiableIKSolverCfg, SolverCfg
from embodichain.learning.layers import DifferentiableIKLayer, KinematicsLayer

pytestmark = pytest.mark.no_sim


def make_solver(dtype="float64"):
    path = Path(__file__).resolve().parents[3] / "fixtures/kinematics/seven_dof.urdf"
    return DifferentiableIKSolverCfg(
        urdf_path=str(path),
        end_link_name="link6",
        dtype=dtype,
        ik=DifferentiableIKCfg(max_iterations=4),
    ).init_solver("cpu")


def test_config_registration_and_injected_chain():
    cfg = SolverCfg.from_dict(
        {
            "class_type": "DifferentiableIKSolver",
            "dtype": "float64",
            "ik": {"max_iterations": 3},
        }
    )
    source = make_solver()
    solver = cfg.init_solver("cpu", pk_serial_chain=source.pk_serial_chain)
    assert solver.dof == 7
    assert solver.cfg.ik.max_iterations == 3
    assert solver.get_fk(torch.zeros(1, 7)).dtype == torch.float64


def test_layer_solver_parity_and_tcp_update():
    solver = make_solver()
    tcp = np.eye(4)
    tcp[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    tcp[:3, 3] = [0.1, 0.05, -0.02]
    solver.set_tcp(tcp)
    layer = DifferentiableIKLayer(
        KinematicsLayer(solver.model, torch.tensor(tcp)), solver.cfg.ik
    )
    seed = torch.full((2, 7), 0.1, dtype=torch.float64, requires_grad=True)
    goal = solver.get_fk(seed.detach() + 0.05).requires_grad_()
    actual = solver.solve(goal, seed)
    expected = layer.solve(goal, seed)
    torch.testing.assert_close(actual.qpos, expected.qpos)
    torch.testing.assert_close(
        solver.get_jacobian(seed), layer.kinematics.jacobian(seed)
    )
    actual_grad = torch.autograd.grad(
        actual.qpos.sum(), (seed, goal), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected.qpos.sum(), (seed, goal))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)
    ok, candidates = solver.get_ik(
        goal.detach(), seed.detach(), return_all_solutions=True
    )
    assert candidates.shape == (2, 1, 7) and ok.shape == (2, 1)


def test_batch_adapter_and_live_limits():
    solver = make_solver()
    seed = torch.full((2, 3, 7), 0.1, dtype=torch.float64)
    target = solver.get_fk_batch(seed)
    ok, q = solver.get_ik_batch(target, seed)
    assert ok.shape == (2, 3) and q.shape == seed.shape and ok.all()
    limits = torch.tensor([[-0.05, 0.05]] * 7, dtype=torch.float64)
    solver.update_with_robot_limit(limits)
    _, q = solver.get_ik_batch(target, seed)
    assert (q <= 0.05).all() and (q >= -0.05).all()


def test_robot_batch_facade_retains_gradients():
    from embodichain.lab.sim.objects.robot import Robot
    from types import SimpleNamespace

    solver = make_solver()
    q = torch.full((1, 2, 7), 0.1, dtype=torch.float64, requires_grad=True)
    target = solver.get_fk_batch(q.detach() + 0.01).requires_grad_()
    robot = SimpleNamespace(
        _solvers={"arm": solver},
        _all_indices=[0],
        device=torch.device("cpu"),
        get_link_pose=lambda **kwargs: torch.eye(4, dtype=torch.float32)[None],
    )
    ok, result = Robot.compute_batch_ik(robot, target, q, name="arm")
    assert result.shape == q.shape
    grads = torch.autograd.grad(result.sum(), (q, target))
    assert all(torch.isfinite(g).all() for g in grads)


def test_batched_jacobian_locations_match_independent_offsets():
    solver = make_solver()
    q = torch.full((2, 7), 0.1, dtype=torch.float64, requires_grad=True)
    offsets = torch.tensor([[0.1, 0.05, 0.02], [0.2, -0.03, 0.05]], dtype=torch.float64)
    batched = solver.get_jacobian(q, locations=offsets)
    independent = torch.cat(
        [solver.get_jacobian(q[i : i + 1], locations=offsets[i]) for i in range(2)]
    )
    torch.testing.assert_close(batched, independent)
    assert torch.isfinite(torch.autograd.grad(batched.square().sum(), q)[0]).all()
