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

"""Native/functional adapters delegate to the same independent NeRD transition."""

from __future__ import annotations

import pytest
import torch
import warp as wp

newton = pytest.importorskip("newton")
pytest.importorskip("dexsim")

from dexsim.engine.newton_physics import NewtonSolverContext
from embodichain_models.dynamics.nerd import NeRDState
from embodichain_solvers.physics.nerd.provider import NeRDProvider
from examples.learning.nerd._support._reference_model import make_reference_model
from examples.learning.nerd._support._reference_assets import reference_profile
from nerd_support import small_nerd_export


@pytest.fixture(params=["cartpole", "ant"])
def deployment(request, tmp_path):
    from examples.learning.nerd._resources import prepare_reference_assets

    try:
        prepare_reference_assets(local_files_only=True)
    except (OSError, ValueError):
        pytest.skip("Pinned NeRD reference assets are not prepared")
    robot = request.param
    checkpoint = small_nerd_export(tmp_path / robot, robot)
    model = make_reference_model(robot, "cpu", 2, "native")
    context = NewtonSolverContext(
        model=model,
        control=model.control(),
        contacts=None,
        device=model.device,
        dt=1 / 60,
        scene_indices={},
    )
    extension = NeRDProvider().create(context, {"checkpoint": str(checkpoint)})
    solver = extension.native_solver
    initial, output = model.state(), model.state()
    profile = reference_profile(robot)
    q = torch.tensor(profile["default_q"]).expand(2, -1).contiguous()
    qd = torch.tensor(profile["default_qd"]).expand(2, -1).contiguous()
    wp.to_torch(initial.joint_q).copy_(
        solver.binding._q_template.scatter(
            0, solver.binding.q_indices.flatten(), q.flatten()
        )
    )
    wp.to_torch(initial.joint_qd).copy_(
        solver.binding._qd_template.scatter(
            0, solver.binding.qd_indices.flatten(), qd.flatten()
        )
    )
    newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    try:
        yield robot, context, extension, initial, output
    finally:
        extension.close()


def test_native_functional_and_direct_transition_agree(deployment):
    robot, context, extension, initial, output = deployment
    solver = extension.native_solver
    functional = extension.functional.create_state(initial)
    q, qd = solver.binding.read(functional.physical)
    force = torch.zeros(context.model.joint_dof_count)
    actuated = (
        solver.binding.qd_indices[:, :1]
        if robot == "cartpole"
        else solver.binding.qd_indices[:, 6:]
    )
    force[actuated] = 0.02
    canonical_effort = force[solver.binding.qd_indices]
    if robot == "ant":
        canonical_effort = canonical_effort[:, 6:]
    direct = solver.transition.step(
        NeRDState(q, qd, functional.history, functional.valid_history_lengths),
        canonical_effort,
    )
    result = extension.functional.step(functional, {"joint_f": force}, {}, 1 / 60)
    wp.to_torch(context.control.joint_f).copy_(force)
    solver.step(initial, output, context.control, None, 1 / 60)
    actual_q, actual_qd = solver.binding.read(
        {
            "joint_q": wp.to_torch(output.joint_q),
            "joint_qd": wp.to_torch(output.joint_qd),
        }
    )
    torch.testing.assert_close(actual_q, direct.q)
    torch.testing.assert_close(actual_qd, direct.qd)
    torch.testing.assert_close(result.physical["joint_q"], wp.to_torch(output.joint_q))
    torch.testing.assert_close(result.physical["body_q"], wp.to_torch(output.body_q))
    for key in direct.history:
        torch.testing.assert_close(solver._history[key], direct.history[key])


def test_native_reset_does_not_overwrite_retained_functional_graph(deployment):
    robot, context, extension, initial, output = deployment
    solver = extension.native_solver
    functional = extension.functional.create_state(initial)
    q = functional.physical["joint_q"].detach().clone().requires_grad_()
    functional.physical["joint_q"] = q
    force = torch.zeros(context.model.joint_dof_count, requires_grad=True)
    retained = extension.functional.step(functional, {"joint_f": force}, {}, 1 / 60)
    before = retained.physical["joint_q"].detach().clone()
    extension.functional.commit(retained)
    solver.step(initial, output, context.control, None, 1 / 60)
    solver.reset(output, torch.tensor([True, False, False]))
    assert solver._lengths.tolist() == [0, 2]
    torch.testing.assert_close(retained.physical["joint_q"], before)
    loss = (
        retained.physical["joint_q"].square().sum()
        + retained.physical["joint_qd"].square().sum()
    )
    gradients = torch.autograd.grad(loss, (q, force))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert gradients[1].abs().sum() > 0
    fresh = extension.functional.create_state(initial)
    history, lengths = extension.functional.reset_history(retained, fresh, None)
    fresh = type(fresh)(fresh.physical, history, lengths)
    next_force = torch.zeros_like(force, requires_grad=True)
    independent = extension.functional.step(fresh, {"joint_f": next_force}, {}, 1 / 60)
    independent.physical["joint_q"].square().sum().backward()
    assert next_force.grad is not None and torch.isfinite(next_force.grad).all()


def test_wrong_interval_and_unmodeled_base_force_fail_without_output_mutation(
    deployment,
):
    robot, context, extension, initial, output = deployment
    solver = extension.native_solver
    before = wp.to_torch(output.joint_q).clone()
    with pytest.raises(ValueError, match="trained dt"):
        solver.step(initial, output, context.control, None, 1 / 120)
    torch.testing.assert_close(wp.to_torch(output.joint_q), before)
    if robot == "ant":
        wp.to_torch(context.control.joint_f)[solver.binding.qd_indices[0, 0]] = 1
        with pytest.raises(ValueError, match="free base"):
            solver.step(initial, output, context.control, None, 1 / 60)
