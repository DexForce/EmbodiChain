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
"""Independent numerical expectations for the packaged spring-slider provider."""

from __future__ import annotations

import math

import newton
import numpy as np
import pytest
import torch
import warp as wp

from dexsim.engine.newton_physics import NewtonSolverContext, resolve_newton_solver
from embodichain.lab.sim.cfg import NewtonPhysicsCfg


def _model(device="cpu", *, joint="prismatic", collision=False, gravity=(0, 0, 0)):
    builder = newton.ModelBuilder(gravity=gravity)
    body = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)))
    rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi / 2)
    add_joint = getattr(builder, "add_joint_" + joint)
    index = add_joint(
        -1, body, axis=(1, 0, 0), parent_xform=wp.transform((1, 2, 3), rotation)
    )
    builder.add_articulation([index])
    if collision:
        builder.add_shape_sphere(body=body, radius=0.1)
    return builder.finalize(device)


def _extension(model, **options):
    provider = resolve_newton_solver("simple")
    return provider.create(
        NewtonSolverContext(model, model.control(), None, model.device, 0.1, {}),
        options,
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_discovered_solver_native_parity_and_analytical_gradients(device):
    model = _model(device)
    extension = _extension(model, stiffness=0, damping=0)
    dynamics = extension.functional
    native = model.state()
    newton.eval_fk(model, native.joint_q, native.joint_qd, native)
    placed = wp.to_torch(native.body_q).clone()
    placed[:, 0] += 5.0
    native.body_q.assign(placed.detach().cpu().numpy())
    initial = dynamics.create_state(native)
    force = torch.tensor([4.0], device=device, requires_grad=True)
    parameters = dynamics.parameters()
    first = dynamics.step(initial, force, parameters, 0.1)
    second = dynamics.step(first, force, parameters, 0.1)
    # m=2, F=4: v1=.2, q1=.02; v2=.4, q2=.06.
    torch.testing.assert_close(
        second.physical["joint_q"], torch.tensor([0.06], device=device)
    )
    torch.testing.assert_close(
        second.physical["body_q"][0, :3], torch.tensor([6.0, 2.06, 3.0], device=device)
    )
    grad_force, grad_scale = torch.autograd.grad(
        second.physical["joint_q"].sum(), (force, parameters["force_scale"])
    )
    torch.testing.assert_close(grad_force, torch.tensor([0.015], device=device))
    torch.testing.assert_close(grad_scale, torch.tensor(0.06, device=device))
    control, output = model.control(), model.state()
    control.joint_f.assign([4.0])
    extension.native_solver.step(native, output, control, None, 0.1)
    extension.native_solver.step(output, native, control, None, 0.1)
    wp.synchronize_device(model.device)
    torch.testing.assert_close(wp.to_torch(native.joint_q), second.physical["joint_q"])
    torch.testing.assert_close(wp.to_torch(native.body_qd), second.physical["body_qd"])
    torch.testing.assert_close(
        initial.physical["joint_q"], torch.zeros(1, device=device)
    )
    torch.testing.assert_close(
        first.physical["joint_q"], torch.tensor([0.02], device=device)
    )


def test_spring_damping_limits_and_mass_refresh():
    model = _model()
    extension = _extension(model, stiffness=2, damping=1)
    native = model.state()
    native.joint_q.assign([0.2])
    native.joint_qd.assign([0.3])
    service = extension.functional
    state = service.create_state(native)
    force = torch.tensor([4.0])
    out = service.step(state, force, service.parameters(), 0.1)
    # a=(4-2*.2-1*.3)/2=1.65; v=.465, q=.2465.
    torch.testing.assert_close(out.physical["joint_q"], torch.tensor([0.2465]))
    model.body_mass.assign([4.0])
    model.joint_limit_upper.assign([0.23])
    extension.native_solver.notify_model_changed(
        newton.ModelFlags.BODY_INERTIAL_PROPERTIES
    )
    out = service.step(state, force, service.parameters(), 0.1)
    torch.testing.assert_close(out.physical["joint_q"], torch.tensor([0.23]))
    torch.testing.assert_close(out.physical["joint_qd"], torch.tensor([0.0]))
    # Move the limit out again to observe the changed mass rather than only clamp.
    model.joint_limit_upper.assign([1.0])
    extension.native_solver.notify_model_changed(
        newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    )
    out = service.step(state, force, service.parameters(), 0.1)
    torch.testing.assert_close(out.physical["joint_q"], torch.tensor([0.23825]))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"joint": "revolute"}, "prismatic leaf"),
        ({"collision": True}, "collision shapes"),
        ({"gravity": (0, 0, -9.81)}, "zero gravity"),
    ],
)
def test_unsupported_model_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _extension(_model(**kwargs))


def test_named_config_and_independent_provider_parameters():
    cfg = NewtonPhysicsCfg(
        device="cpu",
        requires_grad=True,
        solver_cfg={"solver_type": "simple", "stiffness": 3.0},
    ).to_dexsim_cfg(gpu_id=0)
    assert cfg.solver_cfg.solver_type == "simple"
    assert cfg.solver_cfg.options == {"stiffness": 3.0}
    first, second = _extension(_model()), _extension(_model())
    with torch.no_grad():
        first.functional.parameters()["force_scale"].fill_(2)
    assert second.functional.parameters()["force_scale"].item() == 1


@pytest.mark.requires_sim
@pytest.mark.parametrize("owner", ["articulation", "manager"])
def test_effort_only_reset_restores_selected_environment(tmp_path, owner):
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import ArticulationCfg

    asset = tmp_path / "slider.urdf"
    asset.write_text("""<robot name="slider">
      <link name="base"><inertial><mass value="1"/><inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/></inertial></link>
      <link name="slider"><inertial><mass value="2"/><inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/></inertial></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="slider"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="100" velocity="10"/></joint>
    </robot>""")
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            enable_entity_gizmo=False,
            startup_summary="off",
            physics_cfg=NewtonPhysicsCfg(
                device="cpu",
                gravity=(0, 0, 0),
                requires_grad=True,
                collision_cfg=None,
                solver_cfg={"solver_type": "simple"},
            ),
        ),
        requires_native_renderer=False,
    )
    art = None
    try:
        art = sim.add_articulation(ArticulationCfg(uid="slider", fpath=str(asset)))
        sim.prepare()
        art.set_qpos(torch.tensor([[0.2], [0.4]]), target=False)
        art.set_qvel(torch.tensor([[0.3], [0.5]]), target=False)
        with pytest.raises(NotImplementedError, match="joint_target_q"):
            art.set_qpos(torch.zeros(2, 1), target=True)
        if owner == "articulation":
            art.reset(env_ids=[0])
        else:
            sim.reset_objects_state(env_ids=[0])
        torch.testing.assert_close(art.get_qpos(), torch.tensor([[0.0], [0.4]]))
        torch.testing.assert_close(art.get_qvel(), torch.tensor([[0.0], [0.5]]))
        torch.testing.assert_close(art.get_qpos(target=True)[0], torch.zeros(1))
    finally:
        art = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
