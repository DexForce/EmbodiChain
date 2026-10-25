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

"""Functional facade tests using a real CPU Newton model and installed provider."""

from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
import warp as wp
from dexsim.engine.newton_physics import NewtonCfg, RegisteredNewtonSolverCfg
from dexsim.engine.newton_physics.newton_backend import NewtonBackend
from dexsim.scene import Scene
from embodichain.lab.sim.diff.dynamics import FunctionalDynamics, detach_dynamics_state

_PROVIDER = """
import torch
import warp as wp
from newton.solvers import SolverBase
from dexsim.engine.newton_physics import FunctionalSolverState, NewtonSolverCapabilities, NewtonSolverExtension
class Native(SolverBase):
    def __init__(self, model):
        super().__init__(model)
        self.memory = torch.zeros(model.world_count)
        self.lengths = torch.zeros(model.world_count,dtype=torch.long)
    def step(self,state_in,state_out,control,contacts,dt):
        raise AssertionError('Native solver must not be stepped by functional env')
    def reset(self,state,world_mask=None,flags=None):
        mask=torch.ones_like(self.lengths,dtype=torch.bool) if world_mask is None else wp.to_torch(world_mask)[:-1]
        self.memory=torch.where(mask,0,self.memory)
        self.lengths=torch.where(mask,0,self.lengths)
class Provider:
    def register_custom_attributes(self,builder,options):
        pass
    def create(self,context,options):
        native=Native(context.model)
        gain=torch.tensor(2.,requires_grad=True)
        def initial(state):
            physical={k:wp.to_torch(getattr(state,k)).clone() for k in ('joint_q','joint_qd','body_q','body_qd')}
            return FunctionalSolverState(physical,{'memory':native.memory.clone()},native.lengths.clone())
        def transition(state,control,parameters,dt):
            q=state.physical['joint_q']+control*parameters['gain']*dt
            velocity=control*parameters['gain']
            body=torch.cat((q[:,None],state.physical['body_q'][:,1:]),-1)
            body_velocity=torch.cat((velocity[:,None],torch.zeros_like(state.physical['body_qd'][:,1:])), -1)
            return FunctionalSolverState({'joint_q':q,'joint_qd':velocity,'body_q':body,'body_qd':body_velocity},{'memory':state.history['memory']+control*dt},state.valid_history_lengths+1)
        def commit(state):
            native.memory=state.history['memory'].clone()
            native.lengths=state.valid_history_lengths.clone()
        def reset_history(state,fresh,world_mask):
            mask=torch.ones_like(state.valid_history_lengths,dtype=torch.bool) if world_mask is None else wp.to_torch(world_mask)[:-1]
            return {'memory':torch.where(mask,torch.zeros_like(state.history['memory']),state.history['memory'])},torch.where(mask,0,state.valid_history_lengths)
        return NewtonSolverExtension(native_solver=native,capabilities=NewtonSolverCapabilities(differentiable=True,reset=True),functional_step=transition,initial_functional_state=initial,named_parameters=lambda:{'gain':gain},commit_functional_state=commit,reset_functional_history=reset_history)
"""


class _SceneManager:
    """Renderer-free manager shell; physics, ownership merging and time are real."""

    def __init__(self, backend: NewtonBackend) -> None:
        self.device = torch.device("cpu")
        self.backend = backend
        self.scene = Scene("newton", SimpleNamespace(_newton_backend=backend))
        self.published_steps = 0
        self.elapsed = 0.0

    def create_functional_state(self):
        return self.scene.create_functional_state()

    def functional_parameters(self):
        return self.scene.functional_parameters()

    def functional_step(
        self,
        state,
        control,
        parameters=None,
        dt=None,
        *,
        publish=True,
        render_final_step=True,
    ):
        assert render_final_step is False
        result = self.scene.functional_step(
            state, control, parameters, dt, publish=publish
        )
        if publish:
            self.published_steps += 1
            self.elapsed += self.backend.cfg.dt if dt is None else dt
        return result

    def reset_functional_state(self, state, world_mask=None):
        # This shell has no native arena registry: exercise the public backend
        # counterpart with its actual Newton ownership map for selective rows.
        selected = (
            None
            if world_mask is None
            else wp.from_torch(
                torch.cat((world_mask, world_mask.new_zeros(1))), dtype=wp.bool
            )
        )
        fresh = self.scene.create_functional_state()
        return self.backend.reset_functional_state(state, selected, reset_state=fresh)

    def render_frame(self):
        return nullcontext()


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    info = tmp_path / "functional_dynamics_fixture-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: functional-dynamics-fixture\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text(
        "[dexsim.newton_solvers]\nfunctional_fixture = functional_provider_fixture:Provider\n"
    )
    (tmp_path / "functional_provider_fixture.py").write_text(_PROVIDER)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "functional_provider_fixture", raising=False)
    cfg = NewtonCfg(
        device="cpu",
        dt=0.1,
        num_substeps=2,
        requires_grad=True,
        sync_to_dexsim=False,
        collision_pipeline_cfg=None,
        solver_cfg=RegisteredNewtonSolverCfg("functional_fixture"),
    )
    cfg.use_cuda_graph = False
    backend = NewtonBackend(cfg)
    from dexsim.engine.newton_physics._runtime import create_model_builder

    source = create_model_builder()
    body = source.add_link(mass=1.0)
    joint = source.add_joint_prismatic(-1, body, axis=wp.vec3(1.0, 0.0, 0.0))
    source.add_articulation([joint])
    source.add_shape_sphere(body, radius=0.1)
    for _ in range(2):
        backend.builder.add_world(source)
    backend.finalize()
    try:
        yield _SceneManager(backend)
    finally:
        backend.close()


def test_macro_steps_do_not_double_solver_substeps_or_time(manager) -> None:
    dynamics = FunctionalDynamics(manager)
    state = dynamics.create_state()
    action = torch.tensor([1.0, 2.0], requires_grad=True)
    out = dynamics.step(state, action, physics_steps=3, physics_dt=0.1)
    torch.testing.assert_close(out.physical["joint_q"], torch.tensor([0.6, 1.2]))
    assert manager.backend.sim_time == pytest.approx(0.3)
    assert manager.backend.step_index == 3 and manager.published_steps == 3
    assert out.valid_history_lengths.tolist() == [6, 6]
    out.physical["joint_q"].sum().backward()
    torch.testing.assert_close(action.grad, torch.tensor([0.6, 0.6]))
    torch.testing.assert_close(dynamics.parameters()["gain"].grad, torch.tensor(0.9))
    # A later native overwrite cannot alter retained observations or old graphs.
    manager.backend.state_0.joint_q.zero_()
    torch.testing.assert_close(out.physical["joint_q"], torch.tensor([0.6, 1.2]))


def test_isolated_rollout_and_detach_do_not_mutate_live_state(manager) -> None:
    dynamics = FunctionalDynamics(manager)
    action = torch.ones(2, requires_grad=True)
    result = dynamics.step(
        dynamics.create_state(), action, physics_steps=2, publish=False
    )
    assert manager.backend.step_index == 0 and manager.elapsed == 0.0
    assert manager.backend.solver.lengths.tolist() == [0, 0]
    saved = result.physical["joint_q"]
    detached = detach_dynamics_state(result)
    assert saved.requires_grad and not detached.physical["joint_q"].requires_grad
    assert not detached.history["memory"].requires_grad
    torch.testing.assert_close(saved, detached.physical["joint_q"])


def test_partial_reset_preserves_other_physical_and_history_gradients(manager) -> None:
    dynamics = FunctionalDynamics(manager)
    action = torch.tensor([1.0, 2.0], requires_grad=True)
    state = dynamics.step(dynamics.create_state(), action)
    # Normal manager reset occurred first, including native provider reset.
    manager.backend.reset_state(
        wp.array([True, False, False], dtype=wp.bool, device="cpu")
    )
    merged = dynamics.reset(state, torch.tensor([True, False]))
    assert manager.backend.step_index == 1
    torch.testing.assert_close(merged.physical["joint_q"], torch.tensor([0.0, 0.4]))
    assert merged.valid_history_lengths.tolist() == [0, 2]
    loss = merged.physical["joint_q"].sum() + merged.history["memory"].sum()
    loss.backward()
    torch.testing.assert_close(action.grad, torch.tensor([0.0, 0.3]))


def test_invalid_macro_steps_fail_before_stepping(manager) -> None:
    dynamics = FunctionalDynamics(manager)
    state = dynamics.create_state()
    for count in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="positive integer"):
            dynamics.step(state, torch.ones(2), physics_steps=count)
    assert manager.backend.step_index == 0
