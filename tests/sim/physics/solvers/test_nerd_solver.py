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

"""CPU NeRD integration contracts with exact assets and installed entry points.

Small seeded networks prove execution/gradient contracts, not trained accuracy.
"""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
import warp as wp
from dexsim.engine.newton_physics import (
    NewtonCfg,
    RegisteredNewtonSolverCfg,
    FunctionalSolverState,
)
from dexsim.engine.newton_physics.newton_backend import NewtonBackend
from dexsim.scene import Scene
from embodichain.lab.sim.physics.solvers._reference_assets import (
    make_reference_articulation,
    reference_profile,
)
from embodichain.lab.sim.physics.solvers._nerd_binding import NeRDBinding
from embodichain.learning.models.nerd import NeRDModel, NeRDModelCfg, export_model

DT = 1 / 60


def _source(robot: str):
    """Use production descriptor lowering, with no render allocations."""
    from dexsim.spawn.adapters.newton_articulation_adapter import (
        _translate_explicit_articulation,
    )
    from dexsim.engine.newton_physics.articulation.articulation import (
        NewtonArticulation,
    )
    from dexsim.engine.newton_physics._runtime import create_model_builder

    descriptor = make_reference_articulation(robot)
    builder = create_model_builder()
    specs, joints = _translate_explicit_articulation(
        SimpleNamespace(
            get_render_body=lambda name: None,
            get_link_names=lambda *_: [link.name for link in descriptor.links],
        ),
        None,
        descriptor,
        {},
        "mujoco_warp",
        reuse_render=True,
    )
    handle = NewtonArticulation(
        builder,
        None,
        descriptor.name,
        descriptor.root_link_name,
        specs,
        joints,
        build_device="cpu",
    )
    handle.finalize(
        descriptor.pose, fixed_base=descriptor.fixed_base, disable_self_collision=True
    )
    builder.joint_q[:] = reference_profile(robot)["default_q"]
    builder.joint_qd[:] = reference_profile(robot)["default_qd"]
    return builder


@contextmanager
def _scene(
    robot: str,
    checkpoint: Path | None = None,
    *,
    num_envs: int = 2,
    dt: float = DT,
    mutate=None,
    device: str = "cpu",
):
    cfg = NewtonCfg(
        device=device,
        dt=dt,
        num_substeps=1,
        requires_grad=checkpoint is not None,
        sync_to_dexsim=False,
        collision_pipeline_cfg=None,
        solver_cfg=(
            RegisteredNewtonSolverCfg("semi_implicit")
            if checkpoint is None
            else RegisteredNewtonSolverCfg(
                "nerd", {"checkpoint": str(checkpoint), "trainable": True}
            )
        ),
    )
    cfg.use_cuda_graph = False
    backend = NewtonBackend(cfg)
    try:
        source = _source(robot)
        for _ in range(num_envs):
            backend.builder.add_world(source)
        if robot == "ant":
            backend.builder.add_ground_plane()
        if mutate is not None:
            mutate(backend.builder)
        backend.finalize()
        yield Scene("newton", SimpleNamespace(_newton_backend=backend)), backend
    finally:
        backend.close()


def _small_export(scene, destination: Path, robot: str) -> Path:
    binding = NeRDBinding.from_metadata(dict(scene.dynamics_metadata()), robot)
    physical = scene.dynamics_snapshot().physical
    q, qd = binding.read(physical)
    from embodichain.compute.dynamics import NeRDStateCodec

    codec = NeRDStateCodec(**binding.metadata(DT)["codec"])
    features = codec.encode(
        q,
        qd,
        torch.zeros_like(qd),
        binding.contacts(physical["body_q"]),
        binding.gravity_dir,
    )
    torch.manual_seed(42)
    model = NeRDModel(
        NeRDModelCfg(
            input_dims={k: v.shape[-1] for k, v in features.items()},
            output_dim=q.shape[-1] + qd.shape[-1],
            n_embd=24,
            n_head=3,
            n_layer=2,
            history_length=3,
            block_size=4,
        )
    )
    with torch.no_grad():
        model.model.output_net.weight.mul_(0.02)
        model.model.output_net.bias.zero_()
        if robot == "ant":
            model.model.output_net.bias[6] = 1.0
    export_model(destination, model, binding.metadata(DT))
    return destination


def _export(tmp_path: Path, robot: str) -> Path:
    with _scene(robot) as (scene, _):
        return _small_export(scene, tmp_path / robot, robot)


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_native_matches_functional_and_commits_history(
    tmp_path: Path, robot: str
) -> None:
    checkpoint = _export(tmp_path, robot)
    with _scene(robot, checkpoint) as (scene, backend):
        initial = scene.create_functional_state()
        binding = backend.solver.binding
        force = torch.zeros(backend.model.joint_dof_count)
        force[binding.qd_indices[:, -1]] = 0.3
        expected = scene.functional_step(initial, {"joint_f": force}, publish=False)
        assert backend.step_index == 0
        backend.control.joint_f.assign(force.numpy())
        backend.step()
        actual = scene.create_functional_state()
        for name in expected.physical:
            torch.testing.assert_close(
                actual.physical[name], expected.physical[name], atol=1e-6, rtol=1e-5
            )
        assert actual.valid_history_lengths.tolist() == [1, 1]
        # Publish a functional transition, then native stepping must continue its
        # history rather than reusing the history before publication.
        published = scene.functional_step(actual, {"joint_f": force})
        expected_next = scene.functional_step(
            published, {"joint_f": force}, publish=False
        )
        backend.step()
        after = scene.create_functional_state()
        torch.testing.assert_close(
            after.physical["joint_q"],
            expected_next.physical["joint_q"],
            atol=1e-6,
            rtol=1e-5,
        )
        assert after.valid_history_lengths.tolist() == [3, 3]
        assert backend.step_index == 3


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_multistep_gradients_and_native_storage_independence(
    tmp_path: Path, robot: str, device: str
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    checkpoint = _export(tmp_path, robot)
    with _scene(robot, checkpoint, device=device) as (scene, backend):
        state = scene.create_functional_state()
        initial_q = state.physical["joint_q"].requires_grad_()
        initial_qd = state.physical["joint_qd"].requires_grad_()
        controls = []
        for _ in range(5):
            force = torch.zeros(
                backend.model.joint_dof_count, device=device, requires_grad=True
            )
            controls.append(force)
            state = scene.functional_step(state, {"joint_f": force}, publish=True)
        saved_q = state.physical["joint_q"].detach().clone()
        saved_body = state.physical["body_q"].detach().clone()
        backend.state_0.joint_q.zero_()
        backend.state_0.body_q.zero_()
        torch.testing.assert_close(state.physical["joint_q"], saved_q)
        torch.testing.assert_close(state.physical["body_q"], saved_body)
        loss = (
            state.physical["joint_q"].square().sum()
            + state.physical["body_q"].square().sum()
            + state.physical["body_qd"].square().sum()
        )
        loss.backward()
        for value in [initial_q, initial_qd, *controls]:
            assert value.grad is not None and torch.isfinite(value.grad).all()
            assert value.grad.abs().sum() > 0
        parameters = scene.functional_parameters()
        assert parameters and all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in parameters.values()
        )
        assert sum(p.grad.abs().sum() for p in parameters.values()) > 0


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_partial_reset_preserves_unselected_history_graph(
    tmp_path: Path, robot: str, device: str
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    checkpoint = _export(tmp_path, robot)
    with _scene(robot, checkpoint, device=device) as (scene, backend):
        before = scene.create_functional_state()
        force = torch.zeros(
            backend.model.joint_dof_count, device=device, requires_grad=True
        )
        state = scene.functional_step(before, {"joint_f": force})
        mask = wp.array([True, False, False], dtype=wp.bool, device=device)
        # Exercise real public backend ownership mapping in the renderer-free
        # Scene shell; actual manager arena selection has its own integration test.
        reset = backend.reset_functional_state(state, mask)
        assert reset.valid_history_lengths.tolist() == [0, 1]
        for value in reset.history.values():
            assert not value[0].any()
        state2 = scene.functional_step(
            reset, {"joint_f": torch.zeros_like(force)}, publish=False
        )
        assert state2.valid_history_lengths.tolist() == [1, 2]
        q = state2.physical["joint_q"][backend.solver.binding.q_indices]
        q.sum().backward()
        assert force.grad is not None
        selected = backend.solver.binding.qd_indices[0]
        unselected = backend.solver.binding.qd_indices[1]
        assert not force.grad[selected].any()
        assert force.grad[unselected].abs().sum() > 0


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("dt", 0.02, "timestep"),
        ("contact_schema", "wrong", "contact"),
        ("physics_signature", "wrong", "physics signature"),
        ("physics_signature_schema", None, "signature schema"),
        ("physics_signature_schema", "unsupported", "signature schema"),
        ("joint_names", ["wrong"], "joint order"),
    ],
)
def test_incompatible_manifest_rejected_at_creation(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    checkpoint = _export(tmp_path, "cartpole")
    path = checkpoint / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        with _scene("cartpole", checkpoint):
            pass


def test_wrong_timestep_and_control_schema_fail_without_advance(tmp_path: Path) -> None:
    checkpoint = _export(tmp_path, "cartpole")
    with _scene("cartpole", checkpoint) as (scene, backend):
        state = scene.create_functional_state()
        with pytest.raises(ValueError, match="timestep"):
            scene.functional_step(state, {"joint_f": torch.zeros(4)}, dt=0.02)
        with pytest.raises(ValueError, match="joint_f"):
            scene.functional_step(state, {"joint_target_pos": torch.zeros(4)})
        assert backend.step_index == 0


def test_non_float32_export_rejected_before_newton_execution(tmp_path: Path) -> None:
    from embodichain.learning.models.nerd import load_model

    original = _export(tmp_path, "cartpole")
    model, manifest = load_model(original)
    checkpoint = tmp_path / "float64-export"
    export_model(checkpoint, model.double(), manifest)
    with pytest.raises(TypeError, match="float32 model"):
        with _scene("cartpole", checkpoint):
            pass


def test_ant_rejects_unmodeled_free_base_force(tmp_path: Path) -> None:
    checkpoint = _export(tmp_path, "ant")
    with _scene("ant", checkpoint) as (scene, backend):
        force = torch.zeros(backend.model.joint_dof_count)
        force[backend.solver.binding.qd_indices[0, 0]] = 1.0
        with pytest.raises(
            (ValueError, NotImplementedError), match="base|root|force|effort"
        ):
            scene.functional_step(scene.create_functional_state(), {"joint_f": force})
        assert backend.step_index == 0


def test_joint_axis_change_invalidates_physics_signature(tmp_path: Path) -> None:
    checkpoint = _export(tmp_path, "cartpole")

    def mutate(builder):
        builder.joint_axis[0] = wp.vec3(0.0, 0.0, 1.0)
        builder.joint_axis[2] = wp.vec3(0.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="physics signature"):
        with _scene("cartpole", checkpoint, mutate=mutate):
            pass


def test_ant_plane_friction_change_invalidates_physics_signature(
    tmp_path: Path,
) -> None:
    checkpoint = _export(tmp_path, "ant")

    def mutate(builder):
        builder.shape_material_mu[-1] = 0.12

    with pytest.raises(ValueError, match="physics signature"):
        with _scene("ant", checkpoint, mutate=mutate):
            pass


def _check_real_manager_public_scene_and_partial_arena_reset(
    tmp_path: Path, robot: str
) -> None:
    from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import NewtonPhysicsCfg, RenderCfg

    def make(checkpoint=None):
        physics = NewtonPhysicsCfg(
            device="cpu",
            physics_dt=DT,
            num_substeps=1,
            requires_grad=checkpoint is not None,
            use_cuda_graph=False,
            collision_cfg=None,
            solver_cfg=(
                {"solver_type": "semi_implicit"}
                if checkpoint is None
                else {
                    "solver_type": "nerd",
                    "options": {"checkpoint": str(checkpoint), "trainable": True},
                }
            ),
        )
        cfg = SimulationManagerCfg(
            headless=True,
            num_envs=2,
            device="cpu",
            physics_cfg=physics,
            render_cfg=RenderCfg(renderer="no-render"),
            enable_ground_plane=robot == "ant",
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
        )
        sim = SimulationManager(cfg)
        try:
            sim.add_articulation_descriptor(make_reference_articulation(robot))
            sim.prepare()
        except BaseException:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()
            raise
        return sim

    teacher = make()
    try:
        checkpoint = _small_export(
            teacher.spawn_result, tmp_path / "manager-model", robot
        )
    finally:
        teacher.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    sim = make(checkpoint)
    try:
        initial = sim.create_functional_state()
        control = torch.zeros_like(initial.physical["joint_qd"], requires_grad=True)
        state = sim.functional_step(
            initial, {"joint_f": control}, render_final_step=False
        )
        sim.reset_objects_state(env_ids=torch.tensor([0]))
        reset = sim.reset_functional_state(state, torch.tensor([True, False]))
        assert reset.valid_history_lengths.tolist() == [0, 1]
        assert sim._visualization_sim_step == 1
        reset.physical["joint_q"].sum().backward()
        assert control.grad is not None and torch.isfinite(control.grad).all()
        binding = NeRDBinding.from_metadata(
            dict(sim.spawn_result.dynamics_metadata()), robot
        )
        assert not control.grad[binding.qd_indices[0]].any()
        assert control.grad[binding.qd_indices[1]].abs().sum() > 0
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def test_fk_vjp_matches_directional_difference(tmp_path: Path) -> None:
    from embodichain.lab.sim.physics.solvers._kinematics import differentiable_fk

    with _scene("ant") as (scene, backend):
        physical = scene.dynamics_snapshot().physical
        q = physical["joint_q"].clone().requires_grad_()
        qd = physical["joint_qd"].clone().requires_grad_()
        # Perturb scalar joints only so the root quaternion remains unit.
        direction = torch.zeros_like(q)
        for row in NeRDBinding(backend.model, "ant").q_indices:
            direction[row[7:]] = torch.linspace(0.2, 0.8, 8)
        velocity_direction = torch.linspace(-0.3, 0.3, qd.numel())
        weights = torch.linspace(0.1, 1.0, backend.model.body_count * 7).reshape(-1, 7)

        def objective(q, qd):
            body, velocity = differentiable_fk(backend.model, q, qd)
            return (body * weights).sum() + velocity.square().sum()

        loss = objective(q, qd)
        gq, gv = torch.autograd.grad(loss, (q, qd))
        epsilon = 1e-3
        finite = (
            objective(
                q.detach() + epsilon * direction,
                qd.detach() + epsilon * velocity_direction,
            )
            - objective(
                q.detach() - epsilon * direction,
                qd.detach() - epsilon * velocity_direction,
            )
        ) / (2 * epsilon)
        analytic = (gq * direction).sum() + (gv * velocity_direction).sum()
        torch.testing.assert_close(analytic, finite, rtol=2e-3, atol=2e-4)


def test_ant_initial_joint_position_contact_gradient_matches_fd(tmp_path: Path) -> None:
    checkpoint = _export(tmp_path, "ant")
    with _scene("ant", checkpoint) as (scene, backend):
        initial = scene.create_functional_state()
        q = initial.physical["joint_q"].clone().requires_grad_()
        state = FunctionalSolverState(
            {**initial.physical, "joint_q": q},
            initial.history,
            initial.valid_history_lengths,
        )
        force = torch.zeros_like(initial.physical["joint_qd"])

        def feature_loss(value):
            # Deliberately retain the initial detached body snapshot: q is the
            # authoritative reduced state and must rebuild contact geometry.
            current = FunctionalSolverState(
                {**state.physical, "joint_q": value},
                state.history,
                state.valid_history_lengths,
            )
            result = scene.functional_step(current, {"joint_f": force}, publish=False)
            return result.history["contact_depths"][:, 0].sum()

        grad = torch.autograd.grad(feature_loss(q), q)[0]
        direction = torch.zeros_like(q)
        direction[backend.solver.binding.q_indices[:, 2]] = 1.0
        epsilon = 1e-3
        finite = (
            feature_loss(q.detach() + epsilon * direction)
            - feature_loss(q.detach() - epsilon * direction)
        ) / (2 * epsilon)
        analytic = (grad * direction).sum()
        assert analytic.abs() > 0.5
        torch.testing.assert_close(analytic, finite, rtol=1e-3, atol=1e-4)


def test_model_change_invalidates_future_steps_even_after_failed_notification(
    tmp_path: Path,
) -> None:
    checkpoint = _export(tmp_path, "cartpole")
    with _scene("cartpole", checkpoint) as (scene, backend):
        state = scene.create_functional_state()
        original = backend.model.joint_axis.numpy().copy()
        changed = original.copy()
        changed[:, :] = [0.0, 0.0, 1.0]
        backend.model.joint_axis.assign(changed)
        with pytest.raises(ValueError, match="changed|signature"):
            backend.solver.notify_model_changed(0)
        backend.model.joint_axis.assign(original)
        with pytest.raises(RuntimeError, match="invalid|rebuild|changed"):
            scene.functional_step(
                state, {"joint_f": torch.zeros_like(state.physical["joint_qd"])}
            )
        assert backend.step_index == 0


@pytest.mark.subprocess_sim
@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_real_manager_public_scene_and_partial_arena_reset(
    tmp_path: Path, robot: str
) -> None:
    # DexSim renderer is process-global; isolate no-render tests from the main
    # suite's hybrid renderer fixture without resetting native global state.
    code = "import importlib.util,sys; from pathlib import Path; spec=importlib.util.spec_from_file_location('nerd_provider_test',sys.argv[1]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module._check_real_manager_public_scene_and_partial_arena_reset(Path(sys.argv[2]),sys.argv[3])"
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(__file__).resolve()),
            str(tmp_path),
            robot,
        ],
        check=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": str(Path(__file__).resolve().parents[4]),
        },
    )


def _check_real_differentiable_environment(tmp_path: Path) -> None:
    import gymnasium as gym
    from embodichain.lab.gym.envs.differentiable_dynamics_env import (
        DifferentiableDynamicsEnv,
    )
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
    from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import NewtonPhysicsCfg, RenderCfg

    def sim_cfg(checkpoint=None):
        physics = NewtonPhysicsCfg(
            device="cpu",
            physics_dt=DT,
            num_substeps=1,
            requires_grad=checkpoint is not None,
            use_cuda_graph=False,
            collision_cfg=None,
            solver_cfg=(
                {"solver_type": "semi_implicit"}
                if checkpoint is None
                else {
                    "solver_type": "nerd",
                    "options": {"checkpoint": str(checkpoint), "trainable": True},
                }
            ),
        )
        return SimulationManagerCfg(
            headless=True,
            num_envs=2,
            device="cpu",
            physics_cfg=physics,
            render_cfg=RenderCfg(renderer="no-render"),
            enable_ground_plane=False,
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
        )

    teacher = SimulationManager(sim_cfg())
    try:
        teacher.add_articulation_descriptor(make_reference_articulation("cartpole"))
        teacher.prepare()
        checkpoint = _small_export(
            teacher.spawn_result, tmp_path / "env-model", "cartpole"
        )
    finally:
        teacher.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    class Task(DifferentiableDynamicsEnv):
        def _setup_scene(self, **kwargs):
            self.sim = SimulationManager(self.sim_cfg)
            self.sim.add_articulation_descriptor(
                make_reference_articulation("cartpole")
            )

        def _setup_robot(self, **kwargs):
            self.single_action_space = gym.spaces.Box(
                -1.0, 1.0, shape=(2,), dtype=np.float32
            )
            self.single_observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float32
            )
            return self.sim.get_articulation("cartpole")

        def _action_to_control(self, action, state):
            return {"joint_f": action.flatten()}

        def _read_outputs(self, state):
            q = state.physical["joint_q"].reshape(2, 2)
            done = (torch.arange(2, device=q.device) == 0) & (self._elapsed_steps >= 1)
            return {
                "obs": q,
                "reward": q.square().sum(-1),
                "terminated": done,
                "truncated": torch.zeros(2, dtype=torch.bool, device=q.device),
            }

    env = Task(
        EmbodiedEnvCfg(
            num_envs=2,
            sim_cfg=sim_cfg(checkpoint),
            sim_steps_per_control=1,
            max_episode_steps=5,
        )
    )
    try:
        obs, _ = env.reset(seed=7)
        assert obs.shape == (2, 2)
        action = torch.ones(2, 2, requires_grad=True)
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated.tolist() == [True, False]
        assert env.functional_state.valid_history_lengths.tolist() == [0, 1]
        assert env.sim._visualization_sim_step == 1
        loss = reward.sum() + obs.square().sum()
        loss.backward()
        assert action.grad is not None and torch.isfinite(action.grad).all()
        assert action.grad.abs().sum() > 0
        assert all(
            value.grad is not None for value in env.dynamics_parameters().values()
        )
        assert not env.detach_state().requires_grad
    finally:
        env.sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.subprocess_sim
def test_actual_differentiable_environment_constructor_reset_and_backward(
    tmp_path: Path,
) -> None:
    code = "import importlib.util,sys; from pathlib import Path; spec=importlib.util.spec_from_file_location('nerd_provider_test',sys.argv[1]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module._check_real_differentiable_environment(Path(sys.argv[2]))"
    subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        check=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": str(Path(__file__).resolve().parents[4]),
        },
    )


def test_frozen_network_accepts_explicit_trainable_parameter_mapping(
    tmp_path: Path,
) -> None:
    checkpoint = _export(tmp_path, "cartpole")
    with _scene("cartpole", checkpoint) as (scene, backend):
        backend.solver.network.requires_grad_(False)
        state = scene.create_functional_state()
        force = torch.ones_like(state.physical["joint_qd"], requires_grad=True)
        baseline = scene.functional_step(state, {"joint_f": force}, publish=False)
        parameters = {
            name: value.detach().clone().requires_grad_()
            for name, value in scene.functional_parameters().items()
        }
        explicit = scene.functional_step(
            state, {"joint_f": force}, parameters=parameters, publish=False
        )
        torch.testing.assert_close(
            explicit.physical["joint_q"], baseline.physical["joint_q"]
        )
        explicit.physical["joint_q"].square().sum().backward()
        assert all(
            value.grad is None for value in scene.functional_parameters().values()
        )
        assert all(
            value.grad is not None and torch.isfinite(value.grad).all()
            for value in parameters.values()
        )
        assert force.grad is not None and force.grad.abs().sum() > 0
        altered = dict(parameters)
        altered["model.output_net.bias"] = parameters["model.output_net.bias"] + 0.1
        changed = scene.functional_step(
            state, {"joint_f": force}, parameters=altered, publish=False
        )
        assert not torch.allclose(
            changed.physical["joint_q"], baseline.physical["joint_q"]
        )


@pytest.mark.gpu
@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_cuda_native_functional_and_contact_backward_on_torch_stream(
    tmp_path: Path, robot: str
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    checkpoint = _export(tmp_path, robot)
    with _scene(robot, checkpoint, device="cuda:0") as (scene, backend):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
            state = scene.create_functional_state()
            force = torch.zeros_like(state.physical["joint_qd"], requires_grad=True)
            force_data = force.detach()
            expected = scene.functional_step(state, {"joint_f": force}, publish=False)
            wp.copy(backend.control.joint_f, wp.from_torch(force_data))
            backend.step()
            actual = scene.create_functional_state()
            torch.testing.assert_close(
                actual.physical["joint_q"],
                expected.physical["joint_q"],
                atol=1e-6,
                rtol=1e-5,
            )
            expected.physical["body_q"].square().sum().backward()
            assert force.grad is not None and torch.isfinite(force.grad).all()
            if robot == "ant":
                q = state.physical["joint_q"].detach().clone()
                q[backend.solver.binding.q_indices[:, 2]] = 0.3
                q.requires_grad_()
                lowered = FunctionalSolverState(
                    {**state.physical, "joint_q": q},
                    state.history,
                    state.valid_history_lengths,
                )
                result = scene.functional_step(
                    lowered, {"joint_f": torch.zeros_like(force)}, publish=False
                )
                # Foot penetration yields active continuous plane features; masks
                # are discrete and held fixed in the local derivative contract.
                features = result.history["contact_depths"][:, 0]
                assert (features < 0).any()
                features.sum().backward()
                assert q.grad is not None and torch.isfinite(q.grad).all()
                assert q.grad[backend.solver.binding.q_indices[:, 7:]].abs().sum() > 0
        stream.synchronize()


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_explicit_incoming_history_gradient(
    tmp_path: Path, robot: str, device: str
) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    checkpoint = _export(tmp_path, robot)
    with _scene(robot, checkpoint, device=device) as (scene, backend):
        first = scene.create_functional_state()
        force = torch.zeros_like(first.physical["joint_qd"])
        warm = scene.functional_step(first, {"joint_f": force}, publish=False)
        history = {
            name: value.detach().clone().requires_grad_()
            for name, value in warm.history.items()
        }
        state = FunctionalSolverState(
            {name: value.detach() for name, value in warm.physical.items()},
            history,
            warm.valid_history_lengths,
        )
        result = scene.functional_step(state, {"joint_f": force}, publish=False)
        (
            result.physical["joint_q"].square().sum()
            + result.physical["body_q"].square().sum()
        ).backward()
        for value in history.values():
            assert value.grad is not None and torch.isfinite(value.grad).all()
            assert value.grad[:, 0].abs().sum() > 0
            assert not value.grad[:, 1:].any()


def test_cartpole_fixed_root_orientation_is_bound(tmp_path: Path) -> None:
    checkpoint = _export(tmp_path, "cartpole")

    def invert_root(builder):
        for joint, parent in enumerate(builder.joint_parent):
            if parent == -1:
                builder.joint_X_p[joint] = wp.transform(
                    (0.0, 0.0, 0.0), wp.quat(1.0, 0.0, 0.0, 0.0)
                )

    with pytest.raises(ValueError, match="physics signature"):
        with _scene("cartpole", checkpoint, mutate=invert_root):
            pass


@pytest.mark.parametrize(
    "field",
    [
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_restitution",
        "shape_material_ka",
        "shape_material_mu_torsional",
        "shape_gap",
    ],
)
def test_ant_robot_contact_properties_are_bound(tmp_path: Path, field: str) -> None:
    checkpoint = _export(tmp_path, "ant")

    def change_material(builder):
        # Modify every robot equally: distinguish incompatible export from mere
        # heterogeneous-batch rejection. Leave the shared plane unchanged.
        values = getattr(builder, field)
        for shape, body in enumerate(builder.shape_body):
            if body >= 0:
                values[shape] = float(values[shape]) + 0.123

    with pytest.raises(ValueError, match="physics signature"):
        with _scene("ant", checkpoint, mutate=change_material):
            pass


@pytest.mark.parametrize("mutation", ["self_collision", "disabled_plane", "topology"])
def test_ant_contact_policy_and_topology_are_bound(
    tmp_path: Path, mutation: str
) -> None:
    checkpoint = _export(tmp_path, "ant")

    def change(builder):
        if mutation == "self_collision":
            builder.shape_collision_filter_pairs.clear()
        elif mutation == "disabled_plane":
            builder.shape_collision_group[-1] = 0
        else:
            roots = [
                joint
                for joint, parent in enumerate(builder.joint_parent)
                if parent == -1
            ]
            for root in roots:
                builder.joint_parent[root + 2] = builder.joint_child[root]

    with pytest.raises(ValueError, match="plane-only|physics signature"):
        with _scene("ant", checkpoint, mutate=change):
            pass


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_complete_signature_matches_public_snapshot_and_replica_count(
    robot: str,
) -> None:
    with _scene(robot, num_envs=1) as (one, backend):
        direct = NeRDBinding(backend.model, robot)
        snapshot = NeRDBinding.from_metadata(dict(one.dynamics_metadata()), robot)
        assert direct.physics_signature == snapshot.physics_signature
        assert (
            direct.metadata(DT)["physics_signature_schema"]
            == "nerd-reference-physics-v2"
        )
        signature = direct.physics_signature
    with _scene(robot, num_envs=2) as (two, backend):
        assert NeRDBinding(backend.model, robot).physics_signature == signature
        assert (
            NeRDBinding.from_metadata(
                dict(two.dynamics_metadata()), robot
            ).physics_signature
            == signature
        )
