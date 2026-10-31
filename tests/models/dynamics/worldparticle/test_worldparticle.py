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
"""Optional WorldParticle adapter contracts, without downloading weights."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from embodichain_models.dynamics.worldparticle._runtime import load_assets


@pytest.fixture
def assets(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "data: fancy_fluid\ntimestep: 0.01\ngravity: [0., 0., 0.]\nk_partite: 2\n"
    )
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"network loader is replaced only in adapter tests")
    data = tmp_path / "inputs.npz"
    np.savez(
        data,
        feats=np.asarray([[1.0, 1000.0, v] for v in range(4)], np.float32),
        box=np.asarray([[0.0, -1.0, 0.0]], np.float32),
        box_feats=np.asarray([[0.0, 1.0, 0.0]], np.float32),
    )
    return dict(config=str(cfg), checkpoint=str(ckpt), inputs=str(data))


def test_assets_preserve_training_feature_normalization(assets):
    cfg, (features, box, normals), rotation = load_assets(assets)
    values = np.arange(4, dtype=np.float32)
    np.testing.assert_array_equal(
        features[:, 0], (values - values.mean()) / (values.std() + 1e-8)
    )
    np.testing.assert_array_equal(rotation, np.eye(3))
    assert cfg["resolved_other_feats_channels"] == 1
    assert box.shape == normals.shape == (1, 3)


@pytest.mark.parametrize(
    "change,match",
    [
        (
            {"model_to_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]},
            "proper orthonormal",
        ),
        ({"checkpoint": "/nonexistent/worldparticle.ckpt"}, "existing local file"),
        ({"unexpected_option": 1}, "Unknown WorldParticle options"),
    ],
)
def test_invalid_assets_fail_before_loading_network(assets, change, match):
    with pytest.raises(ValueError, match=match):
        load_assets(dict(assets, **change))


def test_unsupported_checkpoint_task_is_rejected(assets):
    Path(assets["config"]).write_text("data: unsupported_task\ntimestep: 0.02\n")
    with pytest.raises(ValueError, match="supported data"):
        load_assets(assets)


class _TranslationNetwork(torch.nn.Module):

    def forward(self, inputs, *, k_partite, external_force):
        q, qd, *_ = inputs
        next_qd = qd + 0.01 * external_force
        return q + 0.005 * (qd + next_qd), next_qd


@pytest.mark.skipif(
    not wp.is_cuda_available(), reason="Public fluid authoring requires CUDA"
)
def test_public_worldparticle_provider_advances_and_resets(assets, monkeypatch):
    import dexsim
    from dexsim.engine.newton_physics import NewtonCfg, ProviderNewtonSolverCfg
    from embodichain.lab.sim.physics.worldparticle import WorldParticleProvider
    from embodichain_models.dynamics.worldparticle import _runtime
    from dexsim.spawn import SceneBuilder, FluidRenderMode

    monkeypatch.setattr(
        _runtime, "load_network", lambda *_: _TranslationNetwork().cuda()
    )
    config = dexsim.WorldConfig()
    config.open_windows = False
    config.init_ground = False
    config.use_default_physics = False
    config.backend = dexsim.types.Backend.VULKAN
    config.renderer = dexsim.types.Renderer.NORENDER
    rotation = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    config.newton_cfg = NewtonCfg(
        device="cuda:0",
        gravity=0.0,
        dt=0.01,
        num_substeps=1,
        sync_to_dexsim=False,
        collision_pipeline_cfg=None,
        solver_cfg=ProviderNewtonSolverCfg(
            provider_factory=WorldParticleProvider,
            options=dict(assets, model_to_world=rotation),
        ),
    )
    config.newton_cfg.use_cuda_graph = False
    world = dexsim.World(config)
    builder = SceneBuilder(world)
    initial = np.asarray([[i, 0.0, 1.0] for i in range(4)], np.float32)
    builder.add_fluid(
        "fluid",
        initial,
        density=1000.0,
        particle_spacing=0.1,
        render_mode=FluidRenderMode.PARTICLES,
    )
    scene = builder.finalize()
    try:
        fluid = scene.get_fluid_particle_set("fluid")
        velocity = np.tile([0.0, 0.0, 2.0], (4, 1))
        fluid.set_particle_velocities(velocity)
        world.update()
        np.testing.assert_allclose(
            fluid.get_particle_positions().numpy(),
            initial + [0.0, 0.0, 0.02],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            fluid.get_particle_velocities().numpy(), velocity, atol=1e-6
        )
        assert scene.dynamics_capabilities().supported_particle_kinds == ("fluid",)
        scene.reset()
        np.testing.assert_allclose(
            fluid.get_particle_positions().numpy(), initial, atol=1e-6
        )
        np.testing.assert_array_equal(
            fluid.get_particle_velocities().numpy(), np.zeros((4, 3))
        )
    finally:
        scene.close()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA inference adapter")
def test_particle_force_units_rotation_buffers_and_invalid_dt():
    import newton
    from dexsim.engine.newton_physics.particle_inference import ParticleInferenceSolver
    from embodichain_models.dynamics.worldparticle.model import WorldParticleModel

    builder = newton.ModelBuilder()
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=2.0
    )
    model = builder.finalize(device="cuda:0")
    model.set_gravity((0.0, 0.0, -10.0))
    first, second = model.state(), model.state()
    wp.to_torch(first.particle_f).copy_(
        torch.tensor([[4.0, 0.0, 0.0]], device="cuda:0")
    )
    rotation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], np.float32
    )
    solver = ParticleInferenceSolver(
        model,
        lambda device: WorldParticleModel(
            _TranslationNetwork().cuda(),
            (torch.zeros((1, 1), device="cuda:0"), None, None),
            rotation,
            {"timestep": 0.01, "k_partite": 2},
        ),
        0.01,
    )
    try:
        solver.step(first, second, model.control(), None, 0.01)
        wp.synchronize_device(model.device)
        np.testing.assert_allclose(
            second.particle_q.numpy(), [[0.0001, 0.0, -0.0005]], atol=1e-7
        )
        np.testing.assert_allclose(
            second.particle_qd.numpy(), [[0.02, 0.0, -0.1]], atol=1e-6
        )
        np.testing.assert_array_equal(first.particle_q.numpy(), np.zeros((1, 3)))
        retained = second.particle_q.numpy().copy()
        with pytest.raises(ValueError, match="step dt"):
            solver.step(first, second, model.control(), None, 0.02)
        np.testing.assert_array_equal(second.particle_q.numpy(), retained)
    finally:
        solver.close()
    with pytest.raises(RuntimeError, match="closed"):
        solver.step(first, second, model.control(), None, 0.01)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA provider creation")
@pytest.mark.parametrize(
    "change,match",
    [
        ({"dt": 0.02}, "substep dt"),
        ({"requires_grad": True}, "inference only"),
    ],
)
def test_unsupported_runtime_rejected_before_weights(
    assets, monkeypatch, change, match
):
    import newton
    from dexsim.engine.newton_physics import NewtonSolverContext
    from embodichain.lab.sim.physics.worldparticle import WorldParticleProvider
    from embodichain_models.dynamics.worldparticle import _runtime

    def forbidden_load(*_):
        raise AssertionError("Weights allocated before runtime validation")

    monkeypatch.setattr(_runtime, "load_network", forbidden_load)
    builder = newton.ModelBuilder()
    for i in range(4):
        builder.add_particle(
            pos=wp.vec3(float(i), 0.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0
        )
    model = builder.finalize(
        device="cuda:0", requires_grad=change.get("requires_grad", False)
    )
    model.set_gravity((0.0, 0.0, 0.0))
    context = NewtonSolverContext(
        model=model,
        control=model.control(),
        contacts=None,
        device=model.device,
        dt=change.get("dt", 0.01),
        scene_indices={},
    )
    with pytest.raises(ValueError, match=match):
        WorldParticleProvider().create(context, assets)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA inference adapter")
@pytest.mark.parametrize("mode", ["zero_mass", "inactive"])
def test_runtime_fixed_particle_remains_fixed(mode):
    import newton
    from dexsim.engine.newton_physics.particle_inference import ParticleInferenceSolver
    from embodichain_models.dynamics.worldparticle.model import WorldParticleModel

    builder = newton.ModelBuilder()
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, 1.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=1.0
    )
    model = builder.finalize(device="cuda:0")
    model.set_gravity((0.0, 0.0, -10.0))
    solver = ParticleInferenceSolver(
        model,
        lambda device: WorldParticleModel(
            _TranslationNetwork().cuda(),
            (torch.zeros((1, 1), device="cuda:0"), None, None),
            np.eye(3, dtype=np.float32),
            {"timestep": 0.01, "k_partite": 2},
        ),
        0.01,
    )
    first, second = model.state(), model.state()
    # Public particle inverse-mass writes can fix a particle after creation.
    if mode == "zero_mass":
        model.particle_inv_mass.zero_()
    else:
        model.particle_flags.zero_()
    try:
        solver.step(first, second, model.control(), None, 0.01)
        wp.synchronize_device(model.device)
        np.testing.assert_array_equal(
            second.particle_q.numpy(), first.particle_q.numpy()
        )
        np.testing.assert_array_equal(second.particle_qd.numpy(), np.zeros((1, 3)))
    finally:
        solver.close()


@pytest.mark.parametrize(
    "task,raw,expected",
    [
        ("fluid", [[0.125, 0.01], [0.25, 0.02]], [[0.125, 0.01], [0.25, 0.02]]),
        ("duck_drop", [[0.2], [0.4]], [[0.2], [0.4]]),
        (
            "cloth_on_ball",
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 5000.0, 50.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 300000.0, 500.0],
            ],
            [[-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ),
    ],
)
def test_task_specific_feature_semantics(task, raw, expected):
    from embodichain_models.dynamics.worldparticle._runtime import normalize_features

    np.testing.assert_allclose(
        normalize_features(task, np.asarray(raw, np.float64)), expected, atol=1e-7
    )


def test_topology_layout_and_capabilities_are_model_scoped(assets):
    from embodichain.lab.sim.physics.worldparticle import WorldParticleProvider

    Path(assets["config"]).write_text(
        "data: cloth_on_ball\ntimestep: .02\ngravity: [0.,0.,0.]\n"
    )
    raw = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 5000.0, 50.0], (4, 1))
    values = dict(
        feats=raw,
        box=np.zeros((1, 3)),
        box_feats=np.ones((1, 3)),
        neighbors_index=np.asarray([1, 0, 3, 2]),
        neighbors_row_splits=np.arange(5),
    )
    np.savez(assets["inputs"], **values)
    cfg, arrays, _ = load_assets(assets)
    assert cfg["resolved_use_topology_conv"] is True
    assert cfg["resolved_other_feats_channels"] == 3
    np.testing.assert_array_equal(arrays[3], values["neighbors_index"])
    cloth = WorldParticleProvider()
    cloth.register_custom_attributes(None, assets)
    assert cloth.capabilities.supported_particle_kinds == ("cloth",)
    assert WorldParticleProvider().capabilities.supported_particle_kinds == ("fluid",)
    values["neighbors_index"][0] = 4
    np.savez(assets["inputs"], **values)
    with pytest.raises(ValueError, match="topology CSR"):
        load_assets(assets)


def test_topology_checkpoint_requires_original_neighbors(assets):
    Path(assets["config"]).write_text(
        "data: duck_drop\ntimestep: .02\ngravity: [0.,0.,0.]\n"
    )
    np.savez(
        assets["inputs"],
        feats=np.ones((4, 1)),
        box=np.zeros((1, 3)),
        box_feats=np.ones((1, 3)),
    )
    with pytest.raises(ValueError, match="requires CSR"):
        load_assets(assets)


def test_legacy_fluid_buffer_migration_does_not_hide_missing_weights():
    from embodichain_models.dynamics.worldparticle._runtime import load_checkpoint_state

    network = torch.nn.Module()
    network.weight = torch.nn.Parameter(torch.ones(1))
    network.local_feature_extractor = torch.nn.Module()
    network.local_feature_extractor.register_buffer(
        "topology_filter_extent_tensor", torch.tensor(0.01)
    )
    state = {"weight": torch.tensor([2.0])}
    with pytest.warns(UserWarning, match="unused topology"):
        load_checkpoint_state(network, state, legacy_fluid=True)
    torch.testing.assert_close(network.weight, torch.tensor([2.0]))
    torch.testing.assert_close(
        network.local_feature_extractor.topology_filter_extent_tensor,
        torch.tensor(0.01),
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint_state(network, {}, legacy_fluid=True)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint_state(network, state, legacy_fluid=False)


def test_explicit_provider_config_does_not_resolve_builtin_solvers():
    from embodichain.lab.sim.cfg import NewtonPhysicsCfg
    from embodichain.lab.sim.physics.worldparticle import WorldParticleProvider
    from dexsim.engine.newton_physics import ProviderNewtonSolverCfg

    provider = ProviderNewtonSolverCfg(
        provider_factory=WorldParticleProvider, options={}
    )
    cfg = NewtonPhysicsCfg(
        solver_cfg=provider, collision_cfg=None, use_cuda_graph=False
    )
    result = cfg.to_dexsim_cfg(gpu_id=0)
    assert result.solver_cfg is cfg.solver_cfg
    assert result.solver_cfg.provider_factory is WorldParticleProvider
    assert result.collision_pipeline_cfg is None
    assert result.use_cuda_graph is False
