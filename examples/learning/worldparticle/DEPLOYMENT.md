# WorldParticle native solver quickstart

Complete the [operator installation check](BUILDING.md#install-and-check-the-built-operators)
first. Run from a source checkout with the main EmbodiChain package/dependencies
installed and bundled extension mappings refreshed. This example uses public
SDK SceneBuilder and the direct WorldParticle Provider. It does not create a
Gym/learning environment or use the manager's DexUni/SPH fluid route.

## Prepare and verify assets

```bash
python -m embodichain run-example learning/worldparticle prepare --assets outputs/worldparticle/assets
python -m embodichain run-example learning/worldparticle prepare --assets outputs/worldparticle/assets --local-files-only
```

Preparation needs access to the existing internal resource service when files
are absent. It verifies the archive and all 19 committed file identities. The
following example uses Fancy Fluid; the four complete CLI demos and rendering
options are in [README.md](README.md).

## Build, step, read back, reset and close

```python
from pathlib import Path
import numpy as np
import yaml
import dexsim
from dexsim.engine.newton_physics import NewtonCfg, ProviderNewtonSolverCfg
from dexsim.spawn import SceneBuilder, FluidRenderMode
from embodichain_solvers.physics.worldparticle.provider import WorldParticleProvider

assets = Path("outputs/worldparticle/assets").resolve()
checkpoint = assets / "checkpoints/fancy_fluid/final.ckpt"
config_path = assets / "checkpoints/fancy_fluid/config.yaml"
sample = assets / "data/fancy_fluid/66/66.npz"
model_cfg = yaml.safe_load(config_path.read_text())
R = np.asarray([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], np.float32)
with np.load(sample, allow_pickle=False) as data:
    frame = int(model_cfg.get("frame_start", 0))
    q0 = np.asarray(data["position"][frame], np.float32) @ R.T
    qd0 = np.asarray(data["velocity"][frame], np.float32) @ R.T
    raw = data["feats"]
    assert np.allclose(raw[:, :2], raw[0, :2])
    mass, density = map(float, raw[0, :2])
options = {
    "checkpoint": str(checkpoint),
    "config": str(config_path),
    "inputs": str(sample),
    "model_to_world": R.tolist(),
}
config = dexsim.WorldConfig()
config.open_windows = False
config.cpu_num = 2
config.init_ground = False
config.use_default_physics = False
config.backend = dexsim.types.Backend.VULKAN
config.renderer = dexsim.types.Renderer.NORENDER
config.newton_cfg = NewtonCfg(
    device="cuda:0",
    dt=float(model_cfg["timestep"]),
    gravity=R @ np.asarray(model_cfg["gravity"], np.float32),
    num_substeps=1,
    requires_grad=False,
    sync_to_dexsim=False,
    collision_pipeline_cfg=None,
    solver_cfg=ProviderNewtonSolverCfg(
        provider_factory=WorldParticleProvider, options=options,
    ),
)
config.newton_cfg.use_cuda_graph = False
world = dexsim.World(config)
scene = particles = None
try:
    builder = SceneBuilder(world)
    builder.add_fluid(
        "prediction", q0,
        density=density,
        particle_spacing=(mass / density) ** (1 / 3),
        particle_radius=model_cfg["particle_radius"],
        material=None,
        render_mode=FluidRenderMode.PARTICLES,
    )
    scene = builder.finalize()
    particles = scene.get_fluid_particle_set("prediction")
    particles.set_particle_velocities(qd0)
    for step in range(20):
        world.update()
        q = particles.get_particle_positions().numpy()
        qd = particles.get_particle_velocities().numpy()
        assert np.isfinite(q).all() and np.isfinite(qd).all()
    assert np.max(np.abs(q - q0)) > 0
    scene.reset()
    np.testing.assert_allclose(particles.get_particle_positions().numpy(), q0, atol=1e-7)
    # Velocity installed after finalize is not the builder's reset baseline.
    particles.set_particle_velocities(qd0)
    world.update()
    assert np.isfinite(particles.get_particle_positions().numpy()).all()
    print("WorldParticle native step/reset passed", q.shape)
finally:
    particles = None
    if scene is not None:
        scene.close()
    world = None
```

The carrier mass is inferred from density × spacing³ and matches the sample's
uniform Fancy Fluid mass. Boundary points/features stay in the model NPZ. No
native ground, rigid body or collision shape is declared. SDK machinery converts
force to total acceleration and writes predictions to native buffers. Reading
after every step is deliberate verification, not a throughput benchmark.

## Named configuration and other scenes

After installing the current wheel/editable metadata, the equivalent registered
selection is:

```python
from dexsim.engine.newton_physics import RegisteredNewtonSolverCfg

registered = RegisteredNewtonSolverCfg(solver_type="worldparticle", options=options)
```

Use `registered` as `NewtonCfg.solver_cfg` at World construction instead of the
direct Provider config. This block selects a Provider; it does not create a
second scene. Direct Provider selection does not use installed-name discovery.

Cloth uses `SceneBuilder.add_clothbody` with its authored display surface; Duck
and Deep Fluid use the unconnected-particle carrier with their own feature,
rotation and initial-state rules. Use their existing demos rather than
substituting a new physical graph. [INPUTS.md](INPUTS.md) defines all loader
keys, NPZ layouts, frame conversions and admission constraints.

Reset restores physical state without reloading fixed model inputs. Recreate
the model/scene when features, boundaries, topology or rotation change. Missing
SDK facilities, rejected runtime versions or undefined operator symbols require
the matching environment/build. Count, gravity and timestep errors require a
matching authored scene/export pair. [VALIDATION.md](VALIDATION.md) states the
tested budgets, rendering scope and numerical limits.
