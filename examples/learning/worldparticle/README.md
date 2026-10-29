# WorldParticle

The model layer lives in `embodichain_models.dynamics.worldparticle`: original
network, strict checkpoint loading, authored features/topology and coordinate
conversion. It imports neither simulation nor examples. The integration layer
is `embodichain.lab.sim.physics.worldparticle.WorldParticleProvider`; the native
solver adapter is DexSim's model-independent `ParticleInferenceSolver`.
Providers are constructed per scene through the existing Newton lifecycle.

## Prepared environment

From this repository, the prepared environment is `.venv-worldparticle`.
The normal `.venv` is unchanged. Linux x86_64, glibc >=2.35, Python 3.12,
Torch 2.7.1+cu128/CXX11 ABI and SM120 are the currently verified operator profile.
NumPy 2.2.6 is used to satisfy EmbodiChain/LeRobot; Torchvision matches Torch.

To reproduce, build the matching DexSim SDK using its normal CMake workflow
(`DexSim`, `pybind`, `pip-jit-package` targets), preserving its cached Python
interpreter. Use a local PEP 440 package version such as
`-DDEXSIM_PACKAGE_VERSION=0.5.1rc1+worldparticle.6367b2` so it satisfies this
checkout's SDK dependency; the unmodified published SDK lacks the new adapter.
Then pass the produced CPython 3.12 wheel explicitly:

```bash
uv venv --python 3.12 .venv-worldparticle --seed
SDK_WHEEL=/absolute/path/to/dexsim_engine-0.5.1rc1+worldparticle.6367b2-cp312-cp312-manylinux_2_35_x86_64.whl
uv pip install --python .venv-worldparticle/bin/python --no-build-isolation \
  -e '.[worldparticle]' "$SDK_WHEEL" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
uv pip check --python .venv-worldparticle/bin/python
```

The requested DexSim source checkout and EmbodiChain's existing default/mapping
Newton configuration have an AutoSolver API mismatch. This change adapts only
explicit SDK configuration objects used by the Provider; it does not change
default solver selection or add an AutoSolver-to-DexUni fallback. Use the
explicit Provider configuration for this environment. The normal `.venv`
remains untouched.

The optional requirements in `pyproject.toml` are the dependency authority.
`runtime-manifest.json`, the Open3D patch and relocation utility record operator
build/reproduction provenance. No environment, weight or native wheel is committed.

## Resources and demos

```bash
source .venv-worldparticle/bin/activate
embodichain run-example learning/worldparticle prepare
embodichain run-example learning/worldparticle demo --scene fancy_fluid
embodichain run-example learning/worldparticle demo --scene cloth_on_ball
embodichain run-example learning/worldparticle demo --scene inverse_design
embodichain run-example learning/worldparticle demo --scene deep_fluid
```

The four profiles retain the official checkpoints. Cloth starts at configured
frame 10; duck uses stride 5. Deep Fluid uses generated dam-break initial
conditions with the official model and has no reference trajectory. Duck is a
forward inference demo, not inverse optimization. Rendering uses DexSim's native
1920x1080 OfflineRT window and public Spawn materials/geometry. Repeated display
runs reset and infer fresh states. `--once --hold-seconds 0` exits after one run;
`--headless` runs without a window. `--capture-dir DIR` saves native camera images.

Assets default to `outputs/worldparticle/assets`, predictions to
`outputs/worldparticle/demos`. The explicit prepare command verifies all 19
file sizes and SHA256 hashes; `--local-files-only` forbids downloading.
The immutable archive remains on the existing internal resource service;
its original DexSim URL is a storage location, not an SDK runtime dependency.

For application scenes, pass `ProviderNewtonSolverCfg(provider_factory=
WorldParticleProvider, options={"checkpoint": ..., "config": ..., "inputs": ...})`
to `NewtonPhysicsCfg(solver_cfg=..., use_cuda_graph=False, collision_cfg=None)`.
For `SimulationManager`, set `enable_default_ground=False` because neural
boundaries are authored in the input NPZ. The ordinary manager ground default
and built-in solver selection are unchanged.
Alternatively, the installed `worldparticle` solver entry point belongs to the
EmbodiChain distribution. Use the checkpoint dt/gravity and matching authored
particle order. Only a single CUDA particle world is supported, without native
rigid collision shapes, gradients or CUDA Graph capture. Static boundaries and
features come from model inputs. Changing them requires closing/recreating the
scene; reset restores physical state and does not reload model assets.

## Licensing

WorldParticle's root MIT license and provenance are included in the model package.
Its vendored ToMe merging component retains CC-BY-NC-4.0 (`LICENSE.tome`).
Moving this code does not grant commercial redistribution permission. The model
code currently ships with the EmbodiChain wheel; this integration has not been
commercially cleared. Weight/data rights also need independent release review.
