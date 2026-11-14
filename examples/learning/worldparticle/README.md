# WorldParticle

The model layer lives in `embodichain_models.dynamics.worldparticle`: original
network, strict checkpoint loading, authored features/topology and coordinate
conversion. It imports neither simulation nor examples. The integration layer
is `embodichain.lab.sim.physics.worldparticle.WorldParticleProvider`; the native
solver adapter is DexSim's model-independent `ParticleInferenceSolver`.
Providers are constructed per scene through the existing Newton lifecycle.

## Prepared environment

The demos use a dedicated Conda `py312` environment with `dexsim_engine==0.6.0rc2`.
Linux x86_64, glibc >=2.35, Python 3.12,
Torch 2.7.1+cu128/CXX11 ABI and SM120 are the currently verified operator profile.
NumPy is 2.2.6; Torchvision matches Torch. The published CPython 3.12 SDK wheel
contains `ParticleInferenceSolver`.

Run from the repository root. Export the model's pinned requirements from the
existing extra, then install the SDK and simulation imports needed by this
source-checkout example:

```bash
conda create -n py312 python=3.12 pip -y
conda activate py312
python - <<'PY'
from pathlib import Path
import tomllib

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
output = Path("outputs/worldparticle/model-requirements.txt")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(project["optional-dependencies"]["worldparticle"]) + "\n")
PY
uv pip install --python "$CONDA_PREFIX/bin/python" \
  --index-url https://pypi.org/simple \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r outputs/worldparticle/model-requirements.txt \
  'dexsim_engine==0.6.0rc2' 'tensordict==0.13.0' \
  'pytorch_kinematics==0.10.0' 'viser==1.0.21' \
  'gymnasium>=0.29.1' prettytable fvcore h5py
uv pip check --python "$CONDA_PREFIX/bin/python"
export PYTHONPATH="$PWD:$PWD/extensions/embodichain_models:$PWD/extensions/embodichain_robots${PYTHONPATH:+:$PYTHONPATH}"
```

This validates the source demo's simulation runtime. The main distribution's
Python 3.12 training profile currently requires Torch 2.11 and Tensordict 0.15,
which conflicts with these model operators. Full `.[worldparticle]` installation
still requires reconciling those dependency profiles.

The optional requirements in `pyproject.toml` are the dependency authority.
`runtime-manifest.json`, the Open3D patch and relocation utility record operator
build/reproduction provenance. No environment, weight or native wheel is committed.

## Resources and demos

```bash
python -m embodichain run-example learning/worldparticle prepare
python -m embodichain run-example learning/worldparticle demo --scene fancy_fluid
python -m embodichain run-example learning/worldparticle demo --scene cloth_on_ball
python -m embodichain run-example learning/worldparticle demo --scene inverse_design
python -m embodichain run-example learning/worldparticle demo --scene deep_fluid
```

The four profiles retain the official checkpoints. Cloth starts at configured
frame 10; duck uses stride 5. Deep Fluid uses generated dam-break initial
conditions with the official model and has no reference trajectory. Duck is a
forward inference demo, not inverse optimization. Rendering uses DexSim's native
1920x1080 OfflineRT window and public Spawn materials/geometry. Repeated display
runs reset and infer fresh states. `--once --hold-seconds 0` exits after one run;
`--headless` runs without a window. `--capture-dir DIR` saves native camera images,
including offscreen captures with `--headless`:

```bash
python -m embodichain run-example learning/worldparticle demo \
  --scene cloth_on_ball --headless --capture-dir outputs/worldparticle/cloth-captures
```

Record a complete rollout to H.264 MP4 with `--video FILE.mp4`. Every simulation
step contributes one native camera frame; `--fps` controls video playback speed
without changing the checkpoint timestep. The initial and final frames each
hold for one second:

```bash
python -m embodichain run-example learning/worldparticle demo \
  --scene cloth_on_ball --headless --video outputs/worldparticle/cloth.mp4 --fps 15
```

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
Alternatively, pass `RegisteredNewtonSolverCfg(solver_type="worldparticle", options=...)`
to the same physics config; its installed entry point belongs to the EmbodiChain
distribution. Use the checkpoint dt/gravity and matching authored
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
