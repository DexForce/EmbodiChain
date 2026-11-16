# WorldParticle

The model layer lives in `embodichain_models.dynamics.worldparticle`: original
network, strict checkpoint loading, authored features/topology and coordinate
conversion. It imports neither simulation nor examples. The integration layer
is `embodichain_solvers.physics.worldparticle.provider.WorldParticleProvider`; the native
solver adapter is DexSim's model-independent `ParticleInferenceSolver`.
Providers are constructed per scene through the existing Newton lifecycle.

Start with the [runtime requirements](#runtime-requirements),
[native solver quickstart](DEPLOYMENT.md) and [input contract](INPUTS.md).

## Runtime requirements

Use Linux x86_64 / Python 3.12, CUDA 12.8, CXX11 ABI and an SM120 GPU with
SDK 0.6.0rc2. The current project profile uses Torch 2.11.0+cu128, Torchvision
0.26.0+cu128, NumPy 2.2.6 and Einops 0.8.2. Supply Open3D
`0.19.0+1e7b1743` and FlashAttention `2.8.3` operators built against that same
Torch/CUDA profile. Operator build/patch tooling belongs to those dependencies
and is outside this model integration.

Compatible operator wheels are not bundled or published by this PR, and there
is no `worldparticle` dependency extra. A default project installation alone
does not provide these binaries. The loader checks recorded versions, ABI and
GPU architecture before loading weights; package versions alone do not ensure
that native symbols match Torch. Check your provisioned environment before
running the examples:

```bash
python -c 'import torch, open3d.ml.torch, flash_attn; from dexsim.engine.newton_physics.particle_inference import ParticleInferenceSolver; print(torch.__version__, torch.cuda.get_device_capability())'
```

An undefined-symbol or missing-kernel error requires a matching operator build,
not weaker loader checks. The older standalone Torch 2.7.1+cu128 / Open3D
`0.19.0+1e7b174` combination remains recognized by the loader; it is not a
replacement for the project's Torch 2.11 environment.

## Independent transition and optional deployment

After preparing the default asset directory, run this independent tensor call
from the repository root in the checked CUDA environment:

```python
from pathlib import Path
import numpy as np
import torch
import yaml
from embodichain_models.dynamics.worldparticle import load_model

assets = Path("outputs/worldparticle/assets").resolve()
config_path = assets / "checkpoints/fancy_fluid/config.yaml"
sample = assets / "data/fancy_fluid/66/66.npz"
cfg = yaml.safe_load(config_path.read_text())
R = np.asarray([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], np.float32)
options = {
    "checkpoint": str(assets / "checkpoints/fancy_fluid/final.ckpt"),
    "config": str(config_path),
    "inputs": str(sample),
    "model_to_world": R.tolist(),
}
with np.load(sample, allow_pickle=False) as data:
    frame = int(cfg.get("frame_start", 0))
    q = torch.tensor(data["position"][frame] @ R.T, dtype=torch.float32, device="cuda:0")
    qd = torch.tensor(data["velocity"][frame] @ R.T, dtype=torch.float32, device="cuda:0")
acceleration = torch.tensor(R @ np.asarray(cfg["gravity"], np.float32), device="cuda:0").expand_as(q)
model = load_model(options, device="cuda:0")
with torch.no_grad():
    next_q, next_qd = model(q, qd, acceleration, float(cfg["timestep"]))
assert torch.isfinite(next_q).all() and torch.isfinite(next_qd).all()
print(next_q.shape, next_qd.shape)
```

Acceleration includes gravity. The model owns
static normalized features, authored CSR topology and the frame rotation; it
does not read native state or apply forces. World/frame transport is FP32,
while the original frozen network uses its recorded BF16 inference mode.
Offline preparation remains independent of native deployment.

The optional Provider delegates buffers, streams, force-to-acceleration
conversion, reset and native writes to the SDK's model-independent adapter.
The model and Provider share one transition implementation. There is no
WorldParticle learning-env or functional-gradient claim in this stage.

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

Once the runtime is qualified, application scenes can pass `ProviderNewtonSolverCfg(provider_factory=
WorldParticleProvider, options={"checkpoint": ..., "config": ..., "inputs": ...})`
to `NewtonPhysicsCfg(solver_cfg=..., use_cuda_graph=False, collision_cfg=None)`.
Use the direct SDK `SceneBuilder` carrier shown by the demo. Neural boundaries
are authored in the input NPZ and native ground/collision shapes must be absent.
`SimulationManager.add_fluid_object` owns DexUni/SPH fluid semantics and is not
a generic WorldParticle particle carrier; that route remains deferred.
Alternatively, pass `RegisteredNewtonSolverCfg(solver_type="worldparticle", options=...)`
to the same physics config; its installed entry point belongs to the EmbodiChain
distribution. Discovery does not qualify a runtime. Use the checkpoint dt/gravity
and matching authored particle order. The Provider validates count, not particle
identity or a permutation. Only a single CUDA particle world is supported, without native
rigid collision shapes, gradients or CUDA Graph capture. Static boundaries and
features come from model inputs. Changing them requires closing/recreating the
scene; reset restores physical state and does not reload model assets.

## Verification

On implementation commit `b0875186c`, the Torch 2.11 / SDK rc2 / SM120 setup
above completed the four default headless budgets (199 / 80 / 15 / 199 steps)
with finite state and normal exit. Each scene also produced two 1920×1080 RGB
offscreen captures in a separate two-step run. Reproduce inference with
`demo --scene SCENE --headless --once --hold-seconds 0`; add
`--steps 2 --capture-dir DIR` for the bounded offscreen check.

These runs do not qualify desktop window interaction, accuracy, historical
runtime equivalence, native/model numerical parity or bitwise replay. The
unchanged network contains CUDA reductions/hash-neighbor construction without
an established determinism guarantee. Runtime/unit checks separately cover
CSR validation, FP32 transport under autocast, force units, native reset and
unsupported-configuration rejection.

## Licensing

WorldParticle's root MIT license and provenance are included in the model package.
Its vendored ToMe merging component retains CC-BY-NC-4.0 (`LICENSE.tome`).
Moving this code does not grant commercial redistribution permission. The model
code currently ships with the EmbodiChain wheel; this integration has not been
commercially cleared. Weight/data rights also need independent release review.
