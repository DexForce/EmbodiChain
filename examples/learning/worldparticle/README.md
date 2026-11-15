# WorldParticle

The model layer lives in `embodichain_models.dynamics.worldparticle`: original
network, strict checkpoint loading, authored features/topology and coordinate
conversion. It imports neither simulation nor examples. The integration layer
is `embodichain_solvers.physics.worldparticle.provider.WorldParticleProvider`; the native
solver adapter is DexSim's model-independent `ParticleInferenceSolver`.
Providers are constructed per scene through the existing Newton lifecycle.

## Runtime qualification

The historical PR demo used a dedicated Python 3.12 environment with `dexsim_engine==0.6.0rc2`.
Linux x86_64, glibc >=2.35, Python 3.12,
Torch 2.7.1+cu128/CXX11 ABI and SM120 form that historical operator profile.
NumPy is 2.2.6; Torchvision matches Torch. The published CPython 3.12 SDK wheel
contains `ParticleInferenceSolver`.

`runtime-manifest.json`, the Open3D patch and relocation utility retain that
profile's immutable build provenance. They are not an installation profile for
the main distribution. The main distribution's
Python 3.12 training profile currently requires Torch 2.11 and Tensordict 0.15,
which conflicts with those historical model operators. Main now pins SDK
0.6.0rc2, including the generic `ParticleInferenceSolver` for native deployment.
The model loader rejects an unqualified runtime before loading weights; a
missing SDK facility fails explicitly.

Open3D and FlashAttention have also been rebuilt against Torch 2.11.0+cu128.
`runtime-manifest.torch211.json` records the local wheel identities and
[BUILDING.md](BUILDING.md) gives the source recipe. CUDA operator checks and
two-step inference with all four published models passed, both standalone and
through SDK 0.6.0rc2. Standalone execution uses no Newton/DexSim imports and preserves
input storage. Native adapter transport/reset is separately tested with a
deterministic predictor. Real BF16 network rollouts do not establish bitwise
determinism, native/standalone numerical parity, historical-runtime equivalence
or long-horizon accuracy.

No conflicting `worldparticle` dependency extra is provided. This integration
preserves main's Torch 2.11 / SDK 0.6.0rc2 requirements. The Torch 2.11 binaries
remain local artifacts; operator distribution must be resolved before
advertising a default installable native runtime. Source-only
experiments with the historical profile must use an isolated environment and
the recorded wheel hashes. No environment, weight or native wheel is committed.

## Independent transition and optional deployment

Load an independent model and call it with world-space particle tensors:

```python
from embodichain_models.dynamics.worldparticle import load_model

model = load_model(options, device="cuda:0")
next_q, next_qd = model(q, qd, acceleration, dt)
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

## Licensing

WorldParticle's root MIT license and provenance are included in the model package.
Its vendored ToMe merging component retains CC-BY-NC-4.0 (`LICENSE.tome`).
Moving this code does not grant commercial redistribution permission. The model
code currently ships with the EmbodiChain wheel; this integration has not been
commercially cleared. Weight/data rights also need independent release review.
