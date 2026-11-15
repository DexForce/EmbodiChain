# WorldParticle: Frozen Particle Dynamics

## Use cases and ownership

WorldParticle predicts particle positions and velocities from the current
state, total acceleration and fixed authored features/boundaries/topology.
The family preserves the original frozen networks and published checkpoints
for Fancy Fluid, Cloth on Ball, Duck forward deformation and Deep Fluid.
Its [source provenance](https://github.com/DexForce/EmbodiChain/blob/main/extensions/embodichain_models/embodichain_models/dynamics/worldparticle/PROVENANCE.json)
records the upstream revision and retained licenses, including the separate
ToMe notice.

| Owner | Responsibility |
| --- | --- |
| `embodichain_models.dynamics.worldparticle` | Frozen network, strict weights, feature normalization, topology and coordinate conversion |
| `embodichain_solvers.physics.worldparticle.provider` | Native scene admission and the optional `worldparticle` Provider |
| SDK `ParticleInferenceSolver` | Streams, native buffers, force-to-acceleration conversion, writes, reset and close |
| `examples/learning/worldparticle` | Explicit resource preparation, operator-build provenance, scene construction, demos and validation |

The model runs without Newton/DexSim imports. Native deployment calls the same
model through a thin Provider. NeRD reuses the existing Provider lifecycle but
has a separate articulated-state/history contract. WorldParticle currently has
no shared-SL recipe, learning-env or APG/functional-gradient contract.

## Runtime and inputs

The checked profile is Linux x86_64 / Python 3.12, Torch 2.11.0+cu128,
CXX11 ABI, SDK 0.6.0rc2 and SM120. Open3D `0.19.0+1e7b1743` and FlashAttention
2.8.3 need the corresponding Torch/CUDA build. Historical Torch 2.7 wheels do
not replace main's Torch 2.11 operators. Compatible wheels remain local build
artifacts, without a published `worldparticle` installation extra.

Follow the [build, install and CUDA preflight recipe](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/worldparticle/BUILDING.md)
before loading model assets or creating a World. It checks versions, the SDK
adapter and actual operator execution; other architectures are not qualified.

`load_model(options, device="cuda:0")` takes local `checkpoint`, `config` and
`inputs` paths, plus optional proper 3×3 `model_to_world` rotation R. Fixed NPZ
inputs are raw `(N, C)` features, nonempty `(M, 3)` boundary positions/features
and, for topology-enabled checkpoints, integer CSR neighbors/row splits.

Call `next_q, next_qd = model(q, qd, acceleration, dt)` with world-space
float32 CUDA tensors of shape `(N, 3)` in the authored particle order. Position,
velocity and acceleration use metres, metres/second and metres/second². Total
acceleration includes gravity; the SDK adds gravity to force/mass once. The
checkpoint dt is in seconds and must match native stepping. Transport is FP32;
the source network uses BF16.

For row vectors, `q_world = q_model @ R.T`; boundary NPZ fields remain in model
coordinates and native gravity is `R @ config_gravity`. The Provider validates
count, not a permutation or geometry identity. The
[input contract](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/worldparticle/INPUTS.md)
defines all fields, feature transforms, task-name aliases, frame/stride rules
and reset/admission constraints. The
[README](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/worldparticle/README.md)
contains a complete standalone tensor example with explicit options.

## Prepare and run

In the checked environment, from the repository root:

```bash
python -m embodichain run-example learning/worldparticle prepare
python -m embodichain run-example learning/worldparticle demo --scene fancy_fluid --headless --once --hold-seconds 0
python -m embodichain run-example learning/worldparticle demo --scene cloth_on_ball --headless --once --hold-seconds 0
python -m embodichain run-example learning/worldparticle demo --scene inverse_design --headless --once --hold-seconds 0
python -m embodichain run-example learning/worldparticle demo --scene deep_fluid --headless --once --hold-seconds 0
```

Assets default to `outputs/worldparticle/assets`. Preparation needs the existing
internal resource service when files are absent and verifies all 19 hashes/sizes.
Use `--assets DIR` for another prepared location; `prepare --local-files-only`
forbids downloads. Predicted NPZ state and JSON reports stay outside Git/the wheel.

The [native solver quickstart](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/worldparticle/DEPLOYMENT.md)
shows a complete Fancy Fluid World/Scene/Provider with step/readback/reset/close
and the registered-name alternative. The demos use direct SDK SceneBuilder;
the manager's `add_fluid_object` retains its DexUni/SPH semantics. Cloth keeps
its authored surface; other demos use an unconnected-particle carrier.

## Execution evidence and limits

At implementation commit `b0875186c` on 2026-11-15, default headless runs
completed 199 / 80 / 15 / 199 steps for Fancy Fluid / Cloth / Duck / Deep Fluid,
with finite state and normal exit. Each scene also produced two 1920×1080 RGB
offscreen captures in a separate two-step run. This supplements the earlier
operator/standalone checks preserved in the runtime manifests.

The [execution record](https://github.com/DexForce/EmbodiChain/blob/main/examples/learning/worldparticle/VALIDATION.md)
gives the environment, commands and measured scope. Desktop window interaction
and new full-length video encoding were not qualified in this run. Finite
rollout/captures do not establish accuracy, historical equivalence, native/model
numerical parity or bitwise reproducibility. CUDA reductions/hash-neighbor
construction in the unchanged source have no established determinism guarantee.

One CUDA particle world with static model boundaries and forward inference/reset
is supported. Dynamic rigid coupling, native contact-force reports and CUDA
Graph capture remain outside the scope. Duck's `inverse_design` name means a
forward-deformation demo here; Deep Fluid uses generated initial conditions
without an official reference trajectory.

See the {doc}`model API </api_reference/model_catalogs>` for public imports.
