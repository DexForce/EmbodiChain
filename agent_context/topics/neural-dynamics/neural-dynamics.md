# Neural dynamics models and training

## Entry points and ownership

The main wheel bundles `embodichain.extensions.models` with the same version and dependencies.
It is an import package under `extensions/`, not a separate distribution or a
runtime registry. Shared MLPs and feature normalization stay in core
`embodichain.learning.models`; generic supervised orchestration stays in
[SL](../supervised-learning/supervised-learning.md).

The [model catalog contract](../../../embodichain/extensions/models/README.md)
owns domain and public-import naming. Policy models predict actions; dynamics
models predict future state or geometry. Canonical RWM and RigidFormer paths are
`dynamics.rwm` and `dynamics.rigidformer`; the catalog tracks implemented families.
RWM ownership and graph/data contracts are in
[robot world model](../robot-world-model/robot-world-model.md).
World-model naming does not introduce a parallel `world_models` domain.

| Request | Owner |
|---|---|
| NeRD network, portable checkpoint, physical codec | `embodichain/extensions/models/dynamics/nerd/{model,checkpoint,codec}.py` |
| Online canonical state/history, injected FK/contact geometry and prediction | Model package `transition.py`, `geometry.py` |
| Lightweight Cartpole/Ant learning factories | Example `_support/{_cartpole_env,_ant_env,_model_env,_model_factory}.py` |
| Optional native Provider, indices and physics admission | `embodichain/extensions/solvers/physics/nerd/` |
| Committed HDF5 shards and bounded windows | Model package `data.py`, private `_resident.py` |
| Encoding, frozen moments and offline errors | Model package `integration.py` |
| Objective and common trainer factory | Model package `supervised.py`, `sl_recipe.py` |
| Prepare/train/export/eval/generate commands | `examples/learning/nerd/__main__.py`; `embodichain run-example learning/nerd` |
| Pinned datasets, exports and reference assets | Example `_resources.py`; explicit `prepare` command |
| Analytical teacher and physical profile | Example `_support/{_generation,_newton,_reference_assets}.py` |
| Historical trainer and checkpoint workflow | Example `_support/{_trainer,_workflow}.py` |
| Original-repository comparison and acceptance | Example `_support/` runners; protocol in `research/` |

The showcase defaults to `sl.train.yaml` and the installable
`embodichain.extensions.models.dynamics.nerd.sl_recipe:build_supervised_trainer` factory.
Direct `embodichain train-sl --config FILE` works independently of example
bootstrap. An explicitly selected `dynamics.train.*.yaml` uses the historical
trainer for controlled comparisons and its legacy checkpoint contract.

The model extension imports neither examples nor simulation. Generation uses Newton's
built-in Featherstone/MuJoCo solvers directly without a DexSim custom solver or
Scene API. Native learned-solver deployment and lightweight learning environments
are separate consumers of the same independent transition. There are no NeRD-specific core `compute/dynamics`,
`data_pipeline/dynamics`, `learning/dynamics` modules or core `dynamics` command.

Online rollout owns canonical `q`, COM `qd`, named history and valid lengths in
`NeRDState`. `NeRDTransition.step` computes contact conditioning from that same
state through injected FK; geometry caches only detached fixed tensors. Offline
SL continues to accept measured contact features. Learning reset/terminal/TBPTT
semantics use the existing [learning contract](../rl-learning/rl-learning.md).
Full reset creates independent episode graphs; partial reset preserves peers.
The optional native adapter retains indexing, physical admission and publication.
The FK-only reference factory isolates SDK descriptor lowering without creating
a World, Scene or solver; signatures guard that SDK compatibility surface.
Its injected Newton backend uses the private `lab.sim.diff._newton_fk` gradient
adapter. Reusable tensor FK/IK remains owned by `compute.kinematics` and
`learning.layers`; the native adapter does not introduce another public FK API.
The example's [native quickstart](../../../examples/learning/nerd/DEPLOYMENT.md)
owns complete resource/configuration/step/reset/close instructions;
[validation](../../../examples/learning/nerd/VALIDATION.md) owns measured budgets,
runtime identity and viewer limits.

## RigidFormer author model

The native family lives in
`embodichain/extensions/models/dynamics/rigidformer/`.
`model.py` owns geometry/configuration; `_native/` preserves licensed author
code. Checkpoint import validates recorded architecture and named tensors.
The community `rigidformer` package is not a runtime dependency.
`data.py` reads verified packed scenes; `supervised.py` separates shifted
conditioning and anchor targets; `sl_recipe.py` constructs the shared trainer
with committed sampling, epoch scheduling and EMA participants.

Integration dt=1 differs from physical seconds. Stored vertices are unscaled;
metadata sizes apply about each origin once. Geometry prediction requires CUDA
and at least two valid objects. Attention backend and source semantics belong
to recipe/recovery identity. Source-owned epoch sampling does not claim the
legacy research DataLoader's worker-RNG trajectory.

The [showcase](../../../examples/learning/rigidformer/README.md) owns downloads,
conversion, fixed-weight evaluation and independent author-Trainer comparison.
Its `demo` command in `_demo.py` compares live Newton spheres with render-only
model predictions after four observed frames, through a native window or an
engine-camera recording. It uses explicit prepared geometry and weight paths.
Assets/results stay outside the wheel. Numerical agreement, recovery and full
benchmark/retraining qualification remain distinct evidence requirements.

## WorldParticle frozen particle transition

`dynamics.worldparticle.WorldParticleModel` owns the stateless world-space
particle transition, authored features/boundaries/CSR topology, checkpoint
loading and model-frame conversion. Its package imports neither simulation
nor examples. Coordinate transport stays FP32 even under caller autocast;
the original frozen network uses BF16. No learned-robot codec or history
contract is shared with NeRD. Both families use the existing optional Newton
Provider lifecycle in `embodichain.extensions.solvers.physics` for native deployment.

`physics.worldparticle.provider.WorldParticleProvider` owns native admission and
delegates stream/state/force/write/reset machinery to the SDK's generic
`ParticleInferenceSolver`, included in SDK 0.6.0rc2. The example's
[native quickstart](../../../examples/learning/worldparticle/DEPLOYMENT.md),
[input contract](../../../examples/learning/worldparticle/INPUTS.md) and
[runtime/verification summary](../../../examples/learning/worldparticle/README.md)
own reproduction details and checked budgets. Compatible Open3D/FlashAttention
operators must be provisioned separately; dependency build/patch tooling is
outside this model integration.
Tests with a small deterministic predictor
qualify adapter behavior; real BF16 inference does not establish bitwise
determinism, native/model parity or historical/paper accuracy.

Only one CUDA particle world with static authored network boundaries is
supported. It does not provide functional gradients/APG, dynamic rigid-body
coupling, native contact reports or CUDA Graph capture. Keep its learning-env
adapter deferred until an actual differentiable training contract is qualified.
Preparation and four forward demonstrations belong to
`examples/learning/worldparticle`; weights, data and native wheels stay external.

## Resource and physical identity

`prepare --robot cartpole|ant --resource-root DIR` downloads the pinned dataset,
portable model and raw reference assets. Revisions and SHA256 checks are owned by
`_resources.py`; `--assets-only` skips data/model downloads and
`--local-files-only` forbids network access. The default resource root is
`EMBODICHAIN_NERD_RESOURCE_ROOT`, otherwise the showcase cache under
`XDG_CACHE_HOME` or `~/.cache`. Downloads do not live in Git or the wheel; model
third-party licensing remains in the model package.

A trajectory has T+1 states and T effort/contact inputs. Splits belong to whole
trajectories; windows never cross episodes or splits. Only manifest-committed
shards are visible. Generation commits its cursor and rejection evidence with
shards; rejection-only progress preserves data fingerprints. Legacy progress
sidecars remain readable, while collector source and physical identity still
must match to continue collection. A path move does not authorize cross-version
resume or reconstruct missing historical rejection evidence.

The direct teacher records importer profile, asset/physical-array hashes, solver
settings and actual Newton/Warp versions. Its `nerd-direct-newton-builder-v1`
signature differs from published Scene-generated data, which remains readable
without simulation. `warp_reference` reconstructs historical asset parameters
and Ant friction mapping; matching that name does not prove identical physics
across model-building paths, versions or solvers.

The codec owns COM-to-spatial-twist transport and relative targets in the frame
at t. Ant uses 25 ordered plane-contact slots; Cartpole is contact-free. Controls
are generalized efforts with zero free-base force. Frozen train-only moments
are reused for evaluation, and supplied statistics must bind the training data
fingerprint. Do not relabel physical records or replace their normalizers.

## Training and validation

SL checkpoints use the common trainer format and resume through the same
recipe/signature. The historical trainer keeps `embodichain.nerd.training.v1`:
`last.pt` retains optimizer, scheduler, RNG, committed sampling state and one
owned CPU snapshot of the selected best model; automatic `best.pt` is export-only.
Both trainers use shared run-directory/history admission from SL. Common `train`
does not turn its final `last.pt` into a validation-selected model.

The showcase `export` accepts common SL and historical NeRD checkpoints and
writes the same portable tensor/JSON model format. Precision, ordered feature
layout, normalization and attention policy are part of recovery/export identity.
Historical fine-tuning and shared sample-order options require a historical
configuration. Only explicitly trusted legacy conversion unpickles model objects.

Frozen source revisions, budgets and gates live in the
[acceptance protocol](../../../examples/learning/nerd/research/acceptance.md).
Historical evidence remains bound to its recorded source, data and runtime;
relocating code does not rerun it. NeRD validation uses the runnable showcase
and its reference/acceptance workflows; `tests/learning/sl/` covers the common
training contracts. The [feature overview](../../../docs/source/features/neural_dynamics.md)
owns model selection and the shared training entry point; the
[NeRD guide](../../../docs/source/features/neural_dynamics/nerd.md) and
[RigidFormer guide](../../../docs/source/features/neural_dynamics/rigidformer.md)
own their model principles and user workflows; the
[RWM guide](../../../docs/source/features/neural_dynamics/rwm.md) covers prediction
and replay under the linked robot-world-model contract. The
[WorldParticle guide](../../../docs/source/features/neural_dynamics/worldparticle.md)
owns frozen inference, operator provisioning and native demo navigation. The
[model API](../../../docs/source/api_reference/model_catalogs.rst) owns exported
definitions. Transition error, accumulated rollout error and paper-scale
accuracy remain separate claims.
