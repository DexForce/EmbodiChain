# Named Newton solvers and the NeRD lifecycle

Status: accepted design; implementation and qualification in progress.
Date: 2026-10-20. This document is the implementation and acceptance contract,
not a claim that the capabilities or numerical results already exist.

Executed experiments and remaining qualification gates are recorded separately
in [the evidence report](reports/neural-dynamics-20261020.md). The subsequent
[v2 completion audit](reports/neural-dynamics-implementation-v2-20261020.md) records
contract fixes, refreshed artifacts and remaining requirements.
The latest [bounded five-seed comparison](reports/neural-dynamics-bounded-five-seed-20261020.md)
and [historical rollout evidence](reports/neural-dynamics-historical-rollout-20261020.md)
cover the subsequent implementation work without claiming full qualification.
The [trained-model and clean-environment report](reports/neural-dynamics-trained-comparison-20261020.md)
records the Ant conversion correction, scratch-training comparison and the active
full-budget matrix at that report's snapshot time using official-version data.
The subsequent [CUDA training and full-v2 report](reports/neural-dynamics-cuda-alignment-20261020.md)
records actual optimizer/resume alignment and the frozen common-teacher,
disjoint-seed long-rollout protocol; full accuracy acceptance remains pending.
The [supported reference-evaluation report](reports/neural-dynamics-supported-reference-evaluation-20261020.md)
records the data provenance: both original training and held-out
teacher data are generated in the supported old environment and replayed through
separately recorded model runtimes.
The latest [original-entry small-budget comparison](reports/neural-dynamics-small-budget-20261020.md)
supersedes the full-budget execution plan for the current task. The user requested
four 1,000-update main runs and a workflow check; the forty-cell queues and their
evaluation waiters are stopped with artifacts preserved. That round does not
claim project-wide accuracy qualification or public-model convergence.
The [numerical attribution follow-up](reports/neural-dynamics-numerical-attribution-20261020.md)
adds 1,000-update same-environment parity for Ant, identifies CPU attention and
gradient-norm reduction differences, validates a separate common numerical policy
through the actual migrated trainer, and isolates Ant transport effects during
training. The official default baseline and production numerical defaults remain
unchanged.

## Objective and decisions

Implement the complete path from analytical simulation data through supervised
training, evaluation, export, named solver deployment, and differentiable control.
The first qualified robots are Cartpole and Ant. Both require CPU/CUDA forward
execution and gradients with respect to controls, initial state, recurrent history,
and neural model parameters. Training is single-device, supports training from
scratch, fine-tuning, and resume. Ant contact derivatives are local derivatives
of continuous features with discrete selections held fixed.

There is exactly one extension mechanism: installed Python entry points in
`dexsim.newton_solvers`, selected by `solver_type`. Do not add a `custom` solver
name, a provider path in deployment configuration, an instance-injection route,
or a public imperative registration API. Editable installs use the same path.

```toml
[project.entry-points."dexsim.newton_solvers"]
nerd = "embodichain.lab.sim.physics.solvers.nerd:NeRDProvider"
```

```yaml
physics: newton
physics_config:
  physics_dt: 0.016666666666666666
  num_substeps: 1
  requires_grad: true
  use_cuda_graph: false
  collision_cfg: null
  solver_cfg:
    solver_type: nerd
    options:
      checkpoint: /path/to/exported-model
```

One solver owns the complete scene. Solver-internal composition is permitted;
framework-managed cross-solver contact coupling is not in v1. CUDA Graph,
second-order gradients, physical-parameter identification, and multi-GPU DDP
are outside v1. NeRD-specific restrictions must not restrict other providers.

## Ownership and public contracts

DexSim owns discovery, construction, lifecycle, stepping, clock, reset and native
state publication. It must not import EmbodiChain. EmbodiChain owns the NeRD
provider, training and data workflows. Shared tensor mathematics and codecs must
not import simulation. Networks and offline training must work without loading
DexSim or initializing a simulator.

The DexSim extension protocol exposes:

- `RegisteredNewtonSolverCfg(solver_type, options)` retaining the actual name.
- `NewtonSolverProvider.register_custom_attributes(builder, options)` before
  model finalization, and `create(context, options)` after model, control and
  external contacts are available.
- `NewtonSolverExtension` containing the native `SolverBase` implementation,
  explicit device/differentiability/contact-report capabilities, optional
  functional execution, and a resource-release hook.
- `FunctionalSolverState` containing named physical tensors and named history
  tensors, including per-world valid history lengths. Physical tensor layouts
  must be explicitly documented; they cannot depend on articulation-count guesses.

Construction context supplies model, control, contacts, actual solver timestep,
device and scene index metadata. It is not an escape hatch to private backend or
World mutation. Provider objects and mutable histories are per-World instances.

Native methods retain Newton signatures: `step(state_in, state_out, control,
contacts, dt)`, `reset(state, world_mask=None, flags=None)`, and
`notify_model_changed(flags)`. Functional execution accepts explicit state,
control, parameters and dt, and returns a new functional state. The extension
must make trainable parameters accessible without serializing Python objects.

Discovery indexes entry-point metadata without importing unused implementations.
Duplicate names, collisions with built-in aliases, missing names and incompatible
interfaces fail with package provenance. Built-ins use the same resolver while
retaining their configuration compatibility and AutoSolver selection semantics.
Adding another extension must not require another hardcoded dispatch branch.

DexSim continues to schedule collisions, runtime controls and substeps. Reduced
coordinate solvers publish consistent body state through FK; the host must not
unconditionally overwrite maximal-coordinate solver results with FK. Reset is
idempotent, handles local/global world masks, and preserves unselected histories.
Topology replacement closes and rebuilds providers. Unsupported contact impulses
are unavailable, not reported as zero.

Add a separate `DifferentiableDynamicsEnv`, preserving the existing kinematic
`DifferentiableEnv` contract. Functional trajectories own immutable intermediates;
native ping-pong buffers are detached publication storage, never backward storage.
The native and functional NeRD paths share the same transition function. Explicit
Warp tape/VJP bridges connect FK to Torch; zero-copy views alone are not autograd.
Unsupported differentiable runtime controls fail before stepping. Simulation time
is advanced exactly once for executed substeps. Functional terminal-row reset
preserves old reward graphs and other rows; `detach_state()` is the explicit BPTT
boundary. Optimizer updates occur after backward and before a detached next rollout.

## NeRD model and state contract

Pin reference source: main `370e89d2b8ce68aa3578aa7812859c681b7502bc` for
generation/training; Newton preview `476dc67fccc6a99be293bf611f897b1c06f13e30`
for solver integration. The requested SDK MR !1483 artifact at `b61e920f` runs
Newton 1.7.0 / Warp 1.17.0; the initial integration used Newton 1.6.
Record the actual DexSim build, Torch, CUDA and driver identities in every report.

Use upstream-matching Cartpole/Ant assets and explicit joint/shape mapping. Bind
asset hashes, topology, inertia/collision intent, ordered joints, timestep,
coordinate conventions, history window and contact schema in a model manifest.
Validate these at deployment. Both reference models use dt=1/60 and one neural
substep. V1 NeRD consumes effort controls; unconverted position/velocity targets
or unmodeled external forces must not be silently ignored. Additional dynamic
objects or heterogeneous articulations require a separately supported model.

Asset identity alone does not establish physical equivalence. The isolated
official Warp importers and Newton 1.6 produce different masses, inertia and
friction from these assets. Persist the actual imported physical arrays and their
hashes in teacher provenance. Official weights must not be relabeled with a
current-Newton physics signature; current-Newton deployment uses a model trained
and validated for that domain. An exact historical-physics reconstruction would
require separate qualification.

Physical signatures have an explicit schema version. The current v2 contract
includes fixed-root orientation, parent/child topology, relevant body/joint flags,
robot and plane contact parameters, and the effective collision relationships.
Changing an omitted field must not evade validation. Contact-free Cartpole root
translations may vary under uniform gravity; fixed-root rotation may not. A new
signature schema requires regenerated/verified data and exports, not relabeling
an older checkpoint.

Implement a simulation-independent `NeRDStateCodec` shared by data, training,
evaluation and deployment. It handles Newton coordinates versus legacy twist,
body frames, relative targets, angle wrapping and quaternion increments. Targets
are expressed in the frame at t, not the frame at t+1. Ant conversion includes
the COM/reference-point shift, not only reordering velocity components.

Ant contact inputs use stable robot points projected onto a static plane. The
continuous geometry is differentiable. Mask decisions, wrap cuts and quaternion
sign choices have branch-local derivatives; tests exclude discontinuities when
checking finite differences. Neural contact features are not force reports.

Histories are per-world, with explicit valid lengths. Fixed-capacity eager buffers
must preserve upstream variable-prefix positional encoding and causal-attention
semantics after partial reset. Functional histories must not overwrite retained
graphs. Model parameters and normalizer statistics use standard Torch parameters
and buffers so dtype/device migration and serialization are complete.

## Data generation, storage and training

Generate trajectories through EmbodiChain scene ownership and DexSim analytical
solvers, with MuJoCo-Warp as the initial teacher default. Teacher and neural
deployments separately own physics configuration; share embodiment/assets rather
than overriding an environment component's physics settings at runtime. Record
teacher substeps and use a common macro timestep of 1/60.

Support sampled/default initial state, random effort and passive trajectories.
Translate upstream robot-specific sampling ranges to current state conventions.
Record state/contact features at t, apply the recorded effort through the macro
interval, then record state at t+1. Invalid numerical trajectories have bounded
retry and explicit rejection counts, not silent quality-based filtering.

Store chunked HDF5 shards with a manifest. Physical states have T+1 samples;
controls and contact inputs have T samples. Include q, qd, base poses, gravity,
actual generalized forces, contact data, trajectory IDs and valid lengths.
Record schema/codec versions, units, asset/config hashes, teacher versions and
sampling seeds. Use bounded staging/writer buffers and commit complete shards;
resume only committed work. Dataset workers open independent HDF5 handles and
read windows lazily by default. Do not materialize the expanded overlapping-window
dataset or a dense full-budget sampling index matrix. A single-process learner
may explicitly opt into raw CPU/CUDA field residency with a strict byte limit
and bounded host staging; it must fail before exceeding that budget. Repeated
window gathers retain independent ownership, and serialized worker datasets drop
resident caches. Cache placement cannot change preprocessing precision or samples.

Default dataset splits are trajectory-level 80/10/10. Windows cannot cross resets,
trajectories or splits. Normalization uses training data only and is frozen for
validation, test and inference. Official-parity experiments share exact official
statistics rather than comparing different statistical weighting schemes.

Provide `embodichain dynamics generate|train|eval|export|convert-checkpoint` with
lazy subcommand imports. Only generation/online evaluation start simulation.
Models belong to learning/models, supervised training/export to learning/dynamics,
storage to data_pipeline/dynamics, simulation sampling/adaptation to lab/sim.
Do not route supervised dynamics training through an RL trainer.

Port the causal Transformer and relative-state normalized MSE objective. Defaults:
history 10, six layers, twelve heads, embedding 192 (Cartpole) / 384 (Ant), Adam,
FP32, gradient norm clip 1.0, linear lr 1e-3 to 1e-4. Keep normalization
`sqrt(var + 1e-5)`. Training budget/batch are configurable. Upstream
`truncate_grad` means gradient clipping, not truncated temporal backpropagation.

Training checkpoints include network, weights, statistics, optimizer, scheduler,
global step, RNG state, committed sampler progress and dataset fingerprints.
Prefetch cannot advance the durable cursor. Resume checks the feature/data
contract. Export writes configuration, state_dict, buffers, manifest and checksums,
without optimizer or serialized Python objects. Newly trained exports load directly
through the named solver. A separate tool converts trusted upstream checkpoints.

## Official correctness and accuracy qualification

Qualification is a required deliverable. Passing smoke tests, finite gradients or
small-dataset overfitting does not establish parity with official implementation
or training results. Never mark missing runs as passed or silently change limits.

### Reference identity and fixtures

Freeze source commits, asset/checkpoint/dataset hashes, precision, software and
hardware. Preserve an unmodified reference checkout outside the production
package. Reference adapters may change data transport or compatibility boundaries,
not the algorithm. Record their patches. Experimental initialization,
normalization, sampling and checkpoint-selection controls must be recorded
separately; a controlled comparison with changed controls is not proof of literal
reference-recipe equivalence. No test imports downloaded code implicitly.

The official sampler allocates gravity features with `torch.empty` and writes only
the up-axis component. Preserve raw output and fail on invalid components unless
an explicit gravity-transport repair is enabled. Record repaired component counts,
the physical gravity direction and hashes of raw and transported data. Label this
reference as corrected transport, not untouched official data. Likewise record
the Y-up to Z-up rotation and origin-twist to COM-velocity conversion explicitly.

Export deterministic fixtures for preprocessing, history lengths 1/9/10, partial
reset, normalization, network output, relative targets and decoded next state.
Use identical weights, state, contacts, effort and valid history. In FP32 on the
same device start with atol=1e-5 and rtol=1e-4; every discrepancy must be localized
before changing a threshold. Do not require bitwise CPU/CUDA identity.

### Training-step equivalence

Use identical initialized weights, minibatches, normalizer statistics and optimizer
state. Compare encoded inputs/targets, loss reduction, parameter gradients, clipping,
one Adam update and a deterministic short training run. Freeze learning-rate update
timing and checkpoint selection. In particular, official statistics computed over
overlapping sampled windows differ from unique-transition statistics: share the
same statistics artifact for parity experiments and label other profiles separately.

### Matched-data training matrix

Run both trainers against both immutable datasets:

| Teacher dataset | Official trainer | EmbodiChain trainer |
|---|---|---|
| Official Warp-generated data | Reference | Training-port qualification |
| Current DexSim/Newton-generated data | Format-adapted reference | Deployment qualification |

Within each row fix train/validation/test partitions, initialization seeds,
normalization, sample order, optimizer update budget, network architecture and
selection metric. Run five training seeds. Keep actual configured update counts
in reports; a shortened run must be labeled and cannot stand in for full-budget
qualification. Paper reproduction and current-teacher qualification are distinct:
the official Ant paper baseline requires Warp 1.5.1, not current Newton physics.

The current matrix is a controlled training-port experiment, not a literal
reproduction of the paper's full data recipe. It shares frozen statistics and
sampling indices and uses one common validation-selection protocol. Its default
replacement sampling, window-population statistics and single ordered validation
split differ from upstream shuffled/drop-last loaders, RunningMeanStd accumulation,
epoch-zero eligibility and separate experiment/passive validation selections.
Those differences must remain visible even if the matched experiment passes.
Matching architecture and optimizer budget does not by itself establish paper
reproduction. Inspect the ordered feature list in the actual immutable initial
model artifact; sorted experiment JSON mappings cannot prove feature order.

Historical-domain rollout qualification runs in a fresh process with the pinned
original Warp environment and Transformer neural integrator. Restore portable
weights from either trainer into the same original network implementation and
frozen normalizers. This controls the physics/runtime domain while measuring the
training port. Validate recorded original physical arrays, source and asset
identity before running. Use explicit canonical coordinate/velocity transport
only for common policies and metrics. This does not imply that the historical
model can be deployed through the current Newton reference assets.

The historical runner is an explicitly trusted reference tool, separate from
the installed solver entry-point mechanism. It must not replace the installed
Warp package or import DexSim/Newton into the old-Warp process. Each simulator
recomputes fixed-policy actions from its own state. Record which reference
runtime executed the network; only current-domain tests establish the native
Newton/functional adapter behavior.

Historical comparison teacher data must also be generated in the declared
supported original environment. Preserve raw and canonical states/controls
without filtering failures, and reuse immutable traces across models. Verify
generation versions, source/asset/physical arrays, coordinate transport and
teacher feedback. Regenerating reference data under a different framework
version requires a separate explicit scope. The model evaluation runtime is
identified independently.

### Evaluation and acceptance

For Cartpole and Ant evaluate passive motion, random effort and fixed-policy
closed-loop trajectories. Use 2048 common test trajectories at horizons 1, 10,
100 and 1000. Report timestep error curves, q/qd errors, Ant base-position error,
quaternion geodesic error, non-finite/failure rates and policy return. Periodic
angles use wrapped error; orientations are not evaluated only by component MSE.

Single-step implementation/gradient/update parity is a hard gate. For matched-data
training, the one-sided 95% confidence upper bound of paired relative degradation
in predeclared primary errors must not exceed 5%. Use seed-level paired bootstrap
(10,000 resamples, fixed seed 0), not independent per-timestep samples. If the
official metric is effectively zero, report the absolute difference and use the
predeclared numerical-parity tolerance rather than an unstable ratio. Failures,
non-finite states and policy returns are separately reported; an improved average
cannot hide instability. Validation chooses checkpoints; test never selects them.

Before computing paired statistics, verify source/model/trace hashes, the robot,
teacher domain and test split. Paired runs must identify the same ordered teacher
trajectory content and fixed policy, independent of output paths or batching.
Numerical qualification requires measured residuals and pinned reference identity;
an unbound pass flag is insufficient. Missing or undefined errors remain explicit
and must not be replaced with zero or excluded to improve acceptance rates.

Artifacts include machine-readable results and manifests, reproduction commands,
fixtures, training curves, rollout error plots and a human-readable comparison
report. Report unsupported/unavailable reference environments as pending. Do not
assert equivalence to paper results from a different teacher or training budget.

## Implementation work packages and completion ledger

1. DexSim named discovery, public contracts, lifecycle/functional execution and
   focused regression tests; package metadata must work in the installed artifact.
2. Shared codec, causal network, manifests, checkpoint conversion and reference
   fixtures with feature/forward/gradient tests.
3. Sharded datasets, trainer/resume/export, CLI and data/training tests.
4. EmbodiChain configuration, NeRD provider, simulation data generation,
   differentiable environment and Cartpole/Ant deployments.
5. End-to-end data→train→eval→export→native/functional simulation validation,
   official matched-data qualification and benchmark/quality reports.
6. Public API documentation, solver/training guides and routed project context.

All items remain pending until backed by authoritative code and executed results.
The implementation log must distinguish smoke, numerical-parity, full-budget
quality, performance and regression evidence. Project-wide qualification requires
all its artifacts. A user-selected bounded round has its own completion conditions
and does not confer project-wide qualification.

Current evidence status (2026-10-21): packages 1–4 and 6 have implementation,
focused runtime/tests and scoped artifact evidence. Package 5 includes native
end-to-end smoke, numerical parity, original-entry small-budget controls and
supported-environment historical rollouts. The user stopped the 40-cell
full-budget matrix and dependent waiters; its checkpoints, logs and configurations
remain preserved. The subsequent one-seed-per-robot round uses original supported
Warp data, captured original shuffle order and explicit reference/math numerics.
Cartpole completed 4,995,000 actual updates and passed fixed offline acceptance;
its validation-selected best is update 4,940,000. Ant final-budget acceptance is
still pending in this source snapshot. Historical-domain success does not qualify
current-Newton deployment. Larger multi-seed/long-rollout qualification remains
separate from this round, and neither comparison establishes same-data public
model convergence reproduction. The [current evidence index](reports/neural-dynamics-pr-status-20261020.md)
links the reports and records their scopes.
