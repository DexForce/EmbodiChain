# Neural dynamics

## Entry points and ownership

| Request | Owner (relative to `embodichain/`) |
|---|---|
| Named solver selection | `lab/sim/cfg/physics.py`; installed `dexsim.newton_solvers` metadata |
| NeRD native/functional adapter | `lab/sim/physics/solvers/nerd.py` |
| Scene indexing and source assets | `lab/sim/physics/solvers/_nerd_binding.py`, `_reference_assets.py` |
| Tensor state/contact codec | `compute/dynamics/codec.py` |
| Shared rollout error/content calculations | `compute/dynamics/_evaluation.py` |
| Network and portable checkpoints | `learning/models/nerd/` |
| Teacher collection and rollout evaluation | `lab/sim/physics/solvers/collection.py` |
| Shard persistence and sequence windows | `data_pipeline/dynamics/shards.py` |
| Supervised training, resume and export | `learning/dynamics/trainer.py`, `workflow.py` |
| Official comparisons | `learning/dynamics/reference.py`, `comparison.py`, `report.py` |
| Historical original-Warp rollout | `learning/dynamics/reference_rollout.py` |
| Command dispatch | `cli/dynamics.py`: `embodichain dynamics` |

DexSim owns the generic registry and provider contracts. EmbodiChain registers
`nerd` in `pyproject.toml`; third-party packages use the same entry-point group.
There is no direct provider-path deployment or imperative registration route.
Discovery does not import implementations. Unknown/duplicate names fail instead
of falling back. A DexSim build exposing the extension and Scene snapshot/
functional APIs is required.

These physics solvers are separate from FK/IK in `lab/sim/motion/solvers`.
Creation follows [simulation ownership](../simulation-system/simulation-system.md).
Providers receive model-owned context; tasks/collectors use public manager/Scene
APIs, not native backend mutation.

## Data and state contract

`DynamicsSimulation`, generation and rollout configs accept an explicit
`physics_profile: native|warp_reference`. The default retains modern importer
semantics. `warp_reference` reconstructs measured legacy asset mass/inertia,
body armature, force-mode damping and ordered capsule contact slots before
Spawn materialization; its shared ground has explicit material parameters.
It still runs current Newton and records that teacher identity. Featherstone
teachers use complete public articulation writes at macro boundaries; their
mass-matrix update interval must divide the macro substep count. Old and new
contact algorithms can differ even after parameter matching. The isolated
`scripts/experiments/neural_dynamics/physics_alignment.py` runner preserves
identical canonical inputs and hash-verified traces for that comparison.

NeRD validates supported homogeneous articulations, named joint mapping, trained
timestep, gravity, physics signature and contact schema. Reference Cartpole is
contact-free: disable the physical default plane at manager creation, not only
its visibility. Ant uses a free base and fixed plane-contact candidates. Controls
are generalized efforts; unsupported base/external forces are rejected.

Newton COM velocities and NeRD spatial twists differ in both component ordering
and reference point. The codec owns both conversions. Relative targets use the
frame at t, not t+1. Training and runtime share the codec and frozen normalizers.
Quaternion arithmetic and scripted body-transform execution are part of FP32
reference alignment. Training targets and native decoding reuse the current
unnormalized embedding rather than transforming that state twice. CUDA graph
specialization can otherwise change low-order bits within one transition.
Periodic positions are wrapped before differencing, matching the original
trainer; loss uses its literal reciprocal-sqrt and native MSE reduction.
Reference adapters reuse the codec's velocity transport on the model device,
so raw CPU/CUDA residency cannot change preprocessing precision.
For original-data training, `import_legacy_datasets` can explicitly preserve
`states/legacy_velocity` before COM conversion. This unrotated FP32 channel
requires bit-exact transition continuity and supplies both current and next
velocity directly to the codec; native `q`/`qd` and the default import retain
their contracts. Source fidelity does not qualify a model for modern physics.
Compare both implementations in the same floating-point world frame. Comparing
an original Y-up result with its Z-up input rotation also measures the reference's
own finite-precision frame non-invariance.

Committed HDF5 shards hold T+1 states and T controls/contact inputs with provenance.
Splits are trajectory-level; windows cannot cross episodes or splits. Workers own
independent file handles and bounded caches. Long single-process learners can
explicitly budget raw CPU/CUDA residency through `resident_cache_bytes`; cache
gathers preserve precision, padding and returned ownership and fail before
exceeding the field budget. Serialized worker datasets drop that cache. Offline
training does not initialize simulation. Teacher/neural deployments separately
own physics configuration.

Training checkpoints retain optimizer, scheduler, RNG and committed sample
progress. Portable exports contain tensor weights and JSON metadata without
executable model objects. Legacy conversion is explicitly trusted and offline.
Official comparisons share exact statistics and sample order; overlapping-window
statistical weighting must not be confused with unique-transition weighting.
`_sample_order.py` supports checkpointed random-with-replacement streams and
explicit original-DataLoader shuffle descriptors. The latter bind source pass
seeds and full-permutation hashes, preserve epoch-tail dropping, and cache at
most one permutation; they must not be replaced by independent seeded shuffles.
Old descriptors and `.npy` matrices remain readable. Both trainers checkpoint
optimizer/RNG/best-model progress and
bind the preprocessing execution contract. Three real-layout CUDA preprocessing
warmups precede updates and RNG restoration, preserving exact resumed execution.
Sparse `log_interval` controls observation volume without changing updates or
selection. Keep virtualenv executable paths intact: resolving their symlinks
can silently select the base interpreter instead.

The optional training `numerical_profile="reference"` selects math attention
and ordered FP64 accumulation of FP32 per-parameter gradient norms. Defaults
retain their existing Torch behavior. Checkpoints bind this policy; portable
exports persist the model's attention backend so inference uses the same choice.
Changing policy requires a new fine-tuning run. The original-method reference
adapter accepts only its default policy and rejects unsupported math exports.
An explicit common policy enables controlled comparisons; it does not promise
bitwise equivalence across CPU and GPU execution.

## Differentiable lifecycle

`DifferentiableDynamicsEnv` and `sim.diff.FunctionalDynamics` advance functional
trajectories through public SimulationManager APIs. Existing `DifferentiableEnv`
retains its [kinematics contract](../differentiable-env/differentiable-env.md).

Backward tensors are independent of native ping-pong buffers. Ordinary and
functional NeRD share a transition; publication adopts detached state/history
without stepping twice. World masks clear selected histories. Functional reset
preserves terminal reward graphs and unselected trajectories. `detach_state()`
is an explicit rollout boundary; optimizer updates follow backward.

Torch and Warp FK use an explicit VJP bridge. Shared storage alone does not
connect autograd. Contact selections, angle wrapping and quaternion signs have
branch-local derivatives. Unsupported native runtime controls fail in functional
execution. First-order gradients do not imply force reports, second-order
derivatives or CUDA Graph support.

## Validation

Tests are under `tests/compute/dynamics`, `tests/learning/models`,
`tests/learning/dynamics`, `tests/data_pipeline/dynamics`, `tests/sim/physics/solvers`,
`tests/sim/diff` and `tests/gym/envs/test_differentiable_dynamics_env.py`.
Official task modules/configs use `classic_control/cartpole` and
`classic_control/ant` in the task extension.

Distinguish feature/forward parity, original-trainer update parity, matched-data
multi-seed accuracy and deployment accuracy. Small synthetic datasets or finite
gradients do not prove the broader claims. Reference identities, budgets and
thresholds are frozen in comparison manifests; missing runs remain pending.
The acceptance contract is `design/neural-dynamics-solvers.md`; user commands
are documented in `docs/source/guides/neural_dynamics.md`.

Equal seeds do not prove identical GPU teacher trajectories. Freeze shared
evaluation corpora and verify their ordered content hashes for paired runs.
Verify held-out seeds against the actual stored trajectory seeds: collection
uses the generation seed plus candidate index, so comparing top-level seeds
alone does not exclude training/evaluation overlap. Check actual canonical
initial states when establishing independent evaluation corpora.
Recorded live PD feedback traces can be reused only with matching policy
identity and action-by-action checks against teacher states. Neural feedback
still uses each neural trajectory's own state; training or random-effort files
cannot stand in for a verified closed-loop teacher.

Historical rollout runs in a fresh process with the original pinned Warp and
source checkout. It restores either trainer's portable weights into the same
original runtime and checks recorded physical arrays including gravity. This is
training-port evidence, not deployment of historical physics through current
Newton. Shared metrics remain simulation-independent so old Warp never imports
DexSim/Newton. The controlled matrix's shared sampling/statistics/validation
protocol is distinct from literal paper-recipe reproduction; its report labels
must retain that distinction.

Fixed historical teacher data can be generated with the standalone
`scripts/experiments/neural_dynamics/official_evaluation_data.py` in Python3.8 /
Torch2.2 without importing the EmbodiChain package or SDK; its pure coordinate
transport modules are explicitly hashed. Historical `teacher_dataset` replay
validates generation provenance, source/model physics, raw/canonical state
agreement and feedback actions. It retains all requested outcomes while the
neural state evolves independently. Data-generation and model-runtime versions
are recorded separately.
