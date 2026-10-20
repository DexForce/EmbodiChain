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

NeRD validates supported homogeneous articulations, named joint mapping, trained
timestep, gravity, physics signature and contact schema. Reference Cartpole is
contact-free: disable the physical default plane at manager creation, not only
its visibility. Ant uses a free base and fixed plane-contact candidates. Controls
are generalized efforts; unsupported base/external forces are rejected.

Newton COM velocities and NeRD spatial twists differ in both component ordering
and reference point. The codec owns both conversions. Relative targets use the
frame at t, not t+1. Training and runtime share the codec and frozen normalizers.

Committed HDF5 shards hold T+1 states and T controls/contact inputs with provenance.
Splits are trajectory-level; windows cannot cross episodes or splits. Workers own
independent file handles and bounded caches. Offline training does not initialize
simulation. Teacher/neural deployments separately own physics configuration.

Training checkpoints retain optimizer, scheduler, RNG and committed sample
progress. Portable exports contain tensor weights and JSON metadata without
executable model objects. Legacy conversion is explicitly trusted and offline.
Official comparisons share exact statistics and sample order; overlapping-window
statistical weighting must not be confused with unique-transition weighting.

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

Historical rollout runs in a fresh process with the original pinned Warp and
source checkout. It restores either trainer's portable weights into the same
original runtime and checks recorded physical arrays including gravity. This is
training-port evidence, not deployment of historical physics through current
Newton. Shared metrics remain simulation-independent so old Warp never imports
DexSim/Newton. The controlled matrix's shared sampling/statistics/validation
protocol is distinct from literal paper-recipe reproduction; its report labels
must retain that distinction.
