# Neural dynamics architecture

Neural dynamics replaces the state-transition solver while retaining scene
construction, controls, collision geometry and simulation lifecycle. The user
workflow, downloadable artifacts and examples are documented in
[Neural Dynamics Models](../docs/source/features/neural_dynamics.md).

## Ownership

| Layer | Owner | Responsibility |
| --- | --- | --- |
| Solver discovery and native lifecycle | DexSim `dexsim.newton_solvers` | Installed provider discovery, construction, stepping, reset and publication |
| Tensor mathematics | `embodichain/compute/dynamics/` | State/velocity transport, history operations and physical-unit errors |
| Model implementation | `embodichain/learning/models/nerd/` | Network, normalizers and portable weights |
| Offline data | `embodichain/data_pipeline/dynamics/` | Committed HDF5 shards, trajectory splits and sequence windows |
| Offline learning | `embodichain/learning/dynamics/` | Training, resume, evaluation, export and optional reference adapters |
| Scene integration | `embodichain/lab/sim/physics/solvers/` | Physical binding, contact adaptation and native/functional transitions |
| Task configuration | `extensions/embodichain_tasks/configs/tasks/classic_control/` | Reference robot scenes and runnable workflow examples |
| Research tools | `scripts/experiments/neural_dynamics/` | Isolated upstream-environment and cross-version comparisons |

Compute does not import simulation or learning. Model loading and offline
training do not initialize DexSim or a simulator. Generic discovery belongs to
DexSim; adding a provider does not require an EmbodiChain dispatch branch.
The current model is `NeRDModel`; generic extension points do not require an
additional abstract neural-model hierarchy.

## Solver deployment

The installed package registers `NeRDProvider` through the
`dexsim.newton_solvers` entry-point group. A deployment selects
`solver_type: nerd` and supplies a portable export directory. Unknown names,
duplicate providers and incompatible interfaces fail instead of falling back.
There is no provider-path import or imperative registration API.

Each World owns its solver, histories and buffers. A solver handles the complete
scene. The initial NeRD implementation supports homogeneous Cartpole or Ant
articulations, generalized joint efforts, and its declared static-plane contact
schema. Capability metadata distinguishes reset, first-order differentiability,
contact reports, external forces and CUDA Graph support. Unsupported quantities
are unavailable rather than fabricated as zero.

A compatible DexSim build must expose the named-solver and functional Scene
contracts; a package version string alone does not establish that capability.

## Physical and numerical contracts

Portable exports bind robot topology, joint order, timestep, gravity, contact
schema, base COM and a measured physics signature. The runtime reconstructs and
validates that physical domain before stepping; users cannot relabel historical
weights to bypass the checks.

`physics_profile=warp_reference` reconstructs the reference assets. Ant with
current Featherstone also uses a coupled conversion from legacy Euclidean
friction to Newton Huber/Coulomb parameters. Both material coefficients and the
smoothing scale form one mapping. The export retains that mapping so inference
recreates the trained physical signature. Other solver profiles retain their
own semantics. Equal numeric parameter trees do not imply equal physics across
versions.

The codec owns conversions between Newton COM-linear-first velocities and
NeRD angular-first spatial twists, including reference-point transport. Relative
state targets use the frame at the current timestep. Training and inference use
the same codec and frozen input/output normalizers. Exported attention policy is
part of the numerical contract; changing it is an explicit model variant.

## Functional execution

Native and functional NeRD calls share one transition implementation. Functional
state consists of owned physical tensors, history tensors and per-world valid
history lengths. Native ping-pong buffers are publication storage, not tensors
retained for backward.

Warp FK has an explicit first-order Torch VJP bridge; shared storage alone does
not connect autograd. Contact selection, angle wrapping and quaternion signs
have branch-local derivatives. Functional publication advances time once and
adopts detached state/history without executing a second transition. Selected
resets preserve unselected trajectories and previous reward graphs.
`detach_state()` defines the rollout differentiation boundary.

## Data, training and exports

Each committed trajectory stores T+1 states and T efforts/contact inputs.
Training, validation and test splits are trajectory-level; sampled windows do
not cross split or episode boundaries. Worker processes own their file handles.
Optional resident caches have explicit byte budgets and preserve tensor values.

Training supports scratch initialization, fine-tuning and exact checkpoint
resume with optimizer, scheduler, RNG, normalization and consumed-sample state.
Reference DataLoader descriptors preserve shuffled pass order and dropped tails.
Offline training checkpoints remain local training artifacts. Published inference
packages contain tensor weights and JSON deployment metadata without optimizer
states, process logs or workstation-specific paths.

Reusable datasets are published under
`DexForceAI/embodichain_datasets/neural_dynamics/nerd/<robot>/v1`; inference
models are published under
`DexForceAI/embodichain_model/neural_dynamics/nerd/<robot>/v1`.
Cards state the actual data scale, supported physics and measured limitations.
The source repository contains the implementation, configuration, tests and
feature documentation; chronological training reports remain outside the PR.

## Validation boundaries

Validate codecs/history in `tests/compute/dynamics`, data persistence/windows in
`tests/data_pipeline/dynamics`, and model/trainer contracts in
`tests/learning/models` and `tests/learning/dynamics`. Solver lifecycle, physical
signatures and contact adaptation are covered in `tests/sim/physics/solvers`;
functional environments are covered in `tests/sim/diff` and `tests/gym/envs`.

Published artifacts must load through the production readers and match their
metadata, checksums and physical domain. Model accuracy is measured against the
matching analytical solver using common initial states and actual action tapes.
Own-state feedback-policy comparisons are labelled closed-loop behavior.

Network parity, trainer parity, solver deployment, task success and reproduction
of a paper's final accuracy are separate claims. Accuracy reports distinguish
endpoint error from temporal mean, retain failed trajectories, and record the
sample population and horizon. Performance comparisons use synchronized warmed
calls, equivalent macro timesteps and the analytical solver's supported CUDA
Graph path; throughput does not establish an accuracy acceptance claim.
