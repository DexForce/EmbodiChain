# Neural dynamics models and training

## Entry points and ownership

| Request | Owner (relative to `embodichain/`) |
|---|---|
| Tensor state, contact and history codecs | `compute/dynamics/codec.py` |
| Model and portable checkpoints | `learning/models/nerd/` |
| Analytical teacher generation | `data_pipeline/dynamics/generation.py`; private `_newton.py` owns its Newton model |
| Reference assets and physical profiles | `data_pipeline/dynamics/_reference_assets.py`; packaged files in `data/nerd/` |
| Committed shards and sequence windows | `data_pipeline/dynamics/shards.py` |
| Training, resume, evaluation and export | `learning/dynamics/trainer.py`, `workflow.py` |
| Shared SL objective and encoded batches | `learning/dynamics/supervised.py` |
| Common offline trainer recipe | `learning/dynamics/sl_recipe.py`; `embodichain train-sl` |
| Original-repository comparisons | `learning/dynamics/reference.py`, `comparison.py`, `report.py` |
| Command dispatch | `cli/dynamics.py`: `embodichain dynamics` |

Model, codec, training and offline evaluation do not import simulation. Generation
uses Newton's built-in Featherstone/MuJoCo solver directly without DexSim, its
custom provider registry or Scene APIs. Native learned-solver deployment and
functional Gym integration are separate consumers, not dependencies of training.

## Data and physical identity

A trajectory contains T+1 states and T controls/contact inputs. Splits belong to
whole trajectories and windows cannot cross episodes or splits. Writers publish
only committed shards; resume advances candidate seeds past accepted and rejected
candidates. File handles and bounded caches belong to their reader/worker.

The direct teacher stores its importer profile, asset/physical-array hashes,
solver settings and actual Newton/Warp versions. Its
`nerd-direct-newton-builder-v1` signature is distinct from Scene-built datasets.
Existing published Scene-generated data remains readable without simulation.
`warp_reference` reconstructs measured legacy asset parameters and the coupled
Ant Featherstone friction mapping; current Newton still owns integration.
Matching a profile name alone does not establish matching physics across
model-building paths, versions or solvers.

Newton COM velocities and NeRD twists differ in ordering and reference point.
The codec owns transport, relative targets use the frame at t, and frozen train
normalizers are reused for evaluation. Ant uses 25 ordered plane-contact slots;
Cartpole is contact-free. Controls are generalized efforts with zero free-base
force. Physical records cannot be relabeled through training manifest overrides.

The collector keeps one bounded batch of history on the model device, then writes
owned CPU arrays. CUDA teachers use graph capture by default where supported;
CPU MuJoCo execution does not capture. Snapshots retain independent ownership.
Mass-matrix update intervals divide the macro substep count so resets can replay.

## Training and validation

Training checkpoints preserve optimizer, scheduler, RNG and committed sample
progress. Portable exports contain tensors and JSON, including normalizers and
attention policy. Resume restores the same run; fine-tuning starts a new one.
Only explicitly trusted legacy conversion may deserialize Python model objects.

`DynamicsTrainer` delegates optimizer updates to `learning.sl.SupervisedLearner`
while retaining its historical sampling, numerical profiles, LR schedule and
training-checkpoint format. The standalone `sl_recipe.build_supervised_trainer`
instead composes the common trainer and uses SL checkpoints. Both reuse the
same codec and normalized physical-target objective. Models and datasets do not
depend on either trainer. The paired acceptance runner under
`scripts/experiments/supervised/nerd_acceptance.py` compares independent reference
and candidate processes; fixed-budget parity is separate from full convergence.

Optional shared sample-order/statistics artifacts support controlled original
trainer comparisons; equal seeds alone do not prove identical batches or physics.
Original Warp references run in isolated supported environments. Keep virtualenv
executable paths intact instead of resolving their interpreter symlinks.

Tests cover codecs, model conversion, bounded shards, reset/replay and partial
collection batches, training/resume and the actual offline CLI. Public usage is
in [Neural Dynamics Models](../../../docs/source/features/neural_dynamics.md).
Fixed-set transition error, live rollout error and paper protocol accuracy are
separate claims; this workflow does not establish their equivalence.
