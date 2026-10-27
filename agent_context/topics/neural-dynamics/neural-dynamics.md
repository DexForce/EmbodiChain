# Neural dynamics models and training

## Entry points and ownership

The main wheel bundles `embodichain_models` with the same version and dependencies.
It is an import package under `extensions/`, not a separate distribution or a
runtime registry. Shared MLPs and feature normalization stay in core
`embodichain.learning.models`; generic supervised orchestration stays in
[SL](../supervised-learning/supervised-learning.md).

The [model catalog contract](../../../extensions/embodichain_models/README.md)
owns domain and public-import naming. Policy models predict actions; dynamics
models predict future state or geometry. Canonical RWM and RigidFormer paths are
`dynamics.rwm` and `dynamics.rigidformer`; the catalog tracks implemented families.
RWM ownership and graph/data contracts are in
[robot world model](../robot-world-model/robot-world-model.md).
World-model naming does not introduce a parallel `world_models` domain.

| Request | Owner |
|---|---|
| NeRD network, portable checkpoint, physical codec | `extensions/embodichain_models/embodichain_models/dynamics/nerd/{model,checkpoint,codec}.py` |
| Committed HDF5 shards and bounded windows | Model package `data.py`, private `_resident.py` |
| Encoding, frozen moments and offline errors | Model package `integration.py` |
| Objective and common trainer factory | Model package `supervised.py`, `sl_recipe.py` |
| Prepare/train/export/eval/generate commands | `examples/learning/nerd/__main__.py`; `embodichain run-example learning/nerd` |
| Pinned datasets, exports and reference assets | Example `_resources.py`; explicit `prepare` command |
| Analytical teacher and physical profile | Example `_support/{_generation,_newton,_reference_assets}.py` |
| Historical trainer and checkpoint workflow | Example `_support/{_trainer,_workflow}.py` |
| Original-repository comparison and acceptance | Example `_support/` runners; protocol in `research/` |

The showcase defaults to `sl.train.yaml` and the installable
`embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer` factory.
Direct `embodichain train-sl --config FILE` works independently of example
bootstrap. An explicitly selected `dynamics.train.*.yaml` uses the historical
trainer for controlled comparisons and its legacy checkpoint contract.

The extension imports neither examples nor simulation. Generation uses Newton's
built-in Featherstone/MuJoCo solvers directly without a DexSim custom solver or
Scene API. Native learned-solver deployment and functional Gym integration are
separate consumers. There are no NeRD-specific core `compute/dynamics`,
`data_pipeline/dynamics`, `learning/dynamics` modules or core `dynamics` command.

## RigidFormer author model

The native family lives in
`extensions/embodichain_models/embodichain_models/dynamics/rigidformer/`.
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

Numerical agreement, recovery and full benchmark/retraining qualification
remain separate evidence requirements.

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
training contracts. The [feature guide](../../../docs/source/features/neural_dynamics.md)
owns user commands; the [model API](../../../docs/source/api_reference/model_catalogs.rst)
owns exported definitions. Transition error, accumulated rollout error and
paper-scale accuracy remain separate claims.
