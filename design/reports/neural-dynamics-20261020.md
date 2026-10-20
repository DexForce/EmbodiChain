# NeRD implementation and training evidence — 2026-10-20

**Status: implementation/short-run evidence exists; full official-training
qualification remains pending.** No 100-update run below replaces the declared
4,995,000-update budget, five paired seeds, or full rollout evaluation.

## Reference and execution identity

- Official source: `NVlabs/neural-robot-dynamics@370e89d2b8ce68aa3578aa7812859c681b7502bc`,
  untouched checkout `/tmp/embodichain-nerd-main-reference`.
- Official data: original `NeuralEnvironment(default_env_mode="ground-truth")`,
  `TrajectorySampler` / `TrajectorySamplerAnt`, sampling ranges and Featherstone
  integration. Cartpole uses isolated Warp **1.8.0**, five substeps; Ant uses the
  paper baseline's Warp **1.5.1**, sixteen substeps. Both macro timesteps are 1/60 s.
- The original `VanillaTrainer.preprocess_data_batch`, `compute_loss`, `one_epoch`
  and `get_scheduled_learning_rate`, original `NeuralIntegrator` math and original
  `ModelMixedInput` execute directly. No official source methods are replaced.
  The wrapper binds common initialization, lazy data, frozen statistics, logging
  and validation evaluation; it bypasses the constructor's eager data allocation.
- Current teacher: DexSim 0.5.1 development build, Newton 1.6.0, Warp 1.17.0,
  MuJoCo CPU with separate worlds, ten substeps. Installed artifact identity is
  recorded in each dataset manifest; this is not a claim about the rc1 wheel.
- Training/benchmark hardware: NVIDIA GeForce RTX 5090; driver **595.58.03**,
  PyTorch **2.7.1+cu128**, CUDA runtime **12.8**, Python **3.11.14**.

## Official checkpoint and numerical parity

The separate Newton-preview reference commit is
`476dc67fccc6a99be293bf611f897b1c06f13e30`. Both official pretrained checkpoints
were checked on CPU in FP32 with history lengths 1, 9 and 10. Maximum forward,
input-gradient, parameter-gradient and three-Adam-update parameter errors were
**zero** for both robots. Maximum codec errors:

| Robot | Features | Targets | Decoded state |
|---|---:|---:|---:|
| Cartpole | 1.19e-7 | 1.19e-7 | 1.19e-7 |
| Ant | 1.43e-6 | 2.52e-7 | 2.98e-7 |

These passed the declared `atol=1e-5`, `rtol=1e-4` tolerances. They verify the
checkpoint/network/codec port for these fixtures, not general training quality.
Checked-in receipts retain checkpoint and fixture hashes:
[Cartpole](nerd-cartpole-parity-20261020.json),
[Ant](nerd-ant-parity-20261020.json).
The full fixtures and portable exports remain under
`/tmp/embodichain-nerd-parity-20261020/{cartpole,ant}`.
Reproduce each with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tests/learning/models/_nerd_reference_check.py /path/to/pinned-newton-preview Cartpole /path/to/new-output
```

Use `Ant` for the other checkpoint. This runner executes explicitly selected
trusted reference code; it does not download or import an arbitrary repository.

## Genuine teacher data and explicit compatibility transport

Both official robots produced **48 real trajectories × 100 transitions**:
32 train, 8 validation and 8 test. Every raw original HDF5 artifact is retained.
Canonical conversion preserves the declared split and checks within-trajectory
continuity. Dataset paths:

- `/tmp/nerd-official-real-cartpole-20261020/canonical`
- `/tmp/nerd-official-real-ant-20261020/canonical`

Each parent directory contains `train.raw.h5`, `validation.raw.h5`, `test.raw.h5`,
`source-physics.json` and `generation-result.json`, including source/file hashes.
The production wrapper now streams one sampler batch at a time; a separately
executed regression verifies multiple batches and a partial final batch.

**Confirmed official-source bug:** `TrajectorySampler` allocates `gravity_dir`
with `torch.empty` and initializes only the up-axis component. The actual CPU
probe produced NaNs and values around 849,000 in the other components. The
collection explicitly enables `repair_uninitialized_gravity: true`; raw HDF5
values remain unchanged. Only the canonical gravity field is replaced by the
known physical direction, with changed/nonfinite counts recorded per split.
These datasets are therefore labeled **source-bug-corrected official teacher
inputs**, not byte-for-byte reproductions of an unspecified published dataset.

Both original environments are Y-up. Explicit world-frame transport rotates Ant
by +90° about X; Cartpole additionally rotates +90° about Z to match the current
reference asset placement. Base position/orientation, COM linear/angular velocity,
world contact points/normals and gravity rotate together. Body-local contact
points remain local. Angular-first world-origin twist is converted to native
linear-first COM velocity. Source-coordinate metadata and rotation are retained.

### Identical asset bytes do not imply identical physical systems

Actual initialized model arrays, not XML declarations alone, give these values:

| Property | Official Warp teacher | Current Newton asset import |
|---|---:|---:|
| Cartpole cart/pole mass | 50 / 2.5 | 1 / 10 |
| Ant torso mass | 31.6534519 | 0.483880247 |
| Ant upper/lower leg mass | 7.83155060 / 13.5184412 | 0.0391577530 / 0.0675922039 |
| Ant robot-shape friction | 0.75 | 1.5 |
| Ant actuated armature | 0.01 | 0.01 |
| Official ground friction | 0.5 | Not independently measured in this comparison |

The old importers' inertia/density handling differs. Both versions use the same
Ant asset SHA256 `23e9a720465a71fde3fd9fd2d5a18f8c11ec642f3dca8921c66cb4e169421fa7`
and Cartpole SHA256
`f31276531c03050a3ac3a3e24443899f38a4424586db17140978383433dec237`.
Original Ant contact thickness is radius (0.25 / 0.08) plus a ground thickness
around 1e-5; current Newton contact feature construction includes its configured
margins. Full masses, inertia, COM, joint parameters and materials remain in
`source-physics.json`.

**Deployment implication:** an official checkpoint cannot be declared physically
compatible using only asset SHA or joint count. Preserve its old physical profile
or retrain against the current teacher. Do not replace an official-data manifest
with current-Newton physics identity merely to bypass deployment validation.

## Executed real-data matched training matrix

All **8 cells completed**: 2 robots × 2 teacher datasets × 2 trainers, **one seed
(0), 100 Adam updates per cell**, batch 512, history 10, six layers, twelve heads,
embedding 192 for Cartpole / 384 for Ant. Both trainers within a row share exact
initial weights, train-only statistics, explicit batch-index order, split and
validation selection. These runs use a deliberately short learning-rate schedule;
they are not the first 100 steps of the full epoch schedule.

Official-data rows use the genuine 48×100 datasets above. Current-Newton rows use
64×32-transition smoke datasets, with observed split counts **57/1/6**. Their
initial states are defaults; Cartpole uses full force scale, Ant uses **0.05** of
the full force scale. The small validation/test sets limit any accuracy inference.

| Robot | Teacher | Trainer | Test q error norm | Test qd error norm | Worker wall s |
|---|---|---|---:|---:|---:|
| Cartpole | Official Warp | Official | 0.031869880 | 0.307753670 | 29.403 |
| Cartpole | Official Warp | EmbodiChain | 0.031835163 | 0.307730865 | 29.094 |
| Cartpole | Current Newton | Official | 0.028611711 | 0.708493175 | 30.270 |
| Cartpole | Current Newton | EmbodiChain | 0.028664216 | 0.700447635 | 29.686 |
| Ant | Official Warp | Official | 0.077358087 | 7.257187259 | 33.837 |
| Ant | Official Warp | EmbodiChain | 0.077455009 | 7.250118046 | 32.056 |
| Ant | Current Newton | Official | 0.090978993 | 7.981286798 | 34.236 |
| Ant | Current Newton | EmbodiChain | 0.091046900 | 7.983307150 | 33.434 |

Ant base position error (m) / orientation error (rad): official-Warp row,
0.005254870 / 0.018305524 (official) versus 0.005307779 / 0.018332163 (migrated);
current-Newton row, 0.004343362 / 0.015835387 versus 0.004330985 / 0.015851749.

Errors are within-row matched comparisons. Cartpole q and generalized qd norms
combine coordinates with different units and must not be described as pure
Cartesian distances. A one-seed result has no five-seed confidence interval and
cannot establish the predeclared 5% noninferiority gate. Rounded values above
are presentation only; machine reports retain full precision.

Evidence directories (each includes immutable experiment/run specs, common
weights/statistics/order hashes, subprocess commands, stdout/stderr, result
receipts, exports and generated plots):

- `/tmp/nerd-real-matched-cartpole-20261020/report/REPORT.md`
- `/tmp/nerd-real-matched-ant-20261020/report/REPORT.md`
- Compact precise metrics: `/tmp/nerd-real-matched-summary-20261020.json`

The report states `training_matrix_status: completed` and overall qualification
`pending`. Earlier synthetic-fixture tests exercised infrastructure only and are
not included in the table above.

## Measured training-budget feasibility

Device-resident FP32, TF32 disabled, batch 512, history 10; 20 warmup updates and
100 synchronized timed updates per implementation/robot. The operation includes
forward, normalized MSE, backward, norm clipping and Adam. Initialization and
first-update latency are reported separately. Final parameters match exactly
between models for this seeded compute-only workload.

| Robot | Implementation | Median ms/update | One 4,995,000-update run, hours |
|---|---|---:|---:|
| Cartpole | Official | 7.539 | 10.460 |
| Cartpole | EmbodiChain | 7.495 | 10.399 |
| Ant | Official | 14.488 | 20.103 |
| Ant | EmbodiChain | 14.263 | 19.790 |

The declared 40-run matrix (2 robots × 2 datasets × 2 trainers × 5 seeds) has an
**optimistic compute-only lower bound of 607.52 GPU-hours / 25.31 days on one
RTX 5090**. Peak allocated GPU memory was at most 1.08 GB for the harness.
This excludes data loading, encoding, validation, checkpoints, teacher generation
and rollout evaluation. No 40-run full-budget job was launched.

The current input pipeline is a material bottleneck: independent CPU timing of
HDF5 read + collate + encode, batch 512/history 10, measured medians **259 ms for
Cartpole and 295 ms for Ant**. cProfile attributes most time to individual HDF5
window/group access. This agrees with the 29–34 s observed worker times for 100
updates, including startup and evaluation. Full-data timing is unmeasured;
batched reads/caching/prefetch need representative validation before a practical
full-budget duration can be promised.

Artifacts: `/tmp/nerd-full-budget-estimate-20261020.json` and
`/tmp/nerd-input-pipeline-profile-20261020.json`.

## Reproduction and remaining gates

Use a separate pinned reference checkout and isolated dependencies; never replace
EmbodiChain's installed Warp:

```bash
GIT_LFS_SKIP_SMUDGE=1 git -C /path/to/neural-robot-dynamics worktree add --detach /tmp/nerd-main-reference 370e89d2b8ce68aa3578aa7812859c681b7502bc
python -m pip install --target /tmp/nerd-warp18 --no-deps warp-lang==1.8.0
python -m pip install --target /tmp/nerd-warp151 --no-deps warp-lang==1.5.1
PYTHONPATH=/tmp/nerd-warp18:/tmp/nerd-main-reference:$PWD python -m embodichain.learning.dynamics.reference --spec generation.json --trust-external-code --generate
python -m embodichain.learning.dynamics.comparison --prepare comparison.json
python -m embodichain.learning.dynamics.comparison --run /path/to/experiment.json --trust-external-code
python -m embodichain.learning.dynamics.report --experiment /path/to/experiment.json
python -m scripts.benchmark.learning.dynamics_training --source /tmp/nerd-main-reference --trust-external-code --output budget.json
```

For Ant generation, use the Warp 1.5.1 PYTHONPATH and `warp_version: "1.5.1"`.
Generation specs used here are `/tmp/nerd-official-generate-{cartpole,ant}.json`;
exact training specs and commands are stored under the matrix directories.
Reference code execution and gravity repair require explicit opt-ins. The source
checkout remains unchanged; its hashes and the adapter identity are recorded.

Still pending: full-size data distributions, five paired seeds at the declared
full budget, numerical qualification reports attached to final runs, all matched
2048-trajectory horizons/scenarios, fixed-policy/stability gates, physical-profile
compatible official-model deployment, and representative end-to-end throughput.
User budget guidance is needed before launching a multiweek full matrix. Until
those gates are backed by executed artifacts, the original qualification objective
remains open.

## Integration and packaging validation

Focused EmbodiChain regression groups completed after integration:

- CLI, simulation manager/configuration and Gym construction: **398 passed,
  3 skipped**.
- Articulation drive/state compatibility, official task layout and package data:
  **23 passed**.
- Context schema/routing: **40 passed**. The changed contracts have a new neural
  dynamics owner and links from simulation, environment and learning topics.
  Broad watches also match Task Programs, Gen Sim and existing asset/RL context;
  their unchanged contracts do not require unrelated rewrites.
- API coverage: **2747/2747** public exports documented; checker regressions:
  **9 passed**.
- Black **26.3.1** applied to the repository; changed Python headers, future
  imports and public-module export declarations passed structural checks.
- Built an EmbodiChain wheel and inspected the NeRD entry point, modules,
  third-party licenses, robot assets and all 16 workflow YAML files.

The full Sphinx dummy build is **not verified**: the existing architecture
extension refuses generation while relevant source files are uncommitted. Its
failure occurred before page rendering. No source was committed merely to bypass
that precondition.

An isolated Sphinx dummy build of the new neural-dynamics guide and API sections
passed with `-W` and no warnings. This focused check excludes the architecture
extension and does not replace the full repository build.

The normal dependency pin still names `dexsim_engine==0.5.1rc1`; it must be
updated with the eventual extension-enabled release. Until then this is a paired
local-development deployment, not an install-ready claim about that published
pin. The tested local DexSim artifact and source/build hashes are supplied by
its adjacent implementation workstream. Native teardown warnings also occurred
in the clean DexSim baseline and are not claimed fixed by this work.

A dependency-consistent local wheel pair is available. The EmbodiChain wheel was
repacked with only its `Requires-Dist` DexSim pin changed to the exact tested local
version and its `RECORD` regenerated; all runtime files match the source-built
wheel. This overlay is not a source dependency release update. Paths, hashes and
the exact metadata change are recorded in
[the local artifact manifest](neural-dynamics-local-artifacts-20261020.json).
Install both listed paired wheels together in an environment with the ordinary
project dependencies. Installing the unmodified editable checkout instead still
leaves its rc1 requirement unsatisfied; `--no-deps` does not make that conflict
disappear.

## Subsequent completion audit (implementation still in progress)

The previous local wheel pair was independently installed and exercised:
**156 tests passed, 21 deselected**, with import paths and dependency matching
recorded in DexSim's `validation-content/manifest.json`. This verifies that
specific snapshot, not later source edits.

A subsequent runtime audit reproduced missing fixed-root-frame and Ant contact/
collision fields in the physical signature. Those omissions can accept a model
for a changed physical system. Their corrections and regenerated simulation
fixtures are required before the current worktree or old portable exports can
be considered qualified. Old fingerprints must not be relabeled to bypass the
new checks. Data-input batching and report provenance checks are also undergoing
focused implementation and validation; rebuild the paired artifact after those
changes settle.

The report reader now accepts the measured numerical receipts directly. Both
historical 8-cell experiments were re-evaluated under `report-audit/`: numerical
parity **passed**, recorded teacher provenance verified, overall qualification
still **pending**. Rollout attachments must prove model/manifest/source/trace
hashes, the test split, teacher domain and identical selected teacher content;
fixed-policy pairs must identify the same policy. Bare `passed: true` summaries
and mismatched rollout populations cannot establish qualification.
