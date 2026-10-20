# NeRD draft PR status and evidence index — 2026-10-20

**Development implementation and bounded experiments are available for review.
Full accuracy qualification and release integration remain pending.** This is the
current status index; the dated reports below retain their original experimental
scope and may describe limitations subsequently addressed by later reports.

The authoritative acceptance contract is [the complete design](../neural-dynamics-solvers.md).
There is no claim of full-budget training, production dynamics accuracy, or
literal reproduction of published paper results.

## Implemented scope

| Work package | Current progress | Evidence boundary |
|---|---|---|
| Named Newton solver extensions | DexSim discovery, capability/lifecycle and public functional/metadata APIs implemented in the adjacent SDK workstream | Tested local SDK artifact; SDK source/release is a separate prerequisite |
| NeRD integration | Installed `dexsim.newton_solvers` entry point named `nerd`, native and functional stepping, state/history/reset, supported controls and physical-profile checks | Homogeneous Cartpole/Ant scenes, CPU/CUDA, first-order gradients |
| Shared mathematics and network | Simulation-independent codec, contact inputs, upstream causal Transformer, normalizers and portable checkpoint conversion | Official checkpoint/fixture and update parity; no full-accuracy claim |
| Data lifecycle | EmbodiChain-owned analytical teacher generation, HDF5 committed shards, trajectory splits and bounded lazy sequence windows | Genuine bounded historical/current datasets; full-size distributions pending |
| Training lifecycle | Supervised train/fine-tune/resume, train-only statistics, offline evaluation and export, CLI and task workflow YAMLs | Exact-resume and original-trainer checks, 40 bounded training runs |
| Differentiable control | Public manager/Scene adapter and separate `DifferentiableDynamicsEnv`, detached publication, partial reset and explicit BPTT boundary | State/control/history/model gradients and scoped CPU/CUDA integration |
| Comparison tooling | Frozen two-teacher/two-trainer plans, shared initialization/statistics/order, evidence validation, paired bootstrap and historical runtime | Formal reports retain `pending` when required evidence is absent |
| Documentation and project context | Full design, guide/API entries, task/configuration examples and routed neural-dynamics context | API coverage and focused documentation checks; broader build caveats below |

The only deployment mechanism is installed entry-point metadata selected by
`solver_type: nerd`. There is no `custom + provider` configuration, direct
provider-path deployment or public imperative registration API. One solver owns
the scene. CUDA Graph, second-order derivatives, physical-parameter gradients,
heterogeneous scene coupling and DDP are outside v1.

## Reference and execution identity

| Component | Identity |
|---|---|
| Official generation/training source | `NVlabs/neural-robot-dynamics` main `370e89d2b8ce68aa3578aa7812859c681b7502bc` |
| Official Newton-preview/checkpoint reference | `476dc67fccc6a99be293bf611f897b1c06f13e30` |
| Current runtime | Python 3.11.14, Torch 2.7.1+cu128, Newton 1.6.0, Warp 1.17.0 |
| Historical runtime | Cartpole Warp 1.8.0; Ant Warp 1.5.1; isolated fresh processes |
| Workstation | NVIDIA GeForce RTX 5090, driver 595.58.03, CUDA runtime 12.8 |
| EmbodiChain starting revision | `66114674668b258c15feae43c8939a0ef1ec4ac4` |
| Adjacent DexSim starting revision | `4f093c4a88ffd869570faf800341a1ad496b7fec` |

The historical and current teacher domains are physically different. Measured
examples include Cartpole masses 50/2.5 versus 1/10 and Ant torso mass 31.65345
versus approximately 0.48388. Historical/current Ant friction also differs.
Asset byte identity therefore cannot qualify a physical model. Compare trainers
within each measured domain; historical weights cannot acquire current-Newton
deployment compatibility by editing manifest annotations.

## Executed experiments

### Official implementation correctness

Official pretrained checkpoints were compared on CPU FP32 at history lengths
1, 9 and 10. Maximum forward, input-gradient, parameter-gradient and
three-Adam-update parameter differences were **zero** for both robots.

| Robot | Maximum feature error | Maximum target error | Maximum decoded-state error |
|---|---:|---:|---:|
| Cartpole | 1.19e-7 | 1.19e-7 | 1.19e-7 |
| Ant | 1.43e-6 | 2.52e-7 | 2.98e-7 |

Receipts: [Cartpole](nerd-cartpole-parity-20261020.json),
[Ant](nerd-ant-parity-20261020.json). These are fixture-level correctness results
under `atol=1e-5`, `rtol=1e-4`, rather than general prediction accuracy.

### Controlled five-seed training

Two robots × two teacher datasets × two trainers × five seeds = **40 completed
runs**, each with **100 optimizer updates**, batch 512 and history 10. Official
network widths are retained: embedding 192/384, six layers, twelve heads, FP32.
Paired runs share exact initialization, statistics, indices and selection rules.

Current data: 64 trajectories × 32 transitions, splits 57/1/6; default initial
states and Ant effort scale 0.05. Fresh historical data: 48 × 100 transitions,
splits 32/8/8, with measured historical physical-v2 gravity provenance.

Every short-run offline metric has a one-sided 95% paired-bootstrap upper bound
below the predeclared 5% margin. The largest is **2.1676%**, current-Newton
Cartpole velocity error. This conditional comparison concerns agreement between
training implementations at 100 updates; it does not qualify the learned model.

The [five-seed report](neural-dynamics-bounded-five-seed-20261020.md) contains
all twelve metrics. The [machine-readable summary](neural-dynamics-five-seed-20261020.json)
retains experiment identities. Observed sequential matrix execution was about
98 s for Cartpole and 139 s for Ant, excluding preparation; do not extrapolate
this small-data timing to the full corpus.

### Historical/current rollout comparison

Seed 0 was evaluated in 24 runs: two robots × two domains × two trainers ×
passive/random-effort/fixed-policy modes. Each run used three trajectories,
horizons 1/10/12 and batches 2 + 1. All twelve pairs have matching teacher-content
and policy identities, with source/model/manifest/HDF5/JSONL hashes checked.

The historical harness uses the original environment, Featherstone teacher and
Transformer integrator. All twelve original-versus-portable restoration probes
had maximum absolute error zero through eleven steps, including history rollover.
Historical runs had no nonfinite states or failures within this bounded horizon.

Material model errors remain:

| Scenario | Teacher | Official-trained model | EmbodiChain-trained model |
|---|---:|---:|---:|
| Historical Ant fixed-PD return | 9.7618 | 1.4588 | 1.5726 |
| Current-Newton Ant fixed-PD return | 4.3053 | 2.2850 | 2.3616 |
| Current Cartpole passive failure rate | 1/3 | 2/3 | 2/3 |
| Current Cartpole random-effort failure rate | 2/3 | 2/3 | 2/3 |
| Current Cartpole fixed-PD failure rate | 0/3 | 1/3 | 1/3 |

The fixed PD is a diagnostic baseline, not the paper policy. Cartpole returns
stop at failure; a less negative return cannot hide an earlier failure.
See [historical measured errors](neural-dynamics-historical-rollout-20261020.md)
and [bounded pairing evidence](neural-dynamics-bounded-pairing-20261020.json).

### Input and training-compute measurements

Bounded grouped HDF5 reads preserve request order, repeated-index ownership,
prefixes, split boundaries and exact training/resume behavior. Reads span at most
266 frames in the measured configuration and keep at most four file handles;
there is no whole-dataset cache or speculative prefetch.

| CPU read/collate/encode workload, B512/H10 | Individual reads | Grouped reads | Speedup |
|---|---:|---:|---:|
| Current Cartpole | 257.87 ms | 12.75 ms | 20.23× |
| Current Ant | 285.03 ms | 25.88 ms | 11.01× |
| Historical Cartpole | 241.61 ms | 10.62 ms | 22.75× |
| Historical Ant | 271.97 ms | 20.77 ms | 13.10× |
| 128-shard, 512-distinct-trajectory stress | 337.13 ms | 181.72 ms | 1.86× |

Warm-page, single-CPU-thread measurements; the stress corpus tests poor I/O
reuse and is not physics data qualification. Full scope and reproduction commands
are in [the input-pipeline report](neural-dynamics-input-20261020.md).

Device-resident official-width FP32 training, TF32 disabled, measured medians:
Cartpole 7.539/7.495 ms and Ant 14.488/14.263 ms per update for official/migrated
implementations. The full 40-run, 4,995,000-update matrix has an optimistic
**607.52 GPU-hour / 25.31-day compute-only lower bound** on this RTX 5090.
Data loading, validation, generation, checkpoints and rollout evaluation are extra.
Full-corpus end-to-end throughput is unmeasured.

## Correctness audit and protocol limits

- Physical signature `nerd-reference-physics-v2` binds fixed-root rotation,
  topology/flags, contact materials and effective collision relationships.
  Datasets/exports were regenerated; older artifacts were not relabeled.
- Training metadata may add unrelated annotations but cannot change/drop
  measured physics fields or invent missing reserved deployment provenance.
- Deployment fails early for unsupported controls, incompatible scenes/physics
  and non-FP32 NeRD weights/buffers. Empty functional reset masks are true no-ops.
- Official generation has an uninitialized gravity-feature defect. Raw output is
  preserved; opt-in transport repair and Y-up/twist-to-Z-up/COM conversion are
  recorded. Actual original gravity is `[0, -9.81, 0]`.
- Acceptance checks measured residuals and immutable artifacts, actual ordered
  inputs and FP32 weights, common teacher content/policy, completed update budget
  and selected test split. Missing/undefined evidence remains pending or fails.
- The controlled comparison changes outer-loop sampling/statistics/validation
  relative to the literal original recipe. Shared replacement indices,
  population moments and one ordered validation split are disclosed in
  [the official protocol audit](neural-dynamics-official-protocol-audit-20261020.md).
  Even a full-budget controlled pass would not by itself prove paper reproduction.

The translated official budget is 4,995,000 updates because epoch zero evaluates
without training. LR starts at 0.0009991 and ends at 0.0001009, with interval 5000
and offset 1. The source and saved pretrained configs also name different dataset
sizes/validation collections; an exact paper recipe must be identified explicitly.

## Local installation and dependency handoff

The adjacent SDK implements the required generic extension APIs. The tested
unpublished local SDK version is
`0.5.1+g4f093c4a88ff.d2bf6eb9a5b93fd36`; wheel SHA-256:
`2b0a37872ddb690392450b4b9764f128d27b8e53138d7973e87459e722504b3f`.
The paired EmbodiChain wheel SHA-256 is
`722046aab36844669862b9feff3165800576316872858d1c0719f1b094926c38`.

Only the paired EC wheel's dependency metadata and RECORD were overlaid to
require that exact SDK. Repository `pyproject.toml` still requires
`dexsim_engine==0.5.1rc1`, which does not establish compatibility with the new
APIs. SDK source/release integration and the final source dependency pin remain
merge prerequisites. This PR does not publish SDK source or wheels.

Independently installed SDK regressions passed 157 tests (21 deselected).
Installed EC historical CPU probes passed for both robots without loading or
attempting to load DexSim/Newton. Tiny installed historical CUDA probes also
passed at B1/count1/horizons1/3: Cartpole max difference zero, Ant 1.4782e-5
within `atol=1e-5`, `rtol=1e-4`. Old Warp emits a cuDeviceGetUuid initialization
diagnostic; successful tiny execution is not general old-Warp GPU qualification.
Artifacts and validation receipts are indexed in
[the latest local artifact manifest](neural-dynamics-local-artifacts-historical-20261020.json).

## Validation scope and remaining work

Earlier focused groups cover CLI/manager/configuration, task packaging,
CPU/CUDA gradients and stream behavior, partial resets, exact training resume,
reference transport and report integrity. Counts in different dated reports
overlap and must not be added into a single total.

The submission-time combined regression selected 479 cases with the actual
official reference, isolated historical runtimes and CUDA tests enabled:
**476 passed, 2 skipped, 1 failed** in 191.49 s. The failure is
`test_robot_mounted_camera_backend_lifecycle[newton]`: `Robot.link_names` indexes
an empty entity list while the mounted camera is declared before articulation
binding. The same test and failure were reproduced from an unchanged archive of
the starting EmbodiChain commit against the same local SDK (8.16 s). This is
baseline evidence under this SDK environment, not a passing test or a claim about
all published SDK combinations. It remains recorded rather than being hidden or
fixed as unrelated scope.

Initial combined collection also exposed three new test-file names colliding
with modules elsewhere under pytest's default import mode. The new files were
renamed to `test_dynamics_task_reference.py`, `test_dynamics_metrics.py` and
`test_dynamics_runtime_identity.py`; production behavior is unchanged. Default
collection with the previously conflicting existing modules now succeeds (106
cases). The full behavioral run used `--import-mode=importlib`. Submission
commands, final counts, skips and log hashes are retained in
[the validation receipt](neural-dynamics-pr-validation-20261020.json).
The three renamed modules subsequently passed all 17 cases with GPU tests
enabled. The two checkpoint cases skipped by the combined command were also
rerun with the explicit pinned Newton-preview checkout: both passed (7.25 s).
These follow-ups overlap the combined selection and do not increase its size.

Public API coverage is 2748/2748. Black 26.3.1, changed-source structural checks
and context validation pass. New guide/API pages built with warnings treated as
errors. A broader Sphinx dummy build succeeded from a clean source snapshot with
785 existing warnings; that snapshot predates the final historical harness phase.
Do not claim a warning-free full current documentation build.

The two vendored official assets retain their exact upstream bytes and six
upstream trailing-whitespace lines. SHA-256 identities match the pinned main
checkout (`ant.xml`: `23e9a720465a71fde3fd9fd2d5a18f8c11ec642f3dca8921c66cb4e169421fa7`;
`cartpole_single.urdf`: `f31276531c03050a3ac3a3e24443899f38a4424586db17140978383433dec237`).
Staged `git diff --check` passes with only these byte-preserved asset files
excluded; the unrestricted check reports their inherited whitespace.

Affected simulation, data, environment and learning context has been revised,
and the new neural-dynamics owner added to MAP. Broader routing matches do not
require unrelated edits where their contracts remain accurate.

- [ ] Release/integrate the extension-enabled SDK and update the source dependency.
- [ ] Run full-size data distributions and 40 × 4,995,000-update training cells.
- [ ] Run five-seed, 2048-trajectory evaluation at horizons 1/10/100/1000 for
  passive, random-effort and fixed-policy scenarios.
- [ ] Pass final accuracy, stability/nonfinite/failure and policy-return gates.
- [ ] Measure representative full-corpus end-to-end throughput.
- [ ] Establish the exact data/statistics/sampling/validation recipe before
  asserting literal published-result reproduction.

No full-budget training job or long evaluation is running. The original
qualification goal is blocked pending the experiment scope/compute-budget
decision. Creating this Draft makes current code, design and evidence reviewable;
it does not close the qualification goal.

## Complete checked-in evidence index

| Record | Scope |
|---|---|
| [Complete design](../neural-dynamics-solvers.md) | Architecture, state/data/training lifecycle, comparisons and formal gates |
| [Initial evidence](neural-dynamics-20261020.md) | Reference fixtures, first real-data experiments, compute budget and initial installation |
| [v2 implementation audit](neural-dynamics-implementation-v2-20261020.md) | Physics-signature fixes, regenerated current-domain flow and integration checks |
| [v2 smoke receipt](neural-dynamics-v2-smoke-20261020.json) | Measured current-domain end-to-end artifacts |
| [Input-pipeline experiment](neural-dynamics-input-20261020.md) | Bounded reader correctness, throughput and stress limits |
| [Official protocol audit](neural-dynamics-official-protocol-audit-20261020.md) | Architecture, optimizer schedule and reproduction differences |
| [Five-seed comparison](neural-dynamics-bounded-five-seed-20261020.md) | Latest 40-cell bounded matrix and both-domain rollout limitations |
| [Five-seed metrics](neural-dynamics-five-seed-20261020.json) | Machine-readable aggregate results |
| [Historical rollout](neural-dynamics-historical-rollout-20261020.md) | Original-runtime protocol, measured state errors and returns |
| [Pairing receipt](neural-dynamics-bounded-pairing-20261020.json) | Both-domain evidence/teacher/policy integrity |
| [Cartpole parity](nerd-cartpole-parity-20261020.json), [Ant parity](nerd-ant-parity-20261020.json) | Official checkpoint and numerical residual identities |
| [Initial artifacts](neural-dynamics-local-artifacts-20261020.json), [v2 artifacts](neural-dynamics-local-artifacts-v2-20261020.json) | Historical artifact snapshots; superseded for latest installation |
| [Latest artifacts](neural-dynamics-local-artifacts-historical-20261020.json) | Current paired wheel identities and installed CPU/CUDA probes |
| [Draft submission validation](neural-dynamics-pr-validation-20261020.json) | Combined regression, baseline failure reproduction and test-discovery fix |

The committed reports/aggregate receipts are reviewable in GitHub. Large raw
HDF5 datasets, weights, traces, plots, source snapshots and complete process logs
remain local at the paths in those reports; hashes identify them, but a local
`/tmp` path is not a remotely downloadable artifact. No binary or durable public
experiment archive is implied by this PR.
