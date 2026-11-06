# Offline Expansion Engine

`embodichain expansion` collects independent expert episodes in one batched World.
It reuses Task Program/Atomic execution with native or Real2Sim prototype motion,
ExpansionSession accounting and the existing LeRobot writer. ODS remains unchanged.

## Draft status

This foundation draft is synchronized with `main`, including Real2Sim #896,
Franka #916 and scene/workspace #912. Release qualification remains pending.
Native Default collection and Task Program Newton runs have positive evidence.
The tested Newton DROID carton run was rejected by physical-effect/segment checks;
it is **not qualified**, and no thresholds or source protections were relaxed.
Complete source/backend matrices and broader atlas/motion recipe qualification
remain follow-up work. Portable calibration/attachment export and independent
postprocessing recovery are implemented and validated. The CLI rejects unsupported combinations rather than claiming full coverage.

## Interactive acceptance showcase

The [full showcase](showcase/README.md) pairs planning-generated UR5 experts with
Real2Sim prototype-driven Franka experts under the same Task Program execution
chain. Run `embodichain run-example expansion/engine/showcase --output outputs/expansion_showcase --open`
for measured 3D/RGB-D comparisons, per-capability checks, recovery diagnostics and
portable dataset verification. `--mode quick` samples a smaller subset; unrun
capabilities remain explicit. `--reuse` rebuilds the offline page without a World.

## Run a configured job

From the repository root, with the simulation dependencies installed:

```bash
embodichain expansion preflight --job examples/expansion/engine/job.default.yaml
embodichain expansion run --job examples/expansion/engine/job.default.yaml
embodichain expansion inspect --output /tmp/embodichain-expansion/default
embodichain expansion resume --job examples/expansion/engine/job.default.yaml
```

The output must be unused for `run`. `resume` checks pinned source/recipe semantics and commit receipts. Batch size,
resource budgets and postprocessing requests may change without resampling a recipe. It retries frozen writes without another
rollout, and, with `retry_interrupted: true`, starts a new attempt of the same immutable
recipe for an interrupted physical episode. Disabling this policy skips interrupted
recipes when admitting new work. A bounded
run that misses its target exits with status 2. Inspect reports physical commits,
per-attempt outcomes and separate observation-variant counts.

`job.newton.yaml` selects a separate Newton environment component. A job cannot
switch the physical backend of a selected environment. Both example deployments
use the same one-cycle Task Program and independent measured lift/place/release
and collision checks. `job.motion.yaml` additionally applies the existing combined
profile's spatial trajectory variants; nominal source planning is still lazy.

The Python entry point is:

```python
from embodichain.lab.expansion import ExpansionEngine, load_expansion_job
engine = ExpansionEngine.from_job(load_expansion_job("job.yaml"))
summary = engine.preflight()
result = engine.run()  # or engine.resume()
```

Custom integrations can supply `ExpansionHost` and `EpisodeSink` to
`ExpansionEngine(ExpansionJobCfg(...), session=..., host=..., sink=...)`.
`ExpertSource`, `ExpansionContext` and `ExpansionOperator` expose read-only source
identity and versioned sampling. `ExpansionEpisode` is the canonical ExpertEpisode
envelope, including immutable metadata references to media and source attachments;
there is no second numeric episode representation.

## Session and host integration boundaries

`ExpansionSession` and `ExpansionSessionCfg` are owned by `embodichain.lab.expansion`.
Old `sim.motion.expansion` Session imports resolve to the identical class. Existing
trajectory configs remain accepted and select their original phase-geometry and
manipulability policy through `TrajectorySessionCoverage`; the new generic session
config defaults to confirmed episode counting. Use an explicit domain coverage
adapter when trajectory deduplication is required.

The native `GymExpansionHost` facade owns one World and composes internal recipe,
cohort preparation, frame-collection and episode-assembly components. Its public
constructor and execution ports remain compatible. Components share an explicit
context; the facade retains reset/close ownership.

Adapters consume public bridge identity (`program_id`, owned `env_ids`) and an
exception-safe `task_program_bridge_scope`. Real2Sim source validation accepts an
explicit measured evidence object instead of a fake environment/buffer interface.
LeRobot export calls the public `persist_episode_payload` method, which shares the
existing writer journal, fragment deduplication and uncertain-commit handling.

## Shared scheduling and recovery

Configured jobs and custom integrations use one `ExpansionEngine` scheduler.
`GymExpansionHost` owns reset, scene/physics changes, settling and measured execution;
`TaskProgramBatchExecutor` drives both native planning and per-row prototype bindings.
`motion_source: prototype` is inferred from deployment-owned `real2sim.profile`.
Legacy `source` and `source_profile` fields remain accepted; an explicit collection
profile must retain the declared deployment's source artifact hashes.

Immutable `ExpansionRecipe` values contain source revision, logical ordinal,
operator versions and sampled payloads. `BatchCompatibility` pins topology, joint
order, Task Program structure, backend, clock, observation schema and shared World
resources. The host verifies these declarations and regenerated values before reset
or parameter mutation. The supplied host has a closed supported operation catalog;
custom operators need an adapter, rather than arbitrary imperative callbacks.

A semantic job ID pins recipe inputs. Execution IDs describe capacity, budgets,
postprocessing requests and runtime versions. A persistent job-instance ID separates
physical episode/commit identities of independent runs of the same recipe. Each
retry has a new attempt number, while its recipe ID stays fixed. Journals retain
invocation history and cumulative observed time; a hard process loss marks elapsed
time as an uncertain lower bound instead of charging offline downtime.

Confirmed parents are eligible for rendering/export even before the collection
reaches its target. Render failure does not prevent parent export. Each stage has
its own status/history; `inspect` derives pending visual children from the current
confirmed parents and pinned profile definitions. Retry only an export using:

```bash
embodichain expansion resume --output /tmp/embodichain-expansion/default --stage export
```

This path opens no World and does not resolve the original deployment or source
files. `stage_complete` reports the selected export, while `complete` still reports
the whole job. Ordinary `resume --job ...` can retry collection and rendering.

Version 2 artifacts store typed manifests with relative, hashed numeric/media/source
references. Export includes a self-contained `embodichain_artifacts` directory with
parent and child evidence, calibration streams and source attachments. Moving the
exported dataset does not require the producing job directory. Legacy artifacts
remain readable in place; export creates a portable projection without rewriting
the original committed evidence. Resuming legacy collection requires unchanged
resolved deployment evidence; interrupted legacy attempts without replayable recipes
are abandoned, with their original records preserved.

## Native RGB-D and observation variants

```bash
embodichain expansion run --job examples/expansion/engine/job.rgbd.yaml
```

This selects a reusable RGB-D embodiment, captures actual frames and calibration,
then produces material/camera variants from complete recorded scene states.
`observation_independent: true` is an explicit assertion about expert decision
semantics: changing rendered images must not invalidate the parent's actions.
It is inappropriate for an expert whose decisions depend on those images.

The World pauses dynamics for the entire rerender cohort. Every child keeps the
parent's actions and T+1 state timestamps; a child does not count as a new physical
success. The original scene and parameters are restored afterward. Old qpos-only
files cannot be used as complete render sources.

`media_buffer_bytes` bounds raw frame storage across the active cohort. Each spool
writes owned NPZ blocks and a final hashed manifest. Encoding/stacking needs an
additional bounded transient copy. Numeric evidence has a separate per-episode
budget and a 128 MiB serialized archive ceiling. No full dense image trajectory is
kept for every candidate in RAM. Sensor schema changes within a job are rejected.

`export_lerobot: true` streams both physical parents and visual children through
LeRobotRecorder, including the existing lossless depth-video sidecar. The published
`lerobot-export.json` identifies a finalized dataset with verified recording journals.
An interrupted export remains private and can be retried from frozen source artifacts.
The reader streams individual frames from stored blocks under a frame-byte budget;
full artifact validation still checks bounded blocks. Writer encoding has its own
buffering. Export revisions track the exact accepted artifact set.
Validate the published dataset with:

```bash
embodichain dataset validate DATASET_PATH --check-media --media-frame-counts
```

## Real2Sim

Prepare the pinned DROID carton example using its unchanged raw recordings and
qualification annotations. The shared geometry fingerprint identifies collision
mesh content and named joint structure independently of assembly comments,
visual sections, inertial values and cache locations. Inertial and dynamic
parameters remain separate measured evidence. Exact legacy byte identities and
independently pinned aliases follow the upstream Real2Sim qualification contract.
Never edit a source pin to bypass qualification.

```bash
embodichain data download --name Real2SimCarton
python examples/expansion/trajectory/real2sim_carton/prepare.py prepare \
  --source-dir ~/.cache/embodichain_data/extract/Real2SimCartonV1/raw \
  --output /tmp/embodichain-expansion/carton
embodichain expansion run --job examples/expansion/engine/job.real2sim.yaml
```

For RGB-D, make a separate prepared deployment selecting the reusable
`franka_panda_robotiq85_rgbd.yaml` embodiment and set `capture_media: true` in its
job. Preserve the original program, integration and source profile. The batch host
supports nominal, joint-residual, task-space-residual and mapped retiming recipes
from the existing source profile. Protected phases, occurrence bindings and no
native-planning fallback remain enforced. Source originals, normalized prototypes,
time maps, actual trace and measured geometry are archived with each committed artifact.

## Scene/workspace and resource compatibility

`scene_atlas` accepts a content-verified SceneRobotFeasibility atlas. Each proposal
keeps its scene layout and robot installation together; prior atlas success never
replaces fresh task/collision validation. Source proposals are selected by logical
ordinal. Geometry/topology changes require another deployment World. Fixed asset
variants from the existing asset-generalization pipeline can therefore be collected
as separate jobs while retaining the upstream deployment and source configuration.

Native cohorts share backend, clock, robot/joint layout, physical topology,
observation schema and global appearance. Material/camera/light profiles apply at
World boundaries. Per-row mass scales also scale inertia; friction is absolute. `center_of_mass`
provides three body-local metre ranges per rigid UID. `joint_stiffness_scale` and
`joint_damping_scale` map explicit robot joint names to scalar ranges. These derive
from the original baseline and verify the native readback.
Every parameter change is read back and compared against the current candidate's
snapshot after execution. Undeclared parameter drift rejects the row.

The supplied native host supports camera/stereo observations and explicit job-owned
scene/physics changes. Custom observation functors, deployment event functors,
non-camera sensor histories, arbitrary topology edits and unsupported collision
shapes need an explicit custom host/sink adapter; preflight does not silently fall
back to B=1. Single-object Newton damping is deliberately not advertised. Source
restrictions still apply to any requested motion or layout change.

## Validation and measurement

CPU tests cover receipts/recovery, media clocks and hashes, source ancestry,
logical randomness at B=1/8/32, capacity/backpressure, selected states/parameters,
source permissions and the existing trajectory/Task Program regressions. Native
probes run in isolated processes; configure `EXPANSION_REAL2SIM_DEPLOYMENT` and
`EXPANSION_REAL2SIM_PROFILE` with a freshly qualified fixture for that source test.

```bash
python scripts/benchmark/expansion/offline_collection.py \
  --job examples/expansion/engine/job.default.yaml \
  --output /tmp/expansion-benchmark
```

The harness runs B=1,8,32 in state and RGB-D modes, records fresh-process wall time,
confirmed episodes/second, success rate, RSS, process GPU memory and native timing
sections. `runtime.json` separates startup, preparation, execution/validation,
commit, rerender and export. Physics, render and sensor-fetch timings remain in the
nested native profiler. Atomic planning has a separate native profiler section. The common engine reports
preparation, admission, execution, staging and submission; the host separates state
capture, source binding, media buffering and acceptance. Nested times must not be
added as independent wall-clock totals. Those costs must not be misreported as physics time.
`coverage.json` separates geometry buckets, actual physical fingerprints/ranges,
visual descendants and source ancestor groups; visual children grant no physical
success credit. These are diagnostics, not an automatic deduplication policy. Compare equal
quality checks and report thread settings, startup and rendering costs. No speedup
is assumed from batch size alone.

## Foundation measurements

Single sequential run on NVIDIA RTX PRO 5000 72GB Blackwell. Process startup, planning,
independent acceptance and confirmed writes are included. Thread settings are in
`qualification.json`. Cache warmup and one-run variance prevent release speedup claims.

| Mode | B | Confirmed | Seconds | Confirmed episodes/s | Peak process-tree RSS (GiB) |
|---|---:|---:|---:|---:|---:|
| state | 1 | 1 | 57.77 | 0.017 | 14.29 |
| state | 8 | 8 | 69.05 | 0.116 | 14.12 |
| state | 32 | 32 | 269.67 | 0.119 | 14.12 |
| rgbd | 1 | 1 | 42.52 | 0.024 | 15.79 |
| rgbd | 8 | 8 | 85.50 | 0.094 | 15.61 |
| rgbd | 32 | 32 | 352.01 | 0.091 | 15.77 |

## Shared-engine validation

The revised scheduler was exercised with Default native motion (B=2), Default
Real2Sim prototype motion (B=2), and Newton native motion (B=2): each confirmed two
physical episodes. The prototype job resumed at B=8 with no duplicate attempts.
A Default RGB-D run confirmed two parents, generated four visual children and
exported six episodes / 2,382 frames. After moving that dataset and hiding its
original job directory, validation reported zero errors/warnings; actions, states
and endpoint RGB frames matched the frozen evidence. These probes establish the
tested combinations, not a complete source/backend qualification matrix.

`calibration_readout.py` measures the native camera getter/packing path independently
of task throughput. With CPU observation tensors, one thread, 30 measured iterations
and three warmups on the same RTX PRO 5000 host, old per-row versus batched mean
readout times were 0.0255/0.0265 ms (B=1), 0.926/0.145 ms (B=8), and
10.573/0.456 ms (B=32). Calibration was exactly equal and physics did not advance.
B=1 has no measured improvement. These numbers exclude planning, rendering and
writing and must not be reported as end-to-end collection speedups.
