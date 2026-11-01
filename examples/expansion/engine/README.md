# Offline Expansion Engine

`embodichain expansion` collects independent expert episodes in one batched World.
It reuses Task Program/Atomic planning, the original Real2Sim qualification path,
ExpansionSession accounting and the existing LeRobot writer. ODS remains unchanged.

## Draft status

This is the foundation draft on top of Real2Sim #896 and Franka #916; scene/workspace
#912 has merged. Resume release qualification after those prerequisites merge.
Native Default collection and Task Program Newton runs have positive evidence.
The tested Newton DROID carton run was rejected by physical-effect/segment checks;
it is **not qualified**, and no thresholds or source protections were relaxed.
Complete source/backend matrices, broader atlas/motion recipe qualification,
portable calibration/attachment export and final performance tuning remain follow-up
work. The CLI rejects unsupported combinations rather than claiming full coverage.

## Run a configured job

From the repository root, with the simulation dependencies installed:

```bash
embodichain expansion preflight --job examples/expansion/engine/job.default.yaml
embodichain expansion run --job examples/expansion/engine/job.default.yaml
embodichain expansion inspect --output /tmp/embodichain-expansion/default
embodichain expansion resume --job examples/expansion/engine/job.default.yaml
```

The output must be unused for `run`. `resume` checks the pinned configuration,
source evidence and commit receipts. It retries frozen writes without another
rollout, and starts a new attempt for an interrupted physical episode. A bounded
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
Validate the published dataset with:

```bash
embodichain dataset validate DATASET_PATH --check-media --media-frame-counts
```

## Real2Sim

Prepare the pinned DROID carton example using its unchanged raw recordings and
qualification annotations. The stable physical geometry fingerprint excludes URDF
comments, rendering-only material/visual sections and generated file locations; it
still includes physical geometry bytes, joint structure and inertial properties.
Old exact raw-file fingerprints remain accepted only when they match exactly.
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
nested native profiler. Planning/runtime and independent validation are included
in execution; those costs must not be misreported as physics time. Compare equal
quality checks and report thread settings, startup and rendering costs. No speedup
is assumed from batch size alone.

## Foundation measurements

Single sequential run on NVIDIA RTX PRO 5000 72GB Blackwell, NVIDIA RTX PRO 5000 72GB Blackwell, NVIDIA RTX PRO 5000 72GB Blackwell, NVIDIA RTX PRO 5000 72GB Blackwell, NVIDIA RTX PRO 5000 72GB Blackwell, NVIDIA RTX PRO 5000 72GB Blackwell, NVIDIA RTX PRO 5000 72GB Blackwell, NVIDIA RTX PRO 5000 72GB Blackwell. Process startup, planning,
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
