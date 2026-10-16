# Source-independent real2sim foundation

This example exercises the reusable qpos source contracts proposed in #709.
It does not import a DROID experiment, reconstruct a scene, tune a robot, or
start a second collection lifecycle. The original recording remains external
and read-only. Ordinary tests use synthetic arrays, not this optional file.

## Implemented slice

- Owned raw trajectory, joint-layout, phase and optional EEF evidence values.
- Explicit unit/joint mapping, initial-state checks, declared holds and mimic derivation.
- Fixed-clock resampling with exact event samples and bidirectional timing evidence.
- A `Real2SimSourceAdapter` for the existing `CandidateCoordinator`/`ExpansionSession`.
- A complete nominal-plan injection boundary in the Atomic engine, retained during
  recovery and invocation revision, and a call-scoped Task Program factory hook.
- Regression tests for bypassing ordinary planning, preserving effect verification,
  deterministic variants, protected derivative boundaries and malformed data.

The shared registered provider catalog/configuration entry, Real2Sim provider's
Pick-specific semantic assembly, physical qualification and receipt-backed array
attachments remain subsequent milestones. The runtime hook is a trusted Python
integration boundary, not permission to put executable imports in `program.yaml`.
No `ExpertSourceProviderFactory` registry or configured service is claimed to exist
in this slice. `eef_pose` and `hybrid` explicitly require a later IK/replanning
provider; the qpos normalizer rejects these authorities.

## Contracts

Create `Real2SimTrajectory` and `RobotJointLayout` independently of a dataset
format, then call `normalize_real2sim(source, layout, cfg)`. The host supplies
`control_dt`; normalization never changes the simulator physics clock.

Source timestamps strictly increase. Duplicate timestamps must be coalesced by
an explicit upstream operation that preserves event provenance. Each phase is
half-open `[start, stop)`, with protected samples at `start` and `stop - 1`.
Hard anchors round forward to control ticks. The default maximum delay is one
control tick per anchor. Incompatible short event sequences raise instead of
silently dropping contact phases. Interpolation-only gaps between remapped
phases remain unannotated and therefore locked.

The v1 optional `EefPoseSamples` stream is aligned to source joint timestamps.
An independent EEF clock must be aligned explicitly upstream and its operation
recorded in provenance. Translations in frame/TCP transforms are in metres;
pose translation and quaternion conventions are declared separately. An
`fk_derived` pose is never reported as independent measurement evidence.

The adapter always rejects unverified input prerequisites. Offline diagnostic
normalization is possible with `allow_unverified=True`, but that output cannot
enter the adapter. Verified prerequisites still do not imply physical success.
V1 permits only same-grid `joint_residual` and requires enough free samples to
protect both endpoints and their adjacent derivative stencils. Template free
windows are inset by one sample at each end; full semantic phase ranges remain
in the normalized unit/report and must be used by future Skill plan assembly.
Expandable mimic-driver groups are rejected until an explicit coupling-aware
candidate projection is available. Ordinary gripper groups can remain locked.

The template source revision is the normalization content digest. The host's
`SourceContext` must match source ID, unit ID, revision and control period;
candidate IDs are still allocated only by the existing coordinator/session.

## One optional real recording

Run from a checkout with the normal EmbodiChain dependencies installed:

```bash
python examples/real2sim/normalize_recording.py \
  --input /path/to/recording.npz \
  --output outputs/real2sim-source-sample/run01 \
  --joint-names joint_a joint_b \
  --joint-units rad rad \
  --robot-id recorded_robot \
  --robot-revision unverified-recording-model \
  --time-unit ms --control-dt 0.01
```

The NPZ defaults to `joint_positions`, `timestamps` and optional
`joint_velocities`; key names are CLI arguments. `--start-index` and
`--stop-index` select one source slice. The provided channels form a diagnostic
layout, not a verified full robot. The script deliberately marks calibration
and phase annotations unverified and cannot certify a Pick or commit a dataset
episode. It refuses to write into the source directory or overwrite its reports.

Optional EEF evidence requires both `--eef-key` and `--eef-description`.
The JSON description supplies `reference_frame`, `observed_link`, `tcp_id`,
`reference_to_base`, `link_to_tcp`, `origin`, `quaternion_order` and
`position_unit`. Supplying a TCP pose requires an identity link-to-TCP transform;
do not apply the same tool offset twice.

The output contains `trajectory.npz` (raw and normalized arrays) and
`normalization.json` (source digest, unit mappings, phase/timing maps and explicit
qualification limitations). These are offline diagnostic artifacts, not
`EpisodeSink` commit receipts. Gripper channels must not be reinterpreted from
their numeric range. Decode a known width/angle convention upstream, or leave
it unresolved and keep the sample diagnostic-only.

## Focused validation

```bash
pytest tests/compute/test_trajectory_anchors.py tests/data_pipeline/real2sim \
  tests/sim/motion/expansion/test_real2sim.py
pytest tests/sim/atomic_actions/test_engine.py tests/sim/atomic_actions/test_engine_per_env.py
pytest tests/lab/task_program/test_semantic_executor.py tests/gym/envs/task_program/test_environment.py
```

Tests exercise production contracts. No tutorial, experiment outputs, local
recording path or reconstructed asset is a CI dependency.
