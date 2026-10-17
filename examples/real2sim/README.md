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

The shared `ExpertSourceRegistry` now fingerprints exact provider versions and
expanded call selections in the simulation catalog. The configured
`expert_source_providers` service admits a content-pinned source bundle before
simulation construction. `Real2SimExpertSourceProviderFactory` creates fresh B=1
Pick providers, and PickUp owns complete nominal-plan assembly. Neither source
code nor callables belong in `program.yaml`.

Physical qualification, receipt-backed array attachments, registered-source
combined expansion, time-grid changes and `eef_pose`/`hybrid` planning remain
subsequent milestones. The existing same-grid source adapter and coordinator
remain available; configured combined expansion with a registered source is
rejected rather than changing its phase permissions.

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

## Four-panel kinematic video

`render_preview.py` adds a native EmbodiChain renderer preview for the optional
Franka arm recording. It shows the source camera, nominal arm qpos, and two
seeded joint-residual variants. The small plots show **flange XY paths**, not a
calibrated gripper TCP. The source NPZ must explicitly contain seven Franka
`joint_positions` columns in `fr3_joint1`–`fr3_joint7` order and `timestamps` in
milliseconds. The source video must be 640×360 with N or N−1 frames.

```bash
PYTHONPATH="$PWD:$PWD/extensions" python examples/real2sim/render_preview.py \
  --input /path/to/gt_action_full.npz \
  --video /path/to/camera_rgb.mp4 \
  --output outputs/real2sim-preview/run01 \
  --free-stop 60 --scale 0.025
```

Choose `--free-stop` only after reviewing the recording: it is the exclusive
source sample index of a conservative free-motion prefix. For the selected
sample, 60 protects the final approach and all later contact, transport and
release motion. The protected suffix uses a conservative contact classification;
it does not claim to have finely segmented those actions. The editable window
is inset to preserve boundary positions and velocity stencils.

The preview deliberately remains outside qualified source admission: calibration
and annotations remain unverified, `Real2SimSourceAdapter` is not relaxed, and the
example applies the existing `joint_residual` operator directly to a diagnostic
`TrajectoryTemplate`. It does not claim a `CandidateCoordinator` acceptance or
an `EpisodeSink` receipt. Synthetic production tests cover the qualified adapter
separately.

Three render arenas display the same official Franka arm and Robotiq 2F85 assets,
assembled inside the output directory. There are no reconstructed scene objects.
The gripper defaults to fixed open; the optional explicit lookup below animates
the declared scalar convention. The mount remains an uncalibrated identity transform. Robot states are written for
kinematic inspection and rendered without physics stepping. Joint-position and
velocity limits, protected qpos/qvel and runtime qpos readback are checked;
collisions, tracking under dynamics and task success are not qualified.

The video follows the state clock, not the camera file's nominal 30 FPS. Default
video frame i maps to source state i; `--video-state-offset -1|0|1` makes that
assumption explicit. Nearest source samples and endpoint clamping handle N−1
frames. The offset still requires independent projection-based calibration.

Outputs include `comparison.mp4`, review stills, `trajectories.npz` (all three
paths, qvel, FK, protected mask and render/source/video mappings), `report.json`
(provenance, asset hashes, amplitude/limit metrics and limitations), and the
assembled robot asset. `--max-frames 1` runs a bounded render smoke check. Outputs
must be in a new directory outside the original recording directory.


## Registered B=1 Pick integration

A task-local integration can select a verified source using only stable IDs and
content-pinned data references:

```yaml
runtime_services:
  expert_source_providers:
    - call_id: pick
      call_index: 0
      provider_id: real2sim_qpos
      revision: "1"
      target_profile_id: my_franka_profile
      target_asset_sha256: <SHA-256 of the resolved target robot URDF/USD>
      bundle_path: ./sources/pick/bundle.json
      bundle_sha256: <SHA-256 of bundle.json>
```

`call_index` selects one global expanded program occurrence. The catalog rejects
missing/mismatched occurrences, unknown versions, duplicate selections and
unsupported capabilities. Data paths resolve relative to the integration file.
The normal execution-policy component must set both `max_replans: 0` and
`max_action_retries: 0` for fixed replay. External nominal sources currently
require sequential segments. Fresh providers share the existing planning scene
provider and exact embodiment profile; Python callbacks cannot override a
registered source binding.

Create source bundles with `write_real2sim_bundle(source, layout, cfg, directory)`.
`load_real2sim_bundle` verifies the declaration/NPZ digests and closed fields,
reconstructs raw contracts, reruns normalization and rejects unreviewed inputs by
default. Large arrays stay in the NPZ sidecar. There are no executable loaders or
pickled arrays. Data digests are rechecked when creating each live source provider.

The initial Pick source has full `approach`, `close`, `lift` semantic phases.
Close and lift use protected `contact`/`hold` kinds. Interpolation-only gaps remain
locked by the adapter; the plan assembler attributes those gaps to the preceding
semantic segment. All source/execution phase intervals remain in diagnostics.
The provider checks B=1, exact profile/model digest/joint order, initial state, control period,
endpoint ownership and live joint-position limits. PickUp then builds commands,
qvel/tracking, segments, dependency boundaries and uncommitted held-object effects.
Source failures propagate without ordinary planning fallback. A fresh reset is
required to run another fixed source attempt.

The B=1 bridge regression uses reviewed **synthetic** source values and projected
semantic completion. It proves the interface and lifecycle, not physical grasp
success. Existing verified-runtime tests continue to cover measured-effect failure.

## Original-recording calibration audit

`prepare_pick_sample.py` consumes only one original HDF5, independently of any
real2sim experiment code or generated scene:

```bash
PYTHONPATH="$PWD:$PWD/extensions" python examples/real2sim/prepare_pick_sample.py \
  --hdf5 /path/to/original_sample.hdf5 \
  --output outputs/real2sim-source-sample/pick_audit \
  --upstream-revision c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be \
  --close-start 97 --lift-start 106 --stop 121
```

The selected source has an open approach, a closing transition at samples 97–105,
and subsequent lift. These indices are example annotations, not an automatic
contact detector. The script preserves raw gripper samples, camera matrices,
source pose and pipeline calibration metadata in a separate evidence sidecar.
It writes a diagnostic source bundle, explicit jaw-width/driver lookup and audit
report. Original files remain read-only.

The [pinned upstream DROID implementation](https://github.com/droid-dataset/droid/blob/c5737e40a6b18859b5b78dbcdbf1e3b3f5e461be/droid/franka/robot.py)
defines gripper position as normalized closure (`1 - width/max_width`). The audit
uses this explicit convention, preserves the exported canonical values separately
in the original NPZ, and does not interpret the scalar as driver radians. The
width-to-driver table comes from the bundled Robotiq contact-pad mesh and includes
estimated endpoint extensions; it remains unverified hardware calibration.

To animate that estimate in the diagnostic renderer, add
`--gripper-calibration /path/to/gripper_calibration.json` to the four-panel command.
This replaces the fixed-open visual policy without granting formal admission.

The original sample still lacks independently reviewed real mount/TCP, hardware
jaw calibration, camera reprojection/frame offset and live target-joint order.
Its audit therefore reports `rejected_until_reviewed`. The near-zero source-pose
versus flange-FK residual is a conversion consistency check, not an independent
measurement-accuracy result. Do not promote these input states automatically.

## Physical interaction and expansion preview

`physics_preview.py` drives the same recorded arm path and two seeded residual
variants through the shared `play_joint_trajectory` host, with 1 ms Default
physics and 10 ms PD command periods. Each candidate runs sequentially at B=1
from the same reset state. The block is dynamic; table and container proxies are
static. Object poses are written only at reset. Gravity, contact and friction
produce grasping, transport, release and settling; no object attachment or
object-trajectory playback is used.

```bash
PYTHONPATH="$PWD:$PWD/extensions" python examples/real2sim/physics_preview.py \
  --input /path/to/gt_action_full.npz \
  --gripper-calibration /path/to/gripper_calibration.json \
  --source-video /path/to/camera_rgb.mp4 \
  --output outputs/real2sim-physics/run01 --repeats 3
```

Generate the explicit gripper lookup with `prepare_pick_sample.py` first. This
example uses the selected recording's source indices 105 and 160 to anchor the
estimated grasp and release geometry. The block width follows the declared jaw
width at closure; length/height, mass, friction, container and table are shared
model estimates. The robot base is translated to 0.75 m in the preview world.
The actual measured gripper position is converted into a controller target with
an explicit 0.015 rad squeeze preload while closing. This preload is shared by
all candidates, stored separately from raw driver values, and configurable with
`--grip-preload`. It is a control assumption, not recovered real command evidence.

Use `--no-render --cases 0 1 2 --repeats 3` for a bounded physics-only run, or select
one case while investigating failure. Parameters such as `--mass`, `--friction`,
`--grip-effort` and `--grip-stiffness` are explicit and recorded. They must remain
fixed across a comparison; changing them produces another diagnostic condition.

The contact sensor resolves exact left/right inner-finger actor identities. The
measured proxy-scene criteria require the block to rise at least 4 cm with
continuous bilateral contact for at least 1 s, finish wholly inside the container
(using oriented box corners), and remain stable in position and orientation for
the final 1 s. Stability also limits linear and angular speed. A motion-limit pass
or a robot reaching its endpoint does not establish this physical result.

Outputs include per-candidate native videos, the four-panel
`physical_comparison.mp4` when `--source-video` is supplied, controller targets,
measured object/joint/flange traces, contact/impulse evidence, actor identities,
settings and per-trial results. The first repeat of each candidate is rendered;
every repeat is physically executed and evaluated. Interpolation-only and contact
motion remains protected exactly as in the kinematic diagnostic.

These results qualify the **estimated proxy-scene diagnostic only**. The real
mount/TCP, hardware jaw calibration, camera/frame alignment and reconstructed
asset geometry remain unverified. Physical-preview success does not change the
source bundle's admission state, allocate accepted generation candidates or
produce an EpisodeSink commit receipt. The registered real2sim source still
rejects the unreviewed original recording.
