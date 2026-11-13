# Video and joints to atomic motion

Run in `py312`. The production command supports explicit `pick_up`, `slide`,
`twist` and `hand_over` motion goals. It needs no object poses, object meshes,
or running simulator. Existing annotation files select the segment; automatic
video contact recognition is not part of this command.

## Reproduce the Pickup round trip

```bash
conda activate py312
embodichain run-example sim/atomic_skill --skill pickup --headless \
  --save-video --output-dir /tmp/pickup-source
embodichain run-example toolkits/skill_fitting \
  --replay-dir /tmp/pickup-source --output-dir /tmp/pickup-fit
```

The second command reads only the replay's joint commands and MP4. Its
sample-owned robot profile supplies the UR5 URDF, mounted PGI TCP, named arm/hand
bindings, open/grasp commands and explicit reference limits. It does not read the
replay JSON's sampled grasps or object information. The reference limits in this
example are demonstration settings, not calibrated hardware safety limits.

Outputs are `robot_profile.json` and `pickup_fit.npz`. The command exits with
status 2 when a generated candidate fails the acceptance gates. Such results
retain `native_qpos` and `proposed_qpos`, with diagnostics; `patched_qpos` remains
the unchanged original reference.

## Fit an annotated recording

```bash
embodichain fit-trajectory \
  --video /data/episode.mp4 --joints /data/joints.npz \
  --annotations /data/episode.segments.json --segment-id seg_example \
  --skill pick_up --robot-profile /data/robot_profile.json \
  --qpos-semantics measured_state --time-offset 0 \
  --output /data/episode.pickup.fit.npz
```

Video intervals are zero-based and inclusive. The video adapter uses decoded
presentation timestamps; the command maps the last selected frame's display
duration into a half-open joint interval. `--time-offset` is the joint timestamp
of video time zero. The joint sampling clock is never inferred from video FPS.
Without an annotation file, `--start-frame` and `--end-frame` select an interval;
their defaults cover the full video.

An annotation may retain a broad real-motion interval and optionally declare
`core_interval: {"start_frame": 120, "end_frame": 185}` inside it. Both endpoints
are inclusive absolute video frames. The CLI and website reconstruct only this
core and preserve all surrounding samples; the report retains both the outer
annotation and the actual fitted video interval. The core must still contain a
complete supported skill. Schema 3 stores this optional field; legacy schema 1/2
files without it remain supported. The annotation menu includes all built-in
skills, including DemoSkill; parameter reconstruction remains limited to the four
skills above. A DemoSkill label preserves intent without generating an executable
demonstration asset or substituting a different fitting model.

For existing HDF5 camera annotations, replace `--video` with `--hdf5` and
optionally select `--image-key` / `--fps`. This reuses the annotator's source
identity and explicitly identifies FPS-derived timing.

Joint inputs:

- NPZ: `qpos` shaped `(T,D)` and strictly increasing `timestamps` in seconds.
  Optional `joint_names` allows explicit column reordering.
- HDF5: select numeric fields using `--qpos-key observations/qpos` and
  `--timestamps-key observations/timestamps`. Missing timestamps require
  `--joint-dt`; they are then declared uniformly sampled.
- Local Atomic PT replay: weights-only loading of the selected environment's
  `positions` and cumulative `dt`, including recorded final holds. Use
  `--qpos-key planned_positions` to explicitly select the planned clock.

Declare whether the input is measured state, submitted commands or a design
reference. Reconstructed output is always a design reference.
Complete joint layouts currently include the configured hand columns. For a
hand whose recorded state does not expose an open/close event, a complete
`--phase-hints` partition can provide event indices. Arm-only recordings require
explicitly preparing the missing hand reference; the CLI does not invent an
observed aperture or closing command from RGB.

`--robot-profile` is a closed JSON configuration. The generated example profile
shows the fields. Joint lists and participant bindings are explicit; custom
names are retained. Solver configurations use existing FK/IK solvers with an
explicit URDF, root/end links, TCP and fixed arm base poses. Paths resolve against
the profile file. Both arms of a handover need a calibrated common frame.

`--fit-config` accepts the declared `SkillFitCfg` fields, including finite sample,
candidate and wall-clock budgets and independent joint/TCP/contact tolerances.
The wall-clock budget is cooperative between native plans, not a preemptive
native-solver timeout.

The website additionally offers relaxed, standard and custom tolerance choices.
The relaxed choice sets joint RMSE to `0.5` in source joint units, TCP/phase
position errors to `0.1 m`, and rotation/phase rotation errors to `1 rad`.
It preserves unrelated fitting options and requires a new fit to take effect.
Standard mode uses the canonical `SkillFitCfg` error defaults, which also remain
the CLI defaults. Robot position, velocity and acceleration qualification is
independent of these geometric error settings.

Boundary connections retain the first/last three original samples exactly,
including the finite-difference stencils at both outer joins. The default
`blend_duration_s=0.5` requests a quintic transition over physical time, with
`blend_samples=12` as an additional minimum sample width. Entrance and exit
widths are capped independently by phase anchors and available segment length.
For a longer connection, use `{"blend_duration_s": 0.8}` in `--fit-config`
or the website's configuration JSON. No samples or time are appended, and values
outside the selected interval remain unchanged. A longer transition can reduce
endpoint spikes; it does not remove noise already in the recording or bypass
contact/derivative acceptance gates. Leave enough surrounding motion in the
selected interval for the connection.

Pre-grasp anchor inference tolerates orientation noise up to
`max_approach_rotation_rad` (default `0.03` rad, about 1.7 degrees). Set it in
`--fit-config`, or the annotation website's fitting-configuration JSON. This
controls anchor inference separately from the final TCP/contact acceptance
limits. Complete Slide/Twist phase hints use the explicit `reach.start` anchor
without first requiring automatic orientation-based inference.

## Phases and limitations

The command uses hand opening/closure and FK to estimate phases. `--phase-hints`
can supply a JSON array of `{name, start, stop}` with half-open, segment-relative
joint indices. It must cover the whole selected segment contiguously.

| Skill | Source phases |
| --- | --- |
| `pick_up` | `approach`, `close`, `lift` |
| pull `slide` | `approach`, `reach`, `close`, `pull`, `open` |
| push `slide` | `approach`, `reach`, `close`, `push`, `open`, `return` |
| `twist` | `approach`, `reach`, `close`, `twist`, `open`, `retract` |
| transfer-only `hand_over` | `transfer`, `approach`, `close`, `hold`, `release`, `retreat` |

Handover optionally prepends `pickup_approach`, `pickup_close` and appends
`place`, `receive_release`, `receive_retreat`. Receiving closure must precede
source release. Participants are always explicitly bound.

Pickup requires an observable stable-orientation approach and a world-Z lift.
Slide fits a translating rail; arbitrary tabletop sliding may have a different
skill contract. Twist needs sufficient rotation around one line; the stored
axis point is the closest point on that line to the initial TCP, not an object
origin. Unobservable phases fail explicitly. A measurable positive Pickup lift
with excess lateral motion or rotation still produces a vertical,
fixed-orientation native candidate for inspection. Its report lists the measured
template errors and rejects the replacement; `patched_qpos` retains the original
reference. Final native, contact and derivative checks remain independent of
approach inference. Tests cover all four skills using an independent Cartesian
robot oracle; the recorded robot round-trip is qualified on the UR5 Pickup example.

## Standalone result

The NPZ contains original/native/patched/proposed joints, physical timestamps,
source-derived monotone phase progress, raw native phase joints, modification
mask, and decimated FK link/TCP geometry. `metadata_json` stores parameters,
native options, phases, initial qpos, calibration/profile/annotation/source
identities, timing interpretation and independent native/patched diagnostics.
`boundary_connection` records the transition target and fixed endpoint samples.
`native_qpos` is the time-aligned native skill before boundary connection; its
joins may jump. Inspect `patched_qpos` (accepted) or `proposed_qpos` (candidate)
to assess the connected reference.
It uses numeric arrays and JSON, with pickle disabled.

For plotting:

```python
from embodichain.data_pipeline.skill_fitting import load_skill_fit

arrays, report = load_skill_fit("/data/episode.pickup.fit.npz")
error = arrays["patched_qpos"] - arrays["original_qpos"]
```

All original joint samples are included; visualization geometry uses the saved
`geometry_indices`. Viewing does not require the source video, joint recording,
URDF, or simulator. The annotation website also runs this pipeline and displays
video-linked joint curves, errors, rotatable FK skeletons, TCP paths and geometric
error magnification. See the [web example](../trajectory_annotation/README.md).
