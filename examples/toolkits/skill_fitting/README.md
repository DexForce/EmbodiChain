# Video and joints to atomic motion

Run in the installed project environment (for the local reconstructed episode,
`py312_bk`). The command fits `pick_up`, `slide`, `twist` and `hand_over` through
their native planners. Reconstructed scenes provide valid antipodal grasp anchors
for the single-arm skills. The original motion-only route remains available when
no scene is supplied. Fitting itself does not start a simulator. Accepted whole episodes can export a native run-task bundle.
Existing annotation files select the segment; automatic RGB contact recognition
is not part of this command.

## Fit a reconstructed scene

For a complete portable web project, start with only the extracted resource folder:

```bash
conda activate py312_bk
embodichain annotate-trajectory --bundle /data/pickup-workflow.inputs --port 8898 --no-browser
```

Its manifest supplies the video and all joint, robot and scene defaults. Edit skill
intervals, fit the whole recording and download the run-task ZIP in the webpage.
No original recording directory is needed. The resource contract and packaging
commands are below. Individual recording inputs remain supported:

Set `TRAJ_EP` to your recording folder. The cup and pen frame ranges below are
examples of two independent pickup intervals.

```bash
conda activate py312_bk
TRAJ_EP=/path/to/recording

python -m embodichain annotate-trajectory \
  --video "$TRAJ_EP/main.mp4" --annotations "$TRAJ_EP/cup.segments.json" \
  --joints "$TRAJ_EP/joints.npz" --robot-profile "$TRAJ_EP/robot_profile.json" \
  --scene "$TRAJ_EP/blender_scene/physical_export" \
  --participant auto --qpos-semantics design_reference --time-offset 0 \
  --device cpu --port 8898 --no-browser
```

Use the same robot profile for the right-hand cup segment `[45,148]` and the
left-hand pen segment `[196,299]`. The webpage displays the selected motion/hand
groups and object. A standalone fit of the first segment is:

```bash
python -m embodichain fit-trajectory \
  --video "$TRAJ_EP/main.mp4" --joints "$TRAJ_EP/joints.npz" \
  --robot-profile "$TRAJ_EP/robot_profile.json" \
  --scene "$TRAJ_EP/blender_scene/physical_export" \
  --skill pickup --start-frame 45 --end-frame 148 \
  --participant auto --qpos-semantics design_reference --time-offset 0 \
  --output /tmp/cup.scene-fit.npz
```

Change the frame bounds to `196` and `299` for the pen. `--object-id` can pin a
scene object; otherwise the nearest mesh-bounds centre to the observed grasp is
selected. The tool never infers an object pose from the video. Without `--scene`,
single-arm fitting also discovers `blender_scene/physical_export` beside the joint
recording when that directory contains `scene_assets.yaml`.

The directory format reads local mesh files from `objects.<id>.shape.fpath` in
`scene_assets.yaml`, and world-from-object matrices from the corresponding
`export_report.json` entry's `anchor`. A standalone YAML may instead declare
`objects.<id>.pose` as a proper 4×4 world matrix. Coordinates must be in metres
and the same world frame as the supplied robot calibration; no extra robot-root
transform is applied to already-world object poses. Optional `graspable: false`
excludes a declared obstacle from automatic target selection.

Automatic participant selection uses the profile's explicit joint lists and URDF
ancestry to pair hands with motion groups, then finds a unique closing hand in
the selection. It does not interpret control-part names or require separate left
and right profiles. Ambiguous evidence fails explicitly; `--participant <part>`
selects one configured motion/hand name and `--participant configured` preserves
the profile's role bindings. HandOver continues to require configured
source/destination roles; an explicit scene request for it requires a future
time-resolved object/attachment binding and fails rather than guessing one.

Scene grasps reuse `AntipodalAffordance.get_grasp_candidates` and the actual
open-hand collision surfaces. The supplied URDF must include the finger collision
meshes. Automatic geometry currently supports two opposing prismatic finger
joints. Contact-frame X follows their configured opening/closing travel; contact
Z follows the calibrated TCP approach axis. The canonical contact origin uses the
overlapping collision-tip bounds minus `grasp_contact_inset_m`; public candidates
are converted back to the user's unchanged TCP. This geometric convention is
recorded in the report and is not measured hardware contact calibration.

The fitter estimates approach direction and skill motion parameters from FK,
ranks valid grasps by translation distance plus `grasp_rotation_weight_m` times
SO(3) angle, and uses the closest candidate with a complete native kinematic plan.
For Pickup this is an object-bound `GraspGoal`, followed by normal approach,
closure and lift generation. It does not require a perfectly constant observed
pre-grasp rotation or pointwise agreement with every noisy source sample.
Slide/Twist retain their native axis-motion contracts with the selected anchor.
Grasp generation tests both the nominal TCP pose and its exact local-Z half
turn (`T_tcp @ Rz(pi)`), keeping TCP position and approach Z fixed. Each physical
hand realization is independently checked against the target mesh; rotations
and translations between the contact frame and virtual TCP are included.
Full TCP-pose ranking chooses the closer qualified alternative to avoid wrist
flips. Pickup's native approach follows the selected TCP Z. There are no random
cone proposals on each fit.

Scene fitting defaults to `grasp_approach_deviation_rad = pi/3` (60 degrees) and
`grasp_filter_ground_collision = false`. The wider sampling angle admits grasps
near the recorded orientation that the former 30-degree restriction rejected.
Disabling the inferred plane avoids rejecting candidates solely because the
gripper intersects a plane placed at the target's lowest vertex; this also
disables that conservative table proxy. Target-object surface checks and joint,
velocity, acceleration and boundary checks still apply. Complete robot/table
collision freedom remains unverified. To restore stricter sampling, pass a
`--fit-config` JSON file (or fill the web panel's JSON field), for example:

```json
{"grasp_approach_deviation_rad": 0.5235987755982988, "grasp_filter_ground_collision": true}
```

The scene report records the symmetry convention, selected roll, approach angle,
and ground-filter policy. `parameters.grasp_rotation_error_rad` records the
selected grasp's rotation difference from the observed TCP.

Each replacement stays on the recorded clock and changes only the selected
motion/hand columns. Automatic Pickup fitting may redistribute approach/lift
time within that fixed outer clock when the selected grasp needs it, while
preserving the observed closing duration. Explicit phase hints remain fixed;
`source_phases` records the observed partition and `phases` records the chosen
one. At most six schedules are checked per grasp, keeping one best path's arrays.
Quintic free-phase bridges preserve the source's outer
position/velocity/acceleration stencils and keep the contact core native. Every
changed position and derivative stencil, including the inner joins, must meet
the configured limits. Existing violations elsewhere in the source are retained
and reported separately under `source_limits`; passing this check does not
certify the entire original recording. Rejected candidates remain downloadable
without replacing `patched_qpos`. Source video, joints and object assets remain
read-only. Fitting multiple intervals does not accumulate replacements.

Candidate generation is bounded by `grasp_sample_count`, `grasp_max_candidates`
and `max_candidates`; there is one temporary generator per fit. Scene, robot
geometry and recording content hashes are checked before returning a result.
Finite open-hand surface checks concern the target and inferred support plane;
full-arm swept collision, closing contacts and physical task success remain
unverified.

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

`hand_commands.<grasp_part>.grasp` supplies the full-close qpos target; `open`
supplies opening, in `control_parts[grasp_part]` joint order and native units.
Use the model's joint limits: a loaded gripper can stop short of the commanded
closure because of contact. Replacing the target with that measured opening
removes closing preload. Native sequence spans therefore update only bound
motion/grasp parts and hold every other joint at its previous command. DemoSkill
spans continue to use their original full-joint commands, including releases.

`solvers.<motion_part>.tcp` is the 4×4 **end-link-from-TCP** transform; translation
is `tcp[:3,3]` in metres, along the end-link axes. Runtime export copies this into
`task.yaml` at `robot.solver_cfg.<motion_part>.tcp`. For the recording's Marvin
right arm, shortening the offset changes `tcp[0][3]` from 0.13 to 0.12 m; the
left arm keeps its existing calibration. This is separate from the scene's world-Z
grasp compensation. Refit/export after source calibration changes to refresh
derived grasp goals and sealed input hashes together.

`--fit-config` accepts the declared `SkillFitCfg` fields, including finite sample,
candidate and wall-clock budgets and independent joint/TCP/contact tolerances.
The wall-clock budget is cooperative between native plans, not a preemptive
native-solver timeout.

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
origin. Unobservable or incompatible segments fail explicitly. Tests cover all
four skills using an independent Cartesian robot oracle; the recorded robot
round-trip is qualified on the UR5 Pickup example.

## Minimal portable scene inputs

The portable resource contract is `scene-skill-fitting-inputs/v1`. It needs these resources:

| Resource | Contents |
| --- | --- |
| `fitting.json` | Version, relative paths, fitting settings and SHA-256 inventory |
| `scene/scene_assets.yaml` | Object-local meshes, world-from-object 4×4 poses, dynamics and fitting defaults |
| `scene/*.obj` | Visual meshes and precomputed convex parts only where requested by collision configuration |
| `robot_profile.json` | Explicit joint layout, control parts, solvers, TCP/base calibration, limits and simulation overlays |
| `robot/robot.urdf`, referenced assets | Complete robot and only the mesh/texture dependencies it uses |
| `joints.npz` | Original full-joint `(T,DOF)` qpos, joint names and strictly increasing timestamps in seconds |
| `annotations.json` | Inclusive frame intervals, optional cores and original skill labels |
| `frame_clock.npz` | Actual frame presentation times and display durations; video pixels are unnecessary for fitting |
| `video/<filename>` | Original encoded video, required for web startup and declared by optional `fitting.json.frame_source` |

Lengths are metres, angles radians, transforms are world-from-local. Grasp targets and robot calibration share one world frame. The folder needs no Blender file, export report, USDA, original directory or previously fitted trajectory. Offline fitting uses the frame clock; the web workflow additionally includes the original video, with its frame order and timestamps preserved. The installed EmbodiChain runtime, annotation extras and solver dependencies remain required for the complete workflow.

Add video to an existing offline input package, without running a fit:

```bash
embodichain fit-trajectory --bundle /data/portable-inputs \
  --package-output /data/pickup-workflow.inputs --package-video /data/main.mp4
```

The packager verifies that the video content/stream matches the annotation seed.
All dependency paths are rewritten to contained relative paths and pinned by
SHA-256. Moving the resulting folder requires no source lookup. Repackaging an
existing video-enabled bundle preserves its video by default. ZIP the folder to
upload/share it; extract it before passing the folder to `--bundle`.

`annotate-trajectory --bundle` saves mutable work to `annotations.edited.json`,
outside the sealed inventory; `--annotations` selects another work file. The web
fit uses the current edited document. The standalone `fit-trajectory --bundle`
command uses the sealed annotation seed. Do not overwrite sealed resource files
with annotation edits or generated outputs.

Create it from an existing annotated recording:

```bash
conda activate py312_bk
embodichain fit-trajectory \
  --video /data/main.mp4 --joints /data/joints.npz \
  --robot-profile /data/robot_profile.json --scene /data/blender_scene/physical_export \
  --annotations /data/main.segments.json --all-segments \
  --qpos-semantics measured_state --time-offset 0 \
  --output /data/whole.fit.npz --package-output /data/portable-inputs \
  --package-video /data/main.mp4 \
  --run-task-output /data/whole-run-task
```

Move the folder anywhere, then fit and export using only its contents:

```bash
embodichain fit-trajectory --bundle /data/portable-inputs \
  --output /data/new.fit.npz --run-task-output /data/new-run-task
```

The manifest seals dependencies. Editing a sealed input requires rebuilding the bundle/inventory; checksum mismatches reject. New output paths must not replace source resources.

## Native run-task output

```bash
cd /data/new-run-task
embodichain run-task --gym_config task.yaml --skill-sequence skills.yaml --headless --viser
# Native EmbodiChain window, with a desktop display:
embodichain run-task --gym_config task.yaml --skill-sequence skills.yaml --no-headless
```

Open the printed Viser URL while execution runs. Native execution is automatic and exits at completion; use the annotation webpage's synchronized dual-robot viewer for timeline inspection. The native-window command omits `--viser`, which forces headless rendering.

`skills.yaml` retains the chronological skill identities and source spans. Pickup exports an object ID and the fitted object-relative TCP calibration; the live object pose grounds its grasp when invoked. Native approach, closure and lift are generated by AtomicSkillEngine from the current robot state. Other supported motion-only goals retain their explicit fitted axis/pose contracts. Control parts come from the actual participant selection and preserve custom names.

Only uncovered/explicit DemoSkill intervals retain `skills/*.execute_demo_skill.pt`, with exact original **DOF × n_frame** qpos, absolute timestamps and joint names. Native atomic spans have no trajectory PT. There is no full-episode `trajectory.pt`, `qpos.pt` or fitting NPZ in the runtime package; the separate offline NPZ retains all comparison arrays. Plain demonstration gaps do not pretend to meet the monitored ExecuteDemoSkill contact/effect contract.

At runtime, raw demo clocks are resampled to the configured control clock without changing the saved source. Quintic joint connectors begin/end at adjacent command postures and obey the profile's velocity and acceleration limits. Atomic skills replan from observed state and update only their bound parts; unbound joints retain previous commands, including full-close targets on loaded grippers. Source frame timing is provenance; native execution duration may change. Completion writes `execution.json` with measured endpoint tracking and object heights. These observations are separate from a physical task acceptance contract.

`--skill-sequence` also accepts `--expansion-profile`. Each exported runtime folder includes an editable `expansion.yaml`:

```bash
embodichain run-task --gym_config task.yaml --skill-sequence skills.yaml \
  --expansion-profile expansion.yaml --expansion-output-dir ./expanded \
  --headless --device cpu
```

The default profile runs one nominal episode and two spatial variants, using `joint_residual` and `via_points` on native motion phases. Demo-only sequences run one nominal episode. Native contact phases, grasp joints, unbound joints and phase boundary stencils remain protected. DemoSkill tensors and their resampled command paths are identical across variants; transition connectors can change to join the preceding native posture. The simulation's measured states can differ because physics still runs.

The profile declares `source.kind: skill_sequence`, `source_id: skills` and `source_revision: native-skill-sequence/v1`. `source.call_count` counts **all** chronological timeline spans; `trajectory_call_indices` uses those zero-based indices and defaults to native atomic spans only. Demo indices are excluded even if requested explicitly. Change `trajectory.variants_per_family` and `collection.target_episodes` together to collect more episodes. Recipe zero is nominal; subsequent recipes sample reproducible spatial variants using the profile and `--seed`. A rejected requested variant is retried, never silently replaced by nominal.

This host supports B=1, a fixed scene and spatial variants only. Scene, affordance, timing, visual and LeRobot expansion are rejected before simulation. It writes `expansion_manifest.json` and `recipe-*/attempt-*/{execution.json,trajectory.h5}` to a **new** output directory. Each HDF5 contains `commands/qpos` with T submitted active-joint commands and `states/qpos`, `states/qvel`, root and object poses with T+1 measured states, including the initial state. Joint names and the control period are stored as attributes. The reports identify path command ranges separately from connectors/holds, retain rejected attempts, and count command completion without claiming physical task success. `expansion.yaml` is outside the sealed physical-input inventory so sampling settings remain editable.

Scene defaults remain:

```yaml
skill_fitting:
  pick_up:
    grasp_world_z_offset_m: -0.01
    replay_interpolation_factor: 2
```

Omitted values are zero offset/factor one. The offset is applied before fitting and retained in object-relative grasp calibration; TCP calibration and mesh poses stay explicit. The interpolation factor retains its existing configuration spelling and becomes native duration scaling at an unchanged command period. Execution has a 200,000-command budget.

The profile's optional `simulation` carries explicit `root_props`, `joint_drive_props`, `attrs` and `link_attrs` into the generated robot configuration. Tuned stiffness/damping/effort and mass/inertia remain simulation calibration, not manufacturer specifications. Mesh collision decompositions are prepared at import/package time, supplied through `decomposition_fpath`, and loaded directly at runtime. They are independent of an external cache on the receiving machine.
