# Trajectory annotation, fitting and native execution

Video-selected reconstruction enters through `embodichain fit-trajectory`
(`toolkits/skill_fitting/cli.py`) or the annotation website's whole-trajectory
fitting panel (`toolkits/trajectory_annotator/_fitting.py`). Both hosts reuse
`toolkits/skill_fitting/_pipeline.py`; the web host retains one background job
and result, keeps decoding on its existing owner, and invalidates results after
annotation revisions change. Annotation choices cover all built-in skills without
importing simulation; only supported reconstruction types enter fitting. A saved
optional core interval separates loose annotation boundaries from reconstruction,
preserving surrounding motion. `_episode.py` fits each annotation independently
from the immutable recording, combines only its core, and partitions every
remaining source sample into DemoSkill spans. Unsupported authored skills fail
explicitly. The CLI selects this path with `--all-segments --annotations ...`;
its existing single-segment mode remains available.
Its bounded joint charts follow video time. `_viewer.py` owns a local Viser worker
with independent original-material and translucent-blue URDF models; playback
samples all recorded joints on the full clock, including explicit mimic columns.
Model bases use the profile calibration relative to each solver's URDF root,
without TCP offsets. One latest-time slot bounds playback work; replacing the fit
or closing the host stops its viewer. The downloadable NPZ retains full samples.
`_export.py` packages accepted episodes for `run-task --skill-sequence skills.yaml`.
`_native_export.py` writes object-bound Pickup goals, actual participant bindings,
native options/policy and a sealed local resource inventory. `_execution.py` is a
direct AtomicSkillEngine caller: each native invocation plans from measured robot
state and current object pose, then submits ControllerAction commands through Gym.
Only explicitly bound motion/grasp parts take new native-plan commands. Unbound
joints retain the previous commanded targets, not measured positions: a loaded
opposite gripper must keep its closing preload. Demo spans retain full source
command ownership, including deliberate releases. Grasp command/TCP calibration
belongs to the input robot profile; export copies solver TCPs into the Gym robot.
It is not a Task Program lowerer. Only DemoSkill gaps carry original DOF-by-frame
PT matrices and source timestamps; there is no atomic/full-episode trajectory PT.
The host resamples demo clocks and inserts derivative-bounded quintic connections
between adjacent postures. Its execution report separates command completion from
measured object heights and leaves physical success unqualified.
`_sequence_expansion.py` accepts a matching Combined Expansion Profile with
`source.kind=skill_sequence`. This B=1 fixed-scene host applies shared
`atomic_skills/_plan_expansion.py` transforms only to native spans; DemoSkill
source tensors and resampled command paths are excluded. The timeline indices
include every span. Native free-phase permissions, bound motion joints and
protected boundary stencils constrain perturbations. Each recipe resets physics,
and rejected proposals retry without nominal substitution. `_sequence_trace.py`
streams T submitted commands and T+1 measured robot/object states into bounded
HDF5 chunks. These artifacts record command completion with unknown physical
success, not verified Expansion Engine dataset receipts. Editable `expansion.yaml`
is separate from the sealed physical inventory. The
[fitting example](../../../examples/toolkits/skill_fitting/README.md) owns commands
and profile settings.
`_bundle.py` owns `scene-skill-fitting-inputs/v1`: local scene meshes/colliders and
explicit poses, calibrated robot model/profile, original joints, annotations and
frame presentation clock. `fit-trajectory --bundle` requires no image pixels or
original exporter directory; `--package-output` creates portable input resources.
Optional `frame_source` adds a content-bound local video for
`annotate-trajectory --bundle`. This entry resolves all fitting defaults from the
same inventory and persists mutable `annotations.edited.json` separately from
the sealed annotation seed. `--package-video` creates video-enabled resources;
the web fit and exported native runtime need no original recording directory.
A v2 bundle additionally pins a host-owned execution context and declares the
qualified design reference as its fitting trajectory. CLI fitting accepts saved
annotation edits for the same frame source via `--annotations`.
`export_task_program` uses that context to bind a scene-bound Pickup to one
named invocation. It separates the fitted core from the surrounding annotation
samples. Each retained interval needs an explicit named DemoSkill and measured
boundary contract. A retained slice on the same post-step clock can be built
from the pinned source asset with its original commands, shifted phases, events
and scoped evidence. The output links recorded event ticks and asset hashes to
compiled calls. The Expansion Engine owns physical qualification and writes a
new receipt before dataset conversion.
`_collision.py` prepares content-addressed scene convex parts; web import and
packaging supply precomputed files for decomposition-free runtime startup.
Scene-owned Z compensation precedes fitting; the existing pickup interpolation
factor becomes native duration scaling at an unchanged control cadence.
`RobotProfile.simulation` carries explicit physics overlays without changing
kinematic calibration or custom control names. See the
[web example](../../../examples/toolkits/trajectory_annotation/README.md).
`toolkits/skill_fitting/_scene.py` grounds reconstructed local meshes in explicit
world poses and obtains calibrated, finite-surface-qualified antipodal grasps.
Single-arm scene fitting ranks valid TCP poses against the recorded grasp, then
`atomic_skills/fitting.py` calls the ordinary native planner. PickUp uses an
object-bound `GraspGoal`; Slide/Twist retain their motion/axis contracts with the
selected grasp anchor. Parallel-hand alternatives apply an exact half turn in
local TCP Z; contact-frame calibration is conjugated into the physical surface
query so tilted/offset virtual TCPs do not change the rotation axis. Both variants
must pass target-surface checks before ranking. Fitting uses a wider configurable
approach angle and disables the inferred support-plane filter by default;
these host policies do not change native affordance defaults or certify table
clearance. PickUp approaches along the selected grasp's Z. Source motion supplies approach, phase clocks and motion
parameters rather than pointwise acceptance targets. Quintic free-phase bridges
preserve the outer derivative stencils and the contact core. Pickup may rebalance
its free-phase clock inside the unchanged outer interval; explicit phase hints
remain fixed and observed phases remain in provenance. Qualification covers
every changed joint/derivative stencil, while pre-existing source-limit violations
remain separately visible. Rejected results retain candidates without replacing
the original reference. The native fitter does not execute physics; the host
exports the native invocation deployment described above.

`_participants.py` pairs configured motion/hand groups through URDF ancestry and
selects a unique closing hand; ambiguous evidence requires an explicit choice.
Names do not establish capabilities. A profile must supply explicit runtime TCP
calibration and actual finger collision assets; automatic scene geometry supports
opposing prismatic fingers, including two URDF followers of one aperture driver.
Without a scene, the effect-free world-TCP motion
fitter remains available for PickUp, Slide, Twist and explicitly bound HandOver.
Scene HandOver needs time-resolved object/attachment bindings and is not inferred
from an initial object pose. Both paths retain explicit source identity, inclusive
frame conversion, joint semantics and clock offsets. Automatic RGB contact
recognition and physical-success certification are absent. The
[runnable guide](../../../examples/toolkits/skill_fitting/README.md) owns scene
formats, calibration conventions, phase partitions and result arrays.
