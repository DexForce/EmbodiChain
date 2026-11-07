# Real2Sim motion

`data_pipeline.real2sim` owns source units, explicit joint/frame/time conversion,
event maps, frozen reference prototypes and hash-checked JSON/NPZ bundles.

Normalization preflights output sample count and estimated numeric working
memory before allocating its grid or casting event indices. Preparation,
prototype readers and CLI accept caller-owned budgets; bundle metadata cannot
raise the reader's limits. These estimates exclude the already loaded source
and do not claim a process-RSS peak bound.
Source, normalized, DROID-evidence and pending-episode NPZ readers check each
NPY header's decoded shape against its byte budget and actual member payload
before NumPy materializes arrays; ZIP member sizes and content hashes alone
do not bound the declared allocation.

`embodichain real2sim prepare|validate` operates offline. qpos-only prototypes
can be prepared without TCP data; object-relative transfer needs TCP poses and
frozen object/target references. FK-derived poses are not independent measured
pose evidence. Source commands retain explicit alignment and are not silently
substituted for qpos.

`real2sim.droid` adapts selected original HDF5 streams with explicit observed
gripper calibration and manual phases. It rejects external or virtual storage;
raw state/command clocks and optional measured arm velocities remain distinct.
The qualified sink requires both the original recording and typed evidence for
DROID sources, verifies their imported crop, and archives them with the prototype.

`sim.motion.expansion.Real2SimSourceAdapter` exports fixed-scene qpos templates.
`atomic_skills.PrototypeMotionProvider` supplies qpos, EEF or explicitly
phase-bound hybrid nominal motion. Cartesian transfer is limited to B=1,
tabletop XY/yaw and compatible geometry; it checks IK/FK and sampled joint limits.
Hybrid qpos phases retain their recorded joints without a fresh IK solve;
a leading qpos phase requires its recorded entry joint state instead of a
Cartesian entry correction. Independent FK, inter-phase continuity and motion
checks still apply.
For a subsequent Pick, a qualified tabletop host can additionally verify that
the observed object rests on the target support and permit its changed height
and at most two degrees of measured tilt. Grasp orientation follows that
observation while lift displacement stays gravity-aligned. Place aligns the
source release anchor to the current semantic target and records the correction.
An optional planner collision check is reported separately from task acceptance.
The tabletop host requires source robot, object and target geometry hashes to
match live physical assets. Robot identity uses named joint/kinematic and
collision descriptions plus mesh contents, excluding assembly timestamps,
cache paths and visuals. `data_pipeline.real2sim._geometry` owns the versioned
digest and independently pinned legacy-reference aliases; aliases must never
be inferred from the live model or written into immutable source bundles.
Other legacy robot digests still require exact original URDF/mesh bytes.
Older source bundles without hashes remain valid
for explicit diagnostics but cannot use the qualified tabletop owner. In EEF
phases, numerical IK jitter can be smoothed only if the TCP error tube and
motion limits remain satisfied. Contact
phases, qpos-authority samples and phase boundary derivative stencils stay fixed.
When a Cartesian IK result itself exceeds its motion budget, a bounded
feasibility pass can adjust hold-phase IK samples inside the declared TCP
error tube before the geometric baseline is frozen; it records those passes.
Input source permissions can only narrow a skill's free-phase permissions.
Pick lift must be declared contact/hold and cannot authorize deformation.
Phasewise IK carries the preceding solution forward; stationary contact poses
repeat the same arm configuration instead of drifting in a redundant nullspace.

Configured source bindings may opt into `MotionRefinementCfg` with explicit
`eef_pose` authority. `motion.refinement.KinematicMotionRefiner` repairs a
scene-retargeted nominal path before expansion freezes its protected samples.
The source provider grounds the anchor from the skill goal, the current held
relation, or existing antipodal candidates with aperture and endpoint-IK
checks. Registered mesh services return runtime TCP poses; refinement uses their
explicit canonical/TCP mapping once and includes actual collision-geometry
identity in its evidence. Full hand/robot/scene collision and closing-contact
checks remain provider-owned; scalar finger envelopes are not inferred. Pick commits the corrected grasp through ordinary skill
effects; later transport and Place reuse that relation. Place refinement needs
the tabletop host's physical footprint and signed release-gap validator. A
geometry/authority/budget failure rejects the bound source without fallback.
The numeric fitter uses sparse fixed-grid pose QPs, bounded local rotation
coordinates, exact stationary close/release constraints and gravity-preserving
hold references; it is not a joint-space collision optimizer. Independent IK,
motion, collision and measured-effect checks remain mandatory. Source data,
timestamps and events are unchanged; neither calibration nor source uncertainty
is inferred. Original/refined TCP arrays and hashed baseline reports belong to
typed trace attachments, whose digests and correction budgets are checked before
receipt confirmation. Episode recipe identity includes the refinement profile;
realized floating-point pose hashes remain call-level evidence rather than
renaming the recipe after a small physical variation.
Explicit projection options can repair approach entry, gravity-aligned lift
and free-phase boundary stops inside the same correction budget before freezing
the baseline. Box-side grasp regions are grounded by
`toolkits.graspkit.box_region`; they do not replace gripper, IK or collision checks.
See `examples/expansion/trajectory/real2sim_carton/README.md` for real-source
prototype rebuilding, refinement comparisons and collection through the official
`carton_stocking` task.
Its local `_audit.py` independently verifies persisted real-source episode evidence.

`AtomicSkillEngine` accepts a call-scoped nominal source before native planning;
only PickUp, MoveHeldObject and Place opt in. Their skill-owned assembly retains
effects, tracking, segments and scene dependencies. Source plans reject retries
and replans. Pick requires the configured grasp command by the end of closure
and throughout lift. External Place checks the requested vertical retract
clearance and optional world-Z cap while retaining the source's lateral retreat.
External MoveHeldObject and Place accept one target pose and reject ordered
multi-waypoint goals; the native planners retain their waypoint support.
Programmatic and configured Task Program bindings use a compiled
program fingerprint and call occurrence; configured prototype content is
rechecked by digest. Prototype manifests and typed source evidence may use closed
`{asset, path}` file references resolved by `data._references`; registered bundle
verification precedes the ordinary content checks. These references cannot escape
the asset root. Catalog browsing does not initialize the assets. Parallel calls are unsupported for this source path.

`retime_with_time_map` and `rebuild_plan_from_mapped_trajectory` provide an
explicit B=1 timing route with protected samples, derivative stencils, limits,
scene dependency boundaries and timeout checks. Each authorized phase's total
duration is rounded up to the control grid; protected intervals remain fixed.
The tabletop profile may narrow retiming by skill and phase with
`retime_phase_allowlist`; a protected short phase then retains its original grid.
The configured combined Expansion Profile does not select these timing variants.
The separate `run-task` tabletop owner selects nominal,
same-grid and mapped-time recipes once per B=1 episode. Programmatic
`TaskProgramMappedRetimeFactory` and `TaskProgramSameGridExpansionFactory`
bind an episode-owned recipe to call occurrences, preserve boundary stencils,
reject out-of-limit proposals and require host collision evidence outside
explicit diagnostics. The optional motion-headroom policy removes nearly
saturated joints from joint-residual edits without changing the configured
amplitude for the remaining joints; the locked indices are recorded and checked
against the nominal plan. For task-space variants the same threshold requests
at most eight free-phase IK smoothing passes and keeps the best path only if
it stays within the TCP error tube and motion limits. Unauthorized, contact,
hold and boundary samples remain fixed by this optional headroom pass. The
tabletop guard uses physical URDF meshes and scene
boxes with per-phase contact and penetration limits, predicts a contact-limited
gripper for planned trajectories, and separately checks actual recorded robot
and object poses. These are sampled geometric checks, not continuous collision
or hardware qualification. Velocity and acceleration limits are explicit
profile inputs with recorded provenance rather than inferred simulator limits.
`motion.expansion.gripper_contact` selects independent prismatic or single-driver
mimic projection. Coupled closure advances every follower together until the
first declared finger contact; measured coupling tolerances never relax commands.
The same-grid factory's geometry family and initial
state stream IDs are supplied by the episode owner, not derived from a moving
robot state. The synchronous
B=1 `Real2SimLeRobotEpisodeSink` saves source and terminal-observation artifacts,
requires a typed action-to-plan trace with source phase events and call-level
lineage, closes buffered writers and checks the existing journal before a receipt;
its Gym helper suppresses the normal reset-time duplicate and discards the
buffer after confirmation. It can reserve measured coverage with an
`ExpansionSession` and applies only the sink's returned receipt; persistence
retry uses the same stored episode rather than another rollout. The tabletop
run-task owner stages an actual T+1 payload before submission and writes a
verified identity-matching receipt before reset. A failed write or receipt
publication retains that payload for write-only repair. Its measured object
poses, fixed target pose and timestamps are a separate verified NPZ sidecar.
The B=1 scene integration may
opt into `trace_actions` to retain acknowledged action-to-plan references;
hosts must still assemble the typed trace and qualified validation before sink
submission. Ordinary `run-task` recording still rejects source-bound collection
without an installed Real2Sim owner. A deployment selects that owner with
`real2sim: {profile: profile.yaml}`; the profile resolves from the deployment
directory and `--real2sim-profile` overrides it. The launcher keeps this binding
out of `EnvCfg` and gives its recorder the `--dataset-dir` destination.
The tabletop profile requires exactly one
dynamic box, one static box and a fixed plane. All sequential Pick,
MoveHeldObject and Place call occurrences require pinned sources; repeats use
distinct call indices even when they share a prototype. Per-Pick lift,
continuous held intervals and each Place's target, support footprint and
stability are measured separately. Other scene topologies and unbound calls
fail closed. Gym still owns reset and stepping.
Optional directed yaw acceptance compares observed object +X to the requested
semantic object pose, not approximate FK release; solver error consumes the same
task tolerance. `real2sim.replay.export_real2sim_replay_manifest` exports existing
identities read-only after joining every episode to its complete recording
journal and confirmed receipt. Observed geometry is verified whenever present
and is mandatory for episodes declaring observed-collision checks; trace and
actual episode evidence remain mandatory for every replayed commit.
Failed attempts retain available actual T+1
partial rollouts without becoming successful episodes.

Focused coverage: `tests/data_pipeline/test_real2sim_normalize.py`,
`tests/sim/motion/expansion/test_mapped_timing.py`, and matching Atomic Skill
and Task Program source-binding tests. Synthetic fixtures establish software
contracts, not real-data physical qualification.

## Batched offline collection

The original qualified tabletop owner retains its B=1/no-sensor contract.
[Expansion Engine](../expansion-engine/expansion-engine.md) provides a separate
World owner that binds each source runtime to one physical row and merges actual
commands into one batched step. It reuses the same source preparation and measured
qualification through `Real2SimTabletopSource` and explicit measured
`Real2SimTabletopEpisodeEvidence`; the original owner adapts into the same interface.
It adds bounded native camera media, frozen artifacts
and LeRobot export. Source permissions and contact protections are unchanged.
