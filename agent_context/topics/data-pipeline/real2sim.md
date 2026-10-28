# Real2Sim motion

`data_pipeline.real2sim` owns source units, explicit joint/frame/time conversion,
event maps, frozen reference prototypes and hash-checked JSON/NPZ bundles.
`embodichain real2sim prepare|validate` operates offline. qpos-only prototypes
can be prepared without TCP data; object-relative transfer needs TCP poses and
frozen object/target references. FK-derived poses are not independent measured
pose evidence. Source commands retain explicit alignment and are not silently
substituted for qpos.

`sim.motion.expansion.Real2SimSourceAdapter` exports fixed-scene qpos templates.
`atomic_actions.PrototypeMotionProvider` supplies qpos, EEF or explicitly
phase-bound hybrid nominal motion. Cartesian transfer is limited to B=1,
tabletop XY/yaw and compatible geometry; it checks IK/FK and sampled joint limits.
An optional planner collision check is reported separately from task acceptance.
The tabletop host requires source robot, object and target geometry hashes to
match live physical assets. Older source bundles without hashes remain valid
for explicit diagnostics but cannot use the qualified tabletop owner. In EEF
phases, numerical IK jitter can be smoothed only if the TCP error tube and
motion limits remain satisfied. Contact
phases, qpos-authority samples and phase boundary derivative stencils stay fixed.
Input source permissions can only narrow a skill's free-phase permissions.
Pick lift must be declared contact/hold and cannot authorize deformation.

`AtomicActionEngine` accepts a call-scoped nominal source before native planning;
only PickUp, MoveHeldObject and Place opt in. Their skill-owned assembly retains
effects, tracking, segments and scene dependencies. Source plans reject retries
and replans. Programmatic and configured Task Program bindings use a compiled
program fingerprint and call occurrence; configured prototype content is
rechecked by digest. Parallel calls are unsupported for this source path.

`retime_with_time_map` and `rebuild_plan_from_mapped_trajectory` provide an
explicit B=1 timing route with protected samples, derivative stencils, limits,
scene dependency boundaries and timeout checks. Each authorized phase's total
duration is rounded up to the control grid; protected intervals remain fixed.
The configured combined Expansion Profile does not select these timing variants.
The separate `run-task --real2sim-profile` tabletop owner selects nominal,
same-grid and mapped-time recipes once per B=1 episode. Programmatic
`TaskProgramMappedRetimeFactory` and `TaskProgramSameGridExpansionFactory`
bind an episode-owned recipe to call occurrences, preserve boundary stencils,
reject out-of-limit proposals and require host collision evidence outside
explicit diagnostics. The optional motion-headroom policy removes nearly
saturated joints from joint-residual edits without changing the configured
amplitude for the remaining joints; the locked indices are recorded and checked
against the nominal plan. The tabletop guard uses physical URDF meshes and scene
boxes with per-phase contact and penetration limits, predicts a contact-limited
gripper for planned trajectories, and separately checks actual recorded robot
and object poses. These are sampled geometric checks, not continuous collision
or hardware qualification. Velocity and acceleration limits are explicit
profile inputs with recorded provenance rather than inferred simulator limits.
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
without an installed Real2Sim owner. The tabletop profile requires exactly one
dynamic box, one static box, a fixed plane and three configured call occurrences;
other scene topologies fail closed. Gym still owns reset and stepping.

Focused coverage: `tests/data_pipeline/test_real2sim_normalize.py`,
`tests/sim/motion/expansion/test_mapped_timing.py`, and matching Atomic Action
and Task Program source-binding tests. Synthetic fixtures establish software
contracts, not real-data physical qualification.
