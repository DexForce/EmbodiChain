# Affordance sampling boundary

Read this when changing geometric pose variation in Atomic Skills.
[Topic overview](atomic-actions.md).

## Ownership

`Affordance` owns sampling inside legal geometry: antipodal grasp candidates,
interaction points, assembly symmetry, Press contact-frame roll, and declared
Twist grasp-roll symmetry. The public entry point is
`Affordance.sample_candidates()`. Do not add a competing candidate `select()`
method or a strategy hierarchy.

`AffordanceSamplingContext` is immutable caller-owned identity: `count`, `seed`,
`episode_id`, and `attempt_id`. It flows through `PlanningContext`; Affordances
never retain it. `AffordancePoseCandidates` retains poses, costs, and an
explicit validity mask. `AffordanceSample` returns row-local success, poses,
and owned metadata.

A caller can supply keyed `preferred_local_poses` from a previously verified
episode. A non-nominal pose selection may match one of the current valid,
geometrically distinct alternatives in the same reference frame. The ordinary
rank applies when no equivalent alternative exists. The sample records the
current candidate ID and whether it matched. The caller owns selection history.

Actions own IK, reachability, complete trajectory feasibility, and diagnostics.
They intersect `AffordanceSample.success` with their feasibility result and put
the sampling record under `diagnostics.metadata["affordance_sample"]`.
Slide distance, Twist angle, and OpenDoor opening fraction remain exact
Action/Goal inputs rather than Affordance variation.

An Atomic Skill using the Antipodal generator requires a GraspValidator cache
and two configured contact pads. The worker validates proposals on the scaled
target mesh in an isolated PhysX process and caches object-local TCPs and
opening widths by hand URDF, mesh and validation settings. An explicit refresh
regenerates each cache key once per service instance. The standalone PickUp example
prepares its live cube, hand commands and final TCP before compilation; the planner
only reads the matching labels. For a bound hand the
validator applies the mounted joint drives and link contact properties under
their native URDF names.
Changing a finger gain, effort limit or pad friction selects a new cache entry.
PickUp, AxisAlign, HandOver, Slide,
OpenDoor and CoordinatedPickment rank eligible labels by live closing-axis
and contact-center cost, then TCP +Z alignment, and check each skill's IK
waypoints in groups of 50. HandOver and coordinated grasps retain their
per-arm regions. HandOver can additionally limit the source and receiving TCP
approach angles independently in degrees. The live angle limits reuse the same
physical labels. Handle validation tests the isolated link mesh; Slide and
OpenDoor check motion against the live joint. Suspended handles and receiving
hands skip the inferred tabletop. Explicit mirror, upright and retraction
adjustments and PickUp's `grasp_frame_to_eef` are applied before physical
validation and included in the cache key. Branch selection and preferred-pose
matching operate on the ranked labels that pass waypoint IK.

## Current host boundary

The only host integration is direct simulation. The PickUp, AxisAlign,
HandOver, OpenDoor, Press, Slide, and Twist examples map
`--affordance_branches` one-to-one to simulation rows and inject the context
through the shared host helpers in
`examples/sim/atomic_skill/support.py`. HandOver samples pickup
and receiving grasps in separate named streams while preserving opposite object
ends and per-row arm assignment; its diagnostics retain the selected assignment's
candidate metadata and control parts. When the hands use different providers,
a provider-specific diagnostic is null on rows where it does not apply.
Press samples local contact-frame roll;
the Twist example explicitly declares `grasp_roll_range=(-pi, pi)` for parallel
branches. Press, Slide, and OpenDoor examples use `ik_interp` for their exact
Cartesian contact/path samples.

The seed and attempt identity control Affordance selection, not upstream grasp
generation. Antipodal approach-direction perturbations still use PyTorch's
global RNG and each row has its own candidate pool. Candidate IDs and reuse
flags are row-local; they do not certify cross-row uniqueness. One branch keeps
sampling disabled. See the human-facing
[example matrix](../../../docs/source/overview/sim/atomic_skills/affordance_sampling.md)
for supported geometric freedom and CLI examples.

Do not route this feature through Task Program, the Gym bridge, environment
configuration, episode retry/commit, or dataset recording until those hosts
define an explicit expansion policy.

## Change and validation sites

- Sampling contracts and selection: `affordance/sampling.py`.
- Geometric samplers: `affordance/interactions.py`.
- Feasibility and diagnostics: matching file under `atomic_skills/primitives/`.
- Direct host examples: `examples/sim/atomic_skill/` and its shared
  `support.py` sampling helpers.
- Tests: `tests/sim/atomic_skills/test_affordance_sampling.py`,
  `test_affordance.py`, `test_actions.py`, `test_core.py`, and `test_engine.py`.
