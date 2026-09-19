# Affordance sampling boundary

Read this when changing geometric pose variation in Atomic Actions.
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

Actions own IK, reachability, complete trajectory feasibility, and diagnostics.
They intersect `AffordanceSample.success` with their feasibility result and put
the sampling record under `diagnostics.metadata["affordance_sample"]`.
Slide distance, Twist angle, and OpenDoor opening fraction remain exact
Action/Goal inputs rather than Affordance variation.

## Current host boundary

The only host integration is direct simulation. The PickUp, AxisAlign,
HandOver, OpenDoor, Press, Slide, and Twist tutorials map
`--affordance_branches` one-to-one to simulation rows and inject the context
through the shared host helpers in
`scripts/tutorials/atomic_action/tutorial_utils.py`. HandOver samples pickup
and receiving grasps in separate named streams while preserving opposite object
ends and per-row arm assignment; its diagnostics retain the selected assignment's
candidate metadata and control parts. Press samples local contact-frame roll;
the Twist tutorial explicitly declares `grasp_roll_range=(-pi, pi)` for parallel
branches. Press, Slide, and OpenDoor tutorials use `ik_interp` for their exact
Cartesian contact/path samples.

The seed and attempt identity control Affordance selection, not upstream grasp
generation. Antipodal approach-direction perturbations still use PyTorch's
global RNG and each row has its own candidate pool. Candidate IDs and reuse
flags are row-local; they do not certify cross-row uniqueness. One branch keeps
sampling disabled. See the human-facing
[tutorial matrix](../../../docs/source/overview/sim/atomic_actions/affordance_sampling.md)
for supported geometric freedom and CLI examples.

Do not route this feature through Task Program, the Gym bridge, environment
configuration, episode retry/commit, or dataset recording until those hosts
define an explicit expansion policy.

## Change and validation sites

- Sampling contracts and selection: `atomic_actions/affordance_sampling.py`.
- Geometric samplers: `atomic_actions/affordance.py`.
- Feasibility and diagnostics: matching file under `atomic_actions/primitives/`.
- Direct host examples: `scripts/tutorials/atomic_action/` and its shared
  `tutorial_utils.py` sampling helpers.
- Tests: `tests/sim/atomic_actions/test_affordance_sampling.py`,
  `test_affordance.py`, `test_actions.py`, `test_core.py`, and `test_engine.py`.
