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

The only host integration is direct simulation. The PickUp tutorial maps
`--affordance_branches` one-to-one to simulation rows and injects the context
through `AtomicActionEngine.initial_context()`.

Do not route this feature through Task Program, the Gym bridge, environment
configuration, episode retry/commit, or dataset recording until those hosts
define an explicit expansion policy.

## Change and validation sites

- Sampling contracts and selection: `atomic_actions/affordance_sampling.py`.
- Geometric samplers: `atomic_actions/affordance.py`.
- Feasibility and diagnostics: matching file under `atomic_actions/primitives/`.
- Direct host example: `scripts/tutorials/atomic_action/pickup.py`.
- Tests: `tests/sim/atomic_actions/test_affordance_sampling.py`,
  `test_affordance.py`, `test_actions.py`, and `test_tutorial_utils.py`.
