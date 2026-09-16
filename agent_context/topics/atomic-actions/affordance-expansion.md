# Geometry-constrained affordance expansion

The [user guide](../../../docs/source/overview/sim/atomic_actions/affordance_expansion.md)
contains runnable commands, the detailed pose-to-trajectory pipeline, sampling
algorithm, and drawer tuning/workaround. This page records implementation
ownership and integration contracts for code navigation.

## Ownership and entry points

`atomic_actions/affordance_sampling.py` owns `AffordanceExpansionCfg`, immutable
`AffordanceSamplingContext`, and `AffordancePoseCandidates`. Geometry remains in
`affordance.py`; grasp generators remain standalone engine-injected services.
Atomic Skills own feasibility and trajectory planning. Gym owns environment
allocation, execution, episode acceptance and recorder commit boundaries.

For configured Task Programs, `run-env --n_affordance_expand N` overrides the
runtime environment count to **N total rows**, including one nominal branch.
The YAML equivalent is `env.expert_trajectory.affordance_expansion.count`.
Omitting the flag or using count one preserves ordinary execution. An explicit
`--num_envs` must match enabled expansion. The CLI rejects non-Task-Program
expansion rather than silently ignoring it. Direct Atomic Skill callers can
pass `AffordanceSamplingContext` through `PlanningContext`,
`AtomicActionEngine.initial_context()` or `SimulationExecutionAdapter`.

The simulation integration reads the effective environment seed, demo episode
index and explicit attempt ID when observing. Context projection and semantic
executor normalization preserve this immutable value. Stream keys also include
the invocation identity and stable environment IDs, so observing again,
reordering rows or changing the active subset does not advance random state.
This guarantees selection reproducibility for the same candidate pool; it does
not guarantee bitwise simulator determinism or regenerate a grasp annotation
cache independently of its provider.

## Candidate and frame contracts

Candidate poses use `(B, K, 4, 4)`, costs and validity use `(B, K)`; the selected
pose is `(B, 4, 4)`. B is the actual environment count, never B times K.
The candidate dimension ends at `select()`. The selected pose anchors each
action's phase waypoints; motion generation and hand interpolation then build
`(B, T, D)` full-joint positions, wrapped in `TimedTrajectory` with the same
`env_ids` and control interval. `affordance_sampling.py` itself does not run IK
or construct trajectories.
Variable-length and empty grasp lists have explicit invalid padding. Nonfinite
costs/poses cannot be selected. Candidates are quality ranked, then diversified
by position and rotation in the object frame when supplied. Near duplicates
are suppressed within the best `max(32, 4 * count)` candidates before branch
allocation. Each row uses its own candidate pool, with group `env_id // count`
and branch `env_id % count`; this does not promise globally distinct outputs
when pools differ. A branch reuses a valid candidate only after unique choices
in its shortlist are exhausted; diagnostics expose candidate IDs, valid
counts, unique counts and reuse. No valid candidate is a row-local planning
failure, never an executable identity placeholder.

PickUp preserves approach, pre-grasp, grasp, lift and supplied downstream IK
filters, as well as TCP calibration and existing symmetric-roll selection.
Each accepted grasp produces its own `HeldObjectState.object_to_eef`, used by
subsequent transport and placement. Explicit grasp targets and fixed
object-to-EEF relations remain exact. AxisAlign samples among its existing
axis-compatible preferred candidates; the final alignment constraint remains
unchanged. Door and Slide sample their handle candidates before ordinary
trajectory feasibility checks; they do not promise every candidate succeeds.

## Allowed variations

- Press: rotate local x/y about contact z. Contact point, approach/press axis and
  press depth stay fixed, and all phases use the selected contact frame.
- OpenDoor: optional `OpenDoorGoal.open_fraction_range` declares the accepted
  absolute interval. Sampling intersects it with forward opening from the
  observed hinge position. The nominal fraction must be inside that interval.
  Missing range preserves the exact goal. Handle/EEF waypoints follow the hinge
  arc and retain their relative transform.
- Twist: optional `TwistOptions.twist_angle_range` declares a signed interval
  containing the nominal angle without reversing direction. The grasp position
  and axis origin stay fixed. `TwistAffordance.grasp_roll_range` permits initial
  roll only when the asset declares that contact symmetry.
- Slide: optional `SlideOptions.translation_distance_range` declares positive
  task-accepted travel. Configured slide action options decode the same field.
- Randomized Twist/Slide travel requires joint limits, an unambiguous live joint
  observation and the axis-to-joint sign. Geometry resolution derives
  `joint_axis_sign`; manually authored geometry may supply it explicitly.
  Sampling intersects task bounds and joint limits per row. Empty intersections
  fail only that row; ambiguous or missing metadata fails configuration/planning.
- InteractionPoints: select existing target-local points of the requested type;
  normals are required and define inward contact z. No spatial interpolation.
- Assemble: `symmetry_transforms` declares equivalent object-local rotations.
  Identity is retained; translations and improper rotations are rejected.
  Without symmetry metadata the assembly relation stays exact.
- Base Affordance: preserve the supplied exact pose when no geometric freedom
  is declared.

Geometric and IK validity are not measured task success. The packaged repeated
pick/place deployment uses projected effect assurance and an open-loop IK
policy, so its completion metadata must not be described as physical success
verification.

## Collection and validation

Expansion collection submits only rows with successful completion and recorded
frames. `--max_episodes` counts actual committed rows across batches; a partial
last batch selects at most the remaining quota. A batch with no accepted rows
uses the bounded existing attempt budget. Successful subsets commit once, and
the other rows are discarded during the same full reset. Dataset, trajectory
and camera recorders honor the same explicit `commit_env_ids` subset; see
[data persistence](../data-pipeline/data-pipeline.md).

Focused tests: `tests/sim/atomic_actions/test_affordance_sampling.py`,
`test_actions.py`, `tests/lab/scripts/test_run_env.py`,
`tests/gym/envs/task_program/test_simulation_environment.py`, and
`tests/gym/envs/managers/test_dataset_manager.py`.
