(affordance-expansion)=

# Parallel affordance expansion

Affordance expansion runs several geometrically valid alternatives of the same
Task Program in parallel. `--n_affordance_expand N` selects **N total simulation
environments**, including the nominal branch. Alternatives come from grasp
candidates or explicitly allowed geometric freedoms, such as rolling a press
frame around its contact axis. Each environment receives one selected pose and
one corresponding trajectory for each action.

## Run the packaged tasks

Run these commands from the repository root in the `embodichain2` environment:

```bash
conda activate embodichain2

embodichain run-env \
  --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --device cuda --headless --seed 42 \
  --max_episodes 10 --n_affordance_expand 9

embodichain run-env \
  --gym_config embodichain_tasks/configs/tasks/manipulation/open_drawer/task.ur5.yaml \
  --device cuda --headless --seed 42 \
  --max_episodes 180 --n_affordance_expand 9
```

To save joint trajectories, append
`--record_trajectory --trajectory_save_dir ./outputs/affordance_trajectories`.
The equivalent expansion setting in the environment configuration is:

```yaml
env:
  expert_trajectory:
    affordance_expansion:
      count: 9
```

The default count is one, which leaves expansion disabled. Enabled expansion
sets the runtime environment count; an explicitly supplied `--num_envs` must
match it. This CLI collection path requires a Task Program deployment. Direct
Atomic Skill callers can instead supply `AffordanceSamplingContext` through
`AtomicActionEngine.initial_context()`, `PlanningContext`, or
`SimulationExecutionAdapter`.

`--max_episodes` counts accepted per-environment episodes across all batches.
With nine environments and all rows accepted, 20 episodes require three batches
(9 + 9 + 2); 180 episodes require twenty batches. The final batch still executes
the parallel environments, but commits only the remaining quota. Failed rows
can increase the number of batches, and a batch with no accepted rows uses the
existing bounded attempt budget.

## From candidate poses to trajectories

`embodichain/lab/sim/atomic_actions/affordance_sampling.py` owns candidate
packing, reproducible selection, and bounded scalar sampling. Affordance
geometry remains in `affordance.py`; the action planners own IK and trajectory
generation; Gym owns simulation stepping and collection.

```text
Observed object/link poses + mesh + action constraints
                         |
          get_valid_grasp_poses() for each environment
                         |
      AffordancePoseCandidates: poses (B, K, 4, 4)
                 costs/validity (B, K)
                         |
       PickUp: calibrate and filter candidate IK variants
                         |
       select(): one pose per environment, (B, 4, 4)
                         |
       Action-specific approach/contact/motion waypoints
                         |
          MotionGenerator / IK + hand interpolation
                         |
       TimedTrajectory: joint positions (B, T, D)
                    + env_ids + control_dt
                         |
       Execution runner / Gym bridge -> parallel env.step()
                         |
             Accept and commit selected episode rows
```

For example, nine environments with fifty candidate grasps each produce a
`(9, 50, 4, 4)` candidate tensor. Selection reduces this to nine poses
`(9, 4, 4)`. An action then produces a joint trajectory `(9, T, D)`, where `T`
is its number of control samples and `D` is the robot's full joint count. The
candidate dimension is an internal search dimension; it does not allocate 450
environments or create a Cartesian product of choices across successive actions.

### 1. Generate and pack candidates

`AntipodalAffordance.get_grasp_candidates()` calls the engine-injected grasp
generator's `get_valid_grasp_poses()` with the target mesh, current object poses,
approach direction, and any requested object-region restriction. The generator
returns a separate list of poses and costs for each environment.

`AffordancePoseCandidates.from_rows()` pads those variable-length lists to the
largest candidate count. Real finite candidates are marked valid; padding and
nonfinite candidates remain invalid. An empty row receives a masked identity
placeholder so tensor shapes remain usable. Selection returns `success=False`
and candidate ID `-1` for that row; the placeholder cannot make it a successful
plan.

### 2. Apply action-specific feasibility checks

`PickUp._resolve_grasp_pose()` checks candidates before sampling. For each
candidate, `_select_feasible_grasp_variants()` evaluates its two symmetric roll
variants, applies the configured upright adjustment and grasp-frame-to-TCP
calibration, and checks approach alignment plus pre-grasp, grasp, and lift IK.
It also checks any supplied `downstream_object_target_poses`, starting from the
corresponding lifted joint state. The existing orientation preference chooses
one feasible roll variant per candidate; its IK mask is combined with candidate
validity before calling `select()`.

`Slide` and `OpenDoor` select from the grasp generator's candidates first and
then perform their ordinary motion/IK checks. They do not try every alternative
automatically if the selected pose fails trajectory planning. A geometrically
valid candidate is therefore not a guarantee of a feasible full trajectory.
Explicit PickUp grasp targets and `fixed_object_to_eef` retain their exact poses
and bypass candidate expansion.

### 3. Select a diverse candidate for each branch

With expansion enabled, `AffordancePoseCandidates.select()` performs these steps
independently for each environment row:

1. Reject masked or nonfinite entries and sort the remainder by ascending cost.
2. Keep at most `max(32, 4 * count)` of the best candidates for diversification
   (36 when `count=9`). The nominal choice is the lowest-cost eligible candidate.
3. If an object reference pose is supplied, transform the shortlist into that
   object's local frame for diversity comparisons. Returned poses remain in
   their original frame.
4. Build a greedy diversity order starting with the nominal candidate. Distance
   combines translation divided by the shortlist's spatial extent (at least
   0.01 m) and the Frobenius distance between rotation matrices. A seeded factor
   in `[0.75, 1)` weights each candidate's distance to its nearest selected pose.
   Candidates within both 0.0001 m in translation and 0.001 in rotation-matrix
   distance are treated as duplicates.
5. Assign branch `env_id % count` to that position in the order. If there are
   fewer distinct candidates than branches, wrap around and report reuse.

Grouping uses `env_id // count`, so a reordered or partial batch retains its
branch identities. The diversity order is computed from each row's own pool;
different environment geometry or feasibility masks can produce different
orders. Expansion does not promise that all N outputs are distinct. It also
does not fabricate arbitrary pose noise when only one legal candidate exists.

Diagnostics expose `candidate_ids`, `valid_candidate_counts`,
`unique_candidate_counts`, and `reused`. Under expansion, unique counts refer to
the bounded shortlist, while valid counts cover the entire eligible pool.
Disabled expansion selects the minimum-cost eligible pose.

### 4. Build a coherent action trajectory

The selected pose anchors every phase of its action. `PickUp` derives pre-grasp
and lift poses from that same grasp, asks `MotionGenerator` for arm motion,
resamples the approach and lift segments, and inserts hand-close interpolation
and any configured settling frames. Uncontrolled joints retain their previous
positions. Candidate validity and trajectory success are combined per row.

The resulting full-joint tensor is wrapped by
`TimedTrajectory.from_uniform_step()` with the matching `context.env_ids` and
`context.require_control_dt()`. The action plan carries segment boundaries,
success masks, and sampling diagnostics. The runner and Gym bridge dispatch
the corresponding trajectory rows on the environment's control grid.

For PickUp, the selected grasp also defines each row's expected
`HeldObjectState.object_to_eef = inverse(object_pose) @ grasp_pose`. Subsequent
transport and placement use that row's transform, preserving the relationship
between the sampled grasp and later object motion.

`Slide` constructs approach, reach, pull/push, and optional return waypoints on
its translation axis, with hand-close/open segments inserted at contact.
`OpenDoor` constructs the opening arc around the hinge and carries the same
handle-to-EEF transform along it. These paths preserve the task geometry across
the whole action.

## Allowed variation and random streams

| Affordance/action | Variation with expansion enabled | Constraint retained |
|---|---|---|
| Antipodal grasps / PickUp | Select among generated grasp candidates | Mesh contact geometry, approach restrictions, and PickUp feasibility filters |
| AxisAlign | Select among existing preferred candidates | Requested axis alignment |
| Press | Roll the contact frame about local z in `[-pi, pi]` | Contact point, z direction, press axis, and depth |
| OpenDoor | Select handle grasps; optionally sample `OpenDoorGoal.open_fraction_range` | Hinge arc, legal limits, and forward opening from the observed joint state |
| Slide | Select handle grasps; optionally sample `SlideOptions.translation_distance_range` | Translation axis, direction, and legal joint travel |
| Twist | Optional `TwistOptions.twist_angle_range` and affordance `grasp_roll_range` | Rotation axis/origin; initial grasp roll requires declared contact symmetry |
| InteractionPoints | Select existing points of the requested type | Target-local points and their inward contact normals |
| Assemble through Place | Select declared `symmetry_transforms` | Equivalent proper local rotations with zero translation; identity retained |
| Base Affordance | Return the supplied pose | Exact pose when no freedom is declared |

Travel/angle ranges are task-accepted bounds and must contain the nominal
value. Absent ranges preserve the exact goal. Randomized Slide/Twist travel also
requires an unambiguous live joint observation, limits, and the axis-to-joint
sign; the planner intersects the task range with reachable joint travel per
environment. An empty intersection fails that row.

For example, to explicitly allow 0.16--0.20 m drawer travel, add the optional
range beside the existing distance in
`embodichain_tasks/configs/tasks/manipulation/open_drawer/task_program/integration.yaml`:

```yaml
profile:
  action_options:
    simulation.articulation_link_slide:
      kind: slide
      direction: pull
      hand_interp_steps: 18
      approach_distance: 0.1
      translation_distance: 0.18
      translation_distance_range: [0.16, 0.20]
```

This is an opt-in example; the packaged drawer configuration keeps travel fixed
at 0.18 m.

`AffordanceSamplingContext` carries `count`, `seed`, `episode_id`, and
`attempt_id`. A SHA-256-derived seed combines `seed`, `episode_id`, `attempt_id`,
the invocation/parameter key, and the environment group to create a private CPU
Torch generator. `count` determines grouping and branch allocation. Sampling
does not consume global random state. Re-observation alone never advances the
sampling stream.

`sample_range()` retains branch zero's nominal value and distributes the other
N-1 branches across permuted strata of the accepted interval with seeded jitter.
Action planners map these samples into each row's legal interval when joint
limits restrict it. Reproducibility assumes the same candidate pool, costs,
constraints, and stream identity; candidate-provider randomness and simulator
determinism are separate concerns.

## Drawer tuning and temporary contact workaround

The packaged UR5 drawer task resolves these settings:

| Setting | Current value | Configuration under `embodichain_tasks/configs/` |
|---|---|---|
| Hand `open` / `grasp` joint targets | `[0.0]` / `[0.040]` | `components/embodiments/ur5_dh_pgi_140_80.yaml`, `skill_profile` hand commands |
| Motion sample budget | `100` | `components/execution_policies/trajectory_open_loop.yaml`, `motion.sample_count` |
| Each close/open interpolation | `18` | `tasks/manipulation/open_drawer/task_program/integration.yaml`, `hand_interp_steps` |
| Pull distance | `0.18` m | Same integration file, `translation_distance` |

For a pull, Slide reserves `2 * hand_interp_steps` samples for the hand and
divides the rest among approach, reach, and pull. The current sequence is
**22 approach + 21 reach + 18 close + 21 pull + 18 open = 100 samples**.
With a fixed control interval, increasing the motion budget allocates more
samples and time to arm motion. Increasing only `hand_interp_steps` within a
fixed total budget reduces the samples available to the arm. The embodiment
and execution policy are shared components, so changing them affects other
deployments that reference them.

The Default/CUDA drawer environment also enables
`refresh_articulation_contact_material` once at startup, scoped to the drawer's
`large_handle_bar` in `tasks/manipulation/open_drawer/env.yaml`. This temporary
workaround addresses the observed first-environment loss of handle contact
during pulling. The backend helper briefly writes dynamic friction plus 0.001
and immediately restores the original value, forcing a native material rebind
without a physics step. It leaves final friction, mass, inertia, and joint
state unchanged and does nothing on CPU or Newton. Remove the startup event
after the underlying DexSim material-binding issue is fixed; it is separate
from affordance selection and trajectory generation.

## Collection and success semantics

Expansion collection accepts rows with successful completion and recorded
frames, commits at most the remaining episode quota, and discards other rows
during the full reset. Dataset, trajectory, and camera recorders receive the
same explicit `commit_env_ids`, including a partial final batch.

The packaged open-loop policy uses `effect_assurance: projected`. Completion
therefore does not establish measured physical success. Checking drawer joint
travel or whether an object was actually lifted requires separate measured
evidence; candidate validity and IK alone cannot provide it.

See {doc}`builtin_actions` for action contracts and
{doc}`/api_reference/public_api` for the sampling API. Focused regression tests
live in `tests/sim/atomic_actions/test_affordance_sampling.py`,
`tests/sim/atomic_actions/test_actions.py`, `tests/lab/scripts/test_run_env.py`,
`tests/gym/envs/task_program/test_simulation_environment.py`,
`tests/gym/envs/managers/test_dataset_manager.py`, and
`tests/gym/envs/managers/test_event_contact_material.py`.
