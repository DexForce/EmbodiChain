# Trajectory Variants

A demonstration collector that replans the same task many times tends to return
the same motion every time. Imitation learning then sees one way of doing the
job, and reinforcement-learning post-training starts from a policy that has
never been shown an alternative. Variant expansion keeps the waypoints a task
has already committed to and varies only the motion between them.

This is narrower than affordance expansion, which changes *where* the robot
makes contact. Here every grasp, placement and dwell stays exactly where
planning put it, and only the free motion in between differs.

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Concern
  - Affordance expansion
  - Trajectory variants
* - What varies
  - Contact poses and motion ranges
  - Free motion between fixed waypoints
* - Where it applies
  - Before IK and path planning
  - After planning, on the joint trajectory
* - Waypoints
  - Resampled per candidate
  - Preserved exactly
```

The two compose: an affordance variation produces a new set of waypoints, and
variant expansion then produces several ways of executing that set.

## Where the variation comes from

```{list-table}
:header-rows: 1
:widths: 22 20 58

* - Factor
  - Operator
  - What changes
* - `spatial`
  - {func}`~embodichain.lab.sim.motion.expansion.joint_residual`,
    {func}`~embodichain.lab.sim.motion.expansion.via_points`
  - The shape of the joint path between two waypoints. `via_points` samples
    several interior knots, so paths differ in shape rather than only in
    amplitude.
* - `ik`
  - {func}`~embodichain.lab.sim.motion.expansion.nullspace_residual`
  - Arm posture, while holding the task rows the caller declares. This is what
    produces visibly different elbow and wrist configurations for one tool path.
* - `timing`
  - {func}`~embodichain.lab.sim.motion.expansion.retime`
  - Free-phase duration, and the velocity profile within a phase. The path is
    untouched; only when the robot is where it is changes.
* - `approach`
  - {func}`~embodichain.lab.sim.motion.expansion.perturb_approach_direction`
  - The standoff pose a contact is reached from. This one produces Cartesian
    poses, so it is consumed *before* IK and planning, by whichever component
    builds the waypoints.
```

The `contact`, `contact_timing` and `recovery` factors are declared in the
configuration schema but not implemented; enabling one is rejected.

## What is and is not guaranteed

Every qpos-level operator uses an envelope that is zero, with zero derivative,
at both endpoints of the phase it modifies. Phases annotated `contact` or
`hold` are never touched at all. So:

- **Guaranteed.** Annotated waypoints, contact windows and dwell durations are
  bit-identical to the reference. Uncontrolled joints, such as a gripper, keep
  their schedule. Sampled joint-limit violations are rejected, and
  {func}`~embodichain.lab.sim.motion.expansion.validate_motion_limits` can
  reject sampled speed and acceleration violations.
- **First order only.** `nullspace_residual` holds the declared task rows by
  projecting onto the Jacobian null space at each sample. Interior samples drift
  as the linearization error grows; the phase endpoints stay exact regardless.
  Verify interior tool poses with forward kinematics if your task needs them.
- **Not guaranteed.** Nothing here establishes collision freedom, dynamic
  feasibility or task success. Deduplication compares measured joint geometry
  and elapsed phase time, which is a similarity measure, not a validity proof.
  Each accepted row must still be executed and validated by the host.

## Enabling it on the official stack-blocks task

`StackBlocksTwo-v1` ships the expansion wired up but switched off, so its
default behavior is unchanged. Set `enabled` to `true` in
`embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/env.json`:

```json
"extensions": {
    "trajectory_variants": {
        "enabled": true,
        "seed": 0,
        "factors": {
            "spatial": {"enabled": true, "method": "via_points", "joint_offset_scale": 0.03, "via_count": 3},
            "ik": {"enabled": true, "normalized_scale": 0.05, "task_rows": [0, 1, 2, 3, 4]},
            "timing": {"enabled": true, "duration_scales": [1.0, 1.25], "profiles": ["uniform", "ease_in", "ease_out"]}
        },
        "coverage": {"target_per_cell": 4, "joint_dedup_normalized_tol": 0.01}
    }
}
```

Then run the task in the environment where EmbodiChain is installed (see
[Installation](../../../quick_start/install.md)). Each parallel environment
receives a different variant, so `--num_envs` is what sets how many variants one
rollout collects:

```bash
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/env.json --num_envs 8 --device cuda --headless --seed 42 --max_episodes 24
```

Append `--record_trajectory --trajectory_save_dir ./outputs/trajectory_variants` to
save the joint trajectories.

Environment zero always replays its unmodified reference, so the trajectory you
already trust stays in the dataset. With `--num_envs 1` the expansion therefore
produces exactly the reference and nothing else.

Successive episodes start further along the list of combinations rather than
repeating the first ones, and each episode draws its own residuals, so a longer run keeps
adding combinations instead of repeating the first batch.

Each demonstration segment records what happened under
`metadata["trajectory_variants"]`: the episode index, the variant assigned to every
row, the accepted sample length of every row, the resolved phase boundaries of
every row, and a count of rows that fell back to their reference along with the
reason. Phase
boundaries are reported per row because retiming moves them, so a consumer
cannot recover contact windows from the nominal annotation alone.

### Which phases the task varies

Phase boundaries come from the atomic actions' own named segments rather than
from re-deriving their internal step arithmetic. The gripper-close and
gripper-open segments become `contact` phases and the dwell after closing
becomes a `hold`, so no operator can move a grasp, a placement, or the settling
pause that follows them.

Retiming is confined to phases at or after the lift. The task's executor clears
the held block's dynamics at one shared step index, so changing how long the
approach takes would desynchronize that event across environments. Path and
posture variation still apply to the approach, because they do not change the
sample count.

## Choosing an API

```{list-table}
:header-rows: 1
:widths: 32 68

* - Function
  - Use it when
* - {func}`~embodichain.lab.sim.motion.expansion.expand_row_variants`
  - The rows are independent scenes, one per parallel environment. Row `i` gets
    enumeration ordinal `i`; nothing is deduplicated across rows because
    geometries from different scenes are not comparable. A rejected row falls
    back to its own reference rather than leaving an environment without a
    command stream.
* - {func}`~embodichain.lab.sim.motion.expansion.expand_trajectory_variants`
  - One fixed scene should yield many variants. Proposals that duplicate an
    accepted row's measured geometry and timing are rejected, and every
    rejection is counted and reported.
* - {func}`~embodichain.lab.sim.motion.expansion.plan_trajectory_variants` with
    {func}`~embodichain.lab.sim.motion.expansion.apply_trajectory_variant`
  - You want the enumeration and the application separately, for example to log
    the intended assignment before planning.
```

The `coverage.target_per_cell` setting caps how many timing variants may share
one geometry family. Timing-only expansion therefore cannot exceed that cap,
because all of its rows have the same measured geometry — raise it when timing
is the only enabled factor.

## Redundancy and the Jacobian frame

`nullspace_residual` takes whatever task rows the caller declares, in the rows
of the Jacobian the caller supplies. A six-row Jacobian on a six-joint arm
generically leaves no redundancy, and the operator raises rather than silently
returning the reference. Dropping the rows the task does not actually constrain
is what creates alternative postures.

The frame convention is the caller's responsibility. `BaseSolver.get_jacobian`
returns a Jacobian in the arm's base frame, so dropping its angular-z row
removes rotation about base z, not about the tool axis. For a top-down grasp
these coincide, which is why the stack-blocks task declares
`task_rows: [0, 1, 2, 3, 4]`. A task that grasps from the side must choose its
rows accordingly, or supply a Jacobian already expressed in the tool frame.

Column order matters too: solver Jacobian columns follow the solver's own joint
names, which need not match the robot's public joint order. Supply
`task_jacobians` already permuted into `controlled_joint_indices` order.

## Limitations

- At most one joint-path operator runs per variant. A null-space projection
  stacked on an already displaced joint path would no longer correspond to the
  Jacobians it was computed from, so the combination is not offered.
- Retiming requires the reference trajectory's phase boundaries to already sit
  on the host control clock. An unaligned reference is rejected rather than
  silently resampled.
- Rows retimed to different lengths are padded by repeating the final valid
  sample. A shorter row holds its last command instead of running past its own
  end; use `CandidateTrajectoryBatch.valid_mask` before recording.
- The generation bookkeeping in
  {class}`~embodichain.lab.sim.motion.expansion.GenerationSession` is not
  used by this path. Episode budgets, commit receipts and durable persistence
  remain the host's, which for the packaged task means the environment's own
  dataset manager.
