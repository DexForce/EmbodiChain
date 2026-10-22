(task-program-recipes)=

# Common program changes

Start from the {doc}`quickstart` and keep the built-in Franka deployment.
Unless stated otherwise, edit your local `program.yaml` and use the complete
no-save command in Quickstart step 3. The complete starting files are
{download}`two_moves.yaml <examples/two_moves.yaml>` and
{download}`two_moves_checked.yaml <examples/two_moves_checked.yaml>`.
Keep `program_id` matched to the integration and use its semantic entity names.

## Change repetition count or execution order

**Need:** make two moves instead of three, or visit B before A.

**Edit:** `program.count` and `targets.drop_pose.values` in `program.yaml`.
A repeat count is the number of times its whole body runs, not the number of
individual calls. With one Pick and one Place, count 2 produces four segments.
Swap entire pose entries, including their orientations. If using acceptance
checks, swap `rest_pose.values` in the same way. Update the root instruction.

The {download}`two-move program <examples/two_moves.yaml>` is the complete
worked example. Reordering `body.items` changes actual execution order; preserve
physical prerequisites such as picking before placing. Cyclic targets are
selected by the nearest enclosing repeat iteration, not by a global call counter.

**Verify:** rerun with seed 7 and one environment. Check four segment entries,
B → A, and the corresponding measured target indices in the checked variant.
Changing only the description will not change the sequence.

## Change a target position or orientation

**Need:** move a Place destination within the existing scene.

**Edit:** a pose entry in `targets.drop_pose.values`. Positions are in **meters**;
orientations are quaternions in **XYZW** order, so `[0, 0, 0, 1]` is identity.
The absolute target is an **object pose in the environment's world frame**,
not a joint command or a TCP/end-effector pose. The integration uses the grasp
relationship to determine robot motion. Do not add a TCP offset to the object
target or interpret these numbers in a camera frame.

For the floor-standing cube, change the matching `rest_pose` XY too. Keep its
center Z at `0.025 m` only while the cube remains 5 cm tall and rests on this
ground plane. Release Z and resting Z serve different purposes. Changing the
quaternion changes the desired object orientation; the position-only validator
does not verify it. Additional surfaces or objects require integration work.

**Verify:** change one pose at a time, run the checked program, and inspect the
measured position error. Check reachability and scene clearance visually.
The tutorial's open-loop policy has dynamic collision checking disabled, so
successful compilation does not establish clearance for arbitrary new targets.
For other frames and semantic targets, see {doc}`scene_registry`.

## Add another Pick/Place pair

**Need:** perform another move with the existing cube.

**Edit:** for the same pair of operations, increase `program.count` first. If the
steps need different instructions or checks, use a `sequence` with explicit
segment entries, following the complete structures in the downloadable files.
Keep `kind: invoke` under `steps` and `kind: pick` / `kind: place` under `call`.
Use `object: cube` and declare every referenced target.

This integration supports the Pick/Place workflow. A call being part of the
language does not mean every deployment supports it. A second object, new
robot, HandOver, or registered skill needs matching scene/resources/lowerers;
follow {doc}`/tutorial/task_program` and
[registered calls](architecture.md#registered-calls).

**Verify:** check that the segment count and order match your intent, then rerun
position/stability checks for every Place. Keep names useful for diagnosis and
avoid assuming that a target advances merely because another call uses it.

## Change task and segment language

**Need:** label the whole demonstration and its subgoals.

**Edit:** root `instruction` and each segment's `instruction` in `program.yaml`.
The root describes the episode; segment text describes that part of the task.
For example, `Move the cube to target B, then to target A.` matches the modified
program. Keep the wording consistent with the actual repeat count and target
order. Do not change `program_id` to change the label.

**Verify:** after saving, run the Quickstart data reader and inspect the printed
Task and segment instructions. Task language is stored in `meta/tasks.parquet`,
segment language in `meta/subtasks.parquet`, with indices in frame data.
See {doc}`/guides/expert_task_language` for overrides and training consumption.

## Wait for settling and check the position

**Need:** reject a run when the cube does not settle near the requested location.

**Edit:** add `post` and `validators` to the Place segment in `program.yaml`, and
provide separate resting targets. Use the complete
{download}`checked program <examples/two_moves_checked.yaml>`:

```{literalinclude} examples/two_moves_checked.yaml
:language: yaml
:start-at:     post:
```

`preset: rigid_object` selects the integration's existing stability preset; it
is not a new tuning field. Post-policies run after the motion and before the
validators. The position tolerance is a 3D distance, not a separate tolerance
per axis. A settling timeout or failed validator rejects the affected row.

**Verify:** inspect `metadata.post_policies` and `metadata.validation` on each
saved Place segment. The Quickstart's observed errors are below `0.03 m`.
For a failed run, add `--debug-mode` to its no-save command to print the failed
demo trace. Inspect the actual pose and settling status before changing the task.
These checks do not independently prove grasp contact or release.

## Change motion timing

**Need:** understand or adjust how long the robot takes to move.

**Edit location:** the **execution-policy component selected by the deployment**,
not the program YAML. The Quickstart policy uses these actual fields:

```{literalinclude} ../../../../extensions/embodichain_tasks/configs/components/execution_policies/trajectory_open_loop.yaml
:language: yaml
```

In this `ik_interp` configuration, `motion.sample_count: 40` generates 40
trajectory samples per Pick/Place. With the environment's `0.04 s` control
interval, the observed unextended segments contain 40 steps (`1.6 s` of recorded
time). Settling adds steps separately. `tracking.kind: timed` controls time-based
tracking; `runner.minimum_cycle_time` is a runner cadence setting, not a uniform
Cartesian speed command. Sample count is not a speed in meters per second,
and its meaning is not interchangeable across motion strategies.

There is no general `speed` field on a repeat, segment or Pick/Place call.
`--task-program` cannot override the execution policy. To change it, follow
{doc}`/tutorial/task_program` to create a local deployment selecting a compatible
policy, or choose an existing compatible deployment from the task catalog.
Do not edit installed package files in place. Preserve policy/embodiment
contracts and explicit assurance. This is an integration-author operation.

**Verify:** compare recorded segment lengths and physical acceptance at the same
seed, and recheck reachability, settling and position. Only the displayed
40-sample policy is physically exercised in this tutorial; a timing change
requires its own validation.

## Choose another existing deployment

**Need:** use a supported robot or task without building an integration.

**Edit:** select its `--gym_config` package reference, using
{doc}`/guides/task_catalog` and `show-task` to discover it. First run its original
program with its documented defaults. Reuse your local program only when its
`program_id`, semantic entities, targets and calls satisfy the new deployment's
integration and contracts. Compatible embodiments can share a program; arbitrary
tasks cannot.

The repeated Pick/Place Franka deployment does not enable expansion. Its UR5
counterpart selects an expansion declaration and can have different batch and
collection defaults. Backend variants also select separate physical components;
`--physics` is not a general way to turn any environment into another backend.

**Verify:** inspect the selected robot, backend, expanded segment count and
collection settings, then run a single controlled episode and check the same
physical criteria. Single-environment success does not qualify randomized or
batch execution. For new scene/robot ownership, continue with
{doc}`/tutorial/task_program`, {doc}`scene_registry` and {doc}`robot_profiles`.
