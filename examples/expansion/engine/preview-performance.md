# Speed up trajectory expansion previews

From the repository root in the `py312` environment, prepare the validated
grasps for the UR5 gripper and cube once. The collector uses the same cache for every point.

```bash
embodichain affordance precompute-grasp-cache \
  --gym-config examples/expansion/engine/task.default.yaml --pair eef cube cube_grasp
```

Run four parameter points in one collection World with eight environment rows:

```bash
embodichain expansion preview --job examples/expansion/engine/job.preview.yaml \
  --interior-samples 1 --num-envs 8 --port 0 --open
```

The lower bound, upper bound and interior point each request four trajectories;
the center requests two. With `batch_size: 2`, each point uses two rows in the
first batch. Six rows finish the remaining recipes in the second batch.
The default capacity is the sum of each point's first batch size. Set
`--num-envs` to cap it for machines with tighter memory budgets.

The CLI closes the collection World before creating the replay World. Both use
the same process renderer. Each recording retains the point's frozen inputs,
local recipe index, and measured validation. The ordered point records in
`preview-plan.json` identify episodes in `jobs/collection`. The `elapsed_s`
field measures collection and verification, excluding viewer startup. Compare
capacity settings using the same seed and check both `confirmed` and
`coverage_complete`. For this command the expected counts are 2, 4, 4 and 4 in
manifest order, with 14 confirmed episodes.

## How command frames are built

The arm and gripper use the same joint indices at every waypoint. For built-in
CPU joint commands, the builder gathers positions and velocities for each
endpoint in blocks of waypoints. It validates each block and emits one
`RuntimeCommandFrame` per waypoint in time order. Sequence snapshots also
validate environment IDs, hold durations, and joint values in blocks. Each
frame owns its tensors. When Task Program records the number of frames in a
plan, it reads the count directly from the archived plan.

## Command frame measurements

On one CPU thread of a Ryzen 9 9950X, 21 paired measurements on one captured
218-frame trajectory gave the following medians for the command-frame changes.

| Operation | Baseline | Batched | Speedup |
| --- | ---: | ---: | ---: |
| Build the command sequence | 18.20 ms | 2.46 ms | 7.41× |
| Snapshot the sequence | 9.68 ms | 3.35 ms | 2.89× |
