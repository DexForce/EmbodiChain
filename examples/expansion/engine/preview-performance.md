# Speed up trajectory expansion previews

From the repository root in the `py312` environment, prepare the validated
grasps for the UR5 gripper and cube once. All sample workers read this cache.

```bash
embodichain affordance precompute-grasp-cache \
  --gym-config examples/expansion/engine/task.default.yaml --pair eef cube cube_grasp
```

Run the preview with four sample workers. The viewer opens after collection
and validation finish.

```bash
embodichain expansion preview --job examples/expansion/engine/job.preview.yaml \
  --interior-samples 1 --sample-workers 4 --port 0 --open
```

## What a sample worker does

`--sample-workers N` runs up to N parameter points at once. With one interior
sample, this job has four points. The center requests two trajectories, and
the lower bound, upper bound, and interior point each request four. Each worker
runs one point's frozen job in a separate process with its own simulation World
and output directory. This overlaps World startup and trajectory collection.
Within each point, the job's `batch_size: 2` still controls how many episodes
run together.

The default is one worker. Increase the count when the machine has enough
memory for the concurrent Worlds. Jobs that render also need GPU memory.
The parent process enforces a shared deadline, verifies the accepted recipes,
and records results in the original sampling order.

## How command frames are built

The arm and gripper use the same joint indices at every waypoint. For built-in
CPU joint commands, the builder gathers positions and velocities for each
endpoint in blocks of waypoints. It validates each block and emits one
`RuntimeCommandFrame` per waypoint in time order. Sequence snapshots also
validate environment IDs, hold durations, and joint values in blocks. Each
frame owns its tensors. When Task Program records the number of frames in a
plan, it reads the count directly from the archived plan.

## Measured performance

On a Ryzen 9 9950X using eight physical cores, the serial baseline
(`8b4c8b212`) and the optimized four-worker preview (`6fc399e91`) each ran three times
with the same job and seed. All six runs confirmed the 14 requested recipes.

| Operation | Baseline | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Collection and recipe validation | 97.85 s | 33.61 s | 2.91× |
| Build a 218-frame command sequence | 18.20 ms | 2.46 ms | 7.41× |
| Snapshot the same sequence | 9.68 ms | 3.35 ms | 2.89× |

The collection row reports the median across the three runs. The frame rows
report medians from 21 alternating measurements on one CPU thread using the
same trajectory captured from this job. Parallel sampling accounts for most
of the collection speedup. Batched construction reduces the cost of building
each command sequence.

To measure the effect of worker count on your machine, run the example with
`--sample-workers 1` and `--sample-workers 4` using the same job, sample count,
and acceptance criteria. The CLI summary and `preview-plan.json` report
collection time in `elapsed_s`. This example requests 14 trajectories.
Compare each point's accepted and rejected recipes across runs. The
`outcomes` field in `preview-plan.json` records the reason for each rejection.
