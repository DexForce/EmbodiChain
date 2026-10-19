# Independent Atomic skill benchmark

```bash
conda activate embodichain
embodichain benchmark atomic-action --robot ur5 --skill all --planner trapezoidal
```

Each case builds its own simulation, compiles `AtomicActionEngine` invocations,
drives the robot, and scores the measured object/joint state. It does not load
`run_task`, `run-task`, Gym deployments, or Task Programs. Shared production
helpers under `embodichain/lab/scripts/_atomic_action/` supply the sample assets,
grippers, mounts, poses and expansion machinery.

UR5 single-arm scenes use DH PGI-140-80. **Handover explicitly mounts
`Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf` on both arms**, matching the
`run-task` Robotiq demo. This applies to every supported Handover arm preset.

## Selection and replay

`--skill` accepts `pickup`, `place`, `pour`, `press`, `slide`, `twist`, `handover`,
`move_joint`, `axis_align`, `open_door`, or `all`. `--planner` accepts
`trapezoidal`, `toppra`, `curobo`, or `all`. Both selectors accept several values.
`--robot ur5` supports all ten scenes; `franka` and `ur10` support Pickup, Place,
MoveJoint and Handover, matching the direct Atomic sample capability matrix.
Unsupported combinations remain visible without starting a simulator.

```bash
# Nominal physical success, without expansion.
embodichain benchmark atomic-action \
  --robot ur5 --skill all --planner trapezoidal --no-expansion

# Compare selected planners with four expansion recipes per scene.
embodichain benchmark atomic-action \
  --robot ur5 --skill pickup press pour \
  --planner trapezoidal toppra --smoke --num-envs 4

# Open the simulation window for a selected benchmark.
embodichain benchmark atomic-action \
  --robot ur5 --skill handover --planner trapezoidal \
  --no-headless --renderer hybrid --smoke

# cuRobo requires its optional runtime and a CUDA device.
embodichain benchmark atomic-action \
  --robot ur5 --skill pickup --planner curobo --device cuda --smoke

# Validate selections and recipe bounds without creating a simulator.
embodichain benchmark atomic-action \
  --robot all --skill all --planner all --dry-run
```

Headless execution is the default. `--physics default` is the default backend;
`--physics newton` selects the corresponding shared sample contact settings.
Contact paths in Press, Slide and OpenDoor retain `ik_interp`, which bypasses
the selected planner to preserve Cartesian constraints. Other invocations use
`motion_gen`. The actual strategies are recorded in the raw results.

Playback uses drive targets with **16 physics steps per waypoint**, following
[`BENCHMARK_STANDARD.md`](BENCHMARK_STANDARD.md). Actual `qpos`, object poses,
TCP poses and articulated joints are read independently. Two terminal holds of
80 steps check post-release stability. `--steps-per-waypoint` and
`--settle-steps` override these settings and are saved with the results; use the
same settings when comparing planners. Tracking error is diagnostic.

## Physical success

The final `success_rate` counts rows passing every applicable stage. Planning
success is reported separately. Returning a finite trajectory or reaching its
last command does not establish task success.

| Skill | Measured stages |
| --- | --- |
| Pickup | grasped, lifted at least 4 cm, held after settling |
| Place | prerequisite Pickup, released, object XY within 1.5 cm of the release goal, stable landing |
| Pour | prerequisite Pickup, peak signed object rotation about its internal axis within 5° |
| Press | actual button movement, peak stroke at least 80% of the asset's 6 mm travel |
| Slide | grasped, drawer within 1 cm of the pull goal, released |
| Twist | grasped, TCP rotation about the knob axis within 5°, released |
| Handover | source grasped, transferred to both arms, source released with receiver holding, delivered within 3 cm |
| MoveJoint | each of three ordered arm targets within 0.3° |
| AxisAlign | grasped, lifted with object axis aligned within 5°, held |
| OpenDoor | grasped, hinge within 5° of the opening goal while controlled, released |

Place and Pour each include one explicit prerequisite Pickup, rather than a
repeated task. Benchmark Place lowers its final TCP release height to 0.03 m
above the ground, retaining the 0.20 m hover waypoint, to reduce free-fall
landing drift. Final Place XY error allows 1.5 cm after landing; settling drift
still allows 1 cm translation and 5° rotation. Slide performs one pull. Grasp
inference uses actual jaw closure, proximity and subsequent lift/holding state;
it is not a force-closure certificate. Holding checks permit object motion
relative to the gripper and evaluate actual height and proximity. Articulation
goals are scored at the controlled phase; release is checked after the final
hold. A failed prerequisite is named
separately. Per-row measurements and the first failed stage explain failures.
Release requires at least 90% reopening relative to each joint's configured
open/close command span, respecting PGI's metres and Robotiq's radians.

## Expansion and repetitions

Default coverage evaluates four Affordance branches × four trajectory variants
(16 recipes), or four trajectory variants for MoveJoint. `--smoke` selects
recipes `0 1 4 5` with the default policy (`0 1` for MoveJoint). `coverage` and
`full` evaluate every configured recipe.

```bash
embodichain benchmark atomic-action \
  --robot ur5 --skill handover --recipe-indices 0 1 4 5 --num-envs 4

embodichain benchmark atomic-action \
  --robot ur5 --skill press --planner trapezoidal \
  --seeds 0 1 2 --repeat 3 --affordance-branches 2 --trajectory-variants 2
```

`--num-envs` controls batch capacity independently of logical recipe count.
Final partial batches use exactly the remaining rows. Every batch, seed and
repetition creates a fresh scene and robot. Failed rows are counted without
retrying; remaining batches and cases continue after ordinary planning or
physical failures. Infrastructure failures/timeouts count unfinished requests
as failures and produce a nonzero exit after writing the report. Missing
hardware or optional planner dependencies are explicitly skipped.

Spatial variants reuse the direct host's conservative phase permissions:
gripper commands, action boundaries and contact phases stay fixed. Handover
only augments the empty-hand pickup approach. Pour's tilt/return segment stays
fixed; its prerequisite Pickup has editable free-motion phases.
Per-row records include selected Affordance poses, operator, actual maximum
joint delta and the protected-sample delta. Rejected expansion proposals can
retain the reference; only a measured nonzero delta counts as effective
spatial expansion.

## Results

Results go to `outputs/benchmarks/atomic_skills_<timestamp>/`, or `--output-dir`.
One `report.md` contains Time & Memory, Success & Other Metrics, and the planner
Leaderboard, weighted by attempted rows. `results.json` contains seeds,
repetitions, recipes, scene/gripper metadata, physical stages, measurements,
failures and expansion proof. Each isolated case subprocess also saves its
request, independent `configuration.json`, result and log.

`--save-trajectories` additionally saves `batch_*.pt` with reference and expanded
commands, actual replay traces, phase ranges and scene goals, for offline
inspection. Timing includes planning, augmentation and physical execution with
observation; scene setup and saving artifacts are excluded. CPU memory is RSS;
GPU memory covers PyTorch allocation, excluding native simulator allocations.
One seed and one repetition are point estimates; use several seeds/repetitions
for success-rate comparisons.

The original five object/position benchmark scripts remain available through
the explicit legacy selector:

```bash
embodichain benchmark atomic-action --action pick_up --smoke
```
