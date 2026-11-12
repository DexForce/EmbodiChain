# Route the Cable (YAM)

This handwritten expert routes a 600 mm anchored cable around two spools and
places its free end in a 10 mm channel. It adapts the
[robosimstudio example](https://github.com/robosimstudio/robosimstudio/tree/73739abe212f963e8a9f5862b27bca370df091a7/examples/08_demos/route_the_cable).
The task uses EmbodiChain's scene components, motion generator, Atomic Skills,
`DemoSegment` executor and trajectory recorder. Only the proximal cable segment
is fixed; grasps use collision and friction, with no runtime attachment or
in-episode state writes.

## Requirements and entry point

Use CUDA and a DexSim build exposing `CableDesc`, `Scene.create_cable_batch`,
`CableBatch.fetch_default_state/apply_state` and `Scene.reset_arenas`. See the
[cable example installation guide](../../../../../../../examples/README.md#newton-cable-demo)
for the development-build dependency and solver migration. Earlier routing
qualification used cable-state commit
`43d078958e6e00158eda0c3024d8af0c5682eb59` with Newton 1.6; those results do not
qualify full routing on a newer engine. A stock wheel lacking the required
cable APIs cannot run this deployment. Permanent cable attachments to articulations
with nonzero initial joint positions are rejected before scene commit: the current
engine resolves their anchors before model-initialization hooks. This task uses
contact-only grasps and does not need that unsupported combination. The file owns Newton/DexUni, dynamic
MuJoCo joints and two-way VBD cable contact; CLI `--physics` does not switch it.

From the EmbodiChain repository:

```bash
python -m embodichain data download --name YamI2rt
python -m embodichain list-task --category manipulation/cable
python -m embodichain run-task \
  --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/cable/route_the_cable/task.yam.yaml \
  --max_episodes 1 --seed 73 --headless --record_trajectory \
  --trajectory_save_dir ./output/route_the_cable
```

`YamI2rtCfg` resolves `get_data_path("YamI2rt/yam_i2rt.urdf")` through the same
asset registry and cache used by Franka. The predownload command above is optional:
constructing the robot downloads and extracts the ready
`robot_assets/YamI2rtV1.zip` bundle when needed. Downloading the bundle does not
require CoACD. `EMBODICHAIN_DOWNLOAD_PREFIX` selects a mirror; the official
`DexForceAI/embodichain_data` Hugging Face dataset is the fallback.

The bundle was
[published on 2026-11-05](https://huggingface.co/datasets/DexForceAI/embodichain_data/commit/8c0b7dcc8a80ab5a1f177c7384ad9d6950281a68)
and is publicly downloadable without a Hugging Face token. Official HF download
into an empty cache, archive checksums, extraction, cache reuse and robot
configuration/FK-chain loading were verified. This asset validation does not
qualify a complete routing episode.

For optional reconstruction from the pinned source in a checkout, use:

```bash
python -m pip install coacd==1.0.14 trimesh==4.12.2 scipy==1.18.1
python scripts/tools/prepare_yam_i2rt_assets.py
```

The preparation script downloads the 9 STL meshes, source MJCF and MIT license
from robosimstudio commit `73739abe212f963e8a9f5862b27bca370df091a7`. It converts
the dual-arm MJCF to URDF and generates 24 left-tip and 15 right-tip collision
hulls with CoACD 1.0.14, threshold 0.06 and seed 73. Every source and generated
file must match the SHA-256 of the qualified robot before the staged directory
is installed. A valid existing bundle is verified and reused; an incomplete or
modified bundle is rejected without being overwritten. This preparation runs
on CPU and does not require the simulation renderer or CUDA.

The download registry extracts under `<data_root>/extract/YamI2rt/`; the source
script defaults to `<data_root>/YamI2rt/`. Both are recognized by `get_data_path`.
The default data root is `~/.cache/embodichain_data/`. Set
`EMBODICHAIN_DATA_ROOT` before both preparation and demo execution to use another
cache root. For offline preparation, pass `--source-dir` pointing to the pinned
upstream `assets/embodiments/yam_i2rt/` directory. `--output` selects an explicit
destination; configure the robot's `fpath` to its `yam_i2rt.urdf` if this differs
from the default cache. The raw meshes, generated URDF and collision hulls are
local assets, excluded from Git and the Python wheel.

For asset publication, `--archive /path/to/YamI2rtV1.zip` additionally builds a
deterministic archive from the verified bundle and includes its MIT license,
NOTICE and provenance. It retains the qualified geometry and all 52 original
file hashes. Archive MD5: `23601ea820ff3e6260336c4962ef01a5`; SHA-256:
`9125380a6240847f42730b79025aa61cd17b665f90a2586b3c6f061d769c503a`.
The repository's `NOTICE` and `licenses/YamI2rt-MIT.txt` record the external
asset attribution even though the binary assets are distributed separately.

The collector accepts complete successful episodes only. A failed intermediate
validator stops its episode. A launched expert is not evidence of task success.
This expert deployment sets `visualization.allow_commands: false`: Viser is an
observer, with camera orbit/pan still available. Dragging robot controls while
the expert writes joint targets introduces a competing controller and can
invalidate grasps. The launcher preserves this read-only policy with `--viser`.
Use a separately configured, stepping scene without the expert for manual IK
experiments. Failed motion validators log planning status, TCP position error
and maximum measured grasp slip with their limits.
Feedback insertion moves may hold the same target for up to one extra second
while physical tracking converges. They retain their original position and
slip tolerances and fail when that bounded settling period is exhausted.

## Stages and reusable boundaries

The right arm places capsule centres at material distances 0.162, 0.282 and
0.438 m; these are the nearest capsule centres to the upstream nominal grasp
locations. The left arm grasps at 0.570 m, selects a wrist branch that can also
reach the channel staging pose, observes the distal tangent, and corrects yaw
and lateral intersection. It lowers through bounded height stages, observes the
actual free end, and accumulates tip-position corrections on the commanded pose
to compensate contact-load bias. Once the end is inside the channel or its
narrow aligned release corridor, it opens the fingers and lets the cable settle.
The final validator requires the actual end inside the channel after release,
retreat and settling. Each stage is
planned lazily from the latest cable state. Both hands return home before the
final settling period.

The YAM IK uses 10 μm position and 1e-5 rad rotation convergence so finite-
difference joint acceleration is not dominated by numerical jitter during
Cartesian retiming. Task moves accept the planner's possibly longer control
grid; grasp-slip and physical placement thresholds are unchanged.

The third carry targets a TCP height of 50 mm above the bench. The former
65 mm target could exhaust the 12 mm slip allowance during `raise_2`, even
with successful IK and acceptable TCP tracking. The shorter lift retains the
same firm grasp and validators. Its CUDA regression checks material-point
clearance, grasp slip, second-spool seating, and robot/spool contact throughout
raising, carrying, lowering, release and retreat.

At each spool, the right hand releases to a 16 mm jaw aperture, retreats to
0.82 m TCP height, and only then opens fully. Opening through the full finger
stroke beside the first spool sweeps a finger through its rim. These explicit
waypoints are task geometry: joint interpolation does not provide obstacle
avoidance, and contact response is not a substitute for a clear commanded path.

- `PickUp` accepts `ContactGraspGoal` for the observed pinch pose. It shares
  approach/close/lift command assembly with rigid pickup and requests physical
  contact verification. The routing host verifies material-point lift and slip.
  `PickUpOptions` retains the 48-frame close, 0.35-second closed hold and
  18.5 mm lift; `MotionPolicy` samples the lift at the previous 0.05 m/s grid.
- `compute.geometry.cable` owns simulator-independent capsule endpoints,
  material-point observations and TCP-relative slip measurement.
- `RouteTheCableEnv.create_demo_segments()` owns feedback, staged execution and
  physical validators. Its `milestone="pinch"` or `"first_spool"` options qualify
  intermediate stages through `execute_demo_episode`; they do not certify a
  complete routing episode.
- `YamI2rtCfg` resolves the versioned robot bundle from the data registry/cache or an explicit
  `fpath`. Its bundle retains the original MIT license and pinned provenance.
  Simulation and FK use
  the same URDF. Fingertips retain separate convex pieces instead of one hull
  spanning their concave grasp region.

The scene follows the pinned upstream `RodSpec` coefficients, including density,
stretch/shear, bend/twist and contact damping. Shape friction combines by a
geometric mean: cable/pad are 60, bench 0.02, and other scenery 2. The task uses
contact history and sticky matching as the upstream contact-only example does;
these values are task settings, not global physics defaults. The routing
component increases left-arm drive stiffness for insertion tracking. The grasp
profile uses the YAM finger-width calibration and cable radius to request a
small interference rather than driving an empty-hand closure against the cable.
The first carry, high third carry and free-end insertion use firm per-invocation
grasp commands through the existing `ActionControlOverrides` interface. The
first grasp needs this holding margin: its light default pinch can reach the
12 mm slip rejection limit during carry/lowering, even without browser input.
Each pinch holds the fully closed command for 0.35 s on the control grid before
lifting. This dwell is part of the Atomic Skill's `close` phase; the material
reference is sampled after closing/settling, before any lift, and is not rebased
during the grasp.

## Validation and recordings

Every grasp must lift the observed material point by more than 8 mm and remain
within 12 mm of its post-closure TCP-relative position. Final metrics include
both spool-seat counts, free-end location, hand clearance, speed and sampled
cable/bench, cable/spool and cable/port penetration. `reference_routed`
reproduces the upstream broad regions. Final `success` additionally requires
the free end inside the actual
channel and penetration no greater than 1 mm. Sampled SDF diagnostics do not
constitute an exact continuous collision-distance query.

These final cable predicates do not certify the robot's swept path. The
first-spool CUDA regression samples backend robot/spool contacts throughout
approach, placement, release and retreat, rejects penetration over 1 mm, and
also requires the cable to remain seated and the hand to end open above the
spool:

```bash
python -m pytest tests/gym/envs/tasks/test_route_the_cable_cuda.py --run-gpu -m gpu -q
```

The contact-pickup migration is covered by 1349 CPU tests spanning the Atomic
Skill and semantic-planning contracts and by both existing NoRender CUDA
regressions with seed 73. Maximum grasp slip was 5.26 mm in the first-spool
case and 4.37 mm in the third carry, with no reported robot/spool penetration
in the latter. The original 12 mm slip and 1 mm penetration limits are retained.
The local runtime reports DexSim build `a3daf24edd88b489702c32ccffc7ffa405741ab2`,
Torch 2.11.0+cu130, Newton 1.7.0 and Warp 1.17.0 on an RTX 5090.

The synchronized FastRT comparison runs the old `GraspCable` implementation at
`df8e304f0` and the unified `PickUp(ContactGraspGoal)` with identical seed 73,
physical configuration, camera and control settings. Its old run stops during
the left-hand pinch at 12.28 mm episode slip; the new run reaches final settling
with 6.96 mm slip but fails the strict port predicate after release. Both report
no robot/spool penetration. These are individual full-task trials. Complete
channel insertion, multi-seed success rates and repeated-reset reliability
require further work.

Recordings contain full robot commands and
`states.cables.cable.segment_state_w`, shaped `(environments, time, 50, 13)` in
world XYZ + XYZW + COM linear/angular velocity order. Segment outcomes retain
planning status and material-coordinate identities. A cable observation is not
a solver-history checkpoint. Construct the replay environment with the same
physical configuration. New recordings retain source Arena origins so a selected
row can replay in a fresh single-environment scene. Older files without origin
metadata require matching Arena placement. `ReplayWrapper(mode="dynamic")` verifies
matching full-reset recorded observations and environment counts; it replays
commands without overwriting cable states. Both `position` and
`position_velocity` expert command modes use the environment control grid; the
latter records derived joint velocities and zero velocities during holds. Cable kinematic scrubbing
and partial replay resets are rejected. The deployment defaults to one episode and
one attempt per process. Reset/episode repeatability with sticky
contact needs separate physical qualification; API state/reset tests alone do
not prove repeatable frictional routing. Use a fresh process for each trial until
that reset path is qualified. Raw trajectory recordings may include failed
attempts; select training data using the segment and final outcomes.
