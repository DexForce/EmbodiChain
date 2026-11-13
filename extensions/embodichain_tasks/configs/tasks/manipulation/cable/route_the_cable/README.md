# Route the Cable (YAM)

This configured Task Program routes a 600 mm anchored cable around two spools and
places its free end in a 10 mm channel. It adapts the
[robosimstudio example](https://github.com/robosimstudio/robosimstudio/tree/73739abe212f963e8a9f5862b27bca370df091a7/examples/08_demos/route_the_cable).
The task uses the common `EmbodiedEnv`, configuration-owned registration,
Semantic Call lowering to existing Atomic Skills, the standard `DemoSegment`
bridge and trajectory recorder. Only the proximal cable segment
is fixed; grasps use collision and friction, with no runtime attachment or
in-episode state writes.

## Requirements and entry point

Use CUDA and a DexSim build exposing `CableDesc`, `Scene.create_cable_batch`,
`CableBatch.fetch_default_state/apply_state` and `Scene.reset_arenas`. See the
[cable example installation guide](../../../../../../../examples/README.md#newton-cable-demo)
for the development-build dependency and solver migration. The current
revalidation runtime uses DexSim `0.6.0.dev12034`, build
`b040478aa5cb4902b26a19a611eff768d229548b`, Newton 1.7.0 and
Torch 2.11.0+cu130. Full-episode revalidation of the revised path is still in
progress; results from earlier engine builds do not qualify this configuration.
A wheel lacking the required cable APIs cannot run this deployment. The
environment file owns Newton/DexUni, dynamic MuJoCo joints and two-way VBD cable
contact; CLI `--physics` does not switch it.

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
Every motion is followed by an explicit bounded hold and a measured TCP
position/orientation check. Insertion corrections read the latest material
geometry when grounded; the finite correction slots retain the commanded pose
once the release condition is reached. They do not run an unbounded Python loop.

## Stages and reusable boundaries

The program contains 51 segments and 129 Semantic Call invocations.

The right arm places capsule centres at material distances 0.162, 0.282 and
0.438 m; these are the nearest capsule centres to the upstream nominal grasp
locations. The left arm grasps at 0.57 m, leaving a 30 mm free span. All four
pinches use a 0.7605 m TCP height. `PickUp` closes and settles with
`lift_height: 0`; a separate `material_lift` target lowers to `MoveEndEffector`
and raises the held material while retracting opposite its observed tangent
projected onto XY. This supplies upstream slack as the cable lifts from the
bench. The lift preserves the measured tool orientation and keeps the original
post-closure slip reference.

| Grasp material distance | Retraction | Lift TCP height |
| --- | --- | --- |
| Right, 0.162 m | 1.5 mm | 0.774 m |
| Right, 0.282 m | 3 mm | 0.780 m |
| Right, 0.438 m | 8 mm | 0.800 m |
| Left, 0.570 m | 3 mm | 0.780 m |

The left arm selects a wrist branch that can reach the staging TCP position
`[0.586, 0.055, 0.805]` m above the open channel. The staging target constructs a
complete orthonormal rotation, including when the measured tool has pitch or
roll. Four bounded XY corrections aim the observed free end at
`[0.586, 0.077]` m. Subsequent TCP height stages are 0.795, 0.7875, 0.780 and
0.7655 m, each followed by two correction slots. Each correction is limited to
5 mm and keeps TCP Y at or below 0.05 m so the hand stays outside the channel
mouth while lowering. Final vertical corrections are limited to 3 mm and retain
a 0.7655 m TCP height floor. Corrections accumulate on the commanded pose to
compensate for contact-load bias.

Release requires the observed endpoint to be inside the strict channel with
`y >= 0.072` m. The configured release region is
`[0.5845, 0.072, 0.753]` to `[0.5875, 0.083, 0.758]` m, entirely within the
channel; readiness also accepts an endpoint already seated in the strict final
region at the same minimum depth. The X bounds account for the 3.5 mm cable
radius between the channel's inner walls. Merely crossing the channel mouth
does not stop the remaining corrections. Release also requires sampled cable
penetration no greater than 1 mm. Once ready, subsequent correction slots retain
the commanded pose and its contact load. Ordinary measured TCP tracking remains
required until the endpoint is seated inside the strict channel.

The left hand first releases with an 8 mm joint target per finger, then retreats
to TCP `[same X, 0.03, 0.82]` m. `open_clear_of_port` opens the hand fully only
after that retreat, avoiding a full finger sweep through the upstream cable at
the insertion pose. The final validator requires the actual end inside the
strict channel after release, retreat and settling. Each stage is planned
lazily from the latest cable state. Both hands return home before the final
settling period.

The YAM IK uses 10 μm position and 1e-5 rad rotation convergence so finite-
difference joint acceleration is not dominated by numerical jitter during
Cartesian retiming. Task moves accept the planner's possibly longer control
grid; grasp-slip and physical placement thresholds are unchanged.

At each spool, the right hand releases with an 8 mm joint target per finger,
retreats to 0.82 m TCP height, and only then opens fully. Opening through the full
finger stroke beside the first spool sweeps a finger through its rim. These explicit
waypoints are task geometry: joint interpolation does not provide obstacle
avoidance, and contact response is not a substitute for a clear commanded path.

- `PickUp` accepts `ContactGraspGoal` for the observed pinch pose. It shares
  approach/close/lift command assembly with rigid pickup and requests physical
  contact verification. The routing host verifies material-point lift and slip.
  `PickUpOptions` retains the 48-frame close, 0.35-second closed hold and
  zero lift height. The following `material_lift` invocation performs the
  retraction and lift before the material-point lift validator runs.
- `compute.geometry.cable` owns simulator-independent capsule endpoints,
  material-point observations and TCP-relative slip measurement.
- `task_program/program.yaml` owns the ordered routing and insertion stages.
  `integration.yaml` binds declared material targets and the board geometry.
  `execution_policy.yaml` owns the adaptive Cartesian sampling policy.
  `evaluation.yaml` requires both spool seats, the strict final channel,
  clearance, speed, penetration and the 12 mm episode slip limit.
- The allowlisted `cable_material` service resolves material-coordinate TCP
  targets. Calls lower to `PickUp(ContactGraspGoal)`, `MoveEndEffector` and
  `MoveJoints`; no cable-specific Atomic Skill or environment subclass exists.
  References are captured after the acknowledged close frame, before lift,
  and belong to one bridge/episode and explicit environment rows.
- The deployment still uses the original `RouteTheCable-YAM-v1` ID. Loading
  `task.yam.yaml` registers it as the common `EmbodiedEnv`; importing the task
  package alone does not register a configuration-owned deployment.
- `YamI2rtCfg` resolves the versioned robot bundle from the data registry/cache or an explicit
  `fpath`. Its bundle retains the original MIT license and pinned provenance.
  Simulation and FK use
  the same URDF. Fingertips retain separate convex pieces instead of one hull
  spanning their concave grasp region.

The scene follows the pinned upstream `RodSpec` coefficients, including density,
stretch/shear, bend/twist and contact damping. Shape friction combines by a
geometric mean: cable/pad are 60, bench 0.02, and other scenery 2. The task uses
`rigid_contact_history: true` with `contact_matching: latest`. Both
`rigid_contact_stick_freeze_translation_eps` and
`rigid_contact_stick_freeze_angular_eps` are set to `0.0`. Contact history still
retains the contact multipliers and local sticking anchors. The zero thresholds
disable VBD's body-level world-space motion freeze for dynamic sticking contacts
without a static/kinematic anchor contact. For sufficiently small motion, that
operation would otherwise restore the previous body pose before computing
velocity. Disabling it allows slowly moving gripper proxies and cable segments
to retain their motion while frictional sticking remains active.
These are task-specific solver settings; the strict final channel bounds and
12 mm slip limit remain unchanged.

The routing component increases left-arm drive stiffness for insertion tracking.
The grasp profile specifies finger joint positions, not the physical gap between the
collision pads. At the grasp depth, two 3 mm joint targets give an unloaded
inner pad gap of approximately 4.45 mm for the 7 mm diameter cable; the nominal
target therefore has about 2.55 mm of interference. Actual finger positions
depend on the drive and contact response. The first and third right-hand
grasps and the left-hand grasp use zero-position firm commands through the
existing `SkillControlOverrides` interface; the second right-hand grasp uses
the shared 3 mm targets.
Each pinch holds its grasp command for 0.35 s on the control grid before
lifting. This dwell is part of the Atomic Skill's `close` phase; the material
reference is sampled after closing/settling, before any lift, and is not rebased
during the grasp.

## Validation and recordings

Every grasp must lift the observed material point by more than 8 mm and remain
within 12 mm of its post-closure TCP-relative position. Final metrics include
both spool-seat counts, free-end location, hand clearance, speed and sampled
cable/bench, cable/spool and cable/port penetration. `reference_routed`
reproduces the upstream broad regions. Final `success` additionally requires
the free end inside the strict region
`[0.5845, 0.064, 0.753]` to `[0.5875, 0.0896, 0.759]` m and penetration no greater
than 1 mm. These final bounds and the original lift/slip limits are unchanged.
Sampled SDF diagnostics do not constitute an exact continuous collision-distance
query.

These final cable predicates do not certify the robot's swept path. The
first-spool CUDA regression samples backend robot/spool contacts throughout
approach, placement, release and retreat, rejects penetration over 1 mm, and
also requires the cable to remain seated and the hand to end open above the
spool:

```bash
python -m pytest tests/gym/envs/tasks/test_route_the_cable_cuda.py --run-gpu -m gpu -q
```

The three NoRender CUDA regressions pass on the development build listed above,
using seed 73 and a fresh process per case. The full-route case naturally
exhausts all Task Program segments, validates every stage, then requires
`task_success=True` after release, retreat and two seconds of physical settling.

| Case | Maximum material-point displacement | Reported robot/obstacle penetration |
| --- | --- | --- |
| First-spool release | 0.080 mm | 0 mm |
| Third carry through release and retreat | 0.379 mm | 0 mm |
| Complete route after the main integration | 0.306 mm | 0 mm |

The full route was repeated successfully after integrating main `1955c640`
and finished at endpoint `[0.58459103, 0.07624141, 0.75500649]` m,
inside the unchanged strict channel. Sampled cable/bench, cable/spool and
cable/port penetration were all zero. The complete-route regression checks raw
robot/bench, robot/spool and robot/port contacts after every control step; it also
checks the final bounds independently of the configured evaluator. The original
12 mm displacement and 1 mm penetration limits remain active.

The production deployment inspector and 45 focused CPU cases cover the new
material lift, staging orientation, commanded hold, release-depth gating and
acceptance behavior. Diagnostic suffix fixtures are excluded from complete-route
qualification. These individual fresh-process cases do not establish multi-seed
success rates or repeated-reset reliability.

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
one attempt per process. Reset/episode repeatability with contact history
needs separate physical qualification; API state/reset tests alone do
not prove repeatable frictional routing. Use a fresh process for each trial until
that reset path is qualified. Raw trajectory recordings may include failed
attempts; select training data using the segment and final outcomes.

## Task Program migration qualification

The migration preserves the Newton/DexUni two-way cable solver, scene and robot
contact coefficients. The shared execution policy applies its joint acceleration
budget to all Cartesian moves, so stage durations may be longer than the former
handwritten direct-motion path. The episode budget is 12,000 control steps.

Static composition/catalog preflight and intermediate physical qualification do
not certify a complete cable task. Full acceptance remains strict after release,
retreat and settling. Failed samples retain their measured stopping stage;
changing orchestration does not by itself establish physical insertion success.
