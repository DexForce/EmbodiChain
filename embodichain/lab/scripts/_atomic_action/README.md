# Atomic Skill CLI scenes

Activate the project environment and select a sample scene directly:

```bash
conda activate embodichain
embodichain run-atomic-action --skill place --physics default --robot franka --planner trapezoidal --headless
```

`--skill` is required. Its choices are `pickup`, `place`, `pour`, `press`,
`slide`, `twist`, `handover`, `move_joint`, `axis_align`, and `open_door`.
`--physics` defaults to `default` and also accepts `newton`; `--robot` defaults
to `ur5`; `--planner` defaults to `trapezoidal` and also accepts `toppra` and
`curobo`.

The sample robot presets are `ur5`, `franka`, and `ur10`. Pickup, Place,
MoveJoint, and Handover accept all three presets. Pour, Press, Slide, Twist,
AxisAlign, and OpenDoor require `ur5`. Unsupported combinations fail before
the simulator is constructed.

Scene layouts, skill options, grasp sampling, and expansion settings are
internal sample defaults. The scenes share the implementations used by
`scripts/tutorials/atomic_action/`, including assets, robot mounts/TCPs,
initial poses, lighting, motion options, and replay callbacks. Place and Pour
retain their PickUp precondition. Slide pulls the drawer, executes it, reads
the moved handle, then plans and executes the push. Handover retains the
dual-arm robot and soda-can/support-surface scene. Its UR5, Franka, and UR10
presets mount two Robotiq 2F-140 grippers from
`get_data_path("Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf")`. Handover
uses a matching 140 mm opening grasp model and a 0.4 rad closing angle with
the URDF's mimic directions. Selecting `--physics newton`
applies the examples' scene-specific Newton contact overlays and solver settings.

Franka Handover currently selects numerical PyTorch IK; its single-arm scenes
retain the preset FEP solver. The Handover factory applies
`FRANKA_ROBOTIQ_2F_140_HANDOVER_TCP` to each Franka arm. UR5 and UR10 retain
their analytical UR solvers. These solver choices remain internal sample defaults.

`MotionGenerator` solves both environments and Cartesian waypoints through
`compute_batch_ik`, with bounded chunks to limit numerical multi-start memory.
Solvers that support continuous batch selection receive the initial path seed.
For other solvers, each chunk's endpoint anchors interpolated seeds for the
interior batch; this encourages consistent branches but does not guarantee
continuous IK selection. Every required waypoint remains part of the solve,
and a failed waypoint fails its environment's complete trajectory.

## Fixture TCP calibration

TCP poses are maintained in `tcp_presets.py`, independently for each mounted
robot/gripper combination. The full 4x4 matrix maps the arm's configured end
link to the selected tool point. Translation is in meters in the end link's
local frame; the upper-left 3x3 block is rotation.

| Mounted fixture / tool point | Pose to edit | Existing local-Z offset |
| --- | --- | --- |
| UR5 + PGI, general point (Press/Twist/MoveJoint) | `UR5_PGI_TCP` | 0.17 m |
| UR5 + PGI, manipulation / generic dual helper | `UR5_PGI_GRASP_TCP` | 0.15 m |
| Franka + PGI, single-arm Pickup/Place/MoveJoint | `FRANKA_PGI_TCP` | 0.15 m |
| Franka + PGI, generic dual helper | `FRANKA_PGI_HANDOVER_TCP` | 0.165 m |
| UR5 + Robotiq 2F-140, Handover | `UR5_ROBOTIQ_2F_140_HANDOVER_TCP` | 0.21 m, with a 90-degree local-Z rotation |
| Franka + Robotiq 2F-140, Handover | `FRANKA_ROBOTIQ_2F_140_HANDOVER_TCP` | 0.21 m, with a 90-degree local-Z rotation |
| UR10 + Robotiq 2F-140 | `UR10_ROBOTIQ_2F_140_TCP` | 0.21 m, with its mounting rotation |

For `--skill place --robot franka`, edit `FRANKA_PGI_TCP`. This calibrates the
single-arm Franka/PGI tool without affecting UR5, Robotiq, or the separately
authored handover calibration. Each solver receives an independent matrix.
For `--skill handover --robot franka`, edit
`FRANKA_ROBOTIQ_2F_140_HANDOVER_TCP`; the corresponding UR5 calibration is
`UR5_ROBOTIQ_2F_140_HANDOVER_TCP`. The generic dual helper retains PGI unless
the scene explicitly selects `gripper="robotiq_2f_140"`.
The named tool points preserve the existing sample offsets; they are sample
calibrations, not measured guarantees for another physical fixture.

Sample scenes select named tool points instead of embedding a generic
`tcp_z`. Python helpers still accept explicit PGI `tcp_z` overrides, including
Franka; an override changes local Z while preserving the fixture's rotation
and lateral offsets. Robotiq keeps its own full mounting pose. TCP calibration
changes the FK/IK tool reference, not the gripper mesh or physical length.

## Expansion

The sample defaults produce four affordance branches and four trajectory
variants per branch (16 recipes). MoveJoint has no affordance and produces four
trajectory variants. Variant zero keeps its reference trajectory. Affordance
branch identities select distinct feasible `AntipodalAffordance` grasp candidates
for Pickup, Place, Pour, AxisAlign and Handover; articulation/contact skills use
their native affordance freedom. Candidate pools can be too small to provide
unique grasps; candidate IDs and reuse flags are preserved in the manifest.

Trajectory expansion uses the same production `joint_residual` and `via_points`
operators as repeated-pick-place, with scale `0.005` and deduplication tolerance
`0.002`. Contact/manipulation phases on the button, knob, drawer or door, hand
transitions, phase endpoints, and hand joint columns stay exact. Motion phases
may change between their endpoints; no timing factor changes the replay
callbacks or control clock.

MoveJoint compiles `ready`, `mid`, and `home` as separate invocations. Each
required joint target is an action boundary preserved by control-grid retiming
and both spatial operators; motion between these targets remains editable.

Pour's complete rotation-and-return phase stays exact, including the interior
rotated target. Its plan exposes both legs as one segment without an interior
waypoint boundary, so spatial operators are withheld from that segment.
The preceding PickUp's free phases still produce trajectory variants.
Protected phases are restored from the original command samples after
expansion, including near-identical zero-time joins removed by adaptation.

Slide expands each row's own compiled reference in both pull and push. Push is
replanned from the robot and handle states measured after pull, so rows keep
their own starting commands, locked contact path, arrival intervals, and grasp
provenance even when they share an affordance branch. This also applies when
internal `runtime.hold_steps` is zero.

Recipe index is `affordance_branch * variants_per_family + trajectory_variant`.
`--num-envs` controls parallel capacity; larger selections run in batches.
The default capacity is 16 for affordance scenes and four for MoveJoint.
Every batch rebuilds the original scene. Seeds and branch identities are stable
across batch sizes. A shortage of accepted trajectory variants fails explicitly.

```bash
# One nominal scene, useful for checking assets/planning.
embodichain run-atomic-action --skill place --headless --no-expansion --num-envs 1

# Nominal + both trajectory operators + a different grasp, four parallel rows.
embodichain run-atomic-action --skill place --headless --num-envs 4 --recipe-indices 0 1 2 4

# Use the example's viewer and record replay videos.
embodichain run-atomic-action --skill handover --num-envs 4 --save-video --keep-alive
```

Omitting `--headless` enables the sample viewer. Replay starts automatically;
`--keep-alive` retains the final scene until Ctrl+C. `--viser --headless
--keep-alive` retains browser visualization. `--real-time` adds the original
example's wall-clock playback pacing. `--save-video` enables the example
camera/MP4 recorder; `--renderer no-render` is available for faster checks without
video. Headless alone does not disable rendering.

## Output and qualification

Outputs default to `/tmp/atomic-action/<skill>`; `--output-dir` selects a different
directory. It contains `expansion_manifest.json` and one subdirectory per batch.
Each replay writes a JSON record and, by default, a `.pt` file with `positions`,
`reference_positions`, `dt`, `env_ids`, and recipe provenance. Slide writes
separate pull and push records.

Status `executed` means planning, expansion and simulation playback completed.
As with the original examples, these scenes have no measured physical success
validator; `measured_task_success` remains null. They do not generate LeRobot
datasets. The repeated-pick-place Task Program collection pipeline continues to
own its acceptance, scene/visual expansion, and structured dataset persistence.
