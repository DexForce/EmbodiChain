# Atomic Skill CLI scenes

Run a fixed-scene reference with one environment, or request the total number
of executions including that reference:

```bash
embodichain run-atomic-action --skill place
embodichain run-atomic-action --skill place --variants 8
embodichain run-atomic-action --skill place --variants 8 --expansion affordance
embodichain run-atomic-action --skill place --variants 8 --expansion trajectory
```

The twelve scenes are `pickup`, `place`, `pour`, `press`, `slide`, `twist`,
`handover`, `move_joint`, `axis_align`, `open_door`, `move_held_object`, and
`push_object`. Defaults are `--robot ur5 --physics default --planner trapezoidal`.
Pickup, Place, MoveJoint, and Handover also accept `franka` and `ur10`; other
scenes currently require UR5. Unsupported selectors fail before simulation.

`--variants N` defaults to one and creates exactly N parallel environments.
Variant zero retains the reference. There is no collection loop or implicit
batching. `--expansion both` is the default for N > 1: each added execution uses
the factors supported by its sample, without creating a Cartesian product.
MoveJoint and PushObject support trajectory expansion only; explicitly selecting
`affordance` for either is an error. `--num-envs` remains a compatibility check
and must equal N; `--no-expansion` requires N = 1. The old fixed-bank
`--recipe-indices` selector has been removed.

## Fixed-scene expansion

Affordance sampling happens before planning. The CLI uses one upstream approach
direction per antipodal pair, so trajectory-only runs retain the same actual
reference grasp. Distinctness is checked on selected affordance geometry in its
reference frame, including comparisons with earlier non-reference variants. Trajectory expansion transforms
each Atomic ActionPlan before compilation and rebuilds its commands, velocities,
and phase metadata together. Both this host and Task Program use the shared
Atomic plan adapter and the production `motion.expansion` operators.

Non-reference trajectory variants alternate `joint_residual` and `via_points`,
with normalized scale 0.005 and deduplication tolerance 0.002. Each logical
variant has an independent seeded stream; physical row numbers do not seed it.
Only explicitly permitted free phases and moving, bound motion joints may
change. Contact paths, hand columns, phase endpoints, required intermediate
targets, stationary participants, and the original time grid remain exact.
Insufficient accepted candidates fail instead of repeating the reference.

Preparation participates in expansion. Place and Pour include PickUp;
MoveHeldObject includes a PickUp followed by object-pose transport. Pour's tilt,
dwell, and return remain protected, so its trajectory changes come from PickUp.
Handover only varies its empty-hand pickup approach and preserves the final
reach, held-object transport, receiving arm, hand transitions, and placement.
Slide executes a pull and then replans the push from each row's measured state.

Every environment uses the same sample's physical initial scene. Object pose,
size, asset, and scene-family generalization belong to benchmark hosts.

## Playback and outputs

```bash
embodichain run-atomic-action --skill handover --variants 2 --save-video --headless
embodichain run-atomic-action --skill place --variants 4 --viser --headless --keep-alive
```

Omitting `--headless` opens the native sample viewer. Replay starts automatically;
`--keep-alive` retains the final scene until Ctrl+C. `--real-time` adds wall-clock
pacing without changing the simulated clock. `--renderer no-render` is available
when video is disabled. Headless alone does not disable rendering. Video framing
includes all parallel environments. Recording cadence is reduced for long
trajectories to keep the entire replay within the existing 2 GB frame buffer;
the actual FPS is recorded in the replay JSON.

Outputs default to `/tmp/atomic-action/<skill>`; `--output-dir` selects another
directory. Manifest schema 2 records the resolved configuration, actual factor
selection, and each replay's stages and per-variant provenance. Each replay has
JSON and trajectory `.pt` files, plus MP4 when `--save-video` is selected. Slide
writes separate pull and push replays.

The trajectory file contains executed `positions`, `velocities`, and `dt`, plus
`planned_positions`, `planned_dt`, `reference_positions`, `reference_dt`,
`env_ids`, stages, and explicit hold durations. Extra Newton contact settling
and final holds are included in the executed clock. Source zero-time joins
retain their command and callback indices and do not advance physics.

Status `executed` means planning, expansion, and simulation playback completed;
`measured_task_success` remains null. These demonstrations do not certify
physical success or produce datasets. Failures and interruptions persist their
reason and release simulation/recording resources.

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
