# Hand Over

The task transfers a can between two grippers, then places and releases it at
the delivery target. Deployments select the physical scene, reusable
embodiment, Task Program integration and verified motion policy.

| Deployment | Config | Scene |
| --- | --- | --- |
| dual_ur5_dh_pgi_140_80 | `task.dual_ur5_dh_pgi_140_80.yaml` | Existing dual-UR5 scene |
| tianji_marvin | `task.tianji_marvin.yaml` | 750 mm tabletop in front of Marvin |
| dual_franka_robotiq_arg2f_140 | `task.dual_franka_robotiq_arg2f_140.yaml` | Upright can and 500 mm support |

## Tianji Marvin

The elongated can lies across a narrow pickup stand in the arms' shared
workspace. The stand leaves room below the finger pads, and both arms start
in the shared compact ready pose with lowered elbows. The left arm picks up the can,
and the right arm receives it and delivers it to the other side of the table.
The receiver approaches from its side, the sender withdraws after releasing,
and the receiver retracts after opening at delivery. Slower hand interpolation
and an explicit receiving hold reduce contact impulses during transfer. Both grippers must
provide measured acquisition/release evidence. After the can settles, the
program checks that its position is within 50 mm of the resting target. A
separate release pose leaves room for the fingers while opening above the table.
Marvin's two arm-chain roots coincide at the torso origin; the handover keeps
the lifted can in the shared pickup region instead of projecting it toward
that origin.

```bash
python -m embodichain run-task \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/hand_over/task.tianji_marvin.yaml \
  --headless --viser --renderer hybrid --seed 0 --max_episodes 1 --filter-dataset-saving
```

Use `--no-headless` without `--viser` for the native window. The shared Marvin
embodiment supplies simulation-calibrated TCPs, gripper commands and joint-position
constraint evidence. Sensor-enabled runs expose the robot preset's
`cam_left_wrist` and `cam_right_wrist`: 640×360 RGB with aligned ideal depth.
These use D405-equivalent nominal optics and mesh-estimated mounts, not
hardware calibration. Use `--disable-sensor` to skip camera acquisition;
`--filter-dataset-saving` disables training-data recording independently.

## Dual Panda and ARG2F-140

`task.dual_franka_robotiq_arg2f_140.yaml` qualifies the ARG2F-140 HandOver variant
against an upright, uniformly scaled can and a 50 cm support surface. It selects
explicit left/right participants, uses 40-step hand transitions and a compatible
280-sample motion budget, and retains the original source hand configuration.

The final target is a release pose. `delivery_rest_pose` accounts for the can's
source mesh height after release onto the support. Acceptance requires a settled
can within 4 cm of that rest position and its local Y axis aligned with world Z
for at least half a second. The can's source longitudinal axis is Y, so a generic
local-Z upright validator would measure the wrong axis.

Use the corresponding deployment rather than substituting this embodiment into
Marvin's tabletop scene/integration. Its stretched can, pickup stand and fixed
handover target describe a different physical setup. Capability-contract
compatibility alone does not qualify that setup for another robot.

The checked path includes expert execution, semantic effect gates, final physical
acceptance and dynamic action replay. The hardware, contact/TCP calibration and
closure commands remain owned by the selected embodiment/gripper configuration.
