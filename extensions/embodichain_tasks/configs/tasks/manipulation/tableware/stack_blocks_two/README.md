# Stack Two Blocks

`task.cobotmagic.yaml` implements `StackBlocksTwo-v1` as a configured Task Program
using the shared `EmbodiedEnv`. The CobotMagic right arm picks block 2, then
releases it on block 1. Both blocks retain their reset-time position and yaw
randomization; placement is grounded against the current base-block pose.

The two recorded segments keep separate physical acceptance gates:

- `pick_block_2`: block 2 must rise at least 50 mm from its segment-start pose.
- `place_block_2_on_block_1`: both blocks must settle, block 2 must be within
  25 mm in X/Y and 12 mm in Z of a point 50 mm above block 1, and each block
  must tilt less than 45 degrees from upright.

The pick declares a calibrated object-to-TCP transform: translation
`[0.02, 0, -0.025]` m and the right arm's initial tool rotation. This retains
the lower-edge side grasp and bypasses antipodal sampling, whose vertical
candidates are not reachable in this scene. The transform follows each
block's randomized yaw.

The shared execution policy uses 90 motion samples. `PickUpOptions` owns the
45-step grasp dwell; all commands and settling holds pass through ordinary
`env.step()`. No task-specific dynamics clearing is applied. Internal semantic
effects are projected; the segment validators measure physical acceptance.

Run from the repository root with the project's simulation Python environment:

```bash
python -m embodichain run-task \
  --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task.cobotmagic.yaml
```

The runnable file selects the physical scene, shared robot/sensors, program,
integration, and execution policy. Static deployment inspection checks their
schemas and contracts; physical qualification additionally requires execution
and successful measured segment results.

Local Default-backend checks with sensors disabled passed expert execution and
dynamic action replay for seed 0. Seeds 1 and 2 also fail in the former handwritten
expert: seed 1 has an unreachable randomized pickup path, while seed 2 fails to
lift the block. The configured runtime rejects seed 2 earlier when contact moves
the target during approach. The existing randomization ranges and physical
acceptance thresholds remain unchanged; this task does not claim reliable success
across those ranges.

The dual-UR5, dual-UR10 and Tianji Marvin deployments reuse this same program,
reset distribution and measured acceptance gates. Their task-local integrations
select the corresponding calibrated grasp and dual-arm policy. See the
[deployment matrix](../../../../../README.md#dual-arm-task-deployments) for
configuration selection and sensor differences.

The Tianji deployment selects the left arm and a Toppra motion policy. Its root
position `[0.125, -0.1, 0.125]` m keeps the fixed stand behind the tabletop, with
at least 15 mm of mesh clearance checked by the native regression. The grasp
inclines 30 degrees about the object's Y axis and uses the object-local TCP
translation `[-0.02, -0.003, 0.034641]` m, centering the jaw contact line while
keeping the wrist reachable from this position. Placement restores the authored
upright orientation; the 25 mm XY, 12 mm Z and 45-degree upright acceptance
limits remain unchanged.
