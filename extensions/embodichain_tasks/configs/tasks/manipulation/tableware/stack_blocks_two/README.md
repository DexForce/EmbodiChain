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
python -m embodichain run-env \
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
