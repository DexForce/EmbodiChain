# Drawer storage Task Program

The catalog offers UR5 with a DH PGI 140/80 gripper and Franka FR3 with a
Panda hand. Both use the same Default-physics scene and four-segment Task
Program. The program opens the rail to 0.26 m, picks the 50 mm cube, places
it in the drawer, releases it, and closes the rail. All movements run
through the shared Task Program and Atomic Skills.

Select `ur5` or `franka` in the config path.

```bash
config=extensions/embodichain_tasks/configs/tasks/manipulation/drawer_storage/task.franka.yaml
python .agents/skills/add-task-program/scripts/inspect_deployment.py "$config"
python -m embodichain run-task --gym-config "$config" \
  --headless --device cpu --renderer no-render --disable-sensor --seed 7
```

The deployments select their own base pose, gripper profile and motion
calibration. The top-down grasp is calibrated to the selected tool frame.
The cube must clear the drawer front during transfer. Franka lifts the
open hand before turning toward the pickup table and approaches the drawer
handle from a clearance waypoint before closing.

The execution policy projects commanded effects. The open rail and lifted
cube have measured segment validators, with measured stability windows after
opening, placement, and closing. After the full program, acceptance requires
all eight cube corners inside the actual drawer frame, a rail position
between -1 mm and 10 mm, measured hand joints within 2 mm of their open
targets, and both finger origins more than 80 mm from the cube. Franka
checks both Panda finger joints and both Panda finger links. The drawer
interior bounds include 1 mm of contact tolerance.

## Franka physical qualification

Default physics on CPU with `dexsim-engine 0.5.1rc1`, NoRender and sensors
disabled. Each seed ran in a fresh process on the fixed scene with one attempt.

| Seeds | Open | Pick | Place and release | Close | Final acceptance | Unintended contacts |
| --- | --- | --- | --- | --- | --- | --- |
| 0–9 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 0 |

The audit checked native positive contact impulses at all 14,380 physics
substeps per seed for every pair involving the robot or cube. Each stage
reported only the expected contacts between fingers and the handle or cube,
and between the cube and its support surface.

All ten runs measured 266.12 mm at the opening validator, 177.70 mm of cube
lift and 3.56 mm of final rail travel from closed. All eight cube corners
passed the drawer-frame bounds. Both Panda fingers opened within tolerance
and remained more than 533 mm from the cube.
