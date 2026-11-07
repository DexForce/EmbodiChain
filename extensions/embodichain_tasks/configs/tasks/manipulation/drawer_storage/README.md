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
opening, placement, and closing. Both deployments select the same
[final acceptance component](task_program/acceptance.yaml). The common
`EmbodiedEnv` remeasures it on every success query and combines it with
program completion. Acceptance requires
all eight cube corners inside the actual drawer frame, a rail position
between -1 mm and 10 mm, measured hand joints within 2 mm of their open
targets, and both finger origins more than 80 mm from the cube. The hand
positions come from the embodiment’s open command preset, and finger link
names come from the embodiment component. The drawer
interior bounds include 1 mm of contact tolerance.

## Physical qualification

Default physics on CPU with `dexsim-engine 0.5.1rc1`, NoRender and sensors
disabled. For each embodiment, seeds 0–9 ran in fresh processes with one
attempt per seed on the fixed scene.

| Embodiment | All four stages and final acceptance | Opening travel (mm) | Cube lift (mm) | Final rail position (mm) | Minimum finger clearance (mm) |
| --- | --- | --- | --- | --- | --- |
| UR5 | 10/10 | 259.36–259.38 | 178.89 | 7.56–7.58 | 545.67 |
| Franka | 10/10 | 266.12 | 177.70 | 3.56 | 533.15 |

In all 20 episodes, the measured cube corners passed the drawer-frame bounds
and the hand opened within tolerance. The Franka contact audit sampled
native positive normal impulses at 143,800 physics substeps for every pair
involving the robot or cube. It found zero unintended contacts. Observed
contacts were limited to fingers touching the handle or cube and the cube
touching its support surface.
