# Hand Over

The task transfers a can between two grippers, then places and releases it at
the delivery target. Deployments select the physical scene, reusable
embodiment, Task Program integration and verified motion policy.

| Deployment | Config | Scene |
| --- | --- | --- |
| dual_ur5_dh_pgi_140_80 | `task.dual_ur5_dh_pgi_140_80.yaml` | Existing dual-UR5 scene |
| tianji_marvin | `task.tianji_marvin.yaml` | 750 mm tabletop in front of Marvin |

## Tianji Marvin

The elongated can lies across a narrow pickup stand in the arms' shared
workspace. The stand leaves room below the finger pads, and both arms start
in the shared compact ready pose with lowered elbows. The left arm picks up the can,
and the right arm receives it and delivers it to the other side of the table. Both grippers must
provide measured acquisition/release evidence. After the can settles, the
program checks that its position is within 50 mm of the resting target. A
separate release pose leaves room for the fingers while opening above the table.
Marvin's two arm-chain roots coincide at the torso origin; the handover keeps
the lifted can in the shared pickup region instead of projecting it toward
that origin.

```bash
python -m embodichain run-env \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/hand_over/task.tianji_marvin.yaml \
  --headless --viser --renderer hybrid --seed 0 --max_episodes 1 --filter-dataset-saving
```

Use `--no-headless` without `--viser` for the native window. The shared Marvin
embodiment supplies the calibrated TCPs, gripper commands and joint-position
constraint evidence. Sensor-enabled runs expose the robot preset's
`cam_left_wrist` and `cam_right_wrist`: 640×360 RGB with aligned ideal depth.
These use D405-equivalent nominal optics and mesh-estimated mounts, not
hardware calibration. Use `--disable-sensor` to skip camera acquisition;
`--filter-dataset-saving` disables training-data recording independently.
