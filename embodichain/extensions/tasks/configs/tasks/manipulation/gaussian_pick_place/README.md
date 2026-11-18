# Pick and Place in captured Gaussian scenes

One configuration-defined Task Program has three FR3 deployments. The common
`EmbodiedEnv` executes `Pick -> Place -> wait_stable` against an independent
dynamic 5 cm / 50 g cube and static support. Both `cam_third` and the camera
mounted on `FrankaPanda/fr3_hand`, `cam_wrist`, provide 960x540 RGB. No task
Python environment subclass is required.

| Deployment | Existing Gym ID | Registered background |
| --- | --- | --- |
| `task.office.yaml` | `FrankaGSPickPlace-Office-v1` | `RoboSplatterOfficeGS` |
| `task.discoverse.yaml` | `FrankaGSPickPlace-Discoverse-v1` | `DiscoverseLab3GS` |
| `task.robosimgs.yaml` | `FrankaGSPickPlace-Robosimgs-v1` | `RoboSimGSTabletopGS` |

```bash
embodichain show-task gaussian_pick_place
embodichain list-task --category manipulation
embodichain run-task \
  --gym-config embodichain/extensions/tasks/configs/tasks/manipulation/gaussian_pick_place/task.office.yaml \
  --headless --seed 42 --max_episodes 2 --record_trajectory \
  --trajectory_save_dir outputs/gaussian_pick_place/office
```

Replace `office` with `discoverse` or `robosimgs` for the other deployments.
The same official configuration references resolve from an installed wheel and
an unrelated working directory. Loading a deployment registers its existing
Gym ID; task discovery does not require a live simulator or asset downloads.

Environments own Default/CPU physics, the scene and settling. All three
deployments select `franka_gs_dual_camera`, which owns the common robot/hand,
wrist calibration and complete camera suite. Each deployment declares its
existing third-person intrinsics/extrinsics through
`embodiment.camera_overrides.cam_third`. The shared
`franka_gs_verified` execution policy retains the original single-arm contract,
timed tracking and verified effects. Program targets and acceptance remain
scene-specific; the common integration binds only the cube and support.

The source Gaussian packages are visual-only and supply no scanned collision
geometry. Task acceptance requires lift >=4 cm, final position error <=2 cm,
tilt <=15 degrees, measured release and stable placement. These deployments
qualify RGB and the independent task physics; metric Gaussian depth remains
unqualified. Captured lighting is baked. Preserve the packages' upstream
licensing/provenance declarations, including the unspecified DISCOVERSE model
license.

The [recording example](../../../../../../../examples/gym/gaussian_pick_place/README.md)
loads these authoritative definitions, writes portable configuration snapshots,
and records complete paired RGB, native trajectories and capture metadata.
Media spooling, video export and offline/negative acceptance remain example
workflows. Their task IDs, camera UIDs and native trajectory layout are retained.
