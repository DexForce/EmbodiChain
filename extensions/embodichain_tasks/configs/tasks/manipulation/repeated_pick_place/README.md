# Repeated Pick and Place

```bash
conda activate embodichain
embodichain run-env --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
```

Each episode repeats Pick/Place three times. UR5, its PGI gripper, initial
pre-pick pose, cube and drop targets follow the Atomic Pickup/Place samples.
The Trapezoidal planner and shared phase permissions expand both Pick and Place.

The default collection contains 16 fixed-scene recipes: four grasp branches
and four trajectory variants, collected in batches of two environments. Scene
and visual randomization are disabled. The recipe selection is cyclic: raising
`--max_episodes` repeats the recipes with a full scene/robot reset and a new
recorded episode after each three-cycle Pick/Place round.
The embodiment supplies the RGB camera, while `env.dataset` owns LeRobot output.
`expansion/repeated_pick_place.yaml` owns the episode schedule and manifest path;
shared expansion parameters live in `components/expansion_policies/atomic_fixed_scene.yaml`.

For a smaller run, including two reset batches:

```bash
embodichain run-env \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --headless --num-envs 2 --max_episodes 4 \
  --expansion-recipe-indices 0 1 4 5 \
  --output-dir /tmp/pick-place/expansion \
  --dataset-dir /tmp/pick-place/datasets
```

Add `--disable-sensor` to collect joints/state without RGB. Explicit recipe
indices must match the requested episode count. `--num-envs` controls parallel
capacity independently of the number of episodes.

`task.franka.yaml` retains the Franka embodiment. `--physics newton` selects
`envs/newton.yaml` for UR5 and Franka. The `task.ur5.objective.yaml` deployment
adds measured physical objectives; ordinary projected completion records the
command trajectory. See [the manipulation guide](../README.md) for all skills,
phase protection, repetition and output behavior.

| Deployment | Robot component | Physics | Configuration |
|---|---|---|---|
| franka | franka_panda | default or newton | `task.franka.yaml` |
| ur5 | ur5_dh_pgi_140_80 | default or newton | `task.ur5.yaml` |
| tianji_marvin | tianji_marvin | default | `task.tianji_marvin.yaml` |

## Tianji Marvin tabletop demo

The Marvin deployment uses `envs/tabletop.default.yaml`, a 750 mm tabletop and
a 50 mm cube. Its left arm makes three transfers between two tabletop targets;
the right arm retains its initial target. Both arms inherit the shared compact
ready pose from the Marvin embodiment. The tabletop program and Marvin
integration select verified Pick/Place effects, with stable-object checks and
a 40 mm final position tolerance after every placement. The 25 mm release
clearance has its own attachment/release hysteresis, so an open hand and the
measured drop onto the support are observed before retracting.

Run without creating a training dataset:

```bash
python -m embodichain run-env \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.tianji_marvin.yaml \
  --headless --viser --renderer hybrid --seed 0 --max_episodes 1 --filter-dataset-saving
```

Use `--no-headless` without `--viser` for the native simulation window. The
shared Marvin embodiment owns the gripper-frame and drive calibration. The
robot preset preserves source inertials. This embodiment explicitly supplies
simulation-calibrated mass/inertia overlays with an assumed 1000× scale of the
pinned URDF entries; its source units are not manufacturer-confirmed. The
simulation retains gravity and physical gripper contacts.

Sensor-enabled runs expose `cam_left_wrist` and `cam_right_wrist`, each with
640×360 RGB and aligned ideal depth. The robot preset owns their
D405-equivalent nominal optics and mesh-estimated mounts; these are simulation
defaults, not hardware calibration. Add `--disable-sensor` to skip acquisition.
Dataset recording remains controlled separately by `--filter-dataset-saving`.
