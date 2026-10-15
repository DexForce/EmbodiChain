# Repeated Pick and Place

```bash
conda activate embodichain
embodichain run-env --gym-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
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
  --gym-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml \
  --headless --num-envs 2 --max_episodes 4 \
  --expansion-recipe-indices 0 1 4 5 \
  --output-dir /tmp/pick-place/expansion \
  --dataset-dir /tmp/pick-place/datasets
```

Add `--disable-sensor` to collect joints/state without RGB. Explicit recipe
indices must match the requested episode count. `--num-envs` controls parallel
capacity independently of the number of episodes.

`task.franka.yaml` retains the Franka embodiment. `--physics newton` selects
`envs/newton.yaml` on either deployment. The `task.ur5.objective.yaml` deployment
adds measured physical objectives; ordinary projected completion records the
command trajectory. See [the manipulation guide](../README.md) for all skills,
phase protection, repetition and output behavior.
