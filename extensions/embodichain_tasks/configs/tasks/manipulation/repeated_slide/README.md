# Repeated Slide

```bash
conda activate embodichain
embodichain run-task --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_slide/task.ur5.yaml
```

The scene uses the UR5/PGI gripper, drawer pose and handle geometry from the
Atomic Slide sample. Each episode opens the drawer toward the absolute 0.18 m
joint position, then closes it toward 0.0 m. The host restores every object and
the robot to their configured initial states before the next round. Absolute
targets use the freshly observed joint position instead of accumulating a fixed
displacement. The Trapezoidal backend remains selected, while
Slide's Cartesian contact path uses its required `ik_interp` strategy.

The default collection runs three sweeps of four grasp branches and four
trajectory variants: 48 independently recorded episodes in batches of two.
Shared phase permissions expand the approach while keeping the
reach, closure, pull/push and release commands exact. Parameters are selected through
`expansion/open_drawer.yaml` and the shared `atomic_fixed_scene` policy.

```bash
embodichain run-task \
  --gym-config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_slide/task.ur5.yaml \
  --headless --num-envs 2 --max_episodes 2 \
  --expansion-recipe-indices 0 1 \
  --output-dir /tmp/open-drawer/expansion \
  --dataset-dir /tmp/open-drawer/datasets
```

Add `--disable-sensor` for joint/state-only recording. `task.franka.yaml` selects
the existing Franka embodiment; `--physics newton` selects its separately owned
Newton environment. See [the manipulation guide](../README.md) for all skills.
