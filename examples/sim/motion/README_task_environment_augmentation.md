# Task-environment trajectory augmentation showcase

Run the PickUp → Place task scene with one nominal execution and several
trajectory variants:

```bash
python examples/sim/motion/task_environment_augmentation_showcase.py \
  --output-dir /tmp/embodichain-task-augmentation \
  --num_envs 4 \
  --device cuda \
  --headless
```

The showcase uses the existing Atomic Action task scene and MotionGenerator
planning path. It keeps contact and release waypoints fixed while varying the
authorized free motion using joint residuals, via points, and timing profiles.
The output directory receives the joint-trajectory and tool-path plots plus
``place_auto_play.mp4`` produced by the tutorial runtime.

Useful overrides include:

```bash
--trajectory_variants 4
--spatial_methods joint_residual via_points
--no_redundancy
--variant_seed 7
```

This is a physical execution showcase. It does not claim confirmed dataset
commits; use the generation coordinator and EpisodeSink flow for collection.
