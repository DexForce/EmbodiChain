# Unitree G1 Rough-Terrain Locomotion

Track planar velocity and yaw-rate commands with the 37-DOF Unitree G1 humanoid across a heightfield terrain curriculum with 187 local elevation samples.
Gym ID: `UnitreeG1RoughRL-v1`.

[env.yaml](env.yaml) and [agents/ppo.yaml](agents/ppo.yaml) configure the `default` catalog deployment on `physics: newton` with `solver_type: dexuni` (`joint_mode: dynamic`, `contact_mode: auto`, `use_mujoco_cpu: false`, `use_mujoco_contacts: false`) and Newton explicit heightfield contact generation (`broad_phase: explicit`, `reduce_contacts: true`, `max_triangle_pairs: 2500000`). Simulation runs 0.005 s outer physics steps, two solver substeps, four outer steps per control interval (50 Hz control), 100 Newton solver iterations, and 50 line-search iterations.

The task loads the 37-DOF `UnitreeG1Rough` robot archive (`g1.usda`) and the `G1RoughTerrains` curriculum (`training/terrain.npz`). The 29-DOF `UnitreeG1Locomotion` flat-terrain asset remains unchanged.

```bash
embodichain show-task embodichain_tasks:g1_rough
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/locomotion/velocity/g1_rough/env.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/locomotion/velocity/g1_rough/agents/ppo.yaml
```

## Evaluation

Evaluate a saved checkpoint with its training run directory:

```bash
embodichain eval-policy outputs/unitree_g1_rough_ppo/<timestamp> --checkpoint latest --num-envs 64 --episodes 64 --renderer no-render
```

Check linear/angular velocity command tracking, upright posture, tilt terminations, terrain curriculum progression, and episode survival. Terminal success is not applicable to this continuous velocity-tracking task.
