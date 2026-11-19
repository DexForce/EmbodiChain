# LEAP Hand Cube Z-Axis Rotation

Continuously rotate a rigid cube around the world Z-axis in-hand with the 16-DOF LEAP Hand.
Gym ID: `LeapCubeRotateZAxisRL-v1`.

[env.yaml](env.yaml) and [agents/ppo.yaml](agents/ppo.yaml) configure the `default` catalog deployment. The scene loads directly from `LeapHandCube` (`usd/M5.usdc`) via `SimulationManager.add_usd()`, running `physics: newton` with `solver_type: dexuni` on GPU MuJoCo Warp CUDA (`use_mujoco_cpu: false`, `use_mujoco_contacts: true`).

```bash
embodichain show-task embodichain_tasks:leap_cube_rotate_z
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_rotate_z/env.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_rotate_z/agents/ppo.yaml
```

## Evaluation

Evaluate a saved checkpoint with its training run directory:

```bash
embodichain eval-policy outputs/leap_cube_rotate_z_ppo/<timestamp> --checkpoint latest --episodes 32 --num-envs 32 --renderer no-render
```

Continuous rotation has no binary target threshold; `info["success"]` is omitted from episode evaluation.
