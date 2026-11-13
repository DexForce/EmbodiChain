# LEAP Hand Cube Reorientation

Reorient a rigid cube in-hand to consecutive 3D target orientations with the 16-DOF LEAP Hand.
Gym ID: `LeapCubeReorientRL-v1`.

[env.yaml](env.yaml) and [agents/ppo.yaml](agents/ppo.yaml) configure the `default` catalog deployment. The scene loads directly from `LeapHandCube` (`usd/M4.usdc`) via `SimulationManager.add_usd()`, running `physics: newton` with `solver_type: dexuni` on GPU MuJoCo Warp CUDA (`use_mujoco_cpu: false`, `use_mujoco_contacts: true`).

```bash
embodichain show-task embodichain_tasks:leap_cube_reorient
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_reorient/env.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_reorient/agents/ppo.yaml
```

## Evaluation

Evaluate a saved checkpoint with its training run directory:

```bash
embodichain eval-policy outputs/leap_cube_reorient_ppo/<timestamp> --checkpoint latest --episodes 32 --num-envs 32 --renderer no-render
```

Episode `info["success"]` records whether at least one target milestone was achieved before termination or truncation.
