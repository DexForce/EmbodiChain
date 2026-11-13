# LEAP Hand Cube Reorientation

Reorient a rigid cube in-hand to consecutive 3D target orientations with the 16-DOF LEAP Hand.
Gym ID: `LeapCubeReorientRL-v1`.

| Deployment | Environment | PPO configuration |
|---|---|---|
| `newton` (DexUni, default) | [env.newton.yaml](env.newton.yaml) | [agents/ppo.newton.yaml](agents/ppo.newton.yaml) |
| `mjwarp` (MuJoCo Warp) | [env.mjwarp.yaml](env.mjwarp.yaml) | [agents/ppo.mjwarp.yaml](agents/ppo.mjwarp.yaml) |

The scene is loaded directly from LeapHandCube (`usd/M4.usdc`) via `SimulationManager.add_usd()`.
`newton` selects `solver_type: dexuni` and dispatches all scene articulations and free bodies to GPU MuJoCo Warp on CUDA (`use_mujoco_cpu: false`, `use_mujoco_contacts: true`); `mjwarp` selects `solver_type: mujoco_warp` directly.

```bash
embodichain show-task embodichain_tasks:leap_cube_reorient
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_reorient/env.newton.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_reorient/agents/ppo.newton.yaml
```

## Evaluation

Evaluate a saved checkpoint with its training run directory:

```bash
embodichain eval-policy outputs/leap_cube_reorient_ppo_newton/<timestamp> --checkpoint latest --episodes 32 --num-envs 32 --renderer no-render
```

Episode `info["success"]` records whether at least one target milestone was achieved before termination or truncation.
