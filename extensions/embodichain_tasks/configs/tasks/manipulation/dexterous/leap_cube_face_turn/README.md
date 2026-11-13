# LEAP Hand Cube Face Turn

Turn the top articulated layer of a Rubik-style cube to discrete 90-degree targets with the 16-DOF LEAP Hand.
Gym ID: `LeapCubeFaceTurnRL-v1`.

| Deployment | Environment | PPO configuration |
|---|---|---|
| `newton` (DexUni, default) | [env.newton.yaml](env.newton.yaml) | [agents/ppo.newton.yaml](agents/ppo.newton.yaml) |
| `mjwarp` (MuJoCo Warp) | [env.mjwarp.yaml](env.mjwarp.yaml) | [agents/ppo.mjwarp.yaml](agents/ppo.mjwarp.yaml) |

The scene is loaded directly from LeapHandCubeFaceTurn (`usd/leap_cube_top_layer.usdc`) via `SimulationManager.add_usd()`.
`newton` selects `solver_type: dexuni` and dispatches all scene articulations and free bodies to GPU MuJoCo Warp on CUDA (`use_mujoco_cpu: false`, `use_mujoco_contacts: true`); `mjwarp` selects `solver_type: mujoco_warp` directly.

```bash
embodichain show-task embodichain_tasks:leap_cube_face_turn
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_face_turn/env.newton.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_face_turn/agents/ppo.newton.yaml
```

## Evaluation

Evaluate a saved checkpoint with its training run directory:

```bash
embodichain eval-policy outputs/leap_cube_face_turn_ppo_newton/<timestamp> --checkpoint latest --episodes 32 --num-envs 32 --renderer no-render
```

Episode `info["success"]` records whether at least one target milestone was achieved before termination or truncation.
