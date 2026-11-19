# LEAP Hand Cube Face Turn

Turn the top articulated layer of a Rubik-style cube to discrete 90-degree targets with the 16-DOF LEAP Hand.
Gym ID: `LeapCubeFaceTurnRL-v1`.

[env.yaml](env.yaml) and [agents/ppo.yaml](agents/ppo.yaml) configure the `default` catalog deployment. The scene loads directly from `LeapHandCubeFaceTurn` (`usd/leap_cube_top_layer.usdc`) via `SimulationManager.add_usd()`, running `physics: newton` with `solver_type: dexuni` on GPU MuJoCo Warp CUDA (`use_mujoco_cpu: false`, `use_mujoco_contacts: true`).

```bash
embodichain show-task embodichain_tasks:leap_cube_face_turn
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_face_turn/env.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/leap_cube_face_turn/agents/ppo.yaml
```

## Evaluation

Evaluate a saved checkpoint with its training run directory:

```bash
embodichain eval-policy outputs/leap_cube_face_turn_ppo/<timestamp> --checkpoint latest --episodes 32 --num-envs 32 --renderer no-render
```

Episode `info["success"]` records whether at least one target milestone was achieved before termination or truncation.
