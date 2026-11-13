# Dexterous Manipulation Tasks (USD-Native `EmbodiedEnv`)

All five dexterous manipulation RL tasks construct their runtime simulation scenes exclusively from USD (`.usdc`) via `SimulationManager.add_usd()` on `EmbodiedEnv`. Runtime execution has zero dependency on MJCF `.xml` files; source MJCF files included in the asset archives are retained solely for offline physical-model and tendon parity audits.

| Task key | Gym ID | Archive | Runtime USD | Initial state |
|---|---|---|---|---|
| `shadow_pen_fixed` | `ShadowPenFixedRL-v1` | `ShadowHandPen` | `usd/shadow_pen.usdc` | `ShadowPenSpin-initial.npz` |
| `shadow_pen_continuous` | `ShadowPenContinuousRL-v1` | `ShadowHandPen` | `usd/shadow_pen.usdc` | `ShadowPenSpin-initial.npz` |
| `leap_cube_face_turn` | `LeapCubeFaceTurnRL-v1` | `LeapHandCubeFaceTurn` | `usd/leap_cube_top_layer.usdc` | `LeapCubeFaceTurn-initial.npz` |
| `leap_cube_reorient` | `LeapCubeReorientRL-v1` | `LeapHandCube` | `usd/M4.usdc` | `M4-initial.npz` |
| `leap_cube_rotate_z` | `LeapCubeRotateZAxisRL-v1` | `LeapHandCube` | `usd/M5.usdc` | `M5-initial.npz` |

The asset archives are published under [`DexForceAI/embodichain_data/robot_assets`](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/e1ebe53462d6f48b5961018e625b7c1304e9e83a/robot_assets) and verified by SHA-256 checksums in `embodichain.data.assets.robot_assets`.

## Deployments and CLI Usage

Every task ships two deployments in its catalog:

- `newton` (default, recommended): `env.newton.yaml` + `agents/ppo.newton.yaml`, using `solver_type: dexuni` (`use_mujoco_cpu: false`, GPU MuJoCo Warp CUDA with full scene joint ownership so tendons, fixed-base hands, and free/hinged objects step inside a single CUDA-graphed solver).
- `mjwarp`: `env.mjwarp.yaml` + `agents/ppo.mjwarp.yaml`, using `solver_type: mujoco_warp`.

```bash
embodichain list-task
embodichain show-task embodichain_tasks:shadow_pen_fixed
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_fixed/env.newton.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_fixed/agents/ppo.newton.yaml
embodichain eval-policy runs/<run_dir> --checkpoint latest --num-envs 64 --episodes 64 --renderer no-render
```
