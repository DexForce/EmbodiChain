# Shadow Hand Fixed-Target Pen Spin

Spin a pen in-hand to sampled target orientations with the 24-DOF Shadow Hand.
Gym ID: `ShadowPenFixedRL-v1`.

[env.yaml](env.yaml) and [agents/ppo.yaml](agents/ppo.yaml) configure the `default` catalog deployment. The scene loads directly from `ShadowHandPen` (`usd/shadow_pen.usdc`) via `SimulationManager.add_usd()`, running `physics: newton` with `solver_type: dexuni` on GPU MuJoCo Warp CUDA (`use_mujoco_cpu: false`, `use_mujoco_contacts: true`).

```bash
embodichain show-task embodichain_tasks:shadow_pen_fixed
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_fixed/env.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_fixed/agents/ppo.yaml
```

## Evaluation

Evaluate a saved checkpoint with its training run directory:

```bash
embodichain eval-policy outputs/shadow_pen_fixed_ppo/<timestamp> --checkpoint latest --episodes 32 --num-envs 32 --renderer no-render
```

Episode `info["success"]` records whether at least one target milestone was achieved before termination or truncation.
