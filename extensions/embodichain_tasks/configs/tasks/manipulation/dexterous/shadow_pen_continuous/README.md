# Shadow Hand Continuous Pen Spin

Continuously spin a pen through sequential orientation targets with the 24-DOF Shadow Hand.
Gym ID: `ShadowPenContinuousRL-v1`.

[env.yaml](env.yaml) and [agents/ppo.yaml](agents/ppo.yaml) configure the `default` catalog deployment. The scene loads directly from `ShadowHandPen` (`usd/shadow_pen.usdc`) via `SimulationManager.add_usd()`, running `physics: newton` with `solver_type: dexuni` on GPU MuJoCo Warp CUDA (`use_mujoco_cpu: false`, `use_mujoco_contacts: true`).

```bash
embodichain show-task embodichain_tasks:shadow_pen_continuous
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_continuous/env.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/dexterous/shadow_pen_continuous/agents/ppo.yaml
```

## Evaluation

Evaluate a saved checkpoint with its training run directory:

```bash
embodichain eval-policy outputs/shadow_pen_continuous_ppo/<timestamp> --checkpoint latest --episodes 32 --num-envs 32 --renderer no-render
```

Episode `info["success"]` records whether at least one target milestone was achieved before termination or truncation.
