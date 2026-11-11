# Unitree Go1 Flat Velocity

Track planar velocity and yaw-rate commands with the Unitree Go1 quadruped on flat terrain.
Gym ID: `UnitreeGo1FlatRL-v1`.

| Deployment | Environment | PPO configuration |
|---|---|---|
| `newton` (DexUni, recommended) | [env.newton.yaml](env.newton.yaml) | [agents/ppo.newton.yaml](agents/ppo.newton.yaml) |
| `default` (Default) | [env.yaml](env.yaml) | [agents/ppo.yaml](agents/ppo.yaml) |

Each PPO file's `trainer.gym_config` selects its matching deployment. The
catalog defaults to `newton`; backend selection belongs to the environment file.
The Newton deployment uses DexUni with dynamic joints and automatic contact
selection. For this rigid articulation scene, DexUni dispatches to MuJoCo Warp
on CUDA (`use_mujoco_cpu: false`). Its MuJoCo settings are explicit, including
20 solver iterations and 50 line-search iterations. Both deployments preserve
unbounded policy actions with `clip: null`.

```bash
embodichain show-task embodichain_tasks:go1_flat
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/locomotion/velocity/go1_flat/env.newton.yaml
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/locomotion/velocity/go1_flat/agents/ppo.newton.yaml
```

## Evaluation

Both PPO configurations disable training-time evaluation. Evaluate a saved
checkpoint with its matching training and environment snapshots using the
[headless evaluation example](https://github.com/DexForce/EmbodiChain/blob/main/docs/source/guides/policy_evaluation.md#headless-locomotion-checkpoint-example).

Check command tracking, tilt failures, foot clearance and episode survival.
See [locomotion qualification](https://github.com/DexForce/EmbodiChain/blob/main/docs/source/guides/policy_evaluation.md#locomotion-qualification)
for metric interpretation and required evidence. Terminal success is not
applicable to this task. Catalog qualification remains unavailable until a
checkpoint has been assessed against recorded task criteria.
