# ANYmal-C Flat Velocity

Track planar velocity and yaw-rate commands with the ANYmal-C quadruped on flat terrain.
Gym ID: `ANYmalCFlatRL-v1`.

| Deployment | Environment | PPO configuration |
|---|---|---|
| `default` (Default) | [env.yaml](env.yaml) | [agents/ppo.yaml](agents/ppo.yaml) |
| `newton` (Newton) | [env.newton.yaml](env.newton.yaml) | [agents/ppo.newton.yaml](agents/ppo.newton.yaml) |

Each PPO file's `trainer.gym_config` selects its matching deployment. The
catalog defaults to `default`; backend selection belongs to the environment file.

```bash
embodichain show-task embodichain_tasks:anymal_c_flat
embodichain run-env --gym_config embodichain_tasks/configs/tasks/locomotion/velocity/anymal_c_flat/env.yaml
embodichain train-rl --config embodichain_tasks/configs/tasks/locomotion/velocity/anymal_c_flat/agents/ppo.yaml
embodichain train-rl --config embodichain_tasks/configs/tasks/locomotion/velocity/anymal_c_flat/agents/ppo.newton.yaml
```

## Evaluation

Both PPO configurations disable training-time evaluation. Evaluate a saved
checkpoint with its matching training and environment snapshots using the
[headless evaluation example](https://github.com/DexForce/EmbodiChain/blob/main/docs/source/guides/policy_evaluation.md#headless-locomotion-checkpoint-example).

Check command tracking, base contacts, undesired thigh contacts and episode survival.
See [locomotion qualification](https://github.com/DexForce/EmbodiChain/blob/main/docs/source/guides/policy_evaluation.md#locomotion-qualification)
for metric interpretation and required evidence. Terminal success is not
applicable to this task. Catalog qualification remains unavailable until a
checkpoint has been assessed against recorded task criteria.
