# Humanoid Run

Run toward a distant target with the torque-controlled Humanoid benchmark.
Gym ID: `HumanoidRun-v1`.

| Deployment | Environment | PPO configuration |
|---|---|---|
| `default` (Default) | [env.yaml](env.yaml) | [agents/ppo.yaml](agents/ppo.yaml) |
| `newton` (Newton) | [env.newton.yaml](env.newton.yaml) | [agents/ppo.newton.yaml](agents/ppo.newton.yaml) |
| `dexuni` (Newton DexUni) | [env.dexuni.yaml](env.dexuni.yaml) | [agents/ppo.dexuni.yaml](agents/ppo.dexuni.yaml) |

Each PPO file's `trainer.gym_config` selects its matching deployment. The
catalog defaults to `default`; backend selection belongs to the environment file.

```bash
embodichain show-task embodichain.extensions.tasks:humanoid
embodichain run-task --gym_config embodichain/extensions/tasks/configs/tasks/classic_control/humanoid/env.yaml
embodichain train-rl --config embodichain/extensions/tasks/configs/tasks/classic_control/humanoid/agents/ppo.yaml
embodichain train-rl --config embodichain/extensions/tasks/configs/tasks/classic_control/humanoid/agents/ppo.newton.yaml
embodichain train-rl --config embodichain/extensions/tasks/configs/tasks/classic_control/humanoid/agents/ppo.dexuni.yaml
```

## Evaluation

The PPO configurations disable training-time evaluation. Evaluate a saved
checkpoint with its matching training and environment snapshots using the
[headless evaluation example](https://github.com/DexForce/EmbodiChain/blob/main/docs/source/guides/policy_evaluation.md#headless-locomotion-checkpoint-example).

Check forward progress, torso height, falls and episode survival.
See [locomotion qualification](https://github.com/DexForce/EmbodiChain/blob/main/docs/source/guides/policy_evaluation.md#locomotion-qualification)
for metric interpretation and required evidence. The DexUni reference
checkpoint at 20,480,000 training transitions completed 900 control steps over
15 seconds at evaluation seed 10042. It traveled 36.44 m forward and maintained
a minimum torso height of 1.15 m during that run.
