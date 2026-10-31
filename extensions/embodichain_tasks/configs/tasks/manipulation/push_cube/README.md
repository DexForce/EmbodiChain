# Push Cube

`env.json` deploys `PushCubeRL` with the default physics backend. Its bundled
`agents/ppo.json` and `agents/grpo.json` each reference this exact environment
configuration through `trainer.gym_config`.

```bash
embodichain show-task embodichain_tasks:push_cube
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/env.json
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/agents/ppo.json
```

The catalog describes configured RL support; it does not claim a trained policy or
physical qualification. No preview or independent physical result is supplied.

## Configuration-defined RL deployment

`env.configured_rl.yaml` deploys `PushCubeConfiguredRL-v1` through the common
`ConfiguredRLEnv`, without a task-specific Python class. It retains the original
robot/TCP, observations, actions and reward formulas. Its measured XY target
condition uses a strict 0.1 m bound; falling below -0.1 m is a failure, whereas
the legacy deployment keeps its truncation behavior.

```bash
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/agents/ppo.configured_rl.yaml
```

The fixed-budget old/new comparison is provided by
`examples/learning/configured_manipulation/compare_push_cube.py`. Training
qualification must use the completed six-run report, not the smoke checks.
