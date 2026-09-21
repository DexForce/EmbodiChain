# Push Cube

`env.json` deploys `PushCubeRL` with the default physics backend. Its bundled
`agents/ppo.json` and `agents/grpo.json` each reference this exact environment
configuration through `trainer.gym_config`.

```bash
embodichain show-task embodichain_tasks:push_cube
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/push_cube/env.json
embodichain train-rl --config embodichain_tasks/configs/tasks/manipulation/push_cube/agents/ppo.json
```

The catalog describes configured RL support; it does not claim a trained policy or
physical qualification. No preview or independent physical result is supplied.
