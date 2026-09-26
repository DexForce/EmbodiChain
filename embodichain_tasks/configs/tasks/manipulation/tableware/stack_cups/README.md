# Stack Cups

`env.json` deploys `StackCups-v1` with the default physics backend. Its environment
checks cup uprightness, horizontal alignment and relative height for success.
The task currently supplies an environment, with no expert demonstration hook or
bundled RL training configuration.

```bash
embodichain show-task embodichain_tasks:stack_cups
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/tableware/stack_cups/env.json
```

Environment availability does not imply a successful stacking controller. No
preview or independent physical qualification result is supplied.
