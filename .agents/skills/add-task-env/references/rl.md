# Reinforcement Learning Task

Use this route when the task should be trainable by PPO, GRPO, APG, or another
registered learning algorithm. EmbodiChain already has an RL system; the first
decision is which environment boundary the prompt requires.

## Route RL environment type

| Requirement | Environment path | Training selector |
|---|---|---|
| Physics simulator, robot, sensors, manager functors | Simulator Gym | `trainer.gym_config` |
| Tensor-only/vectorized dynamics | Lightweight learning env | `trainer.learning_env` |
| Differentiable dynamics or APG | Differentiable lightweight env | `trainer.learning_env` |

Do not use a simulator Gym config for APG. Differentiable algorithms require a
`DifferentiableVecEnv` and are rejected on the simulator path.

## Simulator Gym RL

Use the normal task-first simulator layout and a flat task module that owns
`@register_env`. The subclass may stay thin when manager configuration owns
all task behavior, but the unique simulator task ID still needs import-time
registration.

Register a Python-owned simulator task with explicit RL capability:

```python
@register_env(
    "MyTaskRL",
    max_episode_steps=300,
    supports_rl=True,
)
class MyTaskEnv(EmbodiedEnv):
    ...
```

The runnable Gym deployment owns or selects the physical environment,
embodiment, observations, rewards, actions, events, and randomization. Put
training files under:

```text
<task config>/agents/<algorithm>.yaml
```

Select it from the training config with:

```yaml
trainer:
  gym_config: embodichain_tasks/configs/tasks/<category_path>/<task_name>/<deployment>.yaml
```

Simulator environments use standard rollouts. PPO and GRPO are current
standard examples.

## Lightweight learning RL

Keep the task entry point flat:

```text
embodichain_tasks/embodichain_tasks/<category_path>/<task_name>.py
```

Register the factory with `@register_learning_env` and implement the
`LearningVecEnv` structural contract:

- `num_envs`, `device`, `single_observation_space`, and
  `single_action_space`;
- batched `reset()` and `step()`;
- terminal reward/done for completed rows while auto-resetting those rows to
  their next initial observation;
- `close()`.

For APG or another differentiable algorithm, implement
`DifferentiableVecEnv.detach_state()` as the truncated-backpropagation
boundary. It must detach dynamic state without resetting or resampling the
task.

Select it with:

```yaml
trainer:
  learning_env:
    name: MyTaskRL
    cfg: {}
```

## Training configuration

Every agent config has exact top-level blocks:

- `trainer`: environment selector and runtime/evaluation/checkpoint settings;
- `policy`: registered policy and network configuration;
- `algorithm`: registered algorithm name and algorithm config.

Read values from the current config classes and the closest official example;
do not copy example hyperparameters as global defaults. Keep all algorithms for
the same task in its single `agents/` directory.

Current routing:

- PPO and GRPO: `RolloutKind.STANDARD`;
- APG: `RolloutKind.DIFFERENTIABLE`.

Reference configurations:

- simulator: `classic_control/cart_pole/agents/` and
  `manipulation/push_cube/agents/`;
- lightweight standard/differentiable:
  `classic_control/point_mass/agents/`.

## Validation

Cover the selected boundary:

### Simulator

1. parse the runnable Gym config;
2. prove `@register_env(..., supports_rl=True)` discovery when Python-owned;
3. construct/reset a minimal environment;
4. test observation/action dimensions and reward/termination terms;
5. run trainer routing with the chosen config.

### Lightweight

1. prove `@register_learning_env` discovery;
2. test vector reset/step shapes, devices, async row completion, and auto-reset;
3. test terminal metrics;
4. for differentiable environments, prove reward gradients reach actions and
   `detach_state()` creates the intended TBPTT boundary;
5. run the focused trainer/collector routing tests.

Use `tests/gym/envs/tasks/` for simulator task tests and `tests/learning/`
for lightweight/training tests.
