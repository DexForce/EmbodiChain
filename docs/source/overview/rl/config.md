# Config

This module defines configuration classes for RL algorithms, centralizing the management of training hyperparameters and supporting automatic loading and experiment reproducibility.

## Main Classes and Structure

### AlgorithmCfg
- Base parameter config class for RL algorithms, supports dataclass-based automation.
- Typical fields:
    - `device`: Training device (e.g., "cuda", "cpu").
    - `learning_rate`: Learning rate.
    - `batch_size`: Batch size per training epoch.
    - `gamma`: Discount factor.
    - `gae_lambda`: GAE advantage estimation parameter.
    - `max_grad_norm`: Gradient clipping threshold.
- Supports inheritance and extension (e.g., PPOCfg adds clip_coef, ent_coef, vf_coef; GRPOCfg adds group_size, kl_coef, truncate_at_first_done).

### Automatic Loading
- Supports automatic parsing of JSON and YAML config files; the main training script injects parameters automatically.
- Decouples config from code, making batch experiments and parameter tuning easier.

## Usage Example
```python
from embodichain.learning.rl.utils import AlgorithmCfg
cfg = AlgorithmCfg(learning_rate=1e-4, batch_size=8192, gamma=0.99)
```
Or via config file:
```json
{
    "algorithm": {
        "name": "ppo",
        "cfg": {
            "learning_rate": 0.0001,
            "batch_size": 8192,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5
        }
    }
}
```

GRPO example (for Embodied AI / from-scratch training):

```json
{
    "algorithm": {
        "name": "grpo",
        "cfg": {
            "learning_rate": 0.0001,
            "n_epochs": 10,
            "batch_size": 8192,
            "gamma": 0.99,
            "clip_coef": 0.2,
            "ent_coef": 0.001,
            "kl_coef": 0,
            "group_size": 4,
            "eps": 1e-8,
            "reset_every_rollout": true,
            "max_grad_norm": 0.5,
            "truncate_at_first_done": true
        }
    }
}
```

- **kl_coef**: Set to `0` for from-scratch training (CartPole, dense reward); use `0.02` for VLA/LLM fine-tuning.
- **group_size**: Number of envs per group for within-group return normalization (must divide `num_envs`).

## Extension and Customization
- Custom algorithm parameter classes are supported for multi-algorithm and multi-task experiments.
- Config classes are seamlessly integrated with the main training script for automated experiments and reproducibility.
- Supports parameter validation, default values, and type hints.

## Practical Tips
- It is recommended to manage all experiment parameters via JSON or YAML config files for reproducibility and tuning.
- Supports multi-algorithm config for easy comparison and automation.

## PPO training options

PPO accepts the following optional settings under `algorithm.cfg`:

| Setting | Default | Effect |
|---|---|---|
| `num_mini_batches` | `null` | Split a rollout into this many minibatches; otherwise use `batch_size`. |
| `use_clipped_value_loss` | `false` | Clip value updates around the values stored during collection. |
| `schedule` / `desired_kl` | `fixed` / `null` | Select `adaptive` with a positive target KL to adjust the learning rate. |
| `minimum_learning_rate` / `maximum_learning_rate` | `1e-5` / `1e-2` | Bound the adaptive learning rate. |
| `normalize_advantage_per_mini_batch` | `false` | Normalize advantages within each minibatch. |
| `action_bound_coef` | `0.0` | Weight the squared penalty on action means outside `[-1, 1]`; use this for tasks with normalized actions. |

Adaptive KL scheduling controls the optimizer learning rate directly and cannot
be combined with `algorithm.cfg.lr_scheduler`.

Time-limit truncations bootstrap rewards with the value stored for the
transition.

For `actor_critic`, `policy.actor_obs_normalization` and
`policy.critic_obs_normalization` enable separate running statistics. Both
are disabled by default. These statistics are included in the policy checkpoint;
recreate the policy with the same normalization settings before loading it.
`policy.initial_action_std` sets exploration scale, and `policy.action_std_range`
bounds the standard deviation. `policy.initial_log_std`, when supplied, takes
precedence over the initial standard deviation.

An environment can expose different actor and critic observation groups:

```yaml
policy:
  name: actor_critic
  obs_groups:
    actor: [policy]
    critic: [critic]
  actor_obs_normalization: true
  critic_obs_normalization: true
```

Keep the actor and critic network definitions alongside these settings. The
collector, rollout buffer, and evaluator preserve the selected groups. Omit
`obs_groups` to flatten all observations as before.

`trainer.save_frequency_updates` saves a checkpoint after each specified number
of PPO updates. Its default is `0` (disabled). If it is supplied, the Gym training
entry defaults `save_freq` to `0`; an explicit `save_freq` also enables saving by
environment steps. Checkpoints include `global_step` and `num_updates`.
Evaluation reports terminal scalar metrics under `eval/metrics/`.
