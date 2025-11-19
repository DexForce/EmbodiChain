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
- Supports inheritance and extension (e.g., PPOCfg adds clip_coef, ent_coef, vf_coef).

### Automatic Loading
- Supports automatic parsing of JSON config files; the main training script injects parameters automatically.
- Decouples config from code, making batch experiments and parameter tuning easier.

## Usage Example
```python
from embodichain.agents.rl.utils import AlgorithmCfg
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

## Extension and Customization
- Custom algorithm parameter classes are supported for multi-algorithm and multi-task experiments.
- Config classes are seamlessly integrated with the main training script for automated experiments and reproducibility.
- Supports parameter validation, default values, and type hints.

## Practical Tips
- It is recommended to manage all experiment parameters via JSON config files for reproducibility and tuning.
- Supports multi-algorithm config for easy comparison and automation.

---
