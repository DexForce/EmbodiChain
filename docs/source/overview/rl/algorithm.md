# RL Algorithms

This module contains the core implementations of reinforcement learning algorithms, including PPO (Proximal Policy Optimization) and GRPO (Group Relative Policy Optimization).

## Main Classes and Functions

### BaseAlgorithm
- Abstract base class for RL algorithms, defining common interfaces such as buffer initialization, data collection, and update.
- Key methods:
  - `initialize_buffer(num_steps, num_envs, obs_dim, action_dim)`: Initialize the trajectory buffer.
  - `collect_rollout(env, policy, obs, num_steps, on_step_callback)`: Collect interaction data.
  - `update()`: Update the policy based on collected data.
- Designed to be algorithm-agnostic; Trainer only depends on this interface to support various RL algorithms.
- Supports multi-environment parallel collection, compatible with Gymnasium/IsaacGym environments.

### PPO
- Mainstream on-policy algorithm, supports Generalized Advantage Estimation (GAE), policy update, and hyperparameter configuration.
- Key methods:
  - `_compute_gae(rewards, values, dones)`: Generalized Advantage Estimation.
  - `collect_rollout`: Collect trajectories and compute advantages/returns.
  - `update`: Multi-epoch minibatch optimization, including entropy, value, and policy loss, with gradient clipping.
- Supports custom callbacks, detailed logging, and GPU acceleration.
- Typical training flow: collect rollout → compute advantage/return → multi-epoch minibatch optimization.
- Supports advantage normalization, entropy regularization, value loss weighting, etc.

### GRPO
- Group Relative Policy Optimization: uses group-level return comparison instead of a Critic network, saving memory.
- **Step-wise returns**: Computes per-step discounted returns \(R_t = r_t + \gamma R_{t+1}\) (reverse accumulation), avoiding causal issues and discount bias for dense-reward Embodied AI tasks.
- **Masked group normalization**: For variable-length sequences (e.g. `truncate_at_first_done`), group mean/std uses only alive peers at each step, avoiding dead envs' zeros dragging down the mean.
- **Optional reference policy**: When `kl_coef > 0`, creates a frozen reference policy for KL regularization (e.g. VLA fine-tuning). When `kl_coef = 0`, no ref policy is created (recommended for from-scratch training like CartPole).
- Key methods:
  - `_compute_step_returns_and_mask(rewards, dones)`: Step-wise discounted returns and valid-step mask.
  - `_compute_step_group_advantages(step_returns, seq_mask)`: Per-step group normalization with masked mean/std.
  - `collect_rollout`: Collect trajectories and compute step-wise advantages.
  - `update`: Multi-epoch minibatch optimization with optional KL penalty.
- Supports both **Embodied AI** (dense reward, from-scratch training) and **VLA** (sparse reward, fine-tuning) modes via `kl_coef` configuration.

### Config Classes
- `AlgorithmCfg`, `PPOCfg`, `GRPOCfg`: Centralized management of learning rate, batch size, clip_coef, ent_coef, vf_coef, and other parameters.
- Supports automatic loading from JSON config files for batch experiments and parameter tuning.
- Can be extended via inheritance for multiple algorithms and tasks.

## Code Example
```python
class BaseAlgorithm:
    def initialize_buffer(self, num_steps, num_envs, obs_dim, action_dim):
        ...
    def collect_rollout(self, env, policy, obs, num_steps, on_step_callback=None):
        ...
    def update(self):
        ...

class PPO(BaseAlgorithm):
    def _compute_gae(self, rewards, values, dones):
        ...
    def collect_rollout(self, ...):
        ...
    def update(self):
        ...
```

## Usage Recommendations
- It is recommended to manage all algorithm parameters via config classes and JSON config files for reproducibility and tuning.
- Supports multi-environment parallel collection to improve sampling efficiency.
- Custom algorithm classes can be implemented to extend new RL methods.
- **GRPO**: Use `actor_only` policy (no Critic). Set `kl_coef=0` for from-scratch training (CartPole, dense reward); set `kl_coef=0.02` for VLA/LLM fine-tuning.

## Extension Notes
- Users can inherit from `BaseAlgorithm` to implement custom algorithms and flexibly integrate them into the RL framework.
- Supports multi-environment parallelism and event-driven extension.
- Typical usage:
```python
algo = PPO(cfg, policy)
buffer = algo.initialize_buffer(...)
for _ in range(num_iterations):
    algo.collect_rollout(...)
    algo.update()
```

---
