# RL Algorithms

This module contains the core implementations of reinforcement learning algorithms, including PPO (Proximal Policy Optimization) and GRPO (Group Relative Policy Optimization).

## Main Classes and Functions

### BaseAlgorithm
- Abstract base class for RL algorithms, defining a single update interface over a collected rollout.
- Key methods:
  - `update(rollout)`: Update the policy based on a shared rollout `TensorDict`.
- Designed to be algorithm-agnostic; `Trainer` handles collection while algorithms focus on loss computation and optimization.
- Consumes a shared `[N, T + 1]` rollout `TensorDict` and typically converts it to a transition-aligned view over the valid first `T` steps before optimization.

### PPO
- Mainstream on-policy algorithm, supports Generalized Advantage Estimation (GAE), policy update, and hyperparameter configuration.
- Key methods:
  - `compute_gae(rollout, gamma, gae_lambda)`: Generalized Advantage Estimation over a shared rollout `TensorDict`, using `value[:, -1]` as the bootstrap value and ignoring the padded final transition slot.
  - `update(rollout)`: Multi-epoch minibatch optimization, including entropy, value, and policy loss, with gradient clipping.
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
  - `update(rollout)`: Multi-epoch minibatch optimization with optional KL penalty.
- Supports both **Embodied AI** (dense reward, from-scratch training) and **VLA** (sparse reward, fine-tuning) modes via `kl_coef` configuration.

### Config Classes
- `AlgorithmCfg`, `PPOCfg`, `GRPOCfg`: Centralized management of learning rate, batch size, clip_coef, ent_coef, vf_coef, and other parameters.
- Supports automatic loading from JSON config files for batch experiments and parameter tuning.
- Can be extended via inheritance for multiple algorithms and tasks.

## Code Example
```python
class BaseAlgorithm:
    def update(self, rollout):
        ...

class PPO(BaseAlgorithm):
    def update(self, rollout):
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
rollout = collector.collect(buffer_size, rollout=buffer.start_rollout())
buffer.add(rollout)
algo.update(buffer.get(flatten=False))
```

---
