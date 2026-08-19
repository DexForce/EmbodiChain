# RL Algorithms

This module contains the core implementations of reinforcement learning algorithms, including PPO (Proximal Policy Optimization), GRPO (Group Relative Policy Optimization), and APG (Analytic Policy Gradients).

## Main Classes and Functions

### BaseAlgorithm
- Abstract base class for RL algorithms, defining a single update interface over a collected rollout.
- Key methods:
  - `update(rollout)`: Update the policy based on collected rollout data and return training metrics.
- Declares `rollout_kind` so `train-rl` can route algorithms to the matching trainer:
  - `RolloutKind.STANDARD` (PPO/GRPO) → `Trainer` + shared `[N, T + 1]` rollout `TensorDict`
  - `RolloutKind.DIFFERENTIABLE` (APG) → `DifferentiableTrainer` + graph-preserving `DifferentiableRollout`
- Designed to be algorithm-agnostic; trainers handle collection while algorithms focus on loss computation and optimization.

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

### APG
- Analytic Policy Gradients for **differentiable** environments: backpropagates pathwise gradients from discounted returns through the dynamics graph into policy parameters.
- Requires `RolloutKind.DIFFERENTIABLE`. `train-rl` therefore selects `DifferentiableTrainer` and `DifferentiableCollector` instead of the standard PPO/GRPO path.
- **Segmented discounted returns**: Within each TBPTT segment, computes \(R = \sum_t \gamma^{t} r_t\) per environment (`segmented_discounted_return` / `_discounted_terms`), restarting the discount after `done`.
- **TBPTT accumulation**: `DifferentiableTrainer` collects `segment_length` steps at a time and accumulates gradients until `update_horizon` before one optimizer step. Graph is detached only at segment boundaries.
- **Non-finite safety**: When `skip_nonfinite_updates=True` (default), non-finite losses or gradients skip the optimizer step instead of crashing training.
- Key methods:
  - `begin_update()` / `accumulate_segment(rollout)` / `finish_update()`: Multi-segment pathwise update used by `DifferentiableTrainer`.
  - `update(rollout)`: Convenience wrapper for a single-segment update (`begin` → `accumulate` → `finish`).
  - `segmented_discounted_return(rollout, gamma)`: Standalone helper for per-env discounted returns inside one segment.
- Typical training flow: differentiable collect → accumulate pathwise loss over TBPTT segments → clip grads → optimizer step.
- Does **not** currently support distributed (DDP) training; `build_algo(..., distributed=True)` raises for differentiable algorithms.
- Use an `actor_only` policy (no Critic). PointMass and Newton planar-reach are reference differentiable tasks.

### Config Classes
- `AlgorithmCfg`, `PPOCfg`, `GRPOCfg`, `APGCfg`: Centralized management of learning rate, batch size, clip_coef, ent_coef, vf_coef, gamma, max_grad_norm, and other parameters.
- `APGCfg` adds `ent_coef` and `skip_nonfinite_updates`; TBPTT lengths (`segment_length`, `update_horizon`) live on the trainer config, not the algorithm config.
- Supports automatic loading from JSON or YAML config files for batch experiments and parameter tuning.
- Can be extended via inheritance for multiple algorithms and tasks.

## Code Example
```python
class BaseAlgorithm:
    def update(self, rollout):
        ...

class PPO(BaseAlgorithm):
    def update(self, rollout):
        ...

class APG(BaseAlgorithm):
    def begin_update(self):
        ...
    def accumulate_segment(self, rollout):
        ...
    def finish_update(self):
        ...
```

## Usage Recommendations
- It is recommended to manage all algorithm parameters via config classes and JSON or YAML config files for reproducibility and tuning.
- Supports multi-environment parallel collection to improve sampling efficiency.
- Custom algorithm classes can be implemented to extend new RL methods.
- **GRPO**: Use `actor_only` policy (no Critic). Set `kl_coef=0` for from-scratch training (CartPole, dense reward); set `kl_coef=0.02` for VLA/LLM fine-tuning.
- **APG**: Use a differentiable environment and `actor_only` policy. Prefer `segment_length == update_horizon` for short episodes; shorten `segment_length` when long horizons cause memory pressure. Keep `skip_nonfinite_updates=True` for unstable dynamics.

## Extension Notes
- Users can inherit from `BaseAlgorithm` to implement custom algorithms and flexibly integrate them into the RL framework.
- Set `rollout_kind = RolloutKind.DIFFERENTIABLE` when the algorithm needs an unbroken autograd graph through environment steps.
- Supports multi-environment parallelism and event-driven extension.
- Typical standard-path usage:
```python
algo = PPO(cfg, policy)
rollout = collector.collect(buffer_size, rollout=buffer.start_rollout())
buffer.add(rollout)
algo.update(buffer.get(flatten=False))
```
- Typical differentiable-path usage (orchestrated by `DifferentiableTrainer`):
```python
algo = APG(cfg, policy)
algo.begin_update()
algo.accumulate_segment(segment_rollout)  # repeated until update_horizon
metrics = algo.finish_update()
```
