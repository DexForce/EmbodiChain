# Policy Models

This module contains RL policy networks and related model implementations, supporting various architectures and distributional extensions.

## Main Classes and Structure

### Policy
- Abstract base class for RL policies; all policies must inherit from it.
- Unified interface:
    - `forward(tensordict, deterministic=False)`: Write action, log prob, and value into a `TensorDict`.
    - `get_value(tensordict)`: Estimate state value into a `TensorDict`.
    - `evaluate_actions(tensordict)`: Evaluate action probabilities, entropy, and value from a `TensorDict`.
- `get_action(obs, deterministic=False)` is retained as a compatibility layer for evaluation and legacy callers.
- Supports GPU deployment and distributed training.

### ActorCritic
- Typical actor-critic policy, includes actor (action distribution) and critic (value function). Used with PPO.

### ActorOnly
- Actor-only policy without Critic. Used with GRPO (Group Relative Policy Optimization), which estimates advantages via group-level return comparison instead of a value function.
- Supports Gaussian action distributions, learnable log_std, suitable for continuous action spaces.
- Key methods:
    - `forward`: Actor network outputs mean, samples action, and writes policy outputs into a `TensorDict`.
    - `evaluate_actions`: Used for loss calculation in PPO/GRPO algorithms.
- Custom actor/critic network architectures supported (e.g., MLP/CNN/Transformer).

### MLP
- Multi-layer perceptron, supports custom number of layers, activation functions, LayerNorm, Dropout.
- Used to build actor/critic networks.
- Supports orthogonal initialization and output reshaping.

### Factory Functions
- `build_policy(policy_block, obs_dim, action_dim, device, ...)`: Automatically build policy from config.
- `build_mlp_from_cfg(module_cfg, in_dim, out_dim)`: Automatically build MLP from config.

## Usage Example
```python
actor = build_mlp_from_cfg(actor_cfg, obs_dim, action_dim)
critic = build_mlp_from_cfg(critic_cfg, obs_dim, 1)
policy = build_policy(policy_block, obs_dim, action_dim, device, actor=actor, critic=critic)
action, log_prob, value = policy.get_action(obs)
```

## Extension and Customization
- Supports custom network architectures (e.g., CNN, Transformer) by implementing the Policy interface.
- Can extend to multi-head policies, distributional actors, hybrid action spaces, etc.
- Factory functions facilitate config management and automated experiments.

## Practical Tips
- It is recommended to configure all network architectures and hyperparameters for reproducibility.
- Supports multi-environment parallelism and distributed training to improve sampling efficiency.
- Extend the Policy interface as needed for multi-modal input, hierarchical policies, etc.

---
