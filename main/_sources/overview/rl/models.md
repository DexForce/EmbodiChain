# Policy Models

This module contains RL policy networks and related model implementations, supporting various architectures and distributional extensions.

## Main Classes and Structure

### Policy
- Abstract base class for RL policies; all policies must inherit from it.
- Unified interface:
    - `get_action(tensordict, deterministic=False)`: Sample actions into a `TensorDict` without gradients.
    - `forward(tensordict, deterministic=False)`: Low-level action/value write path used by policy implementations.
    - `get_value(tensordict)`: Estimate state value into a `TensorDict`.
    - `evaluate_actions(tensordict)`: Return optimization-time policy outputs from a `TensorDict`.
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
- `build_policy(policy_block, obs_space, action_space, device, ...)`: Automatically build policy from config.
- `build_mlp_from_cfg(module_cfg, in_dim, out_dim)`: Automatically build MLP from config.

## Usage Example
```python
actor = build_mlp_from_cfg(actor_cfg, obs_dim, action_dim)
critic = build_mlp_from_cfg(critic_cfg, obs_dim, 1)
policy = build_policy(
    policy_block,
    env.flattened_observation_space,
    env.action_space,
    device,
    actor=actor,
    critic=critic,
)
step_td = TensorDict({"obs": obs}, batch_size=[obs.shape[0]], device=obs.device)
step_td = policy.get_action(step_td)
action = step_td["action"]
log_prob = step_td["sample_log_prob"]
value = step_td["value"]
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
