# Rollout Buffer

This module implements the data buffer for RL training, responsible for storing trajectory data from agent-environment interactions.

## Main Classes and Structure

### RolloutBuffer
- Used for on-policy algorithms (such as PPO), efficiently stores observations, actions, rewards, dones, values, and logprobs for each step.
- Supports multi-environment parallelism (shape: [T, N, ...]), all data allocated on GPU.
- Structure fields:
  - `obs`: Observation tensor, float32, shape [T, N, obs_dim]
  - `actions`: Action tensor, float32, shape [T, N, action_dim]
  - `rewards`: Reward tensor, float32, shape [T, N]
  - `dones`: Done flags, bool, shape [T, N]
  - `values`: Value estimates, float32, shape [T, N]
  - `logprobs`: Action log probabilities, float32, shape [T, N]
  - `_extras`: Algorithm-specific fields (e.g., advantages, returns), dict[str, Tensor]

## Main Methods
- `add(obs, action, reward, done, value, logprob)`: Add one step of data.
- `set_extras(extras)`: Attach algorithm-related tensors (e.g., advantages, returns).
- `iterate_minibatches(batch_size)`: Randomly sample minibatches, returns dict (including all fields and extras).
- Supports efficient GPU shuffle and indexing for large-scale training.

## Usage Example
```python
buffer = RolloutBuffer(num_steps, num_envs, obs_dim, action_dim, device)
for t in range(num_steps):
    buffer.add(obs, action, reward, done, value, logprob)
buffer.set_extras({"advantages": adv, "returns": ret})
for batch in buffer.iterate_minibatches(batch_size):
    # batch["obs"], batch["actions"], batch["advantages"] ...
    pass
```

## Design and Extension
- Supports multi-environment parallel collection, compatible with Gymnasium/IsaacGym environments.
- All data is allocated on GPU to avoid frequent CPU-GPU copying.
- The extras field can be flexibly extended to meet different algorithm needs (e.g., GAE, TD-lambda, distributional advantages).
- The iterator automatically shuffles to improve training stability.
- Compatible with various RL algorithms (PPO, A2C, SAC, etc.), custom fields and sampling logic supported.

## Code Example
```python
class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_dim, action_dim, device):
        # Initialize tensors
        ...
    def add(self, obs, action, reward, done, value, logprob):
        # Add data
        ...
    def set_extras(self, extras):
        # Attach algorithm-related tensors
        ...
    def iterate_minibatches(self, batch_size):
        # Random minibatch sampling
        ...
```

## Practical Tips
- It is recommended to call set_extras after each rollout to ensure advantage/return tensors align with main data.
- When using iterate_minibatches, set batch_size appropriately for training stability.
- Extend the extras field as needed for custom sampling and statistics.

---
