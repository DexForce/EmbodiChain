# Rollout Buffer

This module implements the data buffer for RL training, responsible for storing trajectory data from agent-environment interactions.

## Main Classes and Structure

### RolloutBuffer
- Used for on-policy algorithms (such as PPO, GRPO), storing a shared rollout `TensorDict` for collector and algorithm stages.
- Supports multi-environment parallelism with rollout batch shape `[N, T]`, all data allocated on GPU.
- Structure fields:
  - `obs`: Flattened observation tensor, float32, shape `[N, T, obs_dim]`
  - `action`: Action tensor, float32, shape `[N, T, action_dim]`
  - `sample_log_prob`: Action log probabilities, float32, shape `[N, T]`
  - `value`: Value estimates, float32, shape `[N, T]`
  - `next.reward`: Reward tensor, float32, shape `[N, T]`
  - `next.done`: Done flags, bool, shape `[N, T]`
  - `next.terminated`: Termination flags, bool, shape `[N, T]`
  - `next.truncated`: Truncation flags, bool, shape `[N, T]`
  - `next.value`: Bootstrap next-state values, float32, shape `[N, T]`
  - Algorithm-added fields such as `advantage`, `return`, `seq_mask`, and `seq_return`

## Main Methods
- `start_rollout()`: Returns the shared preallocated rollout `TensorDict` for collector write-in.
- `add(rollout)`: Marks the shared rollout as ready for consumption.
- `get(flatten=True)`: Returns the stored rollout, optionally flattened over `[N, T]`.
- `iterate_minibatches(rollout, batch_size, device)`: Shared batching utility in `buffer/utils.py`.

## Usage Example
```python
buffer = RolloutBuffer(num_envs, rollout_len, obs_dim, action_dim, device)
rollout = collector.collect(num_steps=rollout_len, rollout=buffer.start_rollout())
buffer.add(rollout)

rollout = buffer.get(flatten=False)
for batch in iterate_minibatches(rollout.reshape(-1), batch_size, device):
    # batch["obs"], batch["action"], batch["advantage"] ...
    pass
```

## Design and Extension
- Supports multi-environment parallel collection, compatible with Gymnasium-style vectorized environments.
- All tensors are preallocated on device to avoid frequent CPU-GPU copying.
- Algorithm-specific fields are attached directly onto the shared rollout `TensorDict` during optimization.
- The shared minibatch iterator automatically shuffles flattened rollout entries for PPO/GRPO style updates.

## Code Example
```python
class RolloutBuffer:
    def __init__(self, num_envs, rollout_len, obs_dim, action_dim, device):
        # Preallocate rollout TensorDict
        ...
    def start_rollout(self):
        # Return shared rollout storage
        ...
    def add(self, rollout):
        # Mark rollout as full
        ...
    def get(self, flatten=True):
        # Consume rollout
        ...
```

## Practical Tips
- The rollout buffer stores flattened RL observations; structured observations should be flattened or encoded before entering this buffer.
- `next.value` is kept for bootstrap convenience, while `next.obs` is intentionally not stored to reduce duplicated memory.
- Use `buffer/utils.py` for shared minibatch iteration instead of duplicating batching logic in each algorithm.

---
