# Rollout Buffer

This module implements the data buffer for RL training, responsible for storing trajectory data from agent-environment interactions.

## Main Classes and Structure

### RolloutBuffer
- Used for on-policy algorithms (such as PPO, GRPO), storing a shared rollout `TensorDict` for collector and algorithm stages.
- Supports multi-environment parallelism with rollout batch shape `[N, T + 1]`, all data allocated on GPU.
- Structure fields:
  - `obs`: Flattened observation tensor, float32, shape `[N, T + 1, obs_dim]`
  - `action`: Action tensor, float32, shape `[N, T + 1, action_dim]`
  - `sample_log_prob`: Action log probabilities, float32, shape `[N, T + 1]`
  - `value`: Value estimates, float32, shape `[N, T + 1]`
  - `reward`: Reward tensor, float32, shape `[N, T + 1]`
  - `done`: Done flags, bool, shape `[N, T + 1]`
  - `terminated`: Termination flags, bool, shape `[N, T + 1]`
  - `truncated`: Truncation flags, bool, shape `[N, T + 1]`
  - Algorithm-added fields such as `advantage`, `return`, `seq_mask`, and `seq_return`

The final time index is valid for `obs` and `value`, where it stores the last
observation and bootstrap value. For transition-only fields (`action`, `reward`,
`done`, etc.), the final slot is padding so all rollout fields can share the
same `[N, T + 1]` batch shape.

## Main Methods
- `start_rollout()`: Returns the shared preallocated rollout `TensorDict` for collector write-in.
- `add(rollout)`: Marks the shared rollout as ready for consumption.
- `get(flatten=True)`: Returns the stored rollout after converting it to a
  transition view over the valid first `T` steps.
- `transition_view(rollout, flatten=False)`: Builds a transition-aligned view
  that drops the padded final slot from transition-only fields.
- `iterate_minibatches(rollout, batch_size, device)`: Shared batching utility in `buffer/utils.py`.

## Usage Example
```python
buffer = RolloutBuffer(num_envs, rollout_len, obs_dim, action_dim, device)
rollout = collector.collect(num_steps=rollout_len, rollout=buffer.start_rollout())
buffer.add(rollout)

rollout = buffer.get(flatten=False)
flat_rollout = transition_view(rollout, flatten=True)
for batch in iterate_minibatches(flat_rollout, batch_size, device):
    # batch["obs"], batch["action"], batch["advantage"] ...
    pass
```

## Design and Extension
- Supports multi-environment parallel collection, compatible with Gymnasium-style vectorized environments.
- All tensors are preallocated on device to avoid frequent CPU-GPU copying.
- Algorithm-specific fields are attached directly onto the shared rollout `TensorDict` during optimization.
- The shared minibatch iterator automatically shuffles flattened transition entries for PPO/GRPO style updates.

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
- `value[:, -1]` stores the bootstrap value of the final observation. The final slot of transition-only fields is padding and should be ignored during optimization.
- Use `transition_view()` plus `iterate_minibatches()` instead of duplicating rollout slicing logic in each algorithm.

---
