# RL Training Framework Guide

TensorDict-based RL framework supporting standard PPO and asynchronous VLA training.

---

## Quick Start

### Configuration

```json
{
  "trainer": {
    "buffer_size": 2048,
    "model_type": "standard"  // or "vla"
  },
  "policy": {"name": "actor_critic"},
  "algorithm": {
    "name": "ppo",
    "cfg": {
      "learning_rate": 3e-4,
      "gamma": 0.99,
      "n_epochs": 10,
      "batch_size": 64
    }
  }
}
```

### Run Training

```bash
python embodichain/agents/rl/train.py --config configs/agents/rl/my_config.json
```

---

## Architecture

```
Trainer → Collector (sync/async) → Buffer (standard/vla) → Algorithm (PPO)
```

**Components**:
- **Collector**: Gather data from environment (SyncCollector / AsyncCollector)
- **Buffer**: Store transitions (RolloutBuffer / VLABuffer)
- **Algorithm**: Update policy (PPO)
- **Trainer**: Coordinate training loop

---

## Training Modes

### Standard Mode (Default)

**For**: Normal models (<100ms inference/step)

```
SyncCollector → Collect 2048 steps → Train → Clear buffer → Repeat
```

**Config**: `{"trainer": {"model_type": "standard"}}`

**Pros**: Simple, stable, low memory, no staleness

### VLA Async Mode

**For**: Large models (>1 sec inference/step)

```
Background: AsyncCollector → Continuously collect → VLABuffer
Main:       Wait for buffer full → Train → Repeat
```

**Config**: `{"trainer": {"model_type": "vla"}}`

**Pros**: 2-3x speedup via parallel collection  
**Cons**: Data staleness, higher memory

---

## Collectors

### SyncCollector

Collects complete rollout synchronously:

```python
from embodichain.agents.rl.collector import SyncCollector

collector = SyncCollector(env, policy, device, callback)
rollout = collector.collect(num_steps=2048)  # [T, N, ...]
```

### AsyncCollector

Runs in background thread:

```python
from embodichain.agents.rl.collector import AsyncCollector

collector = AsyncCollector(env, policy, buffer, device, callback)
collector.start()   # Begin background collection
# ... buffer fills automatically ...
collector.stop()    # Stop collection
```

---

## Buffers

### RolloutBuffer (Standard)

Single-use buffer:

```python
from embodichain.agents.rl.buffer import RolloutBuffer

buffer = RolloutBuffer(buffer_size=2048, device=device)
buffer.add(rollout)  # [T, N, ...]
data = buffer.get(flatten=True)  # [T*N, ...], auto-clears
```

### VLABuffer (Async)

Circular FIFO buffer:

```python
from embodichain.agents.rl.buffer import VLABuffer

buffer = VLABuffer(buffer_size=4096, device=device)
buffer.add(transition)  # Single step
data = buffer.get(flatten=True)  # [buffer_size, ...] when full
```

**Circular behavior**: `[T0,T1,T2,T3]` → add T4 → `[T4,T1,T2,T3]` (T0 overwritten)

---

## VLA Integration

### 1. Implement Model

```python
class MyVLAModel(nn.Module):
    def forward(self, obs: TensorDict) -> TensorDict:
        # Add 'action', 'sample_log_prob', 'value'
        ...
    def get_value(self, obs: TensorDict) -> TensorDict:
        # Add 'value'
        ...
    def evaluate_actions(self, obs: TensorDict) -> TensorDict:
        # Add 'sample_log_prob', 'entropy', 'value'
        ...
```

### 2. Implement Loading

Edit `embodichain/agents/rl/models/vla_policy.py`:

```python
def load_vla_model(model_path, model_class, model_config, device):
    model = MyVLAModel(**model_config)
    model.load_state_dict(torch.load(model_path))
    return model.to(device)
```

### 3. Configure

```json
{
  "trainer": {"model_type": "vla"},
  "policy": {
    "name": "vla",
    "vla_config": {
      "model_path": "checkpoints/vla.pt",
      "model_class": "MyVLAModel",
      "model_config": {}
    }
  }
}
```

---

## Common APIs

### Trainer

```python
from embodichain.agents.rl.utils import Trainer

trainer = Trainer(
    policy, env, algorithm,
    buffer_size=2048,
    model_type="standard",  # or "vla"
    ...
)
trainer.train(total_timesteps=1000000)
```

### Buffer Methods

```python
buffer.add(data)            # Add data
data = buffer.get(flatten=True)  # Retrieve data
buffer.is_full()            # Check ready status
buffer.clear()              # Clear buffer
buffer.get_stats()          # Statistics
```

### Algorithm

```python
from embodichain.agents.rl.algo import PPO, PPOCfg

algorithm = PPO(PPOCfg(...), policy)
losses = algorithm.update(rollout)  # Returns loss dict
```

---

## FAQ

**Q: When use VLA mode?**  
A: Inference >100ms/step AND GPU training fast

**Q: Buffer size?**  
A: Standard: 2048-4096 (rollout size). VLA: 2048-4096 (buffer capacity)

**Q: Data staleness impact?**  
A: Minor. PPO robust to staleness. 2-3x speedup >> small penalty

**Q: Debug data flow?**  
A: `buffer.get_stats()` or `_print_tensordict_tree(rollout)` in ppo.py

---

## Workflows

### Standard

```python
collector = SyncCollector(env, policy, device, callback)
while step < total:
    rollout = collector.collect(num_steps=2048)
    buffer.add(rollout)
    data = buffer.get(flatten=True)
    losses = algorithm.update(data)
```

### VLA

```python
collector = AsyncCollector(env, policy, buffer, device, callback)
collector.start()
while step < total:
    while not buffer.is_full():
        time.sleep(0.1)
    data = buffer.get(flatten=True)
    losses = algorithm.update(data)
collector.stop()
```

---

## File Structure

```
embodichain/agents/rl/
├── train.py              # Entry point
├── algo/ppo.py          # PPO algorithm
├── buffer/
│   ├── standard_buffer.py  # RolloutBuffer
│   └── vla_buffer.py       # VLABuffer
├── collector/
│   ├── base.py             # BaseCollector
│   ├── sync_collector.py   # SyncCollector
│   └── async_collector.py  # AsyncCollector
├── models/
│   ├── actor_critic.py     # Standard policy
│   └── vla_policy.py       # VLA wrapper
└── utils/trainer.py     # Training coordinator
```

---

## References

- [TensorDict Docs](https://pytorch.org/tensordict/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- Example configs: `configs/agents/rl/`
