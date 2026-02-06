# RL Training Framework

## Overview

Modern **TensorDict-based** RL training framework supporting standard PPO, asynchronous VLA training, and pretrained VLA model fine-tuning.

**Key Features**:
- Pure TensorDict data flow
- Dual modes: Standard synchronous / VLA asynchronous
- Efficient buffers: Single-use / Pre-allocated circular
- VLA model integration: Load and fine-tune pretrained VLA models

---

## Quick Start

### 1. Configuration

```json
{
  "trainer": {
    "buffer_size": 2048,
    "buffer_type": "standard",  // "standard" or "vla"
    "iterations": 500
  },
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

### 2. Run Training

```bash
python embodichain/agents/rl/train.py --config configs/agents/rl/my_config.json
```

---

## Training Modes

### Standard Mode (Default)

**Use Case**: Regular model training

```
Collect data (2048 steps) → Train model → Clear buffer → Repeat
```

**Configuration**:
```json
{"trainer": {"buffer_type": "standard"}}
```

**Characteristics**: Simple, stable, low memory usage

---

### VLA Async Mode

**Use Case**: Large models with slow inference (e.g., VLA models, >1 sec/step)

```
Background Thread: Continuously collect data → Write to buffer
Main Thread:       Wait for buffer full → Train model → Repeat
```

**Configuration**:
```json
{"trainer": {"buffer_type": "vla"}}
```

**Characteristics**:
- ✅ Parallel collection & training, 2-3x speedup
- ✅ Pre-allocated memory, optimized for high-frequency writes
- ⚠️ Slightly stale data (acceptable for on-policy algorithms)

---

## Buffer Explanation

### RolloutBuffer (Standard)

- **Storage**: One complete rollout [T, N, ...]
- **Behavior**: Add → Train once → Clear
- **Usage**: Standard PPO

### VLABuffer (Async)

- **Storage**: Circular buffer [buffer_size, ...]
- **Behavior**: Incremental add → Train when full → Old data overwritten
- **Usage**: VLA async collection

**Circular Overwrite Example** (capacity=4):
```
[T0, _, _, _] → [T0,T1, _, _] → [T0,T1,T2, _] → [T0,T1,T2,T3] (full)
→ [T4,T1,T2,T3] (T0 overwritten) → [T4,T5,T2,T3] (T1 overwritten)
```

---

## Core API

### Trainer

```python
from embodichain.agents.rl.utils import Trainer

trainer = Trainer(
    policy, env, algorithm,
    buffer_size=2048,
    buffer_type="standard",  # or "vla"
    batch_size=64,
    ...
)
trainer.train(total_timesteps=1000000)
```

### Buffer Interface

```python
# Add data
buffer.add(rollout)      # Standard mode: complete rollout
buffer.add(transition)   # VLA mode: single transition

# Get data
data = buffer.get(flatten=True)  # Returns [batch, ...]

# Check status
if buffer.is_full():
    train()
```

---

## FAQ

### When to use VLA mode?

Use VLA mode when inference time > 100ms/step and GPU training is fast.

### How to set buffer capacity?

- Standard mode: `buffer_size` = steps per rollout (typically 2048)
- VLA mode: `buffer_size` = circular buffer capacity (recommended 2048-4096)

### Will data be stale in async mode?

Yes, slightly stale (up to buffer_size steps), but acceptable for PPO and other on-policy algorithms. Performance gain far outweighs staleness cost.

---

## VLA Model Integration

### Overview

The framework supports loading and fine-tuning pretrained Vision-Language-Action (VLA) models. VLA models are loaded from checkpoints and wrapped in `VLAPolicy` to conform to the standard Policy interface.

### VLA Model Requirements

VLA model developers should implement a model class with the following interface:

```python
class MyVLAModel(nn.Module):
    def forward(self, observations: TensorDict) -> torch.Tensor:
        """Generate actions from observations.
        
        Args:
            observations: TensorDict with keys like "rgb", "depth", "proprio", "language"
        Returns:
            Action tensor [B, action_dim]
        """
        
    def get_value(self, observations: TensorDict) -> torch.Tensor:
        """Get value estimate.
        
        Returns:
            Value tensor [B, 1]
        """
        
    def evaluate_actions(
        self, 
        observations: TensorDict, 
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy.
        
        Returns:
            (log_prob [B,], entropy [B,])
        """
```

See [vla_policy.py](../embodichain/agents/rl/models/vla_policy.py) for detailed interface documentation (`VLAModelInterface`).

### Configuration Example

```json
{
  "trainer": {
    "buffer_type": "vla",
    "buffer_size": 2048,
    ...
  },
  "policy": {
    "name": "vla",
    "action_dim": 7,
    "vla_config": {
      "model_path": "checkpoints/pretrained_vla_model.pth",
      "model_class": "vla_models.GPTVLAModel",
      "model_config": {
        "vision_encoder": "resnet50",
        "language_model": "gpt2-medium",
        "freeze_vision_encoder": false
      }
    }
  },
  "algorithm": {
    "name": "ppo",
    "cfg": {
      "learning_rate": 1e-5,
      ...
    }
  }
}
```

See [vla_example/train_config.json](../configs/agents/rl/vla_example/train_config.json) for complete example.

### Implementation Guide for VLA Team

1. **Implement VLA Model Class**: Create a model class conforming to `VLAModelInterface`
2. **Implement Checkpoint Loading**: Implement `load_vla_model()` function in [vla_policy.py](../embodichain/agents/rl/models/vla_policy.py)
3. **Test Integration**: Use example config to verify model loads and trains correctly

The `load_vla_model()` function is currently a placeholder that raises `NotImplementedError` - VLA team should implement actual loading logic.

---

## File Structure

```
embodichain/agents/rl/
├── train.py                 # Entry point
├── algo/ppo.py             # PPO algorithm
├── buffer/
│   ├── standard_buffer.py  # RolloutBuffer
│   └── vla_buffer.py       # VLABuffer
├── models/                 # Policy definitions
│   ├── policy.py           # Policy base class
│   ├── actor_critic.py     # Standard ActorCritic (from scratch)
│   ├── vla_policy.py       # VLA model wrapper (pretrained)
│   └── ...
└── utils/
    ├── trainer.py          # Training coordinator
    └── async_collector.py  # Async data collector
```

---

## References

- [TensorDict Documentation](https://pytorch.org/tensordict/)
- [VLA Policy Interface](../embodichain/agents/rl/models/vla_policy.py)
- Example configs: `configs/agents/rl/`
