# VLA Model Integration Guide

This guide explains how to integrate a VLA (Vision-Language-Action) model with the EmbodiChain RL training framework.

## For VLA Model Developers

### 1. Model Interface Requirements

Your VLA model class must implement the following interface:

```python
class YourVLAModel(nn.Module):
    def __init__(self, **config):
        """Initialize VLA model with configuration."""
        super().__init__()
        # Your initialization code
        
    def forward(self, observations: TensorDict) -> torch.Tensor:
        """Generate actions from observations.
        
        Args:
            observations: TensorDict containing observation data
                Expected keys may include:
                - "rgb": RGB images [B, H, W, C] or [B, C, H, W]
                - "depth": Depth images [B, H, W]
                - "proprio": Proprioceptive state [B, proprio_dim]
                - "language": Language tokens [B, seq_len] or raw strings
                
        Returns:
            Action tensor [B, action_dim]
        """
        # Your action generation code
        pass
        
    def get_value(self, observations: TensorDict) -> torch.Tensor:
        """Get value estimate for observations.
        
        Args:
            observations: TensorDict containing observation data
            
        Returns:
            Value tensor [B, 1]
        """
        # Your value estimation code
        pass
        
    def evaluate_actions(
        self, 
        observations: TensorDict, 
        actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for observation-action pairs.
        
        Args:
            observations: TensorDict containing observation data
            actions: Action tensor [B, action_dim]
            
        Returns:
            Tuple of (log_prob [B,], entropy [B,])
        """
        # Your action evaluation code
        pass
```

**Important Notes**:
- All methods must accept `TensorDict` for observations (not plain tensors)
- Handle missing observation keys gracefully (not all tasks provide all modalities)
- Your model should manage its own tokenization, preprocessing, and internal state
- Value head is required for PPO training (can be a simple MLP on top of your embeddings)

### 2. Implement Checkpoint Loading

Edit `embodichain/agents/rl/models/vla_policy.py` and implement the `load_vla_model()` function:

```python
def load_vla_model(
    model_path: str,
    model_class: Optional[str] = None,
    model_config: Optional[dict] = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load VLA model from checkpoint."""
    import importlib
    
    # Parse model class path
    module_name, class_name = model_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    ModelClass = getattr(module, class_name)
    
    # Initialize model
    model = ModelClass(**model_config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move to device
    model.to(device)
    model.eval()  # Start in eval mode (trainer will set to train)
    
    return model
```

Adapt this to your checkpoint format (may use different keys, compression, etc.).

### 3. Configuration Format

Create a training config JSON:

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
      "model_path": "path/to/your/checkpoint.pth",
      "model_class": "your_package.YourVLAModel",
      "model_config": {
        "vision_encoder": "resnet50",
        "language_model": "gpt2",
        "freeze_vision_encoder": false,
        ... // your model-specific config
      }
    }
  },
  "algorithm": {
    "name": "ppo",
    "cfg": {
      "learning_rate": 1e-5,  // Lower LR for fine-tuning
      ...
    }
  }
}
```

### 4. Testing Your Integration

```bash
# Run training with your VLA model
python embodichain/agents/rl/train.py --config configs/agents/rl/your_vla_config.json
```

Expected workflow:
1. `load_vla_model()` loads your pretrained checkpoint
2. `VLAPolicy` wraps your model and adapts it to Policy interface
3. RL trainer fine-tunes your model using PPO (or other algorithms)

### 5. Tips for VLA Models

**Value Head**:
- PPO requires value estimates
- Add a simple value head (e.g., linear layer) on your embeddings
- Can initialize randomly or pretrain with imitation learning

**Action Distribution**:
- `evaluate_actions()` needs to compute log_prob and entropy
- For continuous actions: use Gaussian distribution (mean from your model, learnable std)
- For discrete actions: use Categorical distribution

**Gradient Flow**:
- Set `freeze_vision_encoder: true` to only fine-tune action/value heads
- Set `freeze_language_model: true` if using large LMs (reduce memory)
- Or fine-tune entire model with lower learning rate

**Memory Optimization**:
- VLA models are large - use `buffer_type: "vla"` with async collection
- Reduce `num_envs` if running out of memory
- Consider gradient checkpointing for very large models

## For RL Framework Users

If VLA model is already integrated, simply configure and run:

```bash
python embodichain/agents/rl/train.py --config configs/agents/rl/vla_example/train_config.json
```

See [RL_TRAINING_FRAMEWORK.md](RL_TRAINING_FRAMEWORK.md) for general usage.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Training Framework                     │
│                                                              │
│  ┌────────────┐      ┌──────────────┐     ┌──────────────┐ │
│  │   Trainer  │─────▶│  VLAPolicy   │────▶│   VLA Model  │ │
│  │            │      │   (wrapper)  │     │  (your code) │ │
│  └────────────┘      └──────────────┘     └──────────────┘ │
│        │                     │                     │        │
│        │                     │                     │        │
│   TensorDict            TensorDict           TensorDict    │
│        │                     │                     │        │
│        ▼                     ▼                     ▼        │
│  [obs, action,        [forward,              [vision,       │
│   reward, done]        get_value,            language,      │
│                        evaluate_actions]      action]       │
└─────────────────────────────────────────────────────────────┘
```

## Reference Implementation

See `embodichain/agents/rl/models/vla_policy.py` for:
- `VLAModelInterface`: Protocol defining required methods
- `VLAPolicy`: Wrapper that adapts VLA model to Policy interface
- `load_vla_model()`: Checkpoint loading function (to be implemented)

Example config: `configs/agents/rl/vla_example/train_config.json`
