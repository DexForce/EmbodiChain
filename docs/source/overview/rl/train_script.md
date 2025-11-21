# Train Script

This module provides the RL training entry script, responsible for parsing configuration, initializing modules, and starting training. It supports multi-task and automated experiments.

## Main Structure and Flow

### train.py
- Main training script, supports command-line arguments (such as --config), automatically loads JSON config.
- Initializes device, random seed, output directory, and logging (TensorBoard/WandB).
- Loads environment config, supports multi-environment parallelism and evaluation environments.
- Builds policy (e.g., actor-critic), algorithm (e.g., PPO), and Trainer.
- Supports event management (e.g., environment randomization, data logging, evaluation events).
- Automatically saves model checkpoints and performs periodic evaluation.

## Argument Parsing
- Supports command-line arguments:
    - `--config`: Specify the path to the config file (JSON only).
- The config file includes parameters for trainer, policy, algorithm, events, and other modules.

## Module Initialization
- Device selection (CPU/GPU), automatic detection and setup.
- Random seed setting to ensure experiment reproducibility.
- Output directory is automatically generated, log files are managed automatically.
- Supports TensorBoard/WandB logging, automatically records the training process.

## Training Flow
1. Load the JSON config file and parse parameters for each module.
2. Initialize environment, policy, algorithm, and Trainer.
3. Enter the main training loop: collect data, update policy, record logs.
4. Periodically evaluate and save the model.
5. Supports graceful interruption and auto-saving with KeyboardInterrupt.

## Usage Example
```bash
python train.py --config configs/agents/rl/push_cube/train_config.json
```

## Extension and Customization
- Supports custom event modules for flexible training flow extension.
- Can integrate multi-task and multi-environment training.
- Config-driven management for batch experiments and parameter tuning.

## Practical Tips
- It is recommended to manage all experiment parameters via JSON config files for reproducibility and tuning.
- Supports multi-environment and event extension to improve training flexibility.
- Logging and checkpoint management help with experiment tracking and recovery.

---
