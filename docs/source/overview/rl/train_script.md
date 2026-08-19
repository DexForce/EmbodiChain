# Train Script

This module provides the RL training entry script, responsible for parsing configuration, initializing modules, and starting training. It supports multi-task and automated experiments.

## Main Structure and Flow

### train.py
- Main training script, supports command-line arguments (such as --config), automatically loads JSON or YAML config.
- Initializes device, random seed, output directory, and logging (TensorBoard/WandB).
- Loads either a simulator `gym_config` or a lightweight registered
  `learning_env`, with independent vectorized evaluation environments.
- Builds the policy and algorithm, then routes standard rollouts
  (PPO/GRPO) to `Trainer` and differentiable rollouts (APG) to
  `DifferentiableTrainer`.
- Supports event management (e.g., environment randomization, data logging, evaluation events).
- Automatically saves model checkpoints and performs periodic evaluation.

## Argument Parsing
- Supports command-line arguments:
    - `--config`: Specify the path to the config file (``.json``, ``.yaml``, or ``.yml``).
    - `--distributed`: Enable multi-GPU distributed training.
    - `--profile` / `--profile_output`: Gym env profiling during training (requires `trainer.gym_config`).
- The config file includes parameters for trainer, policy, algorithm, events, and other modules.
- See [Multi-GPU Training](multi_gpu.md) for distributed training.

## Module Initialization
- Device selection (CPU/GPU), automatic detection and setup.
- Random seed setting to ensure experiment reproducibility.
- Output directory is automatically generated, log files are managed automatically.
- Supports TensorBoard/WandB logging, automatically records the training process.

## Training Flow
1. Load the config file and parse parameters for each module.
2. Initialize environment, policy, algorithm, and Trainer.
3. Enter the main training loop: collect data, update policy, record logs.
4. Periodically evaluate and save the model.
5. Supports graceful interruption and auto-saving with KeyboardInterrupt.

## Usage Example
```bash
embodichain train-rl --config embodichain_tasks/configs/agents/rl/basic/cart_pole/train_config.yaml
embodichain train-rl --config embodichain_tasks/configs/agents/rl/basic/point_mass/train_apg.yaml
embodichain train-rl --config embodichain_tasks/configs/agents/rl/basic/point_mass/train_ppo.yaml
```

PointMass intentionally uses one differentiable PyTorch environment for both
algorithms. The standard collector runs it under `torch.no_grad()`, whereas
APG retains the dynamics graph and detaches it only at configured TBPTT
segment boundaries.

Evaluation always uses a separate environment and deterministic actions.
Metrics are logged under `eval/*`; `num_eval_episodes` is the exact number of
completed episodes rather than episodes per parallel environment.

## Extension and Customization
- Supports custom event modules for flexible training flow extension.
- Can integrate multi-task and multi-environment training.
- Config-driven management for batch experiments and parameter tuning.

## Practical Tips
- It is recommended to manage all experiment parameters via JSON or YAML config files for reproducibility and tuning.
- Supports multi-environment and event extension to improve training flexibility.
- Logging and checkpoint management help with experiment tracking and recovery.
