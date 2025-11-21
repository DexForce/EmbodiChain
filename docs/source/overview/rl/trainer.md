# Trainer

This module implements the main RL training loop, logging management, and event-driven extension.

## Main Classes and Structure

### Trainer
- RL training coordinator, responsible for the interaction between algorithm, environment, and policy.
- Main responsibilities:
    - Manage training loop, evaluation, and model saving.
    - Event-driven extension (e.g., environment randomization, data logging, evaluation events).
    - Logging output (TensorBoard/WandB/console), tracking rewards, episode length, loss, etc.
- Key fields:
    - `policy`: RL policy object.
    - `algorithm`: RL algorithm object.
    - `env`/`eval_env`: Training and evaluation environments.
    - `writer`: TensorBoard logger.
    - `event_manager`/`eval_event_manager`: Event managers.
    - `global_step`, `ret_window`, `len_window`: Training statistics.

## Main Methods
- `train(total_timesteps)`: Main training loop, automatically collects data, updates policy, and logs.
- `_collect_rollout()`: Collect one rollout, supports custom callback statistics.
- `_log_train(losses)`: Log training loss, reward, sampling speed, etc.
- `_eval_once()`: Periodic evaluation, records evaluation metrics.
- `save_checkpoint()`: Save model parameters and training state.

## Event Management
- Supports custom events (e.g., environment randomization, data logging) injected via EventManager.
- Events can be executed by interval/step/trigger, enabling flexible extension.

## Logging and Monitoring
- Supports TensorBoard and WandB logging, automatically records reward, episode length, loss, sampling speed, etc.
- Console output for training progress and statistics.

## Usage Example
```python
trainer = Trainer(policy, env, algorithm, num_steps, batch_size, writer, ...)
trainer.train(total_steps)
trainer.save_checkpoint()
```

## Extension and Customization
- Custom event modules can be implemented for environment reset, data collection, evaluation, etc.
- Supports multi-environment parallelism and distributed training.
- Training process can be flexibly adjusted via config files.

## Practical Tips
- It is recommended to perform periodic evaluation and model saving to prevent loss of progress during training.
- The event mechanism can be used for automated experiments, data collection, and environment reset.
- Logging and monitoring help analyze training progress and tune hyperparameters.

---
