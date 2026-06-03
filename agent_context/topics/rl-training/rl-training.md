# RL Training

## Entry Points

| What | Path |
|------|------|
| CLI entry | `embodichain/agents/rl/train.py` → `parse_args()` + `train_from_config(config_path)` |
| Trainer class | `embodichain/agents/rl/utils/trainer.py` → `Trainer` |
| Package init | `embodichain/agents/rl/__init__.py` — re-exports `algo`, `buffer`, `models`, `utils` |

Run training:
```bash
python -m embodichain.agents.rl.train --config <path-to-yaml-or-json> [--distributed | --no-distributed]
```

## Overview

The RL subsystem implements on-policy reinforcement learning with a modular pipeline:

1. **Config** — JSON/YAML file defines `trainer`, `policy`, and `algorithm` blocks.
2. **Environment** — Built via `build_env()` from `embodichain.lab.gym.envs.tasks.rl`.
3. **Policy** — Neural-network module (`Policy` ABC) producing actions from observations.
4. **Collector** — Steps the env, writes transitions into a preallocated `TensorDict`.
5. **Buffer** — `RolloutBuffer` owns the preallocated storage; marks it full after collection.
6. **Algorithm** — Consumes the rollout, computes losses, and updates policy weights.
7. **Trainer** — Orchestrates the collect → update → log → eval → checkpoint loop.

All rollout data flows as `TensorDict` objects (from the `tensordict` library).

## Architecture

```
train_from_config()
  ├─ build_env()                     → Gym env
  ├─ build_policy(policy_block, ...) → Policy (ActorCritic | ActorOnly | custom)
  ├─ build_algo(name, cfg, policy)   → Algorithm (PPO | GRPO)
  └─ Trainer(policy, env, algorithm, ...)
       ├─ RolloutBuffer  [buffer/standard_buffer.py]
       ├─ SyncCollector  [collector/sync_collector.py]
       └─ .train(total_timesteps)
            loop:
              _collect_rollout()  →  buffer.start_rollout() → collector.collect() → buffer.add()
              algorithm.update(buffer.get())
              _log_train(losses)
              _eval_once()  (if eval_freq hit)
              save_checkpoint()  (if save_freq hit)
```

## PPO Algorithm

**Source**: `embodichain/agents/rl/algo/ppo.py`

- Config: `PPOCfg(AlgorithmCfg)` — `n_epochs=10`, `clip_coef=0.2`, `ent_coef=0.01`, `vf_coef=0.5`.
- Inherits `AlgorithmCfg` defaults: `lr=3e-4`, `batch_size=64`, `gamma=0.99`, `gae_lambda=0.95`, `max_grad_norm=0.5`.
- `update(rollout)` flow:
  1. `compute_gae(rollout, gamma, gae_lambda)` — writes `advantage` and `return` into the TensorDict.
  2. `transition_view(rollout, flatten=True)` — drops padded final slot, flattens to `[N*T]`.
  3. For `n_epochs` × minibatch iterations:
     - Evaluate current policy: `policy.evaluate_actions(batch)` → `logprobs`, `entropy`, `values`.
     - Clipped surrogate objective + value loss + entropy bonus.
     - Adam step with `max_grad_norm` clipping.

### GRPO Algorithm

**Source**: `embodichain/agents/rl/algo/grpo.py`

- Config: `GRPOCfg(AlgorithmCfg)` — `group_size=4`, `kl_coef=0.02`, `ent_coef=0.0`, `reset_every_rollout=True`, `truncate_at_first_done=True`.
- Maintains a frozen `ref_policy` deepcopy for KL penalty when `kl_coef > 0`.
- Requires `group_size >= 2` for within-group advantage normalization.

### Algorithm Registry

**Source**: `embodichain/agents/rl/algo/__init__.py`

```python
_ALGO_REGISTRY = {"ppo": (PPOCfg, PPO), "grpo": (GRPOCfg, GRPO)}
build_algo(name, cfg_kwargs, policy, device, distributed=False)
```

When `distributed=True`, wraps the policy in `DistributedDataParallel` before passing to the algorithm.

## Rollout Buffer

**Source**: `embodichain/agents/rl/buffer/standard_buffer.py`

- `RolloutBuffer(num_envs, rollout_len, obs_dim, action_dim, device)`.
- Preallocates a single TensorDict with batch shape `[num_envs, rollout_len + 1]`.
- The `+1` slot holds the bootstrap observation/value; transition-only fields (`action`, `reward`, `done`) pad the final index.
- API: `start_rollout()` → returns the shared TensorDict for the collector to write into; `add(rollout)` → marks full; `get(flatten=True)` → returns transition view and clears.
- **Invariant**: the buffer holds at most one rollout at a time. Calling `start_rollout()` when full raises `RuntimeError`.

### Buffer Utilities

**Source**: `embodichain/agents/rl/buffer/utils.py`

- `transition_view(rollout, flatten)` — slices `[:, :-1]` on transition fields, optionally reshapes to `[N*T]`.
- `iterate_minibatches(rollout, batch_size, device)` — yields shuffled minibatches from a flattened rollout.

## Actor-Critic Models

**Source**: `embodichain/agents/rl/models/`

### Policy ABC (`policy.py`)
- `Policy(nn.Module, ABC)` — requires `forward()`, `get_value()`, `evaluate_actions()`.
- `get_action()` — convenience wrapper calling `forward()` under `torch.no_grad()`.
- All methods consume and return `TensorDict`.

### ActorCritic (`actor_critic.py`)
- Gaussian policy with learnable `log_std` per action dim (clamped `[-5, 2]`).
- Requires externally injected `actor` and `critic` `nn.Module` instances.
- `forward(td)` → samples action from `Normal(actor(obs), exp(log_std))`, writes `action`, `sample_log_prob`, `value`.

### ActorOnly (`actor_only.py`)
- Same interface but `value` is always zeros (for algorithms like GRPO that don't use a critic).

### MLP (`mlp.py`)
- `MLP(nn.Sequential)` — configurable hidden dims, activation, LayerNorm, dropout, orthogonal init.

### Policy Registry (`__init__.py`)
```python
_POLICY_REGISTRY: {"actor_critic": ActorCritic, "actor_only": ActorOnly}
build_policy(policy_block, obs_space, action_space, device, actor, critic)
build_mlp_from_cfg(module_cfg, in_dim, out_dim)  # expects {"type": "mlp", "network_cfg": {...}}
```

## Training Pipeline

**Source**: `embodichain/agents/rl/utils/trainer.py`

`Trainer.__init__` creates `RolloutBuffer` and `SyncCollector`.

`Trainer.train(total_timesteps)` loop:
1. `_collect_rollout()` — calls `buffer.start_rollout()`, then `collector.collect(buffer_size, rollout, on_step_callback)`, then `buffer.add(rollout)`.
2. `algorithm.update(buffer.get(flatten=False))` — algorithm decides its own flatten/GAE logic.
3. `_log_train(losses)` — writes to TensorBoard + optional W&B.
4. Periodic `_eval_once(num_episodes)` and `save_checkpoint()`.

Distributed training:
- `train_from_config` initializes NCCL process group, offsets seed by rank.
- Only rank 0 creates log dirs, TensorBoard writer, and W&B.
- Timestamps are broadcast from rank 0 to ensure consistent run directories.

### Collector

**Source**: `embodichain/agents/rl/collector/sync_collector.py`

`SyncCollector(env, policy, device, reset_every_rollout)`:
- `collect(num_steps, rollout, on_step_callback)` — steps env synchronously, writing obs/action/reward/done into the preallocated rollout TensorDict.
- Observations are flattened via `flatten_dict_observation()` before storage.
- Requires a preallocated rollout (`rollout=None` raises `ValueError`).

### Helper Utilities

**Source**: `embodichain/agents/rl/utils/helper.py`

- `flatten_dict_observation(obs: TensorDict)` → `[num_envs, obs_dim]` tensor.
- `dict_to_tensordict(obs_dict, device)` → converts env observation mapping to TensorDict.

## Common Failure Modes

| Symptom | Likely Cause |
|---------|-------------|
| `RuntimeError: RolloutBuffer already contains a rollout` | Called `start_rollout()` without consuming via `get()`. |
| `ValueError: Preallocated rollout batch size mismatch` | `buffer_size` in trainer config doesn't match `num_steps` passed to collector. |
| `ValueError: Algorithm 'X' not found` | Algo name not in `_ALGO_REGISTRY`. Check `get_registered_algo_names()`. |
| `ValueError: ActorCritic policy requires external 'actor' and 'critic' modules` | Config uses `actor_critic` policy but doesn't define `actor`/`critic` MLP blocks in the JSON. |
| `ValueError: Configured policy.action_dim=N does not match env action dim M` | `policy.action_dim` in config disagrees with the env's action manager. |
| `RuntimeError: torch.distributed is not initialized` | `distributed=True` but `init_process_group()` was not called (launch via `torchrun`). |
| `GRPO: group_size >= 2` | GRPO requires at least 2 environments per group for normalization. |
| NaN losses | Check `log_std` bounds, gradient clipping, and reward scale. `max_grad_norm` defaults to 0.5. |
| Stale observations after reset | `SyncCollector` resets obs via `_reset_env()` on init; set `reset_every_rollout=True` if episodes must fully reset between rollouts. |
