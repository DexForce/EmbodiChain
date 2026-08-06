# RL Learning

## Entry Points

| What | Path |
|------|------|
| CLI entry | `embodichain/learning/rl/train.py` → `parse_args()` + `train_from_config(config_path)` |
| Trainer class | `embodichain/learning/rl/utils/trainer.py` → `Trainer` |
| Package init | `embodichain/learning/rl/__init__.py` — re-exports `algo`, `buffer`, `models`, `utils` |

Run training:
```bash
python -m embodichain.learning.rl.train --config <path-to-yaml-or-json> [--distributed | --no-distributed]
```

## Overview

The RL subsystem implements on-policy reinforcement learning with a modular pipeline:

1. **Config** — JSON/YAML file defines `trainer`, `policy`, and `algorithm` blocks.
2. **Environment** — Simulator tasks use `build_env()`; lightweight tensor
   tasks use the `learning_env` registry. Official tasks live in
   `embodichain_tasks`.
3. **Policy** — Neural-network module (`Policy` ABC) producing actions from observations.
4. **Collector** — Steps the env, writes transitions into a preallocated `TensorDict`.
5. **Buffer** — `RolloutBuffer` owns the preallocated storage; marks it full after collection.
6. **Algorithm** — Consumes the rollout, computes losses, and updates policy weights.
7. **Trainer** — Orchestrates the collect → update → log → eval → checkpoint loop.

Standard PPO/GRPO rollout data flows as `TensorDict` objects (from the
`tensordict` library).

## Differentiable RL

**Sources**: `env.py`, `collector/differentiable.py`, `algo/apg.py`,
`differentiable_trainer.py`

- `DifferentiableVecEnv.step()` preserves gradients through rewards and the
  differentiable next state.
- `detach_state()` creates the TBPTT boundary without resetting the environment.
- `DifferentiableCollector` keeps graph-connected transitions outside the
  standard preallocated buffer and preserves the policy mode selected by its
  caller.
- Differentiable policy sampling uses `get_differentiable_action()` and
  `rsample()`.
- `APG` maximizes segmented discounted return with optional entropy, gradient
  clipping, and non-finite update protection.
- `DifferentiableTrainer.segment_length` controls the TBPTT boundary, while
  `update_horizon` independently controls environment steps per optimizer
  update. Segment losses backpropagate immediately, state is detached after
  each segment, and policy gradients accumulate until one update horizon is
  complete.
- Checkpoints store policy, optimizer, optional LR scheduler, trainer counters, and best-evaluation
  state.
- Training selects `policy.train()`, while evaluation temporarily selects
  `policy.eval()` and restores the previous mode afterward.

The collector paths are intentionally separate:
- PPO/GRPO use `SyncCollector` and `TensorDict`.
- APG uses `DifferentiableCollector` and `DifferentiableRollout`.

APG is registered with `RolloutKind.DIFFERENTIABLE`. The shared `train.py`
entry point routes it to `DifferentiableTrainer`; PPO/GRPO declare
`RolloutKind.STANDARD` and continue to use `Trainer`. Collectors and rollout
types remain separate even though environment construction, policy building,
evaluation, logging, and checkpoint conventions are shared.

`PointMassRL` in
`embodichain_tasks/embodichain_tasks/rl/basic/point_mass.py` is the canonical
dual-path task. The same differentiable PyTorch dynamics run under
`torch.no_grad()` for PPO and preserve the action-to-reward graph for APG.

### Newton Reference

`experimental/newton/planar_reach.py` is an FK-only two-link test environment.
It bridges reward, joint-state, and end-effector gradients from
`newton.eval_fk` and a Warp tape into PyTorch. It is not a dynamics simulator
or a task implementation for NMG.

The training demo runs APG through `DifferentiableTrainer`, samples new initial
joints and FK-reachable targets for every update, then evaluates the learned
policy on held-out random seeds:

```bash
python -m embodichain.learning.rl.experimental.newton.train_planar_reach \
  --device cuda:0
```

With the default seeds, 600 updates improve the mean minimum distance from
`1.85` to `0.039` and reach an `86.7%` success rate across 512 held-out samples
at the `0.05` threshold. Tests also cover analytical FK, finite-difference
gradients, TBPTT, APG updates, and held-out improvement.

## Architecture

```
train_from_config()
  ├─ build_env() | build_learning_env()
  ├─ build_policy(policy_block, ...) → Policy (ActorCritic | ActorOnly | custom)
  ├─ build_algo(name, cfg, policy)   → Algorithm (PPO | GRPO | APG)
  └─ route by RolloutKind
       ├─ STANDARD       → Trainer + SyncCollector + RolloutBuffer
       └─ DIFFERENTIABLE → DifferentiableTrainer + DifferentiableCollector
```

Both trainers call `evaluation.evaluate_episodes()` with an independent
evaluation environment. It counts actual completed episodes under asynchronous
auto-reset and logs terminal-only metrics as `eval/*`.

```text
Trainer(policy, env, algorithm, ...)
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

**Source**: `embodichain/learning/rl/algo/ppo.py`

- Config: `PPOCfg(AlgorithmCfg)` — `n_epochs=10`, `clip_coef=0.2`, `ent_coef=0.01`, `vf_coef=0.5`.
- Inherits `AlgorithmCfg` defaults: nested `optimizer` (`adam`, `lr=3e-4`), optional `lr_scheduler`, `batch_size=64`, `gamma=0.99`, `gae_lambda=0.95`, `max_grad_norm=0.5`.
- `update(rollout)` flow:
  1. `compute_gae(rollout, gamma, gae_lambda)` — writes `advantage` and `return` into the TensorDict.
  2. `transition_view(rollout, flatten=True)` — drops padded final slot, flattens to `[N*T]`.
  3. For `n_epochs` × minibatch iterations:
     - Evaluate current policy: `policy.evaluate_actions(batch)` → `logprobs`, `entropy`, `values`.
     - Clipped surrogate objective + value loss + entropy bonus.
     - Adam step with `max_grad_norm` clipping.

### GRPO Algorithm

**Source**: `embodichain/learning/rl/algo/grpo.py`

- Config: `GRPOCfg(AlgorithmCfg)` — `group_size=4`, `kl_coef=0.02`, `ent_coef=0.0`, `reset_every_rollout=True`, `truncate_at_first_done=True`.
- Maintains a frozen `ref_policy` deepcopy for KL penalty when `kl_coef > 0`.
- Requires `group_size >= 2` for within-group advantage normalization.

### Algorithm Registry

**Source**: `embodichain/learning/rl/algo/__init__.py`

```python
_ALGO_REGISTRY = {
    "apg": (APGCfg, APG),
    "ppo": (PPOCfg, PPO),
    "grpo": (GRPOCfg, GRPO),
}
build_algo(name, cfg_kwargs, policy, device, distributed=False)
```

`APG.rollout_kind` is `RolloutKind.DIFFERENTIABLE`; PPO/GRPO use
`RolloutKind.STANDARD`. When `distributed=True`, wraps the policy in
`DistributedDataParallel` before passing to the algorithm and rejects
differentiable algorithms.

## Rollout Buffer

**Source**: `embodichain/learning/rl/buffer/standard_buffer.py`

- `RolloutBuffer(num_envs, rollout_len, obs_dim, action_dim, device)`.
- Preallocates a single TensorDict with batch shape `[num_envs, rollout_len + 1]`.
- The `+1` slot holds the bootstrap observation/value; transition-only fields (`action`, `reward`, `done`) pad the final index.
- API: `start_rollout()` → returns the shared TensorDict for the collector to write into; `add(rollout)` → marks full; `get(flatten=True)` → returns transition view and clears.
- **Invariant**: the buffer holds at most one rollout at a time. Calling `start_rollout()` when full raises `RuntimeError`.

### Buffer Utilities

**Source**: `embodichain/learning/rl/buffer/utils.py`

- `transition_view(rollout, flatten)` — slices `[:, :-1]` on transition fields, optionally reshapes to `[N*T]`.
- `iterate_minibatches(rollout, batch_size, device)` — yields shuffled minibatches from a flattened rollout.

## Actor-Critic Models

**Source**: `embodichain/learning/rl/models/`

### Policy ABC (`policy.py`)
- `Policy(nn.Module, ABC)` — requires `forward()`, `get_value()`, `evaluate_actions()`.
- `get_action()` — convenience wrapper calling `forward()` under `torch.no_grad()`.
- `get_differentiable_action()` — explicit graph-preserving action API;
  implementations must provide reparameterized stochastic sampling.
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

**Source**: `embodichain/learning/rl/utils/trainer.py`

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

**Source**: `embodichain/learning/rl/collector/sync_collector.py`

`SyncCollector(env, policy, device, reset_every_rollout)`:
- `collect(num_steps, rollout, on_step_callback)` — steps env synchronously, writing obs/action/reward/done into the preallocated rollout TensorDict.
- Observations are flattened via `flatten_dict_observation()` before storage.
- Requires a preallocated rollout (`rollout=None` raises `ValueError`).

`DifferentiableCollector(env, policy, device)`:
- Collects short segments without `torch.no_grad()` or preallocated copies.
- Returns `DifferentiableRollout` with immutable transition records.
- `rollout.rewards` stacks rewards as `[time, num_envs]` while retaining their
  autograd history.
- `detach_state()` updates the collector to the environment's detached boundary
  observation before the next segment.

### Helper Utilities

**Source**: `embodichain/learning/rl/utils/helper.py`

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
| `Learning environment 'X' is not registered` | Task package was not imported. Call `discover_task_packages()` / `execute_init_hooks()`, or import the task module that uses `@register_learning_env`. |
| `Differentiable algorithms require trainer.learning_env` | APG cannot train simulator `gym_config` envs; use a registered `learning_env` such as `PointMassRL`. |
| Eval success rate stuck near zero | Confirm `evaluate_episodes` reads terminal-only `success`/`metrics`, and that the eval env uses a held-out `eval_seed`. |
