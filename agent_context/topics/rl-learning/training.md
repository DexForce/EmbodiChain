# RL training, evaluation and extension

Read this when the request needs these details. [Topic overview](rl-learning.md).

## Training and Evaluation Lifecycle

The standard trainer repeats:

1. start and collect a rollout;
2. update the algorithm;
3. log train metrics;
4. evaluate when the configured step boundary is reached;
5. save periodic and best-evaluation checkpoints.

Evaluation uses an independent environment and
`evaluate_episodes()`. It counts completed asynchronous episodes, reports
terminal metrics, temporarily switches the policy to evaluation mode, and
restores its prior mode.

Checkpoints include policy parameters, trainer and complete-rollout counters,
best-evaluation state, observation-normalizer state when enabled, and optimizer
or LR-scheduler state when present. Evaluation reuses the frozen training
normalizer without updating its statistics.

`trainer.seed` seeds Python, NumPy, Torch, CUDA, and Warp before environment
construction. Set `trainer.torch_deterministic: true` when a reference run
also requires deterministic PyTorch algorithms.

Complete APG mode resets before every independent microbatch and preserves one
graph across the entire environment-provided horizon. It masks rewards after
the first done, applies the rollout's objective scale, and averages
`gradient_accumulation_steps` full trajectories before one optimizer step. It
never shortens a scheduled horizon to satisfy `total_timesteps`; without an
explicit timestep budget, CLI `iterations` maps to an exact optimizer-update
budget.

Complete mode can clamp actions to the environment space and install a
per-environment action-adjoint norm hook. Non-finite adjoint rows are zeroed;
finite rows are clipped independently with an overflow-safe norm. APG also has
a pre-clip policy-gradient safety limit that skips unsafe updates.

On the simulator path, distributed mode initializes NCCL, assigns one CUDA
device per local rank, wraps the policy in
`DistributedDataParallel`, aggregates step and episode statistics, and
keeps logging, evaluation, and checkpoint ownership on rank zero.
Differentiable algorithms and lightweight learning environments do not
currently support this distributed path.

## Official Examples

| Example | Environment path | Config location |
|---------|------------------|-----------------|
| CartPole | registered simulator Gym env | `embodichain_tasks/configs/tasks/classic_control/cart_pole/agents/` |
| PushCube | registered simulator Gym env | `embodichain_tasks/configs/tasks/manipulation/push_cube/agents/` |
| PointMass PPO | registered lightweight env, standard rollout | `embodichain_tasks/configs/tasks/classic_control/point_mass/agents/ppo.yaml` |
| PointMass APG | differentiable lightweight env | `embodichain_tasks/configs/tasks/classic_control/point_mass/agents/apg.yaml` |
| Newton planar reach | experimental differentiable FK reference | `embodichain/learning/rl/experimental/newton/` |

`PointMassRL` is the reference environment for comparing standard and
differentiable training over the same task dynamics. The Newton planar-reach
example is an experimental gradient reference, not a general simulator task.

## Extension Points

### Add an Algorithm

1. Implement a `BaseAlgorithm` subclass and config under `algo/`.
2. Declare the correct `RolloutKind`.
3. Register the config/class pair in `algo/__init__.py`.
4. Add focused algorithm, routing, and rollout-contract tests.

### Add a Policy

1. Implement the `Policy` contract under `models/`.
2. Register it in `models/__init__.py`.
3. Ensure its outputs satisfy every intended algorithm.
4. Provide graph-preserving sampling if used with differentiable rollouts.

### Add a Lightweight Environment

1. Implement `LearningVecEnv`, or `DifferentiableVecEnv` for APG.
2. Register the factory with `@register_learning_env`.
3. Ensure finished rows auto-reset while returning terminal reward/done with
   the next initial observation.
4. For variable complete APG rollouts, implement
   `ScheduledDifferentiableVecEnv` and return the full non-truncated horizon.
5. Expose `observation_normalize_mask` when semantic dimensions must remain
   raw during normalization.
6. Add an official config under
   `embodichain_tasks/configs/tasks/<domain>/<task>/agents/` when it is a
   bundled task.

Use `add-task-env` for simulator-backed task environments and
`manager-functor` for their observation, reward, event, and action
components.
