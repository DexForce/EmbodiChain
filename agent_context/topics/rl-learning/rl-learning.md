# RL Learning

## Entry Points

| What | Path |
|------|------|
| Unified CLI | `embodichain train-rl --config <train.yaml-or-json>` |
| CLI implementation | `embodichain/learning/rl/train.py` → `cli()` |
| Programmatic training | `embodichain/learning/rl/train.py` → `train_from_config()` |
| Algorithm registry | `embodichain/learning/rl/algo/__init__.py` |
| Policy registry | `embodichain/learning/rl/models/__init__.py` |
| Lightweight env registry | `embodichain/learning/rl/env.py` |
| Standard trainer | `embodichain/learning/rl/utils/trainer.py` → `Trainer` |
| Differentiable trainer | `embodichain/learning/rl/differentiable_trainer.py` |
| Official configs | `embodichain_tasks/configs/tasks/<domain>/<task>/agents/` |

The compatibility module entry point is:

```bash
python -m embodichain.learning.rl.train --config <train.yaml-or-json>
```

Prefer the unified `embodichain train-rl` command.

## Configuration Resolution

Training configs are JSON or YAML mappings with three top-level blocks:

- `trainer`: environment selection, runtime, rollout, evaluation, logging,
  checkpoint, and distributed settings;
- `policy`: registered policy name and network definitions;
- `algorithm`: registered algorithm name plus its config mapping.

The resolution flow is:

```text
CLI --config
  → load_config()
  → trainer / policy / algorithm blocks
  → choose trainer.learning_env or trainer.gym_config
  → build environment
  → build policy and optional MLP modules
  → build algorithm config from the registry
  → route by algorithm.rollout_kind
  → train, evaluate, log, and checkpoint
```

Read concrete values from the selected config and current config classes. Do
not assume example-config values are global defaults.

## Environment Paths

### Simulator Gym Environment

Select this path with `trainer.gym_config`.

1. The CLI discovers installed task packages and executes their init hooks.
2. `train_from_config()` loads the gym config.
3. `config_to_cfg()` builds the environment config and manager functors.
4. Trainer runtime fields override simulation device, GPU, renderer, headless
   mode, environment count, and optional profiling.
5. `build_env()` constructs the registered Gym environment.
6. A sample reset determines flattened observation and action dimensions.

Simulator environments use standard rollouts. A differentiable algorithm on
this path is rejected.

Direct callers of `train_from_config()` that bypass `cli()` must ensure
task packages and init hooks needed by a simulator environment have already
been loaded.

### Lightweight Learning Environment

Select this path with `trainer.learning_env`, either as a registered name or
as a mapping with `name` and `cfg`.

`build_learning_env()` resolves factories registered with
`@register_learning_env`. A learning environment implements the
`LearningVecEnv` protocol. Differentiable algorithms additionally require
`DifferentiableVecEnv.detach_state()` as the truncated-backpropagation
boundary.

This path supports both standard algorithms and differentiable algorithms,
but currently rejects distributed training and environment profiling.

## Rollout and Trainer Routing

`BaseAlgorithm.rollout_kind` is the routing contract:

```text
RolloutKind.STANDARD
  → Trainer
  → SyncCollector
  → RolloutBuffer backed by TensorDict
  → PPO or GRPO update

RolloutKind.DIFFERENTIABLE
  → DifferentiableTrainer
  → DifferentiableCollector
  → graph-connected DifferentiableRollout segments
  → APG update
```

PPO and GRPO use `STANDARD`. APG uses `DIFFERENTIABLE`.
`get_trainer_class()` centralizes this selection for lightweight learning
environments.

The standard buffer reserves shape `[num_envs, rollout_length + 1]`; the
last slot holds the bootstrap observation/value while transition-only fields
use it as padding. The collector writes into the preallocated rollout and the
algorithm consumes it after collection.

The differentiable path does not copy transitions into the standard buffer.
It preserves the action-to-reward autograd graph across short segments.
`segment_length` sets TBPTT boundaries, while `update_horizon` controls
how many environment steps contribute to one optimizer update.

## Component Ownership

| Component | Owner |
|-----------|-------|
| Algorithm base, rollout kind, optimizer scheduling | `algo/base.py`, `utils/optimizer.py` |
| PPO, GRPO, APG implementations | `algo/ppo.py`, `algo/grpo.py`, `algo/apg.py` |
| Standard rollout storage and views | `buffer/` |
| Standard and differentiable collection | `collector/` |
| Policy interface, actor-critic, actor-only, MLP builder | `models/` |
| Standard collect/update loop | `utils/trainer.py` |
| Differentiable TBPTT/update loop | `differentiable_trainer.py` |
| Shared completed-episode evaluation | `evaluation.py` |
| Learning environment protocol and registry | `env.py` |
| End-to-end config and runtime assembly | `train.py` |

Rollout payloads on the standard path are `TensorDict` objects. Policies
consume observations and write action, log-probability, entropy, and value
fields needed by their algorithm. Differentiable policies must expose
graph-preserving action sampling.

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

Checkpoints include policy parameters, trainer counters, best-evaluation
state, and optimizer or LR-scheduler state when present.

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
4. Add an official config under
   `embodichain_tasks/configs/tasks/<domain>/<task>/agents/` when it is a
   bundled task.

Use `add-task-env` for simulator-backed task environments and
`manager-functor` for their observation, reward, event, and action
components.

## Invariants

- The selected environment must expose batched observation/action spaces and
  `num_envs`.
- Policy observation and action dimensions must match the built environment.
- An algorithm's `RolloutKind` must match its collector, rollout type, and
  trainer.
- The standard buffer holds at most one unconsumed rollout.
- APG must retain differentiable rewards until its optimizer boundary;
  `detach_state()` must not reset or resample the task.
- GRPO environment count must satisfy its grouping contract.
- Evaluation must use completed episodes and an independent environment.
- Only rank zero owns external logging and checkpoints in distributed runs.

## Common Failure Modes

| Symptom | Likely cause |
|---------|--------------|
| Algorithm or policy name is not found | Name is absent from the corresponding registry |
| Learning environment is not found | Its task package was not discovered or the module containing its decorator was not imported |
| Differentiable algorithm rejects the config | The config selected `trainer.gym_config` instead of a differentiable `learning_env` |
| Distributed training is rejected | The request uses a lightweight/differentiable path, or the process group/CUDA device is not initialized |
| Policy dimension mismatch | Policy config disagrees with the built environment's observation or action space |
| Standard buffer is already full | A rollout was started before the previous one was consumed with `get()` |
| APG gradients disappear | Actions were sampled under `no_grad`, transitions were copied/detached, or the state was detached too early |
| GRPO reshape or grouping fails | `num_envs` is not divisible by `group_size` |
| Evaluation never completes | The environment does not emit completed asynchronous episodes or terminal metrics correctly |
| Output/checkpoint directories diverge across ranks | Distributed run metadata was not coordinated through rank zero |
