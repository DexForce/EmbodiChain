# RL Learning

## Entry Points

| What | Path |
|------|------|
| Unified CLI | `embodichain train-rl --config <train.yaml-or-json>`; `--resume FILE` or `--resume latest/best --run RUN` |
| CLI implementation | `embodichain/learning/rl/train.py` → `cli()` |
| Standalone evaluation | `embodichain/learning/rl/policy_evaluation/cli.py`; local RUN/checkpoint, pretrained bundle or external Profile |
| Programmatic training | `embodichain/learning/rl/train.py` → `train_from_config()` |
| Algorithm registry | `embodichain/learning/rl/algo/__init__.py` |
| Policy registry | `embodichain/learning/rl/policies/registry.py` |
| Lightweight env registry | `embodichain/learning/rl/env.py` |
| Standard trainer | `embodichain/learning/rl/utils/trainer.py` → `Trainer` |
| Differentiable trainer | `embodichain/learning/rl/differentiable_trainer.py` |
| Official configs | `extensions/embodichain_tasks/configs/tasks/<domain>/<task>/agents/` |

The compatibility module entry point is:

```bash
python -m embodichain.learning.rl.train --config <train.yaml-or-json>
```

Prefer the unified `embodichain train-rl` command. Pretrained evaluation resolves
a downloaded RUN before using the same native runtime and checkpoint loader;
model-cache and repository revision ownership belong to
[data assets](../data-assets/data-assets.md#pretrained-policy-bundles).

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
4. Trainer runtime fields override simulation device, GPU, headless mode,
   environment count, and optional profiling. An explicit `trainer.renderer`
   overrides rendering; omission preserves the task's render configuration.
   Active train events and rank-zero enabled eval events are resolved before
   the first World; built-in recording and nonempty validation cameras reserve
   a native renderer when the effective task renderer is auto.
5. `build_env()` constructs the registered Gym environment.
6. A sample reset determines flattened observation and action dimensions.

Simulator environments use standard rollouts. A differentiable algorithm on
this path is rejected. Simulator tasks with a supported training configuration
declare `supports_rl=True` in `@register_env`. Task listing is implemented
in `embodichain/cli/list_task.py`; capability and config discovery belong to
[environment configuration](../env-framework/configuration.md).

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

Variable-horizon APG environments may also implement
`ScheduledDifferentiableVecEnv.prepare_differentiable_rollout()`. Its
`DifferentiableRolloutSpec` selects the next reset's complete horizon,
per-environment objective scale, and scalar rollout metadata.

This path supports both standard algorithms and differentiable algorithms,
but currently rejects distributed training and environment profiling.

## Rollout and Trainer Routing

### Native locomotion tasks

Official velocity tasks live in `extensions/embodichain_tasks/embodichain_tasks/locomotion/velocity/`;
Humanoid Run lives under `classic_control/humanoid/`. Matching task config
directories own default/Newton environment and PPO files. The package init hook
registers locomotion observations and rewards; joint actions and root-velocity
randomization use the standard manager components.

Headless, contact-only tasks can resolve auto rendering to NoRender. Camera
recording and Viewer paths still require native replica capacity. Read the
selected task's environment and PPO files for batch sizes and capacity budgets;
these are deployment settings, not trainer defaults. See
[render selection](../simulation-system/rendering.md#norender-initialization).

Contact sampling and selective history reset follow the sensor-owned
[contact history contract](../sensor-system/contact-history.md). Root-velocity
writes use `Articulation.set_root_velocity()`; asset resolution belongs to
[data assets](../data-assets/data-assets.md).

### Algorithm selection

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
It supports explicitly configured `segmented` TBPTT rollouts and `complete`
rollouts that reset before each independent microbatch, retain the full
environment-provided horizon, mask post-terminal rewards, and accumulate full
trajectory gradients before one optimizer step. Read the training detail for
the complete-rollout safety and normalization contracts.

## Component Ownership

| Component | Owner |
|-----------|-------|
| Algorithm base, rollout kind, optimizer scheduling | `algo/base.py`, `utils/optimizer.py` |
| PPO, GRPO, APG implementations | `algo/ppo.py`, `algo/grpo.py`, `algo/apg.py` |
| Standard rollout storage and views | `buffer/` |
| Standard and differentiable collection | `collector/` |
| Policy interface, actor-critic, actor-only, policy builder | `learning/rl/policies/` |
| Shared MLP and network builder | `learning/models/mlp.py` |
| Running observation statistics | `normalization.py` |
| Batched action-adjoint stabilization | `gradients.py` |
| Actor and critic observation statistics | `learning/models/normalizer.py` |
| Standard collect/update loop | `utils/trainer.py` |
| Differentiable TBPTT/update loop | `differentiable_trainer.py` |
| Shared completed-episode evaluation | `evaluation.py` |
| Saved-checkpoint CLI and native Viewer adapter | `policy_evaluation/cli.py`, `policy_evaluation/viewer.py` |
| Learning environment protocol and registry | `env.py` |
| End-to-end config and runtime assembly | `train.py` |

Rollout payloads on the standard path are `TensorDict` objects. Policies
consume observations and write action, log-probability, entropy, and value
fields needed by their algorithm. Differentiable policies must expose
graph-preserving action sampling.

Trainer throughput counts transitions collected during the current `train()`
call. Restored `global_step` remains a cumulative checkpoint counter and is
excluded from that call's SPS numerator.

For actor-critic policies, `policy.obs_groups.actor` and `.critic` select ordered
observation groups. The collector and standard buffer preserve separate
`critic_obs` when configured; evaluation applies the same selection. The PPO
collector also records the old Gaussian mean and standard deviation for KL
measurement. These rollout fields have explicit dimensions in the buffer.

`evaluate_episodes` retains `first_completed` allocation as its compatibility
default. Optional `balanced` allocation fixes per-arena quotas before stepping,
preventing completion-time censoring when episode durations differ. Strict peg
final assessment opts into balanced quotas; historical training validation keeps
the default loaded by its running process.
Optional `on_episode` receives CPU records only for scored completions; strict peg evaluation persists and verifies these records as a hashed JSONL file.

Read [training and extension](training.md) for evaluation, checkpoints,
distributed ownership, official examples and adding algorithms/policies/envs.

## Native policy Viewer

`embodichain eval-policy --viewer` supplies `EmbodiChainTaskEnvironment` to
DexSim's Motion Policy Evaluator, preserving the original task's reset, step,
observation and action path. Supplying an Environment means the adapter owns
its camera lifecycle; Kit's default flat-ground camera is not applied to it.

Headless evaluation accepts `--renderer no-render`. When a saved training
configuration uses NoRender, Viewer evaluation defaults to Hybrid; an explicit
native renderer overrides that choice. `--viewer --renderer no-render` is
rejected before creating the simulation.

Tasks opt in through `PolicyViewerCameraCfg` and
`get_policy_viewer_target_pose()` (world XYZ + XYZW). The six bundled flat
velocity tasks own their presets; `policy_evaluation/_viewer_camera.py` owns
reset framing, X-Y translation and tracking/free-view transitions. Other tasks
retain their existing camera. Use native window operations so manual orbit and
zoom persist and the existing `SimulationManager` window recorder sees the
same view. Window camera operations are queued to the render thread; reading a
pose immediately after an update can still return the preceding rendered pose.

Validate changes with `tests/learning/rl/policy_evaluation/test_viewer.py` and
an actual native Viewer run. User controls and recording are described in the
[policy evaluation guide](../../../docs/source/guides/policy_evaluation.md).

General Torch layers and models live beside `rl` under `learning/layers` and
`learning/models`. RL-specific Policy, ActorOnly, ActorCritic and the policy
registry remain in `learning/rl/policies`; their consumers import from
`embodichain.learning.rl.policies`. Shared MLPs, normalization and MLP construction
import from `embodichain.learning.models`. See [learning layers](../learning-layers/learning-layers.md)
for differentiable IK ownership. Root exports load on demand; importing a
lightweight APG component must not eagerly load the simulator-backed Trainer.

## Invariants

Simulator `train_from_config(..., on_trainer_ready=...)` invokes optional
instrumentation after checkpoint restoration and before collection, under the
normal runtime cleanup boundary. The callback receives the native Trainer,
including train/evaluation environments; lightweight environment routing rejects
it. `scripts/benchmark/assembly/train_peg.py` uses this hook for read-only native
capacity checks before auto-reset, while preserving the native PPO path.

Blind assembly uses the registered `proprioceptive_actor_critic` policy with
explicit actor/critic groups. Its `architecture_version` preserves checkpoint
semantics: omitted versions retain `legacy_v1` with previous history followed by
current; `official_v1` matches the author's public actor export with current
followed by history including current. Deployments must pair the policy version
with the manager's `history_layout` and environment's
`proprioception_convention`; equal tensor dimensions do not imply equal feature
semantics. The peg study nominal and development configurations opt into the
official version; other assembly profiles retain their declared legacy version.
The official inference normalizer follows the exported formula, while its
training-statistics updates and privileged critic remain native conventions.

Contact labels travel as state-aligned training-only rollout fields. PPO's
optional auxiliary decoder loss recomputes the latent during minibatch updates
so gradients reach the encoder. Normalization statistics stay fixed through
collection and every PPO epoch, then update for the next rollout. Actor export
excludes the critic and decoder. See `examples/learning/blind_assembly/` for the
public-reference record, nominal development recipe, suite generation and policy
capacity. Inference alignment and tensor capacity do not qualify task learning
or simulator throughput.

Auto-reset timeouts retain completed-row terminal observations and row IDs.
The collector evaluates the terminal state before using its value for timeout
bootstrap; true termination suppresses bootstrap even when truncation is also
set. Legacy environments without terminal observations retain their previous
fallback contract.

Training snapshots freeze physical environment/embodiment/scene components
relative to their declaring deployment before moving configs into a run.
An optional unwrapped environment `export_checkpoint_gym_config` hook owns
materialized task resources; assembly uses it to pin the actual robot and
manifest. Explicit `./` resource paths resolve relative to the saved config,
including `source_urdf` and `assembly_manifest`. Raw checkpoint context remains
unchanged so resource packaging does not redefine resume compatibility.

- The selected environment must expose batched observation/action spaces and
  `num_envs`.
- Policy observation and action dimensions must match the built environment.
- An algorithm's `RolloutKind` must match its collector, rollout type, and
  trainer.
- The standard buffer holds at most one unconsumed rollout.
- APG must retain differentiable rewards until its optimizer boundary;
  `detach_state()` must not reset or resample the task.
- Complete APG must reset once per independent rollout, detach only after
  backward, and exclude post-terminal auto-reset rewards from its objective.
- Observation statistics stay frozen during each complete rollout; semantic
  mask/type fields remain unnormalized.
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
| Long-horizon APG accuracy is lower than the reference | `rollout_mode` is still segmented, the scheduled horizon was truncated, return scaling is missing, or observation normalization differs |
| One environment poisons every APG row | Action-adjoint clipping is disabled or non-finite row filtering is bypassed |
| GRPO reshape or grouping fails | `num_envs` is not divisible by `group_size` |
| Evaluation never completes | The environment does not emit completed asynchronous episodes or terminal metrics correctly |
| Output/checkpoint directories diverge across ranks | Distributed run metadata was not coordinated through rank zero |
