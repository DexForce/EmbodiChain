# Configuration-defined manipulation RL

A supported manipulation task can select observations, actions, rewards, reset
events and measured outcome conditions without defining a Python environment.
The loader registers the common `ConfiguredRLEnv` under the deployment's `id`.
The original Python-owned task route remains available.

## Deployment

Use an ordinary inline Gym configuration, or select existing physical environment
and embodiment components. Keep exactly one file-owned physics backend. Add a
root `rl_task` mapping; it cannot coexist with `task_program` in one deployment.
Existing `env.actions`, `env.observations`, `env.rewards` and `env.events` retain
their current manager contracts and ownership rules.

```yaml
rl_task:
  success:
    goal:
      func: object_near_target
      params:
        entity_cfg: {uid: cube}
        target_pose_key: goal_pose
        axes: [0, 1]
        tolerance: 0.1
  failure:
    fallen:
      func: object_below_height
      params:
        entity_cfg: {uid: cube}
        height: -0.1
```

Success requires every declared success condition; an empty set never defaults
to success. Any failure or custom `truncation` condition ends its own row.
`max_episode_steps` adds the ordinary time limit. Physical success remains
visible in `info["task_outcome"]["physical_success"]`; accepted success excludes
invalid evidence, failure and truncation. The imminent time limit is applied
before success rewards and termination flags are computed.

The initial built-ins use environment-local metres. `object_near_target` checks
a Euclidean distance over selected axes; its default bound is strict.
`object_below_height` checks a strict local Z bound. Task Program's existing
near-target validator shares the measured implementation and retains inclusive
XYZ distance semantics.

## Goal state and lifecycle

Declare virtual target poses with the existing `randomize_target_pose` event in
`reset` mode and a matching `store_key`. Virtual goals that change during an
episode are outside this initial replay contract. Missing or malformed targets
fail after reset events, before returning the initial observation.

Conditions update once after physics and interval events. Queries of info,
evaluation, final task success and episode metadata read cached owned values.
Stateful conditions subclass `Functor` and implement `reset(env_ids)`; selected
row resets do not sample or advance another row's condition history.

New condition functors implement `func(env, **params) -> ConditionResult`.
Register their module through the existing manager-module mechanism, use
`SceneEntityCfg` for entities, and return boolean `(num_envs,)` masks and
batch-aligned metric tensors on the environment device. Metric names, shapes
and dtypes must stay stable during an episode.

Registration is process-local: load the deployment through `config_to_cfg`
before calling `gym.make(id)`. Reloading the same declaration is idempotent;
changing conditions, manager ordering, physical declarations or episode limits
under the same ID fails. Device, batch size and seed are runtime choices.

## PushCube

The new `PushCubeConfiguredRL-v1` deployment preserves the existing concrete
robot, TCP, scene, action, observation and reward setup. It classifies falling
as task failure; the legacy `PushCubeRL` keeps its original falling-as-truncation
behavior. Existing task IDs and agent files are unchanged.

```bash
embodichain train-rl --config extensions/embodichain_tasks/configs/tasks/manipulation/push_cube/agents/ppo.configured_rl.yaml
```

## Recording and dynamic replay

Trajectory metadata includes the original virtual-target context as
`task_initial_state`. Dynamic replay restores that context after restoring the
physical scene and before returning observations or reseeding recordings.
Measured task replay requires one environment and full resets. Missing context
is rejected; a newly randomized goal cannot stand in for the recorded goal.
Kinematic/control playback does not establish measured physical acceptance.
Replay exhaustion stops playback and is distinct from the task's own time limit.

## Fixed-budget training comparison

```bash
python examples/learning/configured_manipulation/compare_push_cube.py \
  --gpus 3 4 --output output/configured_rl_comparison
```

Choose currently available GPU indices. The runner permits at most two simulator
processes, one per selected GPU. It runs legacy/configured variants with seeds
42/43/44, each with 64 environments, rollout length 1024 and 1000 PPO updates.
W&B and camera recording are disabled; monitoring runs every 50 updates.

Each worker captures the untrained policy and reloads the final checkpoint.
Both policies use the same 100 single-environment case seeds (20000–20099),
deterministic actions and recorded initial object/goal/robot state. Final
qualification requires the configured three-seed mean to be within five
percentage points of legacy and at least twenty points above the untrained
configured policy. Actual results, including failure to meet these thresholds,
are preserved without expanding budgets or reducing thresholds.

Each result records its configuration/implementation fingerprint, budget,
complete case list and checkpoint. Reusing an incompatible result directory
fails explicitly. Reduced `--iterations` / `--cases` runs are smoke checks and
cannot claim formal acceptance. The output contains progress JSON, learning
curves, evaluation history, initial policies, checkpoint/run references and a
comparison report.
