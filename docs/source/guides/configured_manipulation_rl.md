# Configuration-defined manipulation RL

A supported manipulation task can select observations, actions, rewards, reset
events and measured outcome conditions without defining a Python environment.
The loader registers the common `EmbodiedEnv` under the deployment's `id` and
sets `EmbodiedEnvCfg.rl_task`. This optional declaration enables `TerminationManager`
on the existing environment; ordinary environments keep their existing success
behavior. The original Python-owned task route remains available.

## Deployment

Use an ordinary inline Gym configuration, or select existing physical environment
and embodiment components. Keep exactly one file-owned physics backend. Add a
root `rl_task` mapping; it cannot coexist with `task_program` in one deployment.
Task Program adapters and acceptance remain separate from RL outcomes.
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

`TerminationManager.compute()` runs once after physics and interval events,
with the host's time-limit mask for that transition. Observations, rewards,
Gym termination flags and episode metadata consume the same cache; recording
does not alter acceptance. The environment owns step timing, reset targets and
replay context. The manager owns condition execution, aggregation and selected
row reset. Queries of info, evaluation and final task success return owned values.
`ignore_terminations` suppresses Gym termination flags while preserving measured
success/failure and time-limit truncation.
Stateful conditions subclass `Functor` and implement `reset(env_ids)`; selected
row resets do not sample or advance another row's condition history.

New condition functors implement `func(env, **params) -> ConditionResult`.
Register their module through the existing manager-module mechanism, use
`SceneEntityCfg` for entities, and return boolean `(num_envs,)` masks and
batch-aligned metric tensors on the environment device. Metric names, shapes
and dtypes must stay stable during an episode.
Use a class functor when entity or geometric parameters need validation at
construction, as the built-in conditions do. Validate those parameters in
`__init__(cfg, env)` without sampling reset targets; the generic manager does not
know individual condition implementations.

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
These requirements apply only when the environment enables RL outcomes;
ordinary and Task Program replay keep their existing contracts.
Kinematic/control playback does not establish measured physical acceptance.
Replay exhaustion stops playback and is distinct from the task's own time limit.
