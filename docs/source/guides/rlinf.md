# Using EmbodiChain with RLinf

RLinf is an optional external training backend for EmbodiChain environments.
EmbodiChain owns the task package, Gym deployment, robot/scene configuration,
and environment semantics. RLinf owns the actor, rollout, trainer, Ray
placement, checkpoints, and distributed execution.

This guide documents the current integration boundary. It is experimental:
the simple inline, low-dimensional state path is supported first, while
componentized deployments, Newton physics, visual observations, and semantic
or chunked actions require the compatibility notes below.

## Choose the training entry point

EmbodiChain's native trainer and RLinf use different configuration formats:

```text
embodichain train-rl
  -> EmbodiChain RL trainer
  -> EmbodiChain Gym environment

RLinf train_embodied_agent.py
  -> RLinf EnvWorker
  -> env_type: embodichain
  -> EmbodiChainEnv
  -> EmbodiChain build_env()
```

Use the native entry point when the algorithm and policy are configured by
EmbodiChain. Use RLinf when the experiment needs RLinf's actor/rollout stack,
placement, or distributed training. An EmbodiChain `agents/*.yaml` or
`agents/*.json` file is not an RLinf configuration and should not be passed to
the RLinf launcher.

## Install the packages in one runtime

Install EmbodiChain, RLinf, and every external task package into the Python
environment used by RLinf workers:

```bash
# EmbodiChain and its official tasks
cd /path/to/EmbodiChain
python -m pip install -e .

# RLinf and its optional runtime dependencies
cd /path/to/RLinf
python -m pip install -e .

# An external EmbodiChain task package, when used
python -m pip install -e /path/to/my_embodichain_tasks
```

For Ray multi-node jobs, install the same versions on every node. Installing a
task source tree only through `PYTHONPATH` is insufficient for automatic task
discovery; the distribution must expose its `embodichain.tasks` entry point.

## Minimal RLinf environment configuration

The RLinf environment stanza selects the EmbodiChain adapter and points to a
runnable EmbodiChain Gym deployment:

```yaml
env_type: embodichain

# Use an absolute path for external or componentized deployments until
# package-resource resolution is enabled by the adapter.
gym_config_path: /path/to/EmbodiChain/embodichain_tasks/configs/tasks/classic_control/cart_pole/env.json

headless: true
sim_device: cuda
state_keys: [qpos, qvel, qf]

seed: 0
group_size: 1
auto_reset: true
ignore_terminations: false
max_episode_steps: 500
```

The current RLinf example is under
`/path/to/RLinf/examples/embodiment/config/embodichain_ppo_cart_pole.yaml`.
Use the task-centric path shown above for the current EmbodiChain checkout;
older RLinf examples may still refer to the retired
`configs/agents/rl/basic/` layout.

The policy configuration must match the resulting observation and action
dimensions. For example, the CartPole example uses a six-dimensional
`states` tensor and two actions.

## Launch RLinf

From the RLinf embodiment example directory, select the RLinf configuration:

```bash
cd /path/to/RLinf/examples/embodiment
python train_embodied_agent.py \
  --config-path config \
  --config-name embodichain_ppo_cart_pole
```

The exact actor, rollout, logging, and cluster options belong to RLinf. Keep
the EmbodiChain `gym_config_path` and observation/action mapping in the RLinf
environment configuration so that the training run is reproducible.

## Register an external task package

An external project uses the same registration contract as the official task
package. Its `pyproject.toml` should declare:

```toml
[project.entry-points."embodichain.tasks"]
my_embodichain_tasks = "my_embodichain_tasks"
```

The package initializer must import every task module that contains a
registration decorator. A custom Python environment is registered as usual:

```python
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env


@register_env("MyTask-v1", max_episode_steps=500, supports_rl=True)
class MyTaskEnv(EmbodiedEnv):
    ...
```

If the package supplies custom observation, reward, event, or action functors,
register them from an `embodichain.init` hook:

```toml
[project.entry-points."embodichain.init"]
my_embodichain_tasks = "my_embodichain_tasks.bootstrap:initialize"
```

The hook must be safe to call more than once because RLinf constructs
environments in worker processes. The RLinf adapter calls EmbodiChain's
`discover_task_packages()` and `execute_init_hooks()` before loading the Gym
deployment, so no RLinf source edit is required for each new task ID.

## Configuration-defined Task Programs

A supported configuration-defined Task Program does not need a Python task
module. Its runnable deployment can select an environment, embodiment,
program, integration, and execution policy components:

```yaml
id: MyTaskProgram-v1

environment:
  component: env.yaml

embodiment:
  component: ../../../components/embodiments/my_robot.yaml

task_program:
  program: task_program/program.yaml
  integration: task_program/integration.yaml
  execution_policy: ../../../components/execution_policies/my_policy.yaml
```

`config_to_cfg()` registers the deployment ID when the runnable configuration
is loaded. RLinf can use this path because the adapter loads the configuration
before calling `build_env()`. Relative component paths currently need special
care; use an absolute deployment path and verify the adapter version before
running a componentized or Task Program deployment.

## Current compatibility boundary

| Capability | Current status |
| --- | --- |
| Inline `env.json`/`env.yaml` deployment | Supported first |
| Python `@register_env` task | Supported when installed in the RLinf runtime |
| External manager or asset resolver | Supported through `embodichain.init` |
| Componentized `task.<embodiment>.yaml` | Experimental; verify relative paths |
| Configuration-defined Task Program | Experimental |
| Default physics backend | Supported first |
| Newton backend and custom simulation settings | Verify adapter version |
| Low-dimensional `robot` state | Supported through `state_keys` |
| Object, camera, language, or nested observations | Requires explicit adapter mapping |
| Flat continuous `Box` actions | Supported first |
| Dict, semantic, or VLA/chunked actions | Requires an action converter |

The current adapter is intentionally conservative:

- it currently builds `states` from keys under `raw_obs["robot"]`;
- actions are converted to a flat tensor and passed to `env.step()`;
- the adapter must preserve the Gym deployment's source directory when
  resolving reusable components;
- the adapter must preserve the deployment's `sim_cfg`, including the selected
  physics backend, render settings, and arena settings.

Consequently, an environment being discoverable does not by itself guarantee
that its observation semantics, action semantics, or physics configuration are
ready for RLinf training. A task should first pass a worker-local `reset()` /
`step()` smoke test, followed by a short RLinf rollout.

## Troubleshooting

### Task ID is not found

Check that the external distribution is installed in the RLinf Python
environment and declares `embodichain.tasks`. Then verify that the Gym
deployment's `id` exactly matches the `@register_env` ID. IDs should be unique;
avoid relying on `override=True` across independent packages.

### Configuration file is not found

Use an absolute `gym_config_path`. `EMBODICHAIN_PATH` can point to an
EmbodiChain checkout, but it does not register an external package and is not a
general package-resource resolver.

### The task starts but the policy cannot learn

Inspect the mapped observation. The current default state mapping only includes
`qpos`, `qvel`, and `qf` under `robot`; object poses and camera outputs are not
automatically included. Confirm that the RLinf policy's `obs_dim` and
`action_dim` match the adapter output and the EmbodiChain action manager.

### The task uses Newton or custom rendering settings

Confirm that the RLinf adapter version preserves the `sim_cfg` produced by
EmbodiChain's `config_to_cfg()`. If it replaces the complete simulation config,
the task may silently fall back to the default physics backend or lose custom
render settings.

## Ownership and support

EmbodiChain owns the task registration and environment contract. RLinf owns the
training runtime. Report task registration, config composition, robot/scene,
manager, or physics issues to the EmbodiChain integration. Report actor,
rollout, placement, checkpoint, or trainer issues to RLinf.
