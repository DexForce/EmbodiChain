# embodichain_tasks

Official task environments for [EmbodiChain](https://github.com/DexForce/EmbodiChain).

This source tree contains the official task environments that used to live
inside the core `embodichain` import package. It is bundled into the main
`embodichain` wheel as the separate `embodichain_tasks` import package and
registered through the `embodichain.tasks` entry point. It has no independent
distribution metadata or version.

Tasks are organized by task family, optional subdomain, and task identity, not
by solution method. Import-registered tasks keep their environment
registration in a task-named Python module; simulator scene and MDP settings
remain in JSON/YAML. Supported Expert Program tasks may instead be entirely
configuration-defined. Optional programs and policy configs live below the
same task:

```text
embodichain_tasks/<category-path>/<task>.py
configs/tasks/<category-path>/<task>/env.{json,yaml}
configs/tasks/<category-path>/<task>/expert/program.yaml
configs/tasks/<category-path>/<task>/agents/<algorithm>.yaml
```

The category path begins with a top-level task family and may include a
subdomain: tableware tasks use `manipulation/tableware`, while general
manipulation tasks can stay directly under `manipulation`. The Python entry
stays flat beneath its owning category; the task-local configuration directory
remains because it can own environment, Expert Program, and policy artifacts.
A configuration-defined Expert Program task uses
`expert_program_runtime` in `env.json`; loading that config registers the
common `EmbodiedEnv`, so no Python task module is needed.

## Migrating from the solution-first layout

Gym IDs are unchanged, but direct module imports and repository-style config
paths must use the task-first locations:

| Previous owner | Task-first owner |
| --- | --- |
| `embodichain_tasks.rl.basic.<task>` | `embodichain_tasks.classic_control.<task>` |
| `embodichain_tasks.rl.push_cube` | `embodichain_tasks.manipulation.push_cube` |
| `embodichain_tasks.tableware.<task>` | `embodichain_tasks.manipulation.tableware.<task>` |
| `embodichain_tasks.expert_program.<task>` | No replacement module for supported config-defined examples; load `configs/tasks/manipulation/<task>/env.json` |
| `configs/tasks/tableware/<task>/` | `configs/tasks/manipulation/tableware/<task>/` |
| `configs/gym/`, `configs/expert_program/`, `configs/agents/rl/` | `configs/tasks/<category-path>/<task>/` |

## Installation

Install the main EmbodiChain distribution. It includes both the core and
official task import packages:

```bash
cd EmbodiChain
pip install -e .
```

The published wheel is installed with the same single-package command:

```bash
pip install embodichain
```

When upgrading a development environment that previously installed this source
tree separately, remove the legacy editable distribution once:

```bash
pip uninstall -y embodichain_tasks
pip install -e .
```

Installing `embodichain` registers the bundled `embodichain_tasks` entry point
so the unified CLI can discover every official task it ships. Repository-style
paths beginning with `embodichain_tasks/configs/` resolve from either a source
checkout or the installed wheel.

## Running a task

Use the unified `embodichain` CLI shipped with EmbodiChain. It discovers all
installed task packages and launches any registered environment; the task is
selected by the `"id"` field of the gym config.

```bash
# Data generation mode
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json

# Preview mode
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json --preview

# Equivalent invocations
python -m embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json
python -m embodichain.lab.scripts.run_env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/env.json
```

## How registration works

Importing `embodichain_tasks` recursively imports every task module, which
triggers its `@register_env` decorator and registers it in the gymnasium
registry. Configuration-defined Expert Program IDs are registered later by
`config_to_cfg()` when their `env.json` is loaded. The unified CLI calls
`discover_task_packages()` (from `embodichain.lab.gym.utils.registration`) at
startup, which imports this package via its entry point. See the
[task-package discovery utilities](../embodichain/lab/gym/utils/registration.py)
and the [official task package initializer](embodichain_tasks/__init__.py) for
the implementation.

## Extending with your own tasks

External projects can ship their own task packages the same way. The easiest
starting point is the
[embodichain_task_template](https://github.com/DexForce/embodichain_task_template)
repository -- fork it and replace the package with your own.

To add a task environment:

1. **Declare the entry point** in your package's `pyproject.toml` so the
   unified CLI discovers it:
   ```toml
   [project.entry-points."embodichain.tasks"]
   "your_package" = "your_package"
   ```
2. **Implement the environment** in `<category-path>/<task>.py` as an
   `EmbodiedEnv` subclass and register it there with
   `@register_env("YourTask-v1")`. Importing your package must reach every task module so the
   decorator runs -- the template uses explicit imports in `__init__.py`;
   `embodichain_tasks` uses the `import_packages()` helper for recursive import.
3. **Write a gym config** (`.json`/`.yaml`) whose `"id"` matches the registered
   env id, defining the robot, scene, sensors, and manager functors.
4. **Install and run**:
   ```bash
   pip install -e .
   embodichain run-env --gym_config path/to/your/gym_config.json
   ```

If your tasks need custom manager modules (observation/reward/event/action
functors) or asset resolvers, register them from an `embodichain.init` hook
(see `register_manager_modules()` in `embodichain.lab.gym.utils.gym_utils`).
