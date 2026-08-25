# embodichain_tasks

Official task environments for [EmbodiChain](https://github.com/DexForce/EmbodiChain).

This source tree contains the official task environments that used to live
inside the core `embodichain` import package. It is bundled into the main
`embodichain` wheel as the separate `embodichain_tasks` import package and
registered through the `embodichain.tasks` entry point. It has no independent
distribution metadata or version.

Tasks are organized by domain and task identity, not by solution method. Every
task package keeps its environment registration in `task.py`; simulator scene
and MDP settings remain in JSON/YAML. Optional expert bindings and policy
configs live below the same task:

```text
embodichain_tasks/<domain>/<task>/task.py
configs/tasks/<domain>/<task>/env.{json,yaml}
configs/tasks/<domain>/<task>/expert/program.yaml
configs/tasks/<domain>/<task>/agents/<algorithm>.yaml
```

Simple tasks contain only `task.py` and an environment config. Expert or RL
files are added only when that task provides the corresponding solution.

## Migrating from the solution-first layout

Gym IDs are unchanged, but direct module imports and repository-style config
paths must use the task-first locations:

| Previous owner | Task-first owner |
| --- | --- |
| `embodichain_tasks.rl.basic.<task>` | `embodichain_tasks.classic_control.<task>.task` |
| `embodichain_tasks.rl.push_cube` | `embodichain_tasks.manipulation.push_cube.task` |
| `embodichain_tasks.expert_program.<task>` | `embodichain_tasks.manipulation.<task>.task` and `<task>.expert.binding` |
| `configs/gym/`, `configs/expert_program/`, `configs/agents/rl/` | `configs/tasks/<domain>/<task>/` |

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
embodichain run-env --gym_config embodichain_tasks/configs/tasks/tableware/pour_water/env.json

# Preview mode
embodichain run-env --gym_config embodichain_tasks/configs/tasks/tableware/pour_water/env.json --preview

# Equivalent invocations
python -m embodichain run-env --gym_config embodichain_tasks/configs/tasks/tableware/pour_water/env.json
python -m embodichain.lab.scripts.run_env --gym_config embodichain_tasks/configs/tasks/tableware/pour_water/env.json
```

## How registration works

Importing `embodichain_tasks` recursively imports every sub-package, which
triggers each task's `@register_env` decorator and registers it in the
gymnasium registry. The unified CLI calls `discover_task_packages()` (from
`embodichain.lab.gym.utils.registration`) at startup, which imports this
package via its entry point. See the
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
2. **Implement the environment** in `<domain>/<task>/task.py` as an
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
