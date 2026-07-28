# embodichain_tasks

Official task environments for [EmbodiChain](https://github.com/DexForce/EmbodiChain).

This package contains the tableware, reinforcement-learning, and special task
environments that used to live inside the core `embodichain` package. It is a
separate, pip-installable package that depends on `embodichain` and registers
itself as a task package through the `embodichain.tasks` entry point.

## Installation

Install EmbodiChain first, then this package, both in development mode:

```bash
cd EmbodiChain

pip install -e embodichain_tasks/
```

Installing `embodichain_tasks` registers its `embodichain.tasks` entry point so
the unified `embodichain` CLI can discover every task it ships.

## Running a task

Use the unified `embodichain` CLI shipped with EmbodiChain. It discovers all
installed task packages and launches any registered environment; the task is
selected by the `"id"` field of the gym config.

```bash
# Data generation mode
embodichain run-env --gym_config embodichain_tasks/configs/gym/pour_water/gym_config.json

# Preview mode
embodichain run-env --gym_config embodichain_tasks/configs/gym/pour_water/gym_config.json --preview

# Equivalent invocations
python -m embodichain run-env --gym_config embodichain_tasks/configs/gym/pour_water/gym_config.json
python -m embodichain.lab.scripts.run_env --gym_config embodichain_tasks/configs/gym/pour_water/gym_config.json
```

## How registration works

Importing `embodichain_tasks` recursively imports every sub-package, which
triggers each task's `@register_env` decorator and registers it in the
gymnasium registry. The unified CLI calls `discover_task_packages()` (from
`embodichain.lab.gym.utils.registration`) at startup, which imports this
package via its entry point. See
`docs/superpowers/specs/2026-07-07-task-env-refactor-design.md` for the full
design.

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
2. **Implement the environment** as an `EmbodiedEnv` subclass and register it
   with `@register_env("YourTask-v1")` (see `embodichain_tasks/tableware/` for
   examples). Importing your package must reach every task module so the
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
