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
remain in JSON/YAML. Supported Task Program tasks may instead be entirely
configuration-defined. Optional programs and policy configs live below the
same task:

```text
embodichain_tasks/<category-path>/<task>.py
configs/tasks/<category-path>/<task>/env.{json,yaml}          # inline runnable, or reusable env.yaml
configs/tasks/<category-path>/<task>/task.<embodiment>.yaml   # componentized runnable deployment
configs/tasks/<category-path>/<task>/task_program/program.yaml
configs/tasks/<category-path>/<task>/task_program/integration.yaml
configs/tasks/<category-path>/<task>/agents/<algorithm>.yaml
configs/components/embodiments/<embodiment>.yaml
configs/components/execution_policies/<policy>.yaml
```

The category path begins with a top-level task family and may include a
subdomain: tableware tasks use `manipulation/tableware`, while general
manipulation tasks can stay directly under `manipulation`. The Python entry
stays flat beneath its owning category; the task-local configuration directory
remains because it can own environment, Task Program, and policy artifacts.
A simple import-registered task may keep all physical and manager values in its
runnable `env.json` or `env.yaml`. A componentized task instead gives the pure
physical `env.yaml` an `environment_id` and places the runnable `id` plus
component selections in `task.<embodiment>.yaml`.
A Gym deployment of any kind may select reusable `environment.component` and
`embodiment.component` files instead of repeating environment, robot, sensor,
and scene fields. A physical-only embodiment may omit `skill_profile`; this is
used by the CobotMagic tableware handwritten demos. Standalone
`scene.component` and inline Gym configs remain valid, but component-owned
fields cannot also be declared inline in the same file.

A configuration-defined Task Program environment uses three explicit owners.
The reusable `env.yaml` owns episode/environment values and physical simulation
entities, and contains no Task Program metadata. The integration's nested
`scene_binding` maps canonical `entity_id` values to physical `simulation_uid`
values and owns semantic types and affordances. A thin
`task.<embodiment>.yaml` deployment selects that
`environment.component`, one reusable embodiment, and all three Task Program
components (`program`, `integration`, and `execution_policy`).

An embodiment owns the simulation robot and its sensor suite. Its optional
`skill_profile` owns the logical resources, command presets, and
embodiment-specific Task Program services. Loading a deployment checks that
scene-binding targets exist in the physical scene, validates explicit
scene/embodiment contracts, and registers the common `EmbodiedEnv`, so no
Python task module is needed. Programs contain no concrete embodiment/profile
IDs; changing the `embodiment.component` reference is sufficient when another
embodiment satisfies the same contract. These files intentionally have no
compatibility `version` field.

## Migrating from the solution-first layout

Gym IDs are unchanged, but direct module imports and repository-style config
paths must use the task-first locations:

| Previous owner | Task-first owner |
| --- | --- |
| `embodichain_tasks.rl.basic.<task>` | `embodichain_tasks.classic_control.<task>` |
| `embodichain_tasks.rl.push_cube` | `embodichain_tasks.manipulation.push_cube` |
| `embodichain_tasks.tableware.<task>` | `embodichain_tasks.manipulation.tableware.<task>` |
| Task Program-specific Python task modules | No replacement module for supported config-defined examples; load `configs/tasks/manipulation/<task>/task.<embodiment>.yaml` |
| `configs/tasks/tableware/<task>/` | `configs/tasks/manipulation/tableware/<task>/` |
| `configs/gym/`, `configs/task_program/`, `configs/agents/rl/` | `configs/tasks/<category-path>/<task>/` |

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
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml

# Preview mode
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml --preview

# Equivalent invocations
python -m embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
python -m embodichain.lab.scripts.run_env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
```

## How registration works

Importing `embodichain_tasks` recursively imports every task module, which
triggers its `@register_env` decorator and registers it in the gymnasium
registry. Configuration-defined Task Program IDs are registered later by
`config_to_cfg()` when their task deployment is loaded. Task listing discovers
top-level JSON/YAML deployments by their non-empty `id` field rather than an
`env*` filename prefix, so the pure `env.yaml` component is ignored. The
unified CLI calls `discover_task_packages()` (from
`embodichain.lab.gym.utils.registration`) at
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
2. **Choose the registration path.** For custom Python behavior, implement
   `<category-path>/<task>.py` as an `EmbodiedEnv` subclass and register it
   there with `@register_env("YourTask-v1")`. Importing your package must reach
   every task module so the decorator runs. A supported configuration-defined
   Task Program may omit that module and register its ID when its runnable
   config is loaded.
3. **Write a runnable gym config** (`.json`/`.yaml`) with `id`. It may define
   physical and manager values inline, or select a reusable `env.yaml`, an
   embodiment, and optional Task Program components from
   `task.<embodiment>.yaml`.
4. **Install and run**:
   ```bash
   pip install -e .
   embodichain run-env --gym_config path/to/your/gym_config.json
   ```

If your tasks need custom manager modules (observation/reward/event/action
functors) or asset resolvers, register them from an `embodichain.init` hook
(see `register_manager_modules()` in `embodichain.lab.gym.utils.gym_utils`).
The template uses explicit package imports; `embodichain_tasks` uses the
`import_packages()` helper for recursive import.
