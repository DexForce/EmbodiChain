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
configs/tasks/<category-path>/<task>/envs/default.yaml  # reusable default backend
configs/tasks/<category-path>/<task>/envs/newton.yaml   # optional Newton backend
configs/tasks/<category-path>/<task>/task.<embodiment>.yaml     # componentized runnable deployment
configs/tasks/<category-path>/<task>/expansion/<profile>.yaml   # task-facing expansion override
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
runnable `env.json` or `env.yaml`. A componentized task instead gives each pure
physical environment variant an `environment_id` and places the runnable `id`
plus component selections in `task.<embodiment>.yaml`.
A Gym deployment of any kind may select reusable `environment.component` and
`embodiment.component` files instead of repeating environment, robot, sensor,
and scene fields. A physical-only embodiment may omit `skill_profile` when its
deployments do not use Task Program. Standalone
`scene.component` and inline Gym configs remain valid, but component-owned
fields cannot also be declared inline in the same file.

A configuration-defined Task Program environment uses explicit owners. The
selected file under `envs/` owns episode/environment values and physical
simulation entities, and contains no Task Program metadata. When a backend needs
native solver/contact values, `envs/newton.yaml` owns those values while
retaining the same scene geometry and control cadence. The integration's nested
`scene_binding` maps canonical `entity_id` values to physical
`simulation_uid` values and owns semantic types and affordances. A thin
`task.<embodiment>.yaml` deployment selects named `default` and `newton`
environment variants. Both select one reusable embodiment and all three Task
Program components (`program`, `integration`, and `execution_policy`). The
launcher selects the concrete backend from the same deployment with
`--physics`; the selected environment file remains the owner of its backend
fields.

Execution policies accept either the original string path or a component
selection with deployment-specific tuning:

```yaml
task_program:
  program: task_program/program.yaml
  integration: task_program/integration.yaml
  execution_policy:
    component: ../../../components/execution_policies/motion_gen_verified.yaml
    overrides:
      motion:
        velocity_limit: 0.5
        dynamic_collision_mode: 'off'
```

Overrides are limited to `motion.sample_count`, `motion.velocity_limit`,
`motion.acceleration_limit`, `motion.dynamic_collision_mode`,
`recovery.action_timeout` and `recovery.goal_rotation_threshold`. They patch
only the declared fields; all other base values are retained. The resolved
values pass the normal policy decoders, including integer, finite-number and
enum validation. Contracts, planner selection, tracking, runner and effect
assurance stay in the base policy. The component path resolves from the
deployment, just as a string reference does. Asset materialization writes the
resolved policy into its standalone output.

Task-facing trajectory expansion is an optional third owner. A runnable task
can reference it without copying the robot or environment declaration:

```yaml
expansion:
  config: expansion/<task-profile>.yaml
```

The referenced file contains `runtime`, `policy`, `overrides`, and optional
`candidate_indices`. Its mappings are merged with any task-local expansion
fields, and its runtime overlay is applied after the selected environment
variant is expanded. This keeps expansion-specific batch size, recorder
events, and augmentation policy beside the task while keeping physical scene
ownership in `envs/`. `run-task` loads the referenced declaration;
relative policy and resource paths inside it resolve from the expansion file.
`--expansion-profile` and `--expansion-candidate-indices` remain
command-line overrides.

An embodiment owns the simulation robot and its sensor suite. Its optional
`skill_profile` owns the logical resources, command presets, and
embodiment-specific Task Program services. Loading a deployment checks that
scene-binding targets exist in the physical scene, validates explicit
scene/embodiment contracts, and registers the common `EmbodiedEnv`, so no
Python task module is needed. Programs contain no concrete embodiment/profile
IDs; changing the `embodiment.component` reference is sufficient when another
embodiment satisfies the same contract. These files intentionally have no
compatibility `version` field.

## Dual-arm task deployments

The following tasks provide `dual_ur5_dh_pgi_140_80` and
`tianji_marvin` catalog deployments:

| Task under `configs/tasks/manipulation/` | Execution |
| --- | --- |
| `hand_over` | Verified two-arm Task Program |
| `repeated_pick_place` | Verified Task Program using the left arm |
| `tableware/pour_water` | Right-arm Task Program with measured segment checks |
| `tableware/blocks_ranking_rgb` | Two-arm Task Program with measured lift/placement checks |
| `tableware/stack_blocks_two` | Single-arm Task Program with measured lift/stack checks |
| `tableware/blocks_ranking_size` | Environment and success checks |
| `tableware/match_object_container` | Environment and success checks |
| `tableware/place_object_drawer` | Environment and success checks |
| `tableware/stack_cups` | Environment and success checks |

Select the file named by the catalog deployment. Most use
`task.<catalog-deployment>.yaml`; the shared-tabletop UR5 hand-over variant uses
`task.dual_ur5.yaml` and the original `task.dual_ur5_dh_pgi_140_80.yaml` keeps its
legacy scene and ID. Existing
CobotMagic deployments, legacy baseline `env.json` paths, and single-arm UR5
pick/place deployments remain available. The baseline JSON files select shared
physical `env.yaml` components; adding a robot variant does not add an expert
to an environment-only task. `ScoopIce-v1` remains retired.

Both embodiments provide `cam_high`; Marvin additionally provides its two
wrist RGB-D cameras. UR5 deployments have one camera, so downstream datasets
must use the selected embodiment's sensor and action layout. Shared tableware
observations normalize named end-effector control parts instead of fixed joint
indices. Robot-specific installation poses live in deployment overrides.

Programs and measured acceptance thresholds are shared across embodiments.
Task integrations retain gripper-specific grasp transforms and arm selections, while the
embodiments own open/grasp commands and grasp generators. The generic
primary/secondary manipulator resources alias the right/left physical resource
leaves; they do not introduce independent arms. Static preflight proves
composition, not physical completion; use the task's measured validators when
qualifying trajectories, seeds, or backend settings.

Verified tabletop policies request explicit planning speed and acceleration
budgets, still capped by robot joint limits. The tabletop verified policy uses
a 0.5 rad/s speed budget with 0.5 rad/s² acceleration. Native trajectory timing is
preserved. The legacy UR5 hand-over keeps its original policy. Verified actions
allow 120 seconds, and the shared hand-over scene
allows 3200 control steps for the full transfer and measured-effect gates.

Example:

```bash
python -m embodichain run-task --gym_config \
  extensions/embodichain_tasks/configs/tasks/manipulation/tableware/stack_blocks_two/task.dual_ur5_dh_pgi_140_80.yaml
```

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
paths beginning with `extensions/embodichain_tasks/configs/` resolve from either a source
checkout or the installed wheel.
The legacy `embodichain_tasks/configs/` prefix remains supported. Sources now
live under `extensions/`; refresh existing editable installations from the
repository root with `pip install -e . --no-deps`.

## Running a task

Use the unified `embodichain` CLI shipped with EmbodiChain. It discovers all
installed task packages and launches any registered environment; the task is
selected by the `"id"` field of the gym config.

```bash
# Data expansion mode
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml

# Newton variant selected from the same deployment
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml --physics newton

# Preview mode
embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml --preview

# Equivalent invocations
python -m embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
python -m embodichain.lab.scripts.run_task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml

# Newton variant selected from the same deployment
python -m embodichain run-task --gym_config extensions/embodichain_tasks/configs/tasks/manipulation/open_drawer/task.ur5.yaml --physics newton
```

## How registration works

Importing `embodichain_tasks` recursively imports every task module, which
triggers its `@register_env` decorator and registers it in the gymnasium
registry. Configuration-defined Task Program IDs are registered later by
`config_to_cfg()` when their task deployment is loaded. Task listing discovers
top-level JSON/YAML deployments by their non-empty `id` field rather than an
`env*` filename prefix, so pure environment components are ignored. The
unified CLI calls `discover_task_packages()` (from
`embodichain.lab.gym.utils.registration`) at
startup, which imports this package via its entry point. See the
[task-package discovery utilities](../../embodichain/lab/gym/utils/registration.py)
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
   physical and manager values inline, or select reusable environment variants,
   an embodiment, and optional Task Program components from
   `task.<embodiment>.yaml`.
4. **Install and run**:
   ```bash
   pip install -e .
   embodichain run-task --gym_config path/to/your/gym_config.json
   ```

If your tasks need custom manager modules (observation/reward/event/action
functors) or asset resolvers, register them from an `embodichain.init` hook
(see `register_manager_modules()` in `embodichain.lab.gym.utils.gym_utils`).
The template uses explicit package imports; `embodichain_tasks` uses the
`import_packages()` helper for recursive import.

## Using registered tasks with RLinf

RLinf workers use the same installed task packages as the EmbodiChain CLI
through the `embodichain.tasks` entry-point group. Install the package in the
worker environment and point the adapter at a runnable Gym deployment. See the
[RLinf integration guide](../../docs/source/guides/rlinf.md) for the integration
boundary and current compatibility notes.

## Dual-arm qualification scope

Local checks use the Default physics backend, one CPU simulation environment,
and seed 0 with the configured reset randomization. The Task Program
combinations use measured program checks. The eight baseline
combinations were constructed, reset and stepped; they do not gain expert
programs. PourWater uses Hybrid for its glTF asset loader. Other motion checks
run headlessly without sensors; an additional Hybrid check verified the shared
UR5 top camera at 960 by 540 pixels. These checks do not qualify every random
seed, physics backend or sensor configuration.
