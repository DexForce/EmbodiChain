# Task Environment Decoupling: Design Spec

**Date**: 2026-07-07
**Status**: Draft
**Reference**: IsaacLab task architecture analysis

## 1. Motivation

Currently all task environments live inside the core `embodichain` package under
`embodichain/lab/gym/envs/tasks/`. This creates several problems:

- **Core package bloat** — tableware, RL, and special tasks are bundled with
  framework code, increasing install size and cognitive load.
- **No third-party extension path** — downstream projects like RoboSynChallenge
  must monkey-patch internals (`DEFAULT_MANAGER_MODULES`, `get_data_path`,
  `LeRobotRecorder`) to integrate.
- **Implicit dependency** — downstream projects use `sys.path` hacks instead of
  proper Python package dependencies.
- **Duplicated launch scripts** — every downstream project copies its own
  `run_env.py` from the core.

## 2. Goals

1. **Extract all task environments** from `embodichain` into a separate
   `embodichain_tasks` package within the same repository.
2. **Enable third-party task packages** via standard setuptools
   `entry_points`, allowing external projects to register tasks and init hooks
   without modifying `embodichain` source.
3. **Provide a unified launch script** — one `run_env` CLI that discovers and
   launches any registered task from any installed package.
4. **Full backward compatibility** — existing env IDs, JSON/YAML config formats,
   `@register_env` API, and `gym.make()` workflow remain unchanged.

## 3. Non-Goals (this iteration)

- RoboSynChallenge is NOT modified in this iteration. It will adopt the new
  mechanism in a follow-up.
- `BaseAgentEnv` is moved out with tasks but will be replaced by a new Agent
  base class in future work. No redesign of the agent infrastructure.
- Config format remains JSON/YAML. No migration to Python config classes.

## 4. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│  python -m embodichain.lab.scripts.run_env            │
│  (unified CLI, no task-specific imports)              │
└──────────┬───────────────────────────────────────────┘
           │
    1. discover_task_packages()
       └─ importlib.metadata.entry_points("embodichain.tasks")
          imports: embodichain_tasks, robosynchallenge, ...
              └─ triggers @register_env → gym.register()
           │
    2. execute_init_hooks()
       └─ importlib.metadata.entry_points("embodichain.init")
          calls: robosynchallenge.init:register()
              └─ register_manager_modules(...)
              └─ install_asset_resolver()
              └─ install_lerobot_recorder_override()
           │
    3. build_env_cfg_from_args(args)
       └─ loads gym_config.json → EmbodiedEnvCfg
           │
    4. gym.make(id, cfg=env_cfg)
       └─ gymnasium registry → instantiated env
```

## 5. Package Structure Changes

### 5.1 New package: `embodichain_tasks/`

```
embodichain_tasks/
├── pyproject.toml
├── VERSION
├── embodichain_tasks/
│   ├── __init__.py              # import_packages() recursive import
│   ├── utils/
│   │   └── importer.py          # import_packages() utility
│   ├── tableware/
│   │   ├── __init__.py
│   │   ├── base_agent_env.py    # migrated from core (to be deprecated)
│   │   ├── pour_water/
│   │   │   ├── __init__.py
│   │   │   ├── pour_water.py    # @register_env("PourWater-v3")
│   │   │   └── action_bank.py
│   │   ├── rearrangement.py
│   │   ├── stack_blocks_two.py
│   │   ├── stack_cups.py
│   │   ├── scoop_ice.py
│   │   ├── blocks_ranking_rgb.py
│   │   ├── blocks_ranking_size.py
│   │   ├── match_object_container.py
│   │   └── place_object_drawer.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── build_env.py         # migrated from tasks/rl/__init__.py
│   │   ├── push_cube.py
│   │   └── basic/
│   │       └── cart_pole.py
│   └── special/
│       └── simple_task.py
└── configs/                     # task JSON configs (migrated from various locations)
    ├── pour_water/
    ├── rearrangement/
    └── ...
```

### 5.2 Core package `embodichain/` removals

```
embodichain/lab/gym/envs/tasks/    # ENTIRE DIRECTORY REMOVED
```

These stay in core (framework-level, not task-specific):
- `embodichain/lab/gym/envs/base_env.py` — `BaseEnv`, `EnvCfg`
- `embodichain/lab/gym/envs/embodied_env.py` — `EmbodiedEnv`, `EmbodiedEnvCfg`
- `embodichain/lab/gym/envs/managers/` — all manager infrastructure
- `embodichain/lab/gym/envs/action_bank/` — `ActionBank`, `@tag_node`, `@tag_edge`
- `embodichain/lab/gym/envs/wrapper/` — `NoFailWrapper`
- `embodichain/lab/gym/utils/` — `gym_utils`, `registration`, `misc`

### 5.3 Core package additions

- `embodichain/lab/gym/utils/registration.py` — new functions:
  - `discover_task_packages()` — entry_points-based task discovery
  - `execute_init_hooks()` — entry_points-based init hook execution
- `embodichain/lab/gym/utils/gym_utils.py` — new functions:
  - `register_manager_modules(modules)` — register custom manager modules
  - `get_manager_modules()` — get all registered manager modules

## 6. Registration & Discovery Mechanism

### 6.1 setuptools entry_points

Task packages declare themselves in `pyproject.toml`:

```toml
# embodichain_tasks/pyproject.toml
[project.entry-points."embodichain.tasks"]
"embodichain_tasks" = "embodichain_tasks"
```

Third-party packages follow the same pattern:

```toml
# robosynchallenge/pyproject.toml (future)
[project.entry-points."embodichain.tasks"]
"robosynchallenge" = "robosynchallenge.tasks"
```

### 6.2 Discovery function

```python
# embodichain/lab/gym/utils/registration.py

import importlib.metadata

def discover_task_packages() -> list[str]:
    """Import all registered task packages via entry_points.

    Returns:
        List of imported package names.
    """
    imported = []
    for ep in importlib.metadata.entry_points(group="embodichain.tasks"):
        importlib.import_module(ep.value)
        imported.append(ep.name)
    return imported
```

### 6.3 Task package auto-import

Each task package's `__init__.py` uses `import_packages()` (same pattern as
IsaacLab's `isaaclab_tasks`):

```python
# embodichain_tasks/__init__.py
from .utils.importer import import_packages
_BLACKLIST = ["utils"]
import_packages(__name__, _BLACKLIST)
```

`import_packages()` recursively walks all sub-packages, importing each module.
This triggers each task's `__init__.py`, which calls `@register_env` →
`gym.register()`. After `discover_task_packages()` returns, all tasks from all
installed packages are registered in `gymnasium.registry`.

### 6.4 @register_env — unchanged

The `@register_env` decorator continues to:
1. Record the env class in `REGISTERED_ENVS` dict
2. Call `gym.register()` with `entry_point=partial(make, env_id=uid)`

All existing env IDs remain unchanged (e.g., `PourWater-v3`, `PushCubeRL`,
`CartPoleRL`, `StackBlocksTwo-v1`, etc.).

## 7. Init Hooks Mechanism

### 7.1 Declaration

Task packages optionally declare init hooks:

```toml
[project.entry-points."embodichain.init"]
"robosynchallenge" = "robosynchallenge.init:register"
```

### 7.2 Hook signature

```python
# robosynchallenge/init.py (future)

def register() -> None:
    """EmbodiChain init hook — called before env creation."""
    from embodichain.lab.gym.utils.gym_utils import register_manager_modules
    from robosynchallenge.data.asset_resolver import install_embodichain_asset_resolver
    from robosynchallenge.managers.datasets import install_lerobot_recorder_override

    register_manager_modules([
        "robosynchallenge.managers.actions",
        "robosynchallenge.managers.datasets",
        "robosynchallenge.managers.events",
        "robosynchallenge.managers.observations",
    ])
    install_embodichain_asset_resolver()
    install_lerobot_recorder_override()
```

### 7.3 Hook execution

```python
# embodichain/lab/gym/utils/registration.py

def execute_init_hooks() -> list[str]:
    """Execute all registered init hooks.

    Hooks are called in entry_points declaration order.
    Exceptions from one hook do not prevent others from executing.

    Returns:
        List of hook names that were executed successfully.
    """
    executed = []
    for ep in importlib.metadata.entry_points(group="embodichain.init"):
        try:
            module_name, func_name = ep.value.split(":")
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            func()
            executed.append(ep.name)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                f"Init hook '{ep.name}' failed", exc_info=True
            )
    return executed
```

### 7.4 manager module registration API

```python
# embodichain/lab/gym/utils/gym_utils.py

_EXTRA_MANAGER_MODULES: list[str] = []

def register_manager_modules(modules: list[str]) -> None:
    """Register additional manager modules for functor resolution.

    These modules are searched when resolving functor function names
    from config strings (in addition to the built-in DEFAULT_MANAGER_MODULES).
    """
    for m in modules:
        if m not in _EXTRA_MANAGER_MODULES:
            _EXTRA_MANAGER_MODULES.append(m)

def get_manager_modules() -> list[str]:
    """Get all registered manager modules."""
    return DEFAULT_MANAGER_MODULES + _EXTRA_MANAGER_MODULES
```

All call sites that currently reference `DEFAULT_MANAGER_MODULES` directly
(e.g., `config_to_cfg()`, `find_function_from_modules()`) are updated to use
`get_manager_modules()`.

## 8. Unified Launch Script

### 8.1 Changes to `run_env.py`

```python
# embodichain/lab/scripts/run_env.py

def main():
    args = parse_args()

    # Step 1: Discover all task packages via entry_points
    discover_task_packages()

    # Step 2: Execute init hooks (register managers, asset resolvers, etc.)
    execute_init_hooks()

    # Step 3: Build env config from CLI args + JSON config file
    env_cfg = build_env_cfg_from_args(args)

    # Step 4: Create environment via gymnasium
    env = gym.make(id=env_cfg["id"], cfg=env_cfg)

    # Step 5: Run data generation or preview
    if args.preview:
        preview(env)
    else:
        generate_and_execute_action_list(env, ...)
```

### 8.2 CLI — unchanged

All existing CLI arguments remain:

```
--gym_config       Path to gym config JSON/YAML
--action_config    Path to action config JSON
--num_envs         Number of parallel environments
--device           cpu | cuda
--headless         Run headless
--renderer         auto | hybrid | fast-rt | rt
--arena_space      Arena space size
--gpu_id           GPU device ID
--preview          Interactive preview mode
--filter_visual_rand       Disable visual randomization
--filter_dataset_saving    Disable dataset saving
--max_episodes     Override max episodes
```

### 8.3 Cross-package config paths

Config files can live anywhere on the filesystem — the `run_env` script does
not assume configs are inside the `embodichain` package:

```bash
# Official tasks
python -m embodichain.lab.scripts.run_env \
    --gym_config ../embodichain_tasks/configs/pour_water/random/gym_config.json

# Third-party tasks (future)
python -m embodichain.lab.scripts.run_env \
    --gym_config ../RoboSynChallenge/configs/click_bell/random/gym_config.json
```

Task identity is determined solely by the `"id"` field in the gym config JSON,
which must match a registered gym environment ID.

## 9. Backward Compatibility

### 9.1 Guarantees

| Item | Strategy |
|------|----------|
| Env IDs | Unchanged — `@register_env(uid)` values preserved |
| JSON/YAML config format | Unchanged — same schema |
| `gym.make(id, cfg=...)` | Unchanged — standard gymnasium interface |
| `@register_env` API | Unchanged — same signature |
| `EmbodiedEnv` / `BaseEnv` import path | Unchanged — stays in `embodichain.lab.gym.envs` |
| `Functor` / manager infrastructure | Unchanged — stays in `embodichain.lab.gym.envs.managers` |
| `ActionBank` | Unchanged — stays in `embodichain.lab.gym.envs.action_bank` |
| `build_env_cfg_from_args()` / `config_to_cfg()` | Unchanged — stays in `embodichain.lab.gym.utils` |
| `run_env` CLI arguments | Unchanged — same flags and semantics |

### 9.2 Breaking change

Direct imports of task classes from `embodichain.lab.gym.envs.tasks.*` will
break. This is not the standard usage path (tasks are created via
`gym.make(env_id)`), but any code doing direct imports needs to update the
import path to `embodichain_tasks.*`.

### 9.3 Deprecation shim

`embodichain/lab/gym/envs/tasks/__init__.py` becomes a thin shim:

```python
# This module is deprecated. Import from embodichain_tasks instead.
import warnings
warnings.warn(
    "embodichain.lab.gym.envs.tasks is deprecated. "
    "Import from embodichain_tasks instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new package for backward compatibility
from embodichain_tasks.tableware.base_agent_env import BaseAgentEnv  # noqa
from embodichain_tasks.tableware.pour_water.pour_water import (
    PourWaterEnv, PourWaterAgentEnv,
)
from embodichain_tasks.tableware.rearrangement import (
    RearrangementEnv, RearrangementAgentEnv,
)
# ... (all other task classes)
```

This ensures existing direct imports continue to work with a deprecation
warning. The shim is removed in the next major version.

## 12. Resolved Decisions

- **Config location**: `embodichain_tasks/configs/` — configs ship with the
  task package, not in a separate top-level directory.
- **Package type**: `embodichain_tasks` is a regular pip-installable package
  that depends on `embodichain`. Not a namespace package.
- **`import_packages()` utility**: A lightweight implementation lives in
  `embodichain_tasks/utils/importer.py`, following the IsaacLab pattern but
  without Omniverse-specific logic.
- **Entry point group names**: `embodichain.tasks` for task packages,
  `embodichain.init` for init hooks. Both follow the convention
  `<package>.<purpose>`.
- **Hook execution order**: Entry points are iterated in declaration order.
  No priority system — if ordering matters, the downstream package controls
  it by its position in the entry_points list.

## 10. Third-Party Integration Pattern (Reference)

When RoboSynChallenge (or any downstream project) adopts this mechanism, the
changes are minimal:

### 10.1 Changes required (3 files)

| File | Change |
|------|--------|
| `pyproject.toml` | Add `[project.entry-points."embodichain.tasks"]` and `[project.entry-points."embodichain.init"]` |
| `robosynchallenge/init.py` | New file with `register()` function |
| `scripts/run_env.py` | Delete (use core `run_env`) |

### 10.2 No changes required

- All JSON config files
- All `@register_env` decorators
- All task environment classes
- All `ActionBank` subclasses
- All custom functors
- Shell launch scripts (just update the `RUN_ENV` variable)

## 11. Implementation Plan (High-Level)

1. Create `embodichain_tasks/` package skeleton with `pyproject.toml`
2. Port `import_packages()` utility from IsaacLab pattern
3. Migrate all task modules from `embodichain/lab/gym/envs/tasks/` to
   `embodichain_tasks/embodichain_tasks/`, updating internal imports
4. Migrate task JSON configs to `embodichain_tasks/configs/`
5. Add `discover_task_packages()` and `execute_init_hooks()` to
   `embodichain/lab/gym/utils/registration.py`
6. Add `register_manager_modules()` / `get_manager_modules()` to
   `embodichain/lab/gym/utils/gym_utils.py`
7. Update all `DEFAULT_MANAGER_MODULES` references to use `get_manager_modules()`
8. Update `run_env.py` to call `discover_task_packages()` + `execute_init_hooks()`
9. Add deprecation shim in `embodichain/lab/gym/envs/tasks/__init__.py`
10. Run full test suite; verify all tasks launch via unified `run_env`
11. Update project documentation

