---
name: add-task-env
description: Use when creating a new task environment for EmbodiChain, including expert demonstration tasks, RL tasks or any EmbodiedEnv subclass
---

# Add Task Environment

Scaffold a new task environment following EmbodiChain's conventions and patterns.

## When to Use

- User asks to create a new task or environment
- User says "add a task", "new env", "create environment for X"

## Steps

### 1. Determine Task Identity

Ask the user:

- **Top-level task family**: `manipulation`, `classic_control`, or `special`
- **Optional subdomain**: a narrower category such as `tableware` under
  `manipulation`
- **Task name** (snake_case, e.g. `pick_place`)
- **Gym ID** (e.g. `PickPlace-v1`)
- **Optional solutions**: scripted expert, Expert Program, or RL policy configs
- **Config format**: JSON or YAML

Combine the task family and optional subdomain into `<category_path>`. Do not
use a solution method such as `rl` or `expert_program` as a category.

### 2. Create the Task Module

Place the environment and its registration at:

```text
embodichain_tasks/embodichain_tasks/<category_path>/<task_name>.py
```

Do not add a same-named task directory or `__init__.py` for a single Python
entry point. Lightweight pure-PyTorch tasks use the same flat
category/task-module layout and register through `@register_learning_env` when
they are not `EmbodiedEnv` subclasses.

Template:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg

__all__ = ["<CamelCaseName>Env"]


@register_env("<GymId>")
class <CamelCaseName>Env(EmbodiedEnv):
    """<One-line description of the task>.

    <Longer description of what the task involves and its reward structure.>
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize the task from its decoded environment config.

        Args:
            cfg: Environment configuration loaded from task-local JSON/YAML.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.
        """
        super().__init__(cfg, **kwargs)

    # Keep only task behavior that cannot be expressed by env config here.
```

Keep `@register_env` in the task-named module; do not create a separate
registration module.

### 3. Add the Environment Config

Create the scene and MDP configuration at:

```text
embodichain_tasks/configs/tasks/<category_path>/<task_name>/env.json
```

Use `env.yaml` when YAML was selected. Robot, scene, sensors, observations,
events, rewards, actions, randomization, and dataset settings belong in this
config. Add Python functors only when the existing registries cannot express
the required behavior.

Optional solution artifacts stay below the same task:

```text
<task config>/expert/program.yaml           # declarative Expert Program
<task config>/agents/<algorithm>.yaml       # RL training configuration
```

Prefer the declarative Expert Program runtime for expert behavior. Do not
scaffold a task-local Action Bank, `BaseAgentEnv`, or expert Python package.
Recorded trajectories are data assets, not Python integration modules.

### 4. Update Exports

Task modules under `embodichain_tasks` are auto-imported via
`import_packages()`. Define exports directly in the task module:

```python
__all__ = ["<CamelCaseName>Env"]
```

The domain `__init__.py` does not need to re-export each task.

### 5. Create Test Stub

Place at `tests/gym/envs/tasks/test_<name>.py` (or `tests/learning/` for
lightweight learning environments).

### 6. Format

```bash
black embodichain_tasks/embodichain_tasks/<category_path>/<task_name>.py
black tests/gym/envs/tasks/test_<name>.py
```

## Checklist

- [ ] File has Apache 2.0 header
- [ ] Uses `from __future__ import annotations`
- [ ] Registration lives in `<task_name>.py` with a unique ID
- [ ] `__all__` defined in the task module
- [ ] No same-named per-task package was created for a single module
- [ ] Python and config paths use the same category hierarchy
- [ ] Scene and MDP are declared in task-local JSON/YAML
- [ ] Expert/RL artifacts exist only when the task provides that solution
- [ ] No task-local Action Bank or `BaseAgentEnv` was introduced
- [ ] Test stub created
- [ ] `black` run on changed Python files
