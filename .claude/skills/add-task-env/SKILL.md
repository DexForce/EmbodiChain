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

### 1. Determine Task Category

Ask the user:

- **Category**: `tableware`, `rl`, or `special` (maps to `embodichain/lab/gym/envs/tasks/<category>/`)
- **Task name** (snake_case, e.g. `pick_place`)
- **Gym ID** (e.g. `PickPlace-v1`)
- **Task type**: RL task (needs reward functors) or expert demonstration task (needs `create_demo_action_list`)

### 2. Create the Task File

Place at `embodichain/lab/gym/envs/tasks/<category>/<name>.py`.

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

import torch
from typing import Dict, Any, Tuple

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.types import EnvObs

__all__ = ["<CamelCaseName>Env"]


@register_env("<GymId>")
class <CamelCaseName>Env(EmbodiedEnv):
    """<One-line description of the task>.

    <Longer description of what the task involves and its reward structure.>
    """

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

    # Expert demo tasks: implement `create_demo_action_list`.
    # RL tasks: implement `check_truncated`, `get_reward`, `compute_task_state`.
```

### 3. Update Exports

Add to `embodichain/lab/gym/envs/tasks/__init__.py`:

```python
from embodichain.lab.gym.envs.tasks.<category>.<name> import <CamelCaseName>Env
```

Add `"<CamelCaseName>Env"` to the `__all__` list.

### 4. Create Test Stub

Place at `tests/gym/envs/tasks/test_<name>.py`.

### 5. Format

```bash
black embodichain/lab/gym/envs/tasks/<category>/<name>.py
black tests/gym/envs/tasks/test_<name>.py
```

## Checklist

- [ ] File has Apache 2.0 header
- [ ] Uses `from __future__ import annotations`
- [ ] `@register_env` decorator with unique gym ID
- [ ] `__all__` defined in the task module
- [ ] Default `cfg = EmbodiedEnvCfg()` in `__init__`
- [ ] Import and `__all__` added to `tasks/__init__.py`
- [ ] Test stub created
- [ ] `black` run on both files
