---
name: add-task-env
description: Use when creating a new task environment for EmbodiChain, including expert demostraction tasks, RL tasks or any EmbodiedEnv subclass
---

# Add Task Environment

Scaffold a new task environment following EmbodiChain's conventions and patterns.

## When to Use

- User asks to create a new task or environment
- User says "add a task", "new env", "create environment for X"
- A new EmbodiedEnv or BaseEnv subclass is needed

## Steps

### 1. Determine Task Category

Ask the user which category the task belongs to:

| Category | Directory | Base Class | Typical Use |
|----------|-----------|------------|-------------|
| `tableware` | `embodichain/lab/gym/envs/tasks/tableware/` | `EmbodiedEnv` | Manipulation tasks (pouring, stacking, rearranging) |
| `rl` | `embodichain/lab/gym/envs/tasks/rl/` | `EmbodiedEnv` | Reinforcement learning tasks (push, reach, lift) |
| `special` | `embodichain/lab/gym/envs/tasks/special/` | `EmbodiedEnv` | Simple or demo tasks |

Also ask:
- Task name (snake_case, e.g. `pick_place`)
- Gym registration ID (e.g. `PickPlace-v1`)
- `max_episode_steps` value
- Whether this is an RL task (needs reward functors) or an agent task (needs action bank / trajectory)

### 2. Create the Task File

Place at `embodichain/lab/gym/envs/tasks/<category>/<name>.py`.

If the task needs multiple files (e.g. action bank), create a subdirectory package:
`embodichain/lab/gym/envs/tasks/<category>/<name>/__init__.py` + `<name>.py`

Use this template:

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


@register_env("<GymId>", max_episode_steps=<N>)
class <CamelCaseName>Env(EmbodiedEnv):
    """<One-line description of the task>.

    <Longer description of what the task involves and its reward structure.>
    """

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        if cfg is None:
            cfg = EmbodiedEnvCfg()
        super().__init__(cfg, **kwargs)

    def compute_task_state(
        self, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Compute success/fail state and metrics for the task.

        Returns:
            is_success: Boolean tensor of shape (num_envs,).
            is_fail: Boolean tensor of shape (num_envs,).
            metrics: Dictionary of metric tensors.
        """
        is_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        is_fail = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        metrics: Dict[str, Any] = {}
        return is_success, is_fail, metrics

    def check_truncated(self, obs: EnvObs, info: Dict[str, Any]) -> torch.Tensor:
        """Check if episodes should be truncated early.

        Returns:
            Boolean tensor of shape (num_envs,).
        """
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)


if __name__ == "__main__":
    env = <CamelCaseName>Env()
```

### 3. Key Methods to Implement

Based on task needs, implement these methods on the `EmbodiedEnv` subclass:

| Method | Required? | Purpose |
|--------|-----------|---------|
| `compute_task_state` | **Yes** | Returns `(is_success, is_fail, metrics)` |
| `check_truncated` | **Yes** | Returns early-truncation boolean tensor |
| `_setup_scene` | If custom scene | Override to add task-specific objects/lights |
| `_reset_idx` | If custom reset | Override for per-env reset logic |

For **RL tasks**, also ensure the `EmbodiedEnvCfg` includes:
- `event_cfg` with any randomization events
- `observation_cfg` with observation functors
- `reward_cfg` with reward functors

For **agent tasks** (trajectory-based), consider adding:
- An action bank at `tasks/<category>/<name>/action_bank.py`
- A companion `BaseAgentEnv` subclass (see `tableware/base_agent_env.py`)

### 4. Update `__init__.py`

Add the import and `__all__` entry to `embodichain/lab/gym/envs/tasks/__init__.py`:

```python
from embodichain.lab.gym.envs.tasks.<category>.<name> import <CamelCaseName>Env
```

And add `"<CamelCaseName>Env"` to the `__all__` list.

If a new category directory was created, also add a new `__init__.py` in that directory.

### 5. Create a Test File

Place at `tests/gym/envs/tasks/test_<name>.py`. For pure-logic tests (no GPU required), use pytest style. For sim-dependent tests, use class style with `setup_method`/`teardown_method`.

### 6. Run `black`

```bash
black embodichain/lab/gym/envs/tasks/<category>/<name>.py
black tests/gym/envs/tasks/test_<name>.py
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Forgetting `@register_env` decorator | Every task must be registered with a unique gym ID |
| Missing `__all__` in the task module | Define `__all__` with the exported class name |
| Not setting `cfg = EmbodiedEnvCfg()` default | Always provide a default config in `__init__` |
| Using `self.device` before `super().__init__` | Call `super().__init__` first |
| Forgetting Apache 2.0 header | Every file must start with the copyright block |
| Adding task to `__init__.py` but not `__all__` | Both import AND `__all__` entry are required |
| Missing `from __future__ import annotations` | Required at top of every file (after header) |

## Quick Reference

| Item | Pattern |
|------|---------|
| File location | `embodichain/lab/gym/envs/tasks/<category>/<name>.py` |
| Base class | `EmbodiedEnv` (most cases), `BaseEnv` (rare) |
| Registration | `@register_env("<GymId>", max_episode_steps=<N>)` |
| Config class | `EmbodiedEnvCfg` (inherited or as-is) |
| Required methods | `compute_task_state`, `check_truncated` |
| Test location | `tests/gym/envs/tasks/test_<name>.py` |
