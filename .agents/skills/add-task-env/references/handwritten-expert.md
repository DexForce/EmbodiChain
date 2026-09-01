# Handwritten Expert Trajectory

Use this route when expert behavior is deliberately implemented in trusted
Python: custom state-dependent planning, task-specific Atomic Skill
composition, bespoke validation, or a prompt that explicitly requests a
scripted/handwritten demonstration.

## Owned files

```text
embodichain_tasks/embodichain_tasks/<category_path>/<task_name>.py
embodichain_tasks/configs/tasks/<category_path>/<task_name>/
tests/gym/envs/tasks/test_<task_name>.py
```

The task module owns `@register_env` and only the behavior that configuration
and existing manager functors cannot express. The task config owns the physical
environment and ordinary manager configuration. It may reuse
`environment.component` and `embodiment.component`.

## Registration skeleton

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["MyTaskEnv"]


@register_env("MyTask-v1", max_episode_steps=600)
class MyTaskEnv(EmbodiedEnv):
    """Describe the task outcome, not the solution method."""

    def __init__(
        self,
        cfg: EmbodiedEnvCfg | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cfg, **kwargs)

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Yield the task's lazily planned expert segments."""
        ...
```

Every new Python file needs the project Apache header, future annotations,
complete public type hints, Google-style docstrings, and `__all__`.

## Prefer segment-native demonstrations

Implement `create_demo_segments()` for new work. It may return or lazily yield
`DemoSegment` values. Each segment should carry:

- a bounded action iterable;
- a stable, descriptive name;
- the target UID and natural-language instruction when meaningful;
- task-relevant metadata, including planning success;
- a validator when physical acceptance is measurable.

Lazy planning is important when a later segment depends on state produced by an
earlier segment. Keep `env.step()`, recording, reset, and persistence in the
normal Gym demo executor; the planner only returns segment actions.

`create_demo_action_list()` remains a legacy fallback and is wrapped as one
`legacy` segment. Do not choose it for new multi-stage work.

## Atomic Skills and task-local planning

Reuse the typed Atomic Action engine for reusable low-level capabilities. If a
required Atomic Skill is missing, invoke `$add-atomic-action`. Keep task-local
goal selection and segment assembly in the task module; do not put task graphs
inside Atomic Skills.

One physical embodiment component can be shared across tasks, but selecting a
component does not rewrite hard-coded Python control-part names, action widths,
or planner assumptions. Either derive those values from the selected robot or
validate the exact supported embodiment.

## Validation

Cover at least:

1. task registration and config parsing;
2. segment shape/type and deterministic metadata;
3. planning failure behavior;
4. validator acceptance and rejection;
5. lazy inter-segment state dependencies, if used; and
6. one real-simulation expert episode when the environment is available.

Use `tests/gym/envs/tasks/` for task-level tests. Real simulation/GPU cases must
use the project's resource markers and clean up `SimulationManager`.

Reference implementations:

- `embodichain_tasks/embodichain_tasks/manipulation/tableware/blocks_ranking_rgb.py`
- `embodichain_tasks/embodichain_tasks/manipulation/tableware/stack_blocks_two.py`
- `embodichain/lab/gym/envs/demo.py`
