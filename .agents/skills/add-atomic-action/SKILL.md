---
name: add-atomic-action
description: Use when adding a new simulation atomic action or motion primitive to EmbodiChain's AtomicActionEngine.
---

# Add Atomic Action

Scaffold a new atomic action following EmbodiChain's `ActionCfg` / `AtomicAction` pattern.

## When to Use

- User asks to add a new motion primitive (push, wipe, insert, hand-over, …)
- User says "add a new atomic action", "create a custom action", "implement a push action"
- User wants to extend `AtomicActionEngine` with a behaviour not covered by the built-ins

## Key Files

| Purpose | Path |
|---------|------|
| Base classes (`ActionCfg`, `AtomicAction`, `ObjectSemantics`) | `embodichain/lab/sim/atomic_actions/core.py` |
| Built-in actions (reference implementations) | `embodichain/lab/sim/atomic_actions/actions.py` |
| Engine + global registry (`register_action`) | `embodichain/lab/sim/atomic_actions/engine.py` |
| Public API exports | `embodichain/lab/sim/atomic_actions/__init__.py` |
| Reference docs | `docs/source/overview/sim/atomic_actions.md` |

## Steps

### 1. Define the config

Add a `@configclass`-decorated class that extends `ActionCfg` (or `MoveActionCfg` /
`GraspActionCfg` if the new action reuses arm/gripper fields).

Place it in `embodichain/lab/sim/atomic_actions/actions.py` alongside the existing configs,
or in a new file if the action is large.

```python
from embodichain.utils import configclass
from embodichain.lab.sim.atomic_actions.core import ActionCfg   # or MoveActionCfg

@configclass
class PushActionCfg(ActionCfg):
    name: str = "push"                # must match the registry key
    push_distance: float = 0.05       # metres to push forward
    push_speed: int = 30              # waypoints for the push phase
    control_part: str = "arm"         # robot segment to control
```

**Rules:**
- `name` must be unique and match the string passed to `register_action`.
- Inherit from `GraspActionCfg` when the action needs hand open/close fields.
- All fields must have defaults — configs are instantiated without arguments in tests.

### 2. Implement the action class

Subclass `AtomicAction` and implement the two abstract methods.

```python
import torch
from typing import Optional, Union
from embodichain.lab.sim.atomic_actions.core import AtomicAction, ObjectSemantics

class PushAction(AtomicAction):
    """Push an object forward by a fixed distance."""

    def __init__(self, motion_generator, cfg: PushActionCfg | None = None):
        super().__init__(motion_generator, cfg=cfg or PushActionCfg())
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)

    # ------------------------------------------------------------------
    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list]:
        """Plan the push motion and return a joint trajectory.

        Args:
            target: EEF pose tensor (n_envs, 4, 4) or ObjectSemantics.
            start_qpos: Starting joint positions (n_envs, dof). Uses current
                robot state when None.

        Returns:
            Tuple of (is_success, trajectory, joint_ids) where
            trajectory has shape (n_envs, n_waypoints, len(joint_ids)).
        """
        # 1. Resolve target pose
        # 2. Plan trajectory with self.motion_generator
        # 3. Return result
        return is_success, trajectory, self.arm_joint_ids

    # ------------------------------------------------------------------
    def validate(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> bool:
        """Fast feasibility check — no trajectory generated.

        Returns:
            True if the action can be attempted.
        """
        return True  # add IK reachability check here if needed
```

**Rules:**
- `execute()` must always return `(is_success: bool, trajectory: Tensor, joint_ids: list)`.
- `trajectory` shape: `(n_envs, n_waypoints, len(joint_ids))`.
- `joint_ids` tells the engine which DOF columns the trajectory covers.
- `validate()` must be cheap — no motion planning allowed.
- Call `super().__init__()` — it sets `self.robot`, `self.motion_generator`, and `self.cfg`.

### 3. Register the action

Register the new class so `AtomicActionEngine` can discover it by name.

**Option A — register at module load (built-ins style)**

In `embodichain/lab/sim/atomic_actions/engine.py`, add to the `_builtin_action_map` dict:

```python
_builtin_action_map: dict[str, type[AtomicAction]] = {
    "move":   MoveAction,
    "pickup": PickUpAction,
    "place":  PlaceAction,
    "push":   PushAction,   # ← add here
}
```

**Option B — register at runtime (custom / plugin style)**

```python
from embodichain.lab.sim.atomic_actions import register_action
register_action("push", PushAction)
```

### 4. Export from the public API

Add config and action class to `embodichain/lab/sim/atomic_actions/__init__.py`:

```python
from .actions import PushAction, PushActionCfg

__all__ = [
    ...,
    "PushAction",
    "PushActionCfg",
]
```

### 5. Update the supported actions table

Add a row to the table in `docs/source/overview/sim/atomic_actions.md` under
"Supported Actions":

```markdown
| `PushAction` | `PushActionCfg` | `Tensor (4,4)` — contact pose | Approach → push forward |
```

### 6. Write a test

Add a test in `tests/sim/atomic_actions/` (append to an existing file or create a new one):

```python
def test_push_action_cfg_defaults():
    cfg = PushActionCfg()
    assert cfg.name == "push"
    assert cfg.push_distance == 0.05

def test_push_action_validate(mock_motion_generator):
    action = PushAction(mock_motion_generator)
    assert action.validate(target=torch.eye(4)) is True
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `name` in config doesn't match registry key | Keep `cfg.name` identical to the string in `register_action("push", ...)` |
| Returning `trajectory` without `joint_ids` | Always return the 3-tuple `(bool, Tensor, list)` |
| `trajectory` shape `(n_envs, dof, n_waypoints)` | Correct shape is `(n_envs, n_waypoints, dof)` |
| Doing motion planning inside `validate()` | `validate()` must be fast — IK check only |
| Not calling `super().__init__()` | Required to set `self.robot`, `self.motion_generator`, `self.cfg` |
| Inheriting `MoveActionCfg` instead of `ActionCfg` | Use `MoveActionCfg` only when the action reuses arm-control fields; otherwise use `ActionCfg` |
| Forgetting to export from `__init__.py` | Users import from the public API — missing exports cause `ImportError` |

## Quick Reference

| Step | Action |
|------|--------|
| 1 | Define `@configclass` config extending `ActionCfg` with `name` field |
| 2 | Subclass `AtomicAction`, implement `execute()` and `validate()` |
| 3 | Register: add to `_builtin_action_map` or call `register_action()` |
| 4 | Export from `__init__.py` |
| 5 | Add row to supported-actions table in overview docs |
| 6 | Write tests for config defaults and `validate()` |
