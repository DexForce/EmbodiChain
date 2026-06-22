---
name: add-atomic-action
description: Use when adding a new simulation atomic action or motion primitive to EmbodiChain's AtomicActionEngine.
---

# Add Atomic Action

Scaffold a new atomic action following EmbodiChain's `AtomicAction` pattern: a typed
target, a `WorldState` threaded across actions, and an `ActionResult` carrying a
full-DoF trajectory.

## When to Use

- User asks to add a new motion primitive (push, wipe, insert, hand-over, …)
- User says "add a new atomic action", "create a custom action", "implement a push action"
- User wants to extend `AtomicActionEngine` with a behaviour not covered by the built-ins

## Key Files

| Purpose | Path |
|---------|------|
| Base classes (`ActionCfg`, `AtomicAction`, `WorldState`, `ActionResult`, typed targets, `ObjectSemantics`) | `embodichain/lab/sim/atomic_actions/core.py` |
| Affordance types (`Affordance`, `AntipodalAffordance`, `InteractionPoints`) | `embodichain/lab/sim/atomic_actions/affordance.py` |
| Stateless trajectory helpers (`TrajectoryBuilder`) | `embodichain/lab/sim/atomic_actions/trajectory.py` |
| Built-in actions (reference implementations) | `embodichain/lab/sim/atomic_actions/actions.py` |
| Engine + global registry (`register_action`, `AtomicActionEngine.register` / `run`) | `embodichain/lab/sim/atomic_actions/engine.py` |
| Public API exports | `embodichain/lab/sim/atomic_actions/__init__.py` |
| Reference docs | `docs/source/overview/sim/atomic_actions.md` |

## The Contract (read first)

Every atomic action is a **sibling** inheriting `AtomicAction` directly — do **not**
inherit from `MoveAction` or any other action. Each action:

1. Declares `TargetType: ClassVar[type]` — the concrete target dataclass it accepts.
2. Holds `self.builder = TrajectoryBuilder(motion_generator)` for shared trajectory math.
3. Implements exactly one method: `execute(self, target, state: WorldState) -> ActionResult`.
   - `target` is an instance of `self.TargetType`.
   - `state.last_qpos` is the full-robot qpos `(n_envs, robot.dof)` to plan from;
     `state.held_object` is the object currently grasped (or `None`).
   - Returns `ActionResult(success, trajectory, next_state)` where `trajectory` is
     full-DoF shaped `(n_envs, n_waypoints, robot.dof)` and `next_state` is the
     successor `WorldState` (advance `last_qpos` to the trajectory's final row;
     set/clear/preserve `held_object` per the action's semantics).

There is **no** `validate` method, **no** `**kwargs`, **no** `start_qpos` parameter,
**no** `updates_held_object_state` flag, and **no** `get_held_object_state`. The
`WorldState` is the single channel for inter-action state.

## Steps

### 1. Define the config

Add a `@configclass`-decorated class that extends `ActionCfg` **directly** (the cfg
hierarchy is flat — do not inherit from another action's cfg). Place it in
`embodichain/lab/sim/atomic_actions/actions.py` alongside the existing configs, or in
a new file if the action is large.

```python
from __future__ import annotations

import torch

from embodichain.utils import configclass
from embodichain.lab.sim.atomic_actions.core import ActionCfg


@configclass
class PushActionCfg(ActionCfg):
    name: str = "push"            # must match the engine registration key
    push_distance: float = 0.05   # metres to push forward
    sample_interval: int = 30     # waypoints for the push phase
    control_part: str = "arm"
```

**Rules:**
- `name` must be unique and match the key used to register the action with the engine.
- Inherit from `ActionCfg` directly. If the action needs hand open/close fields,
  declare them on this cfg (see `PickUpActionCfg` for the pattern) — do not invent a
  shared `GraspActionCfg` parent.
- All fields must have defaults.

### 2. Define a typed target (if needed)

Reuse an existing target when it fits (`PoseTarget(xpos)` for an EEF-pose target,
`GraspTarget(semantics)` for a pickup, `HeldObjectTarget(object_target_pose)` for
moving a grasped object). Only define a new frozen dataclass target when the action
needs inputs the existing targets don't carry. Put new targets in `core.py`.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PushTarget:
    contact_pose: torch.Tensor      # (4, 4) or (n_envs, 4, 4) EEF contact pose
```

### 3. Implement the action class

Subclass `AtomicAction` directly, declare `TargetType`, compose a `TrajectoryBuilder`,
and implement `execute`.

```python
from __future__ import annotations

import torch
from typing import ClassVar

from embodichain.lab.sim.planners import PlanState, MoveType
from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    WorldState,
)
from embodichain.lab.sim.atomic_actions.trajectory import TrajectoryBuilder
from embodichain.utils import logger


class PushAction(AtomicAction):
    """Push an object forward by a fixed distance from a contact pose."""

    TargetType: ClassVar[type] = PushTarget  # set to PoseTarget if you reused it

    def __init__(self, motion_generator, cfg: PushActionCfg | None = None):
        super().__init__(motion_generator, cfg or PushActionCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

    def execute(self, target: PushTarget, state: WorldState) -> ActionResult:
        # 1. Resolve the batched contact pose (n_envs, 4, 4).
        contact_xpos = self.builder.resolve_pose_target(
            target.contact_pose, n_envs=self.n_envs
        )

        # 2. Resolve the arm start qpos from the threaded WorldState.
        start_arm_qpos = self.builder.resolve_start_qpos(
            state.last_qpos[:, self.arm_joint_ids],
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )

        # 3. Plan the arm trajectory via the builder (uses IK + interpolation).
        target_states = [
            [PlanState(xpos=contact_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, arm_traj = self.builder.plan_arm_traj(
            target_states,
            start_arm_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            return self._fail(state)

        # 4. Embed the arm slice into a full-DoF trajectory (n_envs, n_wp, robot.dof).
        full = torch.empty(
            (self.n_envs, arm_traj.shape[1], self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :, self.arm_joint_ids] = arm_traj

        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,  # push does not grasp
            ),
        )

    def _fail(self, state: WorldState) -> ActionResult:
        return ActionResult(
            success=False,
            trajectory=torch.empty(
                (self.n_envs, 0, self.robot_dof),
                dtype=torch.float32,
                device=self.device,
            ),
            next_state=state,
        )
```

**Rules:**
- `execute()` returns an `ActionResult` — never a bare tuple.
- `trajectory` shape is always `(n_envs, n_waypoints, robot.dof)` (full robot DoF).
- Use `self.builder.<helper>` for all trajectory math (`resolve_pose_target`,
  `resolve_start_qpos`, `apply_local_offset`, `plan_arm_traj`, `split_three_phase`,
  `interpolate_hand_qpos`). Do not reimplement that math inline.
- Thread `WorldState` explicitly: advance `last_qpos` to the final trajectory row;
  set/clear/preserve `held_object` per what the action does to the grasp.
- Use `logger.log_error(msg, ValueError)` for contract violations (wrong target type,
  missing cfg fields); use `logger.log_warning` + `_fail(state)` for soft planning
  failures.
- Call `super().__init__()` — it sets `self.robot`, `self.motion_generator`,
  `self.device`, `self.cfg`, `self.control_part`.

### 4. Register the action

Register an **instance** with the engine so `run()` can dispatch it by name.

```python
from embodichain.lab.sim.atomic_actions import AtomicActionEngine, PushAction

engine = AtomicActionEngine(motion_generator=motion_gen)
engine.register(PushAction(motion_gen, cfg=PushActionCfg()))  # keyed by cfg.name "push"
```

For third-party / plugin actions that should be discoverable without the caller
constructing them, register the **class** in the global registry:

```python
from embodichain.lab.sim.atomic_actions import register_action
register_action("push", PushAction)
```

### 5. Export from the public API

Add the config, action class, and any new target to
`embodichain/lab/sim/atomic_actions/__init__.py`:

```python
from .actions import PushAction, PushActionCfg
# (and from .core import PushTarget if you defined one)

__all__ = [
    ...,
    "PushAction",
    "PushActionCfg",
]
```

### 6. Update the supported actions table

Add a row to the table in `docs/source/overview/sim/atomic_actions.md`:

```markdown
| `PushAction` | `PushActionCfg` | `PushTarget` — contact pose | Approach → push forward |
```

### 7. Write a test

Add a test in `tests/sim/atomic_actions/` (append to `test_actions.py` or create a new
file). Mock the `MotionGenerator` (see the `_make_mock_motion_generator` helper in
`test_actions.py`) and assert on behaviour: target type, full-DoF trajectory shape,
and the `WorldState` contract.

```python
def test_push_action_cfg_defaults():
    cfg = PushActionCfg()
    assert cfg.name == "push"
    assert cfg.push_distance == 0.05

def test_push_action_returns_full_dof_trajectory():
    mg = _make_mock_motion_generator()
    action = PushAction(mg, PushActionCfg(sample_interval=10))
    state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
    with patch(
        "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
        return_value=torch.zeros(NUM_ENVS, 10, ARM_DOF),
    ):
        result = action.execute(PushTarget(contact_pose=torch.eye(4)), state)
    assert isinstance(result, ActionResult)
    assert result.success is True
    assert result.trajectory.shape == (NUM_ENVS, 10, TOTAL_DOF)
    # push preserves held_object
    assert result.next_state.held_object is state.held_object
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Inheriting from `MoveAction` | Inherit `AtomicAction` directly and compose a `TrajectoryBuilder`. Actions are siblings, not a tree. |
| Returning `(bool, Tensor, joint_ids)` | Return an `ActionResult` with a full-DoF `(n_envs, n_wp, robot.dof)` trajectory. |
| Declaring `validate` / `updates_held_object_state` / `get_held_object_state` | These were removed. State flows only through `WorldState` and `ActionResult.next_state`. |
| `execute(target, start_qpos=None, **kwargs)` | Signature is `execute(self, target, state: WorldState) -> ActionResult`. No `**kwargs`, no `start_qpos`. |
| Reimplementing IK / interpolation inline | Use `self.builder.plan_arm_traj(...)` and friends. |
| Returning arm-only or arm+hand trajectory | Always embed into full `robot.dof` before returning. |
| `name` not matching the engine registration key | Keep `cfg.name` identical to the key passed to `engine.register(...)` / `register_action(...)`. |
| Forgetting to export from `__init__.py` | Users import from the public API — missing exports cause `ImportError`. |
| Inheriting another action's cfg | Cfgs are flat; extend `ActionCfg` directly and declare the fields you need. |

## Quick Reference

| Step | Action |
|------|--------|
| 1 | Define a flat `@configclass` extending `ActionCfg` with a unique `name` |
| 2 | Define a typed target (or reuse `PoseTarget` / `GraspTarget` / `HeldObjectTarget`) |
| 3 | Subclass `AtomicAction` directly, set `TargetType`, compose `TrajectoryBuilder`, implement `execute(target, state) -> ActionResult` |
| 4 | Register: `engine.register(PushAction(mg, cfg=...))` (instance) or `register_action("push", PushAction)` (class) |
| 5 | Export config + action (+ target) from `__init__.py` |
| 6 | Add a row to the supported-actions table in the overview docs |
| 7 | Write behavioural tests (target type, full-DoF shape, `WorldState` contract) |
