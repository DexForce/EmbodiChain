# Atomic Actions

```{currentmodule} embodichain.lab.sim.atomic_actions
```

Atomic actions are the building blocks for automated robot motion generation. Each action encapsulates a complete, self-contained motion primitive — such as picking up an object or moving to a pose — that can be chained together to form complex manipulation workflows.

## Design Overview

The module is organized into three layers:

```
AtomicActionEngine          ← orchestrates a sequence of actions
    │
    ├── AtomicAction(s)     ← each action plans one motion primitive
    │       │
    │       └── MotionGenerator   ← low-level trajectory planner (IK + trajectory optimization)
    │
    └── SemanticAnalyzer    ← resolves object labels → ObjectSemantics
```

Each action receives a target (object semantics or a pose tensor), runs its planning pipeline,
and returns a joint trajectory.  The engine threads the end state of each action as the start
state of the next, then concatenates all trajectories into one contiguous sequence:

```
ObjectSemantics ──► AffordanceEstimation ──► AtomicAction.execute()
(label + geometry                              │
 + affordance                                  ├─ IK solve
 + entity)                                     ├─ Motion plan
                                               └─ Gripper interpolation
                                                      │
AtomicActionEngine ◄─────────────── PlanResult ───────┘
(sequences actions, accumulates
 full-robot trajectory)
```

### Core Concepts

**`ObjectSemantics`** describes an interaction target. It bundles:
- `geometry` — mesh data (vertices, triangles) used for grasp annotation
- `affordance` — *how* to interact with the object (e.g. antipodal grasp poses)
- `entity` — a live reference to the simulation object, so actions can read its current pose

**`Affordance`** is a data class that encodes a specific interaction capability. The built-in affordance types are:

| Class | Use case |
|---|---|
| `AntipodalAffordance` | Parallel-jaw grasping via antipodal point pairs |
| `InteractionPoints` | Contact-based interactions (push, poke, touch) |

**`AtomicAction`** is the abstract base class for all motion primitives. Every action must implement:
- `execute(target, start_qpos)` — plan and return a joint trajectory
- `validate(target, start_qpos)` — fast feasibility check without full planning

**`AtomicActionEngine`** manages a named registry of actions and runs them in sequence via `execute_static()`, threading the end state of each action as the start state of the next.

---

## Built-in Actions

(supported_atomic_actions)=

The following actions are available out of the box:

| Action | Config class | Target type | Motion phases |
|---|---|---|---|
| `MoveAction` | `MoveActionCfg` | `Tensor (4,4)` — EEF pose | Move arm to pose |
| `PickUpAction` | `PickUpActionCfg` | `ObjectSemantics` or `Tensor (4,4)` | Approach → close gripper → lift |
| `PlaceAction` | `PlaceActionCfg` | `Tensor (4,4)` — EEF release pose | Lower → open gripper → retract |

### `MoveAction`

Moves the end-effector to a target pose in free space.

| Config field | Default | Description |
|---|---|---|
| `control_part` | `"arm"` | Robot control part to move |
| `sample_interval` | `50` | Number of waypoints in the trajectory |

**Target:** `torch.Tensor` of shape `(4, 4)` or `(n_envs, 4, 4)` — a homogeneous EEF pose.

---

### `PickUpAction`

Three-phase grasp motion: *approach → close gripper → lift*.

| Config field | Default | Description |
|---|---|---|
| `approach_direction` | `[0, 0, -1]` | Gripper approach direction in object frame |
| `pre_grasp_distance` | `0.15` | Hover distance before descending (m) |
| `lift_height` | `0.10` | Lift height after grasping (m) |
| `hand_open_qpos` | `None` | **Required.** Gripper open joint positions |
| `hand_close_qpos` | `None` | **Required.** Gripper closed joint positions |
| `hand_control_part` | `"hand"` | Robot control part for the gripper |
| `hand_interp_steps` | `5` | Waypoints for the gripper close phase |
| `sample_interval` | `80` | Total waypoints across all three phases |

**Target:** `ObjectSemantics` (grasp pose computed automatically) **or** a `torch.Tensor` EEF pose.

---

### `PlaceAction`

Three-phase release motion: *lower → open gripper → retract*. Mirrors `PickUpAction`.

Inherits all gripper config fields from `GraspActionCfg`. The `approach_direction` field is not used — the arm moves straight down to the target pose.

**Target:** `torch.Tensor` of shape `(4, 4)` or `(n_envs, 4, 4)` — the EEF pose at release.

---

## Typical Workflow

```python
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ObjectSemantics,
    AntipodalAffordance,
    PickUpActionCfg,
    PlaceActionCfg,
    MoveActionCfg,
)

# 1. Configure each action
pickup_cfg = PickUpActionCfg(
    control_part="arm",
    hand_control_part="hand",
    hand_open_qpos=torch.tensor([0.0, 0.0]),
    hand_close_qpos=torch.tensor([0.025, 0.025]),
)
place_cfg  = PlaceActionCfg(...)
move_cfg   = MoveActionCfg(control_part="arm")

# 2. Build the engine — action order matches target_list order
engine = AtomicActionEngine(
    motion_generator=motion_gen,
    actions_cfg_list=[pickup_cfg, place_cfg, move_cfg],
)

# 3. Describe the object to pick
semantics = ObjectSemantics(
    label="mug",
    geometry={"mesh_vertices": ..., "mesh_triangles": ...},
    affordance=AntipodalAffordance(object_label="mug", ...),
    entity=mug,
)

# 4. Plan the full sequence and replay
is_success, traj = engine.execute_static(
    target_list=[semantics, place_pose, rest_pose]
)
# traj: (n_envs, n_waypoints, dof)
```

---

## How to Extend: Adding a Custom Action

You can add any motion primitive by subclassing `AtomicAction` and registering it with the engine.

### Step 1 — Define the config

```python
from embodichain.utils import configclass
from embodichain.lab.sim.atomic_actions import ActionCfg

@configclass
class PushActionCfg(ActionCfg):
    name: str = "push"
    push_distance: float = 0.05  # metres to push forward
    push_speed: int = 30          # waypoints for the push phase
```

### Step 2 — Implement the action

```python
import torch
from typing import Optional, Union
from embodichain.lab.sim.atomic_actions import AtomicAction, ObjectSemantics
from embodichain.lab.sim.planners import PlanState, MoveType

class PushAction(AtomicAction):
    def __init__(self, motion_generator, cfg: PushActionCfg | None = None):
        super().__init__(motion_generator, cfg=cfg or PushActionCfg())
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)

    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list]:
        # Resolve target to a batched [n_envs, 4, 4] EEF pose
        # ... your planning logic here ...
        return is_success, trajectory, self.arm_joint_ids

    def validate(self, target, start_qpos=None, **kwargs) -> bool:
        return True  # add IK check here if needed
```

### Step 3 — Register and use

```python
from embodichain.lab.sim.atomic_actions import register_action

register_action("push", PushAction, PushActionCfg)

engine = AtomicActionEngine(
    motion_generator=motion_gen,
    actions_cfg_list=[PushActionCfg(push_distance=0.08)],
)
is_success, traj = engine.execute_static(target_list=[target_pose])
```

> **Tip:** The `execute()` return signature is always `(is_success, trajectory, joint_ids)`.  
> `trajectory` has shape `(n_envs, n_waypoints, len(joint_ids))`.  
> `joint_ids` tells the engine which columns of the full robot DOF vector the trajectory covers.

---

## Target Resolution

`AtomicActionEngine` accepts several target formats in `target_list`, giving you flexibility without boilerplate:

| Input type | Resolved to |
|---|---|
| `torch.Tensor (4,4)` or `(n_envs,4,4)` | EEF pose, broadcast across envs |
| `ObjectSemantics` | Passed directly to the action |
| `str` (object label) | Looked up in `SemanticAnalyzer` cache |
| `dict` with `"pose"` key | Unwrapped to tensor |
| `dict` with `"label"` key | Analyzed via `SemanticAnalyzer` |

---

## Further Reading

- {doc}`planners/motion_generator` — the trajectory planner used by every action
- {doc}`sim_robot` — how control parts and IK solvers are configured
- Tutorial: `scripts/tutorials/sim/atomic_actions.py`
