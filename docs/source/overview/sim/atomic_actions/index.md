# Atomic Actions

```{currentmodule} embodichain.lab.sim.atomic_actions
```

Atomic actions are the building blocks for automated robot motion generation. Each action encapsulates a complete, self-contained motion primitive — such as picking up an object or moving to a pose — that can be chained together to form complex manipulation workflows.

## Design Overview

The module is organized into three layers:

```
AtomicActionEngine          ← orchestrates a sequence of (name, typed_target) steps
    │
    ├── AtomicAction(s)     ← each action plans one motion primitive
    │       │
    │       └── MotionGenerator   ← low-level trajectory planner (IK + trajectory optimization)
    │
    └── WorldState           ← threaded action-to-action
                               (last_qpos + held_object/coordinated_held_object)
```

Each action receives a typed target and a `WorldState`, runs its planning pipeline, and
returns an `ActionResult` whose trajectory covers the full robot DOF.  The engine threads
the `next_state` of each action as the input state of the next, then concatenates all
trajectories into one contiguous sequence:

```
GraspTarget(semantics) ──► AtomicAction.execute(target, state)
EndEffectorPoseTarget(xpos)                │
JointPositionTarget(qpos)                  ├─ IK solve when pose-based
NamedJointPositionTarget(name)             ├─ Motion plan / interpolation
HeldObjectPoseTarget(pose)                 └─ Gripper interpolation when needed
CoordinatedPickmentTarget(...)             │
CoordinatedPlacementTarget(...)            │
                                                   │
                                           ActionResult
                                           (success, full-DoF traj, next_state)
                                                   │
AtomicActionEngine ◄───────────────────┘
(run(steps, state) → (is_success, traj, final_state))
```

### Core Concepts

**`ObjectSemantics`** describes an interaction target. It bundles:
- `affordance` — *how* to interact with the object (e.g. an `AntipodalAffordance` carrying mesh data and grasp-generation config)
- `geometry` — plain geometric metadata (e.g. a bounding box). Mesh tensors live on the affordance, not here
- `label` — object category string (also bound onto the affordance for convenience)
- `entity` — a live reference to the simulation object, so actions can read its current pose

**`HeldObjectState`** is runtime state produced after a successful `PickUp`. It stores
the held object's semantics and object-to-end-effector transform so later actions can move the
object without recomputing the grasp. It is intentionally separate from `ObjectSemantics`,
which remains a reusable object description rather than per-execution robot state.

**Typed targets** describe *where* an action should go. Each one is a small frozen dataclass,
and every action declares the target type, or tuple of target types, it accepts via its
`TargetType` class variable:

| Target | Constructor | Used by |
|---|---|---|
| `EndEffectorPoseTarget` | `EndEffectorPoseTarget(xpos)` | `MoveEndEffector`, `Place`, `Press` |
| `JointPositionTarget` | `JointPositionTarget(qpos)` | `MoveJoints` |
| `NamedJointPositionTarget` | `NamedJointPositionTarget(name)` | `MoveJoints` |
| `GraspTarget` | `GraspTarget(semantics)` | `PickUp` |
| `HeldObjectPoseTarget` | `HeldObjectPoseTarget(object_target_pose)` | `MoveHeldObject` |
| `CoordinatedPickmentTarget` | `CoordinatedPickmentTarget(...)` | `CoordinatedPickment` |
| `CoordinatedPlacementTarget` | `CoordinatedPlacementTarget(...)` | `CoordinatedPlacement` |

`Target` is the union of these typed target dataclasses.

**`Affordance`** is a data class that encodes a specific interaction capability. The built-in affordance types are:

| Class | Use case |
|---|---|
| `AntipodalAffordance` | Parallel-jaw grasping via antipodal point pairs |
| `InteractionPoints` | Contact-based interactions (push, poke, touch) |

`AntipodalAffordance` takes its inputs as direct fields — `mesh_vertices`, `mesh_triangles`,
`gripper_collision_cfg`, `generator_cfg`, and `force_reannotate` — rather than a nested config dict.

**`AtomicAction`** is the abstract base class for all motion primitives. Subclasses declare a
`TargetType` class variable and implement a single method:
- `execute(target, state) -> ActionResult` — plan and return a full-DOF trajectory plus the
  successor `WorldState`

**`AtomicActionEngine`** holds a name-keyed registry of action instances and runs a sequence of
`(name, typed_target)` steps via `run(steps, state)`, threading `WorldState` from one action into
the next.

---

## Typed Targets & State Threading

The engine takes a sequence of `(name, typed_target)` steps. Each target is a small
frozen dataclass, and the engine checks that each step's target matches the registered
action's `TargetType` before calling `execute`:

| Target | Holds | Accepted by |
|---|---|---|
| `EndEffectorPoseTarget(xpos)` | EEF pose tensor `(4,4)`, `(n_envs,4,4)` or `(n_envs, n_waypoint, 4, 4)` | `MoveEndEffector`, `Place`, `Press` |
| `JointPositionTarget(qpos)` | Control-part qpos tensor `(control_dof,)`, `(n_envs, control_dof)` or `(n_envs, n_waypoint, control_dof)` | `MoveJoints` |
| `NamedJointPositionTarget(name)` | Name resolved from `MoveJointsCfg.named_joint_positions` | `MoveJoints` |
| `GraspTarget(semantics)` | `ObjectSemantics` (affordance + entity) | `PickUp` |
| `HeldObjectPoseTarget(object_target_pose)` | Desired held-object pose tensor | `MoveHeldObject` |
| `CoordinatedPickmentTarget(...)` | Shared object semantics plus left/right grasp transforms and target object pose | `CoordinatedPickment` |
| `CoordinatedPlacementTarget(...)` | Two held-object states plus object-centric placing/support target poses | `CoordinatedPlacement` |

`WorldState` is threaded between actions and carries the robot's `last_qpos` plus optional
`held_object: HeldObjectState` and `coordinated_held_object: CoordinatedHeldObjectState`.
The built-in actions update it as follows:

| Action | Effect on `held_object` |
|---|---|
| `PickUp` | Populates it (computed object-to-EEF transform) |
| `MoveHeldObject` | Requires it; preserves it unchanged |
| `Place` | Clears it to `None` |
| `CoordinatedPickment` | Leaves `held_object` as `None` and populates `coordinated_held_object` |
| `CoordinatedPlacement` | Returns the support arm's `HeldObjectState`; the placing object is released |
| `MoveEndEffector` | Leaves it unchanged |
| `MoveJoints` | Leaves it unchanged |
| `Press` | Leaves it unchanged |

If a step fails, `run()` returns `success=False` with the partial trajectory concatenated up
to (but not including) the failed step, and the `WorldState` going into that step.

---

## Typical Workflow

```python
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ObjectSemantics,
    AntipodalAffordance,
    GraspTarget,
    EndEffectorPoseTarget,
    JointPositionTarget,
    NamedJointPositionTarget,
    HeldObjectPoseTarget,
    PickUp,
    PickUpCfg,
    MoveJoints,
    MoveJointsCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
    Place,
    PlaceCfg,
    Press,
    PressCfg,
    MoveEndEffector,
    MoveEndEffectorCfg,
)

# 1. Configure each action
pickup_cfg = PickUpCfg(
    control_part="arm",
    hand_control_part="hand",
    hand_open_qpos=torch.tensor([0.0, 0.0]),
    hand_close_qpos=torch.tensor([0.025, 0.025]),
)
place_cfg = PlaceCfg(
    control_part="arm",
    hand_control_part="hand",
    hand_open_qpos=torch.tensor([0.0, 0.0]),
    hand_close_qpos=torch.tensor([0.025, 0.025]),
)
move_held_object_cfg = MoveHeldObjectCfg(
    control_part="arm",
    hand_control_part="hand",
    hand_close_qpos=torch.tensor([0.025, 0.025]),
)
move_cfg = MoveEndEffectorCfg(control_part="arm")
press_cfg = PressCfg(
    control_part="arm",
    hand_control_part="hand",
    hand_close_qpos=torch.tensor([0.025, 0.025]),
)
move_joints_cfg = MoveJointsCfg(
    control_part="arm",
    named_joint_positions={"home": torch.zeros(6)},
)

# 2. Build the engine and register each action instance by name
engine = AtomicActionEngine(motion_generator=motion_gen)
engine.register(PickUp(motion_gen, cfg=pickup_cfg))
engine.register(MoveHeldObject(motion_gen, cfg=move_held_object_cfg))
engine.register(Place(motion_gen, cfg=place_cfg))
engine.register(MoveEndEffector(motion_gen, cfg=move_cfg))
engine.register(Press(motion_gen, cfg=press_cfg))
engine.register(MoveJoints(motion_gen, cfg=move_joints_cfg))

# 3. Describe the object to pick
semantics = ObjectSemantics(
    affordance=AntipodalAffordance(
        mesh_vertices=obj.get_vertices(env_ids=[0], scale=True)[0],
        mesh_triangles=obj.get_triangles(env_ids=[0])[0],
        gripper_collision_cfg=gripper_cfg,
        generator_cfg=generator_cfg,
    ),
    geometry={},
    label="mug",
    entity=obj,
)

# 4. Plan the full sequence — steps are (name, typed_target) pairs
is_success, traj, final_state = engine.run(
    steps=[
        ("pick_up", GraspTarget(semantics=semantics)),
        ("move_held_object", HeldObjectPoseTarget(object_target_pose=carry_pose)),
        ("place", EndEffectorPoseTarget(xpos=place_pose)),
        ("move_joints", NamedJointPositionTarget(name="home")),
    ]
)
# traj: (n_envs, n_waypoints, robot.dof)
```

---

## How to Extend: Adding a Custom Action

You can add any motion primitive by subclassing `AtomicAction`, composing a
`TrajectoryBuilder` for the shared planning math, and registering an instance with the engine.
Built-in primitives live one action per module under
`embodichain/lab/sim/atomic_actions/primitives/`, while
`embodichain.lab.sim.atomic_actions` remains the public import surface and
`embodichain.lab.sim.atomic_actions.actions` stays as a compatibility re-export.

### Step 1 — Define the config

```python
from embodichain.utils import configclass
from embodichain.lab.sim.atomic_actions import ActionCfg

@configclass
class PushCfg(ActionCfg):
    name: str = "push"
    push_distance: float = 0.05  # metres to push forward
    push_speed: int = 30          # waypoints for the push phase
```

### Step 2 — Implement the action

```python
import torch
from typing import ClassVar
from embodichain.lab.sim.atomic_actions import (
    AtomicAction, ActionResult, EndEffectorPoseTarget, Target, WorldState, TrajectoryBuilder,
)

class Push(AtomicAction):
    TargetType: ClassVar[type] = EndEffectorPoseTarget

    def __init__(self, motion_generator, cfg: PushCfg | None = None):
        super().__init__(motion_generator, cfg or PushCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.robot_dof = self.robot.dof
        self.n_envs = self.robot.get_qpos().shape[0]

    def execute(self, target: EndEffectorPoseTarget, state: WorldState) -> ActionResult:
        # ... your planning logic, using self.builder for IK / interpolation ...
        # full must be shaped (n_envs, n_waypoints, robot.dof)
        return ActionResult(
            success=is_success,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,  # push does not change what is held
            ),
        )
```

### Step 3 — Register and use

Register an instance with the engine so it can be referenced by name in `run()`:

```python
from embodichain.lab.sim.atomic_actions import EndEffectorPoseTarget

engine.register(Push(motion_gen, cfg=PushCfg(push_distance=0.08)))
is_success, traj, final_state = engine.run(
    steps=[("push", EndEffectorPoseTarget(xpos=target_pose))]
)
```

To publish an action class for third-party discovery (independent of any engine instance),
use the global registry:

```python
from embodichain.lab.sim.atomic_actions import register_action, unregister_action, get_registered_actions

register_action("push", Push)          # registers the class under "push"
unregister_action("push")                    # removes it
all_actions = get_registered_actions()       # dict[str, type[AtomicAction]]
```

> **Tip:** `execute()` always returns an `ActionResult`. Its `trajectory` is full-robot-DOF
> shaped `(n_envs, n_waypoints, robot.dof)`, and `next_state` carries the `WorldState` the
> engine will feed into the following step. Use `TrajectoryBuilder` for pose broadcasting,
> three-phase splitting, IK/FK, and hand-qpos interpolation so your action matches the
> built-ins.

---

```{toctree}
:maxdepth: 1

builtin_actions
```

## Further Reading

- {doc}`../planners/motion_generator` — the trajectory planner used by every action
- {doc}`../sim_robot` — how control parts and IK solvers are configured
- Focused primitive demos:
  - `scripts/tutorials/atomic_action/move_end_effector.py`
  - `scripts/tutorials/atomic_action/move_joints.py`
  - `scripts/tutorials/atomic_action/pickup.py`
  - `scripts/tutorials/atomic_action/move_held_object.py`
  - `scripts/tutorials/atomic_action/place.py`
  - `scripts/tutorials/atomic_action/press.py`
  - `scripts/tutorials/atomic_action/coordinated_pickment.py`
  - `scripts/tutorials/atomic_action/coordinated_placement.py`

Run a demo in headless CPU mode with `--auto_play --headless --device cpu` to record
an MP4 under `outputs/videos`. For example:

```bash
python scripts/tutorials/atomic_action/move_end_effector.py --headless --auto_play --device cpu
```
