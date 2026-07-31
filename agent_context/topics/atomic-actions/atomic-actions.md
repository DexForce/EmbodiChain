# Atomic Actions

## Entry Points

| What | Path |
|---|---|
| Base classes, configs, runtime state | `embodichain/lab/sim/atomic_actions/core.py` |
| Shared target contracts | `embodichain/lab/sim/atomic_actions/targets.py` |
| Engine and global registry | `embodichain/lab/sim/atomic_actions/engine.py` |
| Trajectory helpers | `embodichain/lab/sim/atomic_actions/trajectory.py` |
| Built-in primitives and their targets | `embodichain/lab/sim/atomic_actions/primitives/` |
| Legacy re-export facade | `embodichain/lab/sim/atomic_actions/actions.py` |
| Public API | `embodichain/lab/sim/atomic_actions/__init__.py` |

## Overview

Atomic actions are env-batched motion primitives chained by `AtomicActionEngine`. Each action receives a typed target and a `WorldState`, plans a full-DoF trajectory for all environments, and returns an `ActionResult`. The engine threads `WorldState` from one action to the next and concatenates trajectories along the time axis.

```
AtomicActionEngine
  ├─ AtomicAction(s)        ← one primitive per class, e.g. MoveEndEffector, PickUp
  │      │
  │      └── TrajectoryBuilder  ← IK/interpolation and MotionGenerator dispatch
  │
  └── WorldState            ← last_qpos + per-control-part held-object maps
```

All tensor shapes carry a leading batch dim `B = n_envs`.

## Core Types

### Typed Targets

Frozen, identity-equality dataclasses accepted by actions via their `TargetType`
class variable. Built-in and third-party targets inherit the open `ActionTarget`
marker; `BuiltinTarget` is only the closed union of targets shipped by EmbodiChain.
Each action-exclusive target is defined beside its owning action and config in
`primitives/<action>.py`. The package root re-exports all targets, so callers
should import them from `embodichain.lab.sim.atomic_actions`. A genuinely shared
target belongs in a neutral target module, not in one primitive that another
primitive must import.

`ObjectActionTarget(semantics)` is the neutral shared base for actions operating
on a semantic object. It intentionally does not define a generic pose field:
object poses, single-arm grasp poses, and dual-arm grasp pairs are distinct
contracts.

| Target | Holds | Used by |
|---|---|---|
| `EndEffectorPoseTarget(xpos)` | `(4,4)`, `(B,4,4)` or `(B,n_waypoint,4,4)` EEF pose | `MoveEndEffector` |
| `PlaceTarget(xpos, tcp_symmetry)` | Release EEF pose plus optional TCP z-roll symmetry | `Place` |
| `PressTarget(xpos)` | One `(4,4)` or `(B,4,4)` contact pose | `Press` |
| `JointPositionTarget(qpos)` | `(dof,)`, `(B,dof)` or `(B,n_waypoint,dof)` joint positions | `MoveJoints` |
| `NamedJointPositionTarget(name)` | Name resolved from `MoveJointsCfg.named_joint_positions` | `MoveJoints` |
| `ObjectActionTarget(semantics)` | Shared semantic-object contract; no generic pose | Base of object-centric targets |
| `GraspTarget(semantics)` | Object semantics plus optional single-arm `grasp_xpos` | `PickUp` |
| `HeldObjectPoseTarget(pose)` | `(4,4)` or `(B,4,4)` target pose for the held object | `MoveHeldObject` |
| `CoordinatedPickTarget(semantics, ...)` | Shared object + target object pose + left/right object-to-EEF transforms | `CoordinatedPickment` |
| `CoordinatedPlacementTarget(...)` | Placing/support target poses and per-call offsets | `CoordinatedPlacement` |

`CoordinatedPickmentTarget` remains an alias of `CoordinatedPickTarget`.

### WorldState

Threaded between actions:
- `last_qpos: torch.Tensor` — shape `(B, robot.dof)`, robot joint positions at the start of the next action.
- `held_objects: dict[str, HeldObjectState]` — independently held objects keyed by arm/control part.
- `coordinated_held_objects: dict[tuple[str, str], CoordinatedHeldObjectState]` — jointly held objects keyed by an ordered control-part pair.

`HeldObjectState` stores the object's semantics plus the object-to-EEF transform and grasp pose (both `(B, 4, 4)`).
Use `get_held_object(control_part)`, `get_coordinated_held_object(first, second)`,
and `with_updates(...)`; `with_updates` copies both maps so successor actions do
not alias their containers.

### ActionResult

Every `execute()` returns:
- `success: bool | torch.Tensor` — per-env boolean tensor of shape `(B,)` for batched actions.
- `trajectory: torch.Tensor` — full-robot trajectory `(B, n_waypoints, robot.dof)`.
- `next_state: WorldState` — state to feed into the next action.

Helpers:
- `ActionResult.success_all` — `True` only when every env succeeded.
- `bool(action_result)` — deprecated; delegates to `success_all` and emits a `DeprecationWarning`.

## Action Configuration

`ActionCfg` (base for all action configs):

| Field | Type | Default | Notes |
|---|---|---|---|
| `name` | `str` | `"default"` | Engine registration key |
| `control_part` | `str` | `"arm"` | Robot control part to move |
| `interpolation_type` | `str` | `"linear"` | Interpolation flavor |
| `velocity_limit` | `float \| None` | `None` | Used on the `motion_gen` path |
| `acceleration_limit` | `float \| None` | `None` | Used on the `motion_gen` path |
| `motion_source` | `str` | `"ik_interp"` | `"ik_interp"` (batched IK + interpolation) or `"motion_gen"` (batched `MotionGenerator`) |

The base config is flat: every action cfg extends `ActionCfg` directly, even if it also carries hand open/close fields (see `PickUpCfg` / `PlaceCfg`).
The engine's `MotionGenerator` owns exactly one planner, so action configs do not
declare a planner type. On the `motion_gen` path, `TrajectoryBuilder` derives
planner-specific options from that owned planner.

## TrajectoryBuilder

Stateless helper owned by each action. Key methods:

| Method | Purpose |
|---|---|
| `resolve_pose_target(target, n_envs)` | Broadcast EEF target to `(B,4,4)` or `(B,n,4,4)` |
| `resolve_joint_target(target, n_envs, joint_dof, control_part)` | Broadcast joint target to `(B,dof)` or `(B,n,dof)` |
| `resolve_start_qpos(start_qpos, n_envs, arm_dof, control_part)` | Broadcast start qpos to `(B, arm_dof)` |
| `plan_arm_traj(target_states_list, start_qpos, n_waypoints, control_part, arm_dof, cfg=None)` | Returns `(success:(B,), trajectory:(B,n_waypoints,arm_dof))`. Selects `ik_interp` or `motion_gen` from `cfg.motion_source`. |
| `plan_joint_traj(start_qpos, target_qpos, n_waypoints)` | Joint-space interpolation; always succeeds. |
| `split_three_phase(...)` | Split sample interval into motion / hand-interp / motion phases. |
| `interpolate_hand_qpos(...)` | Interpolate gripper qpos between two states. |

`plan_arm_traj` input contract for actions: `target_states_list` is `list[list[PlanState]]` where the outer list is per-env and the inner list is per-waypoint. The builder internally converts to a batched `list[PlanState]` (each carrying `(B, ...)` tensors) when dispatching to `MotionGenerator`.

## AtomicActionEngine

```python
engine = AtomicActionEngine(motion_generator)
engine.register(MoveEndEffector(motion_generator, cfg=MoveEndEffectorCfg()))
success, traj, final_state = engine.run(steps=[("move_end_effector", target)])
```

`run(steps, state=None) -> (success, full_traj, final_state)`:
- `success` is a `(B,)` bool tensor indicating which environments completed every step.
- Failed environments hold their last successful joint position in both `full_traj` and `final_state.last_qpos` for the remainder of the sequence.
- If all envs fail, the loop stops early.
- `state` defaults to `WorldState(last_qpos=robot.get_qpos().clone())`.

## Built-in Primitives

| Action | Target | Notes |
|---|---|---|
| `MoveEndEffector` | `EndEffectorPoseTarget` | EEF pose move |
| `MoveJoints` | `JointPositionTarget` / `NamedJointPositionTarget` | Joint-space interpolation |
| `PickUp` | `GraspTarget` | Approach → close gripper → lift; populates `held_objects[cfg.control_part]` |
| `MoveHeldObject` | `HeldObjectPoseTarget` | Moves the object at `held_objects[cfg.control_part]` |
| `Place` | `PlaceTarget` | Lower → open gripper → retract; clears its control-part entry |
| `Press` | `PressTarget` | Close gripper → press down → return |
| `CoordinatedPickment` | `CoordinatedPickTarget` | Replaces the two individual entries with one coordinated held state |
| `CoordinatedPlacement` | `CoordinatedPlacementTarget` | Reads both individual held states from `WorldState` |

## Implementing a New Action

1. Create a flat `@configclass` extending `ActionCfg` with a unique `name`.
2. Define an action-exclusive `@dataclass(frozen=True, slots=True, eq=False)`
   target beside the action. Reuse or promote a target to a neutral module only
   when the contract is genuinely shared. Inherit `ObjectActionTarget` when
   multiple object-centric actions share only `semantics`; keep pose roles in
   the concrete target.
3. Subclass `AtomicAction[YourTarget]` directly (do not inherit from another action). Set `TargetType` for runtime checking and compose a `TrajectoryBuilder`.
4. Implement `execute(self, target, state: WorldState) -> ActionResult`:
   - Resolve batched targets and start qpos via `self.builder`.
   - Call `self.builder.plan_arm_traj(..., cfg=self.cfg)` if using arm motion.
   - Return per-env `success` (a `(B,)` tensor if any env can fail, or `torch.ones(...)` for always-succeeding paths).
   - Embed the arm trajectory into full-DoF shape `(B, n_wp, robot.dof)`.
   - Advance `last_qpos` with `state.with_updates(...)` and preserve/update the held-object maps.
5. Register an instance with the engine or globally via `register_action(name, ActionClass)`.
6. Export from `primitives/__init__.py` and `atomic_actions/__init__.py`.

## Common Failure Modes

- **Forgetting `cfg=self.cfg` in `plan_arm_traj`** — without it, `motion_source` defaults to `"ik_interp"` and per-action planning options are ignored.
- **Treating `success` as scalar** — `ActionResult.success` is `(B,)` for all built-ins; use `success_all` or `success.all()` for a single bool.
- **Using `bool(action_result)` in new code** — still works but emits a `DeprecationWarning`; prefer `.success_all`.
- **Returning arm-only trajectory** — actions must embed into `(B, n_wp, robot.dof)` before returning.
- **Putting runtime held state into a target** — desired state belongs in the
  target; objects already held by a control part belong in `WorldState`.
- **Using a target dataclass with default Tensor equality** — use `eq=False`;
  generated dataclass equality is invalid for multi-element tensors.
- **Importing a target from a sibling primitive** — give the action its own
  contract or promote the shared contract to a neutral module.
- **Putting a generic `xpos` on a shared object target** — use explicit names
  such as `object_target_pose`, `grasp_xpos`, or left/right grasp transforms;
  their frames and cardinalities are not interchangeable.
- **`motion_source="motion_gen"` without a MotionGenerator** — the engine passes its own `motion_generator` to each action's `TrajectoryBuilder`; if it is `None`, the action raises `ValueError` at execute time.
