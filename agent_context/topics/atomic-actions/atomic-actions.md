# Atomic Actions

## Entry Points

| What | Path |
|---|---|
| Base classes, typed targets, configs | `embodichain/lab/sim/atomic_actions/core.py` |
| Engine and global registry | `embodichain/lab/sim/atomic_actions/engine.py` |
| Trajectory helpers | `embodichain/lab/sim/atomic_actions/trajectory.py` |
| Built-in primitives | `embodichain/lab/sim/atomic_actions/primitives/` |
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
  └── WorldState            ← last_qpos + held_object/coordinated_held_object
```

All tensor shapes carry a leading batch dim `B = n_envs`.

## Core Types

### Typed Targets

Frozen dataclasses accepted by actions via their `TargetType` class variable.

| Target | Holds | Used by |
|---|---|---|
| `EndEffectorPoseTarget(xpos)` | `(4,4)`, `(B,4,4)` or `(B,n_waypoint,4,4)` EEF pose | `MoveEndEffector`, `Place`, `Press` |
| `JointPositionTarget(qpos)` | `(dof,)`, `(B,dof)` or `(B,n_waypoint,dof)` joint positions | `MoveJoints` |
| `NamedJointPositionTarget(name)` | Name resolved from `MoveJointsCfg.named_joint_positions` | `MoveJoints` |
| `GraspTarget(semantics)` | `ObjectSemantics` describing the object to grasp | `PickUp` |
| `HeldObjectPoseTarget(pose)` | `(4,4)` or `(B,4,4)` target pose for the held object | `MoveHeldObject` |
| `CoordinatedPickmentTarget(...)` | Shared object + left/right object-to-EEF transforms | `CoordinatedPickment` |
| `CoordinatedPlacementTarget(...)` | Two held-object states + target poses | `CoordinatedPlacement` |

### WorldState

Threaded between actions:
- `last_qpos: torch.Tensor` — shape `(B, robot.dof)`, robot joint positions at the start of the next action.
- `held_object: HeldObjectState | None` — object held by one gripper.
- `coordinated_held_object: CoordinatedHeldObjectState | None` — object jointly held by two grippers.

`HeldObjectState` stores the object's semantics plus the object-to-EEF transform and grasp pose (both `(B, 4, 4)`).

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
| `planner_type` | `str \| None` | `None` | `"toppra"` or `"neural"; required when `motion_source="motion_gen"` |

The base config is flat: every action cfg extends `ActionCfg` directly, even if it also carries hand open/close fields (see `PickUpCfg` / `PlaceCfg`).

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
| `PickUp` | `GraspTarget` | Approach → close gripper → lift; populates `held_object` |
| `MoveHeldObject` | `HeldObjectPoseTarget` | Move held object; preserves `held_object` |
| `Place` | `EndEffectorPoseTarget` | Lower → open gripper → retract; clears `held_object` |
| `Press` | `EndEffectorPoseTarget` | Close gripper → press down → return |
| `CoordinatedPickment` | `CoordinatedPickmentTarget` | Dual-arm shared-object pick |
| `CoordinatedPlacement` | `CoordinatedPlacementTarget` | Dual-arm placement |

## Implementing a New Action

1. Create a flat `@configclass` extending `ActionCfg` with a unique `name`.
2. Reuse an existing target or define a new frozen dataclass in `core.py`.
3. Subclass `AtomicAction` directly (do not inherit from another action). Set `TargetType` and compose a `TrajectoryBuilder`.
4. Implement `execute(self, target, state: WorldState) -> ActionResult`:
   - Resolve batched targets and start qpos via `self.builder`.
   - Call `self.builder.plan_arm_traj(..., cfg=self.cfg)` if using arm motion.
   - Return per-env `success` (a `(B,)` tensor if any env can fail, or `torch.ones(...)` for always-succeeding paths).
   - Embed the arm trajectory into full-DoF shape `(B, n_wp, robot.dof)`.
   - Advance `last_qpos` to the final row and preserve/update/clear `held_object`.
5. Register an instance with the engine or globally via `register_action(name, ActionClass)`.
6. Export from `primitives/__init__.py` and `atomic_actions/__init__.py`.

## Common Failure Modes

- **Forgetting `cfg=self.cfg` in `plan_arm_traj`** — without it, `motion_source` defaults to `"ik_interp"` and `planner_type` is ignored.
- **Treating `success` as scalar** — `ActionResult.success` is `(B,)` for all built-ins; use `success_all` or `success.all()` for a single bool.
- **Using `bool(action_result)` in new code** — still works but emits a `DeprecationWarning`; prefer `.success_all`.
- **Returning arm-only trajectory** — actions must embed into `(B, n_wp, robot.dof)` before returning.
- **`motion_source="motion_gen"` without a MotionGenerator** — the engine passes its own `motion_generator` to each action's `TrajectoryBuilder`; if it is `None`, the action raises `ValueError` at execute time.
- **Wrong planner_type** — `planner_type` must be registered in `MotionGenerator._support_planner_dict` (currently `"toppra"` or `"neural"`).
