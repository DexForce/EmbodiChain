(builtin-actions)=

# Built-in atomic actions

```{currentmodule} embodichain.lab.sim.atomic_actions
```

EmbodiChain ships nine built-in action implementations with stable skill IDs;
applications register the configured instances they need with an engine.
`Place` additionally accepts an `AssembleGoal`, so assembly reuses the same
release primitive instead of introducing a tenth skill ID.

All built-ins implement
`plan(request, context) -> ActionPlan`, where `request` is the engine-resolved
snapshot of an invocation revision. Constructors accept only optional typed
default `*Options`; the owning `AtomicActionEngine` supplies the shared motion
generator, trajectory builder, and control-part command profiles during
`register()` or `plan_action()`. Generic motion and recovery choices belong to
the invocation.

```{note}
The current manipulation primitives consume semantic `open` and `grasp`
commands through the control-part command abstraction. The shipped command
implementation is `JointPositionCommand`. A dexterous hand may register
calibrated joint-position commands immediately; non-position hand policies or
multi-stage in-hand manipulation require additional command types and phases.
```

## Visual catalog

The animations below are the focused simulator demos under
`scripts/tutorials/atomic_action/`.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} `MoveEndEffector`
:link: builtin-move-end-effector
:link-type: ref

`move_end_effector` · free-space EEF pose motion

<img src="../../../_static/atomic_actions/move_end_effector.gif" alt="MoveEndEffector demo" width="480" style="max-width: 100%;" />
:::

:::{grid-item-card} `MoveJoints`
:link: builtin-move-joints
:link-type: ref

`move_joints` · explicit or named joint-space motion

<img src="../../../_static/atomic_actions/move_joints.gif" alt="MoveJoints demo" width="480" style="max-width: 100%;" />
:::

:::{grid-item-card} `PickUp`
:link: builtin-pick-up
:link-type: ref

`pick_up` · approach, close, and lift

<img src="../../../_static/atomic_actions/pickup.gif" alt="PickUp demo" width="480" style="max-width: 100%;" />
:::

:::{grid-item-card} `MoveHeldObject`
:link: builtin-move-held-object
:link-type: ref

`move_held_object` · object-centric transport

<img src="../../../_static/atomic_actions/move_held_object.gif" alt="MoveHeldObject demo" width="480" style="max-width: 100%;" />
:::

:::{grid-item-card} `Place`
:link: builtin-place
:link-type: ref

`place` · approach, release, and retract

<img src="../../../_static/atomic_actions/place.gif" alt="Place demo" width="480" style="max-width: 100%;" />
:::

:::{grid-item-card} Assembly through `Place`
:link: builtin-assemble
:link-type: ref

`place` + `AssembleGoal` · base-relative placement

<img src="../../../_static/atomic_actions/assemble.gif" alt="Assembly demo" width="480" style="max-width: 100%;" />
:::

:::{grid-item-card} `Press`
:link: builtin-press
:link-type: ref

`press` · close, contact, and return

<img src="../../../_static/atomic_actions/press.gif" alt="Press demo" width="480" style="max-width: 100%;" />
:::

:::{grid-item-card} `CoordinatedPickment`
:link: builtin-coordinated-pickment
:link-type: ref

`coordinated_pickment` · dual-arm shared-object pick

<img src="../../../_static/atomic_actions/coordinated_pickment.gif" alt="CoordinatedPickment demo" width="480" style="max-width: 100%;" />
:::

:::{grid-item-card} `CoordinatedPlacement`
:link: builtin-coordinated-placement
:link-type: ref

`coordinated_placement` · align two held objects

<img src="../../../_static/atomic_actions/coordinated_placement.gif" alt="CoordinatedPlacement demo" width="640" style="max-width: 100%;" />
:::

:::{grid-item-card} `HandOver`
:link: builtin-hand-over
:link-type: ref

`hand_over` · transfer an attachment between arms

<img src="../../../_static/atomic_actions/hand_over.gif" alt="HandOver demo" width="480" style="max-width: 100%;" />
:::

::::

## Capability matrix

| Skill ID | Accepted goal | Required binding roles | Required profile commands | Required task state | Expected task effect |
|---|---|---|---|---|---|
| `move_end_effector` | `EndEffectorPoseGoal` | manipulator `primary` | none | none | none |
| `move_joints` | `JointPositionGoal` | manipulator `primary` | named target only: command matching `target` | none | none |
| `pick_up` | `GraspGoal` | manipulator + end effector `primary` | primary: `open`, `grasp` | semantic object/entity | attach object to `primary` manipulator |
| `move_held_object` | `HeldObjectPoseGoal` | manipulator + end effector `primary` | primary: `grasp` | object held by `primary` | preserve attachment |
| `place` | `PlaceGoal`, `AssembleGoal` | manipulator + end effector `primary` | primary: `open`, `grasp` | `AssembleGoal` requires an object held by `primary`; ordinary `PlaceGoal` has no planner-enforced attachment precondition | detach object |
| `press` | `PressGoal` | manipulator + end effector `primary` | primary: `grasp` | none | none |
| `coordinated_pickment` | `CoordinatedPickGoal` | manipulator + end effector `left`, `right` | both: `open`, `grasp` | semantic object/entity | create coordinated attachment; clear individual attachments |
| `coordinated_placement` | `CoordinatedPlacementGoal` | manipulator + end effector `placing`, `support` | placing: `open`, `grasp`; support: `grasp` | one individually held object per arm | optionally detach placing object; preserve support attachment |
| `hand_over` | `GraspGoal` | manipulator + end effector `source`, `destination` | both: `open`, `grasp` | object held by source arm | transfer attachment to destination arm |

### Binding role meanings

Roles are action-local semantic participant slots. They are keys declared by an
action, while the corresponding `ActionBinding` values are concrete
`Robot.control_parts` keys. A role that appears in both binding maps identifies
the manipulator and actuated hand/tool serving the same functional participant;
it does not make the two maps interchangeable.

| Role | Used by | Meaning |
|---|---|---|
| `primary` | Single-participant skills | Principal participant for this invocation; it has no inherent left/right or default-robot meaning |
| `source` | `hand_over` | Participant that initially holds and transfers the object |
| `destination` | `hand_over` | Participant that receives the object |
| `left`, `right` | `coordinated_pickment` | Participants associated with the goal's left/right grasp transforms |
| `placing` | `coordinated_placement` | Participant that aligns and optionally releases the placing object |
| `support` | `coordinated_placement` | Participant that keeps holding and positioning the support object |

The action's `manipulator_roles` and `end_effector_roles` declarations determine
which entries are required. The engine checks that those entries exist and that
every value resolves through `Robot.control_parts`; the caller or capability
binder must select a physically compatible arm and hand/tool combination.

`MoveJoints` is intentionally `agent_visible=False`: it is useful for home,
recovery, calibration, and scripted postures, but is not exposed to an Action
Agent by default.

## Shared goal and configuration rules

### Pose values and dynamic references

Pose-valued fields use `PoseGoalValue`, which accepts either an explicit tensor
or a late-bound scene reference:

```python
fixed = EndEffectorPoseGoal(xpos=target_pose)
tracked = EndEffectorPoseGoal(
    xpos=SceneEntityPose("tray", relative_pose=tray_to_tcp)
)
```

Explicit pose tensors use `(4, 4)` or `(B, 4, 4)`. Waypoint-capable fields in
`EndEffectorPoseGoal` and `PlaceGoal` also accept `(B, N, 4, 4)`.
`SceneEntityPose` resolves to the latest `(B, 4, 4)` pose from each
`SceneSnapshot`, checks optional perception confidence, and registers that
entity as a recovery dependency.

| Skill / field | `SceneEntityPose` accepted | Automatic scene-motion replan |
|---|---:|---:|
| `MoveEndEffector.xpos` | yes | yes |
| `MoveHeldObject.object_target_pose` | yes | yes |
| `Place.xpos` | yes | yes |
| `Press.xpos` | yes | yes |
| `CoordinatedPickGoal.object_target_pose` / `object_initial_pose` | yes | yes |
| `CoordinatedPlacementGoal` placing/support poses | yes | yes |
| `PickUp` / `HandOver` semantic entity lookup | not through `SceneEntityPose` | no automatic scene dependency |
| `AssembleGoal` base entity lookup | not through `SceneEntityPose` | latest pose is used when replanning, but base movement alone does not trigger it |

### Parameter ownership

Use this rule when configuring a built-in or adding a new one:

- the **goal** carries only the requested outcome;
- the **binding** carries semantic-role mappings to control-part names selected
  for this call; every value must be a key in the engine robot's
  `control_parts` mapping;
- typed **skill options** carry phase-specific behavior that may vary by
  invocation; an action may provide defaults;
- the engine's **control-part profiles** carry embodiment-specific semantic
  commands such as `open`, `grasp`, and named postures;
- `MotionPolicy` carries sample count, timing, motion source, limits,
  collision choice, and planner options;
- `RecoveryPolicy` carries all replan/retry thresholds and budgets.

All built-ins resolve participating arm and hand names exclusively from
`ActionBinding`. The engine then resolves the selected control part's profile
and checks each joint-position command against its DoF. Invocation-level
`ActionControlOverrides` may replace a command by binding role for one explicit
revision.

### Planning and effect semantics

Every action returns a per-environment `plan_success` mask and one or more
full-robot trajectories. `plan_success=True` means motion planning succeeded;
it does not prove contact or object transfer. Actions that change attachment
state declare a `StateDelta`. Offline `compile()` projects it hypothetically;
closed-loop execution commits it only after external effect verification.

(builtin-move-end-effector)=

## `MoveEndEffector`

Plans a free-space motion for a bound manipulator to reach one EEF pose or an
ordered set of pose waypoints.

| Contract | Value |
|---|---|
| Skill ID | `move_end_effector` |
| Goal | `EndEffectorPoseGoal(xpos=...)` |
| Binding | manipulator role `primary` |
| Motion | EEF planning from observed arm qpos; output expanded to full robot DoF |
| Completion | `EEF_GOAL_REACHED` |
| Effect | none |
| Skill options | none; reusable motion choices are in `MotionPolicy` |

Use an explicit pose for a fixed target, `SceneEntityPose` for a tracked target,
or `(B, N, 4, 4)` for intermediate waypoints. The action does not command an end
effector/hand resource.

**Example:** `scripts/tutorials/atomic_action/move_end_effector.py`

(builtin-move-joints)=

## `MoveJoints`

Plans directly in joint space. This is appropriate for known safe postures,
homing, scripted recovery, or motions whose desired outcome is a qpos rather
than an EEF pose.

| Contract | Value |
|---|---|
| Skill ID | `move_joints` |
| Goal | `JointPositionGoal(target=...)` |
| Binding | manipulator role `primary` |
| Motion | joint planning/interpolation from observed qpos; supports joint waypoints |
| Completion | `JOINT_GOAL_REACHED` |
| Effect | none |
| Agent visibility | hidden by default (`agent_visible=False`) |

`target` accepts an explicit qpos tensor with shape `(control_dof,)`,
`(B, control_dof)`, or `(B, N, control_dof)`, or a non-empty string resolved
from the bound manipulator's `ControlPartCommandProfile`. Named poses remain
embodiment knowledge without becoming separate goal types:

```python
engine = AtomicActionEngine(
    motion_generator,
    control_profiles={
        "left_arm": ControlPartCommandProfile.joint_positions(home=home_qpos),
    },
)
engine.register(MoveJoints())

explicit_goal = JointPositionGoal(target=home_qpos)
named_goal = JointPositionGoal(target="home")
```

**Example:** `scripts/tutorials/atomic_action/move_joints.py`

(builtin-pick-up)=

## `PickUp`

Plans **approach -> close hand -> lift** and declares the object attached to the
bound manipulator.

| Contract | Value |
|---|---|
| Skill ID | `pick_up` |
| Goal | `GraspGoal(semantics=..., grasp_xpos=None)` |
| Binding | manipulator + end effector role `primary` |
| Precondition | `ObjectSemantics.entity` is set; an `AntipodalAffordance` is required when no explicit grasp pose is supplied |
| Effect | write `HeldObjectState` for the bound manipulator and clear overlapping coordinated attachment state |
| Verification | the attachment effect must be verified during closed-loop execution |

`grasp_xpos` may be `(4, 4)` or `(B, 4, 4)`. When omitted, the action samples
valid affordance grasps, evaluates reachability, and stores the selected
`object_to_eef` transform in the expected held-object state. Later
object-centric skills reuse that transform.

`PickUp` requires `open` and `grasp` commands on the bound end-effector profile.
Important `PickUpOptions` fields:

| Field | Purpose |
|---|---|
| `pre_grasp_distance`, `approach_direction` | Pre-grasp offset and approach direction |
| `lift_height`, `hand_interp_steps` | Lift distance and close-phase discretization |
| `pick_object_part` | Affordance region: currently `center`, `top`, or `bottom` |
| `approach_alignment_max_angle` | Optional TCP approach-alignment filter |
| `downstream_object_target_poses` | Optional future reachability constraints used in grasp selection |
| `obj_upright_direction`, `rotate_upright` | Optional orientation-selection behavior |

The semantic entity pose is read when planning occurs, but it is not currently a
`SceneEntityPose` dependency. Object motion alone therefore does not trigger
automatic dynamic-goal replanning.

**Example:** `scripts/tutorials/atomic_action/pickup.py`

(builtin-move-held-object)=

## `MoveHeldObject`

Moves an already attached object to an object-frame target while keeping the
hand closed. The caller specifies the desired **object pose**, not an EEF pose;
the action derives `target_object_pose @ object_to_eef` from verified task state.

| Contract | Value |
|---|---|
| Skill ID | `move_held_object` |
| Goal | `HeldObjectPoseGoal(object_target_pose=...)` |
| Binding | manipulator + end effector role `primary` |
| Precondition | a `HeldObjectState` exists for the bound manipulator, normally from `PickUp` |
| Motion | single object-centric transport phase with closed-hand qpos |
| Effect | none; the existing attachment is preserved |
| Dynamic target | explicit pose or `SceneEntityPose` |

The bound end-effector profile must provide `grasp`; optional upright-transport
settings belong to `MoveHeldObjectOptions`. The arm and hand are selected by
`ActionBinding`; generic timing and trajectory sampling remain in
`MotionPolicy`.

**Example:** `scripts/tutorials/atomic_action/move_held_object.py`

(builtin-place)=

## `Place`

Plans **approach/descend -> open hand -> retract**. A multi-waypoint
`PlaceGoal` visits all supplied release waypoints in order and opens at the last
one.

| Contract | Value |
|---|---|
| Skill ID | `place` |
| Goal | `PlaceGoal(xpos=..., tcp_symmetry="none")` |
| Binding | manipulator + end effector role `primary` |
| State | consumes the bound manipulator's attachment when present |
| Effect | detach the object and clear overlapping coordinated attachment state |
| Verification | release must be verified during closed-loop execution |
| Dynamic target | explicit pose/waypoints or `SceneEntityPose` |

Set `tcp_symmetry="z_roll_180"` only if TCP x/y can be flipped while TCP z and
translation remain physically equivalent. The action selects the closer
orientation variant from the observed starting state and uses it consistently
across all waypoints.

The bound end-effector profile must provide `open` and `grasp`. Important
`PlaceOptions` fields:

| Field | Purpose |
|---|---|
| `lift_height` | Approach and retract height |
| `hand_interp_steps` | Open-phase discretization |
| `max_approach_retract_z` | Optional world-Z ceiling for approach/retract poses |
| `cartesian_waypoint_count` | Fixed-orientation translation keyframes per segment |

**Example:** `scripts/tutorials/atomic_action/place.py`

(builtin-assemble)=

### Assembly through `Place`

`Place` also accepts `AssembleGoal(affordance=...)`. There is no separate
assembly skill: it derives the assemble-object target from the base object's
live pose and reuses the normal place/release phases.

```text
base_object_pose @ assemble_to_base_pose = assemble_object_target_pose
assemble_object_target_pose @ held.object_to_eef = release_eef_pose
```

The `AssembleAffordance` identifies the base and assemble objects, stores the
relative pose, and must provide `base_object_entity`. A prior verified `PickUp`
must have populated the held object's `object_to_eef` transform. Planning then
declares the same detach effect as a normal place.

The base entity's current pose is read each time `plan()` runs. Because the
goal does not yet encode that entity through `SceneEntityPose`, base movement by
itself does not invalidate an executing plan; another recovery trigger is
required before the newer pose is resolved.

**Example:** `scripts/tutorials/atomic_action/assemble.py`

(builtin-press)=

## `Press`

Plans **close hand -> move to contact pose -> return to the observed starting
arm qpos**. It is intended for button-like or contact interactions where the
arm should retreat along its planned path after reaching the target.

| Contract | Value |
|---|---|
| Skill ID | `press` |
| Goal | `PressGoal(xpos=...)` |
| Binding | manipulator + end effector role `primary` |
| Motion | close, press, joint-space return |
| Effect | none; existing attachment state is unchanged |
| Dynamic target | explicit pose or `SceneEntityPose` |

The bound end-effector profile must provide `grasp`, while
`PressOptions.hand_interp_steps` controls the close interpolation. The arm and
hand control parts come from `ActionBinding`. Contact detection is not itself a
symbolic effect in the current action; applications that require force/contact
confirmation should verify it externally.

**Example:** `scripts/tutorials/atomic_action/press.py`

(builtin-coordinated-pickment)=

## `CoordinatedPickment`

Coordinates two arms around one shared object: **approach both grasps -> close
both hands -> lift -> move object -> hold**.

| Contract | Value |
|---|---|
| Skill ID | `coordinated_pickment` |
| Goal | `CoordinatedPickGoal` |
| Binding | manipulator + end effector roles `left` and `right` |
| Goal geometry | shared-object target pose plus left/right `object_to_eef` transforms; optional initial object pose |
| Effect | clear individual left/right attachments and create `CoordinatedHeldObjectState[(left, right)]` |
| Verification | coordinated attachment must be externally verified |

The object target and optional initial pose may use `SceneEntityPose`. When no
initial pose is supplied, `ObjectSemantics.entity` provides the object's current
pose.

Both bound end-effector profiles must provide `open` and `grasp`. Important
`CoordinatedPickmentOptions` fields group into:

- `pre_grasp_distance` and `lift_height`;
- `object_motion_keyframes`, `hand_interp_steps`, and `hold_steps`.

The left/right arms and hands come exclusively from the corresponding binding
roles. Coordinated dual-arm planning with `motion_source="motion_gen"` is not
supported by the cuRobo backend; use the supported IK/interpolation path for
this primitive.

**Example:** `scripts/tutorials/atomic_action/coordinated_pickment.py`

(builtin-coordinated-placement)=

## `CoordinatedPlacement`

Moves a support object and a placing object together: **align both objects ->
hold -> optionally release the placing hand -> retreat the placing arm**.

| Contract | Value |
|---|---|
| Skill ID | `coordinated_placement` |
| Goal | `CoordinatedPlacementGoal` |
| Binding | manipulator + end effector roles `placing` and `support` |
| Precondition | separate `HeldObjectState` entries exist for both bound arms |
| Goal geometry | placing/support object target poses, optional height offsets, optional release override |
| Effect | preserve support attachment; remove or preserve placing attachment according to `release`; clear overlapping coordinated state |

Both object targets may use `SceneEntityPose`, so either can participate in
dynamic-goal invalidation. Goal-level height/release values override
`CoordinatedPlacementOptions` for that invocation.

The placing profile must provide `open` and `grasp`; the support profile must
provide `grasp`. Important `CoordinatedPlacementOptions` fields group into:

- default `release`, placing/support height offsets, and `lift_height`;
- `hand_interp_steps`, `hold_steps`, and `retreat_steps`.

The placing/support arms and hands come exclusively from the corresponding
binding roles. The same cuRobo restriction as coordinated pickment applies to dual-arm
`motion_source="motion_gen"` planning.

**Example:** `scripts/tutorials/atomic_action/coordinated_placement.py`

(builtin-hand-over)=

## `HandOver`

Transfers an already held object from one arm to another: **move source to the
handover pose -> destination approaches and grasps -> source releases and
retreats -> destination delivers**.

| Contract | Value |
|---|---|
| Skill ID | `hand_over` |
| Goal | `GraspGoal(semantics=...)` |
| Binding | manipulator + end effector roles `source` and `destination` |
| Precondition | source arm has a verified `HeldObjectState`; semantic object supports destination grasp selection |
| Effect | remove source attachment and create destination `HeldObjectState` |
| Verification | attachment transfer must be externally verified |

Both source and destination end-effector profiles must provide `open` and
`grasp`. `HandOverOptions` owns the destination grasp region and approach
direction, middle/final object poses, and phase distances/counts. The
source/destination arm and hand control parts come exclusively from the
corresponding `ActionBinding` roles.

The middle and final poses are currently option tensors rather than
`SceneEntityPose` goal fields. Consequently, handover supports tracking-error
and timeout recovery, but does not automatically invalidate a moving handover
point. An application can submit a newer invocation revision with updated
`HandOverOptions`; the action also queries the semantic object's live
orientation when replanning and preserves it at the supplied middle/final
positions.

As with the other coordinated primitive, cuRobo does not currently support its
dual-arm `motion_source="motion_gen"` path.

**Example:** `scripts/tutorials/atomic_action/hand_over.py`

## Running the demos

Every focused script is interactive by default. Add `--auto_play` to skip
keyboard prompts and combine it with `--headless --device cpu` for a headless
run that records video under `outputs/videos`:

```bash
python scripts/tutorials/atomic_action/move_end_effector.py --headless --auto_play --device cpu
python scripts/tutorials/atomic_action/pickup.py --headless --auto_play --device cpu
python scripts/tutorials/atomic_action/hand_over.py --headless --auto_play --device cpu
```

See {doc}`/tutorial/atomic_actions` for engine setup, static compilation,
closed-loop execution, effect verification, and custom-action guidance.
