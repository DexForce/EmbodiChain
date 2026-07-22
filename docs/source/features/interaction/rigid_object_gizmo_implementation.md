# Rigid-object window gizmo: current implementation and DexSim migration guide

## 1. Document purpose

This document describes the current EmbodiChain implementation of window-driven
gizmo control for rigid objects and proposes a migration boundary for moving the
feature into DexSim.

The intended audience is the DexSim implementer who will replace the current
Python orchestration with an engine-level input controller.

Implementation snapshot:

- EmbodiChain branch: `feat/rigid-object-window-gizmo`
- EmbodiChain implementation commit: `38f0b67b`
- EmbodiChain pull request: `DexForce/EmbodiChain#420`
- DexSim source inspected from branch `dev`, commit `69a51b104`
- Document date: 2026-07-22

## 2. User-visible behavior

The current feature provides the following interaction:

1. The user left-clicks a rigid object in the main viewer. DexSim performs the
   raycast and caches the selected object.
2. The user presses `G`.
3. If the selected object is an eligible EmbodiChain `RigidObject`, its original
   body type is saved and the object is changed to kinematic.
4. A transform gizmo is created and attached to the rigid object.
5. Dragging the gizmo updates the rigid object's transform.
6. Pressing `G` again removes the active gizmo and restores the object's original
   body type.

The second `G` press always exits the active interaction. It does not depend on
the current raycast selection, which may have changed after the gizmo was
enabled.

The same cleanup and body-type restoration also run when:

- `SimulationManager.disable_gizmo(uid)` is called;
- the viewer window is closed; or
- the simulation is destroyed.

Current eligibility rules:

| Condition | Result |
|---|---|
| Dynamic rigid object | Accepted; temporarily changed to kinematic |
| Kinematic rigid object | Accepted; remains kinematic |
| Static rigid object | Rejected |
| Robot, sensor, light, plane, or unregistered DexSim object | Rejected |
| `num_envs != 1` | Rejected |
| Object already has a non-window-owned gizmo | Rejected |

## 3. Current component layout

| Component | Location | Responsibility |
|---|---|---|
| `WindowGizmoCfg` | `embodichain/lab/sim/cfg.py` | Enables or disables default `G` hotkey registration |
| `SimulationManagerCfg.window_gizmo` | `embodichain/lab/sim/sim_manager.py` | Exposes the interaction configuration |
| `_WindowGizmoState` | `embodichain/lab/sim/sim_manager.py` | Stores active EmbodiChain UID and original body type |
| `enable_window_gizmo_hotkey()` | `embodichain/lab/sim/sim_manager.py` | Creates the Python `ObjectManipulator`, enables selection caching, and handles `G` |
| `_resolve_selected_rigid_object_uid()` | `embodichain/lab/sim/sim_manager.py` | Maps the DexSim raycast object back to an EmbodiChain rigid object |
| `toggle_selected_rigid_object_gizmo()` | `embodichain/lab/sim/sim_manager.py` | Implements the enter/exit state machine and rollback |
| `enable_gizmo()` / `disable_gizmo()` | `embodichain/lab/sim/sim_manager.py` | Owns high-level gizmo instances and the shared DexSim drag controller |
| `Gizmo` | `embodichain/lab/sim/objects/gizmo.py` | Wraps native DexSim gizmos for rigid objects, robots, and cameras |
| `Gizmo.destroy()` | `embodichain/lab/sim/objects/gizmo.py` | Detaches callbacks/parents, removes proxies, and calls `Arena.remove_gizmo()` |
| `ObjectManipulator` | DexSim `DFObjectManipulator.hpp` | Receives raycast selection and optionally caches the last valid hit |
| `GizmoController` | DexSim `DFGizmoController.*` | Handles mouse dragging of gizmo axes and rotation rings |
| `IGizmo` / `GizmoX` | DexSim `DFGizmo.*` | Follows a target node and flushes gizmo transforms to it |
| `Arena` | DexSim `Arena.*` | Creates, stores, and removes native gizmos |

## 4. Current runtime state

EmbodiChain keeps one window-owned gizmo state per `SimulationManager`:

```python
@dataclass
class _WindowGizmoState:
    uid: str
    original_body_type: str
```

Related manager fields are:

```python
self._window_gizmo_hotkey_enabled: bool
self._window_gizmo_input_control: ObjectManipulator | None
self._window_gizmo_state: _WindowGizmoState | None
self._gizmos: dict[str, Gizmo]
self._gizmo_controller: GizmoController | None
```

There can be only one window-owned rigid-object gizmo at a time. The generic
`_gizmos` registry may also contain manually enabled robot, camera, or rigid
object gizmos.

## 5. End-to-end event flow

### 5.1 Registration

When `SimulationManager` creates or opens a non-headless window:

1. `SimulationManagerCfg.window_gizmo.enable_hotkey` is read.
2. `enable_window_gizmo_hotkey()` creates a Python subclass of DexSim
   `ObjectManipulator`.
3. `enable_selection_cache(True)` is called on the manipulator.
4. The manipulator is registered with `Windows.add_input_control()`.

Registration is idempotent. If `_window_gizmo_input_control` already exists, no
second controller is added.

### 5.2 Raycast selection

DexSim performs a surface raycast on left mouse down. For each registered
`ObjectManipulator`, the view calls `Select(camera, result)`.

With selection caching enabled, `ObjectManipulator::Select()` updates the
cached result only when the new result contains an object. Clicking empty space
therefore does not clear the last valid object selection.

The Python hotkey handler later reads:

```python
self.selected_object
```

This value is a DexSim `IObject`, not an EmbodiChain `RigidObject`.

### 5.3 First `G` press: enter gizmo control

The Python `WindowGizmoEvent.on_key_down()` calls:

```python
sim.toggle_selected_rigid_object_gizmo(self.selected_object)
```

The manager then performs the following operations in order:

1. Confirm there is no active window-owned gizmo.
2. Require a non-null raycast selection.
3. Require `num_envs == 1`.
4. Read `selected_object.get_user_id()`.
5. Iterate all registered `RigidObject._entities` and compare entity user IDs.
6. Reject a selection that cannot be mapped to an EmbodiChain rigid object.
7. Reject static objects.
8. Reject objects that already have another gizmo.
9. Save `rigid_object.body_type`.
10. Call `rigid_object.set_body_type("kinematic")`.
11. Call `SimulationManager.enable_gizmo(uid)`.
12. If gizmo creation fails, restore the saved body type immediately.
13. On success, store `_WindowGizmoState(uid, original_body_type)`.

The object becomes kinematic before the gizmo is enabled. This ordering prevents
physics from competing with direct transform updates while the object is being
dragged.

### 5.4 Gizmo creation and dragging

For a rigid object, the EmbodiChain `Gizmo` wrapper:

1. calls `Arena.create_gizmo(axis_options, ring_options)`;
2. obtains the first rigid-object entity node with `target._entities[0].node`;
3. calls native `gizmo.follow(target_node)`; and
4. installs a Python local-pose callback that calls
   `node.set_transform(local_pose, flag)`.

The callback is functionally equivalent to DexSim's native
`GizmoX::DefaultTransformFlushCallback()`, which already calls
`target->SetTransform(transform, flag)`. The Python callback is therefore not
required after this feature is migrated into DexSim.

`SimulationManager.enable_gizmo()` also creates one shared native
`GizmoController` and registers it with the window. That controller performs
axis/plane translation and ring rotation based on mouse raycasts.

### 5.5 Second `G` press: exit gizmo control

If `_window_gizmo_state` exists, the toggle method ignores the current selection
and calls `disable_gizmo(active_uid)`.

`disable_gizmo()`:

1. removes the wrapper from `SimulationManager._gizmos`;
2. calls `Gizmo.destroy()`; and
3. restores the original body type in a `finally` path.

`Gizmo.destroy()` performs the native resource cleanup:

1. clear the transform callback;
2. detach the gizmo from its target or proxy;
3. remove a robot/camera proxy actor when present;
4. call `Arena.remove_gizmo(native_gizmo)`; and
5. release Python references.

Calling `Arena.remove_gizmo()` is essential. Detaching the node alone is not
sufficient because `Arena::gizmos_` holds a strong reference. Without explicit
removal, the gizmo remains in the scene and continues to render.

### 5.6 Window and simulation cleanup

`SimulationManager.close_window()` disables the active window-owned gizmo before
closing the DexSim window. It then clears the Python input-control references.

`SimulationManager._deferred_destroy()` also disables the active window-owned
gizmo before removing the remaining generic gizmos and destroying the world.

## 6. State machine and invariants

The current logical state machine is:

```text
IDLE
  | G + eligible cached selection
  v
ACTIVATING
  | set kinematic + create/follow gizmo succeeds
  v
ACTIVE
  | G, disable_gizmo(), close_window(), or destroy()
  v
DEACTIVATING
  | remove gizmo + restore original actor type
  v
IDLE

ACTIVATING -- any failure --> rollback actor type --> IDLE
```

Required invariants:

1. `ACTIVE` implies the target is dynamic-capable and currently kinematic.
2. The original actor type is captured before any mutation.
3. Every exit path attempts actor-type restoration.
4. Gizmo creation failure leaves no native gizmo and restores the actor type.
5. The second `G` exits the active target instead of switching to a newly
   selected target.
6. Native gizmo destruction must call `Arena.remove_gizmo()`.
7. A manually created gizmo is never adopted as a window-owned gizmo.

## 7. Existing DexSim capabilities used by the feature

DexSim already exposes nearly all primitives needed for a native implementation:

| Capability | Existing DexSim API |
|---|---|
| Viewer key callbacks | `DF::InputControl::OnKeyDown()` / `OnKeyUp()` |
| Surface selection | View raycast plus `ObjectManipulator::Select()` |
| Cached last valid selection | `ObjectManipulator::EnableSelectionCache()` |
| Selected scene object | `ObjectManipulator::GetSelectedObject()` |
| Actor-type query | `IObject::GetActorType()` |
| Runtime dynamic/kinematic switch | `IObject::SetActorType()` |
| Gizmo creation | `Arena::CreateGizmo()` |
| Direct target following | `IGizmo::Follow(INodeRef)` |
| Default transform propagation | `GizmoX::DefaultTransformFlushCallback()` |
| Gizmo dragging | `GizmoController` |
| Gizmo removal | `Arena::RemoveGizmo()` |
| Python exposure | pybind classes for `ObjectManipulator`, `GizmoController`, `IObject`, `Gizmo`, and `Arena` |

The main missing piece is an engine-owned controller that coordinates these
existing APIs into one interaction lifecycle.

## 8. Current limitations and migration opportunities

### 8.1 EmbodiChain UID reverse lookup

The raycast result is already the native object that must be manipulated, but
the current Python code converts it to an EmbodiChain UID by scanning every
`RigidObject` entity.

This lookup is:

- unnecessary in DexSim;
- dependent on EmbodiChain private field `_entities`;
- linear in the number of registered entities; and
- the reason non-EmbodiChain native rigid objects are rejected.

A DexSim implementation should retain the selected `IObjectRef` directly.

### 8.2 Single-environment restriction

The current feature rejects `num_envs != 1` because the generic EmbodiChain
`Gizmo` wrapper always follows `target._entities[0]` and its constructor enforces
single-world use.

DexSim selection identifies one concrete `IObject`, so an engine-level controller
can support whichever arena object was actually clicked. The migration can
remove this restriction unless the renderer has a separate multi-view limitation.

### 8.3 Python callback and render-thread boundary

The hotkey is implemented by a Python override of `ObjectManipulator.on_key_down`.
DexSim warms the pybind override cache when adding the control, but key handling
still crosses into Python.

A native controller avoids the GIL boundary and keeps selection, actor mutation,
gizmo creation, and cleanup on the engine side.

### 8.4 Key-repeat behavior

The current implementation toggles on every `on_key_down` callback. If a window
backend emits repeated key-down events while `G` is held, the interaction can
toggle more than once.

The DexSim controller should use an edge guard:

- ignore `OnKeyDown(G)` while `g_key_down_` is already true;
- set `g_key_down_ = true` on the first key down; and
- clear it in `OnKeyUp(G)`.

### 8.5 Stale selection and target removal

The cached raycast result contains a strong/reference-counted object handle. The
native implementation must define behavior when the selected or active object is
removed from its arena.

Recommended behavior:

- validate that the selected object still belongs to a live scene before enter;
- automatically deactivate if the active object is removed; and
- never let selection caching keep a removed scene object alive indefinitely.

### 8.6 Controller ownership

EmbodiChain currently keeps separate references for the hotkey
`ObjectManipulator` and the drag `GizmoController`. Closing the window clears the
Python references, while the window/view owns the active registration.

DexSim should make ownership explicit. A controller registered with a window
must be removed before controller destruction, and controller teardown must
deactivate any active gizmo and restore the target actor type.

## 9. Recommended DexSim design

### 9.1 Preferred abstraction

Add a native controller dedicated to object-level gizmo interaction, for
example:

```cpp
class ObjectGizmoController : public ObjectManipulator {
public:
    ObjectGizmoController(Arena* arena,
                          InputKey hotkey = KEY_SCANCODE_G);

    void OnKeyDown(int key) override;
    void OnKeyUp(int key) override;

    bool Activate(IObjectRef object);
    void Deactivate();
    bool IsActive() const;
    IObjectRef ActiveObject() const;

private:
    Arena* arena_ = nullptr;
    InputKey hotkey_ = KEY_SCANCODE_G;
    bool hotkey_down_ = false;
    IObjectRef active_object_;
    IPhysicBody::ActorType original_actor_type_ =
            IPhysicBody::ActorType::ACTOR_NONE;
    IGizmoRef active_gizmo_;
    GizmoController drag_controller_;
};
```

The exact ownership of `GizmoController` may instead remain at the window/world
level if one shared drag controller is preferred. The important design point is
that the hotkey controller owns the activation state and cleanup contract.

Avoid adding the toggle logic directly to the existing `GizmoController` unless
that class is intentionally expanded from a drag-only controller into a complete
selection-and-lifecycle controller.

### 9.2 Activation algorithm

Recommended native activation logic:

```text
Activate(selected):
    if active:
        return false
    if selected is null or no longer belongs to a live scene:
        return false

    original = selected.GetActorType()
    if original not in {DYNAMIC, KINEMATIC}:
        return false

    selected.SetActorType(KINEMATIC)
    try:
        gizmo = arena.CreateGizmo(options)
        gizmo.Follow(selected as INode)
        commit active_object, original_type, and gizmo
        return true
    on failure:
        remove partially created gizmo if present
        selected.SetActorType(original)
        return false
```

Because `GizmoX` has a native default transform callback, a rigid-object
implementation does not need to install a Python callback.

### 9.3 Deactivation algorithm

Recommended native deactivation logic:

```text
Deactivate():
    move active state into local variables
    clear controller active state to prevent re-entrancy

    detach gizmo target
    arena.RemoveGizmo(gizmo)

    always attempt:
        object.SetActorType(original_type)
```

Actor restoration must be guaranteed even if gizmo removal fails. In C++, use a
scope guard or an equivalent no-fail cleanup structure.

### 9.4 Eligibility policy

At the DexSim layer, the default policy should accept any selected `IObject`
whose actor type is dynamic or kinematic. Static, soft, cloth, and objects with
no physical body should be rejected.

If applications need filtering, prefer engine-native options that do not require
a Python callback on the render thread, such as:

- selectable layer mask;
- actor-type mask;
- user-ID allow/deny set; or
- a C++ predicate interface.

### 9.5 Proposed Python binding

A minimal binding could expose:

```python
controller = dexsim.engine.ObjectGizmoController(
    arena=world.get_env(),
    hotkey=InputKey.SCANCODE_G,
)
controller.enable_selection_cache(True)
window.add_input_control(controller)

controller.is_active
controller.active_object
controller.activate(obj)
controller.deactivate()
```

A higher-level convenience API may additionally be provided by `World` or
`Windows`, but the controller object should remain accessible so applications
can configure and remove it explicitly.

## 10. Migration boundary in EmbodiChain

After DexSim provides the native controller, EmbodiChain should retain:

- `SimulationManagerCfg.window_gizmo.enable_hotkey` as application-level policy;
- creation and registration of the DexSim controller when the window opens;
- generic manual `enable_gizmo()` support for robots and cameras; and
- user-facing documentation.

EmbodiChain can remove or simplify:

- `_WindowGizmoState`;
- `_resolve_selected_rigid_object_uid()`;
- `toggle_selected_rigid_object_gizmo()`;
- the nested Python `WindowGizmoEvent` class;
- body-type restoration hooks inside generic `disable_gizmo()`; and
- the rigid-object-specific Python transform callback.

The EmbodiChain `Gizmo` wrapper remains useful for robot IK and camera proxy
control. Do not remove those paths as part of the rigid-object migration.

`Gizmo.destroy()` should continue to call `Arena.remove_gizmo()` for any gizmo
still created through the EmbodiChain wrapper.

## 11. Suggested migration phases

### Phase 1: implement and test in DexSim

1. Add the native controller and pybind API.
2. Reuse `ObjectManipulator` selection caching.
3. Reuse `GizmoController` for dragging.
4. Add transactional actor-type restoration.
5. Add explicit arena and window teardown behavior.
6. Add an interactive example to DexSim.

### Phase 2: switch EmbodiChain to the DexSim controller

1. Instantiate the native controller from `SimulationManager`.
2. Keep the existing config flag and hotkey behavior.
3. Delete the Python selection-to-UID bridge and state machine.
4. Keep a temporary compatibility fallback only if EmbodiChain must support an
   older DexSim version.

### Phase 3: remove compatibility code

1. Require the DexSim version containing the native feature.
2. Remove the fallback and obsolete tests.
3. Retain cross-layer integration tests in EmbodiChain.

## 12. Required DexSim test matrix

The DexSim implementation should cover at least:

| Scenario | Expected result |
|---|---|
| No cached selection, press `G` | No activation |
| Select dynamic object, press `G` | Actor becomes kinematic; gizmo exists |
| Press `G` again | Gizmo removed; actor returns to dynamic |
| Select kinematic object | Actor remains kinematic after exit |
| Select static object | Rejected without mutation |
| Select non-physical object | Rejected without mutation |
| Gizmo creation failure | Actor type rolled back; no active state |
| Active object removed | Controller deactivates safely |
| Window/controller destroyed while active | Gizmo removed and actor restored |
| Hold `G` and receive repeated key-down events | Only one transition per key press |
| Repeated enable/disable cycles | `Arena::GetAllGizmos()` does not grow |
| Select object in another arena/view | Correct concrete object is controlled |

EmbodiChain currently covers the Python lifecycle with:

- `tests/sim/test_sim_manager.py`; and
- `tests/sim/objects/test_gizmo.py`.

The native DexSim tests should additionally assert the arena's actual gizmo
registry and scene object lifetime rather than only using mocks.

## 13. Migration acceptance criteria

The migration is complete when:

1. Left-click plus `G` works without a Python `ObjectManipulator` override.
2. No EmbodiChain UID reverse lookup is needed.
3. Dynamic and kinematic actor types are restored on every exit and failure path.
4. The second `G` removes the native gizmo from both rendering and
   `Arena::gizmos_`.
5. Window, controller, object, arena, and world teardown are safe in any order.
6. Key repeat cannot cause unintended double toggles.
7. The feature works for a concrete selected object without the current
   EmbodiChain `num_envs == 1` restriction, or DexSim documents why that
   restriction remains necessary.
8. EmbodiChain keeps only configuration/registration glue for rigid objects.

## 14. Open design questions

Before implementation, decide:

1. Should the new controller own a private `GizmoController`, or should the
   window/world provide one shared drag controller?
2. Should `ObjectGizmoController` be registered explicitly by applications, or
   enabled by a `Windows`/`WorldConfig` option?
3. How should the controller observe active-object removal?
4. Should switching selection while active be rejected, or should a single `G`
   atomically transfer control to the new selection? The current contract
   requires an explicit exit first.
5. Should dynamic velocity be preserved and restored after a kinematic editing
   session, or should the object resume with zero velocity? The current feature
   restores only actor type.
6. Which selection filters are needed without introducing a Python callback on
   the render thread?

