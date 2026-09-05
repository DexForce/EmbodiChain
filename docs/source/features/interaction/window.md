# Window interaction

This section describes the default window interaction controls available in the simulation. These controls allow users to interact with the simulation environment using keyboard, mouse, and customizable input events.

The main visualization window is provided by **DexSim**. When
`SimulationManagerCfg.headless=False` or `SimulationManager.open_window()` is
called, DexSim creates the viewer with **ORBIT** camera control by default.
The native window and the Viser backend are mutually exclusive;
`SimulationManager.open_window()` safely returns `False` without opening a
native window while Viser is enabled.
Likewise, `SimulationManager.start_visualization()` rejects Viser startup while
the native window is open.
See {doc}`Viser browser visualization </overview/sim/viser_visualization>` for
the headless browser frontend.

## Default Window Controls

### Mouse Controls

| Input | Operation |
|-------|-----------|
| Left drag / Middle drag | Rotate around the current target point. |
| Right drag | Pan the camera and target together. |
| Mouse wheel | Dolly the camera closer to or farther from the target. |

### Keyboard Controls

| Input | Operation |
|-------|-----------|
| Space | Reset the window camera to its home view. |
| Left Ctrl + W / S | Temporarily translate the view forward / backward. |
| Left Ctrl + A / D | Temporarily translate the view left / right. |
| Left Ctrl + Q / E | Temporarily translate the view down / up. |

In ORBIT mode, plain `W/A/S/D/Q/E` does not move the view. Hold **Left Ctrl** while pressing those keys to translate both the camera eye and target.

### Selection and Focus

| Input | Operation |
|-------|-----------|
| Left click | Select the object under the cursor in the main visualization window. |
| F | Focus the selected object and frame it in the view. |
| L | Toggle selection log output in the terminal. Selection logs are disabled by default. When enabled, left-clicking an object prints its id, name, world position, and rotation. |

### EmbodiChain Extensions

| Input | Operation |
|-------|-----------|
| **Viewer recording (toggle)** | Press **`r`** to **start** recording what the interactive viewer shows, and press **`r`** again to **stop** and save as MP4 videos. Recording uses a hidden camera that follows the live viewer camera pose, so the exported videos match the on-screen view. Useful for debugging and recording demos. |
| **Print camera pose** | Press **`p`** to print the current viewer pose as an executable `window.set_look_at(...)` call. |

Recording hotkey registration is controlled by `SimConfig.window_record.enable_hotkey` (enabled by default). You can also call `SimulationManager.start_window_record()`, `stop_window_record()`, or `toggle_window_record()` programmatically.

The camera-pose hotkey is controlled by `SimulationManagerCfg.window_camera_pose.enable_hotkey` and prints look-at form by default. Set `SimulationManagerCfg.window_camera_pose.convert_to_look_at=False` to print the raw 4x4 pose matrix instead. The same output can be requested programmatically with `SimulationManager.print_window_camera_pose()`.

### Entity Gizmo Control

DexSim owns native entity selection and manipulation. EmbodiChain enables it
automatically when the first native window opens, including a window created
with `SimulationManagerCfg(headless=False)`. Pure headless and Viser runs do not
automatically create native entity gizmos.

To start with native interaction disabled, use
`SimulationManagerCfg(enable_entity_gizmo=False)`. Gym JSON/YAML deployments
accept the top-level field `enable_entity_gizmo: false` as well. At runtime,
`sim.disable_entity_gizmo()` disables interaction and cancels automatic
enablement even before the first window opens. Closing and reopening a window
preserves the controller's current enabled state and custom configuration.

For custom DexSim settings, enable or reconfigure the controller explicitly:

```python
import dexsim

gizmo_config = dexsim.interaction.EntityGizmoConfig()
gizmo_config.max_gizmos = 0  # Unlimited simultaneous bindings.
sim.open_window()
sim.enable_entity_gizmo(gizmo_config)
```

While enabled, left-click a render mesh, dynamic/kinematic rigid body, or
articulation link and press **G** to attach or detach its root gizmo. The
controller supports multiple simultaneous bindings and owns selection,
temporary physics-state changes, and cleanup. No `sim.update_gizmos()` call is
needed for this world-level controller.

EmbodiChain's built-in `default_plane` is registered as an immovable target and
cannot receive an entity gizmo. Other supported scene entities remain
selectable normally.

`sim.enable_entity_gizmo(config)` is a thin helper that also excludes
EmbodiChain's render-only default plane. Query the controller through DexSim's
world object; use the manager's disable helper to preserve your preference
across future window opens:

```python
controller = sim.get_world().get_entity_gizmo()
sim.disable_entity_gizmo()
```

Robot TCP IK controls are registered automatically for parts with configured
IK chain/TCP metadata. The first **I** press activates them; later presses show
or hide their targets. **G** continues to control entity roots. Normal
`sim.update()` calls handle IK updates; no controller-specific call is needed.

The entity gizmo is native-window only. The Viser backend offers an analogous
**click-to-pick** flow (an *Enable click-to-pick Gizmo* checkbox instead of the
**G** hotkey, since browsers do not expose keyboard events); see
:doc:`tutorial/gizmo` for details.

## Customizing Window Events

Users can create their own custom window interaction controls by subclassing the `ObjectManipulator` class (provided by `dexsim`). This allows for the implementation of specific behaviors and responses to user inputs.

Here's an example of how to create a custom window event that responds to key presses:

```python
from dexsim.engine import ObjectManipulator
from dexsim.types import InputKey

class CustomWindowEvent(ObjectManipulator):
    def on_key_down(self, key):
        if key == InputKey.SPACE.value:
            print("Space key pressed!")


# Assuming you already have a SimulationManager instance called `sim_manager`
# (for example, created elsewhere in your code):
# sim_manager = SimulationManager(...)

# Register the custom window event handler with the simulation:
sim_manager.add_custom_window_control([CustomWindowEvent()])
```

The functions table below summarizes the key methods available in the `ObjectManipulator` class for customizing window events:

| Method               | Description                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|
| `on_key_down(key)`   | Triggered when a key is pressed down. The `key` parameter indicates which key was pressed. |
| `on_key_up(key)`     | Triggered when a key is released. The `key` parameter indicates which key was released. |
| `on_mouse_moved(x, y)`| Triggered when the mouse is moved. The `x` and `y` parameters indicate the new mouse position. |
| `on_mouse_down(button, x, y)` | Triggered when a mouse button is pressed. The `button` parameter indicates which button was pressed, and `x`, `y` indicate the mouse position. |
| `on_mouse_up(button, x, y)`   | Triggered when a mouse button is released. The `button` parameter indicates which button was released, and `x`, `y` indicate the mouse position. |
| `on_mouse_wheel(delta)` | Triggered when the mouse wheel is scrolled. The `delta` parameter indicates the amount of scroll. |
| `enable_selection_cache(enable)` | When enabled, caches the last raycast selection so `selected_object`, `selected_position`, and `selected_distance` are available in callbacks. |
