# Window interaction

This section describes the default window interaction controls available in the simulation. These controls allow users to interact with the simulation environment using keyboard, mouse, and customizable input events.

The main visualization window is provided by **DexSim**. When `SimConfig.headless=False` or `SimulationManager.open_window()` is called, DexSim creates the viewer with **ORBIT** camera control by default.

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
| **Rigid-object gizmo (toggle)** | Left-click a rigid object, then press **`g`** to enable its gizmo. Press **`g`** again to disable the gizmo. |

Recording hotkey registration is controlled by `SimConfig.window_record.enable_hotkey` (enabled by default). You can also call `SimulationManager.start_window_record()`, `stop_window_record()`, or `toggle_window_record()` programmatically.

The camera-pose hotkey is controlled by `SimulationManagerCfg.window_camera_pose.enable_hotkey` and prints look-at form by default. Set `SimulationManagerCfg.window_camera_pose.convert_to_look_at=False` to print the raw 4x4 pose matrix instead. The same output can be requested programmatically with `SimulationManager.print_window_camera_pose()`.

The rigid-object gizmo hotkey is controlled by `SimulationManagerCfg.window_gizmo.enable_hotkey` and is enabled by default. Gizmo control supports one environment and dynamic or kinematic rigid objects. A dynamic object is temporarily changed to kinematic while its gizmo is active, then restored to its original body type when `g` is pressed again, the gizmo is disabled programmatically, the window closes, or the simulation is destroyed. Static objects are not modified.

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
