# Window interaction

This section describes the default window interaction controls available in the simulation. These controls allow users to interact with the simulation environment using keyboard, mouse, and customizable input events.

## Default Window Events

The simulation window comes with a set of default controls that enable users to perform various actions, such as selecting objects, manipulating the camera view, and triggering specific events. These controls are implemented using the `ObjectManipulator` class (provided by `dexsim`).

| Events                        | Description                                                                                                                                      |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Raycast Information Display** | Press the right mouse button to select a point and the 'C' key to print the raycast distance and hit position of a surface (world coordinates) to the console. Useful for debugging and checking the position of objects in the simulation. |

> **Note:** We will add more interaction features in future releases. Stay tuned for updates!

## Customizing Window Events

Users can create their own custom window interaction controls by subclassing the `ObjectManipulator` class. This allows for the implementation of specific behaviors and responses to user inputs.

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
sim_manager.add_custom_window_control(CustomWindowEvent())
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
