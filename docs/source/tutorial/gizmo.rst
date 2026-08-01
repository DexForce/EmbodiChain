.. _tutorial_gizmo_robot:

Interactive Robot Control with Gizmo
=====================================

.. currentmodule:: embodichain.lab.sim

This tutorial demonstrates how to use the Gizmo class for interactive robot manipulation in SimulationManager. You'll learn how to create a gizmo attached to a robot's end-effector and use it for real-time inverse kinematics (IK) control, allowing intuitive manipulation of robot poses through visual interaction.

The Code
~~~~~~~~

The tutorial corresponds to the ``gizmo_robot.py`` script in the ``scripts/tutorials/sim`` directory.

.. dropdown:: Code for gizmo_robot.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/gizmo_robot.py
      :language: python
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~


Similar to the previous tutorial on robot simulation, we use the :class:`SimulationManager` class to set up the simulation environment. If you haven't read that tutorial yet, please refer to :doc:`robot` first.



**Important:** Gizmo only supports single environment mode (`num_envs=1`). Using multiple environments will raise an exception.

All gizmo creation, visibility, and destruction operations must be managed via the SimulationManager API:

.. code-block:: python

   # Toggle visibility for a gizmo
   sim.toggle_gizmo_visibility("ur10_gizmo_test", control_part="arm")

   # Set visibility explicitly
   sim.set_gizmo_visibility("ur10_gizmo_test", visible=False, control_part="arm")

Always use the SimulationManager API to control gizmo visibility and lifecycle. Do not operate on the Gizmo instance directly.

The same target behavior is available in either the DexSim window or Viser.
Native robot Gizmos use DexSim Newton IK, while headless Viser Gizmos use the
robot's configured EmbodiChain solver. The standard Viser mode includes
interactive Gizmo control:

.. code-block:: bash

   python scripts/tutorials/sim/gizmo_robot.py --viser

Only expose the Viser endpoint to trusted browser clients because dragging a
Gizmo mutates simulation targets.

What is a Gizmo?
-----------------

A Gizmo is an interactive visual tool that allows users to manipulate simulation objects in real-time through mouse interactions. In robotics applications, gizmos are particularly useful for:

- **Interactive Robot Control**: Drag the robot's end-effector to desired positions
- **Inverse Kinematics**: Automatically solve joint angles to reach target poses
- **Real-time Manipulation**: Provide immediate visual feedback during robot motion planning
- **Debugging and Visualization**: Test robot reachability and workspace limits

The :class:`objects.Gizmo` class provides a unified interface for interactive control of different simulation elements including robots, rigid objects, and cameras.

Setting up Robot Configuration
------------------------------

First, we configure a UR10 robot with an IK solver for end-effector control:

.. literalinclude:: ../../../scripts/tutorials/sim/gizmo_robot.py
   :language: python
   :start-at: # Create UR10 robot configuration
   :end-at: robot = sim.add_robot(cfg=robot_cfg)

Key components of the robot configuration:

- **URDF Configuration**: Loads the robot's kinematic and visual model
- **Control Parts**: Defines which joints can be controlled (``"Joint[1-6]"`` for UR10)
- **IK Solver**: :class:`solvers.PinkSolverCfg` provides inverse kinematics capabilities
- **Drive Properties**: Sets stiffness and damping for joint control

The configured solver drives Viser Gizmos and also provides default chain
metadata to the native controller. A native-only application may instead set
the root link, end link, and optional TCP transform directly in
:class:`objects.GizmoCfg` without configuring an EmbodiChain solver.

Creating and Attaching a Gizmo
-------------------------------



After configuring the robot, enable the gizmo for interactive control using the SimulationManager API (supports robot, rigid object, camera; key is `uid:control_part`):

.. code-block:: python

   from embodichain.lab.sim.objects import GizmoCfg

   # Enable gizmo for the robot's arm
   sim.enable_gizmo(
       uid="ur10_gizmo_test",
       control_part="arm",
       gizmo_cfg=GizmoCfg(
           ik_root_link_name="base_link",
           ik_end_link_name="ee_link",
       ),
       enable_native=native_window_opened,
   )
   if not sim.has_gizmo("ur10_gizmo_test", control_part="arm"):
       logger.log_error("Failed to enable gizmo!")
       return



The Gizmo instance is managed internally by SimulationManager. If you need to access it:

.. code-block:: python

   gizmo = sim.get_gizmo("ur10_gizmo_test", control_part="arm")



The Gizmo system will automatically:

1. **Detect Target Type**: Identify that the target is a robot (vs. rigid object or camera)
2. **Resolve the IK Chain**: Locate the root and end-effector links
3. **Select the Backend**: Build a native DexSim Newton controller or a headless Viser command path
4. **Defer Simulation Writes**: Apply IK drive targets from the simulation update loop

How Gizmo-Robot Interaction Works
----------------------------------



The gizmo-robot interaction follows this workflow:

1. **Target Update**: DexSim or Viser records the requested TCP transform
2. **Deferred Solve**: ``sim.update_gizmos()`` invokes the selected IK backend only when needed
3. **State Bridge**: Native DexSim IK reads and writes the selected EmbodiChain control-part joints through an adapter
4. **Drive Target**: Native solutions use ``Robot.set_qpos(..., target=True)``; Viser solutions use the configured EmbodiChain solver
5. **Robot Motion**: Joint drives move the robot toward the target without teleporting its current state

Native robot Gizmos do not create an EmbodiChain proxy cube. Camera Gizmos
retain their proxy path, while rigid-object Gizmos follow their selected object
directly.

The Simulation Loop
-------------------



In the main loop, simply call `sim.update_gizmos()`. There is no need to manually update any Gizmo instance.



.. code-block:: python

   def run_simulation(sim: SimulationManager):
       step_count = 0
       try:
           last_time = time.time()
           last_step = 0
           while True:
               time.sleep(0.033)  # 30Hz
               sim.update_gizmos()  # Update all gizmos
               sim.capture_visualization_safely()  # Publish Viser state, if enabled
               step_count += 1
               # ...performance statistics, etc...
       except KeyboardInterrupt:
           logger.log_info("\nStopping simulation...")
       finally:
           sim.destroy()  # Release all resources
           logger.log_info("Simulation terminated successfully")



Main loop highlights:

- **Gizmo update**: Only `sim.update_gizmos()` is needed, no `gizmo.update()`
- **Viser update**: Automatic-physics loops also call `sim.capture_visualization_safely()`
- **Performance monitoring**: Optional FPS statistics
- **Resource cleanup**: Only `sim.destroy()` is needed, no manual Gizmo destruction
- **Graceful shutdown**: Supports Ctrl+C interruption

Gizmo Lifecycle Management
--------------------------




Gizmo lifecycle is managed by SimulationManager:

- Enable: `sim.enable_gizmo(...)`
- Update: Main loop automatically calls `sim.update_gizmos()`
- Destroy/disable: `sim.disable_gizmo(...)` or `sim.destroy()` (recommended)

There is no need to manually create or destroy Gizmo instances. All resources are managed by SimulationManager.

Available Gizmo Methods
-----------------------




If you need to access the underlying Gizmo instance (via `sim.get_gizmo`), you can use the following methods:

**Transform Control:**

- ``set_world_pose(pose)``: Set gizmo world position and orientation
- ``get_world_pose()``: Get current gizmo world transform
- ``set_local_pose(pose)``: Set gizmo local transform relative to parent
- ``get_local_pose()``: Get gizmo local transform



**Visual properties (strongly recommend using SimulationManager API):**

- ``sim.toggle_gizmo_visibility(uid, control_part=None)``: Toggle gizmo visibility
- ``sim.set_gizmo_visibility(uid, visible, control_part=None)``: Set gizmo visibility

**Hierarchy Management:**

- ``get_parent()``: Get gizmo's parent node in scene hierarchy
- ``get_name()``: Get gizmo node name for debugging
- ``detach()``: Disconnect gizmo from current target
- ``attach(target)``: Attach gizmo to a new simulation object

Running the Tutorial
--------------------

To run the gizmo robot tutorial:

.. code-block:: bash

   cd scripts/tutorials/sim
   python gizmo_robot.py --device cpu

Command-line options:

- ``--device cpu|cuda``: Choose simulation device
- ``--num_envs N``: Number of parallel environments
- ``--headless``: Run without GUI for automated testing
- ``--renderer``: Enable ray tracing for better visuals

Once running:

1. **Mouse Interaction**: Click and drag the gizmo (colorful axes) to move the robot
2. **Real-time IK**: Watch the robot joints automatically adjust to follow the gizmo
3. **Workspace Limits**: Observe how the robot behaves at workspace boundaries
4. **Performance**: Monitor FPS in the console output

Tips and Best Practices
------------------------



**Performance optimization:**

- Only call ``sim.update_gizmos()`` in the main loop, no need for ``gizmo.update()``
- Reduce IK solver iterations for better real-time performance if needed
- Use ``set_manual_update(False)`` for smoother interaction



**Debugging tips:**

- Check console output for IK solver success/failure messages
- Use ``get_world_pose()`` to check gizmo position (if needed)
- Monitor FPS to identify performance bottlenecks



**Robot compatibility:**

- Ensure your robot is configured with a correct IK solver
- Check the end-effector (EE) link name
- Test joint limits and workspace boundaries



**Visualization customization:**

- Adjust gizmo appearance via Gizmo config (e.g., ``set_line_width()``; requires access to the instance via `sim.get_gizmo`)
- Adjust gizmo scale according to robot size
- Enable collision for debugging if needed

Next Steps
----------

After mastering basic gizmo usage, you can explore:

- **Multi-robot Gizmos**: Attach gizmos to multiple robots simultaneously
- **Custom Gizmo Callbacks**: Implement application-specific interaction logic  
- **Gizmo with Rigid Objects**: Use gizmos for interactive object manipulation
- **Advanced IK Configuration**: Fine-tune solver parameters for specific robots

For more advanced robot control and simulation features, refer to the complete :doc:`robot` tutorial and the API documentation for :class:`objects.Gizmo` and :class:`solvers.PinkSolverCfg`.
