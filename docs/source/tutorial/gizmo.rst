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
   :end-at: robot = sim.add_robot_v2(cfg=robot_cfg)

Key components of the robot configuration:

- **URDF Configuration**: Loads the robot's kinematic and visual model
- **Control Parts**: Defines which joints can be controlled (``"Joint[1-6]"`` for UR10)
- **IK Solver**: :class:`solvers.PinkSolverCfg` provides inverse kinematics capabilities
- **Drive Properties**: Sets stiffness and damping for joint control

The IK solver is crucial for gizmo functionality, as it enables the robot to automatically calculate joint angles needed to reach gizmo target positions.

Creating and Attaching a Gizmo
-------------------------------



After configuring the robot, enable the gizmo for interactive control using the SimulationManager API (supports robot, rigid object, camera; key is `uid:control_part`):

.. code-block:: python

   # Enable gizmo for the robot's arm
   sim.enable_gizmo(uid="ur10_gizmo_test", control_part="arm")
   if not sim.has_gizmo("ur10_gizmo_test", control_part="arm"):
       logger.log_error("Failed to enable gizmo!")
       return



The Gizmo instance is managed internally by SimulationManager. If you need to access it:

.. code-block:: python

   gizmo = sim.get_gizmo("ur10_gizmo_test", control_part="arm")



The Gizmo system will automatically:

1. **Detect Target Type**: Identify that the target is a robot (vs. rigid object or camera)
2. **Find End-Effector**: Locate the robot's end-effector link (``ee_link`` for UR10)
3. **Create Proxy Object**: Generate a small invisible cube at the end-effector position
4. **Set Up IK Callback**: Configure the gizmo to trigger IK solving when moved

How Gizmo-Robot Interaction Works
----------------------------------



The gizmo-robot interaction follows this efficient workflow:

1. **Gizmo Callback**: When the user drags the gizmo, a callback function updates the proxy object's transform
2. **Deferred IK Solving**: Instead of solving IK immediately in the callback (which causes UI lag), the target transform is stored
3. **Update Loop**: During each simulation step, ``gizmo.update()`` solves IK and applies joint commands
4. **Robot Motion**: The robot smoothly moves to follow the gizmo position

This design separates UI responsiveness from computational IK solving, ensuring smooth interaction even with complex robots.

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
               step_count += 1
               # ...performance statistics, etc...
       except KeyboardInterrupt:
           logger.log_info("\nStopping simulation...")
       finally:
           sim.destroy()  # Release all resources
           logger.log_info("Simulation terminated successfully")



Main loop highlights:

- **Gizmo update**: Only `sim.update_gizmos()` is needed, no `gizmo.update()`
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
- ``--enable_rt``: Enable ray tracing for better visuals

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
