.. _tutorial_gizmo_robot:

Interactive Robot Control with Gizmo
=====================================

.. currentmodule:: embodichain.lab.sim

This tutorial demonstrates how to use the Gizmo class for interactive robot manipulation in SimulationManager. Robot gizmos delegate interactive inverse kinematics (IK) to dexsim's Newton IK controller while all joint-state reads and drive-target writes continue to pass through the EmbodiChain ``Robot`` abstraction.

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



**Important:** The target-specific Robot TCP, rigid-object, and Camera
``Gizmo`` wrapper supports only single-environment mode (``num_envs=1``).

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

First, configure a UR10 robot and its controllable arm joints:

.. literalinclude:: ../../../scripts/tutorials/sim/gizmo_robot.py
   :language: python
   :start-at: # Create UR10 robot
   :end-at: robot = sim.add_robot(cfg=robot_cfg)

Key components of the robot configuration:

- **URDF Configuration**: Loads the robot's kinematic and visual model
- **Control Parts**: Defines which joints can be controlled (``"Joint[1-6]"`` for UR10)
- **Drive Properties**: Sets stiffness and damping for joint control

An EmbodiChain kinematics solver is not required by the gizmo. The IK chain is declared when enabling it:

.. code-block:: python

   gizmo_cfg = GizmoCfg(
       ik_root_link_name="base_link",
       ik_end_link_name="ee_link",
   )
   sim.enable_gizmo(
       uid="ur10_gizmo_test",
       control_part="arm",
       gizmo_cfg=gizmo_cfg,
   )

For existing robot configurations, these link names and the TCP can instead be inherited from the selected control part's configured EmbodiChain solver. The solver supplies metadata only; interactive IK is still performed by dexsim's ``NewtonChainIK``.

Creating and Attaching a Gizmo
-------------------------------



After configuring the robot, enable the gizmo for interactive control using the SimulationManager API (supports robot, rigid object, camera; key is `uid:control_part`):

.. code-block:: python

   # Enable gizmo for the robot's arm
   sim.enable_gizmo(
       uid="ur10_gizmo_test",
       control_part="arm",
       gizmo_cfg=GizmoCfg(
           ik_root_link_name="base_link",
           ik_end_link_name="ee_link",
       ),
   )
   if not sim.has_gizmo("ur10_gizmo_test", control_part="arm"):
       logger.log_error("Failed to enable gizmo!")
       return



The Gizmo instance is managed internally by SimulationManager. If you need to access it:

.. code-block:: python

   gizmo = sim.get_gizmo("ur10_gizmo_test", control_part="arm")



The Gizmo system will automatically:

1. **Detect Target Type**: Identify that the target is a robot (vs. rigid object or camera)
2. **Find End-Effector**: Locate the robot's end-effector link (``ee_link`` for UR10)
3. **Build Newton IK Chain**: Build a reduced start-link-to-end-link model from the robot URDF
4. **Bind dexsim Controller**: Attach ``IKGizmoController`` directly to the articulation adapter

How Gizmo-Robot Interaction Works
----------------------------------



The gizmo-robot interaction follows this workflow:

1. **Target Update**: Dragging the dexsim target gizmo updates the Newton IK target state
2. **Deferred Solve**: ``sim.update_gizmos()`` asks ``IKGizmoController`` to solve only when the target changed
3. **State Bridge**: The adapter reads the selected EmbodiChain control-part joints as the solve seed
4. **Drive Target**: The solved positions are written through ``Robot.set_qpos(..., target=True)`` so CPU and CUDA state paths stay synchronized
5. **Robot Motion**: Joint drives move the robot toward the target without teleporting its current state

Robot gizmos no longer create or maintain an EmbodiChain proxy cube. Camera gizmos continue to use their existing proxy path, and rigid-object gizmos continue to follow the selected object directly.

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

- Enable a target-specific gizmo: `sim.enable_gizmo(...)`
- Update: Main loop automatically calls `sim.update_gizmos()`
- Destroy/disable: `sim.disable_gizmo(...)` or `sim.destroy()` (recommended)

There is no need to manually create or destroy Gizmo instances. All resources are managed by SimulationManager.

World-Level Entity Gizmo
------------------------

For selection-based root manipulation, opening a non-headless window enables
dexsim's world-level entity gizmo by default:

.. code-block:: python

   import dexsim

   config = dexsim.interaction.EntityGizmoConfig()
   config.max_gizmos = 0
   sim.open_window(entity_gizmo_config=config)

   # Left-click an entity and press G to attach or detach a gizmo.
   # Multiple entities may remain attached.

   sim.disable_entity_gizmo()

This path supports render meshes, eligible rigid bodies, and articulation
roots. dexsim owns raycast selection, temporary body-state changes, multiple
bindings, and cleanup. It requires no ``sim.update_gizmos()`` call.

EmbodiChain's built-in ``default_plane`` is excluded from manipulation.
Selecting it and pressing **G** does not create a gizmo.

Use ``sim.open_window(enable_entity_gizmo=False)`` for a view-only window, or
set ``SimulationManagerCfg.enable_entity_gizmo_on_window_open=False`` to
change the default. Headless simulations do not create the controller.

``sim.get_entity_gizmo()`` returns the native
``EntityGizmoManipulator`` and ``sim.has_entity_gizmo()`` reports whether it is
enabled. Closing the window or destroying the simulation also disables it.

The Robot end-effector controller remains target-specific because it solves a
TCP pose rather than editing the articulation root. By default, **G** controls
the entity gizmo and **I** toggles Robot TCP IK gizmo visibility.

Available Gizmo Methods
-----------------------




If you need to access the underlying Gizmo instance (via `sim.get_gizmo`), you can use the following methods. For robot targets these methods operate on dexsim's IK target gizmo:

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

- Set valid ``ik_root_link_name`` and ``ik_end_link_name`` values, or configure an EmbodiChain solver whose chain metadata can be inherited
- Set ``ik_tcp_pose`` when the desired tool center point differs from the end-link frame
- Test joint limits and workspace boundaries



**Visualization customization:**

- Adjust gizmo appearance via Gizmo config (e.g., ``set_line_width()``; requires access to the instance via `sim.get_gizmo`)
- Adjust robot target size with ``GizmoCfg.ik_gizmo_scale``
- Enable collision for debugging if needed

Next Steps
----------

After mastering basic gizmo usage, you can explore:

- **Multi-robot Gizmos**: Attach gizmos to multiple robots simultaneously
- **Custom Gizmo Callbacks**: Implement application-specific interaction logic  
- **Gizmo with Rigid Objects**: Use gizmos for interactive object manipulation
- **Advanced IK Configuration**: Tune ``GizmoCfg.ik_iterations``, ``ik_device``, and the TCP pose

For more advanced robot control and simulation features, refer to the complete :doc:`robot` tutorial and the API documentation for :class:`objects.Gizmo`.
