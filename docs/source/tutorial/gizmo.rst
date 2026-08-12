.. _tutorial_gizmo_robot:

Interactive Robot Control with Gizmo
=====================================

.. currentmodule:: embodichain.lab.sim

This tutorial demonstrates native DexSim and browser-based Viser Gizmo control.
DexSim owns native entity and robot IK controllers; EmbodiChain keeps only the
robot control-part adapter and the Viser command path.

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

Viser Gizmo creation, visibility, and destruction are managed through
SimulationManager:

.. code-block:: python

   # Toggle visibility for a gizmo
   sim.toggle_gizmo_visibility("ur10_gizmo_test", control_part="arm")

   # Set visibility explicitly
   sim.set_gizmo_visibility("ur10_gizmo_test", visible=False, control_part="arm")

Native controls use DexSim directly. The standard Viser mode includes
interactive Gizmo control:

.. code-block:: bash

   python scripts/tutorials/sim/gizmo_robot.py --viser

Only expose the Viser endpoint to trusted browser clients because dragging a
Gizmo mutates simulation targets.

Click-to-Pick in Viser
~~~~~~~~~~~~~~~~~~~~~~

Unlike the native DexSim window, the browser does not ray-cast the scene for
you, so EmbodiChain performs the click hit-test against the published scene
geometry. Enable it explicitly in the browser panel:

1. Toggle the **Enable click-to-pick Gizmo** checkbox under the **Interaction**
   folder.
2. Click a rigid object or robot link in the 3D view. A transform control is
   attached to it (replacing any previously picked Gizmo); drag it to move the
   target. Robot IK is solved with DexSim Newton IK, just as in the native
   window.
3. Click empty space, or uncheck the checkbox, to detach the picker-owned
   Gizmo.

The picker manages at most one Gizmo at a time and never touches Gizmos you
created yourself through ``sim.enable_gizmo(...)``. Only rigid objects and
robots are pickable; articulations, soft bodies, and cameras are ignored by the
picker.

What is a Gizmo?
-----------------

A Gizmo is an interactive visual tool that allows users to manipulate simulation objects in real-time through mouse interactions. In robotics applications, gizmos are particularly useful for:

- **Interactive Robot Control**: Drag the robot's end-effector to desired positions
- **Inverse Kinematics**: Automatically solve joint angles to reach target poses
- **Real-time Manipulation**: Provide immediate visual feedback during robot motion planning
- **Debugging and Visualization**: Test robot reachability and workspace limits

The :class:`objects.Gizmo` class is the Viser-side target controller for robots,
rigid objects, and cameras. Native controls are DexSim controllers.

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

The configured EmbodiChain solver is optional: it supplies default IK-chain
metadata (root link, end link, and TCP transform). IK itself is solved by
DexSim Newton IK. Applications may instead set this metadata directly in
:class:`objects.GizmoCfg`.

Creating and Attaching a Gizmo
-------------------------------



For native-window robot control, create DexSim's IK controller through the
small EmbodiChain adapter factory and retain both returned objects:

.. code-block:: python

   from embodichain.lab.sim.objects import (
       GizmoCfg,
       create_robot_ik_gizmo_controller,
   )

   ik_controller, input_controller = create_robot_ik_gizmo_controller(
       robot,
       control_part="arm",
       cfg=GizmoCfg(
           ik_root_link_name="base_link",
           ik_end_link_name="ee_link",
       ),
       world=sim.get_world(),
   )

Call ``ik_controller.update()`` once per frame. DexSim owns the native target,
hotkey, solve trigger, and visibility state. For Viser, use the SimulationManager
command path instead:

.. code-block:: python

   sim.enable_gizmo(
       "ur10_gizmo_test",
       control_part="arm",
       gizmo_cfg=GizmoCfg(
           ik_root_link_name="base_link",
           ik_end_link_name="ee_link",
       ),
   )



The Gizmo system will automatically:

1. **Resolve the IK Chain**: Locate the root and end-effector links
2. **Build Newton IK**: Construct DexSim's reduced-chain solver
3. **Bridge State**: Map one EmbodiChain control part to DexSim's joint API
4. **Own the Frontend**: DexSim owns native interaction; SimulationManager owns Viser commands

How Gizmo-Robot Interaction Works
----------------------------------



The gizmo-robot interaction follows this workflow:

1. **Target Update**: DexSim or Viser records the requested TCP transform
2. **Deferred Solve**: the native controller or ``sim.update_gizmos()`` invokes Newton IK only when needed
3. **State Bridge**: Newton IK reads and writes the selected EmbodiChain control-part joints through an adapter
4. **Drive Target**: Both paths use ``Robot.set_qpos(..., target=True)``
5. **Robot Motion**: Joint drives move the robot toward the target without teleporting its current state

The Simulation Loop
-------------------



Update the DexSim native controller explicitly, then service any Viser controls:



.. code-block:: python

   def run_simulation(sim: SimulationManager, ik_controller=None):
       step_count = 0
       try:
           last_time = time.time()
           last_step = 0
           while True:
               time.sleep(0.033)  # 30Hz
               if ik_controller is not None:
                   ik_controller.update()
               sim.update_gizmos()  # Update Viser gizmos
               sim.capture_visualization_safely()  # Publish Viser state, if enabled
               step_count += 1
               # ...performance statistics, etc...
       except KeyboardInterrupt:
           logger.log_info("\nStopping simulation...")
       finally:
           sim.destroy()  # Release all resources
           logger.log_info("Simulation terminated successfully")



Main loop highlights:

- **Native update**: Call DexSim's ``IKGizmoController.update()`` each frame
- **Viser command update**: Call ``sim.update_gizmos()``
- **Viser frame update**: Automatic-physics loops also call ``sim.capture_visualization_safely()``
- **Performance monitoring**: Optional FPS statistics
- **Resource cleanup**: Only `sim.destroy()` is needed, no manual Gizmo destruction
- **Graceful shutdown**: Supports Ctrl+C interruption

Gizmo Lifecycle Management
--------------------------




Viser Gizmo lifecycle is managed by SimulationManager:

- Enable: `sim.enable_gizmo(...)`
- Update: Call ``sim.update_gizmos()`` from the main loop
- Destroy/disable: `sim.disable_gizmo(...)` or `sim.destroy()` (recommended)

Native controller lifecycle remains in DexSim. Viser visual properties are
available through SimulationManager:

- ``sim.toggle_gizmo_visibility(uid, control_part=None)``: Toggle gizmo visibility
- ``sim.set_gizmo_visibility(uid, visible, control_part=None)``: Set gizmo visibility

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

1. **Mouse Interaction**: Click and drag the gizmo to move the robot
2. **Real-time IK**: Watch the robot joints automatically adjust to follow the gizmo
3. **Workspace Limits**: Observe how the robot behaves at workspace boundaries
4. **Performance**: Monitor FPS in the console output

Tips and Best Practices
------------------------



**Performance optimization:**

- Call the native IK controller's ``update()`` once per frame; call
  ``sim.update_gizmos()`` for Viser
- Reduce IK solver iterations for better real-time performance if needed
- Use ``set_manual_update(False)`` for smoother interaction



**Debugging tips:**

- Check console output for IK solver success/failure messages
- Inspect the robot TCP or Viser target pose when debugging alignment
- Monitor FPS to identify performance bottlenecks



**Robot compatibility:**

- Set the IK chain (root link and end-effector link) in :class:`objects.GizmoCfg`, or configure an EmbodiChain solver to supply them as defaults
- Check the end-effector (EE) link name
- Test joint limits and workspace boundaries



**Visualization customization:**

- Adjust Viser axis lengths, ring radius, and line width through :class:`objects.GizmoCfg`
- Adjust gizmo scale according to robot size
- Enable collision for debugging if needed

Next Steps
----------

After mastering basic gizmo usage, you can explore:

- **Multi-robot Gizmos**: Attach gizmos to multiple robots simultaneously
- **Gizmo with Rigid Objects**: Use gizmos for interactive object manipulation
- **Advanced IK Configuration**: Fine-tune solver parameters for specific robots

For more advanced robot control and simulation features, refer to the complete :doc:`robot` tutorial and the API documentation for :class:`objects.Gizmo` and :class:`solvers.PinkSolverCfg`.
