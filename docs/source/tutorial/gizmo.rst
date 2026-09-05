.. _tutorial_gizmo_robot:

Interactive Robot Control with Gizmo
=====================================

.. currentmodule:: embodichain.lab.sim

This tutorial demonstrates native DexSim and browser-based Viser Gizmo control.
DexSim owns native entity and robot IK controllers; EmbodiChain keeps only the
robot control-part adapter, Viser commands, and controller lifecycle management.

For the cross-frontend capability summary, supported targets, lifecycle rules,
and security boundary, see :doc:`/features/interaction/gizmo`.

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



**Important:** Gizmo supports a single environment (``num_envs=1``). Automatic
registration is skipped for multi-environment simulations.

Robot Gizmo registration, updates, visibility, and destruction are managed by
SimulationManager:

.. code-block:: python

   # Toggle visibility for a gizmo
   sim.toggle_gizmo_visibility("ur10_gizmo_test", control_part="arm")

   # Set visibility explicitly
   sim.set_gizmo_visibility("ur10_gizmo_test", visible=False, control_part="arm")

Native interaction uses DexSim controllers. The standard Viser mode includes
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

The :class:`objects.Gizmo` class manages native robot controllers and Viser
targets. Native entity manipulation remains owned by DexSim.

Setting up Robot Configuration
------------------------------

First, we configure a UR10 robot with an IK solver for end-effector control:

.. literalinclude:: ../../../scripts/tutorials/sim/gizmo_robot.py
   :language: python
   :start-at: # Create UR10 robot
   :end-at: robot = sim.add_robot(cfg=robot_cfg)

Key components of the robot configuration:

- **URDF Configuration**: Loads the robot's kinematic and visual model
- **Control Parts**: Defines which joints can be controlled (``"Joint[1-6]"`` for UR10)
- **IK Solver**: :class:`solvers.PinkSolverCfg` supplies chain metadata and an optional solver override
- **Drive Properties**: Sets stiffness and damping for joint control

The configured EmbodiChain solver is optional: it supplies default IK-chain
metadata (root link, end link, and TCP transform). IK itself is solved by
DexSim Newton IK. Applications may instead set this metadata directly in
:class:`objects.GizmoCfg`.

Automatic Robot Controls
------------------------

With the robot configuration above, no Gizmo-specific configuration or API
call is needed. SimulationManager discovers each control part with existing
root-link and end-link metadata and uses its configured TCP transform.

- In a native window, press **I** to create and show the IK targets. Further
  presses toggle their visibility using DexSim's native controller.
- In Viser, the TCP controls are available automatically when commands are
  allowed. The solver is constructed on the first drag.
- Pure headless, read-only Viser, and multi-environment simulations skip
  automatic registration.

Opening a window or registering a control does not initialize another IK
solver or overwrite existing joint drive targets. DexSim Newton IK is the
default. To use the robot's configured solver instead:

.. code-block:: python

   from embodichain.lab.sim.objects import GizmoCfg

   sim_cfg = SimulationManagerCfg(
       robot_ik_gizmo=GizmoCfg(ik_solver="embodichain"),
   )

Set ``robot_ik_gizmo=None`` to disable automatic setup. Robots without chain
metadata can still use ``sim.enable_gizmo(...)`` with explicit
:class:`objects.GizmoCfg` link settings. Advanced callers can use
:func:`objects.create_robot_ik_gizmo_controller` and manage its updates directly;
SimulationManager will not create a duplicate native controller for that part.

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

The tutorial uses manual physics only. After setting initial joint positions
and drive targets, each iteration advances one physics step:

.. literalinclude:: ../../../scripts/tutorials/sim/gizmo_robot.py
   :language: python
   :start-at: def run_simulation(
   :end-at: sim.update(step=1)

``sim.update()`` processes native and Viser interaction, advances physics, and
publishes visualization state. No separate controller update is required.
The tutorial paces the loop using ``physics_dt`` and releases resources with
``sim.destroy()`` on Ctrl+C.

Gizmo Lifecycle Management
-------------------------

SimulationManager handles automatic robot control registration and cleanup.
Closing a native window detaches input handlers; reopening it reuses existing
controllers and preserves their visibility without writing new drive targets.
Removing a robot also removes its managed controls.

For explicit overrides:

- ``sim.enable_gizmo(uid, control_part, gizmo_cfg)`` replaces that part's settings.
- ``sim.disable_gizmo(uid, control_part)`` disables one part and prevents automatic
  recreation; omitting the part disables every part of the robot.
- ``sim.toggle_gizmo_visibility(uid, control_part)`` and
  ``sim.set_gizmo_visibility(uid, visible, control_part)`` control visibility.

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
- ``--renderer auto|hybrid|fast-rt|rt``: Select the renderer
- ``--viser``: Use browser-based interaction

Once running:

1. **Activate**: Press **I** in the native window, or open the Viser page
2. **Mouse Interaction**: Click and drag the gizmo to move the robot
3. **Real-time IK**: Watch the robot joints automatically adjust to follow the gizmo
4. **Workspace Limits**: Observe how the robot behaves at workspace boundaries
5. **Performance**: Monitor FPS in the console output

Tips and Best Practices
------------------------



**Performance optimization:**

- Use ``sim.update(step=1)`` to service interaction and advance manual physics
- Reduce IK solver iterations for better real-time performance if needed
- Pace manual steps using ``physics_dt``



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
