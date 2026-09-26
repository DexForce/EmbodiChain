embodichain.lab.sim.sim_manager
=========================================

.. automodule:: embodichain.lab.sim.sim_manager

Overview
--------

:class:`SimulationManager` is the central handle around the DexSim scene. It
owns the physics world and the object/sensor registry, drives the simulation
step, owns its profiler, and exposes capture hooks and visualization
configuration. Downstream components (environments, planners, IK solvers, the
visualization runtime) look up the active manager through its class-level
instance registry instead of passing it around explicitly.

Native entity interaction defaults on when the first native window opens.
Set ``SimulationManagerCfg(enable_entity_gizmo=False)`` to opt out, or call
``sim.disable_entity_gizmo()`` at runtime. Explicit enable/disable calls and
custom DexSim controller settings survive window close/reopen. Pure headless
and Viser runs do not automatically create native gizmos.

``SimulationManagerCfg.robot_ik_gizmo`` defaults to ``GizmoCfg()`` and registers
robot control parts with configured IK-chain/TCP metadata during normal updates.
Native IK activates on the first **I** press by default; Viser constructs its solver on
the first drag and requires ``visualization.allow_commands``. Registration does
not write drive targets. Set this field to ``None`` to opt out or select
``GizmoCfg(ik_solver="embodichain")`` to reuse configured solvers. Explicit
``enable_gizmo()`` settings override automatic defaults, and ``disable_gizmo()``
prevents automatic recreation. ``GizmoCfg(ik_start_enabled=True)`` activates
native IK on the first update with an open window, as used by the robot tutorial.

Physics and visual consumption
------------------------------

State-only headless stepping does not publish Newton state to the renderer.
An open native window, camera groups, due recording and due Viser captures
request publication when they consume state. ``render_frame()`` shares that
publication within one read-only consumption phase. Gym arranges this phase
after physics and interval events automatically. Standalone callers combining
camera observations with recording or Viser can use:

.. code-block:: python

   sim.update(physics_dt, step=1, render_final_step=False)
   # Complete direct state edits before entering the read-only frame.
   with sim.render_frame():
       sim.render_camera_group(camera_group_ids)
       # Due recording and Viser captures run on context exit.

The phase never advances physics and never caches publication across frames.
Independent camera/capture calls request fresh publication. If state changes
inside a frame, call ``sync_render_state()`` explicitly before further reads.
Opening a window and rendering while paused also request publication. Native
render-thread recording consumes already-published state without invoking the
blocking physics-to-render bridge from that thread.

.. rubric:: Classes

.. autosummary::

   SimulationManager
   SimulationManagerCfg
   get_physics_scene

.. currentmodule:: embodichain.lab.sim.sim_manager

.. autoclass:: SimulationManager
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: register_contact_material_schedule, register_kinematic_joint_trajectory, register_kinematic_nodal_trajectory, register_particle_contact_material_schedule, visualize_point_cloud

.. rubric:: Newton runtime controls

Runtime controls must be registered after declaring their target assets and
before :meth:`SimulationManager.prepare`. The manager expands logical UIDs to
the concrete paths of every Arena, so callers do not need access to the private
Spawn scene. The particle-material schedule is host-side and disables CUDA
Graph replay; the other controls are graph-compatible.

.. automethod:: SimulationManager.register_kinematic_joint_trajectory

.. automethod:: SimulationManager.register_kinematic_nodal_trajectory

.. automethod:: SimulationManager.register_contact_material_schedule

.. automethod:: SimulationManager.register_particle_contact_material_schedule

.. rubric:: Native point-cloud visualization

.. automethod:: SimulationManager.visualize_point_cloud

.. autoclass:: SimulationManagerCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

Active Physics Scene
--------------------

.. autofunction:: get_physics_scene
