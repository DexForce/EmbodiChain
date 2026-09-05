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
Native IK activates on the first **I** press; Viser constructs its solver on
the first drag and requires ``visualization.allow_commands``. Registration does
not write drive targets. Set this field to ``None`` to opt out or select
``GizmoCfg(ik_solver="embodichain")`` to reuse configured solvers. Explicit
``enable_gizmo()`` settings override automatic defaults, and ``disable_gizmo()``
prevents automatic recreation.

.. rubric:: Classes

.. autosummary::

   SimulationManager
   SimulationManagerCfg

.. currentmodule:: embodichain.lab.sim.sim_manager

.. autoclass:: SimulationManager
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: visualize_point_cloud

.. rubric:: Native point-cloud visualization

.. automethod:: SimulationManager.visualize_point_cloud

.. autoclass:: SimulationManagerCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate
