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
