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

.. currentmodule:: embodichain.lab.sim.sim_manager

.. autoclass:: SimulationManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SimulationManagerCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate
