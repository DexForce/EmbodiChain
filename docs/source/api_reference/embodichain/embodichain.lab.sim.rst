embodichain.lab.sim
=====================

.. automodule:: embodichain.lab.sim

Overview
--------

The ``sim`` package is EmbodiChain's simulation core. It is organized around
the :class:`SimulationManager` (the DexSim scene handle), the scene-object
hierarchy (lights, rigid/soft/cloth bodies, articulations, robots, gizmos,
constraints), the sensor suite (cameras, stereo cameras, contact sensors), IK
solvers and motion planners, the atomic-action motion-primitive layer, and the
shared configuration types and utilities that wire all of these together.

.. rubric:: Submodules

.. autosummary::
   :toctree: .

   sim_manager
   cfg
   common
   material
   shapes
   objects
   robots
   sensors
   solvers
   planners
   atomic_actions
   types
   utility

.. currentmodule:: embodichain.lab.sim

Simulation Manager
------------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.sim_manager

Configuration
-------------

.. automodule:: embodichain.lab.sim.cfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

Common Components
-----------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.common

Materials
---------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.material

Shapes
------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.shapes

Objects
-------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.objects

Sensors
-------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.sensors

Robot Configurations
--------------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.robots

Solvers
-------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.solvers

Planners
--------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.planners

Atomic Actions
--------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.atomic_actions
Shared Types
------------

.. automodule:: embodichain.lab.sim.types
   :members:
   :undoc-members:
   :show-inheritance:

Utility
-------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.utility