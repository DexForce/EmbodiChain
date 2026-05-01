embodichain.lab.sim
=====================

.. automodule:: embodichain.lab.sim

Overview
--------

The ``sim`` package provides simulation-core APIs including scene/object
management, materials, sensors, planning/IK utilities, and action helpers.

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

.. autoclass:: SimulationManager
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SimulationManagerCfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

Configuration
-------------

.. automodule:: embodichain.lab.sim.cfg
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

Common Components
-----------------

.. automodule:: embodichain.lab.sim.common
   :members:
   :undoc-members:
   :show-inheritance:

Materials
---------

.. automodule:: embodichain.lab.sim.material
   :members:
   :undoc-members:
   :show-inheritance:

Shapes
------

.. automodule:: embodichain.lab.sim.shapes
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

Atomic Actions
--------------

.. automodule:: embodichain.lab.sim.atom_actions
   :members:
   :undoc-members:
   :show-inheritance:

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

.. automodule:: embodichain.lab.sim.robots
   :members:
   :undoc-members:
   :show-inheritance:

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