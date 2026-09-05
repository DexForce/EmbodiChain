embodichain.lab.sim
=====================

.. automodule:: embodichain.lab.sim

Overview
--------

The ``sim`` package is EmbodiChain's simulation core. It is organized around
the :class:`SimulationManager` (the DexSim scene handle), the scene-object
hierarchy (lights, rigid/soft/cloth bodies, articulations, robots, gizmos,
constraints), the sensor suite (cameras, stereo cameras, contact sensors), IK
solvers and motion planners, the atomic-action motion-primitive layer, a
reusable workspace-analysis and sampling toolkit, and the shared configuration
types and utilities that wire all of these together.

.. rubric:: Submodules

.. autosummary::
   :toctree: .

   sim_manager
   profiler
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
   workspace
   types
   utility

.. currentmodule:: embodichain.lab.sim

Simulation Manager
------------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.sim_manager

Profiler
--------

.. autoclass:: Profiler
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ProfilerCfg
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

Robot Workspace
---------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.workspace

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

DLSS Configuration
------------------

.. currentmodule:: embodichain.lab.sim

Configure window and offscreen Ray Reconstruction and Super Resolution through
``SimulationManagerCfg.render_cfg.dlss``. Output resolution remains owned by the
window or camera configuration.

.. autoclass:: DLSSCfg
   :members:
   :undoc-members:
