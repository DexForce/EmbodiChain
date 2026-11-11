embodichain.lab.sim
=====================

.. automodule:: embodichain.lab.sim

Overview
--------

The ``sim`` package is EmbodiChain's simulation core. It is organized around
the :class:`SimulationManager` (the DexSim scene handle), the scene-object
hierarchy (lights, rigid/soft/cloth bodies, articulations, robots, gizmos,
constraints), the sensor suite (cameras, stereo cameras, contact sensors), the
``motion`` package for solvers, planners, workspace, and trajectory augmentation,
the atomic-skill motion-primitive layer, and the shared configuration types and
utilities that wire all of these together.

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
   motion
   atomic_skills
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
   :exclude-members: __init__, copy, replace, to_dict, validate, ParallelJawGripperCfg

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

Robot Motion
------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.motion

Atomic Skills
--------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.atomic_skills

Scene Services
--------------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.scene

Shared Types
------------

.. automodule:: embodichain.lab.sim.types
   :members:
   :undoc-members:
   :show-inheritance:

Authored source geometry
------------------------

Object collision-input adapters and static Scene inspections share fresh USD
reads through ``spawn.source_geometry``. ``read_usd_source`` creates an anonymous
stage and binds its source bytes without constructing a simulator. Geometry
consumers require the existing self-contained metre subset through
``require_geometry``. Metadata-only readers can first distinguish absent control
annotations from invalid declared controls. A snapshot
can be explicitly reused within one assembly; freshness checks reject changed
bytes or source selections rather than loading a new baseline silently.

``mesh_arrays`` preserves polygon inputs for stricter consumer qualification.
``link_mesh_arrays`` returns newly allocated, scaled link-local vertices and
fan-triangulated faces. ``link_mesh`` captures these as an owned ``LinkMesh``;
its buffers preserve order/dtype and cannot be made writable. These authored
inputs remain distinct from runtime ``get_link_collision_shapes`` observations.
Control roles, acceptance rules and native collision cooking stay with their
respective owners.

.. currentmodule:: embodichain.lab.sim.spawn.source_geometry

.. autosummary::

   LinkMesh
   UsdSourceSnapshot
   read_usd_source

.. autoclass:: LinkMesh
   :members:

.. autoclass:: UsdSourceSnapshot
   :members:

.. autofunction:: read_usd_source

Utility
-------

.. toctree::
   :maxdepth: 1

   embodichain.lab.sim.utility

Ray-Tracing Denoising and Reconstruction
----------------------------------------

.. currentmodule:: embodichain.lab.sim

Select independent window and offscreen image-processing paths through
``SimulationManagerCfg.render_cfg.denoising``. The public ``dlss`` path maps to
DLSS Ray Reconstruction and ``nrd`` maps to standalone NRD RELAX. DLSS and NRD
tuning remain in their dedicated nested configs; output resolution stays owned
by the window or camera configuration.

.. autodata:: DenoisingMode

.. autoclass:: DenoisingCfg
   :members:
   :undoc-members:

.. autoclass:: DLSSCfg
   :members:
   :undoc-members:

.. autoclass:: NRDCfg
   :members:
   :undoc-members:
