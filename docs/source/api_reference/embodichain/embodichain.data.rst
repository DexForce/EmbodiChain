embodichain.data
================

.. automodule:: embodichain.data

Data Package Overview
---------------------

The ``embodichain.data`` package centralizes dataset resolution, asset download
helpers, preset asset registries, and shared constants/enums used by simulation
tasks and training pipelines.

.. rubric:: Submodules

.. autosummary::

   assets
   constants
   dataset
   download
   enum

Asset Registry
--------------

Preset configuration objects for scene assets (robots, end-effectors, objects,
materials, sensors, planners, solvers, demo scenes) ready to reference from
task configs.

.. automodule:: embodichain.data.assets
   :members:
   :undoc-members:
   :show-inheritance:

Robot Asset Datasets
--------------------

Robot dataset classes register the prefixes used by ``get_data_path()`` and
``embodichain data``. Most fetch their pinned archives on first construction.
``FlexivRizon4S`` downloads its versioned official URDF, meshes, license and
provenance archive from the configured asset source, with Hugging Face as a
fallback, into the standard revision-specific cache. Asset lookup and robot
configuration share that same data.

.. automodule:: embodichain.data.assets.robot_assets
   :members:
   :show-inheritance:

``SO101FollowerAssets`` resolves the official follower URDF, STLs and motor
reference directly from a pinned The Robot Studio revision. File SHA256 checks
and the shared dataset lock protect the complete dependency cache. The original
license and source reference files are retained alongside the meshes.

.. automodule:: embodichain.data.assets.so101_assets
   :members:
   :show-inheritance:

Constants
---------

.. automodule:: embodichain.data.constants
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Resolution
------------------

.. automodule:: embodichain.data.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Asset Download CLI
------------------

.. automodule:: embodichain.data.download
   :members:
   :undoc-members:
   :show-inheritance:

Enums
-----

.. automodule:: embodichain.data.enum
   :members:
   :undoc-members:
   :show-inheritance:
