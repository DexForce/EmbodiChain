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

Package Exports
---------------

Dataset lookup and cache locations are also available from the package root.
Dataset download implementation and preset registries load when requested,
so importing cache paths does not initialize those optional dependencies.

.. currentmodule:: embodichain.data

.. autosummary::

   EmbodiChainDataset
   get_data_class
   get_data_path
   database_dir
   database_2d_dir
   database_agent_prompt_dir
   database_demo_dir

.. autoclass:: EmbodiChainDataset
   :members:

.. autofunction:: get_data_class

.. autofunction:: get_data_path

.. autodata:: database_dir

.. autodata:: database_2d_dir

.. autodata:: database_agent_prompt_dir

.. autodata:: database_demo_dir

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
