embodichain.utils
=================

.. automodule:: embodichain.utils

Overview
--------

Shared utilities used across EmbodiChain: the ``@configclass`` decorator and
the ``CfgNode`` configuration system, logging, math/tensor helpers,
file/string/device/image utilities, non-maximum suppression, a visualizer
helper, and the high-performance ``warp`` kernels for kinematics, collision,
and image processing.

.. currentmodule:: embodichain.utils

.. autodata:: GLOBAL_SEED

.. autofunction:: set_seed

.. autofunction:: is_configclass

.. autofunction:: resolve_config_path

   .. rubric:: Submodules

   .. autosummary::

      warp
      cfg
      configclass
      config_paths
      device_utils
      file
      img_utils
      logger
      math
      module_utils
      nms
      string
      utility
      visualizer

High Performance Computing with Warp
------------------------------------

.. toctree::
   :maxdepth: 1

   embodichain.utils.warp

Configuration Classes
---------------------

.. automodule:: embodichain.utils.configclass
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Paths
-------------------

.. automodule:: embodichain.utils.config_paths
   :members:

Configuration Nodes
-------------------

.. automodule:: embodichain.utils.cfg
   :members:
   :undoc-members:
   :show-inheritance:

File Operations
---------------

.. automodule:: embodichain.utils.file
   :members:
   :undoc-members:
   :show-inheritance:

Logging
-------

.. automodule:: embodichain.utils.logger
   :members:
   :undoc-members:
   :show-inheritance:

Mathematical Operations
-----------------------

.. automodule:: embodichain.utils.math
   :members:
   :undoc-members:
   :show-inheritance:

Module Utilities
----------------

.. automodule:: embodichain.utils.module_utils
   :members:
   :undoc-members:
   :show-inheritance:

String Operations
-----------------

.. automodule:: embodichain.utils.string
   :members:
   :undoc-members:
   :show-inheritance:

General Utilities
-----------------

.. automodule:: embodichain.utils.utility
   :members:
   :undoc-members:
   :show-inheritance:

Device Utilities
----------------

.. automodule:: embodichain.utils.device_utils
   :members:
   :undoc-members:
   :show-inheritance:

Image Utilities
---------------

.. automodule:: embodichain.utils.img_utils
   :members:
   :undoc-members:
   :show-inheritance:

Non-Maximum Suppression
-----------------------

.. automodule:: embodichain.utils.nms
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
-------------

.. automodule:: embodichain.utils.visualizer
   :members:
   :undoc-members:
   :show-inheritance:
