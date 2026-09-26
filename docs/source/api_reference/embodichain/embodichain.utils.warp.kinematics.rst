embodichain.utils.warp.kinematics
=================================

Compatibility aliases for kernels relocated to ``embodichain.compute``.

This subpackage provides Warp kernels and helper functions for inverse/forward
kinematics and batched trajectory warping used across EmbodiChain. The modules
documented below are the main entry points:

- ``opw_solver``: efficient OPW-based forward/inverse kinematics kernels.
- ``warp_trajectory``: kernels to compute, interpolate, and apply trajectory offsets.

.. automodule:: embodichain.utils.warp.kinematics

   .. Rubric:: Submodules

   .. autosummary::

        interpolate
        ur_solver
        opw_solver
        warp_trajectory

OPW Kinematics Solver
-----------------------

.. automodule:: embodichain.utils.warp.kinematics.opw_solver
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:


Trajectory Warping Utilities
----------------------------
.. automodule:: embodichain.utils.warp.kinematics.warp_trajectory
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:

SRS and UR Compatibility
------------------------

.. automodule:: embodichain.utils.warp.kinematics.srs_solver
   :members:
   :imported-members:

.. automodule:: embodichain.utils.warp.kinematics.ur_solver
   :members:
   :imported-members:

Path Resampling Compatibility
-----------------------------

.. automodule:: embodichain.utils.warp.kinematics.interpolate
   :members:
   :imported-members:
