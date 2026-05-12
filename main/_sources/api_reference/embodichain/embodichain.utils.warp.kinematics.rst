embodichain.utils.warp.kinematics
=================================

Utilities for kinematics implemented with Warp (high-performance kernels).

This subpackage provides Warp kernels and helper functions for inverse/forward
kinematics and batched trajectory warping used across EmbodiChain. The modules
documented below are the main entry points:

- ``opw_solver``: efficient OPW-based forward/inverse kinematics kernels.
- ``warp_trajectory``: kernels to compute, interpolate, and apply trajectory offsets.

.. automodule:: embodichain.utils.warp.kinematics

   .. Rubric:: Submodules

   .. autosummary::

        opw_solver
        warp_trajectory

OPW Kinematics Solver
-----------------------

.. automodule:: embodichain.utils.warp.kinematics.opw_solver
   :members:
   :undoc-members:
   :show-inheritance:


Trajectory Warping Utilities
----------------------------
.. automodule:: embodichain.utils.warp.kinematics.warp_trajectory
   :members:
   :undoc-members:
   :show-inheritance:
   