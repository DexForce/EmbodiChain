embodichain.toolkits
====================

The :mod:`embodichain.toolkits` package contains asset-preparation and
manipulation utilities that can be used independently of the simulation loop.

.. automodule:: embodichain.toolkits

   .. rubric:: Submodules

   .. autosummary::

      acd
      graspkit
      urdf_assembly


GraspKit — Parallel-Gripper Grasp Sampling
-------------------------------------------

The ``embodichain.toolkits.graspkit.pg_grasp`` module provides a complete pipeline for generating antipodal grasp poses for parallel-jaw grippers. The pipeline consists of three stages:

1. **Antipodal sampling** — Surface points are uniformly sampled on the mesh and rays are cast to find antipodal point pairs on opposite sides.
2. **Pose construction** — For each antipodal pair, a 6-DoF grasp frame is built aligned with the approach direction.
3. **Filtering & ranking** — Grasp candidates that cause the gripper to collide with the object are discarded; survivors are scored by a weighted cost.

.. rubric:: Public API

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp

The main entry point is :class:`GraspGenerator`. It is configured via :class:`GraspGeneratorCfg` and :class:`GripperCollisionCfg`.

.. autosummary::
   :nosignatures:

   GraspGenerator
   GraspGeneratorCfg
   AntipodalSampler
   AntipodalSamplerCfg
   GripperCollisionChecker
   GripperCollisionCfg
   ConvexCollisionChecker
   ConvexCollisionCheckerCfg


GraspGenerator
~~~~~~~~~~~~~~~

.. autoclass:: GraspGenerator
   :members: generate, annotate, get_grasp_poses, visualize_grasp_pose
   :show-inheritance:

GraspGeneratorCfg
~~~~~~~~~~~~~~~~~~

.. autoclass:: GraspGeneratorCfg
   :members:
   :show-inheritance:

AntipodalSampler
~~~~~~~~~~~~~~~~~

.. autoclass:: AntipodalSampler
   :members: sample
   :show-inheritance:

AntipodalSamplerCfg
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AntipodalSamplerCfg
   :members:
   :show-inheritance:

GripperCollisionChecker
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GripperCollisionChecker
   :members: query
   :show-inheritance:

GripperCollisionCfg
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GripperCollisionCfg
   :members:
   :show-inheritance:

ConvexCollisionChecker
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvexCollisionChecker
   :members: query, query_batch_points
   :show-inheritance:

ConvexCollisionCheckerCfg
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvexCollisionCheckerCfg
   :members:
   :show-inheritance:


URDF Convex Decomposition
-------------------------

The :mod:`embodichain.toolkits.acd.urdf_modifider` module converts concave URDF
collision meshes into CoACD-generated convex hulls. The high-level function can
also scale the model and recompute inertial properties.

.. currentmodule:: embodichain.toolkits.acd.urdf_modifider

.. autofunction:: generate_urdf_collision_convexes


URDF Assembly
-------------

.. automodule:: embodichain.toolkits.urdf_assembly
   :members:
   :undoc-members:
   :show-inheritance:
