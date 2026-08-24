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

The ``embodichain.toolkits.graspkit.pg_grasp`` module provides a reusable
antipodal implementation of the generic parallel-jaw grasp-pose service. The
pipeline consists of three stages:

1. **Antipodal sampling** — Surface points are uniformly sampled on the mesh and rays are cast to find antipodal point pairs on opposite sides.
2. **Pose construction** — For each antipodal pair, a 6-DoF grasp frame is built aligned with the approach direction.
3. **Filtering & ranking** — Grasp candidates that cause the gripper to collide with the object are discarded; survivors are scored by a weighted cost.

.. rubric:: Public API

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp

The application-facing entry point is :class:`AntipodalGraspPoseGenerator`.
Its configuration separates the physical gripper model, grasp algorithm,
collision policy, and annotation/cache policy. Mesh-specific sampling and
collision state remain private implementation details.

.. autosummary::
   :nosignatures:

   AntipodalGraspPoseGenerator
   AntipodalGraspPoseGeneratorCfg
   ParallelJawGraspCollisionCfg
   GraspAnnotationCfg
   AntipodalSampler
   AntipodalSamplerCfg
   GripperCollisionChecker
   GripperCollisionCfg
   ConvexCollisionChecker
   ConvexCollisionCheckerCfg


AntipodalGraspPoseGenerator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AntipodalGraspPoseGenerator
   :members:
   :show-inheritance:

AntipodalGraspPoseGeneratorCfg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: AntipodalGraspPoseGeneratorCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

ParallelJawGraspCollisionCfg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ParallelJawGraspCollisionCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

GraspAnnotationCfg
~~~~~~~~~~~~~~~~~~

.. autoclass:: GraspAnnotationCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

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

Implementation Module
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.pose_generator

.. autosummary::
   :nosignatures:

   AntipodalGraspPoseGenerator
   AntipodalGraspPoseGeneratorCfg
   ParallelJawGraspCollisionCfg
   GraspAnnotationCfg


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
