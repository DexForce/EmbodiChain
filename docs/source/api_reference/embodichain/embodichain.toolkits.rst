embodichain.toolkits
====================

The :mod:`embodichain.toolkits` package contains asset-preparation and
manipulation utilities plus an isolated application-level dynamics-calibration
workflow.

.. automodule:: embodichain.toolkits

   .. rubric:: Submodules

   .. autosummary::

      acd
      dynamics_calibration
      graspkit
      urdf_assembly

.. toctree::
   :maxdepth: 1
   :hidden:

   embodichain.toolkits.dynamics_calibration


GraspKit — Parallel-Gripper Grasp Sampling
-------------------------------------------

The :mod:`embodichain.toolkits.graspkit` package owns the standalone
grasp-pose service contracts. The toolkit does not import
:mod:`embodichain.lab`, so the same generator instance can be called directly
or installed in a higher-level planning runtime.

.. currentmodule:: embodichain.toolkits.graspkit

.. autosummary::
   :nosignatures:

   GraspPoseGenerator
   ParallelJawGraspPoseGenerator
   ParallelJawGripperModelCfg
   get_parallel_jaw_gripper_model

.. autoclass:: GraspPoseGenerator
   :members:

.. autoclass:: ParallelJawGraspPoseGenerator
   :members:

.. autoclass:: ParallelJawGripperModelCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

.. autofunction:: get_parallel_jaw_gripper_model

.. currentmodule:: embodichain.toolkits.graspkit.pose_generator

.. autosummary::
   :nosignatures:

   GraspPoseGenerator
   ParallelJawGraspPoseGenerator
   ParallelJawGripperModelCfg
   get_parallel_jaw_gripper_model

The :mod:`embodichain.toolkits.graspkit.pg_grasp` module provides a reusable
antipodal implementation of these contracts. The pipeline consists of three
stages:

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

   GraspPoseGenerator
   ParallelJawGraspPoseGenerator
   ParallelJawGripperModelCfg
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
