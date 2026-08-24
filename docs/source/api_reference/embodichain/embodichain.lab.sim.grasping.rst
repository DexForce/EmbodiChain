embodichain.lab.sim.grasping
============================

The :mod:`embodichain.lab.sim.grasping` package defines simulator-independent
contracts for grasp-pose generation. A handwritten environment can call a
generator directly; an atomic-action engine or Expert Program adapter can
install the same instance as a planning service. Target geometry remains on
the affordance, while the generator owns end-effector geometry and generation
policy.

.. automodule:: embodichain.lab.sim.grasping

.. currentmodule:: embodichain.lab.sim.grasping

.. autosummary::
   :nosignatures:

   GraspPoseGenerator
   ParallelJawGraspPoseGenerator
   ParallelJawGripperModelCfg

.. autoclass:: GraspPoseGenerator
   :members:

.. autoclass:: ParallelJawGraspPoseGenerator
   :members:

.. autoclass:: ParallelJawGripperModelCfg
   :members:
   :show-inheritance:
   :exclude-members: __init__, copy, replace, to_dict, validate

Implementation Module
---------------------

.. currentmodule:: embodichain.lab.sim.grasping.base

.. autosummary::
   :nosignatures:

   GraspPoseGenerator
   ParallelJawGraspPoseGenerator
   ParallelJawGripperModelCfg
