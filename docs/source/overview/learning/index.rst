Learning Infrastructure
=======================

Embodied learning trains agents to act through a robot's observations and
actions. This section describes the learning infrastructure and its policy
training and evaluation workflows.
Shared Torch models live in ``learning.models``; ``learning.layers`` composes
simulation-independent kinematics from ``compute.kinematics`` as reusable network
layers. For FK/IK usage, pose losses and gradient contracts, see
:doc:`differentiable_kinematics`.

.. toctree::
   :maxdepth: 1

   /overview/rl/index
   differentiable_kinematics

Related Workflows
-----------------

Use the :doc:`/overview/gym/index` to define task observations, actions, and
rewards. For demonstration-based learning workflows, see
:doc:`/tutorial/data_expansion` and :doc:`/features/online_data` for collecting
and supplying training data. The optional :doc:`/guides/rlinf` integration
connects EmbodiChain environments to an external training backend.
