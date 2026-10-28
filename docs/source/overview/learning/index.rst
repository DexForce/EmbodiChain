Learning Infrastructure
=======================

Embodied learning trains agents to act through a robot's observations and
actions. This section covers policy training and evaluation in EmbodiChain;
the built-in training framework currently focuses on reinforcement learning.
Standalone differentiable kinematics layers also support task-parameter
optimization and training without a simulation environment.

.. toctree::
   :maxdepth: 1

   /overview/rl/index
   differentiable_ik

Related Workflows
-----------------

Use the :doc:`/overview/gym/index` to define task observations, actions, and
rewards. For demonstration-based learning workflows, see
:doc:`/tutorial/data_expansion` and :doc:`/features/online_data` for collecting
and supplying training data. The optional :doc:`/guides/rlinf` integration
connects EmbodiChain environments to an external training backend.
