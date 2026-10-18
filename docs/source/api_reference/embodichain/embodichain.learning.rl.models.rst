embodichain.learning.rl.models
=============================

.. currentmodule:: embodichain.learning.rl.models

RL policy interfaces and construction
-------------------------------------

``Policy`` defines the action, action-evaluation and value interfaces used by RL
algorithms. ``ActorCritic`` combines a Gaussian action policy with a value network;
``ActorOnly`` supplies the same rollout contract without a learned value function.
The registry builds policies from registered names and observation-group settings.
General MLP construction and feature normalization are provided separately by
:doc:`embodichain.learning.models`.

.. autosummary::

   Policy
   ActorCritic
   ActorOnly
   register_policy
   get_registered_policy_names
   get_policy_class
   build_policy
   resolve_policy_obs_groups

.. automodule:: embodichain.learning.rl.models
   :members:
   :undoc-members:
   :show-inheritance:

Policy registry
---------------

.. automodule:: embodichain.learning.rl.models.registry
   :members:

Observation normalization
-------------------------

``ActorCritic`` can maintain independent actor and critic observation moments.
It uses the shared ``EmpiricalNormalizer``; policy buffers preserve the statistics
in checkpoints. Observation selection and normalization-update timing remain
part of the RL rollout contract.
