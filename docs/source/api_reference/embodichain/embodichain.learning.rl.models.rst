embodichain.learning.rl.models
==============================

.. automodule:: embodichain.learning.rl.models

Overview
--------

Policy-network registration and model construction APIs for RL agents. Policies
implement the :class:`Policy` ABC; the built-in actor-critic variants are
:class:`ActorCritic` and :class:`ActorOnly`, both built on the :class:`MLP`
backbone. :func:`build_policy` constructs a policy from a config block, with
:class:`~embodichain.learning.rl.utils.config.AlgorithmCfg`-style registration
through :func:`register_policy` / :func:`get_policy_class`.

   .. rubric:: Classes

   .. autosummary::

      Policy
      ActorCritic
      ActorOnly
      MLP
      EmpiricalNormalizer

   .. rubric:: Functions

   .. autosummary::

      build_mlp_from_cfg
      build_policy
      get_policy_class
      get_registered_policy_names
      register_policy
      resolve_policy_obs_groups

.. automodule:: embodichain.learning.rl.models
   :members:
   :undoc-members:
   :show-inheritance:
Observation normalization
-------------------------

``ActorCritic`` can maintain independent actor and critic observation moments.
The statistics are stored in policy buffers and restored with its checkpoint.

.. currentmodule:: embodichain.learning.rl.models.normalizer

.. autosummary::

   EmpiricalNormalizer

.. autoclass:: EmpiricalNormalizer
   :no-index:
