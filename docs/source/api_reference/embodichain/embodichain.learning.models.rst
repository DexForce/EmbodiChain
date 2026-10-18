embodichain.learning.models
==============================

.. currentmodule:: embodichain.learning.models

Overview
--------

Policy-network registration and model construction APIs for RL agents. Policies
implement the :class:`Policy` ABC; the built-in actor-critic variants are
:class:`ActorCritic` and :class:`ActorOnly`, both built on the :class:`MLP`
backbone. :func:`build_policy` constructs a policy from a config block.
:func:`register_policy` and :func:`get_policy_class` access the shared policy
registry.

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

.. automodule:: embodichain.learning.models
   :members:
   :undoc-members:
   :show-inheritance:

Observation normalization
-------------------------

``ActorCritic`` can maintain independent actor and critic observation moments.
The statistics are stored in policy buffers and restored with its checkpoint.

.. currentmodule:: embodichain.learning.models.normalizer

.. autosummary::

   EmpiricalNormalizer

.. autoclass:: EmpiricalNormalizer
   :no-index:

Registry and builders
---------------------

.. automodule:: embodichain.learning.models.registry
   :members:

MLP backbone
------------

.. automodule:: embodichain.learning.models.mlp
   :members:
