embodichain.learning.rl.models
==============================

.. automodule:: embodichain.learning.rl.models

Overview
--------

Policy-network registration and model construction APIs for RL agents. Policies
implement the :class:`Policy` ABC; the built-in actor-critic variants are
:class:`ActorCritic` and :class:`ActorOnly`. Backbones include :class:`MLP` and
the full-context :class:`WaypointTransformerActor` /
:class:`WaypointTransformerCritic` pair for ordered mixed-modality constraints.
:func:`build_policy` constructs a policy from a config block, with
:class:`~embodichain.learning.rl.utils.config.AlgorithmCfg`-style registration
through :func:`register_policy` / :func:`get_policy_class`.

   .. rubric:: Classes

   .. autosummary::

      Policy
      ActorCritic
      ActorOnly
      MLP
      WaypointTransformerEncoder
      WaypointTransformerActor
      WaypointTransformerCritic

   .. rubric:: Functions

   .. autosummary::

      build_model_from_cfg
      build_mlp_from_cfg
      build_policy
      get_policy_class
      get_registered_policy_names
      register_policy
      parse_waypoint_observation
      waypoint_observation_dim
      waypoint_observation_normalize_mask

.. automodule:: embodichain.learning.rl.models
   :members:
   :undoc-members:
   :show-inheritance:
