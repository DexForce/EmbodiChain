RL policies
===========

.. currentmodule:: embodichain.learning.rl.policies

``ActorCritic`` accepts ordinary actor/critic networks and optional named
``AuxiliaryTask`` objectives. Ordinary MLP use needs no feature interface or
training labels. ``TemporalActor`` supplies action means and graph-connected
features in one pass; ``ContactPrediction`` consumes those features during
training. PPO's ``auxiliary_coef`` defaults to zero.

Temporal inputs have one convention: the current frame followed by chronological
history including current (``official_v1``). The observation producer owns history
updates and selective resets. Export copies only the deterministic actor and its
normalizer. The historical policy constructor, old encoder/layout and checkpoint
migration paths have been removed. Complete checkpoints require the current schema
and preserve component semantics in addition to neural tensor shapes.

Composable configuration
------------------------

.. code-block:: yaml

   policy:
     name: actor_critic
     obs_groups:
       actor: [proprioception]
       critic: [privileged]
     actor_obs_normalization: true
     actor:
       type: temporal
       network_cfg:
         frame_dim: 33
         history_length: 30
         latent_dim: 32
         architecture_version: official_v1
         hidden_sizes: [256, 256, 128]
     critic:
       type: mlp
       network_cfg:
         hidden_sizes: [256, 256, 128]
     auxiliary_tasks:
       contact:
         type: contact_prediction
         contact_points: 64
         contact_classes: 3
         observation_group: contact_labels
   algorithm:
     name: ppo
     cfg:
       auxiliary_coef: 1.0

Omit ``auxiliary_tasks`` and keep the default zero coefficient for ordinary PPO.
A configured objective with a zero coefficient still owns its head and declared
training fields. Custom input normalizers remain optional; normalization flags
control their running-statistics updates. Shared parameters appear in one gradient
clipping group.

Policy and component API
------------------------

.. autosummary::

   Policy
   ActorCritic
   ActorOnly
   TemporalActorCfg
   TemporalActor
   TemporalObservationNormalizer
   ObservationHistory
   AuxiliaryTask
   ContactPrediction
   register_policy
   get_registered_policy_names
   get_policy_class
   build_policy
   resolve_policy_obs_groups

.. autoclass:: Policy
   :members:

.. autoclass:: ActorCritic
   :members:

.. autoclass:: ActorOnly
   :members:

.. autoclass:: TemporalActorCfg
   :members:

.. autoclass:: TemporalActor
   :members:

.. autoclass:: TemporalObservationNormalizer
   :members:

.. autoclass:: ObservationHistory
   :members:

.. autoclass:: AuxiliaryTask
   :members:

.. autoclass:: ContactPrediction
   :members:

.. autofunction:: register_policy

.. autofunction:: get_registered_policy_names

.. autofunction:: get_policy_class

.. autofunction:: build_policy

.. autofunction:: resolve_policy_obs_groups

Component implementation modules
--------------------------------

The following modules own the exported components and factory functions.

.. automodule:: embodichain.learning.rl.policies.temporal
   :members:

.. automodule:: embodichain.learning.rl.policies.auxiliary
   :members:

.. automodule:: embodichain.learning.rl.policies.registry
   :members:
