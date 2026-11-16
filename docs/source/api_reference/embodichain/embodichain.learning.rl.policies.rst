embodichain.learning.rl.policies
================================

.. currentmodule:: embodichain.learning.rl.policies

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
   ObservationHistory
   ProprioceptiveActorCriticCfg
   ProprioceptiveActorCritic
   register_policy
   get_registered_policy_names
   get_policy_class
   build_policy
   resolve_policy_obs_groups

.. automodule:: embodichain.learning.rl.policies
   :members:
   :undoc-members:
   :show-inheritance:

Policy registry
---------------

.. automodule:: embodichain.learning.rl.policies.registry
   :members:

Observation normalization
-------------------------

``ActorCritic`` can maintain independent actor and critic observation moments.
It uses the shared ``EmpiricalNormalizer``; policy buffers preserve the statistics
in checkpoints. Observation selection and normalization-update timing remain
part of the RL rollout contract.

Temporal proprioception and training-only supervision
-----------------------------------------------------

``ProprioceptiveActorCritic`` compresses a fixed proprioceptive history with
temporal convolutions and combines it with the current frame. Explicit actor
and critic groups isolate privileged state. Contact labels are rollout-aligned
training fields; their weighted decoder loss updates the encoder without
entering the actor. ``export_actor`` omits critic and decoder. The policy uses
the existing PPO, collector and checkpoint contracts.

Composable temporal actors and training objectives
--------------------------------------------------

``ActorCritic`` accepts an ordinary actor and critic plus optional named
auxiliary objectives. A feature-producing actor evaluates its action mean and
latent together; each objective owns its predictions and loss. The policy
registers all neural heads and persistent buffers, combines state-aligned
training declarations, and exports only the deterministic actor and its input
normalizer. Ordinary shared actor/critic observation groups remain supported.

For configuration-driven construction, ``actor.type: temporal`` reads
``TemporalActorCfg`` fields from ``actor.network_cfg``. The critic retains the
ordinary MLP declaration. A contact objective can be selected as follows within
an existing training configuration; the point/class counts and coefficient are
example settings::

    policy:
      name: actor_critic
      obs_groups:
        actor: [proprioception]
        critic: [privileged]
      actor_obs_normalization: true
      critic_obs_normalization: true
      actor:
        type: temporal
        network_cfg:
          frame_dim: 33
          history_length: 30
          architecture_version: official_v1
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
        auxiliary_coef: 0.1

The environment owns feature order and history advancement. Contact labels are
integer class indices and cannot be selected as actor inputs. The ``official_v1``
layout places current first and also includes it at the end of history; it must
not be substituted for the legacy layout solely because dimensions match.

Existing ``proprioceptive_actor_critic`` configurations remain the compatibility
entry for complete historical checkpoints. Their parameter paths, module
identities, normalization state and optimizer ordering remain unchanged. An
explicitly composed policy can load equivalent actor/critic weights; full
training resume retains the original configuration identity checks.

.. automodule:: embodichain.learning.rl.policies.temporal
   :members:
   :show-inheritance:

.. automodule:: embodichain.learning.rl.policies.auxiliary
   :members:
   :show-inheritance:
