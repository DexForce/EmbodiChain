embodichain.learning.rl.algo
============================

.. automodule:: embodichain.learning.rl.algo

Overview
--------

Algorithm registry and algorithm-construction helpers for RL training. The
on-policy algorithms :class:`PPO` and :class:`GRPO` and the differentiable
algorithms :class:`APG` and :class:`APGActorCritic` all derive from :class:`BaseAlgorithm`. Each algorithm
declares a :class:`RolloutKind` that selects the compatible trainer.
:func:`build_algo` looks up a registered algorithm by name and wires it to a
policy, while :func:`compute_gae` provides generalized advantage estimation.

   .. rubric:: Classes

   .. autosummary::

      BaseAlgorithm
      RolloutKind
      APGCfg
      APG
      APGActorCriticCfg
      APGActorCritic
      PPOCfg
      PPO
      GRPOCfg
      GRPO

   .. rubric:: Functions

   .. autosummary::

      build_algo
      get_registered_algo_names
      compute_gae
      differentiable_lambda_returns
      complete_discounted_return
      segmented_discounted_return

.. automodule:: embodichain.learning.rl.algo
   :members:
   :undoc-members:
   :show-inheritance:

APG Actor-Critic environment inputs
------------------------------------

``apg_actor_critic`` uses the same differentiable trainer as ``apg``. Its policy
exposes disjoint actor and critic parameter groups and a scalar ``get_value``.
The collector selects the configured actor/critic observation groups. An
auto-resetting environment returns a full, graph-preserving successor batch
in ``info["terminal_observation"]`` for value bootstrap at time truncations.
Hard termination suppresses bootstrap while retaining the final reward.

Optional per-environment ``continuation``, ``valid`` and ``next_valid`` tensors
represent soft continuation and task-data availability. The actor differentiates
through continuation and terminal-value inputs, with critic parameters and
statistics fixed. Critic regression detaches observations, lambda targets and
soft-survival weights. With ``update_critic_normalization=True``, the algorithm
updates the policy's critic normalizer after actor backpropagation and then
recomputes lambda targets under the new statistics. The default leaves statistics
unchanged.

``DifferentiableTrainer.update_once()`` collects and optimizes one update,
maintains counters and records metrics. Online model-learning workflows can
call it between supervised updates. ``train()`` also manages periodic evaluation
and checkpoints. The PointMass configuration is
``extensions/embodichain_tasks/configs/tasks/classic_control/point_mass/agents/apg_actor_critic.yaml``.
