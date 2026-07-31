embodichain.learning.rl.algo
============================

.. automodule:: embodichain.learning.rl.algo

Overview
--------

Algorithm registry and algorithm-construction helpers for RL training. The
on-policy algorithms :class:`PPO` and :class:`GRPO` both derive from
:class:`BaseAlgorithm`; :func:`build_algo` looks up a registered algorithm by
name and wires it to a policy, while :func:`compute_gae` provides generalized
advantage estimation.

   .. rubric:: Classes

   .. autosummary::

      BaseAlgorithm
      PPOCfg
      PPO
      GRPOCfg
      GRPO

   .. rubric:: Functions

   .. autosummary::

      build_algo
      get_registered_algo_names
      compute_gae

.. automodule:: embodichain.learning.rl.algo
   :members:
   :undoc-members:
   :show-inheritance:
   