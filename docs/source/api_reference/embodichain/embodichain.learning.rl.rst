embodichain.learning.rl
=======================

.. automodule:: embodichain.learning.rl

Overview
--------

The ``embodichain.learning.rl`` package contains algorithm registries, rollout
collection logic, policy/model builders, and training entry points.

   .. rubric:: Submodules

   .. autosummary::
      :toctree: .

      algo
      buffer
      collector
      models
      train
      utils

   .. rubric:: Top-level APIs

   .. autosummary::

      DifferentiableTrainer
      DifferentiableTrainerCfg
      DifferentiableRolloutSpec
      DifferentiableVecEnv
      LearningVecEnv
      ScheduledDifferentiableVecEnv
      RunningObservationNormalizer
      BatchedGradientNormStats
      build_learning_env
      clip_batched_gradient_norm
      evaluate_episodes
      get_trainer_class
      register_learning_env
      stratified_rollout_value

Algorithms
----------

.. automodule:: embodichain.learning.rl.algo
   :members:
   :undoc-members:
   :show-inheritance:

Environments
------------

.. automodule:: embodichain.learning.rl.env
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation
----------

.. automodule:: embodichain.learning.rl.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Gradient Stabilization
----------------------

.. automodule:: embodichain.learning.rl.gradients
   :members:
   :undoc-members:
   :show-inheritance:

Observation Normalization
-------------------------

.. automodule:: embodichain.learning.rl.normalization
   :members:
   :undoc-members:
   :show-inheritance:

Routing
-------

.. automodule:: embodichain.learning.rl.routing
   :members:
   :undoc-members:
   :show-inheritance:

Differentiable Trainer
----------------------

.. automodule:: embodichain.learning.rl.differentiable_trainer
   :members:
   :undoc-members:
   :show-inheritance:

Rollout Buffer
--------------

.. automodule:: embodichain.learning.rl.buffer
   :members:
   :undoc-members:
   :show-inheritance:

Collectors
----------

.. automodule:: embodichain.learning.rl.collector
   :members:
   :undoc-members:
   :show-inheritance:

Policy Models
-------------

.. automodule:: embodichain.learning.rl.models
   :members:
   :undoc-members:
   :show-inheritance:

Training
--------

.. automodule:: embodichain.learning.rl.train
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: embodichain.learning.rl.utils
   :members:
   :undoc-members:
   :show-inheritance:
