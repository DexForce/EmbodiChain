Robot world-model learning
==========================

Configurable robot-state dynamics integrate frame-local increments in FP32.
The supervised objective retains multi-step gradients, while an owned replay
stores detached physical transitions and keeps episode boundaries explicit.
These components do not construct policy rewards or simulator instances.

embodichain.compute.dynamics.robot_state
----------------------------------------

.. currentmodule:: embodichain.compute.dynamics.robot_state

.. autosummary::

   rotation_exp
   rotation_log
   robot_state_features
   robot_state_increment
   integrate_robot_state
   robot_state_error

.. autofunction:: rotation_exp

.. autofunction:: rotation_log

.. autofunction:: robot_state_features

.. autofunction:: robot_state_increment

.. autofunction:: integrate_robot_state

.. autofunction:: robot_state_error

embodichain.data_pipeline.datasets.transition_replay
----------------------------------------------------

.. currentmodule:: embodichain.data_pipeline.datasets.transition_replay

.. autosummary::

   TransitionReplay

.. autoclass:: TransitionReplay
   :members:

embodichain.learning.models.rwm
-------------------------------

.. currentmodule:: embodichain.learning.models.rwm

.. autosummary::

   RobotWorldModelCfg
   RobotWorldModel
   frozen_world_model

.. autoclass:: RobotWorldModelCfg
   :members:

.. autoclass:: RobotWorldModel
   :members:

.. autofunction:: frozen_world_model

embodichain.learning.sl.rwm
---------------------------

.. currentmodule:: embodichain.learning.sl.rwm

.. autosummary::

   RobotWorldModelObjective
   RWMPredictor
   replay_supervised_batch
   build_rwm_trainer

.. autoclass:: RobotWorldModelObjective
   :members:

.. autoclass:: RWMPredictor
   :members:

.. autofunction:: replay_supervised_batch

.. autofunction:: build_rwm_trainer
