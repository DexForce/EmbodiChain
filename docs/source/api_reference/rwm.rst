Robot world model
=================

``embodichain_models.dynamics.rwm`` predicts action-conditioned robot states
using frame-local increments and FP32 integration. It owns the model, state
representation, autoregressive objective, prediction state, replay and offline
recipe. Shared optimizer updates and recovery use
:doc:`embodichain/embodichain.learning.sl`.

Public entry points
-------------------

The named family lazily exports the model and its supervised integration.
``build_rwm_trainer`` retains its public symbol name at this canonical import
path and at ``embodichain_models.dynamics.rwm.sl_recipe``. The family does not
initialize simulation or import example helpers.

.. currentmodule:: embodichain_models.dynamics.rwm

.. autosummary::

   RobotWorldModelCfg
   RobotWorldModel
   frozen_world_model
   RobotWorldModelObjective
   replay_supervised_batch
   RWMPredictor
   TransitionReplay
   build_rwm_trainer

.. autoclass:: RobotWorldModelCfg

.. autoclass:: RobotWorldModel

.. autofunction:: frozen_world_model

.. autoclass:: RobotWorldModelObjective

.. autofunction:: replay_supervised_batch

.. autoclass:: RWMPredictor

.. autoclass:: TransitionReplay

.. autofunction:: build_rwm_trainer

State representation and integration
------------------------------------

Canonical states contain world position, a row-major body-to-world rotation
matrix, world linear/angular velocity and ordered joint positions/velocities.
Increments use the current body frame and right-multiplied exponential rotation
coordinates. The logarithm selects the shortest rotation; first-order derivatives
away from its pi branch cut are supported.

.. currentmodule:: embodichain_models.dynamics.rwm.robot_state

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

Model and normalization
-----------------------

The network predicts normalized state increments. Input/target statistics change
only through an explicit update from measured training transitions. Checkpoint
identity retains the fixed control period, action semantics and ordered joints.
``frozen_world_model`` freezes parameters and statistics while preserving
state/action gradients through dependent rollout and backward work.

.. currentmodule:: embodichain_models.dynamics.rwm.model

.. autosummary::

   RobotWorldModelCfg
   RobotWorldModel
   frozen_world_model

.. autoclass:: RobotWorldModelCfg
   :members:

.. autoclass:: RobotWorldModel
   :members:

.. autofunction:: frozen_world_model

Supervision and prediction lifecycle
------------------------------------

The objective builds an untruncated autoregressive graph. Invalid prefix-mask
suffixes are excluded before differentiable arithmetic, including nonfinite
padding. The predictor returns newly allocated states; selective reset preserves
unselected rows and outstanding graphs. Callers explicitly detach prediction
state when beginning a new graph.

.. currentmodule:: embodichain_models.dynamics.rwm.supervised

.. autosummary::

   RobotWorldModelObjective
   replay_supervised_batch

.. autoclass:: RobotWorldModelObjective
   :members: compute_loss

.. autofunction:: replay_supervised_batch

.. currentmodule:: embodichain_models.dynamics.rwm.predictor

.. autosummary::

   RWMPredictor

.. autoclass:: RWMPredictor
   :members:

Owned transition replay
----------------------

Append copies detached physical transitions before environment reset. Sampled
windows and oldest-first histories retain episode boundaries and survive ring
overwrite. The fixed allocation can be checked against a byte ceiling. Replay
snapshots retain their sampling RNG; acquisition and refill remain caller-owned.

.. currentmodule:: embodichain_models.dynamics.rwm.replay

.. autosummary::

   TransitionReplay

.. autoclass:: TransitionReplay
   :members:

Offline SL recipe
-----------------

Use ``embodichain_models.dynamics.rwm.sl_recipe:build_rwm_trainer`` as the factory
for ``embodichain train-sl``. Tensor corpora declare their split, action semantics,
control period and ordered joints. Normalization uses valid measured training
transitions. The resulting common SL checkpoint preserves corpus identity,
normalization, optimizer, RNG and the sampling cursor. The controlled fitting
example runs through ``embodichain run-example learning/rwm``.

.. currentmodule:: embodichain_models.dynamics.rwm.sl_recipe

.. autosummary::

   build_rwm_trainer

.. autofunction:: build_rwm_trainer
