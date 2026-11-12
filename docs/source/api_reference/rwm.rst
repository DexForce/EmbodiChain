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

   KinematicWorldModelCfg
   KinematicWorldModel
   RobotWorldModelCfg
   RobotWorldModel
   frozen_world_model
   RobotWorldModelObjective
   replay_supervised_batch
   RWMPredictor
   TransitionReplay
   build_rwm_trainer

.. autoclass:: KinematicWorldModelCfg

.. autoclass:: KinematicWorldModel

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

   pack_robot_state
   unpack_robot_state
   rotation_exp
   rotation_log
   robot_state_features
   robot_state_increment
   integrate_robot_state
   robot_state_error

.. autofunction:: pack_robot_state

.. autofunction:: unpack_robot_state

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

Kinematic-feature model
-----------------------

``KinematicWorldModel`` combines the shared MLP and normalizer with full-body FK
features. For a 29-joint G1 it uses 245 state features, 29 executed position
setpoints and 33 recorded physical parameters. Its default three 2048-wide SiLU
layers contain 9,166,918 parameters. Input normalization uses
``sqrt(variance + epsilon**2)`` and clips at ten; output increments use the same
scale convention with epsilon 1e-4. FK, normalization and integration remain FP32.

This variant uses heading axes for displacement and linear-velocity changes.
Angular-velocity increments compare each state's own body axes. Root rotation
uses the same right-multiplied exponential coordinates as the body-frame model.
Both models accept the canonical matrix-based state and preserve state/action
input gradients when frozen. Geometry XML and the complete model configuration
are included in checkpoint identity.

.. currentmodule:: embodichain_models.dynamics.rwm.kinematic

.. autosummary::

   KinematicWorldModelCfg
   KinematicWorldModel
   heading_state_increment
   integrate_heading_state

.. autoclass:: KinematicWorldModelCfg
   :members:

.. autoclass:: KinematicWorldModel
   :members:

.. autofunction:: heading_state_increment

.. autofunction:: integrate_heading_state

Supervision and prediction lifecycle
------------------------------------

The objective builds an untruncated autoregressive graph. Invalid prefix-mask
suffixes are excluded before differentiable arithmetic, including nonfinite
padding. The predictor returns newly allocated states; selective reset preserves
unselected rows and outstanding graphs. Callers explicitly detach prediction
state when beginning a new graph. The objective accepts feature weights and
``temporal_reduction="sum"`` for a time-summed window loss. With
``error_floor=None`` it uses the model's increment scale directly. Complete
window adaptation excludes episode-crossing windows and optional false
``usable`` transition flags before constructing the prediction graph.

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
-----------------------

Append copies detached physical transitions before environment reset. Sampled
windows and oldest-first histories retain episode boundaries and survive ring
overwrite. The fixed allocation can be checked against a byte ceiling. Replay
snapshots retain their sampling RNG; acquisition and refill remain caller-owned.
Declared auxiliary scalar/vector fields retain reference progress, policy controls
or transition usability with the same ownership and byte budget. Start samples
include ``history_complete``: reset padding is complete, while missing rows lost
to ring overwrite are not. Version-1 snapshots remain loadable for layouts with
no auxiliary fields.

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

Motion policy and model environment
------------------------------------

The motion actor concatenates proprioceptive history with an encoded reference
goal. Its scalar critic consumes separate privileged state features. Actor and
critic normalizers are updated explicitly at their respective learning phases.

.. automodule:: embodichain_models.policies.motion_tracking
   :members:

The tensor-only motion environment starts from measured replay states, clean
history and reference progress. It preserves action/state derivatives across a
fixed horizon and reports invalid clip suffixes through validity masks. The
workflow freezes model parameters until policy backpropagation completes.

.. automodule:: embodichain_tasks.locomotion.tracking.motion_model
   :members:

.. automodule:: embodichain_tasks.locomotion.managers.events
   :members:
