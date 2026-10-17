embodichain.learning
====================

.. automodule:: embodichain.learning

   .. rubric:: Submodules

   .. autosummary::

      layers
      models
      rl

Reinforcement Learning
----------------------

.. automodule:: embodichain.learning.rl
   :members:
   :undoc-members:
   :show-inheritance:

   .. autosummary::
      :toctree: .

      algo
      buffer
      collector
      models
      train
      utils

Differentiable layers
---------------------

``learning.layers`` composes simulation-independent computation as Torch modules.
``KinematicsLayer`` owns fixed TCP and limit buffers; its copied PK geometry follows
``to(device/dtype)``. Reconstruct the same robot model before loading a checkpoint;
its geometry fingerprint is checked. ``DifferentiableIKLayer.forward`` returns joints
for network composition, while ``solve`` returns residuals and a success mask.
Both retain input gradients in training and evaluation mode. Callers own rollout
state, optimizers, and detach boundaries.

.. currentmodule:: embodichain.learning.layers

.. autosummary::

   KinematicsLayer
   DifferentiableIKLayer

.. automodule:: embodichain.learning.layers.kinematics
   :members:

.. automodule:: embodichain.learning.layers.ik
   :members:

Shared trainable models
-----------------------

The generic MLP is shared with RL; existing ``learning.rl.models.MLP`` and
``learning.rl.models.mlp.MLP`` imports refer to the same class. Policy interfaces,
actor-critic models and policy registration remain owned by ``learning.rl.models``.
Root learning packages load on demand, so using standalone layers does not load
RL trainers, simulation, TensorDict, or logging backends.

.. currentmodule:: embodichain.learning.models

.. autosummary::

   MLP

.. automodule:: embodichain.learning.models.mlp
   :members:

.. currentmodule:: embodichain.learning.rl.models.mlp

.. autosummary::

   MLP
