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

``learning.models`` owns MLP, policy interfaces, actor-critic networks,
observation normalization and the policy registry/builders. Import these APIs
from ``embodichain.learning.models``. Exports load on demand: using MLP or the
standalone IK layers does not import RL trainers, simulation or TensorDict.

.. toctree::
   :maxdepth: 1

   embodichain.learning.models
