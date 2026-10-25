embodichain.learning
====================
.. automodule:: embodichain.learning

   .. rubric:: Submodules

   .. autosummary::

      layers
      models
      rl
      sl

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

``learning.models`` owns general MLPs, feature normalization and MLP configuration
builders. RL-specific ``Policy``, ``ActorCritic``, ``ActorOnly`` and policy
registration/builders remain in ``learning.rl.policies``. Shared components and
standalone IK layers do not import RL, simulation or TensorDict.
Concrete model families and their adapters are documented in the
:doc:`official model library </api_reference/model_catalogs>`.

.. toctree::
   :maxdepth: 1

   embodichain.learning.models

Supervised learning
-------------------

Policies and learned dynamics share tensor-tree batches, objective-owned losses,
optimizer updates and committed-update recovery through ``learning.sl``.

.. toctree::
   :maxdepth: 1

   embodichain.learning.sl
