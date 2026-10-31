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
``DifferentiableFKLayer.forward`` maps joints to root-frame TCP transforms without
constructing a geometric Jacobian; ``evaluate`` returns both in one FK pass.
The layer owns fixed TCP and limit buffers, and its copied PK geometry follows
``to(device/dtype)``. Reconstruct the same robot model before loading a checkpoint;
its geometry fingerprint is checked. ``DifferentiableIKLayer`` composes an FK layer:
its ``forward`` returns joints for network composition, while ``solve`` returns
residuals and a success mask.
Both retain input gradients in training and evaluation mode. Callers own rollout
state, optimizers, and detach boundaries.

.. currentmodule:: embodichain.learning.layers

.. autosummary::

   DifferentiableFKLayer
   DifferentiableIKLayer
   DifferentiableRetimingLayer

.. automodule:: embodichain.learning.layers.fk
   :members:

.. automodule:: embodichain.learning.layers.ik
   :members:

``DifferentiableRetimingLayer`` is a stateless Torch module with numerical
hyperparameters in its constructor and positions/motion limits in ``forward``.
It returns ``(q, qd, qdd, dt, success, lengths)`` as ordinary tensors. Compute
owns fixed-width quantity padding. The layer has no parameters, limit buffers,
custom device migration or extra checkpoint state; callers reconstruct settings
and can store fixed robot limits as buffers on their own model. ``eval()`` keeps
gradients, and duration remains the tensor expression ``dt.sum(-1)``.
Only first-order derivatives on the current discrete branch are supported.

.. automodule:: embodichain.learning.layers.toppra
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
