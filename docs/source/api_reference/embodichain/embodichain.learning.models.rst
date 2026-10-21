embodichain.learning.models
===========================

.. currentmodule:: embodichain.learning.models

General neural networks and feature normalization
-------------------------------------------------

These components support supervised learning and RL without importing RL,
TensorDict or simulation. ``build_mlp_from_cfg`` constructs an MLP from a network
configuration; ``EmpiricalNormalizer`` stores feature moments as module buffers.
RL policy interfaces and construction belong to
:doc:`embodichain.learning.rl.policies`.

.. autosummary::

   MLP
   EmpiricalNormalizer
   build_mlp_from_cfg

.. automodule:: embodichain.learning.models
   :members:
   :undoc-members:
   :show-inheritance:

MLP and configuration builder
-----------------------------

.. automodule:: embodichain.learning.models.mlp
   :members:

Feature normalization
---------------------

.. automodule:: embodichain.learning.models.normalizer
   :members:
