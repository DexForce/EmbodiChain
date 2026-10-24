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
   ResNet10Encoder

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

Trainable visual features
-------------------------

``ResNet10Encoder`` maps float32 NCHW RGB images in ``[0, 1]`` at 256 by
256 pixels to 256-dimensional features. It owns ImageNet normalization and
keeps the GroupNorm residual backbone and learned spatial pooling trainable.
Its torchvision dependency is loaded only when the visual encoder is imported.

.. automodule:: embodichain.learning.models.resnet_encoder
   :members:
