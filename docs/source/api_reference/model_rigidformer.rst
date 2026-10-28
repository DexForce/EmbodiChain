RigidFormer geometry dynamics
=============================

The model family implements the author's RigidFormer architecture from revision
160fe0852c51cffae7b7440452b58d0b49db6df6. Licensed implementation code is private
to the model package; public adapters do not import the research showcase or a
simulator. The optional extra installs numerical/data dependencies, without the
community rigidformer package.

Geometry uses aligned material points, explicit reference geometry, valid
object/vertex masks and physical mass/friction/restitution. The author KNN path
requires CUDA and at least two valid objects per scene. Model construction and
weight inspection can run on CPU. The geometry integration unit dt=1 is distinct
from physical intervals step_size / 240 seconds.

Attention uses explicit SDPA or an optional compatible Flash-Attention build.
Backend selection is explicit in model configuration. Inference rollout restores
the caller's module modes and does not retain a gradient graph.

Public model family
-------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer

.. autosummary::

   RIGIDFORMER_REVISION
   RigidFormerCfg
   RigidFormerModel
   load_rigidformer_checkpoint

.. autodata:: RIGIDFORMER_REVISION

.. autoclass:: RigidFormerCfg
   :members:

.. autoclass:: RigidFormerModel
   :members:

.. autofunction:: load_rigidformer_checkpoint

Architecture and geometry contract
----------------------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer.model

.. autosummary::

   RIGIDFORMER_REVISION
   RigidFormerCfg
   RigidFormerModel

.. autodata:: RIGIDFORMER_REVISION

.. autoclass:: RigidFormerCfg
   :members:

.. autoclass:: RigidFormerModel
   :members:

Official inference weights
--------------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer.checkpoint

.. autosummary::

   load_rigidformer_checkpoint

.. autofunction:: load_rigidformer_checkpoint
