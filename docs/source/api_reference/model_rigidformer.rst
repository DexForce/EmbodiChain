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
Backend identity is part of recipe/recovery semantics. Inference rollout restores
the caller's module modes and does not retain a gradient graph.

Public model family
-------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer

.. autosummary::

   RigidFormerCfg
   RigidFormerModel
   load_rigidformer_checkpoint
   export_rigidformer_checkpoint
   MOViSequenceSource
   RigidFormerObjective
   rigidformer_batch
   build_rigidformer_trainer

.. autodata:: RIGIDFORMER_REVISION

.. autoclass:: RigidFormerCfg
   :members:

.. autoclass:: RigidFormerModel
   :members:

.. autofunction:: load_rigidformer_checkpoint

.. autofunction:: export_rigidformer_checkpoint

.. autoclass:: MOViSequenceSource
   :members:

.. autoclass:: RigidFormerObjective
   :members:

.. autofunction:: rigidformer_batch

.. autofunction:: build_rigidformer_trainer

Architecture and geometry contract
----------------------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer.model

.. autosummary::

   RigidFormerCfg
   RigidFormerModel

.. autodata:: RIGIDFORMER_REVISION

.. autoclass:: RigidFormerCfg
   :members:

.. autoclass:: RigidFormerModel
   :members:

Official weights and native SL export
-------------------------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer.checkpoint

.. autosummary::

   load_rigidformer_checkpoint
   export_rigidformer_checkpoint

.. autofunction:: load_rigidformer_checkpoint

.. autofunction:: export_rigidformer_checkpoint

Prepared packed scenes
----------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer.data

.. autosummary::

   MOViSequenceSource

.. autoclass:: MOViSequenceSource
   :members:

Shifted supervision and anchor losses
-------------------------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer.supervised

.. autosummary::

   RigidFormerObjective
   rigidformer_batch

.. autoclass:: RigidFormerObjective
   :members:

.. autofunction:: rigidformer_batch

Recoverable EMA state
---------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer.ema

.. autosummary::

   RigidFormerEMA

.. autoclass:: RigidFormerEMA
   :members:

Shared SL training recipe
-------------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer.sl_recipe

.. autosummary::

   build_rigidformer_trainer

.. autofunction:: build_rigidformer_trainer

Recovery and experiment boundaries
----------------------------------

The source is explicitly registered as a training-state participant so its
epoch/cursor commits after optimizer success. Prepared files are content-checked;
source identity also binds splits, scaling, sampling and seed. EMA shadows remain
FP32 and retain the author's warmup policy. Official weight import is distinct
from native optimizer recovery; export selects EMA or raw weights explicitly.

Applications own acquisition and independent reference experiments. Short
numerical comparisons do not establish published accuracy.
