RigidFormer geometry dynamics
=============================

The lazy optional adapter binds the pinned RigidFormer implementation to causal
point-cloud windows and shared supervised updates. Reference geometry and point
correspondence are explicit. The model has no action input and does not qualify
action gradients, APG or a native physics-solver interface. Install the
``rigidformer`` extra for this family. The Box2D comparison belongs to the
``learning/rigidformer`` example and retains its frozen protocol and provenance.

Public entry points
-------------------

.. currentmodule:: embodichain_models.dynamics.rigidformer

.. autosummary::

   RIGIDFORMER_REVISION
   RigidFormerCfg
   RigidFormerModel
   RigidBodySequenceSource
   RigidFormerObjective
   rigidformer_batch
   build_rigidformer_trainer

.. autodata:: RIGIDFORMER_REVISION

.. autoclass:: RigidFormerCfg

.. autoclass:: RigidFormerModel

.. autoclass:: RigidBodySequenceSource

.. autoclass:: RigidFormerObjective

.. autofunction:: rigidformer_batch

.. autofunction:: build_rigidformer_trainer

Aligned geometry windows
------------------------

The source owns a bounded CPU corpus and returns independent previous, current,
reference and next-frame point arrays without crossing episode boundaries.
The caller owns sampling RNG; material point identities and finite object/point
lengths remain explicit.

.. currentmodule:: embodichain_models.dynamics.rigidformer.data

.. autosummary::

   RigidBodySequenceSource

.. autoclass:: RigidBodySequenceSource
   :members: get_batch, sample_batch

Optional network adapter
------------------------

The optional upstream model loads at construction. Parameter keys and numerical
behavior follow the pinned upstream revision. The per-call budget bounds the
dense nearest-neighbour distance intermediate. Unequal point lengths require a
padded point axis of at most 128 for forward and rollout.

.. currentmodule:: embodichain_models.dynamics.rigidformer.model

.. autosummary::

   RIGIDFORMER_REVISION
   RigidFormerCfg
   RigidFormerModel

.. autodata:: RIGIDFORMER_REVISION

.. autoclass:: RigidFormerCfg
   :members:

.. autoclass:: RigidFormerModel
   :members: forward, rollout

Objective and causal batch conversion
-------------------------------------

The objective preserves the upstream position/acceleration losses. Effective
weight is the sum of valid objects, including when scenes have different object
counts. Targets remain separate from causal model inputs.

.. currentmodule:: embodichain_models.dynamics.rigidformer.supervised

.. autosummary::

   RigidFormerObjective
   rigidformer_batch

.. autoclass:: RigidFormerObjective
   :members: compute_loss

.. autofunction:: rigidformer_batch

Installable supervised recipe
-----------------------------

Use ``factory:
embodichain_models.dynamics.rigidformer.sl_recipe:build_rigidformer_trainer``
with ``embodichain train-sl``. The recipe samples fixed tensor-only archives and
returns the common trainer with source/content/configuration recovery identity.

.. currentmodule:: embodichain_models.dynamics.rigidformer.sl_recipe

.. autosummary::

   build_rigidformer_trainer

.. autofunction:: build_rigidformer_trainer
