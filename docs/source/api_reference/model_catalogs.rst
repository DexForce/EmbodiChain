Official model library
======================

The main ``embodichain`` distribution bundles ``embodichain_models`` with the
same release version and dependencies. This package owns concrete model
families, their physical representations, model-specific data adapters and
supervised recipes. General MLPs and feature normalization remain in
:mod:`embodichain.learning.models`; shared optimizer updates, recovery and
training orchestration remain in :doc:`embodichain/embodichain.learning.sl`.

Import a selected family explicitly. Importing the model-library root does not
load every model or initialize simulation. NeRD's raw teacher assets, data
generation and original-repository comparison runners belong to the
``learning/nerd`` showcase, outside the runtime model package.

NeRD public entry points
-------------------------

``embodichain_models.dynamics.nerd`` lazily exposes the causal network, its
physical codec, portable checkpoints and shared-SL integration. Inputs are named
history tensors; predictions are physical state-increment targets. Model
evaluation retains input gradients. Frozen training statistics travel with the
network as buffers.

.. currentmodule:: embodichain_models.dynamics.nerd

.. autosummary::

   NeRDModelCfg
   NeRDModel
   FeatureNormalizer
   export_model
   load_model
   convert_legacy_checkpoint
   NeRDStateCodec
   NeRDObjective
   make_nerd_batch
   NeRDWindowSource
   build_supervised_trainer

.. autoclass:: NeRDModelCfg

.. autoclass:: NeRDModel

.. autoclass:: FeatureNormalizer

.. autofunction:: export_model

.. autofunction:: load_model

.. autofunction:: convert_legacy_checkpoint

.. autoclass:: NeRDStateCodec

.. autoclass:: NeRDObjective

.. autofunction:: make_nerd_batch

.. autoclass:: NeRDWindowSource

.. autofunction:: build_supervised_trainer

Network and normalization
--------------------------

Ordered input widths and Transformer settings define the network schema.
``forward`` predicts every history position; ``evaluate`` selects the last valid
position for each row. Causal attention preserves valid prefix outputs under
right padding; the objective owns the padded loss mask. Feature conversion
follows the network's floating dtype without detaching input gradients.

.. currentmodule:: embodichain_models.dynamics.nerd.model

.. autosummary::

   NeRDModelCfg
   NeRDModel
   FeatureNormalizer

.. autoclass:: NeRDModelCfg

.. autoclass:: NeRDModel
   :members: forward, evaluate, set_normalization

.. autoclass:: FeatureNormalizer
   :members: forward

Portable checkpoints
---------------------

Exports contain tensor weights and JSON deployment metadata, including a weight
checksum, numerical profile and attention backend. Loading restores the saved
floating precision and returns an evaluation-mode model. Legacy Python-object
conversion requires explicit trust and accepts the supported official
Transformer architecture.

.. currentmodule:: embodichain_models.dynamics.nerd.checkpoint

.. autosummary::

   export_model
   load_model
   convert_legacy_checkpoint

.. autofunction:: export_model

.. autofunction:: load_model

.. autofunction:: convert_legacy_checkpoint

Physical state, contact and history encoding
--------------------------------------------

The Cartpole and Ant codec converts native Newton COM velocities to NeRD's
angular-first spatial twist and expresses relative targets in the current body
frame. Quaternion helpers use xyzw ordering. Contact features use fixed plane
slots; history append and reset return new tensors while retaining gradients
through unaffected values.

.. currentmodule:: embodichain_models.dynamics.nerd.codec

.. autosummary::

   NeRDStateCodec
   quat_multiply
   quat_rotate
   quat_inverse
   plane_contact_features
   append_history
   reset_history

.. autoclass:: NeRDStateCodec
   :members: native_to_legacy, legacy_to_native, encode, targets, decode

.. autofunction:: quat_multiply

.. autofunction:: quat_rotate

.. autofunction:: quat_inverse

.. autofunction:: plane_contact_features

.. autofunction:: append_history

.. autofunction:: reset_history

Committed trajectories and windows
-----------------------------------

The single-writer HDF5 format stores T+1 states and T transitions. Only
manifest-committed shards are visible to readers. Windows respect trajectory
and split boundaries; optional startup windows carry right-padding masks.
Readers bound open shard handles and offer an explicitly budgeted resident
cache. Operational generation progress is excluded from the data fingerprint.

.. currentmodule:: embodichain_models.dynamics.nerd.data

.. autosummary::

   DynamicsShardWriter
   DynamicsWindowDataset
   trajectory_split
   load_manifest
   dataset_fingerprint

.. autoclass:: DynamicsShardWriter
   :members: committed_ids, add_trajectory, commit, close

.. autoclass:: DynamicsWindowDataset
   :members: get_batch, enable_device_cache, close

.. autofunction:: trajectory_split

.. autofunction:: load_manifest

.. autofunction:: dataset_fingerprint

Encoding, statistics and evaluation
-----------------------------------

Raw windows become named feature histories, physical increment targets and a
valid-position mask. Statistics use only the training split and weight each
valid overlapping-window occurrence; artifacts bind the dataset fingerprint.
The normalized MSE uses frozen target variance. Offline evaluation reports
valid-position prediction and decoded state errors without selecting on test.

.. currentmodule:: embodichain_models.dynamics.nerd.integration

.. autosummary::

   encode_batch
   normalized_mse
   compute_statistics
   save_statistics
   load_statistics
   evaluate_model

.. autofunction:: encode_batch

.. autofunction:: normalized_mse

.. autofunction:: compute_statistics

.. autofunction:: save_statistics

.. autofunction:: load_statistics

.. autofunction:: evaluate_model

Shared supervised objective
---------------------------

The adapter preserves the raw reader's sampling contract when creating a
``SupervisedBatch``. Its objective returns a graph-connected normalized
state-increment loss and the number of valid positions used for aggregation.
Computing the loss does not update normalization statistics.

.. currentmodule:: embodichain_models.dynamics.nerd.supervised

.. autosummary::

   NeRDObjective
   make_nerd_batch

.. autoclass:: NeRDObjective
   :members: compute_loss

.. autofunction:: make_nerd_batch

Offline SL recipe
------------------

``NeRDWindowSource`` uses the trainer's explicit CPU sampling generator and
encodes copied windows on the requested device. The factory combines this
source, the physical objective, Adam scheduling and held-out evaluation with
the common SL trainer. Its checkpoints use the SL format; historical NeRD
training checkpoints remain owned by the showcase's historical trainer.

Use the installable factory path
``embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer`` in an
``embodichain train-sl`` configuration. It is available independently of the
working directory and example-runner bootstrap.

.. currentmodule:: embodichain_models.dynamics.nerd.sl_recipe

.. autosummary::

   NeRDWindowSource
   build_supervised_trainer

.. autoclass:: NeRDWindowSource
   :members: sample

.. autofunction:: build_supervised_trainer

Other dynamics families
-----------------------

.. toctree::
   :maxdepth: 2

   rwm
