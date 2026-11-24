Official model library
======================

The main ``embodichain`` distribution bundles ``embodichain.extensions.models`` with the
same release version and dependencies. This package owns concrete model
families, their physical representations, model-specific data adapters and
supervised recipes. General MLPs and feature normalization remain in
:mod:`embodichain.learning.models`; shared optimizer updates, recovery and
training orchestration remain in :doc:`embodichain/embodichain.learning.sl`.

Import a selected family explicitly. Importing the model-library root does not
load every model or initialize simulation. NeRD's raw teacher assets, data
generation and original-repository comparison runners belong to the
``learning/nerd`` showcase, outside the runtime model package.

Neural Motion Generator
-----------------------

``embodichain.extensions.models.policy.nmg`` loads a standalone NMG ONNX export.
Its two backends share a batched FP32 observation-to-action interface. ONNX
Runtime runs inference, and the optional PyTorch backend retains first-order
observation gradients with frozen weights. The export supplies its own
normalization and observation metadata. See the :doc:`NMG model guide
</resources/models/policy/nmg>` for supported configurations, weight downloads and examples.

.. currentmodule:: embodichain.extensions.models.policy.nmg

.. autosummary::

   NeuralMotionGenerator

.. autoclass:: NeuralMotionGenerator
   :members: from_onnx, forward

NeRD public entry points
-------------------------

``embodichain.extensions.models.dynamics.nerd`` lazily exposes the causal network, its
physical codec, portable checkpoints and shared-SL integration. Inputs are named
history tensors; predictions are physical state-increment targets. Model
evaluation retains input gradients. Frozen training statistics travel with the
network as buffers.

.. currentmodule:: embodichain.extensions.models.dynamics.nerd

.. autosummary::

   NeRDModelCfg
   NeRDModel
   FeatureNormalizer
   export_model
   load_model
   convert_legacy_checkpoint
   NeRDStateCodec
   NeRDGeometry
   NeRDState
   NeRDTransition
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

.. autoclass:: NeRDGeometry

.. autoclass:: NeRDState

.. autoclass:: NeRDTransition

.. autoclass:: NeRDObjective

.. autofunction:: make_nerd_batch

.. autoclass:: NeRDWindowSource

.. autofunction:: build_supervised_trainer

Graph-connected rollout and contact geometry
--------------------------------------------

The rollout computes current-state FK and contact features before encoding,
history advancement, prediction and decoding. The geometry owns only detached
fixed parameters and receives a tensor FK callable; neither implementation
imports a simulator. State and history are explicit and treated as immutable.
Learning environments own reset and TBPTT boundaries. Offline supervised
objectives may continue using measured contact features.

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.geometry

.. autosummary::

   NeRDGeometry

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.transition

.. autosummary::

   NeRDState
   NeRDTransition

Optional native NeRD deployment
-------------------------------

The :doc:`NeRD guide </resources/models/neural_dynamics/nerd>` distinguishes measured
window prediction, online rollout, learning-env training and native inference.
Its native quickstart links model preparation with step/reset/close and records
the verified physical domain and execution environment.

The Provider maps native coordinates and controls into the same independent
transition. It owns inference history and resource validation; SDK Scene owns
publication, reset baselines and scheduling. The lightweight learning route
does not construct this solver. Support remains limited to the recorded
Cartpole/Ant physical domains, without general contact-force reports or
external wrenches.

.. currentmodule:: embodichain.extensions.solvers.physics.nerd.provider

.. autosummary::

   NeRDProvider
   NeuralRobotDynamicSolver

.. autoclass:: NeRDProvider

.. autoclass:: NeuralRobotDynamicSolver

Network and normalization
--------------------------

Ordered input widths and Transformer settings define the network schema.
``forward`` predicts every history position; ``evaluate`` selects the last valid
position for each row. Causal attention preserves valid prefix outputs under
right padding; the objective owns the padded loss mask. Feature conversion
follows the network's floating dtype without detaching input gradients.

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.model

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

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.checkpoint

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

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.codec

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

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.data

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

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.integration

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

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.supervised

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
``embodichain.extensions.models.dynamics.nerd.sl_recipe:build_supervised_trainer`` in an
``embodichain train-sl`` configuration. It is available independently of the
working directory and example-runner bootstrap.

.. currentmodule:: embodichain.extensions.models.dynamics.nerd.sl_recipe

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


WorldParticle independent transition
------------------------------------

WorldParticle predicts frozen world-space particle transitions. The model
package owns authored features/topology, network loading and coordinate
conversion. The optional native Provider belongs to the solver extension and
uses the SDK particle inference adapter. Operator/runtime qualification is
separate from discovery; main's Torch/SDK pins are unchanged.

The :doc:`WorldParticle guide </resources/models/neural_dynamics/worldparticle>` links
the complete native quickstart, standalone call, NPZ/frame contract, operator
installation/preflight and measured execution record. Frozen inference does not
establish a learning-env or functional-gradient contract.

.. autoclass:: embodichain.extensions.models.dynamics.worldparticle.WorldParticleModel
   :noindex:

.. autofunction:: embodichain.extensions.models.dynamics.worldparticle.load_model
   :noindex:

.. autoclass:: embodichain.extensions.models.dynamics.worldparticle.model.WorldParticleModel
   :noindex:

.. autofunction:: embodichain.extensions.models.dynamics.worldparticle.model.load_model
   :noindex:

.. autofunction:: embodichain.extensions.models.dynamics.worldparticle.data.load_sample

.. autoclass:: embodichain.extensions.solvers.physics.worldparticle.provider.WorldParticleProvider


LeWM visual dynamics
--------------------

The :doc:`lewm` reference covers visual dynamics, training and planning.

.. toctree::
   :hidden:

   lewm
