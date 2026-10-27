Supervised learning (SL)
========================

The SL layer composes objective-owned losses with a common optimizer update and
offline run lifecycle. Batches retain their tensor-tree structure and causal
masks; source adapters, objectives and downstream execution define their semantics.
Importing this package does not initialize simulation or reinforcement learning.

Single-node launch
------------------

Run the same recipe on one CPU or multiple workers from an installed repository
checkout. Each fresh run needs a separate, artifact-free ``trainer.output_dir``.
The example factory uses a small masked MLP regression corpus generated locally.

.. code-block:: bash

   # One CPU worker.
   python -m embodichain train-sl --config examples/learning/sl_ddp/cpu.yaml

   # Two CPU workers using Gloo; choose a fresh output directory in the recipe.
   OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
     -m embodichain train-sl --config examples/learning/sl_ddp/cpu.yaml

   # Two GPUs using NCCL, one worker per visible GPU.
   CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
     torchrun --standalone --nnodes=1 --nproc-per-node=2 \
     -m embodichain train-sl --config examples/learning/sl_ddp/cuda.yaml

For ``WORLD_SIZE > 1``, the workflow initializes the process group before invoking
the recipe factory. It resolves ``trainer.device`` to ``cpu`` or
``cuda:LOCAL_RANK``. ``trainer.batch_size`` is per worker; the global number of
drawn examples is ``batch_size * world_size``. Sampling semantics belong to each source: the tensor example samples independently
with replacement, while explicit offline adapters may shard an epoch traversal. The learning rate is not scaled
automatically.

Objectives return a local mean loss and its valid supervision count in
``LossResult.weight``. The distributed learner differentiates the global
weighted mean ``sum(weight_rank * loss_rank) / sum(weight_rank)``. For masked
regression, that weight is the number of valid scalar targets, rather than the
number of padded tensor elements or an unweighted rank count.

Only rank 0 writes training artifacts and runs validation. Validation receives
``learner.model`` as an unwrapped model so its forward pass does not require other
ranks. Each rank participates in ordinary ``fit()``, ``save()`` and ``resume()``
calls. A save within rank 0's validation callback uses the already gathered
committed-update snapshot and does not start another collective.

Distributed checkpoints preserve common optimizer/update state together with
each rank's RNG, source-sampling state and model buffers. Resume with the same
world size, recipe, source and training configuration:

.. code-block:: bash

   OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
     -m embodichain train-sl --config /tmp/sl-ddp-resume.yaml \
     --resume outputs/sl_ddp_cpu/step_100.pt

In this example, copy ``cpu.yaml`` to ``/tmp/sl-ddp-resume.yaml`` and change only
``trainer.output_dir`` to a fresh destination. This preserves the completed
original run when demonstrating continuation from its older update 100 checkpoint.
An interrupted run can use its original directory when no later metrics would
be overwritten. See ``examples/learning/sl_ddp/README.md`` for the full recipe.

Programmatic process-group scope
--------------------------------

The CLI owns process-group setup and cleanup automatically. Programmatic callers
can use ``distributed_run(device)`` around recipe construction and training; every
worker enters the scope. It yields the resolved ``torch.device`` and destroys
only a process group that it created. An already initialized group remains owned
by its caller.

.. code-block:: python

   from pathlib import Path
   import yaml
   from embodichain.learning.sl import distributed_run
   from examples.learning.sl_ddp.recipe import build_trainer

   config = yaml.safe_load(Path("examples/learning/sl_ddp/cpu.yaml").read_text())
   with distributed_run(config["trainer"]["device"]) as device:
       config["trainer"]["device"] = str(device)
       trainer = build_trainer(config)
       trainer.fit()

The corresponding PyTorch 2.7 references are
`torchrun <https://github.com/pytorch/pytorch/blob/v2.7.0/torch/distributed/run.py>`_
and `DistributedDataParallel
<https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html>`_.

embodichain.learning.sl
-----------------------

.. currentmodule:: embodichain.learning.sl

.. autosummary::

   LossResult
   SupervisedBatch
   SupervisedObjective
   SupervisedTrainingState
   SupervisedLearner
   SupervisedSource
   TensorBatchSource
   SequenceBatchSource
   SupervisedTrainer
   SupervisedTrainerCfg
   distributed_run

.. autoclass:: LossResult
   :members:

.. autoclass:: SupervisedBatch
   :members:

.. autoclass:: SupervisedObjective
   :members:

.. autoclass:: SupervisedTrainingState
   :members:

.. autoclass:: SupervisedLearner
   :members:

.. autoclass:: SupervisedSource
   :members:

.. autoclass:: TensorBatchSource
   :members:

.. autoclass:: SequenceBatchSource
   :members:

.. autoclass:: SupervisedTrainer
   :members:

.. autoclass:: SupervisedTrainerCfg
   :members:

.. autofunction:: distributed_run

embodichain.learning.sl.distributed
-----------------------------------

.. currentmodule:: embodichain.learning.sl.distributed

.. autosummary::

   distributed_run

.. autofunction:: distributed_run

embodichain.learning.sl.learner
-------------------------------

.. currentmodule:: embodichain.learning.sl.learner

.. autosummary::

   SupervisedLearner

.. autoclass:: SupervisedLearner
   :members:

embodichain.learning.sl.sequence
--------------------------------

.. currentmodule:: embodichain.learning.sl.sequence

.. autosummary::

   SequenceBatchSource

.. autoclass:: SequenceBatchSource
   :members:

The optional bridge samples one canonical causal window and calls an explicit,
versioned encoder. It preserves original masks and provenance under the reserved
``metadata['sequence_source']`` namespace, excluding observations and actions
from that diagnostic context. The encoder deliberately selects measured inputs
and supervision; the adapter infers neither action meaning nor feature layout.
The runnable ``examples/learning/sl_data_source`` example demonstrates action
targets and causal state-transition targets using a tiny synthetic CPU corpus.

Exact recovery is disabled by default. ``immutable=True`` requires an unchanged,
content-hashed resident ``TensorDataSource`` (including ``LeRobotDataSource``)
without sampling quotas. Schema, window, encoder version, language and semantic
metadata enter identity; storage paths and read/cache diagnostics do not.
Construction and resume hash CPU content using bounded blocks. Sampling checks
the immutable promise without rehashing the corpus at every update.

Live or quota-managed sources can train and save snapshots, but exact
``SupervisedTrainer.resume()`` is rejected collectively before learning, RNG or
output admission state changes. Producer state and delivery credits are not
restored by the trainer. Start/stop, refill, waits and handling sampling failures
remain caller-owned; the bridge performs no retry or quota refund.

embodichain.learning.sl.sources
-------------------------------

.. currentmodule:: embodichain.learning.sl.sources

.. autosummary::

   SupervisedSource
   TensorBatchSource

.. autoclass:: SupervisedSource
   :members:

.. autoclass:: TensorBatchSource
   :members:

embodichain.learning.sl.trainer
-------------------------------

.. currentmodule:: embodichain.learning.sl.trainer

.. autosummary::

   SupervisedTrainerCfg
   SupervisedTrainer

.. autoclass:: SupervisedTrainerCfg
   :members:

.. autoclass:: SupervisedTrainer
   :members:

embodichain.learning.sl.types
-----------------------------

.. currentmodule:: embodichain.learning.sl.types

.. autosummary::

   SupervisedBatch
   LossResult
   SupervisedObjective
   SupervisedTrainingState

.. autoclass:: SupervisedBatch
   :members:

.. autoclass:: LossResult
   :members:

.. autoclass:: SupervisedObjective
   :members:

.. autoclass:: SupervisedTrainingState
   :members:

embodichain.learning.sl.workflow
--------------------------------

.. currentmodule:: embodichain.learning.sl.workflow

.. autosummary::

   train_from_config
   cli

.. autofunction:: train_from_config

.. autofunction:: cli

Model-specific recipes
----------------------

Concrete models own their physical codecs, objectives and source adapters in
the :doc:`official model library </api_reference/model_catalogs>`. The NeRD
recipe connects committed trajectory windows to this common trainer without
loading simulation or requiring the example runner.

Recoverable update participants
-------------------------------

Factories may pass ordered ``training_states`` implementing
``SupervisedTrainingState``. Participants commit after optimizer success; their
owned state is saved per rank, validated before live recovery mutation, and
restored after validation callbacks. This supports data cursors and model
averages without model-specific trainer subclasses. Default empty participants
retain compatibility with existing checkpoints.

Checkpoint formats 3/4 mark nonempty training participants or nondefault buffer
broadcasting, so older readers reject required state instead of discarding it.
Default recipes continue to write/read formats 1/2.
