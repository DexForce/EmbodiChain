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
drawn examples is ``batch_size * world_size``. Each worker samples independently
with replacement, so examples can overlap. The learning rate is not scaled
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
   SupervisedLearner
   SupervisedSource
   TensorBatchSource
   SupervisedTrainer
   SupervisedTrainerCfg
   distributed_run

.. autoclass:: LossResult
   :members:

.. autoclass:: SupervisedBatch
   :members:

.. autoclass:: SupervisedObjective
   :members:

.. autoclass:: SupervisedLearner
   :members:

.. autoclass:: SupervisedSource
   :members:

.. autoclass:: TensorBatchSource
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

.. autoclass:: SupervisedBatch
   :members:

.. autoclass:: LossResult
   :members:

.. autoclass:: SupervisedObjective
   :members:

embodichain.learning.sl.workflow
--------------------------------

.. currentmodule:: embodichain.learning.sl.workflow

.. autosummary::

   train_from_config
   cli

.. autofunction:: train_from_config

.. autofunction:: cli
