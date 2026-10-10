embodichain.data_pipeline.engine
================================

.. automodule:: embodichain.data_pipeline.engine

Overview
--------

Online data streaming engine: a process-safe shared buffer for trajectory data.

   .. rubric:: Functions

   .. autosummary::

      OnlineDataEngine
      OnlineDataEngineCfg

.. automodule:: embodichain.data_pipeline.engine
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain.data_pipeline.engine.data
   :members:
   :undoc-members:
   :show-inheritance:

Shared language snapshots
-------------------------

Generated batches include numeric ``task_index`` and ``subtask_index`` columns.
``OnlineDataEngine.resolve_language(batch)`` returns nested task/subtask strings
for tokenization, including in DataLoader transforms. IDs are engine-local and
append-only, so a copied sample still resolves after its trajectory slot is
refilled. They are independent of an offline recorder's vocabulary indices.

The registry uses a bounded shared UTF-8 buffer instead of mutable strings in
TensorDict storage. ``language_buffer_bytes`` limits its lifetime capacity;
overflow fails the producer before publishing that rollout. Consumers use the
same immutable indices and never depend on the current contents of reused rows.

.. automodule:: embodichain.data_pipeline.engine.language
   :members:
