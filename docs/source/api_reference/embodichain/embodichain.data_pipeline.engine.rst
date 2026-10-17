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
   :exclude-members: SharedLanguageRegistry
   :undoc-members:
   :show-inheritance:

.. automodule:: embodichain.data_pipeline.engine.data
   :members:
   :no-index:
   :undoc-members:
   :show-inheritance:

Recorded actions and provenance
-------------------------------

``OnlineDataEngineCfg.action_dim`` optionally separates stored action width
from ``state_dim``; omitting it preserves the historical shared width.
``action_semantics="expert_controller_command"`` is the default and follows
the environment's ``ExpertActionSpec``. ``action_semantics="policy_request"``
requires an explicit width and ordered ``action_contracts`` matching the live
ActionManager. The worker validates resolved semantics, dimensions, term slices
and controlled-joint order before generating transitions. Explicit policy
recording uses raw requests independently of the expert schema and does not
require a DatasetManager.

``refill_policy="manual"`` lets the owner request complete generations with
``request_refill()`` and observe publication with ``wait_for_generation()``.
The default remains sample-threshold refill. ``get_stats()`` returns detached
counters and excludes the active write window from valid-frame counts.
``shutdown_timeout`` is the finite, positive graceful-close wait in seconds
(default ``5.0``). Increasing it allows longer worker teardown; it does not
change forced terminate/kill waits or their terminal failure semantics.

``include_provenance=True`` adds ``ods_run_id``, ``ods_episode_id``,
``ods_generation``, ``ods_frame_index`` and ``ods_accepted_at_ns`` to each frame.
The frame index is its zero-based stored episode position. The timestamp is
successful-publication Unix nanoseconds, stamped under the publication lock
and copied with its payload. Invalid frames and causal-history padding use
``-1`` for all provenance fields. These engine-local identities do not restore
shared-buffer contents or replace offline corpus content identities.

Shared language snapshots
-------------------------

Generated batches include numeric ``task_index`` and ``subtask_index`` columns.
``OnlineDataEngine.resolve_language(batch)`` returns nested task/subtask strings
for tokenization, including in DataLoader transforms. IDs are engine-local and
append-only, so a copied sample still resolves after its trajectory slot is
refilled. They are independent of an offline recorder's vocabulary indices.
History padding retains ``-1`` in both language columns. The resolver rejects
invalid IDs, so select real ``valid`` or supervised ``loss_mask`` frames before
decoding padded histories.

The registry uses a bounded shared UTF-8 buffer instead of mutable strings in
TensorDict storage. ``language_buffer_bytes`` limits its lifetime capacity;
overflow fails the producer before publishing that rollout. Consumers use the
same immutable indices and never depend on the current contents of reused rows.

The package-level alias is available for convenience; the canonical class and
its methods are indexed under the language module.

.. autoclass:: embodichain.data_pipeline.engine.SharedLanguageRegistry
   :no-index:

.. automodule:: embodichain.data_pipeline.engine.language
   :members:
