embodichain.lab.task_program.runtime
====================================

.. automodule:: embodichain.lab.task_program.runtime
   :members:
   :undoc-members:

   Runtime services execute lowered Semantic Calls and synchronize parallel
   branches. They do not own Gym episode lifecycle, ``env.step()``, or
   trajectory recording.

   .. autosummary::

      ParallelSemanticExecutor
      ParallelTimingPolicy
      SemanticCallExecutor
      SemanticExecutionResult
      SemanticExecutionStatus

``SemanticCallExecutor.step()`` and ``advance()`` execute the same due cycle.
``advance()`` returns status and leaves result construction to the caller's
audit boundary. Progress is available through ``status``, ``current_call_index``,
``env_ids`` and ``wait_duration``. Cancellation, observation, dispatch and effect
verification retain the same order under both entry points.
