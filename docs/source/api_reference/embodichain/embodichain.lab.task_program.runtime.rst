embodichain.lab.task_program.runtime
====================================

.. automodule:: embodichain.lab.task_program.runtime
   :members:
   :no-index:

   Runtime services execute lowered Semantic Calls and synchronize parallel
   branches. They do not own Gym episode lifecycle, ``env.step()``, or
   trajectory recording.

   .. autosummary::

      ParallelSemanticExecutor
      ParallelTimingPolicy
      SemanticCallExecutor
      SemanticExecutionResult
      SemanticExecutionStatus
