embodichain.lab.task_program.compiler
=====================================

.. automodule:: embodichain.lab.task_program.compiler
   :members:
   :no-index:

   The compiler expands a validated AST into immutable segments and calls.
   Semantic lowering is provider-independent; live planning starts only after
   an environment integration is assembled.

   .. autosummary::

      CompiledArticulationJointPositionValidator
      CompiledBarrier
      CompiledObjectNearTargetValidator
      CompiledParallelBlock
      CompiledParallelBranch
      CompiledPostPolicy
      CompiledRepeatFrame
      CompiledTargetSelection
      CompiledTaskProgram
      CompiledTaskProgramAnalysis
      CompiledTaskProgramCall
      CompiledTaskProgramSegment
      CompiledTaskProgramValidator
      TaskProgramCompileError
      TaskProgramCompiler
