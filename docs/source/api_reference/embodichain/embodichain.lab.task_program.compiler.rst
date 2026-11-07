embodichain.lab.task_program.compiler
=====================================

.. automodule:: embodichain.lab.task_program.compiler
   :members:
   :undoc-members:

   The compiler expands a validated AST into immutable segments and calls.
   Semantic lowering is provider-independent; live planning starts only after
   an environment integration is assembled.

   .. autosummary::

      CompiledArticulationJointPositionValidator
      CompiledBarrier
      CompiledObjectNearRelativeTargetValidator
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

Validator type aliases
----------------------

.. currentmodule:: embodichain.lab.task_program.compiler

These aliases describe the compiled validator variants.

.. autodata:: CompiledTaskProgramValidator

Relative position acceptance retains both canonical scene references and a
world-axis displacement. The simulation validator observes both objects again
at acceptance time, independently of the pose used during action planning.
