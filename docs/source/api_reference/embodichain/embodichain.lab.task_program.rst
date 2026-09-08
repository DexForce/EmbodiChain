embodichain.lab.task_program
==============================

.. automodule:: embodichain.lab.task_program
   :members:
   :no-index:

   .. autosummary::

      TaskProgramCfg
      TaskProgramIntegrationCfg
      PoseCfg
      TargetRefCfg
      CyclicPoseTargetCfg
      PickCfg
      PlaceCfg
      HandOverCfg
      RegisteredSemanticCallCfg
      InvokeCfg
      SequenceCfg
      RepeatCfg
      SegmentCfg
      ParallelCfg
      BarrierCfg
      WaitStablePostCfg
      ObjectNearTargetValidatorCfg
      ObjectNearRelativeTargetValidatorCfg
      ArticulationJointPositionValidatorCfg
      TaskProgramCompiler
      CompiledTaskProgram
      load_task_program
      loads_task_program_json
      parse_task_program_json
      decode_task_program
      validate_task_program
      render_config_path
      ConfigPath
      ConfigPathPart
      SceneReferenceRole
      TaskProgramValidationContext
      TaskProgramConfigError
      TaskProgramDecodeError
      TaskProgramValidationError
      TaskProgramCompileError

.. currentmodule:: embodichain.lab.task_program

Schema
------

.. autoclass:: TaskProgramCfg
   :members:

.. autoclass:: TaskProgramIntegrationCfg
   :members:

.. autoclass:: PoseCfg
   :members:

.. autoclass:: TargetRefCfg
   :members:

.. autoclass:: CyclicPoseTargetCfg
   :members:

.. autoclass:: PickCfg
   :members:

.. autoclass:: PlaceCfg
   :members:

.. autoclass:: HandOverCfg
   :members:

.. autoclass:: RegisteredSemanticCallCfg
   :members:

.. autoclass:: InvokeCfg
   :members:

.. autoclass:: SequenceCfg
   :members:

.. autoclass:: RepeatCfg
   :members:

.. autoclass:: SegmentCfg
   :members:

.. autoclass:: ParallelCfg
   :members:

.. autoclass:: BarrierCfg
   :members:

.. autoclass:: WaitStablePostCfg
   :members:

.. autoclass:: ObjectNearTargetValidatorCfg
   :members:

.. autoclass:: ObjectNearRelativeTargetValidatorCfg
   :members:

.. autoclass:: ArticulationJointPositionValidatorCfg
   :members:

Loading and validation
----------------------

.. autofunction:: load_task_program

.. autofunction:: loads_task_program_json

.. autofunction:: parse_task_program_json

.. autofunction:: decode_task_program

.. autofunction:: validate_task_program

.. autofunction:: render_config_path

.. autodata:: ConfigPath

.. autodata:: ConfigPathPart

.. autodata:: SceneReferenceRole

.. autoclass:: TaskProgramValidationContext

.. autoclass:: TaskProgramConfigError

.. autoclass:: TaskProgramDecodeError

.. autoclass:: TaskProgramValidationError

Compilation
-----------

Compilation is provider-independent. Environment and simulator binding is
documented under :mod:`embodichain.lab.task_program.integrations`; the final
Gym lifecycle bridge is :mod:`embodichain.lab.gym.envs.task_program`.

.. autoclass:: TaskProgramCompiler
   :members:

.. autoclass:: CompiledTaskProgram
   :members:

.. autoclass:: TaskProgramCompileError

MLLM frontend
-------------

.. autofunction:: embodichain.agents.mllm.decode_mllm_task_program

.. autofunction:: embodichain.agents.mllm.compile_mllm_task_program
