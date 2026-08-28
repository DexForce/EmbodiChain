embodichain.lab.expert_program
==============================

.. automodule:: embodichain.lab.expert_program
   :members:
   :no-index:

   .. autosummary::

      ExpertProgramCfg
      ExpertProgramIntegrationCfg
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
      ArticulationJointPositionValidatorCfg
      ExpertProgramCompiler
      CompiledProgram
      load_expert_program
      loads_expert_program_json
      parse_expert_program_json
      decode_expert_program
      validate_expert_program
      render_config_path
      ConfigPath
      ConfigPathPart
      SceneReferenceRole
      ExpertProgramValidationContext
      ExpertProgramConfigError
      ExpertProgramDecodeError
      ExpertProgramValidationError
      ExpertProgramCompileError

.. currentmodule:: embodichain.lab.expert_program

Schema
------

.. autoclass:: ExpertProgramCfg
   :members:

.. autoclass:: ExpertProgramIntegrationCfg
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

.. autoclass:: ArticulationJointPositionValidatorCfg
   :members:

Loading and validation
----------------------

.. autofunction:: load_expert_program

.. autofunction:: loads_expert_program_json

.. autofunction:: parse_expert_program_json

.. autofunction:: decode_expert_program

.. autofunction:: validate_expert_program

.. autofunction:: render_config_path

.. autodata:: ConfigPath

.. autodata:: ConfigPathPart

.. autodata:: SceneReferenceRole

.. autoclass:: ExpertProgramValidationContext

.. autoclass:: ExpertProgramConfigError

.. autoclass:: ExpertProgramDecodeError

.. autoclass:: ExpertProgramValidationError

Compilation
-----------

Compilation is provider-independent. Environment and simulator binding is
documented under :mod:`embodichain.lab.gym.envs.expert_program`.

.. autoclass:: ExpertProgramCompiler
   :members:

.. autoclass:: CompiledProgram
   :members:

.. autoclass:: ExpertProgramCompileError

MLLM frontend
-------------

.. autofunction:: embodichain.agents.mllm.decode_mllm_expert_program

.. autofunction:: embodichain.agents.mllm.compile_mllm_expert_program
