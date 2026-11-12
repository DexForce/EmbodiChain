embodichain.lab.task_program.language
=====================================

.. automodule:: embodichain.lab.task_program.language
   :members:
   :undoc-members:

   Task Program source is represented by executable-free schema values. The
   same package owns strict JSON/YAML decoding, reference validation, bounded
   loading, and pathful configuration errors.

   .. autosummary::

      ArticulationJointPositionValidatorCfg
      BarrierCfg
      ConfigPath
      ConfigPathPart
      CyclicPoseTargetCfg
      PoseTargetCfg
      SceneRelativePoseTargetCfg
      MoveHeldObjectCfg
      HoldPostCfg
      NormalizedTaskProgram
      normalize_task_program
      HandOverCfg
      InvokeCfg
      ObjectNearRelativeTargetValidatorCfg
      ObjectNearTargetValidatorCfg
      ParallelCfg
      PickCfg
      PlaceCfg
      PoseCfg
      RegisteredSemanticCallCfg
      RepeatCfg
      SceneReferenceRole
      SegmentCfg
      SegmentContractCfg
      SegmentEvidenceCfg
      SequenceCfg
      TargetRefCfg
      TaskProgramCfg
      TaskProgramConfigError
      TaskProgramDecodeError
      TaskProgramIntegrationCfg
      TaskProgramValidationContext
      TaskProgramValidationError
      WaitStablePostCfg
      decode_task_program
      load_task_program
      loads_task_program_json
      parse_task_program_json
      render_config_path
      validate_task_program

Configuration paths and reference roles
---------------------------------------

.. currentmodule:: embodichain.lab.task_program.language

These exported values describe configuration paths and reference categories.

.. autodata:: ConfigPath

.. autodata:: ConfigPathPart

.. autodata:: SceneReferenceRole

Authoring and effective call contracts
-------------------------------------

The normalizer expands shorthand into the same strict AST and retains original
source locations. Deployment catalogs supply Demo Skill aliases and effective
argument contracts; normalization never constructs a simulator.

.. automodule:: embodichain.lab.task_program.language.authoring
   :members:
   :no-index:

.. autoclass:: embodichain.lab.task_program.language.semantic_contracts.SemanticCallContract
   :members:
