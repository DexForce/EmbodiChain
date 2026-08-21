embodichain.lab.gym.envs.expert_program
=======================================

.. automodule:: embodichain.lab.gym.envs.expert_program

   .. autosummary::

      ExpertProgramCfg
      ExpertProgramIntegrationCfg
      ExpertProgramCompiler
      CompiledProgram
      load_expert_program
      loads_expert_program_json
      parse_expert_program_json
      decode_expert_program
      ExpertProgramEnvironmentMixin
      ExpertProgramEnvironmentAdapter
      SimulationSceneBinding
      SimulationResourceEndpointBinding
      SimulationRobotResourceBinding
      RobotResourceBinding
      ControlPartEndpointBinding
      ControlPartResourceBinding
      SimulationRobotSkillProfileBinding
      SimulationExpertProgramFactory
      SimulationSegmentPolicyPort
      ControlCommandStateEvidenceTracker
      AcceptedRuntimeCommandObserver
      AcceptedRuntimeCommandObserverFactory
      AntipodalGraspAffordanceBinding
      ArticulationOperationAffordanceBinding
      ArticulationOperationTargetBinding
      AtomicDemoBridge
      BarrierCfg
      BufferedGymCommandSink
      CompiledBarrier
      CompiledParallelBlock
      CompiledParallelBranch
      CompiledPostPolicy
      CompiledProgramAnalysis
      CompiledProgramCall
      CompiledProgramSegment
      CompiledProgramValidator
      CompiledRepeatFrame
      CompiledTargetSelection
      ConfigPath
      ConfigPathPart
      ControlPartCommandPreset
      CyclicPoseTargetCfg
      DeclarativeCfgValue
      DemoBridgeError
      EXPERT_PROGRAM_SCHEMA_VERSION
      EXPERT_PROGRAM_SCHEMA_VERSION_V2
      EnvironmentStepClock
      EnvironmentStepTimingError
      ExpertProgramCompileError
      ExpertProgramConfigError
      ExpertProgramDecodeError
      ExpertProgramEnvironmentFactory
      ExpertProgramRuntimeAssembly
      ExpertProgramSceneResolver
      ExpertProgramValidationContext
      ExpertProgramValidationError
      GymPlanningObservationProvider
      HandOverCfg
      InvokeCfg
      MAX_DECLARATIVE_DEPTH
      MAX_DECLARATIVE_NODES
      MAX_EXPANDED_CALLS
      MAX_EXPERT_PROGRAM_BYTES
      MAX_PROGRAM_DEPTH
      MAX_PROGRAM_NODES
      MAX_REPEAT_COUNT
      MaterializedCompiledProgram
      MotionGeneratorFactory
      ObjectNearTargetValidatorCfg
      OperateArticulationCfg
      ParallelCfg
      PickCfg
      PlaceCfg
      PlanningObservationPort
      PoseCfg
      PostPolicyCfg
      ProgramNodeCfg
      RegisteredSemanticCallCfg
      RepeatCfg
      RuntimeCommandFrameEncoder
      RuntimeTransportActionEncoder
      SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS
      SceneReferenceRole
      SceneRegistryProgramResolver
      SegmentCfg
      SegmentPostPolicyMetadataPort
      SegmentPostPolicyPort
      SegmentPostPolicyResultPort
      SegmentValidatorMetadataPort
      SegmentValidatorPort
      SemanticCallCfg
      SequenceCfg
      SharedTickSceneProvider
      SimulationArticulationBinding
      SimulationArticulationLinkBinding
      SimulationExpertProgramEnvironment
      SimulationPlanningObservationProvider
      SimulationRigidObjectBinding
      SkillRuntimeAssemblyPort
      TargetCfg
      TargetRefCfg
      UnsupportedRuntimeTransportError
      ValidatorCfg
      WaitStablePostCfg
      create_simulation_expert_program_adapter
      decode_semantic_call
      encode_semantic_call
      render_config_path
      validate_expert_program
      EndpointAdapterDeclaration
      ExpertProgramIntegrationCatalog
      IntegrationFingerprintMismatch
      ParallelCommandSafetyValidatorFactory
      ParallelSafetyDeclaration
      RuntimeTransportDeclaration
      SimulationExpertProgramRegistration
      StandardExtensionDeclarations
      VersionedKey
      default_simulation_settle_presets

.. currentmodule:: embodichain.lab.gym.envs.expert_program

Schema and loading
------------------

The public decoders and file loaders support Expert Program schema versions 1
and 2. Version 2 adds deterministic parallel blocks with explicit barriers.

.. autoclass:: ExpertProgramCfg
   :members:

.. autoclass:: ExpertProgramIntegrationCfg
   :members:

.. autofunction:: load_expert_program

.. autofunction:: loads_expert_program_json

.. autofunction:: parse_expert_program_json

.. autofunction:: decode_expert_program

MLLM frontend
-------------

The MLLM frontend intentionally accepts only the constrained schema version 1
surface. Trusted host code remains responsible for authoring version 2
parallel structure and the integration selection.

.. autofunction:: embodichain.agents.mllm.decode_mllm_expert_program

.. autofunction:: embodichain.agents.mllm.compile_mllm_expert_program

Compilation and environment integration
---------------------------------------

.. autoclass:: ExpertProgramCompiler
   :members:

.. autoclass:: CompiledProgram
   :members:

.. autoclass:: ExpertProgramEnvironmentMixin
   :members:

.. autoclass:: ExpertProgramEnvironmentAdapter
   :members:

.. autoclass:: SkillRuntimeAssemblyPort
   :members:

.. autoclass:: ExpertProgramIntegrationCatalog
   :members:

.. autoclass:: SimulationExpertProgramRegistration
   :members:

Simulation integration
----------------------

.. autoclass:: SimulationSceneBinding
   :members:

.. autoclass:: SimulationResourceEndpointBinding

.. autoclass:: SimulationRobotResourceBinding

.. autoclass:: RobotResourceBinding
   :members:

.. autoclass:: ControlPartEndpointBinding
   :members:

.. autoclass:: ControlPartResourceBinding
   :members:

.. autoclass:: SimulationRobotSkillProfileBinding
   :members:

.. autoclass:: SimulationExpertProgramFactory
   :members:

.. autoclass:: SimulationSegmentPolicyPort
   :members:

.. autoclass:: ControlCommandStateEvidenceTracker
   :members:
