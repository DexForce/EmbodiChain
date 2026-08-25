embodichain.lab.gym.envs.expert_program
=======================================

.. automodule:: embodichain.lab.gym.envs.expert_program

   .. autosummary::

      AntipodalGraspAffordanceBinding
      ArticulationJointPositionValidatorCfg
      AtomicDemoBridge
      BarrierCfg
      BufferedGymCommandSink
      ConfigPath
      ConfigPathPart
      CompiledProgram
      ConfiguredHandOverPoseProvider
      ContainerAffordanceBinding
      ControlPartCommandPreset
      ControlPartEvidenceProviderDeclaration
      ControlPartEvidenceProviderFactory
      ControlPartEndpointBinding
      ControlPartResourceBinding
      CyclicPoseTargetCfg
      CuroboParallelCommandSafetyValidator
      CuroboParallelSafetyValidatorFactory
      DemoBridgeError
      EXPERT_PROGRAM_SCHEMA_VERSION
      EnvironmentStepClock
      EnvironmentStepTimingError
      EndpointAdapterDeclaration
      ExpertProgramCfg
      ExpertProgramCompileError
      ExpertProgramCompiler
      ExpertProgramConfigError
      ExpertProgramDecodeError
      ExpertProgramEnvironmentAdapter
      ExpertProgramEnvironmentFactory
      ExpertProgramIntegrationCfg
      ExpertProgramIntegrationCatalog
      ExpertProgramRuntimeAssembly
      ExpertProgramValidationContext
      ExpertProgramValidationError
      GymPlanningObservationProvider
      HandOverCfg
      InvokeCfg
      IntegrationFingerprintMismatch
      ObjectNearTargetValidatorCfg
      ParallelCfg
      ParallelCommandSafetyValidatorFactory
      ParallelSafetyDeclaration
      PickCfg
      PlaceCfg
      PlanningObservationPort
      PoseCfg
      RegisteredSemanticCallCfg
      RepeatCfg
      RuntimeCommandFrameEncoder
      RuntimeTransportDeclaration
      RuntimeTransportActionEncoder
      SceneReferenceRole
      SegmentCfg
      SegmentPostPolicyPort
      SegmentValidatorPort
      SequenceCfg
      SimulationArticulationBinding
      SimulationArticulationLinkBinding
      SimulationExpertProgramFactory
      SimulationExpertProgramRegistration
      SimulationRigidObjectBinding
      SimulationRobotSkillProfileBinding
      SimulationSceneBinding
      SimulationSegmentPolicyPort
      StandardExtensionDeclarations
      SupportSurfaceAffordanceBinding
      TargetRefCfg
      UnsupportedRuntimeTransportError
      VersionedKey
      WaitStablePostCfg
      create_simulation_expert_program_adapter
      default_simulation_settle_presets
      decode_expert_program
      load_expert_program
      loads_expert_program_json
      parse_expert_program_json
      render_config_path
      validate_expert_program

.. currentmodule:: embodichain.lab.gym.envs.expert_program

Schema and loading
------------------

Expert Program schema version 2 is the only accepted top-level schema. It
contains bounded sequential nodes and deterministic parallel blocks whose
barrier is owned by the enclosing parallel node.

.. autodata:: EXPERT_PROGRAM_SCHEMA_VERSION

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

MLLM frontend
-------------

The MLLM frontend intentionally accepts only the constrained sequential subset
of schema version 2. Trusted host code remains responsible for authoring
parallel structure and selecting the integration.

.. autofunction:: embodichain.agents.mllm.decode_mllm_expert_program

.. autofunction:: embodichain.agents.mllm.compile_mllm_expert_program

Compilation and environment integration
---------------------------------------

Compilation resolves static scene identities through the core
``SceneManifest`` and returns one already bounded, materialized program. The
environment adapter then creates a fresh canonical ``SkillRuntime`` for each
bridge.

.. autoclass:: ExpertProgramCompiler
   :members:

.. autoclass:: CompiledProgram
   :members:

.. autoclass:: ExpertProgramCompileError

.. autoclass:: ExpertProgramEnvironmentAdapter
   :members:

.. autoclass:: ExpertProgramEnvironmentFactory

.. autoclass:: ExpertProgramRuntimeAssembly
   :members:

.. autoclass:: PlanningObservationPort

Registration catalogs and standard extensions
---------------------------------------------

The standard simulation path snapshots one task-owned registration before a
live environment is created. Its fingerprint covers the scene/profile
manifests and the exact endpoint, transport, and parallel-safety declarations
used again during runtime assembly.

.. autoclass:: ExpertProgramIntegrationCatalog
   :members:

.. autoclass:: SimulationExpertProgramRegistration
   :members:

.. autoclass:: IntegrationFingerprintMismatch

.. autoclass:: ControlPartEvidenceProviderDeclaration
   :members:

.. autoclass:: ControlPartEvidenceProviderFactory

.. autoclass:: EndpointAdapterDeclaration
   :members:

.. autoclass:: RuntimeTransportDeclaration
   :members:

.. autoclass:: ParallelSafetyDeclaration
   :members:

.. autoclass:: ParallelCommandSafetyValidatorFactory

.. autoclass:: StandardExtensionDeclarations
   :members:

.. autodata:: VersionedKey

.. autofunction:: default_simulation_settle_presets

Gym bridge ports
----------------

The bridge converts accepted runtime commands to lazy ``DemoSegment`` actions.
Only the normal environment executor calls ``env.step()``.

.. autoclass:: AtomicDemoBridge
   :members:

.. autoclass:: BufferedGymCommandSink
   :members:

.. autoclass:: EnvironmentStepClock
   :members:

.. autoclass:: GymPlanningObservationProvider
   :members:

.. autoclass:: RuntimeCommandFrameEncoder
   :members:

.. autoclass:: RuntimeTransportActionEncoder

.. autoclass:: SegmentPostPolicyPort

.. autoclass:: SegmentValidatorPort

.. autoclass:: DemoBridgeError

.. autoclass:: EnvironmentStepTimingError

.. autoclass:: UnsupportedRuntimeTransportError

Simulation integration
----------------------

Simulation bindings translate explicit scene and robot declarations into the
core scene registry and robot skill profile. Generic non-control-part resources
use the core ``RobotResource`` type directly.

.. autoclass:: SimulationSceneBinding
   :members:

.. autoclass:: SimulationRigidObjectBinding
   :members:

.. autoclass:: SimulationArticulationBinding
   :members:

.. autoclass:: SimulationArticulationLinkBinding
   :members:

.. autoclass:: AntipodalGraspAffordanceBinding
   :members:

.. autoclass:: ControlPartCommandPreset
   :members:

.. autoclass:: ControlPartEndpointBinding
   :members:

.. autoclass:: ControlPartResourceBinding
   :members:

.. autoclass:: SimulationRobotSkillProfileBinding
   :members:

.. autoclass:: SimulationExpertProgramFactory
   :members:

.. autoclass:: ConfiguredHandOverPoseProvider
   :members:

.. autoclass:: CuroboParallelCommandSafetyValidator
   :members:

.. autoclass:: CuroboParallelSafetyValidatorFactory
   :members:

.. autoclass:: ContainerAffordanceBinding
   :members:

.. autoclass:: SupportSurfaceAffordanceBinding
   :members:

.. autoclass:: SimulationSegmentPolicyPort
   :members:

.. autofunction:: create_simulation_expert_program_adapter

Parallel-safety implementation module
-------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.simulation_parallel_safety

.. autosummary::

   CuroboParallelCommandSafetyValidator
   CuroboParallelSafetyValidatorFactory

Catalog implementation module
-----------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.catalog

.. autosummary::

   ExpertProgramIntegrationCatalog
   IntegrationFingerprintMismatch
   SimulationExpertProgramRegistration

Extension declaration implementation module
-------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.extensions

.. autosummary::

   ControlPartEvidenceProviderDeclaration
   ControlPartEvidenceProviderFactory
   EndpointAdapterDeclaration
   ParallelCommandSafetyValidatorFactory
   ParallelSafetyDeclaration
   RuntimeTransportDeclaration
   StandardExtensionDeclarations
   VersionedKey
   build_standard_extension_declarations
   declare_control_part_evidence_factory
   declare_endpoint_adapter
   declare_parallel_safety_factory
   declare_runtime_transport
   validate_immutable_extension_declaration

.. autoclass:: ControlPartEvidenceProviderDeclaration
   :members:

.. autoclass:: ControlPartEvidenceProviderFactory

.. autoclass:: EndpointAdapterDeclaration
   :members:

.. autoclass:: ParallelCommandSafetyValidatorFactory

.. autoclass:: ParallelSafetyDeclaration
   :members:

.. autoclass:: RuntimeTransportDeclaration
   :members:

.. autoclass:: StandardExtensionDeclarations
   :members:

.. autodata:: VersionedKey

.. autofunction:: build_standard_extension_declarations

.. autofunction:: declare_control_part_evidence_factory

.. autofunction:: declare_endpoint_adapter

.. autofunction:: declare_parallel_safety_factory

.. autofunction:: declare_runtime_transport

.. autofunction:: validate_immutable_extension_declaration
