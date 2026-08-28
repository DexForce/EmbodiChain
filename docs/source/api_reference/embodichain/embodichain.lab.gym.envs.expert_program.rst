embodichain.lab.gym.envs.expert_program
=======================================

.. automodule:: embodichain.lab.gym.envs.expert_program
   :members:
   :no-index:

   .. autosummary::

      AntipodalGraspAffordanceBinding
      AtomicDemoBridge
      BufferedGymCommandSink
      ConfiguredHandOverPoseProvider
      ContainerAffordanceBinding
      ControlPartCommandPreset
      ControlPartEvidenceProviderDeclaration
      ControlPartEvidenceProviderFactory
      ControlPartEndpointBinding
      ControlPartResourceBinding
      CuroboParallelCommandSafetyValidator
      CuroboParallelSafetyValidatorFactory
      DemoBridgeError
      EnvironmentStepClock
      EnvironmentStepTimingError
      EndpointAdapterDeclaration
      ExpertProgramAdapterFactory
      ExpertProgramEnvironmentAdapter
      ExpertProgramEnvironmentFactory
      ExpertProgramIntegrationCatalog
      ExpertProgramRuntimeAssembly
      GymPlanningObservationProvider
      IntegrationFingerprintMismatch
      ParallelCommandSafetyValidatorFactory
      ParallelSafetyDeclaration
      PlanningObservationPort
      RegisteredSemanticLowererDeclaration
      RegisteredSemanticLowererFactory
      RuntimeCommandFrameEncoder
      RuntimeTransportDeclaration
      RuntimeTransportActionEncoder
      SegmentPostPolicyPort
      SegmentValidatorPort
      SimulationArticulationBinding
      SimulationArticulationLinkBinding
      SimulationExpertProgramAdapterFactory
      SimulationExpertProgramFactory
      SimulationExpertProgramRegistration
      SimulationRigidObjectBinding
      SimulationRobotSkillProfileBinding
      SimulationSceneBinding
      SimulationSegmentPolicyPort
      StandardExtensionDeclarations
      SupportSurfaceAffordanceBinding
      UnsupportedRuntimeTransportError
      VersionedKey
      create_simulation_expert_program_adapter
      default_simulation_settle_presets

.. currentmodule:: embodichain.lab.gym.envs.expert_program

Environment assembly
--------------------

The provider-independent schema and compiler live in
:mod:`embodichain.lab.expert_program`. This package owns only Gym and
simulation integration.

.. autoclass:: ExpertProgramEnvironmentAdapter
   :members:

.. autoclass:: ExpertProgramAdapterFactory
   :members:

.. autoclass:: ExpertProgramEnvironmentFactory

.. autoclass:: ExpertProgramRuntimeAssembly
   :members:

.. autoclass:: PlanningObservationPort

Registration and extensions
---------------------------

.. autoclass:: ExpertProgramIntegrationCatalog
   :members:

.. autoclass:: SimulationExpertProgramRegistration
   :members:

.. autoclass:: IntegrationFingerprintMismatch

.. autoclass:: ControlPartEvidenceProviderDeclaration
   :members:

.. autoclass:: ControlPartEvidenceProviderFactory

.. autoclass:: RegisteredSemanticLowererDeclaration
   :members:

.. autoclass:: RegisteredSemanticLowererFactory

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

Gym bridge
----------

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

Simulation bindings
-------------------

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

.. autoclass:: ContainerAffordanceBinding
   :members:

.. autoclass:: SupportSurfaceAffordanceBinding
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

.. autoclass:: SimulationExpertProgramAdapterFactory
   :members:

.. autoclass:: ConfiguredHandOverPoseProvider
   :members:

.. autoclass:: CuroboParallelCommandSafetyValidator
   :members:

.. autoclass:: CuroboParallelSafetyValidatorFactory
   :members:

.. autoclass:: SimulationSegmentPolicyPort
   :members:

.. autofunction:: create_simulation_expert_program_adapter

Implementation module exports
-----------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.simulation_parallel_safety

.. autosummary::

   CuroboParallelCommandSafetyValidator
   CuroboParallelSafetyValidatorFactory

.. currentmodule:: embodichain.lab.gym.envs.expert_program.catalog

.. autosummary::

   ExpertProgramIntegrationCatalog
   IntegrationFingerprintMismatch
   SimulationExpertProgramRegistration

.. currentmodule:: embodichain.lab.gym.envs.expert_program.extensions

.. autosummary::

   ControlPartEvidenceProviderDeclaration
   ControlPartEvidenceProviderFactory
   EndpointAdapterDeclaration
   ParallelCommandSafetyValidatorFactory
   ParallelSafetyDeclaration
   RegisteredSemanticLowererDeclaration
   RegisteredSemanticLowererFactory
   RuntimeTransportDeclaration
   StandardExtensionDeclarations
   VersionedKey
   build_standard_extension_declarations
   declare_control_part_evidence_factory
   declare_endpoint_adapter
   declare_parallel_safety_factory
   declare_registered_semantic_lowerer_factory
   declare_runtime_transport
   validate_immutable_extension_declaration

.. autofunction:: build_standard_extension_declarations

.. autofunction:: declare_control_part_evidence_factory

.. autofunction:: declare_endpoint_adapter

.. autofunction:: declare_parallel_safety_factory

.. autofunction:: declare_registered_semantic_lowerer_factory

.. autofunction:: declare_runtime_transport

.. autofunction:: validate_immutable_extension_declaration
