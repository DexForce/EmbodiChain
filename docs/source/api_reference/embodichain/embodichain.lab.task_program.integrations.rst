embodichain.lab.task_program.integrations
=========================================

.. automodule:: embodichain.lab.task_program.integrations
   :members:
   :no-index:

   .. autosummary::

      AntipodalGraspAffordanceBinding
      ConfiguredHandOverPoseProvider
      ContainerAffordanceBinding
      ControlPartCommandPreset
      ControlPartEvidenceProviderDeclaration
      ControlPartEvidenceProviderFactory
      ControlPartEndpointBinding
      ControlPartResourceBinding
      CuroboParallelCommandSafetyValidator
      CuroboParallelSafetyValidatorFactory
      EndpointAdapterDeclaration
      TaskProgramAdapterFactory
      TaskProgramEnvironmentAdapter
      TaskProgramEnvironmentFactory
      TaskProgramIntegrationCatalog
      TaskProgramRuntimeAssembly
      IntegrationFingerprintMismatch
      ParallelCommandSafetyValidatorFactory
      ParallelSafetyDeclaration
      PlanningObservationPort
      RegisteredSemanticLowererDeclaration
      RegisteredSemanticLowererFactory
      RuntimeTransportDeclaration
      SimulationArticulationBinding
      SimulationArticulationLinkBinding
      SimulationTaskProgramAdapterFactory
      SimulationTaskProgramFactory
      SimulationTaskProgramRegistration
      SimulationRigidObjectBinding
      SimulationRobotSkillProfileBinding
      SimulationSceneBinding
      SimulationSegmentPolicyPort
      StandardExtensionDeclarations
      SupportSurfaceAffordanceBinding
      VersionedKey
      create_simulation_task_program_adapter
      default_simulation_settle_presets

.. currentmodule:: embodichain.lab.task_program.integrations

Environment assembly
--------------------

The provider-independent language and compiler live in
:mod:`embodichain.lab.task_program`. This package owns explicit environment,
registration, and simulation assembly. Gym stepping remains in
:mod:`embodichain.lab.gym.envs.task_program`.

.. autoclass:: TaskProgramEnvironmentAdapter
   :members:

.. autoclass:: TaskProgramAdapterFactory
   :members:

.. autoclass:: TaskProgramEnvironmentFactory

.. autoclass:: TaskProgramRuntimeAssembly
   :members:

.. autoclass:: PlanningObservationPort

Registration and extensions
---------------------------

.. autoclass:: TaskProgramIntegrationCatalog
   :members:

.. autoclass:: SimulationTaskProgramRegistration
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

.. autoclass:: SimulationTaskProgramFactory
   :members:

.. autoclass:: SimulationTaskProgramAdapterFactory
   :members:

.. autoclass:: ConfiguredHandOverPoseProvider
   :members:

.. autoclass:: CuroboParallelCommandSafetyValidator
   :members:

.. autoclass:: CuroboParallelSafetyValidatorFactory
   :members:

.. autoclass:: SimulationSegmentPolicyPort
   :members:

.. autofunction:: create_simulation_task_program_adapter

Implementation module exports
-----------------------------

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.parallel_safety

.. autosummary::

   CuroboParallelCommandSafetyValidator
   CuroboParallelSafetyValidatorFactory

.. currentmodule:: embodichain.lab.task_program.integrations.catalog

.. autosummary::

   TaskProgramIntegrationCatalog
   IntegrationFingerprintMismatch
   SimulationTaskProgramRegistration

.. currentmodule:: embodichain.lab.task_program.integrations.extensions

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
