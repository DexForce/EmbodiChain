embodichain.lab.task_program.integrations
=========================================

.. automodule:: embodichain.lab.task_program.integrations
   :members:
   :undoc-members:
   :exclude-members: TaskProgramEnvironmentAdapter, TaskProgramAdapterFactory,
      TaskProgramEnvironmentFactory, TaskProgramRuntimeAssembly,
      PlanningObservationPort, TaskProgramIntegrationCatalog,
      SimulationTaskProgramRegistration, IntegrationFingerprintMismatch,
      ControlPartEvidenceProviderDeclaration, ControlPartEvidenceProviderFactory,
      RegisteredSemanticLowererDeclaration, RegisteredSemanticLowererFactory,
      EndpointAdapterDeclaration, RuntimeTransportDeclaration,
      ParallelSafetyDeclaration, ParallelCommandSafetyValidatorFactory,
      StandardExtensionDeclarations, VersionedKey, default_simulation_settle_presets,
      SimulationSceneBinding, SimulationRigidObjectBinding,
      SimulationRigidizedArticulationObjectBinding, SimulationArticulationBinding,
      SimulationArticulationLinkBinding, AntipodalGraspAffordanceBinding,
      RigidizedArticulationAntipodalGraspBinding, ContainerAffordanceBinding,
      SupportSurfaceAffordanceBinding, ControlPartCommandPreset,
      ControlPartEndpointBinding, ControlPartResourceBinding,
      SimulationRobotSkillProfileBinding, SimulationTaskProgramFactory,
      SimulationTaskProgramAdapterFactory, ConfiguredHandOverPoseProvider,
      CuroboParallelCommandSafetyValidator, CuroboParallelSafetyValidatorFactory,
      SimulationSegmentPolicyPort, create_simulation_task_program_adapter

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
      RigidizedArticulationAntipodalGraspBinding
      RuntimeTransportDeclaration
      SimulationArticulationBinding
      SimulationArticulationLinkBinding
      SimulationRigidizedArticulationObjectBinding
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

.. toctree::
   :maxdepth: 1

   embodichain.lab.task_program.integrations.scene_expansion

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

.. Register shared implementation names once beside their public definitions.
   Re-export packages can then index their own names without duplicate
   canonical registrations for the same implementation.

.. py:class:: SimulationSceneBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: SimulationSceneBinding
   :members:

.. py:class:: SimulationRigidObjectBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: SimulationRigidObjectBinding
   :members:

.. py:class:: SimulationRigidizedArticulationObjectBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: SimulationRigidizedArticulationObjectBinding
   :members:

.. py:class:: SimulationArticulationBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: SimulationArticulationBinding
   :members:

.. py:class:: SimulationArticulationLinkBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: SimulationArticulationLinkBinding
   :members:

.. py:class:: AntipodalGraspAffordanceBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: AntipodalGraspAffordanceBinding
   :members:

.. py:class:: RigidizedArticulationAntipodalGraspBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: RigidizedArticulationAntipodalGraspBinding
   :members:

.. py:class:: ContainerAffordanceBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ContainerAffordanceBinding
   :members:

.. py:class:: SupportSurfaceAffordanceBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: SupportSurfaceAffordanceBinding
   :members:

.. py:class:: ControlPartCommandPreset
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ControlPartCommandPreset
   :members:

.. py:class:: ControlPartEndpointBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ControlPartEndpointBinding
   :members:

.. py:class:: ControlPartResourceBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

.. autoclass:: ControlPartResourceBinding
   :members:

.. py:class:: SimulationRobotSkillProfileBinding
   :module: embodichain.lab.task_program.integrations.simulation.bindings
   :no-typesetting:
   :no-index-entry:

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

.. autodata:: VersionedKey
