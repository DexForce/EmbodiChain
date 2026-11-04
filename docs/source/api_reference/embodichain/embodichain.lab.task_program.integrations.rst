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
      SimulationSegmentPolicyPort, create_simulation_task_program_adapter,
      InvocationPolicyBinding, BoundInvocationPolicies, task_program_fingerprint,

      ConfiguredTaskProgramDeployment, ConfiguredTaskProgramIntegration,
      load_configured_task_program_deployment

   .. autosummary::

      AntipodalGraspAffordanceBinding
      InvocationPolicyBinding
      BoundInvocationPolicies
      task_program_fingerprint
      ConfiguredHandOverPoseProvider
      ConfiguredTaskProgramDeployment
      ConfiguredTaskProgramIntegration
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
      load_configured_task_program_deployment

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
   embodichain.lab.task_program.integrations.simulation

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

``capture_root_pose=True`` includes an owned arena-local robot root transform
in each synchronized planning observation. The default leaves it unset and
does not query root poses. ``hold_qpos_source="observed"`` preserves the default
captured-position holds without another robot read. Explicit ``drive_target``
mode reads controller targets for unaddressed joints; safe stops still use
measured positions from their planning context.

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

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.policies

``SimulationSettleMonitorCfg`` adds an explicit measurement mode to the existing
dynamic settle thresholds and cadence. Use it in a registration's
``settle_presets`` table. ``pose_delta`` requires an explicitly bound native
rigid object and the authoritative control ``step_dt``; provider-free catalog
preflight rejects articulation targets. The first observation has no previous
pose and remains unresolved. Inactive rows and timeout behavior retain the
shared monitor contract.

Plain ``DynamicSettleMonitorCfg`` values and the built-in preset table keep
reported velocities and their existing identity/metadata. Names such as
``contained_rigid_object`` do not select measurement behavior or install a preset.
Configured JSON declarations must be decoded by their integration owner; this
type does not register a second deployment configuration source.

.. autosummary::

   SimulationSettleMonitorCfg

.. autoclass:: SimulationSettleMonitorCfg
   :members:

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

Occurrence policy bindings
--------------------------

An ``InvocationPolicyBinding`` selects an existing named profile preset for one
sequential execution occurrence, identified by the complete compiled-program
fingerprint and ``program_id/segment_id:segment_call_index``. Pass a tuple of these
declarations to ``TaskProgramEnvironmentAdapter.create_bridge`` through its
``invocation_policy_bindings`` keyword. Normal calls use their existing policies
when the tuple is empty.

The adapter validates identities and selections before execution. The compiler
links and analyzes selected presets before constructing the final Atomic
invocation, so repeated calls and cached analyses retain distinct selections.
Selected presets must retain the original effect assurance. A selected preset's
existing runner settings own that occurrence; unselected calls continue to use
the host's runner override. Re-grounding preserves the original occurrence and
selection. Derived workflow-recovery calls keep independent IDs and ordinary
preset selection.

Bindings targeting parallel branches are explicitly rejected. Unbound parallel
calls remain supported. The compiled-program fingerprint uses the same format as
nominal motion sources; it includes the instruction and integration selection.
These declarations do not add YAML fields, execute providers, or mutate an engine.
``validate_manifest`` checks profile identity, runtime-preset selection and each
selected call's policy contract. A scene manifest carries entity metadata rather
than the host's scene-registry selection ID; the standard adapter validates that
selection, and hosts constructing a bridge directly retain that responsibility.

.. currentmodule:: embodichain.lab.task_program.integrations.invocation_policy

.. autosummary::

   InvocationPolicyBinding
   BoundInvocationPolicies
   task_program_fingerprint

.. autoclass:: InvocationPolicyBinding

.. autoclass:: BoundInvocationPolicies
   :members: validate_program, validate_manifest

.. autofunction:: task_program_fingerprint

Configured transport targets
----------------------------

Named ``move_held_object`` routes declare canonical carried-object identities
and absolute or scene-relative target poses. They lower to the shared
``MoveHeldObject`` goal, retain attachment semantics and expose corresponding
Pick lookahead targets. Lowering reads no live state and performs no IK or
planning. Existing single-target configurations keep their factory and effect
contract. The route form is mutually exclusive with the legacy selector fields.

.. code-block:: yaml

   kind: move_held_object
   routes:
     - object_id: bottle
       target_id: over_cup
       pose:
         kind: scene_entity
         entity_id: cup

The optional ``relative_pose`` uses a flattened entity-local SE(3) matrix;
``world_orientation`` optionally supplies a world-frame 3x3 rotation. Absolute
poses use ``kind: pose``, ``position`` and ``quaternion_xyzw``. Every conversion
owns its runtime tensors; declarations remain immutable and fingerprinted.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.targets

.. autoclass:: AbsolutePoseTarget
   :members:

.. autoclass:: SceneEntityTarget
   :members:

.. autodata:: ConfiguredPoseTarget

.. autofunction:: resolve_pose_target

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.transport

.. autoclass:: HeldObjectTransportRoute
   :members:

.. autoclass:: RoutedMoveHeldObjectLowererFactory
   :members:

Declaration-only lowerer validation
----------------------------------

Factories may implement ``RegisteredSemanticLowererSceneValidator`` to reject
missing or ambiguous references against the immutable scene manifest during
registration. The routed transport factory uses this optional contract;
existing factories remain valid and keep their fingerprints. Implementations
must not construct resources or read live state during this preflight.

.. currentmodule:: embodichain.lab.task_program.integrations

.. autosummary::

   RegisteredSemanticLowererSceneValidator

.. currentmodule:: embodichain.lab.task_program.integrations.extensions

.. autoclass:: RegisteredSemanticLowererSceneValidator
   :members:

