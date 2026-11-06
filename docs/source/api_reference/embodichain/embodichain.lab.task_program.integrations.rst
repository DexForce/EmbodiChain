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
      DeclaredAssetFileProvider,
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
      PostPolicyPresetBinding, BoundPostPolicyPresets,

      ConfiguredTaskProgramDeployment, ConfiguredTaskProgramIntegration,
      load_configured_task_program_deployment

   .. autosummary::

      AntipodalGraspAffordanceBinding
      InvocationPolicyBinding
      BoundInvocationPolicies
      PostPolicyPresetBinding
      BoundPostPolicyPresets
      task_program_fingerprint
      ConfiguredHandOverPoseProvider
      ConfiguredTaskProgramDeployment
      ConfiguredTaskProgramIntegration
      ContainerAffordanceBinding
      ControlPartCommandPreset
      ControlPartEvidenceProviderDeclaration
      ControlPartEvidenceProviderFactory
      DeclaredAssetFileProvider
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

.. autoclass:: DeclaredAssetFileProvider
   :members:

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

``declared_asset_files()`` resolves service-owned local assets without creating
an adapter, generator, sampler, robot, or simulator. Each grasp and hand
calibration factory must explicitly implement ``DeclaredAssetFileProvider``
when an inventory is requested. Configured antipodal factories delegate their
effective model and side to the GraspKit model owner; sampled hand calibration
borrows the physical robot's assets. Other registration extensions can declare
their assets through the same protocol. Results are sorted unique absolute
files. Physical declarations and motion-source manifests keep their own input
contracts; the asset protocol does not infer them from a factory's private
state. The configured adapter wrapper exposes the same public method.

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
   DeclaredAssetFileProvider
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

Catalog preflight checks each selected Place preset's waypoint qualification
against its bounded recovery variant, using the same scope rule as Atomic
planning. Contradictory combinations reject before runtime construction;
unselected preset templates retain their ordinary behavior.

Bindings targeting parallel branches are explicitly rejected. Unbound parallel
calls remain supported. The compiled-program fingerprint uses the same format as
nominal motion sources; it includes the instruction and integration selection.
Configured integrations may declare the same tuple under
``runtime_services.invocation_policy_bindings``. Its entries contain exactly
``program_fingerprint``, ``invocation_id``, and ``preset_id``. The public deployment
loader validates them against the compiled program before returning its lazy
factory. The bridge uses these sealed bindings; an explicit second binding tuple
is rejected. Named execution-policy ``presets`` patch policy groups, while
``profile.preset_options`` owns action-option and monitor additions for those
names. Neither path executes providers or mutates an engine.
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

.. currentmodule:: embodichain.lab.task_program.integrations

Post-policy occurrence preset bindings
--------------------------------------

``PostPolicyPresetBinding`` selects an integration-owned condition preset for one
exact ``program_id/segment_id:post:policy_index`` occurrence and complete compiled
program fingerprint. It retains the program's original entity/preset strings.
``BoundPostPolicyPresets`` snapshots exact compiled segments and validates
membership, changed declarations, duplicate/unknown occurrences and foreign
programs. Legal compiler-normalized scene aliases remain valid.

Configured ``runtime_services.post_policy_bindings`` is opt-in. Its current
simulation capability requires complete sequential control-operation coverage,
matching target/source operation and phase, one primary resource, and a terminal
condition only at the explicitly registered cleanup occurrence. Native child-link
identity stays in the declared control operation; no task ID, suffix or cached
analysis index selects runtime behavior. Runtime ports read pre-bound presets.
Empty tables retain existing defaults and registration fingerprints.

.. currentmodule:: embodichain.lab.task_program.integrations.invocation_policy

.. autoclass:: PostPolicyPresetBinding

.. autoclass:: BoundPostPolicyPresets
   :members: validate_program, preset_for


Configured transport targets
----------------------------

Named ``move_held_object`` routes declare canonical carried-object identities
and absolute or scene-relative target poses. They lower to the shared
``MoveHeldObject`` goal, retain attachment semantics and expose corresponding
Pick lookahead targets. Lowering reads no live state and performs no IK or
planning. Existing single-target configurations keep their factory and effect
contract. The route form is mutually exclusive with the legacy selector fields.

The routed-2 contract supplies the route's exact ``object_id`` through
``HeldObjectPoseGoal``. Grounding requires the selected resource to hold that
object in every eligible row before planning. A verified monitor is required;
projected presets reject before dispatch. The existing verified input relation
is guarded during ``transport`` without a new acquisition gate. Named transport
plans additionally request terminal retained-attachment verification. Observed
loss invalidates only the affected resource rows; unresolved terminal evidence
cannot complete those rows successfully. The legacy single-target form keeps its
original effectless default and registration fingerprint.

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



Scoped Twist observation and execution sessions
-----------------------------------------------

``TwistOperationSession`` binds exact declared occurrences, robot, program and
physical episode to one shared Atomic cursor and the real Motion query factory.
Planning, installation, adoption and feedback remain separate guarded boundaries.
The registry selects only exact declared occurrence IDs and implements the
existing plan-lifecycle factory; preview forks change cursor identity without
changing the borrowed physical owners.

``ControlObservationWindow`` projects only newly appended records from an exact
canonical ``JointContactObserver`` generation. Raw frames retain simulation time;
source-relative scalar time is explicitly declared. It preserves missing channel,
loss, capacity and cadence errors and never performs another sensor fetch.
``record_count`` reads the retained prefix length without copying that prefix.
These interfaces install no standard post-policy producer or control factory.

.. automodule:: embodichain.lab.task_program.integrations.simulation.control_observation
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.twist_session
   :members:

Declared articulation continuation and drawer services
------------------------------------------------------

Prismatic lowerers ground explicit source-qualified joint targets, withdrawal
frames and PARK commands into the shared skills. The continuation registry owns
actual ``ExecutionPlanAttempt`` installation history for exact declared
program/episode occurrences. A preview is not an installation and cannot grant
withdrawal or PARK recovery authority.

Drawer and preshape adapters validate original source stamps and native owners
around reads. They provide observed geometry and typed command provenance,
select no alternate skill registry, and import no Task Engine state. These
interfaces do not install the standard Press/Twist post-policy producer.

.. automodule:: embodichain.lab.task_program.integrations.simulation.prismatic_lowering
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.articulation_session
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.drawer_planning
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.slide_preshape
   :members:

Configured geometry planning
----------------------------

``runtime_services.planning_services`` is a closed, explicitly enabled geometry
service declaration. Composition snapshots and fingerprints it without starting
Sim, reading native state, performing IK or constructing collision queries.
It selects discrete input-mesh collision queries and optional native receiver
geometry from the shared Motion and Scene owners. Native receiver geometry
requires an ordered collision roster covering all declared rigid owners, actual
root observations, and Default contact revisions. The host checks backend and
batch eligibility before simulation; these geometry queries retain their B=1
scope. Unconfigured calls retain their defaults and registration fingerprints.

The observation provider reads the actual native source stamp before and after
capturing robot and scene state; drift rejects the observation. Native HandOver
preparation checks that full stamp, observed start, explicit receiver resources
and physical owner inventory. It captures a fresh borrowed-scene query whose
Atomic caller closes. A closed factory cannot authorize further native reads.
HandOver's bounded recovery is selected explicitly by its typed options and
``handover_bounded`` declaration. Unpublished planning-service fields fail the
closed decoder; this interface does not imply that every source capability is
already available from configuration.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.planning_services

.. autosummary::

   CollisionSceneRosterCfg
   SimulationPlanningServicesCfg

.. autoclass:: CollisionSceneRosterCfg
   :members: snapshot

.. autoclass:: SimulationPlanningServicesCfg
   :members: snapshot, validate_num_envs, validate_physics

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.handover_scene

.. autosummary::

   SimulationHandOverSceneProvider

.. autoclass:: SimulationHandOverSceneProvider
   :members: prepare


Registered motion aliases
-------------------------

The closed registered-lowerer declarations ``align_held``, ``clear_released`` and
``idle_arm_clearance`` retain the existing call IDs ``gen_sim.align_held``,
``gen_sim.clear_released`` and ``gen_sim.clear_idle_arm``. They bind explicit route,
resource and geometry operands into ordinary ``MoveHeldObject``,
``MoveEndEffector`` and ``MoveJoints`` goals. Numerical target construction belongs
to Compute; the Atomic and Motion owners plan and qualify emitted commands.

Held alignment requires the declared object to be exclusively held and currently
observed. Its optional retention contract requests verification rather than
establishing a physical fact. Released clearance requires an empty resource and
exact hand-model reserves. Idle clear/restore routes bind an ordered pair of
sequential segments, their resource and a declared joint offset; both calls
require explicit discrete free-motion qualification. The original shared
``simulation.park`` call remains the restore entry point.

Configured linear Cartesian modes require their existing fixed-grid/sequential
IK policies and B=1 scope. Missing services or incompatible policy values reject
before simulation. Bounded staging and compiler-bound released continuations
remain unavailable until their runtime providers are installed; ordinary calls
retain their policies and fingerprints.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.held_alignment

.. autosummary::

   HeldAlignmentRoute
   HeldAlignmentLowererFactory

.. autoclass:: HeldAlignmentRoute
   :members: selector

.. autoclass:: HeldAlignmentLowererFactory
   :members: create, validate_scene_declaration

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.release_clearance

.. autosummary::

   ReleasedHandRoute
   NominalHandBinding
   ClearReleasedLowererFactory

.. autoclass:: ReleasedHandRoute

.. autoclass:: NominalHandBinding

.. autoclass:: ClearReleasedLowererFactory
   :members: create, validate_scene_declaration

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.idle_clearance

.. autosummary::

   IdleClearanceRoute
   IdleClearanceFactory

.. autoclass:: IdleClearanceRoute

.. autoclass:: IdleClearanceFactory
   :members: create

Declared articulation continuation and drawer services
------------------------------------------------------

Explicit runtime observation scope
----------------------------------

``TaskProgramEnvironmentAdapter.assemble_runtime`` and ``create_bridge`` accept
an optional ``runtime_scope_guard`` for a host-owned session, episode or physical
owner epoch. The standard factory forwards it to the planning observation port
and shared scene cache. ``validate_runtime_scope()`` checks factory lifetime and
this explicit authority without acquiring robot/sensor state or stepping physics.
A scope change during acquisition rejects the result. Cached qpos also rejects
closed factory resources; a live factory does not imply that a previous bridge
still has authority.

The default ``None`` keeps legacy custom factory calls unchanged: no new keyword
is passed to them. Explicit scope requests require a factory supporting the
optional keyword. The validator must not acquire observations or mutate physical
state. Control-service construction and evidence collection remain separate.


Scoped control producers and runtime-bound lowerers
---------------------------------------------------

The standard factory creates a fresh operation for every compiled bridge, using
its real program fingerprint, initialized episode, contact sensor and acknowledged
clock. The same operation is supplied to the registered lowerer and physical
post-policy. Ordinary factories continue using ``create``; the optional runtime
factory protocol explicitly requests the bridge's actual service context.

Press preserves observed preparation holds, source tip geometry, bounded physical
release windows and terminal receipts. Twist shares one cursor between Atomic
planning, installed-plan lifecycle and measured chunk feedback. Prismatic
continuation registries reuse the published Atomic recovery owners. This assembly
keeps B1 source qualification and rejects external nominal plans or expansion for
control bridges. Software assembly checks do not establish physical qualification.

.. currentmodule:: embodichain.lab.task_program.integrations.extensions

.. autoclass:: RuntimeBoundSemanticLowererFactory
   :members:

.. automodule:: embodichain.lab.task_program.integrations.runtime_services
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.control_assembly
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.control_lowering
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.control_policy
   :members:
   :exclude-members: ControlOperationRuntime

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.control_policy

.. autoclass:: ControlOperationRuntime
   :members:
   :exclude-members: observation_scope

.. py:attribute:: ControlOperationRuntime.observation_scope
   :type: dict[str, builtins.object]

   Owned JSON provenance for the declared program, episode, operation, native
   actors, observer and clock. Reading this metadata performs no acquisition;
   the declarations do not establish current physical evidence availability.

Control observers opt into retaining the actual before-fetch source stamp. Their
``capture(refresh=False)`` reuses the completed buffer; this does not add a sensor
refresh or repeat the post-policy's physical acquisition. Bridge cleanup detaches
owned hooks and closes cursors/queries without acknowledging pending commands.
The borrowed observation owner remains available for final collectors until host
teardown, while a changed episode invalidates that bridge's live observations.


Native planning deployment services
-----------------------------------

The standard factory resolves explicit planning declarations after the physical
host exists. It supplies shared collision, observed cuRobo, Pick closing, measured
rigid, Place-scene, held staging/command, Pour projection and paired-grasp services
to the ordinary Atomic engine. Static composition creates none of those live
queries. Default service absence retains the existing shared behavior.

Every occurrence permission is checked against the actual compiled program and
selected preset. Place/released-hand continuations bind the same existing
compiler, engine, robot and program; they use the normal complete-plan preview
pipeline. Released authorities additionally guard the actual episode and close
through the bridge's existing cleanup callback. They never create a second
compiler or commit a preview's physical effects.

Public inline composition accepts explicit declarations under ``inline`` and
retains independently owned program data. ``program_source`` and ``load_program``
provide the same strict decoding/preflight for file and inline sources.
``compile_configured_task_program`` computes provider-free occurrence identities
before their final runtime-service bindings exist; full deployment preflight is
still required afterwards. Backend/batch, missing services, incompatible modes,
foreign roles, external nominal incompatibilities and declaration drift reject
without starting simulation. Measurements and physical acceptance remain with
their separate evidence and TaskSpec owners.

.. autofunction:: embodichain.lab.task_program.integrations.composition.compile_configured_task_program

.. autoclass:: embodichain.lab.task_program.integrations.simulation.planning_services.ObservedCuroboServicesCfg
   :members: snapshot

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.collision_observation

.. autosummary::

   SimulationCollisionObservationProvider

.. autoclass:: SimulationCollisionObservationProvider
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.coordinated

.. autosummary::

   CoordinatedHoldLowererFactory
   CoordinatedTargetRoute
   CoordinatedPlacementLowererFactory

.. autoclass:: CoordinatedHoldLowererFactory
   :members:

.. autoclass:: CoordinatedTargetRoute
   :members:

.. autoclass:: CoordinatedPlacementLowererFactory
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.coordinated_grasp

.. autosummary::

   CoordinatedGraspServiceCfg
   ConfiguredCoordinatedGraspServices

.. autoclass:: CoordinatedGraspServiceCfg
   :members:

.. autoclass:: ConfiguredCoordinatedGraspServices
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.effect_contracts

.. autosummary::

   PlacementEffectContract
   TransportEffectContract

.. autoclass:: PlacementEffectContract
   :members:

.. autoclass:: TransportEffectContract
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.held_planning

.. autosummary::

   SimulationHeldPlanningProvider
   resolve_retained_command_authorities

.. autoclass:: SimulationHeldPlanningProvider
   :members:

.. autofunction:: resolve_retained_command_authorities


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.jaw_pour

.. autosummary::

   JawPourFactory

.. autoclass:: JawPourFactory
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.measured_rigid

.. autosummary::

   MeasuredRigidInvocationBinding
   SimulationMeasuredRigidProvider

.. autoclass:: MeasuredRigidInvocationBinding
   :members:

.. autoclass:: SimulationMeasuredRigidProvider
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.pick_closing

.. autosummary::

   PickClosingBinding
   SimulationPickClosingProvider

.. autoclass:: PickClosingBinding
   :members:

.. autoclass:: SimulationPickClosingProvider
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.pick_targets

.. autosummary::

   PickTargetRoute
   ConfiguredPickLowererFactory
   make_pick_alias_factory

.. autoclass:: PickTargetRoute
   :members:

.. autoclass:: ConfiguredPickLowererFactory
   :members:

.. autofunction:: make_pick_alias_factory


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.place_continuation

.. autosummary::

   PlaceContinuationBinding
   HeldAlignmentContinuationBinding
   SimulationPlaceContinuationProvider

.. autoclass:: PlaceContinuationBinding
   :members:

.. autoclass:: HeldAlignmentContinuationBinding
   :members:

.. autoclass:: SimulationPlaceContinuationProvider
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.place_scene

.. autosummary::

   PlaceSceneBinding
   SimulationPlaceSceneProvider

.. autoclass:: PlaceSceneBinding
   :members:

.. autoclass:: SimulationPlaceSceneProvider
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.placement

.. autosummary::

   ObservedRelativePlaceBinding
   ObservedRelativePlaceLowererFactory
   UprightPlaceLowererFactory

.. autoclass:: ObservedRelativePlaceBinding
   :members:

.. autoclass:: ObservedRelativePlaceLowererFactory
   :members:

.. autoclass:: UprightPlaceLowererFactory
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.pour_binding

.. autosummary::

   PourProjectionBinding
   resolve_pour_projection_routes

.. autoclass:: PourProjectionBinding
   :members:

.. autofunction:: resolve_pour_projection_routes


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.pour_projection

.. autosummary::

   ResolvedPourProjectionRoute
   SimulationPourProjectionProvider

.. autoclass:: ResolvedPourProjectionRoute
   :members:

.. autoclass:: SimulationPourProjectionProvider
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.pour_transport

.. autosummary::

   PourStageRoute
   PourTransportFactory
   PourRegionTransportFactory

.. autoclass:: PourStageRoute
   :members:

.. autoclass:: PourTransportFactory
   :members:

.. autoclass:: PourRegionTransportFactory
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.released_continuation

.. autosummary::

   ReleasedHandContinuationBinding
   SimulationReleasedHandContinuationProvider

.. autoclass:: ReleasedHandContinuationBinding
   :members:

.. autoclass:: SimulationReleasedHandContinuationProvider
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.retained_pour

.. autosummary::

   RetainedPourFactory

.. autoclass:: RetainedPourFactory
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.stack_placement

.. autosummary::

   StackPlaceLowererFactory

.. autoclass:: StackPlaceLowererFactory
   :members:


.. currentmodule:: embodichain.lab.task_program.integrations.simulation.transport

.. autosummary::

   TerminalAttachmentTransportLowererFactory
   DrawerTransportLowererFactory

.. autoclass:: TerminalAttachmentTransportLowererFactory
   :members:

.. autoclass:: DrawerTransportLowererFactory

Physical observation services for an actual execution
------------------------------------------------------

After binding the actual active bridge, call the standard adapter or simulation
factory's ``create_observation_services(environment, bridge, observations=...)``.
``SimulationObservationServices`` creates real collectors from the sealed
registration before the first execution step. Unselected services add no sensor
hook or acquisition. It reuses the public State/support collectors and borrowed
control runtime; cleanup closes only its contact hook and Cargo owner. Closed
service sets are released from the factory's active resource list. Factory
teardown also closes live service sets without closing the borrowed raw manager.

The actual environment-owned ObservationManager supplies selected raw pose fields,
exact inclusive windows and cheap per-row completed counts. Original validity,
acquisition errors, revisions and timestamps remain visible; missing finite raw
values or timestamps cannot become valid geometry. Neither an old trajectory
reader nor another policy/substep sampling chain is constructed.

``SimulationControlObservationCollector`` copies already completed exact compiled
policy occurrences, preserving window generations, physical channels, startup
windows and receipts. It performs no current-state fallback and can copy retained
facts after control hooks close. The supplied actual program/episode/native
identity remains separate from task interpretation.

.. automodule:: embodichain.lab.task_program.integrations.simulation.control_observations
   :members:

``SimulationRecordedContactCfg`` selects an existing ordinary substep-suite sensor,
complete single-sided native roster, 1024-contact query and bounded timeline.
The collector registers its observer before stepping and reads pose windows only
from the raw owner. Contact-only capture uses completed counts and copies no pose
history. Closing detaches only the exact owned subscription; replacement, reset,
episode and clock drift are rejected.

.. automodule:: embodichain.lab.task_program.integrations.simulation.recorded_contacts
   :members:

``SimulationCargoEnvelopeCfg`` selects source carriers and exclusions explicitly.
The Cargo collector reads initial source meshes once per required selection, then
checks complete original recorded prefixes and current terminal poses. Recorded
acquisition metadata retains the native simulation clock, while report boundaries
name the execution-step clock. The collector reports facts without evaluating a
TaskSpec, and imports no Task Engine implementation.

.. automodule:: embodichain.lab.task_program.integrations.simulation.cargo_observation
   :members:

.. automodule:: embodichain.lab.task_program.integrations.simulation.observation_services
   :members:

Explicit initial planning diagnostics
-------------------------------------

The configured probe uses the same adapter, compiler, occurrence policy and
Atomic engine as execution, analyzing only the first sequential window and
planning its first occurrence once. It never resets, steps or dispatches.
Process and environment RNG streams restore on success and error; borrowed
host adapters remain open and temporary bridge resources close. Probe-only
nominal sources and a parallel initial window are rejected explicitly.

``create_bridge(initial_plan_observer=...)`` instead emits owned proposal data
from the actual installed first complete plan before dispatch. The callback is
explicit application code, not a declaration or a plan transform. With None,
ordinary creation, nominal support and execution retain their existing paths.
``task_success`` remains None in both forms.

.. currentmodule:: embodichain.lab.task_program.integrations.simulation.planning_probe

.. autofunction:: probe_configured_task_program_initial_plan
