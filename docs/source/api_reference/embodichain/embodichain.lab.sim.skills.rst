embodichain.lab.sim.skills
==========================

.. automodule:: embodichain.lab.sim.skills

   .. rubric:: Scene integration contracts

   .. autosummary::

      SceneRegistry
      RegistrySceneProvider
      SceneEntityRegistration
      SceneEntityRef
      SceneObjectRef
      SceneArticulationRef
      SceneLinkRef
      SceneAffordanceRef
      SceneEntityStateProvider
      SceneGeometryProvider
      SceneDynamics
      SceneCollisionRole
      SceneCollisionWorldMode

   .. rubric:: Robot skill profiles

   .. autosummary::

      RobotSkillProfile
      BoundRobotSkillProfile
      RobotResource
      ResourceEndpoint
      ResourceEndpointAdapter
      EndpointResolution
      ControlPartEndpoint
      ControlPartEndpointAdapter
      ResourceBinding
      ResolvedResourceEndpoint
      ResolvedRobotResource
      ResolvedSkillBinding
      ResourceClaim
      SkillPolicyPreset
      ProfileValidationError
      UnsupportedSkillError
      AmbiguousSkillBindingError

   .. rubric:: Semantic calls and runtime

   .. autosummary::

      SemanticCallSpec
      SemanticPose
      Pick
      Place
      HandOver
      OperateArticulation
      RegisteredSemanticCall
      SemanticCallCatalog
      SemanticSkillCompiler
      AtomicSkills
      SkillRuntime
      SkillResult
      SkillCallTrace
      SkillPlanAttemptTrace
      SkillEffectTrace

   .. rubric:: Effects, evidence, and parallel execution

   .. autosummary::

      SemanticEffectSpec
      EffectMonitorRef
      EffectMonitor
      EffectEvidenceCollector
      ParallelSkillRuntime
      ParallelSkillResult
      ParallelCommandSafetyValidator

   .. rubric:: Additional public contracts

   .. autosummary::

      ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY
      AmbiguousSceneAffordanceError
      AnalyzedSemanticCall
      ArticulationJointEvidenceAddress
      ArticulationJointObservationCallback
      ArticulationJointStateExpectation
      BinaryEffectClause
      BinaryEffectEvidenceBatch
      BinaryEffectEvidenceQuery
      BinaryEffectObservation
      BinaryEvidenceKind
      BinaryObservationCallback
      BoundSemanticCall
      BoundSemanticIntegration
      COMPOSITE_EFFECT_MONITOR_ID
      COMPOSITE_EFFECT_MONITOR_REVISION
      CONSTRAINT_EFFECT_CHANNEL
      CONTACT_EFFECT_CHANNEL
      CONTROL_PART_EVIDENCE_PROVIDER_ID
      CONTROL_PART_EVIDENCE_PROVIDER_REVISION
      CompositeEffectMonitor
      CompositeEffectMonitorCfg
      CompositeEffectMonitorFactory
      ControlPartEvidenceAddress
      ControlPartRobotEvidenceSource
      ControlPartSimulationEvidenceProvider
      CoordinatedHeldObjectCleanupExpectation
      DeclarativeValue
      EffectClause
      EffectEvidenceAddress
      EffectEvidenceBatch
      EffectEvidenceCollectionContext
      EffectEvidenceCollectorPort
      EffectEvidenceProvider
      EffectEvidenceProviderRegistry
      EffectEvidenceQuery
      EffectEvidenceQueryValue
      EffectEvidenceSourceRef
      EffectMonitorDecision
      EffectMonitorFactory
      EffectMonitorParam
      EffectMonitorRegistry
      EffectStateExpectation
      FORCE_EFFECT_CHANNEL
      GRASP_AFFORDANCE_CAPABILITY
      GroundedSemanticCall
      HandOverPoseProvider
      HandOverPoseTargets
      HeldObjectRelation
      HeldObjectStateExpectation
      JOINT_STATE_EFFECT_CHANNEL
      JointStateEffectClause
      JointStateEvidenceBatch
      JointStateEvidenceQuery
      JointStateObservation
      LinkedSemanticCall
      PLACE_IN_AFFORDANCE_CAPABILITY
      PLACE_ON_AFFORDANCE_CAPABILITY
      POSE_RELATION_EFFECT_CHANNEL
      ParallelBarrierUpdate
      ParallelBranchPlan
      ParallelBranchRuntime
      ParallelBranchStaticAnalysis
      ParallelConflictError
      ParallelLaneCommandSink
      ParallelRuntimeBranch
      ParallelSafetyError
      ParallelStateConflictError
      ParallelTimingError
      ParallelTimingPolicy
      PathPart
      PlaceRelationTarget
      PoseRelationClause
      PoseRelationEvidenceBatch
      PoseRelationEvidenceQuery
      PoseRelationExpectation
      RegisteredSemanticLowerer
      RelationTargetGrounder
      ResolvedCorePolicyTrace
      SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID
      SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION
      ScalarEffectClause
      ScalarEffectEvidenceBatch
      ScalarEffectEvidenceQuery
      ScalarEffectObservation
      ScalarEvidenceKind
      ScalarExpectation
      ScalarObservationCallback
      SceneArticulationEvidenceProvider
      SceneArticulationJointStateProvider
      SceneEntityManifest
      SceneEntityMetadata
      SceneManifest
      SemanticCallDescriptor
      SemanticDiagnostic
      SemanticEffectDependency
      SemanticEffectKind
      SemanticHandOverTarget
      SemanticIntegrationManifest
      SemanticLowering
      SemanticObjectTarget
      SemanticRelationTarget
      SemanticValidationError
      SemanticWorkflow
      SkillEndpointBindingTrace
      SkillFailure
      SkillRuntimeProvider
      SkillScene
      SkillStatus
      SymbolicStateDomain
      SymbolicStateKey
      UnsupportedSceneAffordanceError
      align_parallel_commands
      analyze_parallel_branches
      build_effect_evidence_queries
      builtin_semantic_call_catalog
      merge_parallel_effects
      resolve_parallel_barrier
      task_state_to_metadata
      validate_parallel_claims

.. currentmodule:: embodichain.lab.sim.skills

Robot resources and profiles
----------------------------

.. autoclass:: RobotSkillProfile
   :members:

.. autoclass:: BoundRobotSkillProfile
   :members:

.. autoclass:: RobotResource
   :members:

.. autoclass:: ResourceEndpoint
   :members:

.. autoclass:: ResourceEndpointAdapter
   :members:

.. autoclass:: EndpointResolution
   :members:

.. autoclass:: ControlPartEndpoint
   :members:

.. autoclass:: ControlPartEndpointAdapter
   :members:

.. autoclass:: ResourceBinding
   :members:

.. autoclass:: ResolvedResourceEndpoint
   :members:

.. autoclass:: ResolvedRobotResource
   :members:

.. autoclass:: ResolvedSkillBinding
   :members:

.. autoclass:: ResourceClaim
   :members:

.. autoclass:: SkillPolicyPreset
   :members:

Profile errors
--------------

.. autoclass:: ProfileValidationError

.. autoclass:: UnsupportedSkillError

.. autoclass:: AmbiguousSkillBindingError

Semantic calls and runtime
--------------------------

.. autoclass:: SemanticCallSpec
   :members:

.. autoclass:: SemanticPose
   :members:

.. autoclass:: Pick
   :members:

.. autoclass:: Place
   :members:

.. autoclass:: HandOver
   :members:

.. autoclass:: OperateArticulation
   :members:

.. autoclass:: RegisteredSemanticCall
   :members:

.. autoclass:: SemanticCallCatalog
   :members:

.. autoclass:: SemanticSkillCompiler
   :members:

.. autoclass:: AtomicSkills
   :members:

.. autoclass:: SkillRuntime
   :members:

.. autoclass:: SkillResult
   :members:

.. autoclass:: SkillCallTrace
   :members:

.. autoclass:: SkillPlanAttemptTrace
   :members:

.. autoclass:: SkillEffectTrace
   :members:

Effects, evidence, and parallel execution
-----------------------------------------

.. autoclass:: SemanticEffectSpec
   :members:

.. autoclass:: EffectMonitorRef
   :members:

.. autoclass:: EffectMonitor
   :members:

.. autoclass:: EffectEvidenceCollector
   :members:

.. autoclass:: ParallelSkillRuntime
   :members:

.. autoclass:: ParallelSkillResult
   :members:

.. autoclass:: ParallelCommandSafetyValidator
   :members:

Registry and provider
---------------------

.. autoclass:: SceneRegistry
   :members:

.. autoclass:: RegistrySceneProvider
   :members:

Registration contracts
----------------------

.. autoclass:: SceneEntityRegistration
   :members:

.. autoclass:: SceneEntityStateProvider
   :members:

.. autoclass:: SceneGeometryProvider
   :members:

References and enums
--------------------

.. autoclass:: SceneEntityRef
   :members:

.. autoclass:: SceneObjectRef
   :members:

.. autoclass:: SceneArticulationRef
   :members:

.. autoclass:: SceneLinkRef
   :members:

.. autoclass:: SceneAffordanceRef
   :members:

.. autoclass:: SceneDynamics
   :members:

.. autoclass:: SceneCollisionRole
   :members:

.. autoclass:: SceneCollisionWorldMode
   :members:
