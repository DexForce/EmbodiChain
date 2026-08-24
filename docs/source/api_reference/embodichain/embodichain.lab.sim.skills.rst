embodichain.lab.sim.skills
==========================

.. automodule:: embodichain.lab.sim.skills
   :members:
   :no-index:

   .. rubric:: Semantic calls and catalog

   .. autosummary::

      SemanticCallSpec
      SemanticPose
      Pick
      Place
      HandOver
      RegisteredSemanticCall
      SemanticCallDescriptor
      SemanticCallCatalog
      builtin_semantic_call_catalog

   .. rubric:: Semantic compilation and grounding

   .. autosummary::

      SemanticWorkflow
      SemanticLowering
      GroundedSemanticCall
      SemanticObjectTarget
      SemanticRelationTarget
      RegisteredSemanticLowerer
      RelationTargetGrounder
      HandOverPoseTargets
      HandOverPoseProvider
      SemanticSkillCompiler

   .. rubric:: Semantic integration and execution

   .. autosummary::

      SceneEntityManifest
      SceneManifest
      SemanticIntegrationManifest
      SemanticDiagnostic
      SemanticValidationError
      SemanticEffectVerifier
      AtomicSkills
      SkillRuntime
      SkillResult
      SkillStatus
      SkillCallTrace
      SkillPlanAttemptTrace
      SkillEffectTrace
      SkillFailure
      SkillScene

   .. rubric:: Scene integration contracts

   .. autosummary::

      SceneRegistry
      RegistrySceneProvider
      SceneEntityRegistration
      SceneEntityMetadata
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
      GRASP_AFFORDANCE_CAPABILITY
      PLACE_ON_AFFORDANCE_CAPABILITY
      PLACE_IN_AFFORDANCE_CAPABILITY
      UnsupportedSceneAffordanceError
      AmbiguousSceneAffordanceError

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

   .. rubric:: Effects, evidence, and parallel execution

   .. autosummary::

      SemanticEffectSpec
      EffectMonitorRef
      EffectMonitor
      EffectEvidenceCollector
      ParallelSkillRuntime
      ParallelSkillResult
      ParallelCommandSafetyValidator

.. currentmodule:: embodichain.lab.sim.skills

Semantic calls and catalog
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

.. autoclass:: RegisteredSemanticCall
   :members:

.. autoclass:: SemanticCallDescriptor
   :members:

.. autoclass:: SemanticCallCatalog
   :members:

.. autofunction:: builtin_semantic_call_catalog

Semantic compilation and grounding
-----------------------------------

.. autoclass:: SemanticWorkflow
   :members:

.. autoclass:: SemanticLowering
   :members:

.. autoclass:: GroundedSemanticCall
   :members:

.. autoclass:: SemanticObjectTarget
   :members:

.. autoclass:: SemanticRelationTarget
   :members:

.. autoclass:: RegisteredSemanticLowerer
   :members:

.. autoclass:: RelationTargetGrounder
   :members:

.. autoclass:: HandOverPoseTargets
   :members:

.. autoclass:: HandOverPoseProvider
   :members:

.. autoclass:: SemanticSkillCompiler
   :members:

Semantic integration
--------------------

.. autoclass:: SceneEntityManifest
   :members:

.. autoclass:: SceneManifest
   :members:

.. autoclass:: SemanticIntegrationManifest
   :members:

.. autoclass:: SemanticDiagnostic
   :members:

.. autoclass:: SemanticValidationError
   :members:

Semantic runtime
----------------

.. autodata:: SemanticEffectVerifier

.. autoclass:: AtomicSkills
   :members:

.. autoclass:: SkillRuntime
   :members:

.. autoclass:: SkillResult
   :members:

.. autoclass:: SkillStatus
   :members:

.. autoclass:: SkillFailure
   :members:

.. autoclass:: SkillCallTrace
   :members:

.. autoclass:: SkillPlanAttemptTrace
   :members:

.. autoclass:: SkillEffectTrace
   :members:

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

.. autoclass:: SceneEntityMetadata
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

Affordance capabilities and errors
----------------------------------

.. autodata:: GRASP_AFFORDANCE_CAPABILITY

.. autodata:: PLACE_ON_AFFORDANCE_CAPABILITY

.. autodata:: PLACE_IN_AFFORDANCE_CAPABILITY

.. autoclass:: UnsupportedSceneAffordanceError

.. autoclass:: AmbiguousSceneAffordanceError
