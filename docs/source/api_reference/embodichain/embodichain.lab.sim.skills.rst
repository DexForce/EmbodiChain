embodichain.lab.sim.skills
==========================

.. automodule:: embodichain.lab.sim.skills

   .. rubric:: Semantic calls and catalog

   .. autosummary::

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
      SemanticSkillRuntime
      SemanticTask
      SemanticExecution
      SemanticExecutionStatus
      SemanticTaskStatus
      SemanticExecutionStep
      SemanticCallRecord
      SemanticSegmentResult
      SemanticTaskResult

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

.. currentmodule:: embodichain.lab.sim.skills

Semantic calls and catalog
--------------------------

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

.. autoclass:: SemanticSkillRuntime
   :members:

.. autoclass:: SemanticTask
   :members:

.. autoclass:: SemanticExecution
   :members:

.. autoclass:: SemanticExecutionStatus
   :members:

.. autoclass:: SemanticTaskStatus
   :members:

.. autoclass:: SemanticExecutionStep
   :members:

.. autoclass:: SemanticCallRecord
   :members:

.. autoclass:: SemanticSegmentResult
   :members:

.. autoclass:: SemanticTaskResult
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
