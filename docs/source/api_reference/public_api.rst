Public API Supplement
=====================

.. This page is maintained by the update-api-docs agent skill.
   docs/scripts/check_api_docs.py reads it but never writes it.

This page lists module exports declared through ``__all__`` that are not
covered by a more focused API-reference page. Prefer curated pages for APIs
that need deeper explanations or examples. Sphinx obtains signatures and
summaries here from the canonical Python docstrings.

embodichain.agents.mllm.expert_program
--------------------------------------

.. currentmodule:: embodichain.agents.mllm.expert_program

.. autosummary::

   compile_mllm_expert_program
   decode_mllm_expert_program

embodichain.data.assets.planner_assets
--------------------------------------

.. currentmodule:: embodichain.data.assets.planner_assets

.. autosummary::

   download_neural_planner_checkpoint

embodichain.data.assets.solver_assets
-------------------------------------

.. currentmodule:: embodichain.data.assets.solver_assets

.. autosummary::

   download_neural_ik_checkpoint

embodichain.data_pipeline.depth_video
-------------------------------------

.. currentmodule:: embodichain.data_pipeline.depth_video

.. autosummary::

   DEFAULT_DEPTH_MIN
   DEFAULT_DEPTH_MAX
   DEFAULT_DEPTH_SHIFT
   DEFAULT_DEPTH_USE_LOG
   DEFAULT_DEPTH_PIX_FMT
   DEPTH_METER_UNIT
   DEPTH_MILLIMETER_UNIT
   DEPTH_QMAX

embodichain.gen_sim.scene_engine.core.scene_edit_plan
------------------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.core.scene_edit_plan

.. autosummary::

   SceneEditOperation
   SceneEditPlan

embodichain.gen_sim.scene_engine.core.scene_graph
--------------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.core.scene_graph

.. autosummary::

   GENERATED_SCENE_GRAPH_SCHEMA
   GeneratedSceneGraph
   GeneratedSceneNode
   GeneratedSceneRelation
   OrientationState
   PlanarRelationType
   SceneConstraintType
   SupportRelationType
   TABLE_OBJECT_ID
   TABLE_REGIONS
   TableRegion

embodichain.gen_sim.scene_engine.core.scene_object
---------------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.core.scene_object

.. autosummary::

   ObjectPhysics
   SceneObject

embodichain.gen_sim.scene_engine.errors
-----------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.errors

.. autosummary::

   SceneServiceError

embodichain.gen_sim.scene_engine.pipeline
-----------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline

.. autosummary::

   SCENE_BLUEPRINT_SCHEMA
   SCENE_EDIT_BLUEPRINT_SCHEMA
   SceneBlueprintPackage
   SceneEditBlueprintPackage
   SceneMaterialization
   analyze_edit
   analyze_image
   materialize_blueprint
   materialize_edit

embodichain.gen_sim.scene_engine.pipeline.api
---------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.api

.. autosummary::

   SCENE_BLUEPRINT_SCHEMA
   SCENE_EDIT_BLUEPRINT_SCHEMA
   SceneBlueprintPackage
   SceneEditBlueprintPackage
   SceneMaterialization
   analyze_edit
   analyze_image
   materialize_blueprint
   materialize_edit

embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_asset_preparation
-----------------------------------------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_asset_preparation

.. autosummary::

   prepare_scene_edit_assets

embodichain.gen_sim.simready_pipeline.cli.start
-----------------------------------------------

.. currentmodule:: embodichain.gen_sim.simready_pipeline.cli.start

.. autosummary::

   cli_ingest_single
   main

embodichain.lab.gym.envs.base_env
---------------------------------

.. currentmodule:: embodichain.lab.gym.envs.base_env

.. autosummary::

   BaseEnv
   EnvCfg

embodichain.lab.gym.envs.demo
-----------------------------

.. currentmodule:: embodichain.lab.gym.envs.demo

.. autosummary::

   DEMO_ANNOTATION_KEYS
   DEMO_SCHEMA_VERSION
   DemoEpisodeResult
   DemoSegment
   DemoSegmentResult
   ProcessedEnvAction
   execute_demo_episode
   resolve_demo_segments

embodichain.lab.gym.envs.embodied_env
-------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.embodied_env

.. autosummary::

   EmbodiedEnvCfg
   EmbodiedEnv

embodichain.lab.gym.envs.expert_program.bridge
------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.bridge

.. autosummary::

   AcceptedRuntimeCommandObserver
   AtomicDemoBridge
   BufferedGymCommandSink
   CompiledProgramPort
   CurrentQposProvider
   DemoBridgeError
   EnvironmentStepClock
   EnvironmentStepTimingError
   GymPlanningObservationProvider
   JointPositionGymTransportEncoder
   ParallelCommandSafetyValidator
   RuntimeCommandFrameEncoder
   RuntimeTransportActionEncoder
   SegmentPostPolicyMetadataPort
   SegmentPostPolicyPort
   SegmentPostPolicyResultPort
   SegmentValidatorMetadataPort
   SegmentValidatorPort
   SequentialSkillRuntimePort
   UnsupportedRuntimeTransportError

embodichain.lab.gym.envs.expert_program.catalog
-------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.catalog

.. autosummary::

   ExpertProgramIntegrationCatalog
   IntegrationFingerprintMismatch
   SimulationExpertProgramRegistration

embodichain.lab.gym.envs.expert_program.cfg
---------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.cfg

.. autosummary::

   BarrierCfg
   CyclicPoseTargetCfg
   DeclarativeCfgValue
   EXPERT_PROGRAM_SCHEMA_VERSION
   EXPERT_PROGRAM_SCHEMA_VERSION_V2
   ExpertProgramCfg
   ExpertProgramIntegrationCfg
   HandOverCfg
   InvokeCfg
   MAX_DECLARATIVE_DEPTH
   MAX_DECLARATIVE_NODES
   MAX_EXPANDED_CALLS
   MAX_PROGRAM_DEPTH
   MAX_PROGRAM_NODES
   MAX_REPEAT_COUNT
   ObjectNearTargetValidatorCfg
   OperateArticulationCfg
   ParallelCfg
   PickCfg
   PlaceCfg
   PoseCfg
   PostPolicyCfg
   ProgramNodeCfg
   RegisteredSemanticCallCfg
   RepeatCfg
   SUPPORTED_EXPERT_PROGRAM_SCHEMA_VERSIONS
   SegmentCfg
   SemanticCallCfg
   SequenceCfg
   TargetCfg
   TargetRefCfg
   ValidatorCfg
   WaitStablePostCfg

embodichain.lab.gym.envs.expert_program.compiler
--------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.compiler

.. autosummary::

   CompiledBarrier
   CompiledParallelBlock
   CompiledParallelBranch
   CompiledPostPolicy
   CompiledProgram
   CompiledProgramAnalysis
   CompiledProgramCall
   CompiledProgramSegment
   CompiledProgramValidator
   CompiledRepeatFrame
   CompiledTargetSelection
   ExpertProgramCompileError
   ExpertProgramCompiler
   ExpertProgramSceneResolver
   MaterializedCompiledProgram
   SceneRegistryProgramResolver

embodichain.lab.gym.envs.expert_program.decoder
-------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.decoder

.. autosummary::

   ConfigPath
   ConfigPathPart
   ExpertProgramConfigError
   ExpertProgramDecodeError
   ExpertProgramValidationContext
   ExpertProgramValidationError
   SceneReferenceRole
   decode_expert_program
   decode_semantic_call
   encode_semantic_call
   render_config_path
   validate_expert_program

embodichain.lab.gym.envs.expert_program.environment
-----------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.environment

.. autosummary::

   AcceptedRuntimeCommandObserverFactory
   ExpertProgramEnvironmentAdapter
   ExpertProgramEnvironmentFactory
   ExpertProgramEnvironmentMixin
   ExpertProgramRuntimeAssembly
   PlanningObservationPort
   SkillRuntimeAssemblyPort

embodichain.lab.gym.envs.expert_program.extensions
----------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.extensions

.. autosummary::

   EndpointAdapterDeclaration
   ParallelCommandSafetyValidatorFactory
   ParallelSafetyDeclaration
   RuntimeTransportDeclaration
   StandardExtensionDeclarations
   VersionedKey
   build_standard_extension_declarations
   declare_endpoint_adapter
   declare_parallel_safety_factory
   declare_runtime_transport
   validate_immutable_extension_declaration

embodichain.lab.gym.envs.expert_program.loader
------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.loader

.. autosummary::

   MAX_EXPERT_PROGRAM_BYTES
   load_expert_program
   loads_expert_program_json
   parse_expert_program_json

embodichain.lab.gym.envs.expert_program.simulation
----------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.simulation

.. autosummary::

   AntipodalGraspAffordanceBinding
   ArticulationOperationAffordanceBinding
   ArticulationOperationTargetBinding
   ControlPartCommandPreset
   ContainerAffordanceBinding
   ControlPartEndpointBinding
   ControlPartResourceBinding
   RobotResourceBinding
   SimulationArticulationBinding
   SimulationArticulationLinkBinding
   SimulationResourceEndpointBinding
   SimulationRigidObjectBinding
   SimulationRobotResourceBinding
   SimulationRobotSkillProfileBinding
   SimulationSceneBinding
   SupportSurfaceAffordanceBinding

embodichain.lab.gym.envs.expert_program.simulation_environment
----------------------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.simulation_environment

.. autosummary::

   ControlCommandStateEvidenceTracker
   MotionGeneratorFactory
   SharedTickSceneProvider
   SimulationExpertProgramEnvironment
   SimulationExpertProgramFactory
   SimulationPlanningObservationProvider
   create_simulation_expert_program_adapter

embodichain.lab.gym.envs.expert_program.simulation_policies
-------------------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.simulation_policies

.. autosummary::

   SimulationSegmentPolicyPort
   default_simulation_settle_presets

embodichain.lab.gym.envs.expert_program.simulation_parallel_safety
------------------------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.simulation_parallel_safety

.. autosummary::

   CuroboParallelCommandSafetyValidator
   CuroboParallelSafetyValidatorFactory

embodichain.lab.gym.envs.expert_program.simulation_handover
------------------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.simulation_handover

.. autosummary::

   ConfiguredHandOverPoseProvider

embodichain.lab.gym.envs.settling
---------------------------------

.. currentmodule:: embodichain.lab.gym.envs.settling

.. autosummary::

   DynamicSettleMonitor
   DynamicSettleMonitorCfg
   DynamicSettleSample
   DynamicSettleState

embodichain.lab.gym.envs.managers.action_manager
------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.action_manager

.. autosummary::

   ActionTerm
   ActionManager

embodichain.lab.gym.envs.managers.actions
-----------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.managers.actions

.. autosummary::

   DeltaQposTerm
   QposTerm
   QposDenormalizedTerm
   QposNormalizedTerm
   EefPoseTerm
   QvelTerm
   QfTerm

embodichain.lab.gym.envs.wrapper.replay
---------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.wrapper.replay

.. autosummary::

   ReplayWrapper

embodichain.lab.gym.utils
-------------------------

.. currentmodule:: embodichain.lab.gym.utils

.. autosummary::

   EnvProfiler
   EnvProfilerCfg
   capture_trajectory_state
   restore_trajectory_state

embodichain.lab.gym.utils.gym_utils
-----------------------------------

.. currentmodule:: embodichain.lab.gym.utils.gym_utils

.. autosummary::

   DEFAULT_MANAGER_MODULES
   add_env_launcher_args_to_parser
   assign_data_to_dict
   batch
   build_env_cfg_from_args
   cat_tensor_with_ids
   clip_and_scale_action
   config_to_cfg
   convert_observation_to_space
   dict_array_to_torch_inplace
   fetch_data_from_dict
   flatten_state_dict
   get_dtype_bounds
   get_manager_modules
   init_rollout_buffer_from_config
   init_rollout_buffer_from_gym_space
   map_qpos_to_eef_pose
   merge_args_with_gym_config
   register_manager_modules
   to_cpu_tensor
   to_tensor

embodichain.lab.gym.utils.profiler
----------------------------------

.. currentmodule:: embodichain.lab.gym.utils.profiler

.. autosummary::

   EnvProfilerCfg
   EnvProfiler

embodichain.lab.gym.utils.trajectory_state
------------------------------------------

.. currentmodule:: embodichain.lab.gym.utils.trajectory_state

.. autosummary::

   capture_trajectory_state
   restore_trajectory_state

embodichain.lab.scripts.analyze_workspace
-----------------------------------------

.. currentmodule:: embodichain.lab.scripts.analyze_workspace

.. autosummary::

   build_sim_cfg
   build_robot_cfg
   build_preset_robot_cfg
   build_analyzer_config
   preview_cache
   parse_args
   main
   cli

embodichain.lab.scripts.preview_asset
-------------------------------------

.. currentmodule:: embodichain.lab.scripts.preview_asset

.. autosummary::

   build_sim_cfg
   cli
   load_assets
   main
   preview

embodichain.lab.scripts.preview_joint_control
---------------------------------------------

.. currentmodule:: embodichain.lab.scripts.preview_joint_control

.. autosummary::

   ArticulationPreviewController

embodichain.lab.scripts.preview_lerobot_data
--------------------------------------------

.. currentmodule:: embodichain.lab.scripts.preview_lerobot_data

.. autosummary::

   EpisodePreview
   SegmentPreview
   build_episode_preview
   cli
   inspect_dataset
   main
   resolve_dataset_root

embodichain.lab.scripts.run_env
-------------------------------

.. currentmodule:: embodichain.lab.scripts.run_env

.. autosummary::

   cli
   generate_and_execute_action_list
   generate_function
   main
   preview

embodichain.lab.sim
-------------------

.. currentmodule:: embodichain.lab.sim

.. autosummary::

   VisualMaterialCfg
   VisualMaterial
   VisualMaterialInst
   ReuseSegmentState
   BatchEntity
   SimulationManager
   SimulationManagerCfg
   SIM_CACHE_DIR
   MATERIAL_CACHE_DIR
   CONVEX_DECOMP_DIR
   REACHABLE_XPOS_DIR

embodichain.lab.sim.atomic_actions
----------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions

.. autosummary::

   ActionPlanningServices
   Affordance
   AntipodalAffordance
   ArticulationJointState
   ArticulationOperationAffordance
   ArticulationOperationTarget
   AssembleAffordance
   BASE_POSE_CHANNEL
   BUILTIN_ACTION_TYPES
   CoordinatedPickmentOptions
   CoordinatedPlacementOptions
   DynamicCollisionMode
   EndpointTrackingChannelBinding
   EndpointTrackingFeedbackAddress
   EntityState
   EffectVerificationRequirement
   EffectVerificationResult
   EffectExpectationResult
   EffectVerifier
   ExecutionPlanAttempt
   FeedbackTerminalAcceptance
   GRASP_COMMAND
   InFlightTrackingPolicy
   JOINT_POSITION_CHANNEL
   JointPositionTrackingEvaluator
   JointPositionTrackingMetric
   JointPositionTrackingProjector
   JointPositionTrackingState
   HandOverOptions
   HeldObjectGuardVerifier
   InteractionPoints
   MoveEndEffectorOptions
   MoveHeldObjectOptions
   MoveJointsOptions
   ObjectActionGoal
   ObservedArticulationJointState
   OPEN_COMMAND
   OperateArticulation
   OperateArticulationGoal
   OperateArticulationOptions
   PickUpOptions
   PlaceOptions
   PhaseEffectGateVerifier
   PlanningContextTrackingFeedbackProvider
   PoseTrackingEvaluator
   PoseTrackingMetric
   PoseTrackingState
   PoseGoalValue
   RigidObjectSceneProvider
   RigidObjectSceneProviderCfg
   RunnerStepCallback
   SceneProvider
   SceneArticulationOperationGeometry
   SceneSnapshotSupplier
   TimedTerminalAcceptance
   TimedTrackingSequence
   TerminalAcceptance
   TrackingCommandProjector
   TrackingEvaluation
   TrackingEvaluatorRegistry
   TrackingFeedbackAddress
   TrackingFeedbackBatch
   TrackingFeedbackProvider
   TrackingFeedbackProviderRegistry
   TrackingFeedbackSourceRef
   TrackingFrame
   TrackingMetricCfg
   TrackingMetricEvaluator
   TrackingPolicy
   TrackingProjectorRef
   TrackingProjectorRegistry
   TrackingRuntime
   TrackingSetpoint
   TrackingState
   WHOLE_BODY_POSE_CHANNEL
   WholeBodyPoseTrackingEvaluator
   WholeBodyPoseTrackingMetric
   WholeBodyPoseTrackingState
   get_registered_actions
   register_action
   unregister_action

embodichain.lab.sim.atomic_actions.affordance
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.affordance

.. autosummary::

   Affordance
   AntipodalAffordance
   ArticulationOperationAffordance
   ArticulationOperationTarget
   SlideAffordance
   PressAffordance
   TwistAffordance
   InteractionPoints
   AssembleAffordance

embodichain.lab.sim.atomic_actions.bindings
-------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.bindings

.. autosummary::

   ActionBinding
   EndpointBinding
   JointPositionTarget
   RuntimeEndpointTarget

embodichain.lab.sim.atomic_actions.control
------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.control

.. autosummary::

   ActionControlOverrides
   ControlCommand
   ControlPartCommandProfile
   GRASP_COMMAND
   JointPositionCommand
   OPEN_COMMAND

embodichain.lab.sim.atomic_actions.core
---------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.core

.. autosummary::

   AtomicAction
   ObjectSemantics
   SkillDescriptor
   resolve_runtime_device

embodichain.lab.sim.atomic_actions.effects
------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.effects

.. autosummary::

   StateDelta

embodichain.lab.sim.atomic_actions.engine
-----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.engine

.. autosummary::

   AtomicActionEngine

embodichain.lab.sim.atomic_actions.execution
--------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.execution

.. autosummary::

   EffectVerificationRequest
   EffectVerificationResult
   EffectExpectationResult
   ExecutionEvent
   ExecutionEventKind
   ExecutionPlanAttempt
   ExecutionSession
   ExecutionStatus
   ExecutionTick
   HeldObjectGuardRequest
   HeldObjectGuardResult
   PhaseEffectGateRequest
   PhaseEffectGateResult

embodichain.lab.sim.atomic_actions.goals
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.goals

.. autosummary::

   ActionGoal
   ObjectActionGoal
   PoseGoalValue
   SceneArticulationOperationGeometry
   SceneEntityPose
   collect_scene_dependencies
   resolve_pose_goal
   validate_pose_goal
   validate_pose_tensor

embodichain.lab.sim.atomic_actions.invocation
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.invocation

.. autosummary::

   ActionInvocation
   ActionOptions
   GoalT
   OptionsT
   PhaseEffectGateRequirement
   ResolvedActionRequest

embodichain.lab.sim.atomic_actions.plans
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.plans

.. autosummary::

   ActionPlan
   CompiledTrajectory
   EffectVerificationRequirement
   ExecutionFeedbackMode
   PlannerDiagnostics
   TimedTrajectory
   TrajectorySegment
   normalize_success_mask

embodichain.lab.sim.atomic_actions.policies
-------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.policies

.. autosummary::

   DynamicCollisionMode
   MotionPolicy
   RecoveryPolicy

embodichain.lab.sim.atomic_actions.primitives
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.primitives

.. autosummary::

   BUILTIN_ACTION_TYPES
   OperateArticulation
   OperateArticulationGoal
   OperateArticulationOptions

embodichain.lab.sim.atomic_actions.primitives.operate_articulation
-----------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.primitives.operate_articulation

.. autosummary::

   OperateArticulation
   OperateArticulationGoal
   OperateArticulationOptions

embodichain.lab.sim.atomic_actions.requirements
-----------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.requirements

.. autosummary::

   BATCH_INVERSE_KINEMATICS_CAPABILITY
   CARTESIAN_POSE_CAPABILITY
   DisjointResourceSlots
   DisjointSlotEndpoints
   FORWARD_KINEMATICS_CAPABILITY
   GRASP_CAPABILITY
   INVERSE_KINEMATICS_CAPABILITY
   JOINT_POSITION_CAPABILITY
   SkillBindingContract
   SkillEndpointRequirement
   SkillResourceSlot

embodichain.lab.sim.atomic_actions.runner
-----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.runner

.. autosummary::

   CommandAckStatus
   CommandAcknowledgement
   CommandDispatch
   CommandOperation
   CommandSink
   EffectVerifier
   ExecutionClock
   ExecutionRunner
   ExecutionRunnerCfg
   HeldObjectGuardVerifier
   MonotonicExecutionClock
   ObservationProvider
   PhaseEffectGateVerifier
   RunnerStatus
   RunnerStep
   RunnerStepCallback

embodichain.lab.sim.atomic_actions.runtime
------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.runtime

.. autosummary::

   ActionPlanningServices

embodichain.lab.sim.atomic_actions.runtime_commands
---------------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.runtime_commands

.. autosummary::

   EndpointCommand
   JointPositionPayload
   RuntimeCommandFrame
   RuntimeCommandPayload
   TimedCommandSequence

embodichain.lab.sim.atomic_actions.scene
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.scene

.. autosummary::

   SceneProvider

embodichain.lab.sim.atomic_actions.sim_adapter
----------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.sim_adapter

.. autosummary::

   RigidObjectSceneProvider
   RigidObjectSceneProviderCfg
   SceneSnapshotSupplier
   SimulationExecutionAdapter

embodichain.lab.sim.atomic_actions.state
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.state

.. autosummary::

   EntityState
   ArticulationJointState
   CoordinatedHeldObjectState
   HeldObjectState
   ObservedArticulationJointState
   PlanningContext
   RobotObservation
   SceneSnapshot
   TaskState

embodichain.lab.sim.atomic_actions.tracking
-------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.tracking

.. autosummary::

   BASE_POSE_CHANNEL
   FeedbackTerminalAcceptance
   InFlightTrackingPolicy
   JOINT_POSITION_CHANNEL
   JointPositionTrackingEvaluator
   JointPositionTrackingMetric
   JointPositionTrackingProjector
   JointPositionTrackingState
   EndpointTrackingChannelBinding
   EndpointTrackingFeedbackAddress
   PlanningContextTrackingFeedbackProvider
   PoseTrackingEvaluator
   PoseTrackingMetric
   PoseTrackingState
   TerminalAcceptance
   TimedTerminalAcceptance
   TimedTrackingSequence
   TrackingChannelId
   TrackingCommandProjector
   TrackingEvaluation
   TrackingEvaluatorRegistry
   TrackingFeedbackAddress
   TrackingFeedbackBatch
   TrackingFeedbackProvider
   TrackingFeedbackProviderRegistry
   TrackingFeedbackSourceRef
   TrackingFrame
   TrackingMetricCfg
   TrackingMetricEvaluator
   TrackingPolicy
   TrackingProjectorRef
   TrackingProjectorRegistry
   TrackingRuntime
   TrackingSetpoint
   TrackingState
   WHOLE_BODY_POSE_CHANNEL
   WholeBodyPoseTrackingEvaluator
   WholeBodyPoseTrackingMetric
   WholeBodyPoseTrackingState

embodichain.lab.sim.atomic_actions.trajectory_ops
-------------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.trajectory_ops

.. autosummary::

   axis_translation_keyframes
   build_joint_plan_states
   build_pose_plan_states
   interpolate_hand_qpos
   interpolate_joint_trajectory
   resolve_joint_target
   resolve_pose_target
   split_three_segments
   to_full_robot_trajectory
   translate_pose_world

embodichain.lab.sim.atomic_actions.transports
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.transports

.. autosummary::

   EndpointCommandRouter
   EndpointCommandTransport

embodichain.lab.sim.objects.articulation
----------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.articulation

.. autosummary::

   ArticulationData
   Articulation

embodichain.lab.sim.objects.cloth_object
----------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.cloth_object

.. autosummary::

   ClothBodyData
   ClothObject
   ClothObjectCfg

embodichain.lab.sim.objects.constraint
--------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.constraint

.. autosummary::

   RigidConstraint

embodichain.lab.sim.objects.gizmo
---------------------------------

.. currentmodule:: embodichain.lab.sim.objects.gizmo

.. autosummary::

   Gizmo
   GizmoCfg

embodichain.lab.sim.objects.rigid_object
----------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.rigid_object

.. autosummary::

   RigidBodyData
   RigidObject
   RigidObjectCfg

embodichain.lab.sim.objects.rigid_object_group
----------------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.rigid_object_group

.. autosummary::

   RigidBodyGroupData
   RigidObjectGroup
   RigidObjectGroupCfg

embodichain.lab.sim.objects.robot
---------------------------------

.. currentmodule:: embodichain.lab.sim.objects.robot

.. autosummary::

   ControlGroup
   Robot

embodichain.lab.sim.objects.soft_object
---------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.soft_object

.. autosummary::

   SoftBodyData
   SoftObject
   SoftObjectCfg

embodichain.lab.sim.planners.base_planner
-----------------------------------------

.. currentmodule:: embodichain.lab.sim.planners.base_planner

.. autosummary::

   BasePlannerCfg
   CollisionWorldInfo
   PlanOptions
   BasePlanner
   validate_plan_options

embodichain.lab.sim.planners.curobo.curobo_planner
--------------------------------------------------

.. currentmodule:: embodichain.lab.sim.planners.curobo.curobo_planner

.. autosummary::

   CuroboAutoGenCfg
   CuroboPlanOptions
   CuroboPlanner
   CuroboPlannerCfg
   CuroboWorldCfg

embodichain.lab.sim.planners.curobo.curobo_yaml
-----------------------------------------------

.. currentmodule:: embodichain.lab.sim.planners.curobo.curobo_yaml

.. autosummary::

   generate_curobo_robot_yaml
   generate_curobo_world_yaml

embodichain.lab.sim.planners.motion_generator
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.planners.motion_generator

.. autosummary::

   MotionGenerator
   MotionGenCfg
   MotionGenOptions

embodichain.lab.sim.planners.neural_planner
-------------------------------------------

.. currentmodule:: embodichain.lab.sim.planners.neural_planner

.. autosummary::

   NeuralPlanner
   NeuralPlannerCfg
   NeuralPlanOptions

embodichain.lab.sim.planners.toppra_planner
-------------------------------------------

.. currentmodule:: embodichain.lab.sim.planners.toppra_planner

.. autosummary::

   ToppraPlanner
   ToppraPlannerCfg
   ToppraPlanOptions

embodichain.lab.sim.planners.utils
----------------------------------

.. currentmodule:: embodichain.lab.sim.planners.utils

.. autosummary::

   TrajectorySampleMethod
   MovePart
   MoveType
   PlanState
   PlanResult
   normalize_success_mask
   calculate_point_allocations
   interpolate_xpos
   interpolate_xpos_batched

embodichain.lab.sim.robots.cobotmagic
-------------------------------------

.. currentmodule:: embodichain.lab.sim.robots.cobotmagic

.. autosummary::

   CobotMagicCfg

embodichain.lab.sim.robots.dexforce_w1.hand_specs
-------------------------------------------------

.. currentmodule:: embodichain.lab.sim.robots.dexforce_w1.hand_specs

.. autosummary::

   W1HandSideSpec
   W1HandSpec
   get_default_w1_hand_version
   get_w1_hand_spec
   normalize_w1_hand_mappings

embodichain.lab.sim.robots.dexforce_w1.specs
--------------------------------------------

.. currentmodule:: embodichain.lab.sim.robots.dexforce_w1.specs

.. autosummary::

   W1VersionSpec
   get_w1_version_spec

embodichain.lab.sim.robots.dexforce_w1.types
--------------------------------------------

.. currentmodule:: embodichain.lab.sim.robots.dexforce_w1.types

.. autosummary::

   DexforceW1Version
   DexforceW1HandVersion
   DexforceW1ArmSide
   DexforceW1Type
   DexforceW1HandBrand

embodichain.lab.sim.robots.dexforce_w1.utils
--------------------------------------------

.. currentmodule:: embodichain.lab.sim.robots.dexforce_w1.utils

.. autosummary::

   ChassisManager
   TorsoManager
   HeadManager
   ArmManager
   HandManager
   EyesManager
   build_dexforce_w1_assembly_urdf_cfg

embodichain.lab.sim.robots.dual_arm
-----------------------------------

.. currentmodule:: embodichain.lab.sim.robots.dual_arm

.. autosummary::

   DualArmRobotCfg
   build_dual_arm_cfg
   resolve_mounts

embodichain.lab.sim.robots.franka_panda
---------------------------------------

.. currentmodule:: embodichain.lab.sim.robots.franka_panda

.. autosummary::

   FrankaPandaCfg

embodichain.lab.sim.robots.ur_robot
-----------------------------------

.. currentmodule:: embodichain.lab.sim.robots.ur_robot

.. autosummary::

   URRobotCfg

embodichain.lab.sim.sensors.camera
----------------------------------

.. currentmodule:: embodichain.lab.sim.sensors.camera

.. autosummary::

   Camera
   CameraCfg

embodichain.lab.sim.sim_manager
-------------------------------

.. currentmodule:: embodichain.lab.sim.sim_manager

.. autosummary::

   SIM_CACHE_DIR
   MATERIAL_CACHE_DIR
   CONVEX_DECOMP_DIR
   REACHABLE_XPOS_DIR

embodichain.lab.sim.skills.calls
--------------------------------

.. currentmodule:: embodichain.lab.sim.skills.calls

.. autosummary::

   DeclarativeValue
   HandOver
   OperateArticulation
   Pick
   Place
   PlaceRelationTarget
   RegisteredSemanticCall
   SemanticCallCatalog
   SemanticCallDescriptor
   SemanticCallSpec
   SemanticPose
   builtin_semantic_call_catalog

embodichain.lab.sim.skills.compiler
-----------------------------------

.. currentmodule:: embodichain.lab.sim.skills.compiler

.. autosummary::

   AnalyzedSemanticCall
   GroundedSemanticCall
   GroundedHeldObjectGuard
   GroundedPhaseEffectGate
   ContainerRelationTargetGrounder
   HandOverPoseProvider
   HandOverPoseTargets
   HeldObjectGuardBaseline
   RelationTargetGrounder
   RegisteredSemanticLowerer
   SemanticEffectDependency
   SemanticEffectKind
   SemanticHandOverTarget
   SemanticLowering
   SemanticObjectTarget
   SemanticRelationTarget
   SemanticSkillCompiler
   SupportSurfaceRelationTargetGrounder
   SemanticWorkflow

embodichain.lab.sim.skills.effects
----------------------------------

.. currentmodule:: embodichain.lab.sim.skills.effects

.. autosummary::

   ArticulationJointStateExpectation
   BinaryEffectClause
   BinaryEffectEvidenceBatch
   BinaryEvidenceKind
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
   CoordinatedHeldObjectCleanupExpectation
   EffectClause
   EffectEvidenceAddress
   EffectEvidenceBatch
   EffectEvidenceSourceRef
   EffectExpectationDecision
   EffectExpectationDecision
   EffectMonitor
   EffectMonitorDecision
   EffectMonitorFactory
   EffectMonitorParam
   EffectMonitorRef
   EffectMonitorRegistry
   EffectStateExpectation
   FORCE_EFFECT_CHANNEL
   HeldObjectRelation
   HeldObjectStateExpectation
   JOINT_STATE_EFFECT_CHANNEL
   JointStateEffectClause
   JointStateEvidenceBatch
   POSE_RELATION_EFFECT_CHANNEL
   PoseRelationClause
   PoseRelationEvidenceBatch
   PoseRelationExpectation
   ScalarEffectClause
   ScalarEffectEvidenceBatch
   ScalarEvidenceKind
   ScalarExpectation
   SemanticEffectKind
   SemanticEffectSpec
   SymbolicStateDomain
   SymbolicStateKey

embodichain.lab.sim.skills.evidence
-----------------------------------

.. currentmodule:: embodichain.lab.sim.skills.evidence

.. autosummary::

   ArticulationJointObservationCallback
   BinaryEffectEvidenceQuery
   BinaryEffectObservation
   BinaryObservationCallback
   ControlPartRobotEvidenceSource
   ControlPartSimulationEvidenceProvider
   EffectEvidenceCollectionContext
   EffectEvidenceCollector
   EffectEvidenceProvider
   EffectEvidenceProviderRegistry
   EffectEvidenceQuery
   EffectEvidenceQueryValue
   JointStateEvidenceQuery
   JointStateObservation
   PoseRelationEvidenceQuery
   ScalarEffectEvidenceQuery
   ScalarEffectObservation
   ScalarObservationCallback
   SceneArticulationEvidenceProvider
   build_effect_evidence_queries

embodichain.lab.sim.skills.integration
--------------------------------------

.. currentmodule:: embodichain.lab.sim.skills.integration

.. autosummary::

   BoundSemanticCall
   LinkedSemanticCall
   PathPart
   SceneEntityManifest
   SceneManifest
   SemanticDiagnostic
   SemanticIntegrationManifest
   SemanticValidationError

embodichain.lab.sim.skills.parallel
-----------------------------------

.. currentmodule:: embodichain.lab.sim.skills.parallel

.. autosummary::

   ParallelBarrierUpdate
   ParallelBranchPlan
   ParallelConflictError
   ParallelStateConflictError
   ParallelTimingError
   ParallelTimingPolicy
   align_parallel_commands
   merge_parallel_effects
   resolve_parallel_barrier
   validate_parallel_claims

embodichain.lab.sim.skills.parallel_runtime
-------------------------------------------

.. currentmodule:: embodichain.lab.sim.skills.parallel_runtime

.. autosummary::

   ParallelBranchRuntime
   ParallelBranchStaticAnalysis
   ParallelCommandSafetyValidator
   ParallelLaneCommandSink
   ParallelRuntimeBranch
   ParallelSafetyError
   ParallelSkillResult
   ParallelSkillRuntime
   analyze_parallel_branches

embodichain.lab.sim.skills.profiles
-----------------------------------

.. currentmodule:: embodichain.lab.sim.skills.profiles

.. autosummary::

   AmbiguousSkillBindingError
   BoundRobotSkillProfile
   ControlPartEndpoint
   ControlPartEndpointAdapter
   EndpointResolution
   ProfileValidationError
   ResourceEndpoint
   ResourceEndpointAdapter
   ResolvedRobotResource
   ResolvedResourceEndpoint
   ResolvedSkillBinding
   ResourceBinding
   ResourceClaim
   RobotResource
   RobotSkillProfile
   SkillPolicyPreset
   WorkflowRecoveryPolicy
   UnsupportedSkillError

embodichain.lab.sim.skills.runtime
----------------------------------

.. currentmodule:: embodichain.lab.sim.skills.runtime

.. autosummary::

   AtomicSkills
   EffectEvidenceCollectorPort
   ResolvedCorePolicyTrace
   SkillCallTrace
   SkillEndpointBindingTrace
   SkillEndpointTrackingChannelTrace
   SkillEffectTrace
   SkillFailure
   SkillPlanAttemptTrace
   SkillResult
   SkillRuntime
   SkillRuntimeProvider
   SkillScene
   SkillStatus
   SkillWorkflowRecoveryRole
   SkillWorkflowRecoveryTrace
   task_state_to_metadata

embodichain.lab.sim.skills.scene
--------------------------------

.. currentmodule:: embodichain.lab.sim.skills.scene

.. autosummary::

   ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY
   ArticulationJointEvidenceAddress
   ContainerAffordance
   SCENE_ARTICULATION_EVIDENCE_PROVIDER_ID
   SCENE_ARTICULATION_EVIDENCE_PROVIDER_REVISION
   SceneArticulationJointStateProvider
   RegistrySceneProvider
   SceneAffordanceRef
   SceneArticulationRef
   SceneCollisionRole
   SceneCollisionWorldMode
   SceneDynamics
   SceneEntityRef
   SceneEntityMetadata
   SceneEntityRegistration
   SceneEntityStateProvider
   SceneGeometryProvider
   SceneLinkRef
   SceneObjectRef
   SceneRegistry
   AmbiguousSceneAffordanceError
   GRASP_AFFORDANCE_CAPABILITY
   PLACE_IN_AFFORDANCE_CAPABILITY
   PLACE_ON_AFFORDANCE_CAPABILITY
   PLACEMENT_TARGET_AFFORDANCE_REVISION
   SupportSurfaceAffordance
   UnsupportedSceneAffordanceError

embodichain.lab.sim.solvers.neural_ik_solver
--------------------------------------------

.. currentmodule:: embodichain.lab.sim.solvers.neural_ik_solver

.. autosummary::

   NeuralIKSolverCfg
   NeuralIKSolver

embodichain.lab.sim.solvers.srs_solver
--------------------------------------

.. currentmodule:: embodichain.lab.sim.solvers.srs_solver

.. autosummary::

   SRSSolver
   SRSSolverCfg

embodichain.lab.sim.utility.render_utils
----------------------------------------

.. currentmodule:: embodichain.lab.sim.utility.render_utils

.. autosummary::

   select_default_renderer

embodichain.lab.sim.workspace.caches.cache_utils
------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.caches.cache_utils

.. autosummary::

   clean_all_sessions
   clean_session
   format_size
   get_cache_root
   get_dir_size
   list_sessions
   main
   show_session_info
   show_total_size

embodichain.lab.sim.workspace.caches.results_cache
--------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.caches.results_cache

.. autosummary::

   DEFAULT_RESULTS_CACHE_DIR
   ResultsCache
   compute_cache_key
   serialize_results
   deserialize_results

embodichain.lab.sim.workspace.constraints.base_constraint
---------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.constraints.base_constraint

.. autosummary::

   IConstraintChecker
   BaseConstraintChecker

embodichain.lab.sim.workspace.constraints.workspace_constraint
--------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.constraints.workspace_constraint

.. autosummary::

   WorkspaceConstraintChecker

embodichain.lab.sim.workspace.samplers.base_sampler
---------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.samplers.base_sampler

.. autosummary::

   ISampler
   BaseSampler

embodichain.lab.sim.workspace.samplers.gaussian_sampler
-------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.samplers.gaussian_sampler

.. autosummary::

   GaussianSampler

embodichain.lab.sim.workspace.samplers.halton_sampler
-----------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.samplers.halton_sampler

.. autosummary::

   HaltonSampler

embodichain.lab.sim.workspace.samplers.importance_sampler
---------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.samplers.importance_sampler

.. autosummary::

   ImportanceSampler

embodichain.lab.sim.workspace.samplers.iniform_sampler
------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.samplers.iniform_sampler

.. autosummary::

   UniformSampler

embodichain.lab.sim.workspace.samplers.lhs_sampler
--------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.samplers.lhs_sampler

.. autosummary::

   LatinHypercubeSampler

embodichain.lab.sim.workspace.samplers.random_sampler
-----------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.samplers.random_sampler

.. autosummary::

   RandomSampler

embodichain.lab.sim.workspace.samplers.sobol_sampler
----------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.samplers.sobol_sampler

.. autosummary::

   SobolSampler

embodichain.lab.sim.workspace.visualizers.axis_visualizer
---------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.visualizers.axis_visualizer

.. autosummary::

   AxisVisualizer

embodichain.lab.sim.workspace.visualizers.base_visualizer
---------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.visualizers.base_visualizer

.. autosummary::

   IVisualizer
   BaseVisualizer

embodichain.lab.sim.workspace.visualizers.point_cloud_visualizer
----------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.visualizers.point_cloud_visualizer

.. autosummary::

   PointCloudVisualizer

embodichain.lab.sim.workspace.visualizers.sphere_visualizer
-----------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.visualizers.sphere_visualizer

.. autosummary::

   SphereVisualizer

embodichain.lab.sim.workspace.visualizers.visualizer_factory
------------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.visualizers.visualizer_factory

.. autosummary::

   VisualizerFactory
   create_visualizer

embodichain.lab.sim.workspace.visualizers.voxel_visualizer
----------------------------------------------------------

.. currentmodule:: embodichain.lab.sim.workspace.visualizers.voxel_visualizer

.. autosummary::

   VoxelVisualizer

embodichain.lab.visualization.backends
--------------------------------------

.. currentmodule:: embodichain.lab.visualization.backends

.. autosummary::

   VisualizationBackend
   ViserBackend

embodichain.lab.visualization.backends.base
-------------------------------------------

.. currentmodule:: embodichain.lab.visualization.backends.base

.. autosummary::

   VisualizationBackend

embodichain.lab.visualization.backends.viser
--------------------------------------------

.. currentmodule:: embodichain.lab.visualization.backends.viser

.. autosummary::

   ViserBackend

embodichain.lab.visualization.cfg
---------------------------------

.. currentmodule:: embodichain.lab.visualization.cfg

.. autosummary::

   VisualizationCfg
   ViserServerCfg

embodichain.lab.visualization.cli
---------------------------------

.. currentmodule:: embodichain.lab.visualization.cli

.. autosummary::

   add_viser_args_to_parser
   visualization_cfg_from_args

embodichain.lab.visualization.protocol
--------------------------------------

.. currentmodule:: embodichain.lab.visualization.protocol

.. autosummary::

   SCHEMA_VERSION
   CameraImage
   CameraImageFrame
   CameraSpec
   DynamicMeshUpdate
   FrameOverlay
   GizmoCommand
   GizmoSpec
   GizmoState
   JointControlCommand
   JointControlProvider
   JointControlSpec
   JointControlState
   MeshGeometry
   PointCloudOverlay
   SceneFrame
   SceneManifest
   SceneNode
   SceneOverlays
   TargetOverlay
   TrajectoryOverlay
   estimate_camera_image_frame_bytes
   estimate_frame_bytes
   estimate_manifest_bytes
   pose_to_position_wxyz

embodichain.lab.visualization.runtime
-------------------------------------

.. currentmodule:: embodichain.lab.visualization.runtime

.. autosummary::

   GizmoCommandQueue
   JointControlCommandQueue
   LatestFrameQueue
   RuntimeHealth
   RuntimeStats
   VisualizationRuntime

embodichain.lab.visualization.scene_exporter
--------------------------------------------

.. currentmodule:: embodichain.lab.visualization.scene_exporter

.. autosummary::

   CameraImageCaptureResult
   CaptureResult
   SceneExporter
   mesh_geometry_id
   safe_path_component

embodichain.learning.rl.algo.apg
--------------------------------

.. currentmodule:: embodichain.learning.rl.algo.apg

.. autosummary::

   APG
   APGCfg
   segmented_discounted_return

embodichain.learning.rl.algo.base
---------------------------------

.. currentmodule:: embodichain.learning.rl.algo.base

.. autosummary::

   BaseAlgorithm
   RolloutKind

embodichain.learning.rl.algo.common
-----------------------------------

.. currentmodule:: embodichain.learning.rl.algo.common

.. autosummary::

   compute_gae

embodichain.learning.rl.algo.grpo
---------------------------------

.. currentmodule:: embodichain.learning.rl.algo.grpo

.. autosummary::

   GRPO
   GRPOCfg

embodichain.learning.rl.algo.ppo
--------------------------------

.. currentmodule:: embodichain.learning.rl.algo.ppo

.. autosummary::

   PPO
   PPOCfg

embodichain.learning.rl.collector.base
--------------------------------------

.. currentmodule:: embodichain.learning.rl.collector.base

.. autosummary::

   BaseCollector

embodichain.learning.rl.collector.differentiable
------------------------------------------------

.. currentmodule:: embodichain.learning.rl.collector.differentiable

.. autosummary::

   DifferentiableCollector
   DifferentiableRollout
   DifferentiableTransition

embodichain.learning.rl.collector.sync_collector
------------------------------------------------

.. currentmodule:: embodichain.learning.rl.collector.sync_collector

.. autosummary::

   SyncCollector

embodichain.learning.rl.experimental.newton
-------------------------------------------

.. currentmodule:: embodichain.learning.rl.experimental.newton

.. autosummary::

   NewtonPlanarReachEnv
   NewtonPlanarReachEnvCfg

embodichain.learning.rl.experimental.newton.planar_reach
--------------------------------------------------------

.. currentmodule:: embodichain.learning.rl.experimental.newton.planar_reach

.. autosummary::

   NewtonPlanarReachEnv
   NewtonPlanarReachEnvCfg

embodichain.learning.rl.experimental.newton.train_planar_reach
--------------------------------------------------------------

.. currentmodule:: embodichain.learning.rl.experimental.newton.train_planar_reach

.. autosummary::

   NewtonPlanarReachTrainingCfg
   train_planar_reach

embodichain.learning.rl.models.actor_critic
-------------------------------------------

.. currentmodule:: embodichain.learning.rl.models.actor_critic

.. autosummary::

   ActorCritic

embodichain.learning.rl.models.actor_only
-----------------------------------------

.. currentmodule:: embodichain.learning.rl.models.actor_only

.. autosummary::

   ActorOnly

embodichain.learning.rl.models.policy
-------------------------------------

.. currentmodule:: embodichain.learning.rl.models.policy

.. autosummary::

   Policy

embodichain.learning.rl.utils.optimizer
---------------------------------------

.. currentmodule:: embodichain.learning.rl.utils.optimizer

.. autosummary::

   bind_scheduler_horizon
   build_lr_scheduler
   build_optimizer
   coerce_lr_scheduler_cfg
   coerce_optimizer_cfg
   get_registered_lr_scheduler_names
   get_registered_optimizer_names
   scheduler_needs_horizon

embodichain.toolkits.acd
------------------------

.. currentmodule:: embodichain.toolkits.acd

.. autosummary::

   generate_urdf_collision_convexes

embodichain.toolkits.acd.cli
----------------------------

.. currentmodule:: embodichain.toolkits.acd.cli

.. autosummary::

   main

embodichain.toolkits.acd.urdf_modifider
---------------------------------------

.. currentmodule:: embodichain.toolkits.acd.urdf_modifider

.. autosummary::

   URDFModifider

embodichain.toolkits.graspkit.pg_grasp.antipodal_generator
----------------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.antipodal_generator

.. autosummary::

   GraspGenerator
   GraspGeneratorCfg

embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler
--------------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.antipodal_sampler

.. autosummary::

   AntipodalSamplerCfg
   AntipodalSampler

embodichain.toolkits.graspkit.pg_grasp.collision_checker
--------------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.collision_checker

.. autosummary::

   ConvexCollisionCheckerCfg
   ConvexCollisionChecker

embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker
----------------------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker

.. autosummary::

   GripperCollisionCfg
   GripperCollisionChecker
   box_surface_grid

embodichain.toolkits.graspkit.scripts.annotate_grasp
----------------------------------------------------

.. currentmodule:: embodichain.toolkits.graspkit.scripts.annotate_grasp

.. autosummary::

   cli

embodichain.toolkits.urdf_assembly.component
--------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.component

.. autosummary::

   ComponentRegistry
   URDFComponent
   URDFComponentManager

embodichain.toolkits.urdf_assembly.connection
---------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.connection

.. autosummary::

   URDFConnectionManager

embodichain.toolkits.urdf_assembly.file_writer
----------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.file_writer

.. autosummary::

   URDFFileWriter

embodichain.toolkits.urdf_assembly.logging_utils
------------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.logging_utils

.. autosummary::

   URDFAssemblyLogger

embodichain.toolkits.urdf_assembly.mesh
---------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.mesh

.. autosummary::

   URDFMeshManager

embodichain.toolkits.urdf_assembly.sensor
-----------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.sensor

.. autosummary::

   SensorRegistry
   SensorAttachment
   URDFSensorManager

embodichain.toolkits.urdf_assembly.signature
--------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.signature

.. autosummary::

   URDFAssemblySignatureManager

embodichain.toolkits.urdf_assembly.urdf_assembly_manager
--------------------------------------------------------

.. currentmodule:: embodichain.toolkits.urdf_assembly.urdf_assembly_manager

.. autosummary::

   URDFAssemblyManager

embodichain.workspace_cache_cli
-------------------------------

.. currentmodule:: embodichain.workspace_cache_cli

.. autosummary::

   main

embodichain.utils
-----------------

.. currentmodule:: embodichain.utils

.. autosummary::

   GLOBAL_SEED
   is_configclass
   resolve_config_path
   set_seed

embodichain_tasks.configs
-------------------------

.. currentmodule:: embodichain_tasks.configs

.. autosummary::

   get_config_path

embodichain_tasks.multi_segments
--------------------------------

.. currentmodule:: embodichain_tasks.multi_segments

.. autosummary::

   MultiSegmentsCubePickPlaceEnv

embodichain_tasks.multi_segments.cube_pick_place
------------------------------------------------

.. currentmodule:: embodichain_tasks.multi_segments.cube_pick_place

.. autosummary::

   MultiSegmentsCubePickPlaceEnv
   CUBE_EXPERT_PROGRAM_REGISTRATION
   create_cube_robot_profile_binding
   create_cube_scene_binding

embodichain_tasks.rl
--------------------

.. currentmodule:: embodichain_tasks.rl

.. autosummary::

   build_env

embodichain_tasks.rl.basic
--------------------------

.. currentmodule:: embodichain_tasks.rl.basic

.. autosummary::

   CartPoleEnv
   PointMassEnv

embodichain_tasks.rl.basic.point_mass
-------------------------------------

.. currentmodule:: embodichain_tasks.rl.basic.point_mass

.. autosummary::

   PointMassEnv

embodichain_tasks.special.simple_task
-------------------------------------

.. currentmodule:: embodichain_tasks.special.simple_task

.. autosummary::

   SimpleTaskEnv

embodichain_tasks.special.stay_still_save
-----------------------------------------

.. currentmodule:: embodichain_tasks.special.stay_still_save

.. autosummary::

   StayStillSaveEnv

embodichain_tasks.tableware
---------------------------

.. currentmodule:: embodichain_tasks.tableware

.. autosummary::

   HandOverEnv
   OpenDrawerEnv

embodichain_tasks.tableware.hand_over
-------------------------------------

.. currentmodule:: embodichain_tasks.tableware.hand_over

.. autosummary::

   HandOverEnv
   HAND_OVER_EXPERT_PROGRAM_REGISTRATION
   HAND_OVER_POSE_PROVIDER
   create_hand_over_robot_profile_binding
   create_hand_over_scene_binding

embodichain_tasks.tableware.blocks_ranking_rgb
----------------------------------------------

.. currentmodule:: embodichain_tasks.tableware.blocks_ranking_rgb

.. autosummary::

   BlocksRankingRGBEnv

embodichain_tasks.tableware.blocks_ranking_size
-----------------------------------------------

.. currentmodule:: embodichain_tasks.tableware.blocks_ranking_size

.. autosummary::

   BlocksRankingSizeEnv

embodichain_tasks.tableware.match_object_container
--------------------------------------------------

.. currentmodule:: embodichain_tasks.tableware.match_object_container

.. autosummary::

   MatchObjectContainerEnv

embodichain_tasks.tableware.open_drawer
---------------------------------------

.. currentmodule:: embodichain_tasks.tableware.open_drawer

.. autosummary::

   OpenDrawerEnv
   OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION
   create_open_drawer_robot_profile_binding
   create_open_drawer_scene_binding

embodichain_tasks.tableware.place_object_drawer
-----------------------------------------------

.. currentmodule:: embodichain_tasks.tableware.place_object_drawer

.. autosummary::

   PlaceObjectDrawerEnv

embodichain_tasks.tableware.pour_water.action_bank
--------------------------------------------------

.. currentmodule:: embodichain_tasks.tableware.pour_water.action_bank

.. autosummary::

   PourWaterActionBank

embodichain_tasks.tableware.pour_water.pour_water
-------------------------------------------------

.. currentmodule:: embodichain_tasks.tableware.pour_water.pour_water

.. autosummary::

   PourWaterEnv
   PourWaterAgentEnv

embodichain_tasks.tableware.rearrangement
-----------------------------------------

.. currentmodule:: embodichain_tasks.tableware.rearrangement

.. autosummary::

   RearrangementEnv
   RearrangementAgentEnv

embodichain_tasks.tableware.scoop_ice
-------------------------------------

.. currentmodule:: embodichain_tasks.tableware.scoop_ice

.. autosummary::

   ScoopIce

embodichain_tasks.tableware.stack_blocks_two
--------------------------------------------

.. currentmodule:: embodichain_tasks.tableware.stack_blocks_two

.. autosummary::

   StackBlocksTwoEnv

embodichain_tasks.tableware.stack_cups
--------------------------------------

.. currentmodule:: embodichain_tasks.tableware.stack_cups

.. autosummary::

   StackCupsEnv

embodichain_tasks.utils.importer
--------------------------------

.. currentmodule:: embodichain_tasks.utils.importer

.. autosummary::

   import_packages
