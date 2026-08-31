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

Strict MLLM entry points that inject host-owned integration settings before
decoding and compiling the constrained Expert Program schema surface.

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

embodichain.gen_sim.env
-----------------------

.. currentmodule:: embodichain.gen_sim.env

.. autosummary::

   find_gen_sim_env_file
   get_embodichain_root
   load_gen_sim_env

embodichain.gen_sim.gradio_ui.app_articraft
-------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_articraft

.. autosummary::

   build_articraft_panel
   cleanup_articraft_session
   configure_articraft_environment
   generate_articraft_asset
   reset_articraft_asset

embodichain.gen_sim.gradio_ui.app_asset_engine
----------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_asset_engine

.. autosummary::

   build_asset_engine_panel
   cleanup_asset_engine_session
   prepare_asset_input_preview
   reset_simready_asset
   run_simready_asset

embodichain.gen_sim.gradio_ui.app_commands
------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_commands

.. autosummary::

   build_run_agent_command

embodichain.gen_sim.gradio_ui.app_env
-------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_env

.. autosummary::

   ACTION_ENGINE_VISER_PORT
   ARTICULATION_SERVER_BASE_URL
   ARTICULATION_SERVER_POLL_INTERVAL_S
   ARTICULATION_SERVER_TASK_TIMEOUT_S
   ARTICULATION_SERVER_TIMEOUT_S
   ARTICRAFT_CONDA_ENV
   ARTICRAFT_OUTPUT_ROOT
   ARTICRAFT_REPOSITORY_URL
   ARTICRAFT_ROOT
   DIRECT_NO_PROXY_VALUE
   EMBODICHAIN_ROOT
   GRADIO_AUTH_PASSWORD
   GRADIO_AUTH_USERNAME
   PROXY_ENV_KEYS
   SCENE_ENGINE_VISER_PORT
   SERVER_NAME
   SERVER_PORT
   SIMREADY_OPENAI_API_KEY
   SIMREADY_OPENAI_BASE_URL
   SIMREADY_OPENAI_MODEL
   build_gradio_allowed_paths
   build_gradio_blocked_paths
   configure_direct_network_env
   configure_simready_llm_env
   get_inherited_network_env
   get_gradio_auth
   validate_gradio_artifact_root

embodichain.gen_sim.gradio_ui.app_media
---------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_media

.. autosummary::

   latest_audience_output_video

embodichain.gen_sim.gradio_ui.app_processes
-------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_processes

.. autosummary::

   SessionProcessRegistry
   build_codex_env
   build_pipeline_env
   build_run_agent_command
   force_stop_all_child_processes
   get_request_session_id
   kill_process_group
   read_process_output
   register_managed_process
   run_agent_cli_supports_robot_profile
   start_pipeline
   terminate_process_group
   redact_sensitive_text

embodichain.gen_sim.gradio_ui.app_services
------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_services

.. autosummary::

   build_app

embodichain.gen_sim.gradio_ui.app_state
---------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_state

.. autosummary::

   PHASES
   Phase
   RuntimeState
   SessionRuntimeRegistry
   runtime_lock
   runtime_registry
   set_runtime_phase_locked

embodichain.gen_sim.gradio_ui.app_ui
------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_ui

.. autosummary::

   build_app

embodichain.gen_sim.gradio_ui.app_workflows
-------------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.app_workflows

.. autosummary::

   cleanup_workflow_session
   format_status
   preview_editable_scene
   preview_saved_scene
   refresh_saved_scenes
   reset_scene_engine
   run_action_engine_from_current
   run_scene_edit
   run_scene_engine
   stop_action_engine
   ui_snapshot

embodichain.gen_sim.gradio_ui.gradio_app
----------------------------------------

.. currentmodule:: embodichain.gen_sim.gradio_ui.gradio_app

.. autosummary::

   main

embodichain.gen_sim.scene_engine.core.scene_edit_plan
-----------------------------------------------------

.. currentmodule:: embodichain.gen_sim.scene_engine.core.scene_edit_plan

.. autosummary::

   SceneEditOperation
   SceneEditPlan

embodichain.gen_sim.scene_engine.pipeline.editing.scene_edit_asset_preparation
-------------------------------------------------------------------------------

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
   execute_demo_episode
   resolve_demo_segments

embodichain.lab.gym.envs.embodied_env
-------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.embodied_env

.. autosummary::

   EmbodiedEnvCfg
   EmbodiedEnv

embodichain.lab.gym.envs.expert_program.simulation_handover
-----------------------------------------------------------

.. currentmodule:: embodichain.lab.gym.envs.expert_program.simulation_handover

.. autosummary::

   ConfiguredHandOverPoseProvider

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
   AssembleAffordance
   BUILTIN_ACTION_TYPES
   CoordinatedPickmentOptions
   CoordinatedPlacementOptions
   DynamicCollisionMode
   EntityState
   EffectVerifier
   GRASP_COMMAND
   HandOverOptions
   InteractionPoints
   MoveEndEffectorOptions
   MoveHeldObjectOptions
   MoveJointsOptions
   ObjectActionGoal
   OPEN_COMMAND
   PickUpOptions
   PlaceOptions
   PoseGoalValue
   PushObject
   PushObjectGoal
   PushObjectOptions
   PushObjectToolCalibration
   RigidObjectSceneProvider
   RigidObjectSceneProviderCfg
   RunnerStepCallback
   SceneProvider
   SceneSnapshotSupplier

embodichain.lab.sim.atomic_actions.affordance
---------------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.affordance

.. autosummary::

   Affordance
   AntipodalAffordance
   AxisAlignAffordance
   SlideAffordance
   PressAffordance
   TwistAffordance
   InteractionPoints
   AssembleAffordance

.. automodule:: embodichain.lab.sim.atomic_actions.affordance
   :members:
   :no-index:

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

   EffectExpectationResult
   EffectVerificationRequest
   EffectVerificationResult
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

   ObjectActionGoal
   PoseGoalValue
   SceneEntityPose
   collect_scene_dependencies
   resolve_pose_goal
   validate_pose_goal
   validate_pose_tensor

.. automodule:: embodichain.lab.sim.atomic_actions.goals
   :members:
   :no-index:

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
   PlannerDiagnostics
   PlanningFailure
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

   create_simulation_atomic_action_engine
   RigidObjectSceneProvider
   RigidObjectSceneProviderCfg
   SceneSnapshotSupplier
   SimulationExecutionAdapter

embodichain.lab.sim.atomic_actions.state
----------------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.state

.. autosummary::

   EntityState
   HeldObjectState
   PlanningContext
   RobotObservation
   SceneSnapshot
   TaskState

.. automodule:: embodichain.lab.sim.atomic_actions.state
   :members:
   :no-index:

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

embodichain.lab.sim.diff
------------------------

Public differentiable-stepping bridge from manager-owned Newton trajectories
and Warp tapes into PyTorch autograd.

.. currentmodule:: embodichain.lab.sim.diff

.. autosummary::

   NewtonStepFunc
   differentiable_step
   tape_context

embodichain.lab.sim.diff.bridge
-------------------------------

.. currentmodule:: embodichain.lab.sim.diff.bridge

.. autosummary::

   NewtonStepFunc
   differentiable_step
   tape_context

embodichain.lab.sim.diff.runtime
--------------------------------

.. currentmodule:: embodichain.lab.sim.diff.runtime

.. autosummary::

   NewtonDifferentiableRuntime

embodichain.lab.sim.objects.articulation
----------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.articulation

.. autosummary::

   ArticulationData
   Articulation
   ArticulationJointKinematics

embodichain.lab.sim.objects.cloth_object
----------------------------------------

.. currentmodule:: embodichain.lab.sim.objects.cloth_object

.. autosummary::

   ClothBodyData
   ClothObject
   ClothObjectCfg
   SurfaceDeformableData
   SurfaceDeformableObject
   SurfaceDeformableObjectCfg

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
   VolumeDeformableData
   VolumeDeformableObject
   VolumeDeformableObjectCfg

embodichain.lab.sim.physics
---------------------------

Manager-level physics backend selection and lifecycle contracts for the
Default and Newton implementations integrated through DexSim.

.. currentmodule:: embodichain.lab.sim.physics

.. autosummary::

   PhysicsBackend
   DefaultPhysicsBackend
   NewtonPhysicsBackend
   make_physics_backend

embodichain.lab.sim.physics.base
--------------------------------

.. currentmodule:: embodichain.lab.sim.physics.base

.. autosummary::

   PhysicsBackend

embodichain.lab.sim.physics.default
-----------------------------------

.. currentmodule:: embodichain.lab.sim.physics.default

.. autosummary::

   DefaultPhysicsBackend

embodichain.lab.sim.physics.newton
----------------------------------

.. currentmodule:: embodichain.lab.sim.physics.newton

.. autosummary::

   NewtonPhysicsBackend

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
   Pick
   Place
   PlaceRelationTarget
   RegisteredSemanticCall
   SemanticCallCatalog
   SemanticCallDescriptor
   SemanticCallSpec
   SemanticPose
   builtin_semantic_call_catalog

.. automodule:: embodichain.lab.sim.skills.calls
   :members:
   :no-index:

embodichain.lab.sim.skills.compiler
-----------------------------------

.. currentmodule:: embodichain.lab.sim.skills.compiler

.. autosummary::

   AnalyzedSemanticCall
   ContainerRelationTargetGrounder
   GroundedHeldObjectGuard
   GroundedPhaseEffectGate
   GroundedSemanticCall
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
   SemanticWorkflow
   SupportSurfaceRelationTargetGrounder

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

.. automodule:: embodichain.lab.sim.skills.integration
   :members:
   :no-index:

embodichain.lab.sim.skills.effects
----------------------------------

.. automodule:: embodichain.lab.sim.skills.effects
   :members:
   :no-index:

embodichain.lab.sim.skills.evidence
-----------------------------------

.. automodule:: embodichain.lab.sim.skills.evidence
   :members:
   :no-index:

embodichain.lab.sim.skills.parallel
------------------------------------

.. automodule:: embodichain.lab.sim.skills.parallel
   :members:
   :no-index:

embodichain.lab.sim.skills.parallel_runtime
--------------------------------------------

.. automodule:: embodichain.lab.sim.skills.parallel_runtime
   :members:
   :no-index:

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
   UnsupportedSkillError
   WorkflowRecoveryPolicy

embodichain.lab.sim.skills.runtime
----------------------------------

.. automodule:: embodichain.lab.sim.skills.runtime
   :members:
   :no-index:

embodichain.lab.sim.skills.scene
--------------------------------

.. currentmodule:: embodichain.lab.sim.skills.scene

.. autosummary::

   AmbiguousSceneAffordanceError
   GRASP_AFFORDANCE_CAPABILITY
   PLACE_IN_AFFORDANCE_CAPABILITY
   PLACE_ON_AFFORDANCE_CAPABILITY
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
   UnsupportedSceneAffordanceError

.. automodule:: embodichain.lab.sim.skills.scene
   :members:
   :no-index:

embodichain.lab.sim.solvers.neural_ik_solver
--------------------------------------------

.. currentmodule:: embodichain.lab.sim.solvers.neural_ik_solver

.. autosummary::

   NeuralIKSolverCfg
   NeuralIKSolver

embodichain.lab.sim.solvers.null_space_posture_task
---------------------------------------------------

.. currentmodule:: embodichain.lab.sim.solvers.null_space_posture_task

.. autosummary::

   NullSpacePostureTask

embodichain.lab.sim.solvers.pink_solver
---------------------------------------

.. currentmodule:: embodichain.lab.sim.solvers.pink_solver

.. autosummary::

   PinkSolver
   PinkSolverCfg

embodichain.lab.sim.solvers.srs_solver
--------------------------------------

.. currentmodule:: embodichain.lab.sim.solvers.srs_solver

.. autosummary::

   SRSSolver
   SRSSolverCfg

embodichain.lab.sim.spawn
-------------------------

Translation boundary from EmbodiChain object configs and singleton USD assets
into DexSim Spawn descriptors.

.. currentmodule:: embodichain.lab.sim.spawn

.. autosummary::

   articulation_desc_from_cfg
   articulation_desc_from_usd
   cloth_desc_from_cfg
   rigid_desc_from_cfg
   rigid_desc_from_usd
   soft_desc_from_cfg
   surface_deformable_desc_from_cfg
   volume_deformable_desc_from_cfg

embodichain.lab.sim.spawn.descriptors
-------------------------------------

.. currentmodule:: embodichain.lab.sim.spawn.descriptors

.. autosummary::

   articulation_desc_from_cfg
   cloth_desc_from_cfg
   configure_articulation_desc
   rigid_desc_from_cfg
   soft_desc_from_cfg
   surface_deformable_desc_from_cfg
   volume_deformable_desc_from_cfg

embodichain.lab.sim.spawn.scene
-------------------------------

.. currentmodule:: embodichain.lab.sim.spawn.scene

.. autosummary::

   SpawnScene

embodichain.lab.sim.spawn.source
--------------------------------

.. currentmodule:: embodichain.lab.sim.spawn.source

.. autosummary::

   resolve_articulation_source

embodichain.lab.sim.spawn.usd
-----------------------------

.. currentmodule:: embodichain.lab.sim.spawn.usd

.. autosummary::

   articulation_desc_from_usd
   rigid_desc_from_usd

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
   complete_discounted_return
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

embodichain.learning.rl.gradients
---------------------------------

Row-wise action-adjoint clipping and its rollout-level diagnostics.

.. currentmodule:: embodichain.learning.rl.gradients

.. autosummary::

   BatchedGradientNormStats
   clip_batched_gradient_norm

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

embodichain.learning.rl.normalization
-------------------------------------

.. currentmodule:: embodichain.learning.rl.normalization

.. autosummary::

   RunningObservationNormalizer

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

embodichain_tasks.classic_control.cart_pole
--------------------------------------------

Cart-pole environment registration under the task-first import path.

.. currentmodule:: embodichain_tasks.classic_control.cart_pole

.. autosummary::

   CartPoleEnv

embodichain_tasks.classic_control.point_mass
---------------------------------------------

Differentiable lightweight point-mass task and learning-environment registration.

.. currentmodule:: embodichain_tasks.classic_control.point_mass

.. autosummary::

   PointMassEnv

embodichain_tasks.configs
-------------------------

.. currentmodule:: embodichain_tasks.configs

.. autosummary::

   get_config_path

embodichain_tasks.manipulation.push_cube
----------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.push_cube

.. autosummary::

   PushCubeEnv

embodichain_tasks.manipulation.tableware.blocks_ranking_rgb
-----------------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.blocks_ranking_rgb

.. autosummary::

   BlocksRankingRGBEnv

embodichain_tasks.manipulation.tableware.blocks_ranking_size
------------------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.blocks_ranking_size

.. autosummary::

   BlocksRankingSizeEnv

embodichain_tasks.manipulation.tableware.match_object_container
---------------------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.match_object_container

.. autosummary::

   MatchObjectContainerEnv

embodichain_tasks.manipulation.tableware.place_object_drawer
------------------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.place_object_drawer

.. autosummary::

   PlaceObjectDrawerEnv

embodichain_tasks.manipulation.tableware.scoop_ice
--------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.scoop_ice

.. autosummary::

   ScoopIce

embodichain_tasks.manipulation.tableware.stack_blocks_two
---------------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.stack_blocks_two

.. autosummary::

   StackBlocksTwoEnv

embodichain_tasks.manipulation.tableware.stack_cups
---------------------------------------------------

.. currentmodule:: embodichain_tasks.manipulation.tableware.stack_cups

.. autosummary::

   StackCupsEnv

embodichain_tasks.special.franka_reach_apg
-------------------------------------------

Differentiable Franka FR3 reach environment that demonstrates the explicit
kinematics route used by analytic policy-gradient experiments.

.. currentmodule:: embodichain_tasks.special.franka_reach_apg

.. autosummary::

   FrankaReachApgEnv

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

embodichain_tasks.utils.importer
--------------------------------

.. currentmodule:: embodichain_tasks.utils.importer

.. autosummary::

   import_packages
