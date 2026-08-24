embodichain.lab.gym.envs.expert_program
=======================================

.. automodule:: embodichain.lab.gym.envs.expert_program

   .. autosummary::

      AntipodalGraspAffordanceBinding
      ArticulationJointPositionValidatorCfg
      AtomicDemoBridge
      BarrierCfg
      BufferedGymCommandSink
      ConfigPath
      ConfigPathPart
      CompiledProgram
      ControlPartCommandPreset
      ControlPartEndpointBinding
      ControlPartResourceBinding
      CyclicPoseTargetCfg
      DemoBridgeError
      EXPERT_PROGRAM_SCHEMA_VERSION
      EnvironmentStepClock
      EnvironmentStepTimingError
      ExpertProgramCfg
      ExpertProgramCompileError
      ExpertProgramCompiler
      ExpertProgramConfigError
      ExpertProgramDecodeError
      ExpertProgramEnvironmentAdapter
      ExpertProgramEnvironmentFactory
      ExpertProgramIntegrationCfg
      ExpertProgramRuntimeAssembly
      ExpertProgramValidationContext
      ExpertProgramValidationError
      GymPlanningObservationProvider
      HandOverCfg
      InvokeCfg
      ObjectNearTargetValidatorCfg
      ParallelCfg
      PickCfg
      PlaceCfg
      PlanningObservationPort
      PoseCfg
      RegisteredSemanticCallCfg
      RepeatCfg
      RuntimeCommandFrameEncoder
      RuntimeTransportActionEncoder
      SceneReferenceRole
      SegmentCfg
      SegmentPostPolicyPort
      SegmentValidatorPort
      SequenceCfg
      SimulationArticulationBinding
      SimulationArticulationLinkBinding
      SimulationExpertProgramFactory
      SimulationRigidObjectBinding
      SimulationRobotSkillProfileBinding
      SimulationSceneBinding
      SimulationSegmentPolicyPort
      TargetRefCfg
      UnsupportedRuntimeTransportError
      WaitStablePostCfg
      create_simulation_expert_program_adapter
      decode_expert_program
      load_expert_program
      loads_expert_program_json
      parse_expert_program_json
      render_config_path
      validate_expert_program

.. currentmodule:: embodichain.lab.gym.envs.expert_program

Schema and loading
------------------

Expert Program schema version 2 is the only accepted top-level schema. It
contains bounded sequential nodes and deterministic parallel blocks whose
barrier is owned by the enclosing parallel node.

.. autodata:: EXPERT_PROGRAM_SCHEMA_VERSION

.. autoclass:: ExpertProgramCfg
   :members:

.. autoclass:: ExpertProgramIntegrationCfg
   :members:

.. autoclass:: PoseCfg
   :members:

.. autoclass:: TargetRefCfg
   :members:

.. autoclass:: CyclicPoseTargetCfg
   :members:

.. autoclass:: PickCfg
   :members:

.. autoclass:: PlaceCfg
   :members:

.. autoclass:: HandOverCfg
   :members:

.. autoclass:: RegisteredSemanticCallCfg
   :members:

.. autoclass:: InvokeCfg
   :members:

.. autoclass:: SequenceCfg
   :members:

.. autoclass:: RepeatCfg
   :members:

.. autoclass:: SegmentCfg
   :members:

.. autoclass:: ParallelCfg
   :members:

.. autoclass:: BarrierCfg
   :members:

.. autoclass:: WaitStablePostCfg
   :members:

.. autoclass:: ObjectNearTargetValidatorCfg
   :members:

.. autoclass:: ArticulationJointPositionValidatorCfg
   :members:

.. autofunction:: load_expert_program

.. autofunction:: loads_expert_program_json

.. autofunction:: parse_expert_program_json

.. autofunction:: decode_expert_program

.. autofunction:: validate_expert_program

.. autofunction:: render_config_path

.. autodata:: ConfigPath

.. autodata:: ConfigPathPart

.. autodata:: SceneReferenceRole

.. autoclass:: ExpertProgramValidationContext

.. autoclass:: ExpertProgramConfigError

.. autoclass:: ExpertProgramDecodeError

.. autoclass:: ExpertProgramValidationError

Compilation and environment integration
---------------------------------------

Compilation resolves static scene identities through the core
``SceneManifest`` and returns one already bounded, materialized program. The
environment adapter then creates a fresh canonical ``SkillRuntime`` for each
bridge.

.. autoclass:: ExpertProgramCompiler
   :members:

.. autoclass:: CompiledProgram
   :members:

.. autoclass:: ExpertProgramCompileError

.. autoclass:: ExpertProgramEnvironmentAdapter
   :members:

.. autoclass:: ExpertProgramEnvironmentFactory

.. autoclass:: ExpertProgramRuntimeAssembly
   :members:

.. autoclass:: PlanningObservationPort

Gym bridge ports
----------------

The bridge converts accepted runtime commands to lazy ``DemoSegment`` actions.
Only the normal environment executor calls ``env.step()``.

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

Simulation integration
----------------------

Simulation bindings translate explicit scene and robot declarations into the
core scene registry and robot skill profile. Generic non-control-part resources
use the core ``RobotResource`` type directly.

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

.. autoclass:: SimulationSegmentPolicyPort
   :members:

.. autofunction:: create_simulation_expert_program_adapter
