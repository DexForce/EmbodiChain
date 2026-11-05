embodichain.lab.sim.atomic_actions
==================================

.. automodule:: embodichain.lab.sim.atomic_actions
   :members:
   :no-index:

   .. rubric:: Planning contracts

   .. autosummary::

      ActionBinding
      EndpointBinding
      RuntimeEndpointTarget
      JointPositionTarget
      ControlCommand
      JointPositionCommand
      ControlPartCommandProfile
      ActionControlOverrides
      ActionInvocation
      PhaseEffectGateRequirement
      ResolvedActionRequest
      ActionOptions
      MotionPolicy
      RecoveryPolicy
      RobotObservation
      TaskState
      SceneSnapshot
      SceneEntityPose
      PlanningContext
      StateDelta
      TimedTrajectory
      RuntimeCommandPayload
      JointPositionPayload
      EndpointCommand
      RuntimeCommandFrame
      TimedCommandSequence
      TrajectorySegment
      PlannerDiagnostics
      PlanningFailure
      ActionPlan
      CompiledTrajectory

   .. rubric:: Articulation geometry adaptation

   .. autosummary::

      ArticulationAffordanceGeometry
      ArticulationGeometryProvider
      ArticulationJointGeometry
      create_rigidized_articulation_antipodal_affordance
      sample_initial_articulation_geometry

   .. rubric:: Semantic resource contracts

   .. autosummary::

      SkillDescriptor
      SkillBindingContract
      SkillResourceSlot
      SkillEndpointRequirement
      DisjointSlotEndpoints
      DisjointResourceSlots

   .. rubric:: Execution contracts

   .. autosummary::

      AtomicAction
      AtomicActionEngine
      PlanTransform
      ExecutionSession
      ExecutionRunner
      ExecutionRunnerCfg
      RunnerStep
      RunnerStatus
      ObservationProvider
      CommandSink
      EndpointCommandTransport
      EndpointCommandRouter
      CommandAcknowledgement
      CommandAckStatus
      CommandDispatch
      CommandOperation
      ExecutionClock
      create_simulation_atomic_action_engine
      create_rigidized_articulation_antipodal_semantics
      SceneEntity
      SimulationExecutionAdapter
      ExecutionTick
      EffectVerificationRequest
      EffectVerificationRequirement
      EffectVerificationResult
      PhysicalEvidenceBatch
      PhysicalEvidenceFrame
      PhysicalEvidenceProvider
      PhysicalEvidenceRequest
      PhaseEffectGateRequest
      PhaseEffectGateResult
      PhysicalEffectVerifier
      PhysicalPhaseEffectGateVerifier
      HeldObjectGuardRequest
      HeldObjectGuardResult
      PhysicalHeldObjectGuardVerifier
      ExecutionPlanAttempt
      ExecutionEvent
      ExecutionEventKind
      ExecutionStatus

   .. rubric:: Built-in goals and actions

   .. autosummary::

      EndEffectorPoseGoal
      JointPositionGoal
      GraspGoal
      HandOverGoal
      AxisAlignGoal
      AxisAlignOptions
      AxisAlignAffordance
      HeldObjectPoseGoal
      PourGoal
      PourOptions
      PushObjectGoal
      PushObjectOptions
      PushObjectToolCalibration
      PlaceGoal
      AssembleGoal
      PressGoal
      PressOptions
      PressAffordance
      SlideGoal
      SlideOptions
      SlideAffordance
      OpenDoorGoal
      OpenDoorOptions
      OpenDoorAffordance
      TwistGoal
      TwistOptions
      TwistAffordance
      CoordinatedPickGoal
      CoordinatedPlacementGoal
      MoveEndEffector
      MoveJoints
      PickUp
      AxisAlign
      MoveHeldObject
      Pour
      PushObject
      Place
      Press
      Slide
      OpenDoor
      Twist
      CoordinatedPickment
      CoordinatedPlacement
      HandOver

.. toctree::
   :maxdepth: 1
   :hidden:

   embodichain.lab.sim.atomic_actions.primitives
   embodichain.lab.sim.atomic_actions.tracking

.. currentmodule:: embodichain.lab.sim.atomic_actions

Articulation geometry adaptation
--------------------------------

The adapter converts deterministic articulation meshes, FK, and parent-joint
topology into sampled Atomic Action affordance geometry. Initial state and scale
are explicit inputs; the simulation object does not own sampling or semantic
geometry keys.

.. currentmodule:: embodichain.lab.sim.atomic_actions.articulation_geometry

.. autoclass:: ArticulationGeometryProvider
   :members:

.. autoclass:: ArticulationJointGeometry
   :members:

.. autoclass:: ArticulationAffordanceGeometry
   :members:

.. autofunction:: create_rigidized_articulation_antipodal_affordance

.. autofunction:: sample_initial_articulation_geometry

.. currentmodule:: embodichain.lab.sim.atomic_actions

Semantic resource contracts
---------------------------

.. autoclass:: SkillDescriptor
   :members:

.. autoclass:: SkillBindingContract
   :members:

.. autoclass:: SkillResourceSlot
   :members:

.. autoclass:: SkillEndpointRequirement
   :members:

.. autoclass:: DisjointSlotEndpoints
   :members:

.. autoclass:: DisjointResourceSlots
   :members:

Standard capability identifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autodata:: JOINT_POSITION_CAPABILITY

.. autodata:: CARTESIAN_POSE_CAPABILITY

.. autodata:: FORWARD_KINEMATICS_CAPABILITY

.. autodata:: INVERSE_KINEMATICS_CAPABILITY

.. autodata:: BATCH_INVERSE_KINEMATICS_CAPABILITY

.. autodata:: GRASP_CAPABILITY

Planning and state
------------------

.. autoclass:: ActionBinding
   :members:

.. autoclass:: EndpointBinding
   :members:

.. autoclass:: RuntimeEndpointTarget
   :members:

.. autoclass:: JointPositionTarget
   :members:

.. autoclass:: ControlCommand
   :members:

.. autoclass:: JointPositionCommand
   :members:

.. autoclass:: ControlPartCommandProfile
   :members:

.. autoclass:: ActionControlOverrides
   :members:

.. autoclass:: ActionInvocation
   :members:

.. autoclass:: ResolvedActionRequest
   :members:

.. autoclass:: ActionOptions
   :members:

.. autoclass:: MotionPolicy
   :members:
   :exclude-members: __init__, copy, replace, to_dict

.. autoclass:: RecoveryPolicy
   :members:
   :exclude-members: __init__, copy, replace, to_dict

.. autoclass:: PlanningContext
   :members:

.. autoclass:: RobotObservation
   :members:

.. autoclass:: TaskState
   :members:

.. autoclass:: SceneSnapshot
   :members:

.. autoclass:: SceneEntityPose
   :members:

.. autoclass:: StateDelta
   :members:

.. autoclass:: TimedTrajectory
   :members:

.. autoclass:: PlanningFailure
   :members:

.. autoclass:: RuntimeCommandPayload
   :members:

.. autoclass:: JointPositionPayload
   :members:

.. autoclass:: EndpointCommand
   :members:

.. autoclass:: RuntimeCommandFrame
   :members:

.. autoclass:: TimedCommandSequence
   :members:

.. autoclass:: ActionPlan
   :members:

.. autoclass:: ActionPlanTemplateAdapter
   :members:

.. currentmodule:: embodichain.lab.sim.atomic_actions.trajectory_adapter

.. autosummary::
   :nosignatures:

   ActionPlanTemplateAdapter

.. currentmodule:: embodichain.lab.sim.atomic_actions

Engine and execution
--------------------

.. autoclass:: AtomicAction
   :members:

.. autoclass:: AtomicActionEngine
   :members:

.. autodata:: PlanTransform

.. autoclass:: ExecutionSession
   :members:

.. autoclass:: ExecutionRunner
   :members:

.. autoclass:: ExecutionRunnerCfg
   :members:
   :exclude-members: __init__, copy, replace, to_dict

.. autoclass:: ObservationProvider
   :members:

.. autoclass:: PhysicalEvidenceProvider
   :members:

.. autoclass:: PhysicalEvidenceRequest
   :members:

.. autoclass:: PhysicalEvidenceBatch
   :members:

.. autoclass:: PhysicalEvidenceFrame
   :members:

.. autoclass:: CommandSink
   :members:

.. autoclass:: EndpointCommandTransport
   :members:

.. autoclass:: EndpointCommandRouter
   :members:

.. autoclass:: ExecutionClock
   :members:

.. autoclass:: MonotonicExecutionClock
   :members:

.. autofunction:: create_simulation_atomic_action_engine

.. autofunction:: create_rigidized_articulation_antipodal_semantics

.. autoclass:: SceneEntity
   :members:

.. autoclass:: SimulationExecutionAdapter
   :members:

.. autoclass:: CommandAcknowledgement
   :members:

.. autoclass:: CommandAckStatus
   :members:

.. autoclass:: CommandDispatch
   :members:

.. autoclass:: CommandOperation
   :members:

.. autoclass:: RunnerStep
   :members:

.. autoclass:: RunnerStatus
   :members:

.. autoclass:: ExecutionTick
   :members:

.. autoclass:: ExecutionEvent
   :members:

.. autoclass:: ExecutionEventKind
   :members:

.. autoclass:: ExecutionStatus
   :members:

.. autoclass:: EffectVerificationRequest
   :members:

.. autoclass:: EffectVerificationRequirement
   :members:

.. autoclass:: EffectVerificationResult
   :members:

.. autodata:: PhysicalEffectVerifier

.. autodata:: PhysicalPhaseEffectGateVerifier

.. autodata:: PhysicalHeldObjectGuardVerifier

.. autoclass:: ExecutionPlanAttempt
   :members:

Semantic objects and helpers
----------------------------

.. autoclass:: ObjectSemantics
   :members:

.. autoclass:: HeldObjectState
   :members:

Verification implementation module
----------------------------------

.. currentmodule:: embodichain.lab.sim.atomic_actions.evidence

.. autosummary::

   PhysicalEvidenceBatch
   PhysicalEvidenceFrame
   PhysicalEvidenceProvider
   PhysicalEvidenceRequest

.. currentmodule:: embodichain.lab.sim.atomic_actions.runner

.. autosummary::

   PhysicalEffectVerifier
   PhysicalPhaseEffectGateVerifier
   PhysicalHeldObjectGuardVerifier

Evidence-backed execution keeps acquisition separate from the existing
symbolic effect verifier. The runner collects one aligned frame after each
fresh observation and passes it only to the explicitly selected evidence-aware
verifier.

.. currentmodule:: embodichain.lab.sim.atomic_actions.verification

.. autosummary::

   EffectExpectationResult
   EffectVerificationRequest
   EffectVerificationResult
   HeldObjectGuardRequest
   HeldObjectGuardResult
   PhaseEffectGateRequest
   PhaseEffectGateResult

Named held-object transport
---------------------------

``HeldObjectPoseGoal.object_id`` optionally identifies the exact object already
held by the bound resource. A named goal validates that identity before native
or external nominal planning and declares the unchanged attachment as its
terminal effect. The execution owner verifies this effect before committing it;
planning does not establish physical attachment. Motion and timing use the same
path as an unnamed goal. Omitting the field preserves the existing effectless
``MoveHeldObject`` behavior. Task Program's routed transport uses named goals
and additionally selects a verified input-hold guard for its transport segment.


Scoped grasp preparation
------------------------

``ActionPlanningServices.prepare_grasp_query()`` resolves the declared hand
calibration by endpoint target ID and binds Scene capture to GraspKit's explicit
``ClosingContactPolicy`` and the existing isolated Newton backend. It accepts no
experiment revision selector and installs no alternate skill registration.
Ordinary PickUp defaults remain unchanged. Source-compatible capture uses the
caller's explicit 65-sample hand declaration.

Use this service as a context manager. Closing, optional release-opening and
rigid-scene query resources close together on normal exit, failure or cancellation;
borrowed simulation entities are never destroyed. ``GraspOpeningTarget`` carries
already-grounded poses from the same observation scope. An undeclared hand
calibration fails before capture. Diagnostic metadata never selects policy.

.. currentmodule:: embodichain.lab.sim.atomic_actions.runtime

.. autosummary::
   :nosignatures:

   GraspOpeningTarget

.. autoclass:: GraspOpeningTarget
   :members:
   :no-index:

.. automethod:: ActionPlanningServices.prepare_grasp_query

Explicit source contact-policy composition
------------------------------------------

The shared ``GraspOpeningTarget`` definition lives in ``grasp_geometry``;
``atomic_actions`` and ``runtime`` retain the same public class identity. Its
poses and metadata stay owned and bound to one preparation scope.

.. automodule:: embodichain.lab.sim.atomic_actions.grasp_geometry
   :members:

``source_native_contact`` maps the explicitly selected preserved source revision
to GraspKit's existing typed ``ClosingContactPolicy`` and the requested FCL
service. Closing, opening and rigid-clearance algorithms remain solely in
GraspKit. Source builders consume the existing Default/B1 native capture API,
retain the explicit hand sampling grid and original distances, and never step
physics or infer a Task Program policy. Missing FCL fails explicitly; this is
geometric qualification, not physical acquisition/release evidence.

Close each query with its context manager or a ``finally`` block. It releases
only owned query resources, never borrowed simulation entities. Ordinary
``prepare_grasp_query`` keeps its existing Newton backend and defaults.

.. automodule:: embodichain.lab.sim.atomic_actions.source_native_contact
   :members:


Explicit Pick and Place planning
--------------------------------

``PickUpOptions`` and ``PlaceOptions`` select action geometry. Motion sampling,
segment recovery and timing remain in ``MotionPolicy``; bounded candidate and
release-path recovery use the exact ``PickPlanningRecovery`` or
``PlacePlanningRecovery`` value in ``RecoveryPolicy.planning_recovery``. Ordinary
defaults do not enable these capabilities. The descriptor advertises only each
installed skill's actual recovery type and segment handler.

The engine borrows typed planning ports. Their owned snapshots contain geometric
facts and exact occurrence identities, never Task Engine or semantic compiler
objects. Missing required geometry or a different robot rejects before the
corresponding query. A continuation preview predicts a complete release using
the same engine and observation; it authorizes no task-state effect or physical
success. Native measured paths retain their explicit single-environment scope.

``AxisAlignOptions`` rejects inherited Pick-only requirements that its planner
does not implement. Explicit drawer Place and held transport use the installed bounded
drawer handler and declared physical provider; missing inputs fail before
corresponding planning. Existing external nominal capabilities remain
available; explicit measured source policies reject incompatible nominal input.

.. currentmodule:: embodichain.lab.sim.atomic_actions

.. autosummary::
   :nosignatures:

   PickPlanningRecovery
   PlacePlanningRecovery
   PourGraspProjectionInputs
   PourGraspProjectionProvider
   PickClosingQueryProvider
   PickRigidSceneProvider
   PlaceSceneInputs
   PlaceInsideLanding
   PlaceSceneProvider

Explicit Press planning recovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``PressPlanningRecovery`` selects the existing Press skill's failure-only B1
recovery. Nominal planning and at most one declared contact-roll alternative share
the original two-complete-plan budget and RNG progression. Failed Cartesian
phases may retry through the shared unchanged-target, unchanged-time IK resolver.
The final emitted full-robot trajectory retains velocity qualification after
resampling and transforms. Missing motion or excessive velocity remains failure.
Default ``PressOptions`` enables no contact-roll alternatives or recovery.

``validate_press_extra_distance`` accepts only an optional finite addition in
``[0, 0.010]`` metres. It does not alter the authored stroke or determine physical
contact, release or task success. Configured recovery accepts only
``{kind: press_bounded, revision: 1}``; budgets and callbacks cannot be supplied
through that declaration.

.. autosummary::
   :nosignatures:

   PressPlanningRecovery
   validate_press_extra_distance

.. autoclass:: PressPlanningRecovery
   :members:

.. autofunction:: validate_press_extra_distance

Curobo queries
^^^^^^^^^^^^^^

The joint-prefix adapter preserves the exact observed full robot state and backend options while borrowing Motion's owned query lifetime.

.. currentmodule:: embodichain.lab.sim.atomic_actions.curobo_queries

.. autosummary::
   :nosignatures:

   CuroboJointPrefixPlanner

.. autoclass:: CuroboJointPrefixPlanner
   :members:

Measured rigid
^^^^^^^^^^^^^^

Physical hosts supply same-observation rigid geometry and query sessions. Atomic skills own candidate selection, plan transformation and bounded recovery.

.. currentmodule:: embodichain.lab.sim.atomic_actions.measured_rigid

.. autosummary::
   :nosignatures:

   MeasuredInsidePlanningInputs
   MeasuredRigidPlanningSession
   MeasuredRigidPlanningProvider

.. autoclass:: MeasuredInsidePlanningInputs
   :members:

.. autoclass:: MeasuredRigidPlanningSession
   :members:

.. autoclass:: MeasuredRigidPlanningProvider
   :members:

Place continuation
^^^^^^^^^^^^^^^^^^

Release prediction binds distinct source, optional transport and release occurrences. Predictions cannot collect future physical facts or commit effects.

.. currentmodule:: embodichain.lab.sim.atomic_actions.place_continuation

.. autosummary::
   :nosignatures:

   PlaceContinuationRoute
   PlaceContinuationSession
   PlaceContinuationProvider

.. autoclass:: PlaceContinuationRoute
   :members:

.. autoclass:: PlaceContinuationSession
   :members:

.. autoclass:: PlaceContinuationProvider
   :members:

Planning probes
^^^^^^^^^^^^^^^

These ports separate read-only geometry and observed backend queries from Pick and Place phase construction. Missing declared inputs fail explicitly.

.. currentmodule:: embodichain.lab.sim.atomic_actions.planning_probes

.. autosummary::
   :nosignatures:

   JointPrefixPlanner
   CuroboWorldObservationProvider
   ObservedCuroboQueryService
   PourGraspProjectionProvider
   PickClosingQueryProvider
   PickRigidSceneProvider
   PlaceInsideLanding
   PlaceInsideLandingProvider
   PlaceSceneInputs
   PlaceSceneProvider

.. autoclass:: JointPrefixPlanner
   :members:

.. autoclass:: CuroboWorldObservationProvider
   :members:

.. autoclass:: ObservedCuroboQueryService
   :members:

.. autoclass:: PourGraspProjectionProvider
   :members:

.. autoclass:: PickClosingQueryProvider
   :members:

.. autoclass:: PickRigidSceneProvider
   :members:

.. autoclass:: PlaceInsideLanding
   :members:

.. autoclass:: PlaceInsideLandingProvider
   :members:

.. autoclass:: PlaceSceneInputs
   :members:

.. autoclass:: PlaceSceneProvider
   :members:

Pour projection
^^^^^^^^^^^^^^^

Numeric downstream-pour operands constrain Pick grasp sampling without exposing a task graph or changing Pour execution.

.. currentmodule:: embodichain.lab.sim.atomic_actions.pour_projection

.. autosummary::
   :nosignatures:

   PourGraspProjectionInputs

.. autoclass:: PourGraspProjectionInputs
   :members:


Held transport, Pour, Idle and released-hand plans
--------------------------------------------------

``MoveHeldObjectOptions`` preserves the shared transport path unless an explicit
alignment/staging rule is selected. Command observations come from Lab's owned
``RobotCommandObservation`` value; the action consumes static retention authority
and current facts without changing a verified grasp transform. Observed hand
posture is an explicit option and does not prove physical attachment.

``JawPourGoal`` and ``RetainedPourGoal`` add the preserved ordered-pad and retained
arc constructions alongside ``PourGoal``. Relative ``JointOffsetTarget`` values
use the bound model joint; free-motion qualification checks the full command
grid, passive joints, original limits and scene queries. Protected-link offset
search remains in Motion.

``ReleasedHandClearanceRecovery`` and ``ReleasedHandParkRecovery`` require the
exact declared cleanup/Park occurrences, an empty resource and explicit collision
queries. Previews use the same engine and complete pipeline, preserve observed
identity and restore their random streams. They send no commands and commit no
effects. Missing or foreign suffix authority rejects. Drawer transport and
articulation withdrawal/authorization are unavailable in this layer.

.. currentmodule:: embodichain.lab.sim.atomic_actions

.. autosummary::
   :nosignatures:

   MoveHeldObjectPlanningRecovery
   JointOffsetTarget
   JawPourGoal
   RetainedPourGoal
   ReleasedHandClearanceRecovery
   ReleasedHandParkRecovery
   ReleasedHandContinuationSession
   ReleasedHandContinuationProvider

Held planning ports
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: embodichain.lab.sim.atomic_actions.planning_probes

.. autosummary::
   :nosignatures:

   HeldStagingBinding
   HeldStagingProvider
   HeldArticulationStagingProvider
   RetainedCommandAuthority
   RetainedCommandProvider

The typed staging ports declare an exact rigid subject and prepare its owned
distal geometry only when requested. The optional articulation query capability
uses Motion's existing observation scope; it installs no articulation action.
Retained-command authority names serial provenance and monitor identities and
must be checked against an actual command observation by its host.

.. autoclass:: HeldStagingBinding
   :members:

.. autoclass:: HeldStagingProvider
   :members:

.. autoclass:: HeldArticulationStagingProvider
   :members:

.. autoclass:: RetainedCommandAuthority
   :members:

.. autoclass:: RetainedCommandProvider
   :members:

Released continuation authority
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: embodichain.lab.sim.atomic_actions.released_continuation

.. autosummary::
   :nosignatures:

   ReleasedHandClearanceRecovery
   ReleasedHandParkRecovery
   ReleasedHandContinuationSession
   ReleasedHandContinuationProvider

.. autoclass:: ReleasedHandClearanceRecovery
   :members:

.. autoclass:: ReleasedHandParkRecovery
   :members:

.. autoclass:: ReleasedHandContinuationSession
   :members:

.. autoclass:: ReleasedHandContinuationProvider
   :members:


Bounded Twist planning and measured feedback
--------------------------------------------

``TwistPlanningRecovery`` retains the fixed B1/env0, float32, fixed-grid sequential
IK route. The skill reads an explicit ``TwistPlanningProvider`` session for the
same robot, observation, invocation and revision, selects the original ordered
contact candidates and closes the borrowed query on every exit. Final command
qualification and parent dependency proofs reject without changing the source
budget or issuing a physical command. Missing providers do not select a fallback.

The recovery cursor separates planning outcome commit, installed-plan adoption
and measured feedback. Preview forks own separate cursor identities. Geometric
qualification and successful planning do not establish measured contact or task
success; event rules retain incomplete evidence and release failures explicitly.

.. automodule:: embodichain.lab.sim.atomic_actions.twist_planning
   :members:

.. automodule:: embodichain.lab.sim.atomic_actions.primitives.twist_recovery
   :members:

.. automodule:: embodichain.lab.sim.atomic_actions.articulation_evidence
   :members:

.. automodule:: embodichain.lab.sim.atomic_actions.control_validation
   :members:

.. automodule:: embodichain.lab.sim.atomic_actions.twist_feedback

Prismatic articulation and drawer planning
------------------------------------------

``SlidePlanningRecovery`` preserves explicit B1/env0 fixed-grid sequential IK,
the original segment retry grid, 390-frame bound, endpoint order, push/withdrawal
repair and RNG scopes. ``ArticulationWithdrawalPlanningRecovery`` and
``ArticulationParkPlanningRecovery`` consume actual installed predecessor facts.
Preview plans do not authorize later withdrawal or PARK. Generic PARK screening
and bounded vias remain implemented once in ``primitives/_released_park.py``.

Explicit drawer Place and held transport use declared ``DrawerPlanningProvider``
inputs. The complete-plan builder retains bounded entry ranking, upright release
lift choices, reverse entry, selected-transform rejection and containment checks
before motion. Missing geometry or native ownership remains unavailable.
``SlideHandClearanceProvider`` supplies measured preshape aperture facts; the
optional 85%-to-75% retry retains original candidate and RNG boundaries.

.. automodule:: embodichain.lab.sim.atomic_actions.articulation_continuation
   :members:

.. automodule:: embodichain.lab.sim.atomic_actions.drawer_planning
   :members:

.. currentmodule:: embodichain.lab.sim.atomic_actions.planning_probes

.. autoclass:: SlideHandClearanceProvider

Bounded handover and coordinated hold continuation
--------------------------------------------------

``HandOverPlanningRecovery`` names the preserved bounded receiver-direction and
exchange-station search. Select it together with explicit adaptive-receive
options; native receiver/scene screening additionally requires the exact borrowed
``HandOverSceneProvider``. The action closes each owned query, retains declared
participant resources and preserves source grasp/release/retreat timing. Screening
is geometric planning evidence and cannot certify a physical acquisition.

``CoordinatedPickmentOptions.continue_verified_holds`` keeps both verified grasp
transforms instead of resampling. ``prefer_cargo_clearance`` adds the explicit
observed-content preference to the existing paired sampler. Static cargo geometry
is copied at the engine boundary. Missing observations keep base costs; an
accepted envelope or preferred candidate does not establish retention.

These modes do not replace the default implementations, infer presets from task
names, or register another skill set. Unsupported source declarations remain
closed in Task Program configuration until the runtime-assembly owner is present.

.. currentmodule:: embodichain.lab.sim.atomic_actions

.. autosummary::
   :nosignatures:

   HandOverPlanningRecovery
   HandOverSceneProvider

.. currentmodule:: embodichain.lab.sim.atomic_actions.planning_probes

.. autosummary::
   :nosignatures:

   HandOverSceneProvider

.. autoclass:: HandOverSceneProvider
   :members:
