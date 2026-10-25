# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Deprecated import compatibility for :mod:`embodichain.lab.sim.atomic_skills`.

The old package and every submodule reuse canonical module objects. This keeps
class identity, registry state and exception types intact during migration.
Legacy imports are retained for the 0.x migration window; new code should use
``atomic_skills``. Removal requires a separately announced breaking release.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from pkgutil import walk_packages as _walk_packages
import sys as _sys
import warnings as _warnings

__all__ = [
    "ActionBinding",
    "ActionControlOverrides",
    "ActionInvocation",
    "ActionOptions",
    "ActionPlan",
    "ActionPlanningServices",
    "Affordance",
    "AffordancePoseCandidates",
    "AffordanceSample",
    "AffordanceSamplingContext",
    "AntipodalAffordance",
    "ArticulationAffordanceGeometry",
    "ArticulationGeometryProvider",
    "ArticulationJointGeometry",
    "AxisAlignAffordance",
    "create_rigidized_articulation_antipodal_affordance",
    "ArticulationJointState",
    "AssembleAffordance",
    "AssembleGoal",
    "AxisAlign",
    "AxisAlignGoal",
    "AxisAlignOptions",
    "AtomicAction",
    "AtomicActionEngine",
    "PlanTransform",
    "BUILTIN_ACTION_TYPES",
    "BATCH_INVERSE_KINEMATICS_CAPABILITY",
    "CARTESIAN_POSE_CAPABILITY",
    "CompiledTrajectory",
    "CommandAcknowledgement",
    "CommandAckStatus",
    "CommandDispatch",
    "CommandOperation",
    "CommandSink",
    "ControlCommand",
    "ControlPartCommandProfile",
    "SceneEntity",
    "create_rigidized_articulation_antipodal_semantics",
    "create_simulation_atomic_action_engine",
    "CoordinatedHeldObjectState",
    "CoordinatedPickGoal",
    "CoordinatedPickment",
    "CoordinatedPickmentOptions",
    "CoordinatedPlacement",
    "CoordinatedPlacementGoal",
    "CoordinatedPlacementOptions",
    "DynamicCollisionMode",
    "DisjointResourceSlots",
    "DisjointSlotEndpoints",
    "EndEffectorPoseGoal",
    "EndpointBinding",
    "EndpointCommand",
    "EndpointCommandRouter",
    "EndpointCommandTransport",
    "ActionPlanTemplateAdapter",
    "EntityState",
    "EffectExpectationResult",
    "EffectVerificationRequest",
    "EffectVerificationRequirement",
    "EffectVerificationResult",
    "PhysicalEvidenceBatch",
    "PhysicalEvidenceFrame",
    "PhysicalEvidenceProvider",
    "PhysicalEvidenceRequest",
    "EffectVerifier",
    "PhysicalEffectVerifier",
    "ExecutionClock",
    "ExecutionEvent",
    "ExecutionEventKind",
    "ExecutionPlanAttempt",
    "ExecutionRunner",
    "ExecutionRunnerCfg",
    "ExecutionSession",
    "ExecutionStatus",
    "ExecutionTick",
    "HeldObjectGuardRequest",
    "HeldObjectGuardResult",
    "PhaseEffectGateRequest",
    "PhaseEffectGateRequirement",
    "PhaseEffectGateResult",
    "PhaseEffectGateVerifier",
    "PhysicalPhaseEffectGateVerifier",
    "PhysicalHeldObjectGuardVerifier",
    "HeldObjectGuardVerifier",
    "EndpointTrackingChannelBinding",
    "EndpointTrackingFeedbackAddress",
    "FeedbackTerminalAcceptance",
    "GRASP_COMMAND",
    "GRASP_CAPABILITY",
    "GraspGoal",
    "HandOver",
    "HandOverGoal",
    "HandOverOptions",
    "HeldObjectPoseGoal",
    "HeldObjectState",
    "FORWARD_KINEMATICS_CAPABILITY",
    "INVERSE_KINEMATICS_CAPABILITY",
    "InFlightTrackingPolicy",
    "InteractionPoints",
    "JointPositionGoal",
    "JointPositionCommand",
    "JointPositionPayload",
    "JointPositionTarget",
    "JOINT_POSITION_CAPABILITY",
    "JOINT_POSITION_CHANNEL",
    "JointPositionTrackingMetric",
    "JointPositionTrackingState",
    "MotionPolicy",
    "MonotonicExecutionClock",
    "MoveEndEffector",
    "MoveEndEffectorOptions",
    "MoveHeldObject",
    "MoveHeldObjectOptions",
    "MoveJoints",
    "MoveJointsOptions",
    "ObjectActionGoal",
    "ObjectSemantics",
    "OpenDoor",
    "OpenDoorAffordance",
    "OpenDoorGoal",
    "OpenDoorOptions",
    "OPEN_COMMAND",
    "ObservationProvider",
    "ObservedArticulationJointState",
    "PARK_COMMAND",
    "PickUp",
    "PickUpOptions",
    "Place",
    "PlaceGoal",
    "PlaceOptions",
    "Pour",
    "PourGoal",
    "PourOptions",
    "PlannerDiagnostics",
    "PlanningFailure",
    "PlanningContext",
    "PoseGoalValue",
    "PoseTrackingMetric",
    "PoseTrackingState",
    "Press",
    "PressAffordance",
    "PressGoal",
    "PressOptions",
    "PushObject",
    "PushObjectGoal",
    "PushObjectOptions",
    "PushObjectToolCalibration",
    "SlideAffordance",
    "Slide",
    "SlideGoal",
    "SlideJointTarget",
    "SlideOptions",
    "Twist",
    "TwistGoal",
    "TwistOptions",
    "RecoveryPolicy",
    "RigidObjectSceneProvider",
    "RigidObjectSceneProviderCfg",
    "ResolvedActionRequest",
    "RobotObservation",
    "RuntimeCommandFrame",
    "RuntimeCommandPayload",
    "RuntimeEndpointTarget",
    "RunnerStatus",
    "RunnerStep",
    "RunnerStepCallback",
    "SceneProvider",
    "SceneSnapshot",
    "SceneSnapshotSupplier",
    "SceneEntityPose",
    "SkillDescriptor",
    "SkillBindingContract",
    "SkillEndpointRequirement",
    "SkillResourceSlot",
    "StateDelta",
    "SimulationExecutionAdapter",
    "sample_initial_articulation_geometry",
    "TaskState",
    "TimedCommandSequence",
    "TimedTerminalAcceptance",
    "TimedTrackingSequence",
    "TimedTrajectory",
    "TerminalAcceptance",
    "TrajectorySegment",
    "TrackingCommandProjector",
    "TrackingEvaluation",
    "TrackingEvaluatorRegistry",
    "TrackingFeedbackAddress",
    "TrackingFeedbackBatch",
    "TrackingFeedbackProvider",
    "TrackingFeedbackProviderRegistry",
    "TrackingFeedbackSourceRef",
    "TrackingFrame",
    "TrackingMetricCfg",
    "TrackingMetricEvaluator",
    "TrackingPolicy",
    "TrackingProjectorRef",
    "TrackingProjectorRegistry",
    "TrackingRuntime",
    "TrackingSetpoint",
    "TrackingState",
    "BASE_POSE_CHANNEL",
    "JointPositionTrackingEvaluator",
    "JointPositionTrackingProjector",
    "PlanningContextTrackingFeedbackProvider",
    "PoseTrackingEvaluator",
    "WHOLE_BODY_POSE_CHANNEL",
    "WholeBodyPoseTrackingEvaluator",
    "WholeBodyPoseTrackingMetric",
    "WholeBodyPoseTrackingState",
    "TwistAffordance",
]

_RENAMES = {
    "AtomicAction": "AtomicSkill",
    "AtomicActionEngine": "AtomicSkillEngine",
    "ActionInvocation": "SkillInvocation",
    "ResolvedActionRequest": "ResolvedSkillRequest",
    "ActionPlan": "SkillPlan",
    "ActionOptions": "SkillOptions",
    "ActionBinding": "SkillBinding",
    "ActionControlOverrides": "SkillControlOverrides",
    "ActionPlanningServices": "SkillPlanningServices",
    "ActionPlanTemplateAdapter": "SkillPlanTemplateAdapter",
    "ObjectActionGoal": "ObjectSkillGoal",
    "BUILTIN_ACTION_TYPES": "BUILTIN_SKILL_TYPES",
    "create_simulation_atomic_action_engine": "create_simulation_atomic_skill_engine",
}

_warnings.warn(
    "embodichain.lab.sim.atomic_actions is deprecated; use "
    "embodichain.lab.sim.atomic_skills. See the Atomic Skill migration guide.",
    DeprecationWarning,
    stacklevel=2,
)
_canonical = _import_module("embodichain.lab.sim.atomic_skills")
# Import through canonical names before registering aliases, so Python never
# evaluates a module twice under different package names.
_modules = [_canonical] + [
    _import_module(info.name)
    for info in _walk_packages(_canonical.__path__, _canonical.__name__ + ".")
]
for _module in _modules:
    for _old_name, _new_name in _RENAMES.items():
        if hasattr(_module, _new_name):
            setattr(_module, _old_name, getattr(_module, _new_name))
    if _module is not _canonical:
        _sys.modules[_module.__name__.replace("atomic_skills", "atomic_actions", 1)] = (
            _module
        )
for _name in __all__:
    globals()[_name] = getattr(_canonical, _RENAMES.get(_name, _name))
# Preserve package attribute access such as atomic_actions.primitives as well
# as direct imports of the corresponding aliased submodule.
for _module in _modules[1:]:
    if _module.__name__.rpartition(".")[0] == _canonical.__name__:
        globals()[_module.__name__.rpartition(".")[2]] = _module
