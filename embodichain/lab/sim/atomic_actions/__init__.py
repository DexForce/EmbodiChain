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

"""Typed planning contracts and built-in atomic actions.

The engine resolves an :class:`ActionInvocation` into a
:class:`ResolvedActionRequest`, which an action combines with a
:class:`PlanningContext` through :meth:`AtomicAction.plan`. Planning is
side-effect free: it returns an :class:`ActionPlan` with timed motion,
completion criteria, diagnostics, and uncommitted expected task-state effects.
:class:`AtomicActionEngine` can compile a static sequence. For closed-loop use,
:class:`ExecutionSession` owns recovery and invocation-revision state while
:class:`ExecutionRunner` connects it to observations, commands, and time.
"""

from __future__ import annotations

from .affordance import (
    Affordance,
    AntipodalAffordance,
    ArticulationOperationAffordance,
    ArticulationOperationTarget,
    AssembleAffordance,
    InteractionPoints,
)
from .bindings import (
    ActionBinding,
    EndpointBinding,
    JointPositionTarget,
    RuntimeEndpointTarget,
)
from .control import (
    ActionControlOverrides,
    ControlCommand,
    ControlPartCommandProfile,
    GRASP_COMMAND,
    JointPositionCommand,
    OPEN_COMMAND,
)
from .core import AtomicAction, ObjectSemantics, SkillDescriptor
from .effects import StateDelta
from .engine import (
    AtomicActionEngine,
    get_registered_actions,
    register_action,
    unregister_action,
)
from .execution import (
    EffectVerificationRequest,
    EffectVerificationResult,
    ExecutionEvent,
    ExecutionEventKind,
    ExecutionPlanAttempt,
    ExecutionSession,
    ExecutionStatus,
    ExecutionTick,
)
from .goals import (
    ActionGoal,
    ObjectActionGoal,
    PoseGoalValue,
    SceneArticulationOperationGeometry,
    SceneEntityPose,
)
from .invocation import ActionInvocation, ActionOptions, ResolvedActionRequest
from .plans import (
    ActionPlan,
    CompiledTrajectory,
    EffectVerificationRequirement,
    ExecutionFeedbackMode,
    PlannerDiagnostics,
    TimedTrajectory,
    TrajectorySegment,
)
from .policies import DynamicCollisionMode, MotionPolicy, RecoveryPolicy
from .requirements import (
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    DisjointResourceSlots,
    DisjointSlotEndpoints,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    INVERSE_KINEMATICS_CAPABILITY,
    JOINT_POSITION_CAPABILITY,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from .runtime import ActionPlanningServices
from .runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
    RuntimeCommandPayload,
    TimedCommandSequence,
)
from .transports import EndpointCommandRouter, EndpointCommandTransport
from .primitives import (
    AssembleGoal,
    BUILTIN_ACTION_TYPES,
    CoordinatedPickGoal,
    CoordinatedPickment,
    CoordinatedPickmentOptions,
    CoordinatedPlacement,
    CoordinatedPlacementGoal,
    CoordinatedPlacementOptions,
    EndEffectorPoseGoal,
    GraspGoal,
    HandOver,
    HandOverOptions,
    HeldObjectPoseGoal,
    JointPositionGoal,
    MoveEndEffector,
    MoveEndEffectorOptions,
    MoveHeldObject,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    OperateArticulation,
    OperateArticulationGoal,
    OperateArticulationOptions,
    PickUp,
    PickUpOptions,
    Place,
    PlaceGoal,
    PlaceOptions,
    Press,
    PressGoal,
    PressOptions,
)
from .runner import (
    CommandAcknowledgement,
    CommandAckStatus,
    CommandDispatch,
    CommandOperation,
    CommandSink,
    EffectVerifier,
    ExecutionClock,
    ExecutionRunner,
    ExecutionRunnerCfg,
    MonotonicExecutionClock,
    ObservationProvider,
    RunnerStatus,
    RunnerStep,
    RunnerStepCallback,
)
from .scene import SceneProvider
from .sim_adapter import (
    RigidObjectSceneProvider,
    RigidObjectSceneProviderCfg,
    SceneSnapshotSupplier,
    SimulationExecutionAdapter,
)
from .state import (
    ArticulationJointState,
    CoordinatedHeldObjectState,
    EntityState,
    HeldObjectState,
    ObservedArticulationJointState,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)

__all__ = [
    "ActionBinding",
    "ActionControlOverrides",
    "ActionGoal",
    "ActionInvocation",
    "ActionOptions",
    "ActionPlan",
    "ActionPlanningServices",
    "Affordance",
    "AntipodalAffordance",
    "ArticulationOperationAffordance",
    "ArticulationOperationTarget",
    "ArticulationJointState",
    "AssembleAffordance",
    "AssembleGoal",
    "AtomicAction",
    "AtomicActionEngine",
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
    "EntityState",
    "EffectVerificationRequest",
    "EffectVerificationRequirement",
    "EffectVerificationResult",
    "EffectVerifier",
    "ExecutionClock",
    "ExecutionFeedbackMode",
    "ExecutionEvent",
    "ExecutionEventKind",
    "ExecutionPlanAttempt",
    "ExecutionRunner",
    "ExecutionRunnerCfg",
    "ExecutionSession",
    "ExecutionStatus",
    "ExecutionTick",
    "GRASP_COMMAND",
    "GRASP_CAPABILITY",
    "GraspGoal",
    "HandOver",
    "HandOverOptions",
    "HeldObjectPoseGoal",
    "HeldObjectState",
    "FORWARD_KINEMATICS_CAPABILITY",
    "INVERSE_KINEMATICS_CAPABILITY",
    "InteractionPoints",
    "JointPositionGoal",
    "JointPositionCommand",
    "JointPositionPayload",
    "JointPositionTarget",
    "JOINT_POSITION_CAPABILITY",
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
    "OPEN_COMMAND",
    "ObservationProvider",
    "ObservedArticulationJointState",
    "OperateArticulation",
    "OperateArticulationGoal",
    "OperateArticulationOptions",
    "PickUp",
    "PickUpOptions",
    "Place",
    "PlaceGoal",
    "PlaceOptions",
    "PlannerDiagnostics",
    "PlanningContext",
    "PoseGoalValue",
    "Press",
    "PressGoal",
    "PressOptions",
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
    "SceneArticulationOperationGeometry",
    "SceneSnapshot",
    "SceneSnapshotSupplier",
    "SceneEntityPose",
    "SkillDescriptor",
    "SkillBindingContract",
    "SkillEndpointRequirement",
    "SkillResourceSlot",
    "StateDelta",
    "SimulationExecutionAdapter",
    "TaskState",
    "TimedCommandSequence",
    "TimedTrajectory",
    "TrajectorySegment",
    "get_registered_actions",
    "register_action",
    "unregister_action",
]
