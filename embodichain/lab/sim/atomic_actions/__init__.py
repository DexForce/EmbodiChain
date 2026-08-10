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
    AssembleAffordance,
    InteractionPoints,
)
from .bindings import ActionBinding, ResolvedActionBinding, ResolvedControlPart
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
    ExecutionEvent,
    ExecutionEventKind,
    ExecutionSession,
    ExecutionStatus,
    ExecutionTick,
    JointCommand,
)
from .goals import ActionGoal, ObjectActionGoal, PoseGoalValue, SceneEntityPose
from .invocation import ActionInvocation, ActionOptions, ResolvedActionRequest
from .plans import (
    ActionPlan,
    CompiledTrajectory,
    PlannerDiagnostics,
    TimedTrajectory,
    TrajectorySegment,
)
from .policies import DynamicCollisionMode, MotionPolicy, RecoveryPolicy
from .runtime import ActionPlanningServices
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
    CoordinatedHeldObjectState,
    EntityState,
    HeldObjectState,
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
    "AssembleAffordance",
    "AssembleGoal",
    "AtomicAction",
    "AtomicActionEngine",
    "BUILTIN_ACTION_TYPES",
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
    "EndEffectorPoseGoal",
    "EntityState",
    "EffectVerificationRequest",
    "EffectVerifier",
    "ExecutionClock",
    "ExecutionEvent",
    "ExecutionEventKind",
    "ExecutionRunner",
    "ExecutionRunnerCfg",
    "ExecutionSession",
    "ExecutionStatus",
    "ExecutionTick",
    "GRASP_COMMAND",
    "GraspGoal",
    "HandOver",
    "HandOverOptions",
    "HeldObjectPoseGoal",
    "HeldObjectState",
    "InteractionPoints",
    "JointPositionGoal",
    "JointCommand",
    "JointPositionCommand",
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
    "ResolvedActionBinding",
    "ResolvedControlPart",
    "RobotObservation",
    "RunnerStatus",
    "RunnerStep",
    "RunnerStepCallback",
    "SceneProvider",
    "SceneSnapshot",
    "SceneSnapshotSupplier",
    "SceneEntityPose",
    "SkillDescriptor",
    "StateDelta",
    "SimulationExecutionAdapter",
    "TaskState",
    "TimedTrajectory",
    "TrajectorySegment",
    "get_registered_actions",
    "register_action",
    "unregister_action",
]
