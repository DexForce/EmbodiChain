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

An action consumes an :class:`ActionInvocation` and a :class:`PlanningContext`
through :meth:`AtomicAction.plan`. Planning is side-effect free: it returns an
:class:`ActionPlan` with timed motion, completion criteria, diagnostics, and
uncommitted expected task-state effects. :class:`AtomicActionEngine` can compile
a static sequence; closed-loop execution belongs to an execution session.
"""

from __future__ import annotations

from .affordance import (
    Affordance,
    AntipodalAffordance,
    AssembleAffordance,
    InteractionPoints,
)
from .bindings import ActionBinding
from .core import ActionCfg, AtomicAction, ObjectSemantics, SkillDescriptor
from .effects import StateDelta
from .engine import (
    AtomicActionEngine,
    get_registered_actions,
    register_action,
    unregister_action,
)
from .execution import (
    ExecutionEvent,
    ExecutionEventKind,
    ExecutionSession,
    ExecutionStatus,
    ExecutionTick,
    JointCommand,
)
from .goals import ActionGoal, ObjectActionGoal, PoseGoalValue, SceneEntityPose
from .invocation import ActionInvocation
from .plans import (
    ActionPlan,
    CompiledTrajectory,
    CompletionCondition,
    CompletionConditionKind,
    PhaseSpec,
    PlannedPhase,
    PlannerDiagnostics,
    TimedTrajectory,
)
from .policies import MotionPolicy, RecoveryPolicy
from .runtime import ActionPlanningServices
from .primitives import (
    AssembleGoal,
    CoordinatedPickGoal,
    CoordinatedPickment,
    CoordinatedPickmentCfg,
    CoordinatedPlacement,
    CoordinatedPlacementCfg,
    CoordinatedPlacementGoal,
    EndEffectorPoseGoal,
    GraspGoal,
    HandOver,
    HandOverCfg,
    HeldObjectPoseGoal,
    JointPositionGoal,
    MoveEndEffector,
    MoveEndEffectorCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
    MoveJoints,
    MoveJointsCfg,
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
    PlaceGoal,
    Press,
    PressCfg,
    PressGoal,
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
from .trajectory import TrajectoryBuilder

__all__ = [
    "ActionBinding",
    "ActionCfg",
    "ActionGoal",
    "ActionInvocation",
    "ActionPlan",
    "ActionPlanningServices",
    "Affordance",
    "AntipodalAffordance",
    "AssembleAffordance",
    "AssembleGoal",
    "AtomicAction",
    "AtomicActionEngine",
    "CompiledTrajectory",
    "CompletionCondition",
    "CompletionConditionKind",
    "CoordinatedHeldObjectState",
    "CoordinatedPickGoal",
    "CoordinatedPickment",
    "CoordinatedPickmentCfg",
    "CoordinatedPlacement",
    "CoordinatedPlacementCfg",
    "CoordinatedPlacementGoal",
    "EndEffectorPoseGoal",
    "EntityState",
    "ExecutionEvent",
    "ExecutionEventKind",
    "ExecutionSession",
    "ExecutionStatus",
    "ExecutionTick",
    "GraspGoal",
    "HandOver",
    "HandOverCfg",
    "HeldObjectPoseGoal",
    "HeldObjectState",
    "InteractionPoints",
    "JointPositionGoal",
    "JointCommand",
    "MotionPolicy",
    "MoveEndEffector",
    "MoveEndEffectorCfg",
    "MoveHeldObject",
    "MoveHeldObjectCfg",
    "MoveJoints",
    "MoveJointsCfg",
    "ObjectActionGoal",
    "ObjectSemantics",
    "PhaseSpec",
    "PickUp",
    "PickUpCfg",
    "Place",
    "PlaceCfg",
    "PlaceGoal",
    "PlannedPhase",
    "PlannerDiagnostics",
    "PlanningContext",
    "PoseGoalValue",
    "Press",
    "PressCfg",
    "PressGoal",
    "RecoveryPolicy",
    "RobotObservation",
    "SceneSnapshot",
    "SceneEntityPose",
    "SkillDescriptor",
    "StateDelta",
    "TaskState",
    "TimedTrajectory",
    "TrajectoryBuilder",
    "get_registered_actions",
    "register_action",
    "unregister_action",
]
