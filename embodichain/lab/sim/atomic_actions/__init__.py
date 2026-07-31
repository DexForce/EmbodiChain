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

"""Atomic action abstraction layer for embodied AI motion generation.

This module provides a unified interface for the atomic motion primitives
(``move_end_effector``, ``move_joints``, ``pick_up``, ``move_held_object``,
``place``, ``press``, ``coordinated_pickment``, ``coordinated_placement``),
with typed targets, a ``WorldState`` threaded across sequenced actions, and
extensible custom action registration.
"""

from __future__ import annotations

from .affordance import (
    Affordance,
    AntipodalAffordance,
    InteractionPoints,
)
from .core import (
    ActionTarget,
    ActionCfg,
    ActionResult,
    AtomicAction,
    CoordinatedHeldObjectState,
    HeldObjectState,
    ObjectSemantics,
    Target,
    WorldState,
)
from .engine import (
    AtomicActionEngine,
    register_action,
    unregister_action,
    get_registered_actions,
)
from .targets import ObjectActionTarget
from .primitives import (
    CoordinatedPickTarget,
    CoordinatedPickment,
    CoordinatedPickmentCfg,
    CoordinatedPickmentTarget,
    CoordinatedPlacement,
    CoordinatedPlacementCfg,
    CoordinatedPlacementTarget,
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectPoseTarget,
    JointPositionTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
    MoveJoints,
    MoveJointsCfg,
    NamedJointPositionTarget,
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
    PlaceTarget,
    Press,
    PressCfg,
    PressTarget,
)
from .trajectory import TrajectoryBuilder

BuiltinTarget = (
    EndEffectorPoseTarget
    | JointPositionTarget
    | NamedJointPositionTarget
    | GraspTarget
    | HeldObjectPoseTarget
    | PlaceTarget
    | PressTarget
    | CoordinatedPickTarget
    | CoordinatedPlacementTarget
)
"""Union of target types shipped by EmbodiChain.

Use :class:`ActionTarget` rather than this closed union at extension boundaries.
"""

__all__ = [
    # Core classes
    "ActionTarget",
    "Affordance",
    "AntipodalAffordance",
    "InteractionPoints",
    "ObjectSemantics",
    "ObjectActionTarget",
    "HeldObjectState",
    "CoordinatedHeldObjectState",
    "HeldObjectPoseTarget",
    "JointPositionTarget",
    "NamedJointPositionTarget",
    "EndEffectorPoseTarget",
    "PlaceTarget",
    "PressTarget",
    "CoordinatedPickTarget",
    "CoordinatedPickmentTarget",
    "CoordinatedPlacementTarget",
    "GraspTarget",
    "Target",
    "BuiltinTarget",
    "WorldState",
    "ActionResult",
    "ActionCfg",
    "AtomicAction",
    # Action implementations
    "CoordinatedPickment",
    "CoordinatedPlacement",
    "MoveEndEffector",
    "MoveJoints",
    "MoveHeldObject",
    "PickUp",
    "Place",
    "Press",
    "CoordinatedPickmentCfg",
    "CoordinatedPlacementCfg",
    "MoveEndEffectorCfg",
    "MoveJointsCfg",
    "MoveHeldObjectCfg",
    "PickUpCfg",
    "PlaceCfg",
    "PressCfg",
    # Engine
    "AtomicActionEngine",
    "register_action",
    "unregister_action",
    "get_registered_actions",
    # Trajectory helpers
    "TrajectoryBuilder",
]
