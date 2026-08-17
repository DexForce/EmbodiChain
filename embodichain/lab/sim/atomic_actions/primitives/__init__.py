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

"""Built-in atomic action primitive implementations."""

from __future__ import annotations

from ..core import AtomicAction
from .coordinated_pickment import (
    CoordinatedPickGoal,
    CoordinatedPickment,
    CoordinatedPickmentOptions,
)
from .coordinated_placement import (
    CoordinatedPlacement,
    CoordinatedPlacementGoal,
    CoordinatedPlacementOptions,
)
from .hand_over import HandOver, HandOverOptions
from .move_end_effector import (
    EndEffectorPoseGoal,
    MoveEndEffector,
    MoveEndEffectorOptions,
)
from .move_held_object import (
    HeldObjectPoseGoal,
    MoveHeldObject,
    MoveHeldObjectOptions,
)
from .move_joints import JointPositionGoal, MoveJoints, MoveJointsOptions
from .pick_up import GraspGoal, PickUp, PickUpOptions
from .place import AssembleGoal, Place, PlaceGoal, PlaceOptions
from .press import Press, PressGoal, PressOptions
from .press_button import PressButton, PressButtonGoal, PressButtonOptions
from .turn_knob import TurnKnob, TurnKnobGoal, TurnKnobOptions

BUILTIN_ACTION_TYPES: tuple[type[AtomicAction], ...] = (
    MoveEndEffector,
    MoveJoints,
    PickUp,
    MoveHeldObject,
    Place,
    Press,
    PressButton,
    TurnKnob,
    CoordinatedPickment,
    CoordinatedPlacement,
    HandOver,
)
"""Built-in action implementations instantiated once per action engine."""

__all__ = [
    "AssembleGoal",
    "BUILTIN_ACTION_TYPES",
    "CoordinatedPickGoal",
    "CoordinatedPickment",
    "CoordinatedPickmentOptions",
    "CoordinatedPlacement",
    "CoordinatedPlacementGoal",
    "CoordinatedPlacementOptions",
    "EndEffectorPoseGoal",
    "GraspGoal",
    "HandOver",
    "HandOverOptions",
    "HeldObjectPoseGoal",
    "JointPositionGoal",
    "MoveEndEffector",
    "MoveEndEffectorOptions",
    "MoveHeldObject",
    "MoveHeldObjectOptions",
    "MoveJoints",
    "MoveJointsOptions",
    "PickUp",
    "PickUpOptions",
    "Place",
    "PlaceGoal",
    "PlaceOptions",
    "Press",
    "PressButton",
    "PressButtonGoal",
    "PressButtonOptions",
    "PressGoal",
    "PressOptions",
    "TurnKnob",
    "TurnKnobGoal",
    "TurnKnobOptions",
]
