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

This module provides a unified interface for atomic actions like reach, grasp,
move, etc., with support for semantic object understanding and extensible
custom action registration.
"""

from .core import (
    Affordance,
    GraspPose,
    InteractionPoints,
    ObjectSemantics,
    ActionCfg,
    AtomicAction,
)
from .actions import (
    ReachAction,
    GraspAction,
    ReleaseAction,
    MoveAction,
)
from .engine import AtomicActionEngine, register_action, unregister_action, get_registered_actions

__all__ = [
    # Core classes
    "Affordance",
    "GraspPose",
    "InteractionPoints",
    "ObjectSemantics",
    "ActionCfg",
    "AtomicAction",
    # Action implementations
    "ReachAction",
    "GraspAction",
    "ReleaseAction",
    "MoveAction",
    # Engine
    "AtomicActionEngine",
    "register_action",
    "unregister_action",
    "get_registered_actions",
]
