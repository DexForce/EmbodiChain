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

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, TYPE_CHECKING, Union

from embodichain.lab.sim.common import BatchEntity
from embodichain.utils import configclass

from .affordance import Affordance

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator


# =============================================================================
# ObjectSemantics
# =============================================================================


@dataclass
class ObjectSemantics:
    """Semantic information about an interaction target."""

    affordance: Affordance
    """Affordance data describing how the object can be interacted with."""

    geometry: dict[str, Any]
    """Non-affordance geometric metadata (e.g., bounding_box). Mesh tensors live
    on AntipodalAffordance, not here."""

    properties: dict[str, Any] = field(default_factory=dict)
    """Physical properties: mass, friction, etc."""

    label: str = "none"
    """Object category label (e.g., 'mug', 'apple')."""

    entity: BatchEntity | None = None
    """Optional reference to the simulation entity for this object."""

    def __post_init__(self) -> None:
        # Bind only the label onto the affordance for convenience. DO NOT
        # alias the geometry dict — that was the footgun fixed by this redesign.
        self.affordance.object_label = self.label


# =============================================================================
# Typed targets
# =============================================================================


@dataclass(frozen=True)
class PoseTarget:
    """End-effector pose target. Used by MoveAction and PlaceAction."""

    xpos: torch.Tensor
    """(4, 4) or (n_envs, 4, 4) homogeneous transform."""


@dataclass(frozen=True)
class GraspTarget:
    """Pickup target. The grasp pose is solved from the affordance + entity at execute time."""

    semantics: ObjectSemantics


@dataclass(frozen=True)
class HeldObjectTarget:
    """Move the currently-held object to a desired object pose."""

    object_target_pose: torch.Tensor
    """(4, 4) or (n_envs, 4, 4) target pose for the held object."""


Target = Union[PoseTarget, GraspTarget, HeldObjectTarget]


# =============================================================================
# World state threaded between actions
# =============================================================================


@dataclass
class HeldObjectState:
    """State of an object currently held by the robot."""

    semantics: ObjectSemantics
    """Semantics of the held object."""

    object_to_eef: torch.Tensor
    """Batched transform from object frame to end-effector frame, shape [n_envs, 4, 4]."""

    grasp_xpos: torch.Tensor
    """Batched end-effector pose used to grasp the object, shape [n_envs, 4, 4]."""


@dataclass
class WorldState:
    """State the engine threads through a sequence of actions."""

    last_qpos: torch.Tensor
    """Robot joint positions at the start of the next action, shape [n_envs, robot.dof]."""

    held_object: HeldObjectState | None = None
    """Object currently held by the gripper, or None."""


@dataclass
class ActionResult:
    """Return value of every AtomicAction.execute call."""

    success: bool
    """Whether the action produced a valid full-DoF trajectory."""

    trajectory: torch.Tensor
    """Full-robot trajectory, shape (n_envs, n_waypoints, robot.dof)."""

    next_state: WorldState
    """World state to feed into the next action."""


# =============================================================================
# Configuration base
# =============================================================================


@configclass
class ActionCfg:
    """Configuration shared by all atomic actions."""

    name: str = "default"
    control_part: str = "arm"
    interpolation_type: str = "linear"
    velocity_limit: float | None = None
    acceleration_limit: float | None = None


# =============================================================================
# AtomicAction ABC (slim)
# =============================================================================


class AtomicAction(ABC):
    """Abstract base for atomic actions.

    Subclasses declare ``TargetType`` to advertise the concrete target dataclass
    they accept. ``execute`` is the only required method; ``validate`` has been
    dropped from the contract in this redesign.
    """

    TargetType: ClassVar[type]
    """Concrete target dataclass accepted by ``execute``."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: ActionCfg | None = None,
    ) -> None:
        self.motion_generator = motion_generator
        self.cfg = cfg if cfg is not None else ActionCfg()
        self.robot = motion_generator.robot
        self.device = self.robot.device
        self.control_part = self.cfg.control_part

    @abstractmethod
    def execute(self, target: Target, state: WorldState) -> ActionResult:
        """Plan and return a full-DoF trajectory for this action.

        Args:
            target: Typed target dataclass; must be an instance of ``self.TargetType``.
            state: World state inherited from the previous action (or the engine seed).

        Returns:
            ActionResult with the planned trajectory and the successor world state.
        """


__all__ = [
    "ActionCfg",
    "ActionResult",
    "AtomicAction",
    "GraspTarget",
    "HeldObjectState",
    "HeldObjectTarget",
    "ObjectSemantics",
    "PoseTarget",
    "Target",
    "WorldState",
]
