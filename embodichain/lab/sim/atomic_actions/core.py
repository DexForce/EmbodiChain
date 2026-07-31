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
from typing import Any, ClassVar, Generic, Mapping, TYPE_CHECKING, TypeVar

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
# Target foundation
# =============================================================================


class ActionTarget:
    """Open marker base for atomic-action target value objects.

    Third-party actions should define a target dataclass that inherits from this
    class. The engine performs the action-specific runtime check using
    :attr:`AtomicAction.TargetType`; the marker keeps the public engine contract
    open to targets outside the built-in set.
    """

    __slots__ = ()


TargetT = TypeVar("TargetT", bound=ActionTarget)


def _validate_pose_tensor(
    value: torch.Tensor,
    name: str,
    *,
    allow_waypoints: bool,
) -> None:
    """Validate the environment-independent part of a pose tensor contract."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    valid_dims = {2, 3, 4} if allow_waypoints else {2, 3}
    if value.dim() not in valid_dims or value.shape[-2:] != (4, 4):
        supported = "(4, 4), (n_envs, 4, 4)"
        if allow_waypoints:
            supported += ", or (n_envs, n_waypoint, 4, 4)"
        raise ValueError(
            f"{name} must have shape {supported}, got {tuple(value.shape)}."
        )


# ``Target`` used to be a closed union of built-in target classes. Keep the
# public name as an open compatibility alias so extension targets are accepted.
Target = ActionTarget


# =============================================================================
# World state threaded between actions
# =============================================================================


@dataclass(slots=True, eq=False)
class HeldObjectState:
    """State of an object currently held by the robot."""

    semantics: ObjectSemantics
    """Semantics of the held object."""

    object_to_eef: torch.Tensor
    """Batched transform from object frame to end-effector frame, shape [n_envs, 4, 4]."""

    grasp_xpos: torch.Tensor
    """Batched end-effector pose used to grasp the object, shape [n_envs, 4, 4]."""


@dataclass(slots=True, eq=False)
class CoordinatedHeldObjectState:
    """State of a single object jointly held by two robot hands."""

    semantics: ObjectSemantics
    """Semantic object currently held by the two grippers."""

    left_object_to_eef: torch.Tensor
    """Transform from object frame to left end-effector frame, shape ``[n_envs, 4, 4]``."""

    right_object_to_eef: torch.Tensor
    """Transform from object frame to right end-effector frame, shape ``[n_envs, 4, 4]``."""

    left_grasp_xpos: torch.Tensor
    """Left end-effector grasp pose for the shared object, shape ``[n_envs, 4, 4]``."""

    right_grasp_xpos: torch.Tensor
    """Right end-effector grasp pose for the shared object, shape ``[n_envs, 4, 4]``."""


@dataclass(slots=True, eq=False)
class WorldState:
    """State the engine threads through a sequence of actions."""

    last_qpos: torch.Tensor
    """Robot joint positions at the start of the next action, shape [n_envs, robot.dof]."""

    held_objects: dict[str, HeldObjectState] = field(default_factory=dict)
    """Objects held by individual control parts, keyed by control-part name."""

    coordinated_held_objects: dict[tuple[str, str], CoordinatedHeldObjectState] = field(
        default_factory=dict
    )
    """Objects jointly held by two control parts, keyed by their ordered pair."""

    def get_held_object(self, control_part: str) -> HeldObjectState | None:
        """Return the object held by ``control_part``, if any."""
        return self.held_objects.get(control_part)

    def get_coordinated_held_object(
        self,
        first_control_part: str,
        second_control_part: str,
    ) -> CoordinatedHeldObjectState | None:
        """Return the object jointly held by an ordered control-part pair."""
        return self.coordinated_held_objects.get(
            (first_control_part, second_control_part)
        )

    def with_updates(
        self,
        *,
        last_qpos: torch.Tensor | None = None,
        held_objects: Mapping[str, HeldObjectState] | None = None,
        coordinated_held_objects: (
            Mapping[tuple[str, str], CoordinatedHeldObjectState] | None
        ) = None,
    ) -> WorldState:
        """Return a successor state without aliasing held-state dictionaries."""
        return WorldState(
            last_qpos=self.last_qpos if last_qpos is None else last_qpos,
            held_objects=dict(
                self.held_objects if held_objects is None else held_objects
            ),
            coordinated_held_objects=dict(
                self.coordinated_held_objects
                if coordinated_held_objects is None
                else coordinated_held_objects
            ),
        )


@dataclass(slots=True, eq=False)
class ActionResult:
    """Return value of every AtomicAction.execute call."""

    success: bool | torch.Tensor
    """Whether the action produced a valid full-DoF trajectory.
    Can be a bool or a per-environment boolean tensor of shape (n_envs,)."""

    trajectory: torch.Tensor
    """Full-robot trajectory, shape (n_envs, n_waypoints, robot.dof)."""

    next_state: WorldState
    """World state to feed into the next action."""

    @property
    def success_all(self) -> bool:
        """True only if all environments succeeded."""
        if isinstance(self.success, torch.Tensor):
            return bool(torch.all(self.success).item())
        return bool(self.success)

    def __bool__(self) -> bool:
        import warnings as _w

        _w.warn(
            "ActionResult bool() is deprecated; use .success_all",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.success_all


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
    motion_source: str = "ik_interp"
    """Trajectory source: 'ik_interp' (default, batched IK + linear interp)
    or 'motion_gen' (batched MotionGenerator)."""

    def __post_init__(self) -> None:
        valid_sources = {"ik_interp", "motion_gen"}
        if self.motion_source not in valid_sources:
            raise ValueError(
                f"motion_source must be one of {sorted(valid_sources)}, "
                f"but got {self.motion_source!r}."
            )


# =============================================================================
# AtomicAction ABC (slim)
# =============================================================================


class AtomicAction(Generic[TargetT], ABC):
    """Abstract base for atomic actions.

    Subclasses declare ``TargetType`` to advertise the concrete target dataclass
    they accept. ``execute`` is the only required method; ``validate`` has been
    dropped from the contract in this redesign.
    """

    TargetType: ClassVar[type[ActionTarget] | tuple[type[ActionTarget], ...]]
    """Concrete target dataclass or dataclasses accepted by ``execute``."""

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
    def execute(self, target: TargetT, state: WorldState) -> ActionResult:
        """Plan and return a full-DoF trajectory for this action.

        Args:
            target: Typed target dataclass; must be an instance of ``self.TargetType``.
            state: World state inherited from the previous action (or the engine seed).

        Returns:
            ActionResult with the planned trajectory and the successor world state.
        """


__all__ = [
    "ActionTarget",
    "ActionCfg",
    "ActionResult",
    "AtomicAction",
    "CoordinatedHeldObjectState",
    "HeldObjectState",
    "ObjectSemantics",
    "Target",
    "TargetT",
    "WorldState",
]
