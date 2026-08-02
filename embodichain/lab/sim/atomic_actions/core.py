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
    from embodichain.lab.sim.planners import MotionGenerator, PlanOptions


def _resolve_runtime_device(device: torch.device | str) -> torch.device:
    """Resolve an indexless CUDA device to the active concrete GPU index."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return resolved


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
    """Object-to-end-effector transform, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    grasp_xpos: torch.Tensor
    """Grasp pose, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    env_mask: torch.Tensor | None = None
    """Environments in which the held-object relation is active, shape ``(n_envs,)``."""

    def __post_init__(self) -> None:
        object_batch_size = _validate_held_pose(
            self.object_to_eef, "HeldObjectState.object_to_eef"
        )
        grasp_batch_size = _validate_held_pose(
            self.grasp_xpos, "HeldObjectState.grasp_xpos"
        )
        known_batch_sizes = {
            size for size in (object_batch_size, grasp_batch_size) if size is not None
        }
        if len(known_batch_sizes) > 1:
            raise ValueError(
                "HeldObjectState pose tensors must use the same batch size, "
                f"got {object_batch_size} and {grasp_batch_size}."
            )
        if self.grasp_xpos.device != self.object_to_eef.device:
            raise ValueError("HeldObjectState pose tensors must use the same device.")
        batch_size = next(iter(known_batch_sizes), None)
        self.env_mask = _normalize_optional_env_mask(
            self.env_mask,
            batch_size=batch_size,
            device=self.object_to_eef.device,
            name="HeldObjectState.env_mask",
        )


@dataclass(slots=True, eq=False)
class CoordinatedHeldObjectState:
    """State of a single object jointly held by two robot hands."""

    semantics: ObjectSemantics
    """Semantic object currently held by the two grippers."""

    left_object_to_eef: torch.Tensor
    """Left object-to-end-effector transform, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    right_object_to_eef: torch.Tensor
    """Right object-to-end-effector transform, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    left_grasp_xpos: torch.Tensor
    """Left grasp pose, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    right_grasp_xpos: torch.Tensor
    """Right grasp pose, shape ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    env_mask: torch.Tensor | None = None
    """Environments in which the coordinated hold is active, shape ``(n_envs,)``."""

    def __post_init__(self) -> None:
        pose_fields = {
            "left_object_to_eef": self.left_object_to_eef,
            "right_object_to_eef": self.right_object_to_eef,
            "left_grasp_xpos": self.left_grasp_xpos,
            "right_grasp_xpos": self.right_grasp_xpos,
        }
        batch_sizes = {
            name: _validate_held_pose(value, f"CoordinatedHeldObjectState.{name}")
            for name, value in pose_fields.items()
        }
        known_batch_sizes = {size for size in batch_sizes.values() if size is not None}
        if len(known_batch_sizes) > 1:
            raise ValueError(
                "CoordinatedHeldObjectState pose tensors must use the same batch "
                f"size, got {batch_sizes}."
            )
        devices = {value.device for value in pose_fields.values()}
        if len(devices) != 1:
            raise ValueError(
                "CoordinatedHeldObjectState pose tensors must use the same device."
            )
        batch_size = next(iter(known_batch_sizes), None)
        self.env_mask = _normalize_optional_env_mask(
            self.env_mask,
            batch_size=batch_size,
            device=self.left_object_to_eef.device,
            name="CoordinatedHeldObjectState.env_mask",
        )


def _validate_held_pose(value: torch.Tensor, name: str) -> int | None:
    """Validate a held-state pose and return its explicit batch size, if any."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.shape == (4, 4):
        return None
    if value.dim() != 3 or value.shape[-2:] != (4, 4) or value.shape[0] == 0:
        raise ValueError(
            f"{name} must have shape (4, 4) or (n_envs, 4, 4) with n_envs > 0, "
            f"got {tuple(value.shape)}."
        )
    return int(value.shape[0])


def _normalize_optional_env_mask(
    value: torch.Tensor | None,
    *,
    batch_size: int | None,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    """Normalize a mask when a held-state batch can already be inferred."""
    if batch_size is None and value is None:
        return None
    if batch_size is None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None.")
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool, got {value.dtype}.")
        if value.dim() != 1 or value.shape[0] == 0:
            raise ValueError(
                f"{name} must have shape (n_envs,) with n_envs > 0, "
                f"got {tuple(value.shape)}."
            )
        batch_size = int(value.shape[0])
    return _normalize_env_mask(
        value,
        batch_size=batch_size,
        device=device,
        name=name,
    )


def _normalize_env_mask(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Return an owned boolean environment mask with shape ``(batch_size,)``."""
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None.")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {value.dtype}.")
    if value.shape != (batch_size,):
        raise ValueError(
            f"{name} must have shape ({batch_size},), got {tuple(value.shape)}."
        )
    return value.to(device=device).clone()


def _broadcast_held_pose(
    value: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Resolve an optionally batched held-state pose to a world-state batch."""
    pose_batch_size = _validate_held_pose(value, name)
    if value.device != device:
        raise ValueError(f"{name} must use the same device as WorldState.last_qpos.")
    if pose_batch_size is None:
        return value.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if pose_batch_size != batch_size:
        raise ValueError(
            "Held-object state batch size must match WorldState.last_qpos; "
            f"expected {batch_size}, got {pose_batch_size}."
        )
    return value


def _normalize_held_object_state(
    value: HeldObjectState,
    *,
    batch_size: int,
    device: torch.device,
) -> HeldObjectState:
    """Return a held-object state normalized to a world-state batch."""
    object_batch_size = _validate_held_pose(
        value.object_to_eef, "HeldObjectState.object_to_eef"
    )
    grasp_batch_size = _validate_held_pose(
        value.grasp_xpos, "HeldObjectState.grasp_xpos"
    )
    if (
        object_batch_size == batch_size
        and grasp_batch_size == batch_size
        and value.object_to_eef.device == device
        and value.grasp_xpos.device == device
        and isinstance(value.env_mask, torch.Tensor)
        and value.env_mask.dtype == torch.bool
        and value.env_mask.shape == (batch_size,)
        and value.env_mask.device == device
    ):
        return value
    return HeldObjectState(
        semantics=value.semantics,
        object_to_eef=_broadcast_held_pose(
            value.object_to_eef,
            batch_size=batch_size,
            device=device,
            name="HeldObjectState.object_to_eef",
        ),
        grasp_xpos=_broadcast_held_pose(
            value.grasp_xpos,
            batch_size=batch_size,
            device=device,
            name="HeldObjectState.grasp_xpos",
        ),
        env_mask=_normalize_env_mask(
            value.env_mask,
            batch_size=batch_size,
            device=device,
            name="HeldObjectState.env_mask",
        ),
    )


def _normalize_coordinated_held_object_state(
    value: CoordinatedHeldObjectState,
    *,
    batch_size: int,
    device: torch.device,
) -> CoordinatedHeldObjectState:
    """Return a coordinated-held state normalized to a world-state batch."""
    pose_fields = {
        "left_object_to_eef": value.left_object_to_eef,
        "right_object_to_eef": value.right_object_to_eef,
        "left_grasp_xpos": value.left_grasp_xpos,
        "right_grasp_xpos": value.right_grasp_xpos,
    }
    pose_batch_sizes = {
        name: _validate_held_pose(
            pose,
            f"CoordinatedHeldObjectState.{name}",
        )
        for name, pose in pose_fields.items()
    }
    if (
        all(size == batch_size for size in pose_batch_sizes.values())
        and all(pose.device == device for pose in pose_fields.values())
        and isinstance(value.env_mask, torch.Tensor)
        and value.env_mask.dtype == torch.bool
        and value.env_mask.shape == (batch_size,)
        and value.env_mask.device == device
    ):
        return value
    normalized_poses = {
        name: _broadcast_held_pose(
            pose,
            batch_size=batch_size,
            device=device,
            name=f"CoordinatedHeldObjectState.{name}",
        )
        for name, pose in pose_fields.items()
    }
    return CoordinatedHeldObjectState(
        semantics=value.semantics,
        **normalized_poses,
        env_mask=_normalize_env_mask(
            value.env_mask,
            batch_size=batch_size,
            device=device,
            name="CoordinatedHeldObjectState.env_mask",
        ),
    )


def _merge_held_object_state(
    previous: HeldObjectState | None,
    candidate: HeldObjectState | None,
    update_mask: torch.Tensor,
) -> HeldObjectState | None:
    """Merge one held-object entry using a per-environment update mask."""
    if previous is not None:
        assert previous.env_mask is not None
    if candidate is not None:
        assert candidate.env_mask is not None
    if previous is None and candidate is None:
        return None
    if previous is None:
        assert candidate is not None
        env_mask = candidate.env_mask & update_mask
        if not env_mask.any():
            return None
        return HeldObjectState(
            semantics=candidate.semantics,
            object_to_eef=candidate.object_to_eef,
            grasp_xpos=candidate.grasp_xpos,
            env_mask=env_mask,
        )
    if candidate is None:
        env_mask = previous.env_mask & ~update_mask
        if not env_mask.any():
            return None
        return HeldObjectState(
            semantics=previous.semantics,
            object_to_eef=previous.object_to_eef,
            grasp_xpos=previous.grasp_xpos,
            env_mask=env_mask,
        )

    previous_retained = bool((previous.env_mask & ~update_mask).any().item())
    candidate_applied = bool((candidate.env_mask & update_mask).any().item())
    if (
        previous_retained
        and candidate_applied
        and previous.semantics is not candidate.semantics
    ):
        raise ValueError(
            "Cannot merge different held-object semantics for one control part "
            "across environments."
        )
    env_mask = torch.where(update_mask, candidate.env_mask, previous.env_mask)
    if not env_mask.any():
        return None
    selector = update_mask[:, None, None]
    return HeldObjectState(
        semantics=candidate.semantics if candidate_applied else previous.semantics,
        object_to_eef=torch.where(
            selector, candidate.object_to_eef, previous.object_to_eef
        ),
        grasp_xpos=torch.where(selector, candidate.grasp_xpos, previous.grasp_xpos),
        env_mask=env_mask,
    )


def _merge_coordinated_held_object_state(
    previous: CoordinatedHeldObjectState | None,
    candidate: CoordinatedHeldObjectState | None,
    update_mask: torch.Tensor,
) -> CoordinatedHeldObjectState | None:
    """Merge one coordinated-held entry using a per-environment update mask."""
    if previous is not None:
        assert previous.env_mask is not None
    if candidate is not None:
        assert candidate.env_mask is not None
    if previous is None and candidate is None:
        return None
    if previous is None:
        assert candidate is not None
        env_mask = candidate.env_mask & update_mask
        if not env_mask.any():
            return None
        return CoordinatedHeldObjectState(
            semantics=candidate.semantics,
            left_object_to_eef=candidate.left_object_to_eef,
            right_object_to_eef=candidate.right_object_to_eef,
            left_grasp_xpos=candidate.left_grasp_xpos,
            right_grasp_xpos=candidate.right_grasp_xpos,
            env_mask=env_mask,
        )
    if candidate is None:
        env_mask = previous.env_mask & ~update_mask
        if not env_mask.any():
            return None
        return CoordinatedHeldObjectState(
            semantics=previous.semantics,
            left_object_to_eef=previous.left_object_to_eef,
            right_object_to_eef=previous.right_object_to_eef,
            left_grasp_xpos=previous.left_grasp_xpos,
            right_grasp_xpos=previous.right_grasp_xpos,
            env_mask=env_mask,
        )

    previous_retained = bool((previous.env_mask & ~update_mask).any().item())
    candidate_applied = bool((candidate.env_mask & update_mask).any().item())
    if (
        previous_retained
        and candidate_applied
        and previous.semantics is not candidate.semantics
    ):
        raise ValueError(
            "Cannot merge different coordinated-held semantics for one control-part "
            "pair across environments."
        )
    env_mask = torch.where(update_mask, candidate.env_mask, previous.env_mask)
    if not env_mask.any():
        return None
    selector = update_mask[:, None, None]
    return CoordinatedHeldObjectState(
        semantics=candidate.semantics if candidate_applied else previous.semantics,
        left_object_to_eef=torch.where(
            selector, candidate.left_object_to_eef, previous.left_object_to_eef
        ),
        right_object_to_eef=torch.where(
            selector, candidate.right_object_to_eef, previous.right_object_to_eef
        ),
        left_grasp_xpos=torch.where(
            selector, candidate.left_grasp_xpos, previous.left_grasp_xpos
        ),
        right_grasp_xpos=torch.where(
            selector, candidate.right_grasp_xpos, previous.right_grasp_xpos
        ),
        env_mask=env_mask,
    )


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

    def __post_init__(self) -> None:
        if not isinstance(self.last_qpos, torch.Tensor):
            raise TypeError("WorldState.last_qpos must be a torch.Tensor.")
        if (
            self.last_qpos.dim() != 2
            or self.last_qpos.shape[0] == 0
            or self.last_qpos.shape[1] == 0
        ):
            raise ValueError(
                "WorldState.last_qpos must have shape (n_envs, robot_dof) with "
                f"both dimensions non-zero, got {tuple(self.last_qpos.shape)}."
            )
        held_objects: dict[str, HeldObjectState] = {}
        for control_part, held in self.held_objects.items():
            if not isinstance(control_part, str) or not control_part:
                raise TypeError(
                    "WorldState.held_objects keys must be non-empty strings."
                )
            if not isinstance(held, HeldObjectState):
                raise TypeError(
                    "WorldState.held_objects values must be HeldObjectState instances."
                )
            held_objects[control_part] = _normalize_held_object_state(
                held,
                batch_size=self.batch_size,
                device=self.last_qpos.device,
            )
        coordinated_held_objects: dict[tuple[str, str], CoordinatedHeldObjectState] = {}
        for control_parts, held in self.coordinated_held_objects.items():
            if (
                not isinstance(control_parts, tuple)
                or len(control_parts) != 2
                or not all(isinstance(part, str) and part for part in control_parts)
            ):
                raise TypeError(
                    "WorldState.coordinated_held_objects keys must be pairs of "
                    "non-empty control-part names."
                )
            if not isinstance(held, CoordinatedHeldObjectState):
                raise TypeError(
                    "WorldState.coordinated_held_objects values must be "
                    "CoordinatedHeldObjectState instances."
                )
            coordinated_held_objects[control_parts] = (
                _normalize_coordinated_held_object_state(
                    held,
                    batch_size=self.batch_size,
                    device=self.last_qpos.device,
                )
            )
        self.held_objects = held_objects
        self.coordinated_held_objects = coordinated_held_objects

    @property
    def batch_size(self) -> int:
        """Number of vectorized environments represented by this state."""
        return int(self.last_qpos.shape[0])

    @property
    def robot_dof(self) -> int:
        """Number of robot joint-position columns represented by this state."""
        return int(self.last_qpos.shape[1])

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

    def masked_merge(
        self,
        candidate: WorldState,
        update_mask: torch.Tensor,
    ) -> WorldState:
        """Merge a candidate successor for selected environments.

        Args:
            candidate: Candidate successor returned by an atomic action.
            update_mask: Boolean tensor of shape ``(n_envs,)``. Candidate robot
                and held-object state is committed only where this mask is true.

        Returns:
            A new state that preserves the current values in unselected rows.

        Raises:
            TypeError: If ``candidate`` or ``update_mask`` has an invalid type.
            ValueError: If state shapes, devices, or semantics are incompatible.
        """
        if not isinstance(candidate, WorldState):
            raise TypeError("candidate must be a WorldState instance.")
        if candidate.last_qpos.shape != self.last_qpos.shape:
            raise ValueError(
                "Candidate WorldState.last_qpos must match the current shape, "
                f"got {tuple(candidate.last_qpos.shape)} and "
                f"{tuple(self.last_qpos.shape)}."
            )
        if candidate.last_qpos.device != self.last_qpos.device:
            raise ValueError("WorldState values being merged must use the same device.")
        update_mask = _normalize_env_mask(
            update_mask,
            batch_size=self.batch_size,
            device=self.last_qpos.device,
            name="update_mask",
        )

        held_objects: dict[str, HeldObjectState] = {}
        held_keys = dict.fromkeys((*self.held_objects, *candidate.held_objects))
        for key in held_keys:
            merged = _merge_held_object_state(
                self.held_objects.get(key), candidate.held_objects.get(key), update_mask
            )
            if merged is not None:
                held_objects[key] = merged

        coordinated_held_objects: dict[tuple[str, str], CoordinatedHeldObjectState] = {}
        coordinated_keys = dict.fromkeys(
            (*self.coordinated_held_objects, *candidate.coordinated_held_objects)
        )
        for key in coordinated_keys:
            merged = _merge_coordinated_held_object_state(
                self.coordinated_held_objects.get(key),
                candidate.coordinated_held_objects.get(key),
                update_mask,
            )
            if merged is not None:
                coordinated_held_objects[key] = merged

        return WorldState(
            last_qpos=torch.where(
                update_mask[:, None], candidate.last_qpos, self.last_qpos
            ),
            held_objects=held_objects,
            coordinated_held_objects=coordinated_held_objects,
        )


@dataclass(slots=True, eq=False)
class ActionResult:
    """Return value of every AtomicAction.execute call."""

    success: torch.Tensor
    """Per-environment planning success, normalized to shape ``(n_envs,)``."""

    trajectory: torch.Tensor
    """Full-robot trajectory, shape (n_envs, n_waypoints, robot.dof)."""

    next_state: WorldState
    """World state to feed into the next action."""

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, torch.Tensor):
            raise TypeError("ActionResult.trajectory must be a torch.Tensor.")
        if self.trajectory.dim() != 3:
            raise ValueError(
                "ActionResult.trajectory must have shape "
                f"(n_envs, n_waypoints, robot_dof), got {tuple(self.trajectory.shape)}."
            )
        if not isinstance(self.next_state, WorldState):
            raise TypeError("ActionResult.next_state must be a WorldState instance.")
        expected_shape = (
            self.next_state.batch_size,
            self.next_state.robot_dof,
        )
        if (self.trajectory.shape[0], self.trajectory.shape[2]) != expected_shape:
            raise ValueError(
                "ActionResult trajectory batch/DoF must match next_state.last_qpos; "
                f"got trajectory {tuple(self.trajectory.shape)} and state "
                f"{tuple(self.next_state.last_qpos.shape)}."
            )
        if self.trajectory.device != self.next_state.last_qpos.device:
            raise ValueError(
                "ActionResult trajectory and next_state.last_qpos must use the "
                "same device."
            )

        batch_size = self.next_state.batch_size
        if isinstance(self.success, bool):
            success = torch.full(
                (batch_size,),
                self.success,
                dtype=torch.bool,
                device=self.trajectory.device,
            )
        elif isinstance(self.success, torch.Tensor):
            if self.success.dtype != torch.bool:
                raise TypeError(
                    "ActionResult.success must have dtype torch.bool, "
                    f"got {self.success.dtype}."
                )
            success = self.success.to(device=self.trajectory.device)
            if success.dim() == 0 or success.shape == (1,):
                success = success.reshape(1).expand(batch_size)
            if success.shape != (batch_size,):
                raise ValueError(
                    f"ActionResult.success must have shape ({batch_size},), "
                    f"got {tuple(success.shape)}."
                )
            success = success.clone()
        else:
            raise TypeError("ActionResult.success must be a bool or torch.Tensor.")
        self.success = success

    @property
    def success_all(self) -> bool:
        """True only if all environments succeeded."""
        return bool(torch.all(self.success).item())

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
    """Interpolation policy. Only ``"linear"`` is currently implemented."""

    velocity_limit: float | None = None
    acceleration_limit: float | None = None
    plan_opts: PlanOptions | None = None
    """Optional planner-specific options copied for each motion-generator call."""

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
        if self.interpolation_type != "linear":
            raise ValueError(
                "interpolation_type currently supports only 'linear', "
                f"but got {self.interpolation_type!r}."
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
