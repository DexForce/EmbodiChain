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

"""Expected symbolic effects produced by side-effect-free action planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch

from .state import (
    CoordinatedHeldObjectState,
    HeldObjectState,
    TaskState,
    _normalize_coordinated_held,
    _normalize_held,
    _normalize_mask,
)


def _with_held_mask(
    value: HeldObjectState,
    env_mask: torch.Tensor,
) -> HeldObjectState:
    """Copy a held-object relation with a replacement mask."""
    return HeldObjectState(
        semantics=value.semantics,
        object_to_eef=value.object_to_eef,
        grasp_xpos=value.grasp_xpos,
        env_mask=env_mask,
    )


def _with_coordinated_mask(
    value: CoordinatedHeldObjectState,
    env_mask: torch.Tensor,
) -> CoordinatedHeldObjectState:
    """Copy a coordinated held-object relation with a replacement mask."""
    return CoordinatedHeldObjectState(
        semantics=value.semantics,
        left_object_to_eef=value.left_object_to_eef,
        right_object_to_eef=value.right_object_to_eef,
        left_grasp_xpos=value.left_grasp_xpos,
        right_grasp_xpos=value.right_grasp_xpos,
        env_mask=env_mask,
    )


def _merge_held(
    previous: HeldObjectState | None,
    candidate: HeldObjectState | None,
    update_mask: torch.Tensor,
) -> HeldObjectState | None:
    """Apply one optional held-object update per environment."""
    if previous is None and candidate is None:
        return None
    if previous is None:
        assert candidate is not None and candidate.env_mask is not None
        env_mask = candidate.env_mask & update_mask
        return _with_held_mask(candidate, env_mask) if env_mask.any() else None
    assert previous.env_mask is not None
    if candidate is None:
        env_mask = previous.env_mask & ~update_mask
        return _with_held_mask(previous, env_mask) if env_mask.any() else None
    assert candidate.env_mask is not None

    previous_retained = bool((previous.env_mask & ~update_mask).any().item())
    candidate_applied = bool((candidate.env_mask & update_mask).any().item())
    if (
        previous_retained
        and candidate_applied
        and previous.semantics is not candidate.semantics
    ):
        raise ValueError(
            "Cannot merge different held-object semantics for one resource "
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


def _merge_coordinated(
    previous: CoordinatedHeldObjectState | None,
    candidate: CoordinatedHeldObjectState | None,
    update_mask: torch.Tensor,
) -> CoordinatedHeldObjectState | None:
    """Apply one optional coordinated relation update per environment."""
    if previous is None and candidate is None:
        return None
    if previous is None:
        assert candidate is not None and candidate.env_mask is not None
        env_mask = candidate.env_mask & update_mask
        return _with_coordinated_mask(candidate, env_mask) if env_mask.any() else None
    assert previous.env_mask is not None
    if candidate is None:
        env_mask = previous.env_mask & ~update_mask
        return _with_coordinated_mask(previous, env_mask) if env_mask.any() else None
    assert candidate.env_mask is not None

    previous_retained = bool((previous.env_mask & ~update_mask).any().item())
    candidate_applied = bool((candidate.env_mask & update_mask).any().item())
    if (
        previous_retained
        and candidate_applied
        and previous.semantics is not candidate.semantics
    ):
        raise ValueError(
            "Cannot merge different coordinated held-object semantics for one "
            "resource pair across environments."
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


@dataclass(frozen=True, slots=True, eq=False)
class StateDelta:
    """Expected task-state changes that require post-execution verification.

    A mapping value of ``None`` removes the corresponding relation. Planning
    only declares this delta; an execution runtime applies it after verifying
    the semantic effect for the successful environment rows.
    """

    held_object_updates: Mapping[str, HeldObjectState | None] = field(
        default_factory=dict
    )
    """Per-resource attachment replacements or removals."""

    coordinated_held_object_updates: Mapping[
        tuple[str, str], CoordinatedHeldObjectState | None
    ] = field(default_factory=dict)
    """Per-resource-pair coordinated attachment replacements or removals."""

    def __post_init__(self) -> None:
        held = dict(self.held_object_updates)
        coordinated = dict(self.coordinated_held_object_updates)
        for resource, value in held.items():
            if not isinstance(resource, str) or not resource:
                raise ValueError(
                    "held_object_updates keys must be non-empty resource names."
                )
            if value is not None and not isinstance(value, HeldObjectState):
                raise TypeError(
                    "held_object_updates values must be HeldObjectState or None."
                )
        for resources, value in coordinated.items():
            if (
                not isinstance(resources, tuple)
                or len(resources) != 2
                or not all(isinstance(item, str) and item for item in resources)
            ):
                raise ValueError(
                    "coordinated_held_object_updates keys must be resource pairs."
                )
            if value is not None and not isinstance(value, CoordinatedHeldObjectState):
                raise TypeError(
                    "coordinated_held_object_updates values must be "
                    "CoordinatedHeldObjectState or None."
                )
        object.__setattr__(self, "held_object_updates", MappingProxyType(held))
        object.__setattr__(
            self,
            "coordinated_held_object_updates",
            MappingProxyType(coordinated),
        )

    @property
    def is_empty(self) -> bool:
        """Whether this delta declares no symbolic state changes."""
        return not self.held_object_updates and not self.coordinated_held_object_updates

    def apply(
        self,
        state: TaskState,
        update_mask: torch.Tensor,
    ) -> TaskState:
        """Apply expected effects to selected environment rows.

        This operation is used for hypothetical state propagation while
        compiling a sequence. A runtime must apply the same delta only after
        effect verification.

        Args:
            state: Input task state.
            update_mask: Successful and verified rows, shape ``(n_envs,)``.

        Returns:
            New task state with masked updates.
        """
        if not isinstance(state, TaskState):
            raise TypeError("state must be a TaskState.")
        mask = _normalize_mask(
            update_mask,
            batch_size=state.batch_size,
            device=state.device,
            name="update_mask",
        )
        held = dict(state.held_objects)
        for resource, candidate in self.held_object_updates.items():
            normalized = (
                None
                if candidate is None
                else _normalize_held(
                    candidate,
                    batch_size=state.batch_size,
                    device=state.device,
                )
            )
            merged = _merge_held(held.get(resource), normalized, mask)
            if merged is None:
                held.pop(resource, None)
            else:
                held[resource] = merged

        coordinated = dict(state.coordinated_held_objects)
        for resources, candidate in self.coordinated_held_object_updates.items():
            normalized = (
                None
                if candidate is None
                else _normalize_coordinated_held(
                    candidate,
                    batch_size=state.batch_size,
                    device=state.device,
                )
            )
            merged = _merge_coordinated(coordinated.get(resources), normalized, mask)
            if merged is None:
                coordinated.pop(resources, None)
            else:
                coordinated[resources] = merged

        return TaskState(
            batch_size=state.batch_size,
            device=state.device,
            held_objects=held,
            coordinated_held_objects=coordinated,
        )


__all__ = ["StateDelta"]
