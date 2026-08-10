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

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.common import BatchEntity

from .state import (
    CoordinatedHeldObjectState,
    HeldObjectState,
    TaskState,
    _normalize_coordinated_held,
    _normalize_held,
    _normalize_mask,
)

if TYPE_CHECKING:
    from .core import ObjectSemantics


def _effect_snapshot_memo(value: object) -> dict[int, object]:
    """Preserve live entities and private runtime caches during effect copies."""
    memo: dict[int, object] = {}
    visited: set[int] = set()

    def visit(nested: object) -> None:
        nested_id = id(nested)
        if nested_id in visited:
            return
        visited.add(nested_id)
        if isinstance(nested, BatchEntity):
            memo[nested_id] = nested
            return
        if is_dataclass(nested) and not isinstance(nested, type):
            for data_field in fields(nested):
                child = getattr(nested, data_field.name)
                if data_field.name == "_generator" and child is not None:
                    memo[id(child)] = None
                elif not data_field.init and child is not None:
                    memo[id(child)] = child
                else:
                    visit(child)
            return
        if isinstance(nested, Mapping):
            for key, child in nested.items():
                visit(key)
                visit(child)
            return
        if isinstance(nested, (list, tuple, set, frozenset)):
            for child in nested:
                visit(child)

    visit(value)
    return memo


def _snapshot_semantics(value: ObjectSemantics) -> ObjectSemantics:
    """Copy semantic data while retaining live simulation-entity identity."""
    try:
        copied = deepcopy(value, _effect_snapshot_memo(value))
    except Exception as exc:
        raise TypeError(
            "ObjectSemantics effect metadata must be copyable without cloning "
            "live simulation entities."
        ) from exc
    if type(copied) is not type(value) or copied is value:
        raise TypeError(
            "ObjectSemantics effect snapshots must produce a distinct value "
            "of the same exact type."
        )
    return copied


def _snapshot_held(value: HeldObjectState) -> HeldObjectState:
    """Return an independently owned held-object effect value."""
    return HeldObjectState(
        semantics=_snapshot_semantics(value.semantics),
        object_to_eef=value.object_to_eef.clone(),
        grasp_xpos=value.grasp_xpos.clone(),
        env_mask=None if value.env_mask is None else value.env_mask.clone(),
    )


def _snapshot_coordinated(
    value: CoordinatedHeldObjectState,
) -> CoordinatedHeldObjectState:
    """Return an independently owned coordinated held-object effect value."""
    return CoordinatedHeldObjectState(
        semantics=_snapshot_semantics(value.semantics),
        left_object_to_eef=value.left_object_to_eef.clone(),
        right_object_to_eef=value.right_object_to_eef.clone(),
        left_grasp_xpos=value.left_grasp_xpos.clone(),
        right_grasp_xpos=value.right_grasp_xpos.clone(),
        env_mask=None if value.env_mask is None else value.env_mask.clone(),
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


def _merge_held(
    previous: HeldObjectState | None,
    candidate: HeldObjectState | None,
    update_mask: torch.Tensor,
) -> HeldObjectState | None:
    """Apply one optional held-object update per environment."""
    from .core import _same_object_identity

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
        and not _same_object_identity(previous.semantics, candidate.semantics)
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
        semantics=(previous.semantics if previous_retained else candidate.semantics),
        object_to_eef=torch.where(
            selector, candidate.object_to_eef, previous.object_to_eef
        ),
        grasp_xpos=torch.where(selector, candidate.grasp_xpos, previous.grasp_xpos),
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

    def __post_init__(self) -> None:
        held = dict(self.held_object_updates)
        for resource, value in held.items():
            if not isinstance(resource, str) or not resource:
                raise ValueError(
                    "held_object_updates keys must be non-empty resource names."
                )
            if value is not None and not isinstance(value, HeldObjectState):
                raise TypeError(
                    "held_object_updates values must be HeldObjectState or None."
                )
        object.__setattr__(self, "held_object_updates", MappingProxyType(held))

    @property
    def is_empty(self) -> bool:
        """Whether this delta declares no symbolic state changes."""
        return not self.held_object_updates

    def snapshot(self) -> StateDelta:
        """Return an independently owned symbolic-effect snapshot.

        Live simulation entities retain identity, while semantic metadata,
        affordance data, and every attachment tensor are copied.

        Returns:
            Independently owned state delta.
        """
        return StateDelta(
            held_object_updates={
                resource: None if value is None else _snapshot_held(value)
                for resource, value in self.held_object_updates.items()
            },
            coordinated_held_object_updates={
                resources: (None if value is None else _snapshot_coordinated(value))
                for resources, value in self.coordinated_held_object_updates.items()
            },
        )

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
            update_mask: Successful and verified rows, shape ``(num_envs,)``.

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

        return TaskState(
            batch_size=state.batch_size,
            device=state.device,
            held_objects=held,
        )


__all__ = ["StateDelta"]
