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

"""Immutable scene proposals, independent of task and execution lifecycles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

__all__ = ["ScenePoseChange", "SceneVariant"]


from .._validation import _identifier, _nonnegative_integer, _rigid_pose


def _variant_id(
    parent_scene_id: str,
    seed: int,
    ordinal: int,
    pose_changes: tuple,
    metadata: tuple,
) -> str:
    """Hash already validated, canonically ordered proposal values."""
    payload = {
        "schema": "scene-variant/v1",
        "parent_scene_id": parent_scene_id,
        "seed": seed,
        "ordinal": ordinal,
        "pose_changes": pose_changes,
        "metadata": metadata,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "scene-variant:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ScenePoseChange:
    """A proposed pose for one existing physical entity.

    Args:
        entity_uid: Stable physical entity UID, independent of environment slots.
        arena_pose: Homogeneous 4x4 transform from the entity frame into its
            local Z-up arena frame. Inputs are copied into immutable tuples.

    .. attention::
        An arena pose excludes replication offsets. The host converts it to
        backend coordinates when applying the candidate. This contract does
        not establish support, collision freedom, or manipulation feasibility.
    """

    entity_uid: str
    arena_pose: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        _identifier(self.entity_uid, "entity_uid")
        object.__setattr__(self, "arena_pose", _rigid_pose(self.arena_pose))


@dataclass(frozen=True)
class SceneVariant:
    """An immutable scene proposal with reproducible proposal identity.

    Args:
        parent_scene_id: Content/revision identity of the original scene.
        seed: Nonnegative proposal-generation seed.
        ordinal: Nonnegative position in the candidate source.
        pose_changes: Unique existing entities and their proposed arena poses.
        metadata: Unique string key/value provenance entries. Entries and pose
            changes are sorted so equivalent input ordering has one identity.

    .. attention::
        ``variant_id`` identifies a proposal, not a settled initial state or a
        successful execution. Hosts record those identities and evidence
        separately. Metadata describes provenance, not executable operations.
        Since ``ordinal`` participates in the hash, consuming the same layout
        at a different source position changes ``variant_id``. File-backed
        consumers should retain the original layout identity and proposal
        ordinal in metadata for provenance across selection orders; this ID
        is not a geometry-only deduplication key.
    """

    parent_scene_id: str
    seed: int
    ordinal: int
    pose_changes: tuple[ScenePoseChange, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.parent_scene_id, "parent_scene_id")
        _nonnegative_integer(self.seed, "seed")
        _nonnegative_integer(self.ordinal, "ordinal")
        try:
            changes = tuple(self.pose_changes)
        except TypeError as exc:
            raise ValueError(
                "pose_changes must contain ScenePoseChange values"
            ) from exc
        if any(not isinstance(change, ScenePoseChange) for change in changes):
            raise ValueError("pose_changes must contain ScenePoseChange values")
        if len({change.entity_uid for change in changes}) != len(changes):
            raise ValueError("pose_changes must contain unique entity UIDs")
        object.__setattr__(
            self,
            "pose_changes",
            tuple(sorted(changes, key=lambda change: change.entity_uid)),
        )
        try:
            entries = tuple(self.metadata)
            if any(isinstance(entry, str) for entry in entries):
                raise ValueError("metadata must contain string key/value pairs")
            metadata = tuple(tuple(entry) for entry in entries)
        except TypeError as exc:
            raise ValueError("metadata must contain string key/value pairs") from exc
        if any(len(entry) != 2 for entry in metadata):
            raise ValueError("metadata must contain string key/value pairs")
        for key, value in metadata:
            _identifier(key, "metadata key")
            if not isinstance(value, str):
                raise ValueError("metadata values must be strings")
        if len({key for key, _ in metadata}) != len(metadata):
            raise ValueError("metadata must contain unique keys")
        object.__setattr__(self, "metadata", tuple(sorted(metadata)))

    @property
    def variant_id(self) -> str:
        """Return a stable identity including geometry, source, seed, and ordinal.

        Returns:
            SHA-256 identity independent of Python hash randomization and
            temporary physical environment assignments.
        """
        return _variant_id(
            self.parent_scene_id,
            self.seed,
            self.ordinal,
            tuple(
                (change.entity_uid, change.arena_pose) for change in self.pose_changes
            ),
            self.metadata,
        )
