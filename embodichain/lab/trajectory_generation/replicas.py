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

"""Validated mapping from logical candidates to real fixed-scene replicas."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.motion.expansion import MotionSnapshot, SceneCase

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions.state import PlanningContext
    from .initial_state import FixedSceneHost, PreparedBatch

__all__ = ["CandidateSlotAssignment", "SceneReplicaPool"]


@dataclass(frozen=True)
class CandidateSlotAssignment:
    """One logical candidate bound to a real simulator row for one planning wave.

    Source indices identify canonical cases, not physical execution slots.
    Coordinates are local arena coordinates: verified replicas have the same
    root and entity poses, so this binding has an identity coordinate mapping.
    A binding does not authorize execution or certify collision/task success.

    Args:
        candidate_index: Position in the caller's ordered candidate jobs.
        source_index: Canonical logical source index.
        source_row: Physical row capturing that canonical source's initial state.
        slot_index: Physical tensor row used in this planning wave.
        env_id: Actual environment ID, equal to the physical row in this backend.
        candidate_id: Optional stable identity supplied by the generation owner.
        host_epoch: Host preparation epoch, or ``None`` for planning-only inputs.
    """

    candidate_index: int
    source_index: int
    source_row: int
    slot_index: int
    env_id: int
    candidate_id: str | None = None
    host_epoch: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "candidate_index",
            "source_index",
            "source_row",
            "slot_index",
            "env_id",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.env_id != self.slot_index:
            raise ValueError("env_rows requires env_id to equal the physical slot")
        if self.candidate_id is not None and (
            not isinstance(self.candidate_id, str) or not self.candidate_id.strip()
        ):
            raise ValueError("candidate_id must be a nonempty string or None")
        if self.host_epoch is not None and (
            type(self.host_epoch) is not int or self.host_epoch < 0
        ):
            raise ValueError("host_epoch must be a nonnegative integer or None")


class SceneReplicaPool:
    """Schedule candidate jobs over verified complete physical environment rows.

    Rows sharing ``(scene_case_id, initial_state_id)`` must carry identical
    SceneCase metadata, complete joint state, local root/entity poses, and
    trusted fixed-condition signatures. The first row is their canonical
    source. Geometry, material, calibration and controller equivalence cannot
    be inferred from poses: the caller's signature must cover these conditions.

    .. attention::
        The direct constructor is planning-only. It verifies the supplied
        snapshots, not live simulation state. Use :meth:`from_host` for an
        epoch-bound pool whose host checks live initial-state drift. Neither
        constructor creates, resets or duplicates physical environments.

    Args:
        snapshots: One complete initial snapshot per real physical row, already
            expressed in local arena coordinates, excluding global arena offsets.
        fixed_condition_signatures: Trusted geometry, physical, tool and control
            signatures in physical-row order. Equal-case rows must match.
        env_ids: Complete physical row IDs; defaults to ``range(E)``. Reordering,
            repeated IDs and invented logical candidate IDs are not supported.
        entity_id_map: Optional physical UID to atomic SceneSnapshot ID mapping.
            Unspecified physical UIDs use the identity mapping.
        atol: Finite nonnegative absolute equality tolerance; relative tolerance
            is always zero.
    """

    def __init__(
        self,
        snapshots: Sequence[MotionSnapshot],
        *,
        fixed_condition_signatures: Sequence[str],
        env_ids: Sequence[int] | None = None,
        entity_id_map: Mapping[str, str] | None = None,
        atol: float = 1e-6,
    ) -> None:
        values = tuple(snapshots)
        if not values or any(not isinstance(value, MotionSnapshot) for value in values):
            raise ValueError("snapshots must cover a nonempty physical batch")
        if (
            isinstance(atol, bool)
            or not isinstance(atol, (int, float))
            or not math.isfinite(atol)
            or atol < 0
        ):
            raise ValueError("atol must be finite and nonnegative")
        self._atol = float(atol)
        self._snapshots = tuple(replace(value) for value in values)
        if isinstance(fixed_condition_signatures, str):
            raise ValueError("provide one trusted fixed-condition signature per row")
        signatures = tuple(fixed_condition_signatures)
        if len(signatures) != len(values) or any(
            not isinstance(value, str) or not value.strip() for value in signatures
        ):
            raise ValueError("provide one trusted fixed-condition signature per row")
        ids = tuple(range(len(values))) if env_ids is None else tuple(env_ids)
        if any(type(value) is not int for value in ids) or ids != tuple(
            range(len(values))
        ):
            raise ValueError("env_rows requires the complete ordered physical env IDs")
        mapping = dict(entity_id_map or {})
        all_entities = {key for value in values for key in value.entity_poses}
        if not set(mapping).issubset(all_entities) or any(
            not isinstance(value, str) or not value.strip()
            for value in mapping.values()
        ):
            raise ValueError(
                "entity_id_map must map captured physical UIDs to entity IDs"
            )
        targets = [mapping.get(key, key) for key in all_entities]
        if len(set(targets)) != len(targets):
            raise ValueError("entity_id_map must not merge distinct physical entities")
        self._entity_id_map = mapping
        self._env_ids = ids
        self._signatures = signatures
        self._host: FixedSceneHost | None = None
        self._binding: PreparedBatch | None = None
        self._context_baseline: dict[str, object] | None = None

        cases: list[SceneCase] = []
        source_rows: list[int] = []
        source_indices: list[int] = []
        groups: list[list[int]] = []
        source_keys: dict[tuple[str, str], int] = {}
        for row, snapshot in enumerate(self._snapshots):
            first = self._snapshots[0]
            if snapshot.joint_names != first.joint_names or (
                snapshot.scene_case.robot_profile_id,
                snapshot.scene_case.calibration_id,
                snapshot.scene_case.task_id,
            ) != (
                first.scene_case.robot_profile_id,
                first.scene_case.calibration_id,
                first.scene_case.task_id,
            ):
                raise ValueError(
                    "one replica pool requires a shared robot/task/calibration"
                )
            if (
                snapshot.joint_positions.dtype != first.joint_positions.dtype
                or snapshot.joint_positions.device != first.joint_positions.device
            ):
                raise ValueError("replica rows must share joint dtype and device")
            key = (
                snapshot.scene_case.scene_case_id,
                snapshot.scene_case.initial_state_id,
            )
            if key not in source_keys:
                source_keys[key] = len(cases)
                cases.append(snapshot.scene_case)
                source_rows.append(row)
                groups.append([])
            source = source_keys[key]
            canonical = self._snapshots[source_rows[source]]
            self._assert_equivalent(canonical, snapshot, source_rows[source], row)
            source_indices.append(source)
            groups[source].append(row)
        self._cases = tuple(cases)
        self._source_rows = tuple(source_rows)
        self._source_indices = tuple(source_indices)
        self._groups = tuple(tuple(rows) for rows in groups)

    @classmethod
    def from_host(
        cls,
        host: FixedSceneHost,
        batch: PreparedBatch,
        *,
        entity_id_map: Mapping[str, str] | None = None,
        atol: float = 1e-6,
    ) -> SceneReplicaPool:
        """Bind a pool to a prepared host and its trusted initial-state profile.

        The profile's verifier must certify fixed geometry and physical/control
        conditions for each declared SceneCase. A global signature alone does
        not establish equivalence between rows with different physical assets.

        Args:
            host: Exclusive owner of the complete physical batch.
            batch: Current successfully prepared host token.
            entity_id_map: Physical UID to atomic scene entity mapping.
            atol: Absolute snapshot and context equality tolerance.

        Returns:
            A pool that checks ownership, epoch and physical drift before reuse.
        """
        host.assert_current(batch)
        if not host.verify_initial(batch).accepted:
            raise RuntimeError("replica pool requires verified host initial state")
        signature = host.profile.signature()
        pool = cls(
            host.snapshots(batch),
            fixed_condition_signatures=(signature,) * len(batch.cases),
            entity_id_map=entity_id_map,
            atol=atol,
        )
        pool._host = host
        pool._binding = batch
        return pool

    @property
    def batch_size(self) -> int:
        """Number of real physical rows, independent of candidate count."""
        return len(self._snapshots)

    @property
    def source_count(self) -> int:
        """Number of unique logical scene/initial-state sources."""
        return len(self._cases)

    @property
    def cases(self) -> tuple[SceneCase, ...]:
        """Canonical cases in deterministic first-physical-occurrence order."""
        return self._cases

    @property
    def source_rows(self) -> tuple[int, ...]:
        """Canonical source index to initial physical row mapping."""
        return self._source_rows

    @property
    def source_indices(self) -> tuple[int, ...]:
        """Physical row to canonical logical source index mapping."""
        return self._source_indices

    @property
    def env_ids(self) -> tuple[int, ...]:
        """Complete, unique real environment IDs in tensor-row order."""
        return self._env_ids

    @property
    def snapshots(self) -> tuple[MotionSnapshot, ...]:
        """Owned copies of physical-row initial snapshots, safe for caller mutation."""
        return tuple(replace(snapshot) for snapshot in self._snapshots)

    @property
    def host_epoch(self) -> int | None:
        """Captured preparation epoch, or ``None`` for a planning-only pool."""
        return None if self._binding is None else self._binding.epoch

    def source_snapshot(self, source_index: int) -> MotionSnapshot:
        """Copy the initial snapshot belonging to a canonical logical source.

        Args:
            source_index: Source index in :attr:`cases`.

        Returns:
            An independent copy preserving canonical source identity.
        """
        self._validate_source(source_index)
        return replace(self._snapshots[self._source_rows[source_index]])

    def rounds(
        self,
        candidate_sources: Sequence[int],
        *,
        candidate_ids: Sequence[str] | None = None,
    ) -> Iterator[tuple[CandidateSlotAssignment, ...]]:
        """Yield deterministic bounded physical-row batches for ordered jobs.

        Every candidate appears once; each wave uses a physical slot at most
        once. Surplus jobs wait for the next wave without changing their source
        or candidate identity. Unused physical slots are omitted; the consumer
        must mask them in its full-size planning request.

        Args:
            candidate_sources: Canonical source index for every ordered job.
            candidate_ids: Optional unique stable IDs in the same order.

        Yields:
            Assignments ordered by physical slot, with no duplicate env IDs.
        """
        sources = tuple(candidate_sources)
        for source in sources:
            self._validate_source(source)
        ids = (None,) * len(sources) if candidate_ids is None else tuple(candidate_ids)
        if len(ids) != len(sources) or (
            candidate_ids is not None
            and (
                any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != len(ids)
            )
        ):
            raise ValueError("candidate_ids must be unique nonempty IDs matching jobs")
        pending: list[deque[int]] = [deque() for _ in self._cases]
        for index, source in enumerate(sources):
            pending[source].append(index)
        self.assert_current()
        while any(pending):
            self.assert_current()
            assignments = []
            for source, slots in enumerate(self._groups):
                for slot in slots:
                    if not pending[source]:
                        break
                    candidate = pending[source].popleft()
                    assignments.append(
                        CandidateSlotAssignment(
                            candidate_index=candidate,
                            source_index=source,
                            source_row=self._source_rows[source],
                            slot_index=slot,
                            env_id=self._env_ids[slot],
                            candidate_id=ids[candidate],
                            host_epoch=self.host_epoch,
                        )
                    )
            yield tuple(
                sorted(assignments, key=lambda assignment: assignment.slot_index)
            )

    def assert_current(self, context: PlanningContext | None = None) -> None:
        """Reject stale bindings or initial-state/context drift before planning.

        Args:
            context: Optional complete, unprojected initial atomic context. Its
                first validation freezes the task/scene/control inputs for this
                pool; later calls must use the same initial context values.

        Raises:
            RuntimeError: If a host token, physical initial state or previously
                validated atomic context changed.
            ValueError: If an initial context does not match the captured rows.
        """
        if self._host is not None:
            assert self._binding is not None
            self._host.assert_current(self._binding)
            if not self._host.verify_initial(self._binding).accepted:
                raise RuntimeError(
                    "replica host initial state drifted; restore and rebind"
                )
        if context is None:
            return
        if (
            context.batch_size != self.batch_size
            or tuple(context.env_ids.tolist()) != self.env_ids
        ):
            raise ValueError("context must use the complete physical replica batch")
        context.require_control_dt()
        if context.robot.root_pose is None and self._host is None:
            raise ValueError(
                "planning-only replica context requires explicit root_pose"
            )
        for row, snapshot in enumerate(self._snapshots):
            self._require_close(
                context.robot.qpos[row], snapshot.joint_positions, "context qpos"
            )
            self._require_close(
                context.robot.qvel[row], snapshot.joint_velocities, "context qvel"
            )
            if context.robot.root_pose is not None:
                self._require_close(
                    context.robot.root_pose[row],
                    snapshot.root_pose,
                    "context root_pose",
                )
            for uid, pose in snapshot.entity_poses.items():
                entity_id = self._entity_id_map.get(uid, uid)
                state = context.scene.entities.get(entity_id)
                if state is None:
                    raise ValueError(
                        f"context is missing captured entity {uid!r} ({entity_id!r})"
                    )
                value = state.pose[row] if state.pose.ndim == 3 else state.pose
                self._require_close(value, pose, f"context entity {entity_id!r}")
        values = self._context_values(context)
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                continue
            if not torch.isfinite(value).all():
                raise ValueError(f"replica {name} must contain finite initial values")
            if value.ndim == 0:
                continue
            if value.shape[0] != self.batch_size:
                continue
            for slots in self._groups:
                for row in slots[1:]:
                    self._require_close(value[row], value[slots[0]], f"replica {name}")
        if self._context_baseline is None:
            self._context_baseline = {
                name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in values.items()
            }
            return
        if values.keys() != self._context_baseline.keys():
            raise RuntimeError("replica initial context structure drifted")
        for name, value in values.items():
            previous = self._context_baseline[name]
            try:
                if isinstance(value, torch.Tensor) and isinstance(
                    previous, torch.Tensor
                ):
                    self._require_close(value, previous, name)
                elif value != previous:
                    raise ValueError(name)
            except ValueError as exc:
                raise RuntimeError(f"replica initial context drifted: {name}") from exc

    def _validate_source(self, source: int) -> None:
        if type(source) is not int or not 0 <= source < self.source_count:
            raise ValueError("candidate source must identify a canonical pool source")

    def _require_close(
        self, first: torch.Tensor, second: torch.Tensor, name: str
    ) -> None:
        if (
            first.shape != second.shape
            or first.dtype != second.dtype
            or first.device != second.device
            or not torch.allclose(first, second, atol=self._atol, rtol=0)
        ):
            raise ValueError(f"{name} differs from the verified replica initial state")

    def _assert_equivalent(
        self, first: MotionSnapshot, second: MotionSnapshot, first_row: int, row: int
    ) -> None:
        if (
            first.scene_case != second.scene_case
            or self._signatures[first_row] != self._signatures[row]
            or first.dependency_revisions != second.dependency_revisions
            or first.entity_poses.keys() != second.entity_poses.keys()
        ):
            raise ValueError(
                "same-case replicas have different metadata or fixed conditions"
            )
        for name in ("joint_positions", "joint_velocities", "root_pose"):
            self._require_close(
                getattr(first, name), getattr(second, name), f"replica {name}"
            )
        for uid, pose in first.entity_poses.items():
            self._require_close(
                pose, second.entity_poses[uid], f"replica entity {uid!r}"
            )

    def _context_values(self, context: PlanningContext) -> dict[str, object]:
        """Copy comparable inputs without depending on mutable semantic objects."""
        values: dict[str, object] = {
            "control_dt": context.control_dt,
            "scene_version": context.scene.version,
            "collision_entities": context.scene.collision_entity_ids,
            "collision_revisions": torch.tensor(
                context.scene.collision_world_revisions(self.batch_size)
            ),
        }
        for name in ("qpos", "qvel", "qeffort", "root_pose", "root_twist"):
            value = getattr(context.robot, name)
            if value is not None:
                values[f"robot.{name}"] = value
        for entity_id, state in context.scene.entities.items():
            values[f"scene.{entity_id}"] = (
                state.pose
                if state.pose.ndim == 3
                else state.pose.unsqueeze(0).expand(self.batch_size, -1, -1)
            )
            values[f"confidence.{entity_id}"] = state.confidence
        for name in ("held_objects", "coordinated_held_objects", "articulation_joints"):
            for key, state in getattr(context.task, name).items():
                for field_name in state.__dataclass_fields__:
                    value = getattr(state, field_name)
                    values[f"task.{name}.{key!r}.{field_name}"] = (
                        value
                        if isinstance(value, torch.Tensor) or value is None
                        else id(value)
                    )
        for key, state in context.scene.articulation_joints.items():
            position = state.position
            values[f"articulation.{key!r}.position"] = (
                position
                if position.ndim == 2
                else position.unsqueeze(0).expand(self.batch_size, -1)
            )
            values[f"articulation.{key!r}.valid_mask"] = state.valid_mask
        return values
