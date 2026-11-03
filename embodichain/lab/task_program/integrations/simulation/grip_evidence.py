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

"""Native paired-pad grip observations with explicitly declared robot bindings."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import MISSING, dataclass
import math
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.sensors.contact_sensor import (
    ContactSensorCfg,
    ArticulationContactFilterCfg,
)
from embodichain.lab.sim.atomic_actions.evidence import PhysicalEvidenceRequest
from embodichain.lab.sim.atomic_actions.scene import SceneProvider
from embodichain.lab.task_program.semantics.evidence import (
    BinaryEffectEvidenceQuery,
    BinaryEffectObservation,
    EffectEvidenceCollectionContext,
    ControlPartSimulationEvidenceProvider,
    ControlPartRobotEvidenceSource,
)
from embodichain.lab.task_program.semantics.effects import (
    BinaryEvidenceKind,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    CONSTRAINT_EFFECT_CHANNEL,
    ControlPartEvidenceAddress,
    HeldObjectRelation,
    HeldObjectStateExpectation,
)
from embodichain.lab.task_program.semantics.scene import SceneObjectRef, SceneRegistry
from embodichain.utils import configclass
from .contact_evidence import (
    ContactEvidenceSampler,
    CONTACT_ACTOR_IDS,
    CONTACT_VALID_MASK,
    CONTACT_NORMALS,
    CONTACT_IMPULSES,
)

__all__ = [
    "DriverOpeningBinding",
    "NativeGripEvidenceCfg",
    "NativeGripEvidenceFactory",
    "NativeGripEvidenceProvider",
    "NativeGripObserver",
    "classify_contacts",
    "create_joint_opening_reader",
]

_RELEASE_SCOPE = (
    "experimental_open_hand_without_bilateral_grip_not_contact_free_separation"
)
_INVALID_CONTACT = (
    "Target grip contact is incomplete, one-sided, non-opposing, or saturated."
)


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact identifier.")


def _finite_trace_copy(value: Any, path: str, errors: list[dict[str, str]]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        label = (
            "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
        )
        errors.append({"path": path, "value": label})
        return {"nonfinite": label}
    if isinstance(value, torch.Tensor):
        return _finite_trace_copy(value.detach().cpu().tolist(), path, errors)
    if isinstance(value, dict):
        return {
            key: _finite_trace_copy(item, f"{path}.{key}", errors)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, deque)):
        return [
            _finite_trace_copy(item, f"{path}[{index}]", errors)
            for index, item in enumerate(value)
        ]
    return deepcopy(value)


def classify_contacts(
    pairs: torch.Tensor,
    normals: torch.Tensor,
    impulses: torch.Tensor,
    *,
    target: int,
    pads: tuple[int, int],
) -> tuple[bool, bool]:
    """Distinguish opposed bilateral contact, clear release, and unresolved contact.

    Args:
        pairs: Native actor pairs, shaped (contacts, 2).
        normals: Arena-frame actor-0-to-actor-1 normals, shaped (contacts, 3).
        impulses: Native scalar normal impulses, shaped (contacts,).
        target: Exact object actor ID for this environment.
        pads: Exact two pad actor IDs for this environment.

    Returns:
        Measured bilateral contact and whether this positive or negative
        observation is available. Availability is separate from a task verdict.
    """
    if not all(torch.isfinite(v).all() for v in (normals, impulses)):
        return False, False
    sides = []
    touching = False
    for pad in pads:
        selected = (
            ((pairs[:, 0] == target) & (pairs[:, 1] == pad))
            | ((pairs[:, 1] == target) & (pairs[:, 0] == pad))
        ) & (impulses > 0)
        touching |= bool(selected.any())
        outward = torch.where(
            (pairs[selected, 0] == target)[:, None],
            normals[selected],
            -normals[selected],
        )
        lengths = torch.linalg.vector_norm(outward, dim=-1)
        if bool((lengths <= 1e-8).any()):
            return False, False
        sides.append(outward / lengths[:, None])
    if (
        sides[0].numel()
        and sides[1].numel()
        and bool((sides[0] @ sides[1].T <= -0.5).any())
    ):
        return True, True
    return False, not touching


@dataclass(frozen=True, slots=True)
class DriverOpeningBinding:
    """Explicit joint-order calibration for the measured open-hand observation.

    The declaring robot/profile owns driver selection; this layer does not infer
    mimic joints or hand geometry from a model name or a control-part suffix.
    """

    control_part: str
    joint_names: tuple[str, ...]
    driver_indices: tuple[int, ...]
    open_qpos: tuple[float, ...]
    tolerance: float

    def __post_init__(self) -> None:
        _identifier(self.control_part, "control_part")
        if (
            type(self.joint_names) is not tuple
            or not self.joint_names
            or len(set(self.joint_names)) != len(self.joint_names)
        ):
            raise ValueError("joint_names must be a nonempty unique tuple.")
        for name in self.joint_names:
            _identifier(name, "joint_names")
        if (
            type(self.driver_indices) is not tuple
            or not self.driver_indices
            or len(set(self.driver_indices)) != len(self.driver_indices)
            or any(
                type(i) is not int or not 0 <= i < len(self.joint_names)
                for i in self.driver_indices
            )
        ):
            raise ValueError("driver_indices must select unique declared joints.")
        if (
            type(self.open_qpos) is not tuple
            or len(self.open_qpos) != len(self.joint_names)
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                for value in self.open_qpos
            )
        ):
            raise ValueError("open_qpos must match the complete finite joint order.")
        if (
            isinstance(self.tolerance, bool)
            or not math.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            raise ValueError("Opening tolerance must be finite and positive.")


def create_joint_opening_reader(
    robot: object, bindings: tuple[DriverOpeningBinding, ...]
) -> Callable[[str, int], Mapping[str, object]]:
    """Read declared drivers while requiring the complete hand row to be finite.

    Args:
        robot: Selected robot exposing joint names, control indices and qpos.
        bindings: Explicit calibration for each observed control part.

    Returns:
        Per-part/per-environment reader with the original strict open threshold.
    """
    declared = {item.control_part: item for item in bindings}
    if len(declared) != len(bindings):
        raise ValueError("Opening bindings require unique control parts.")
    for part, binding in declared.items():
        names = tuple(
            robot.joint_names[int(index)] for index in robot.get_joint_ids(name=part)
        )
        if names != binding.joint_names:
            raise ValueError(
                "Release opening requires the exact configured joint order."
            )

    def observe(part: str, env_id: int) -> Mapping[str, object]:
        binding = declared[part]
        qpos = robot.get_qpos(name=part)
        if (
            qpos.ndim != 2
            or qpos.shape[1] != len(binding.joint_names)
            or not 0 <= env_id < qpos.shape[0]
        ):
            raise ValueError(
                "Release joint observation has an invalid shape or environment."
            )
        indices = list(binding.driver_indices)
        measured = qpos[env_id, indices]
        target = measured.new_tensor(binding.open_qpos)[indices]
        finite = bool(torch.isfinite(qpos[env_id]).all())
        return {
            "finite": finite,
            "open": finite
            and bool(((measured - target).abs() < binding.tolerance).all()),
            "driving_joint_names": [binding.joint_names[i] for i in indices],
            "measured_qpos": measured.detach().cpu().tolist(),
            "open_qpos": target.detach().cpu().tolist(),
            "tolerance": binding.tolerance,
            "scope": _RELEASE_SCOPE,
        }

    return observe


@configclass
class NativeGripEvidenceCfg:
    """Task-independent physical grip sampling and explicit hand bindings."""

    sensor_uid: str = MISSING
    object_bindings: tuple[tuple[str, str], ...] = MISSING
    pad_bindings: tuple[tuple[str, tuple[str, str]], ...] = MISSING
    algorithm_revision: int = 1
    opening_bindings: tuple[DriverOpeningBinding, ...] = ()
    max_contacts_per_env: int = 1024
    trace_capacity: int = 4096


class NativeGripObserver:
    """Observe target-specific bilateral contact through physical evidence frames.

    Revision 1 requires opposed positive-impulse pad contacts or a measured clear
    release. Revision 2 additionally permits the explicitly configured open-hand
    observation for one-sided release; it never certifies acquisition, bilateral
    release, or geometric separation.
    """

    def __init__(
        self,
        sampler: ContactEvidenceSampler,
        object_actors: Mapping[str, torch.Tensor],
        pad_actors: Mapping[str, torch.Tensor],
        *,
        sensor_uid: str,
        algorithm_revision: int = 1,
        trace_capacity: int = 4096,
        release_opening: Callable[[str, int], Mapping[str, object]] | None = None,
    ) -> None:
        """Own raw actor bindings and bounded observer-lifetime audit state.

        Args:
            sampler: Instance-owned contact acquisition source.
            object_actors: One native actor per object and environment, shape (B, 1).
            pad_actors: Two explicitly declared pad actors per control part, (B, 2).
            sensor_uid: Exact configured sensor identity, retained in audit metadata.
            algorithm_revision: Physical interpretation revision, either 1 or 2.
            trace_capacity: Maximum retained query records.
            release_opening: Required only for explicit revision-2 release observation.
        """
        _identifier(sensor_uid, "sensor_uid")
        if type(algorithm_revision) is not int or algorithm_revision not in (1, 2):
            raise ValueError("Unsupported native grip algorithm revision.")
        if type(trace_capacity) is not int or trace_capacity < 1:
            raise ValueError("trace_capacity must be a positive integer.")
        if algorithm_revision == 2 and not callable(release_opening):
            raise ValueError("Grip revision 2 requires an explicit opening reader.")
        self.sampler, self.sensor_uid = sampler, sensor_uid
        self.algorithm_revision = algorithm_revision
        self._release_opening = release_opening if algorithm_revision == 2 else None
        self._object_actors = self._actors(object_actors, sampler.num_envs, 1)
        self._pad_actors = self._actors(pad_actors, sampler.num_envs, 2)
        self.trace: deque[dict[str, Any]] = deque(maxlen=trace_capacity)
        self.cumulative_query_count = 0
        self.recorded_query_count = 0
        self.evicted_query_count = 0
        self.nonfinite_query_count = 0
        self.dropped_query_count = 0
        self.maximum_dropped_contacts = 0

    @staticmethod
    def _actors(
        bindings: Mapping[str, torch.Tensor], count: int, width: int
    ) -> dict[str, torch.Tensor]:
        if not bindings:
            raise ValueError("Native grip actor bindings must not be empty.")
        result = {}
        for name, value in bindings.items():
            _identifier(name, "actor binding")
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype not in (torch.int32, torch.int64)
                or value.shape != (count, width)
                or bool((value < 0).any())
            ):
                raise ValueError(
                    "Native grip actor IDs have invalid shape, dtype or identity."
                )
            if width == 2 and bool((value[:, 0] == value[:, 1]).any()):
                raise ValueError("Native grip pads must be two distinct actors.")
            result[name] = value.detach().cpu().clone()
        return result

    def snapshot_trace(self) -> dict[str, Any]:
        """Return raw trace data without sampling or claiming task success.

        The frontend owns any application JSON schema wrapper. Field meanings and
        observer-lifetime loss/nonfinite accounting retain the original contract.
        """
        nonfinite: list[dict[str, str]] = []
        records = _finite_trace_copy(self.trace, "trace", nonfinite)
        return {
            "source": "native_contact_sensor",
            "observed": self.recorded_query_count > 0,
            "sensor_uid": self.sensor_uid,
            "scope": "observer_lifetime_raw_queries_not_task_success",
            "actor_bindings": {
                "scope": "observer_lifetime_native_ids_not_contact_evidence",
                "env_axis": "native_env_id",
                "object_actor_ids": {
                    key: values.tolist() for key, values in self._object_actors.items()
                },
                "pad_actor_ids": {
                    key: values.tolist() for key, values in self._pad_actors.items()
                },
            },
            "cumulative_query_count": self.cumulative_query_count,
            "recorded_query_count": self.recorded_query_count,
            "retained_query_count": len(self.trace),
            "evicted_query_count": self.evicted_query_count,
            "unrecorded_query_count": self.cumulative_query_count
            - self.recorded_query_count,
            "trace_capacity": self.trace.maxlen,
            "overflowed": self.evicted_query_count > 0,
            "trace_complete": self.cumulative_query_count
            == self.recorded_query_count
            == len(self.trace)
            and self.evicted_query_count == 0,
            "dropped_contacts": {
                "queries_with_drops": self.dropped_query_count,
                "maximum_per_query": self.maximum_dropped_contacts,
            },
            "finite_audit": {
                "valid": self.nonfinite_query_count == 0 and not nonfinite,
                "nonfinite_query_count": self.nonfinite_query_count,
                "retained_nonfinite_locations": nonfinite,
                "nonfinite_encoding": "explicit_value_marker_at_original_location",
            },
            "trace": records,
        }

    def __call__(
        self, query: BinaryEffectEvidenceQuery, context: EffectEvidenceCollectionContext
    ) -> BinaryEffectObservation:
        """Acquire one clause-local constraint observation for requested rows.

        Args:
            query: Held-object constraint query with its original evidence ID.
            context: Authoritative observation time, revision and environment order.

        Returns:
            Physical boolean and independent validity/error masks, never a verdict.
        """
        if (
            type(query) is not BinaryEffectEvidenceQuery
            or query.clause.evidence_kind is not BinaryEvidenceKind.CONSTRAINT
        ):
            raise TypeError(
                "Native grip observations require a binary constraint query."
            )
        expectation, address = query.expectation, query.source.address
        if (
            type(expectation) is not HeldObjectStateExpectation
            or type(address) is not ControlPartEvidenceAddress
            or address.channel != CONSTRAINT_EFFECT_CHANNEL
        ):
            raise ValueError(
                "Native grip query requires a held-object control-part constraint route."
            )
        uid, part = expectation.object_id, address.control_part
        if uid not in self._object_actors or part not in self._pad_actors:
            raise ValueError(
                "Native grip query is outside the declared actor bindings."
            )
        self.cumulative_query_count += 1
        sample = self.sampler.sample(
            PhysicalEvidenceRequest(
                context.timestamp,
                context.observation_revision,
                context.env_ids.detach().cpu(),
                (
                    CONTACT_VALID_MASK,
                    CONTACT_ACTOR_IDS,
                    CONTACT_NORMALS,
                    CONTACT_IMPULSES,
                ),
            )
        )
        batches = sample.frame.batches
        if (
            sample.acquisition_error is not None
            and batches[CONTACT_VALID_MASK].values.shape[1] == 0
        ):
            return BinaryEffectObservation(
                torch.zeros(
                    len(context.env_ids),
                    dtype=torch.bool,
                    device=context.env_ids.device,
                ),
                torch.zeros(
                    len(context.env_ids),
                    dtype=torch.bool,
                    device=context.env_ids.device,
                ),
                (sample.acquisition_error,) * len(context.env_ids),
            )
        values, valid, errors, records = [], [], [], []
        available_rows = (
            torch.stack(tuple(batch.valid for batch in batches.values()))
            .all(0)
            .tolist()
        )
        finite_query = True
        for row, env_id in enumerate(sample.frame.env_ids.tolist()):
            mask = batches[CONTACT_VALID_MASK].values[row]
            trace_pairs = batches[CONTACT_ACTOR_IDS].values[row, mask]
            trace_normals = batches[CONTACT_NORMALS].values[row, mask]
            trace_impulses = batches[CONTACT_IMPULSES].values[row, mask]
            # Host records already need CPU ownership. Keep public Frame checks
            # there, but preserve the original backend arithmetic at physical
            # thresholds by evaluating only the packed contacts on their source.
            compute_device = sample.source_device or trace_normals.device
            pairs = trace_pairs.to(compute_device)
            normals = trace_normals.to(compute_device)
            impulses = trace_impulses.to(compute_device)
            finite = bool(
                torch.isfinite(normals).all() and torch.isfinite(impulses).all()
            )
            finite_query &= finite
            target = int(self._object_actors[uid][env_id, 0])
            pads = tuple(self._pad_actors[part][env_id].tolist())
            value, ok = classify_contacts(
                pairs, normals, impulses, target=target, pads=pads
            )
            opening = None
            available = available_rows[row]
            if (
                self._release_opening is not None
                and query.clause.expected is False
                and expectation.relation is HeldObjectRelation.DETACHED
                and not value
                and not ok
                and finite
                and sample.dropped_contacts == 0
                and available
                and bool((torch.linalg.vector_norm(normals, dim=-1) > 1e-8).all())
                and bool((impulses >= 0).all())
            ):
                touching = [
                    bool(
                        (
                            (
                                ((pairs[:, 0] == target) & (pairs[:, 1] == pad))
                                | ((pairs[:, 1] == target) & (pairs[:, 0] == pad))
                            )
                            & (impulses > 0)
                        ).any()
                    )
                    for pad in pads
                ]
                if sum(touching) == 1:
                    opening = dict(self._release_opening(part, env_id))
                    finite &= opening["finite"] is True
                    finite_query &= finite
                    ok = opening["open"] is True and finite
            ok &= sample.dropped_contacts == 0 and available
            values.append(value)
            valid.append(ok)
            errors.append(None if ok else _INVALID_CONTACT)
            selected = (trace_pairs == target).any(-1)
            records.append(
                {
                    "env_id": env_id,
                    "value": value,
                    "valid": ok,
                    "finite": finite,
                    "error": errors[-1],
                    "pairs": trace_pairs[selected].tolist(),
                    "normals": trace_normals[selected].tolist(),
                    "impulses": trace_impulses[selected].tolist(),
                    "raw_contacts": {
                        "pairs": trace_pairs.tolist(),
                        "normals": trace_normals.tolist(),
                        "impulses": trace_impulses.tolist(),
                    },
                    **({"release_opening": opening} if opening is not None else {}),
                }
            )
        if len(self.trace) == self.trace.maxlen:
            self.evicted_query_count += 1
        self.recorded_query_count += 1
        self.nonfinite_query_count += int(not finite_query)
        self.dropped_query_count += int(sample.dropped_contacts > 0)
        self.maximum_dropped_contacts = max(
            self.maximum_dropped_contacts, sample.dropped_contacts
        )
        self.trace.append(
            {
                "query_index": self.cumulative_query_count,
                "timestamp": context.timestamp,
                "observation_revision": context.observation_revision,
                "object_id": uid,
                "control_part": part,
                "expected": query.clause.expected,
                "dropped_contacts": sample.dropped_contacts,
                "rows": records,
            }
        )
        return BinaryEffectObservation(
            torch.tensor(values, dtype=torch.bool, device=context.env_ids.device),
            torch.tensor(valid, dtype=torch.bool, device=context.env_ids.device),
            tuple(errors),
        )


class NativeGripEvidenceProvider(ControlPartSimulationEvidenceProvider):
    """Standard control-part provider exposing its owned grip observer for audits."""

    def __init__(
        self, robot: object, observer: NativeGripObserver, *, scene_provider: object
    ) -> None:
        """Construct through the public observer-injection boundary.

        Args:
            robot: Exact assembled robot evidence source.
            observer: Instance-owned native grip observer.
            scene_provider: Shared scene observation source.
        """
        super().__init__(
            robot, scene_provider=scene_provider, constraint_observer=observer
        )
        self._grip_observer = observer

    @property
    def grip_observer(self) -> NativeGripObserver:
        """Return the observer owned by this provider, not by a global sensor slot."""
        return self._grip_observer


@dataclass(frozen=True, slots=True, init=False)
class NativeGripEvidenceFactory:
    """Immutable physical declaration for the existing control-part evidence ID."""

    provider_id: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_REVISION
    sensor_uid: str
    object_bindings: tuple[tuple[str, str], ...]
    pad_bindings: tuple[tuple[str, tuple[str, str]], ...]
    algorithm_revision: int
    opening_bindings: tuple[DriverOpeningBinding, ...]
    max_contacts_per_env: int
    trace_capacity: int

    def __init__(self, cfg: NativeGripEvidenceCfg) -> None:
        """Validate and freeze explicit bindings before native resources are created.

        Args:
            cfg: Complete physical bindings and optional release-revision selection.
        """
        cfg.validate()
        _identifier(cfg.sensor_uid, "sensor_uid")
        if type(cfg.algorithm_revision) is not int or cfg.algorithm_revision not in (
            1,
            2,
        ):
            raise ValueError("Unsupported native grip algorithm revision.")
        for name in ("max_contacts_per_env", "trace_capacity"):
            if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(cfg.object_bindings, (tuple, list)) or any(
            not isinstance(item, (tuple, list)) or len(item) != 2
            for item in cfg.object_bindings
        ):
            raise ValueError(
                "object_bindings must contain explicit identity/UID pairs."
            )
        if not isinstance(cfg.pad_bindings, (tuple, list)) or any(
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or not isinstance(item[1], (tuple, list))
            for item in cfg.pad_bindings
        ):
            raise ValueError(
                "pad_bindings must contain explicit control-part/link-list pairs."
            )
        objects = tuple((key, uid) for key, uid in cfg.object_bindings)
        pads = tuple((part, tuple(names)) for part, names in cfg.pad_bindings)
        if not objects or len({key for key, _ in objects}) != len(objects):
            raise ValueError("Object bindings must have unique semantic identities.")
        if not pads or len({key for key, _ in pads}) != len(pads):
            raise ValueError("Pad bindings must have unique control parts.")
        for key, uid in objects:
            _identifier(key, "object identity")
            _identifier(uid, "simulation UID")
        for part, names in pads:
            _identifier(part, "control_part")
            if type(names) is not tuple or len(names) != 2 or len(set(names)) != 2:
                raise ValueError(
                    "Each grip binding requires two explicit pad link names."
                )
            for name in names:
                _identifier(name, "pad link")
        opening = tuple(cfg.opening_bindings)
        if not all(type(item) is DriverOpeningBinding for item in opening) or len(
            {item.control_part for item in opening}
        ) != len(opening):
            raise ValueError(
                "Opening bindings must be unique DriverOpeningBinding values."
            )
        if cfg.algorithm_revision == 2 and {item.control_part for item in opening} != {
            part for part, _ in pads
        }:
            raise ValueError(
                "Grip revision 2 requires explicit opening bindings for every hand."
            )
        for name, value in (
            ("sensor_uid", cfg.sensor_uid),
            ("object_bindings", objects),
            ("pad_bindings", pads),
            ("algorithm_revision", cfg.algorithm_revision),
            ("opening_bindings", opening),
            ("max_contacts_per_env", cfg.max_contacts_per_env),
            ("trace_capacity", cfg.trace_capacity),
        ):
            object.__setattr__(self, name, value)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: object,
        scene_provider: object,
    ) -> NativeGripEvidenceProvider:
        """Build a fresh observer with explicit native actors and unchanged provider IDs.

        Args:
            simulation: Explicit simulation owner of the sensor.
            robot: Exact selected robot.
            scene_registry: Semantic identities for declared object bindings.
            engine: Engine assembled for the same robot.
            scene_provider: Shared synchronized scene provider.

        Returns:
            Standard control-part evidence provider with an instance-owned observer.
        """
        if not isinstance(robot, ControlPartRobotEvidenceSource):
            raise TypeError("Native grip evidence requires a robot evidence source.")
        if scene_provider is not None and not isinstance(scene_provider, SceneProvider):
            raise TypeError("scene_provider must implement SceneProvider or be None.")
        if engine.robot is not robot:
            raise ValueError("Native grip evidence requires the engine's exact robot.")
        reader = (
            create_joint_opening_reader(robot, self.opening_bindings)
            if self.algorithm_revision == 2
            else None
        )
        native = {}
        for entity_id, uid in self.object_bindings:
            canonical = scene_registry.resolve(
                entity_id, expected_type=SceneObjectRef
            ).entity_id
            if canonical in native:
                raise ValueError(
                    "Object bindings must resolve to distinct scene objects."
                )
            native[canonical] = uid
        links = [name for _, names in self.pad_bindings for name in names]
        sensor = simulation.get_sensor(self.sensor_uid)
        if sensor is None:
            sensor = simulation.add_sensor(
                ContactSensorCfg(
                    uid=self.sensor_uid,
                    rigid_uid_list=list(native.values()),
                    articulation_cfg_list=[
                        ArticulationContactFilterCfg(
                            articulation_uid=robot.uid, link_name_list=links
                        )
                    ],
                    max_contacts_per_env=self.max_contacts_per_env,
                )
            )
        else:
            cfg = sensor.cfg
            declared_links = [
                (item.articulation_uid, tuple(item.link_name_list))
                for item in cfg.articulation_cfg_list
            ]
            if (
                tuple(cfg.rigid_uid_list) != tuple(native.values())
                or declared_links != [(robot.uid, tuple(links))]
                or cfg.max_contacts_per_env != self.max_contacts_per_env
                or cfg.filter_need_both_actor is not True
            ):
                raise ValueError(
                    "Existing grip sensor does not match the declared contact scope."
                )
        observer = NativeGripObserver(
            ContactEvidenceSampler(
                sensor,
                topology_revision_reader=lambda: int(
                    simulation.spawn_result.topology_revision
                ),
            ),
            {key: sensor.get_actor_ids(uid) for key, uid in native.items()},
            {
                part: sensor.get_actor_ids(robot.uid, names)
                for part, names in self.pad_bindings
            },
            sensor_uid=self.sensor_uid,
            algorithm_revision=self.algorithm_revision,
            trace_capacity=self.trace_capacity,
            release_opening=reader,
        )
        return NativeGripEvidenceProvider(
            robot, observer, scene_provider=scene_provider
        )
