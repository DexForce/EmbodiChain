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

"""Native contact support observations with explicit acquisition ownership."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np
import torch

from embodichain.compute.geometry.contact_support import support_from_contacts
from embodichain.lab.sim.evidence import (
    PhysicalEvidenceBatch,
    PhysicalEvidenceRequest,
)
from embodichain.lab.sim.sensors.contact.sensor import ContactSensor, ContactSensorCfg
from .contact.sampling import (
    CONTACT_ACTOR_IDS,
    CONTACT_IMPULSES,
    CONTACT_NORMALS,
    CONTACT_POSITIONS,
    CONTACT_VALID_MASK,
    ContactEvidenceSample,
    ContactEvidenceSampler,
    ContactSourceStamp,
    native_contact_revision,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "NATIVE_SUPPORT_CHANNEL",
    "SupportControlClock",
    "NativeSupportResult",
    "NativeSupportObservation",
    "SupportObservationRegistry",
]

NATIVE_SUPPORT_CHANNEL = "support.com_patch"
_RAW_CHANNELS = (
    CONTACT_ACTOR_IDS,
    CONTACT_POSITIONS,
    CONTACT_NORMALS,
    CONTACT_IMPULSES,
)
_ERRORS = (
    AttributeError,
    AssertionError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


class SupportControlClock(Protocol):
    """Host control cadence used only by explicitly bound strong observations."""

    @property
    def step_index(self) -> int:
        """Number of completed control steps."""
        ...

    @property
    def step_dt(self) -> float:
        """Control period in seconds."""
        ...

    def now(self) -> float:
        """Return elapsed control time without advancing it."""
        ...


@dataclass(frozen=True, slots=True, eq=False)
class NativeSupportResult:
    """A physical support fact and independently owned source metadata.

    ``support.valid`` separates unavailable acquisition from an instantaneous
    measured negative. An empty contact sample does not establish absence of
    support over a stability window. Task acceptance remains a consumer concern.

    Args:
        support: Boolean physical channel with explicit validity and errors.
        metadata: Native support measurements, preserving the selected contract.
    """

    support: PhysicalEvidenceBatch
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        self._validate_support(self.support)
        object.__setattr__(self, "support", self.support.snapshot())
        object.__setattr__(
            self, "metadata", MappingProxyType(deepcopy(dict(self.metadata)))
        )

    @staticmethod
    def _validate_support(support: PhysicalEvidenceBatch) -> None:
        if type(support) is not PhysicalEvidenceBatch or (
            support.evidence_id != NATIVE_SUPPORT_CHANNEL
            or support.values.dtype != torch.bool
            or support.values.shape != support.valid.shape
        ):
            raise TypeError("support must be a boolean native-support batch.")

    @classmethod
    def _from_owned(
        cls, support: PhysicalEvidenceBatch, metadata: dict[str, Any]
    ) -> NativeSupportResult:
        """Consume one newly validated batch and unexposed per-call metadata.

        The observation producer has copied its persistent actor-pair records;
        all other lists/dictionaries were created by this observation. Public
        construction still snapshots arbitrary caller-owned mutable inputs.
        """
        cls._validate_support(support)
        result = object.__new__(cls)
        object.__setattr__(result, "support", support)
        object.__setattr__(result, "metadata", MappingProxyType(metadata))
        return result


class ContactWitness:
    """Retain a measured positive patch only while both body poses persist."""

    def __init__(self) -> None:
        self.value: tuple[torch.Tensor, torch.Tensor, tuple[np.ndarray, ...]] | None = (
            None
        )
        self.sample_info: dict[str, Any] | None = None
        self.last_native_revision: int | None = None

    def update(
        self,
        obj: torch.Tensor,
        ref: torch.Tensor,
        sample: tuple[np.ndarray, ...] | None,
        *,
        invalid: bool = False,
    ) -> tuple[np.ndarray, ...] | None:
        """Own a new patch, or invalidate retention after either body moves."""
        if (
            invalid
            or self.value is not None
            and (
                not torch.allclose(obj, self.value[0], atol=1e-4, rtol=0)
                or not torch.allclose(ref, self.value[1], atol=1e-4, rtol=0)
            )
        ):
            self.value = None
            self.sample_info = None
        if not invalid and sample is not None:
            self.value = (
                obj.clone(),
                ref.clone(),
                tuple(value.copy() for value in sample),
            )
            self.sample_info = None
        return None if self.value is None else self.value[2]


@dataclass(frozen=True, slots=True)
class _SampleToken:
    stamp: ContactSourceStamp
    timestamp: float


def _identifier(value: str, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact identifier.")
    return value


def _timestamp(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value) or value < 0)
    ):
        raise ValueError(
            "Native support acquisition time must be finite and nonnegative."
        )
    return float(value)


def _native_instances(body: RigidObject) -> tuple[object, ...]:
    values = tuple(body._entities)
    if not values:
        raise ValueError("Native support requires materialized rigid bodies.")
    return values


def _same_instances(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


class NativeSupportObservation:
    """Observe one explicitly bound native pair through a borrowed contact sensor.

    The acquisition clock is mandatory for every physical frame. ``bind_clock``
    separately enables strong control-window qualification; constructing this
    observer never upgrades legacy metadata to support contract revision 1.
    One last frame and one packed witness per environment bound retained storage.
    The COM tolerance is the sum of each body's maximum native rest offset plus
    0.001 metres, and must lie in ``(0, 0.01]``. A retained witness is invalidated
    when either body pose differs beyond absolute tolerance ``1e-4``.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        obj: RigidObject,
        reference: RigidObject,
        *,
        entity_id: str,
        reference_id: str,
        sensor: ContactSensor,
        observer_name: str,
        acquisition_clock: Callable[[], float],
    ) -> None:
        """Attach one owned sampling hook without owning native resource lifetime.

        Args:
            simulation: Exact owner of the selected rigid bodies and sensor.
            obj: Native payload object.
            reference: Native support object.
            entity_id: Canonical payload identity used in observation metadata.
            reference_id: Canonical support identity used in metadata.
            sensor: Borrowed sensor declaring exactly the two rigid UIDs.
            observer_name: Unique hook name in this sensor.
            acquisition_clock: Read-only canonical host acquisition time in seconds.
        """
        self.entity_id = _identifier(entity_id, "entity_id")
        self.reference_id = _identifier(reference_id, "reference_id")
        self._observer_name = _identifier(observer_name, "observer_name")
        if not callable(acquisition_clock):
            raise TypeError("acquisition_clock must be callable.")
        if obj is reference or obj.uid == reference.uid:
            raise ValueError("Native support requires two distinct rigid bodies.")
        if simulation.get_rigid_object(obj.uid) is not obj or (
            simulation.get_rigid_object(reference.uid) is not reference
        ):
            raise ValueError("Support bodies must belong to the exact simulation.")
        if simulation.get_sensor(sensor.cfg.uid) is not sensor:
            raise ValueError("Support sensor must belong to the exact simulation.")
        _validate_sensor_cfg(sensor.cfg, obj, reference)
        count = obj.num_instances
        if (
            type(count) is not int
            or count < 1
            or (reference.num_instances != count or sensor.num_instances != count)
        ):
            raise ValueError(
                "Support bodies and sensor must have equal nonempty batches."
            )
        self.object, self.reference, self.sensor = obj, reference, sensor
        self._sensor_cfg = deepcopy(sensor.cfg.to_dict())
        self._simulation, self._acquisition_clock = simulation, acquisition_clock
        self._body_instances = (_native_instances(obj), _native_instances(reference))
        rest = [
            max(float(entity.get_physical_attr().rest_offset) for entity in entities)
            for entities in self._body_instances
        ]
        self._tolerance = sum(rest) + 0.001
        if not math.isfinite(self._tolerance) or not 0 < self._tolerance <= 0.01:
            raise ValueError("Native support rest envelopes are unqualified.")
        self._witnesses = [ContactWitness() for _ in range(count)]
        self._episode_generations = [0] * count
        self._blocked_rows: set[int] = set()
        self._clock: SupportControlClock | None = None
        self._base: tuple[int, float] | None = None
        self._offset, self._substep, self._serial = 0.0, 0, 0
        self._capture_revision = 0
        self._actor_pairs: list[dict[str, Any]] = []
        self._topology_revision: int | None = None
        self._sample_revision: int | None = None
        self._legacy_sample_step: int | None = None
        self._last_sample: ContactEvidenceSample | None = None
        self._error: str | None = None
        self._closed = False
        self._sampler = self._new_sampler()
        self._refresh_identity()
        _timestamp(acquisition_clock())
        sensor.add_sampling_observer(self._observer_name, self)

    @property
    def tolerance(self) -> float:
        """Return the qualified native rest-envelope COM tolerance in metres."""
        return self._tolerance

    def _new_sampler(self) -> ContactEvidenceSampler:
        return ContactEvidenceSampler(
            self.sensor,
            native_revision_reader=(
                None
                if self._clock is None
                else lambda: native_contact_revision(self._simulation)
            ),
            topology_revision_reader=lambda: int(
                self._simulation.spawn_result.topology_revision
            ),
        )

    def _physics_step(self) -> int:
        step = self._simulation.physics_step_index
        if type(step) is not int or step < 0:
            raise ValueError("Native support requires a valid physics-step identity.")
        return step

    def _revision_or_none(self) -> int | None:
        if self._clock is None:
            return None
        try:
            return native_contact_revision(self._simulation)
        except _ERRORS:
            return None

    def bind_clock(self, clock: SupportControlClock) -> None:
        """Enable strong qualification using an already-owned control clock.

        Args:
            clock: Canonical execution clock; this method never advances it.
        """
        if self._closed:
            raise ValueError("Native support observation is closed.")
        _timestamp(clock.now())
        if (
            type(clock.step_index) is not int
            or clock.step_index < 0
            or (
                isinstance(clock.step_dt, bool)
                or not math.isfinite(clock.step_dt)
                or clock.step_dt <= 0
            )
        ):
            raise ValueError("Native support control clock is unqualified.")
        if clock is not self._clock:
            self._clock, self._base = clock, None
            self._sampler = self._new_sampler()
            self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Invalidate selected witnesses and advance each episode generation once.

        Args:
            env_ids: Sensor-compatible indices, including negative aliases.
        """
        if self._closed:
            return
        self._legacy_sample_step = None
        selected = torch.ones(len(self._witnesses), dtype=torch.bool)
        if env_ids is not None:
            selected.zero_().index_fill_(
                0, torch.as_tensor(env_ids, dtype=torch.long).cpu(), True
            )
        revision = self._revision_or_none()
        for row in selected.nonzero(as_tuple=False).flatten().tolist():
            witness = self._witnesses[row]
            witness.value, witness.sample_info = None, None
            witness.last_native_revision = revision
            self._episode_generations[row] += 1
            self._blocked_rows.add(row)

    def _unavailable(self, error: Exception) -> None:
        self._error = str(error) or type(error).__name__
        for witness in self._witnesses:
            witness.value, witness.sample_info = None, None

    def _refresh_identity(self) -> None:
        if self._closed:
            raise ValueError("Native support observation is closed.")
        if self._simulation.get_sensor(self.sensor.cfg.uid) is not self.sensor or (
            self.sensor.cfg.to_dict() != self._sensor_cfg
        ):
            raise ValueError(
                "Native support sensor changed; explicitly rebind the pair."
            )
        for body, original in zip((self.object, self.reference), self._body_instances):
            if self._simulation.get_rigid_object(
                body.uid
            ) is not body or not _same_instances(_native_instances(body), original):
                raise ValueError(
                    "Native support bodies changed; explicitly rebind the pair."
                )
        capabilities = self.sensor.contact_capabilities
        if not capabilities.geometry or not capabilities.impulse:
            raise ValueError("Native support geometry/impulse capability unavailable.")
        topology = int(self._simulation.spawn_result.topology_revision)
        if topology < 0:
            raise ValueError("Native support topology revision is invalid.")
        actors = [
            self.sensor.get_actor_ids(body.uid)
            for body in (self.object, self.reference)
        ]
        if any(ids.shape != (len(self._witnesses), 1) for ids in actors):
            raise ValueError("Native support actor mapping is ambiguous.")
        pairs = []
        for row in range(len(self._witnesses)):
            ids = [int(value[row, 0]) for value in actors]
            infos = [asdict(self.sensor.get_actor_info(actor)) for actor in ids]
            if (
                min(ids) < 0
                or ids[0] == ids[1]
                or any(
                    info["env_id"] not in {-1, row}
                    or info["link_name"] is not None
                    or info["actor_id"] != actor
                    for info, actor in zip(infos, ids)
                )
            ):
                raise ValueError("Native support actor identity is unqualified.")
            pairs.append(
                dict(
                    object_actor=ids[0],
                    reference_actor=ids[1],
                    object_uid=self.object.uid,
                    reference_uid=self.reference.uid,
                    object_info=infos[0],
                    reference_info=infos[1],
                )
            )
        changed = bool(self._actor_pairs) and (
            pairs != self._actor_pairs or topology != self._topology_revision
        )
        self._actor_pairs, self._topology_revision = pairs, topology
        self._object_actors, self._reference_actors = [
            ids.detach().cpu() for ids in actors
        ]
        if changed:
            self.reset()
            self._sampler = self._new_sampler()

    def begin_control_step(self) -> None:
        """Begin the next complete strong interval after the sensor's own reset."""
        if self._closed or self._clock is None:
            return
        try:
            index, now, dt = (
                self._clock.step_index,
                _timestamp(self._clock.now()),
                self._clock.step_dt,
            )
            if type(index) is not int or index < 0 or not math.isfinite(dt) or dt <= 0:
                raise ValueError("Native support clock is unqualified.")
            if self._base is not None and (
                index != self._base[0] + 1 or now <= self._base[1]
            ):
                raise ValueError("Native support control clock did not advance once.")
            self._base, self._offset, self._substep = (index, now), 0.0, 0
            self._refresh_identity()
            self._blocked_rows.clear()
            self._error = None
        except _ERRORS as error:
            self._unavailable(error)

    def before_sample(self) -> _SampleToken | None:
        """Capture the source stamp and actual host acquisition time before fetch."""
        if self._closed:
            return None
        try:
            return _SampleToken(
                self._sampler.source_stamp(), _timestamp(self._acquisition_clock())
            )
        except _ERRORS as error:
            self._unavailable(error)
            return None

    def after_sample(self, dt: float, token: object) -> None:
        """Consume already-captured raw state without a second sensor fetch.

        Args:
            dt: Physics duration represented by the completed sample.
            token: This observer's corresponding before-fetch token.
        """
        if self._closed:
            return
        self._capture_revision += 1
        try:
            if type(token) is not _SampleToken:
                raise ValueError("Native support before-fetch stamp is unavailable.")
            if self._clock is not None:
                if self._error is not None or self._base is None:
                    raise ValueError(
                        self._error or "Native support interval is missing."
                    )
                if (
                    not math.isfinite(dt)
                    or dt <= 0
                    or self._clock.step_index != self._base[0]
                ):
                    raise ValueError("Native support substep timing is unqualified.")
                self._offset += dt
                self._substep += 1
                self._serial += 1
                if self._offset > self._clock.step_dt + 1e-9:
                    raise ValueError(
                        "Native support interval exceeds the control cadence."
                    )
            self._refresh_identity()
            legacy_step = self._physics_step() if self._clock is None else None
            sample = self._sampler.sample(
                PhysicalEvidenceRequest(
                    token.timestamp,
                    self._capture_revision,
                    torch.arange(len(self._witnesses)),
                ),
                refresh=False,
                source_stamp=token.stamp,
            )
            self._capture(
                sample,
                self.object.get_local_pose(to_matrix=True),
                self.reference.get_local_pose(to_matrix=True),
            )
            if legacy_step is not None:
                if self._physics_step() != legacy_step:
                    raise ValueError(
                        "Native support physics step changed during capture."
                    )
                self._legacy_sample_step = legacy_step

        except _ERRORS as error:
            self._unavailable(error)

    def _capture(
        self, sample: ContactEvidenceSample, obj: torch.Tensor, ref: torch.Tensor
    ) -> None:
        batches = sample.frame.batches
        if (
            sample.acquisition_error
            or sample.dropped_contacts
            or any(not bool(batch.valid.all()) for batch in batches.values())
        ):
            reasons = [
                error
                for batch in batches.values()
                for error in batch.acquisition_errors
                if error
            ]
            raise ValueError(
                sample.acquisition_error
                or "; ".join(dict.fromkeys(reasons))
                or "Native support contact sample is unavailable."
            )
        count = len(self._witnesses)
        if any(
            not isinstance(pose, torch.Tensor)
            or pose.shape != (count, 4, 4)
            or not torch.isfinite(pose).all()
            for pose in (obj, ref)
        ):
            raise ValueError(
                "Native support poses must be finite full-batch transforms."
            )
        active = batches[CONTACT_VALID_MASK].values
        if bool((batches[CONTACT_ACTOR_IDS].values[active] < 0).any()):
            raise ValueError("Native support counterpart is unqualified.")
        if self._clock is not None and sample.native_revision is None:
            raise ValueError("Native support source revision is unavailable.")
        self._last_sample, self._sample_revision = sample, sample.native_revision
        if self._clock is None:
            self._error = None
        for row, witness in enumerate(self._witnesses):
            if self._clock is not None:
                witness.update(obj[row], ref[row], None)
                if (
                    row in self._blocked_rows
                    or witness.last_native_revision == sample.native_revision
                ):
                    continue
                witness.last_native_revision = sample.native_revision
            mask = active[row]
            pairs = batches[CONTACT_ACTOR_IDS].values[row, mask]
            impulses = batches[CONTACT_IMPULSES].values[row, mask]
            selected = (
                (
                    (pairs[:, 0] == self._reference_actors[row, 0])
                    & (pairs[:, 1] == self._object_actors[row, 0])
                )
                | (
                    (pairs[:, 1] == self._reference_actors[row, 0])
                    & (pairs[:, 0] == self._object_actors[row, 0])
                )
            ) & (impulses > 0)
            patch = (
                tuple(
                    batches[channel].values[row, mask][selected].numpy()
                    for channel in _RAW_CHANNELS
                )
                if bool(selected.any())
                else None
            )
            witness.update(obj[row], ref[row], patch)
            if self._clock is not None and patch is not None:
                assert self._base is not None
                witness.sample_info = dict(
                    timestamp=self._base[1] + self._offset,
                    control_step=self._base[0] + 1,
                    substep=self._substep,
                    serial=self._serial,
                    episode_generation=self._episode_generations[row],
                    native_revision=sample.native_revision,
                    topology_revision=self._topology_revision,
                )

    def _validate_clocked_observation(self) -> None:
        if self._clock is None or self._error is not None:
            raise ValueError(
                self._error or "Native support execution clock unavailable."
            )
        self._refresh_identity()
        if (
            native_contact_revision(self._simulation) != self._sample_revision
            or self.sensor.dropped_contacts
            or (
                not math.isfinite(self._clock.now())
                or self._base is None
                or self._clock.step_index != self._base[0] + 1
                or not math.isclose(
                    self._offset, self._clock.step_dt, rel_tol=0, abs_tol=1e-9
                )
            )
        ):
            raise ValueError(
                "Native support has an unobserved step or invalid interval."
            )

    def observe(
        self,
        object_pose: torch.Tensor,
        reference_pose: torch.Tensor,
        *,
        support_contract_revision: int | None = None,
    ) -> NativeSupportResult:
        """Measure support without stepping physics or deciding task acceptance.

        Args:
            object_pose: Observed payload poses, shape ``(environments, 4, 4)``.
            reference_pose: Matching support poses in the same Arena frame.
            support_contract_revision: ``None`` preserves legacy metadata; ``1``
                requires an explicitly bound clock and complete observed interval.

        Returns:
            Physical values, availability/errors and independently owned metadata.
        """
        if support_contract_revision is not None and (
            type(support_contract_revision) is not int or support_contract_revision != 1
        ):
            raise ValueError("Unsupported native support contract revision.")
        timestamp = _timestamp(self._acquisition_clock())
        count = len(self._witnesses)
        if not isinstance(object_pose, torch.Tensor) or object_pose.shape != (
            count,
            4,
            4,
        ):
            raise ValueError("object_pose must contain one transform per environment.")
        clocked = support_contract_revision is not None
        sample_is_fresh = False
        metadata: dict[str, Any] = {
            "scope": "native_contact_support",
            "entity": self.entity_id,
            "reference": self.reference_id,
        }
        try:
            if (
                not isinstance(reference_pose, torch.Tensor)
                or reference_pose.shape != object_pose.shape
                or (
                    not torch.isfinite(object_pose).all()
                    or not torch.isfinite(reference_pose).all()
                )
            ):
                raise ValueError(
                    "Native support poses must be finite full-batch transforms."
                )
            if clocked or self._clock is not None:
                self._validate_clocked_observation()
            else:
                self._refresh_identity()
                self._capture_revision += 1
                sample = self._sampler.sample(
                    PhysicalEvidenceRequest(
                        timestamp, self._capture_revision, torch.arange(count)
                    )
                )
                step = self._physics_step()
                if (
                    self._last_sample is None
                    or self._legacy_sample_step != step
                    or sample.acquisition_error
                    or sample.dropped_contacts
                    or any(
                        not bool(batch.valid.all())
                        for batch in sample.frame.batches.values()
                    )
                ):
                    self._capture(sample, object_pose, reference_pose)
                    self._legacy_sample_step = step
                    sample_is_fresh = True
            assert self._last_sample is not None
            if self.sensor.latest_sample_dropped_contacts:
                raise ValueError("Native support contact query saturated.")
            com = self.object.body_data.com_pose[:, :3].to(object_pose)
            center = (object_pose[:, :3, :3] @ com[:, :, None]).squeeze(
                -1
            ) + object_pose[:, :3, 3]
            accepted, valid, errors, rows = [], [], [], []
            for row, witness in enumerate(self._witnesses):
                patch = witness.update(object_pose[row], reference_pose[row], None)
                info = witness.sample_info
                if clocked:
                    assert self._clock is not None
                    if info is not None and (
                        info["timestamp"] > self._clock.now() + 1e-9
                        or info["episode_generation"] != self._episode_generations[row]
                    ):
                        raise ValueError(
                            "Native support sample clock/generation is unqualified."
                        )
                    if info is None:
                        patch = (
                            np.empty((0, 2)),
                            np.empty((0, 3)),
                            np.empty((0, 3)),
                            np.empty(0),
                        )
                if patch is None:
                    mask = self._last_sample.frame.batches[CONTACT_VALID_MASK].values[
                        row
                    ]
                    if not sample_is_fresh:
                        # A cached contact cannot re-establish a witness invalidated by motion.
                        mask = torch.zeros_like(mask)
                    patch = tuple(
                        self._last_sample.frame.batches[channel]
                        .values[row, mask]
                        .numpy()
                        for channel in _RAW_CHANNELS
                    )
                result, measurements = support_from_contacts(
                    *patch,
                    center[row].detach().cpu().numpy(),
                    object_actor=int(self._object_actors[row, 0]),
                    reference_actor=int(self._reference_actors[row, 0]),
                    tolerance=self._tolerance,
                )
                accepted.append(result)
                blocked = self._clock is not None and row in self._blocked_rows
                valid.append(not blocked)
                errors.append(
                    "Native support interval is incomplete after reset."
                    if blocked
                    else None
                )
                if clocked:
                    measurements.update(
                        sample=None if info is None else dict(info),
                        retained=info is not None and info["serial"] != self._serial,
                        raw_contacts={
                            key: value.tolist()
                            for key, value in zip(
                                ("user_ids", "position", "normal", "impulse"), patch
                            )
                        },
                    )
                rows.append(measurements)
            metadata.update(
                object_pose=object_pose.detach().cpu().tolist(),
                reference_pose=reference_pose.detach().cpu().tolist(),
                supported=accepted,
                env_ids=list(range(count)),
                rows=rows,
            )
            if clocked:
                assert self._clock is not None
                metadata.update(
                    support_contract_revision=1,
                    clock="environment_step",
                    observed_at=self._clock.now(),
                    observed_step=self._clock.step_index,
                    step_dt=self._clock.step_dt,
                    episode_generations=list(self._episode_generations),
                    actor_pairs=deepcopy(self._actor_pairs),
                )
        except _ERRORS as error:
            self._unavailable(error)
            metadata = {
                "scope": "native_contact_support",
                "entity": self.entity_id,
                "reference": self.reference_id,
            }
            accepted, valid, errors = (
                [False] * count,
                [False] * count,
                [self._error] * count,
            )
        return NativeSupportResult._from_owned(
            PhysicalEvidenceBatch(
                NATIVE_SUPPORT_CHANNEL,
                torch.tensor(accepted, dtype=torch.bool, device=object_pose.device),
                torch.tensor(valid, dtype=torch.bool, device=object_pose.device),
                tuple(errors),
                timestamp,
                torch.arange(count, device=object_pose.device),
                self._capture_revision,
            ),
            metadata,
        )

    def close(self) -> None:
        """Detach only this hook and clear owned frames/witnesses, idempotently.

        Delayed callbacks from a sensor token snapshot become no-ops. Borrowed
        sensor and native-object lifetimes remain with the simulation owner.
        """
        if not self._closed:
            self._closed = True
            try:
                self.sensor.remove_sampling_observer(self._observer_name, expected=self)
            except (KeyError, ValueError):
                # A host may already have detached/replaced this registration.
                # The expected-owner guard leaves the new observer untouched.
                pass
            self._last_sample = None
            self._base = None
            self._unavailable(ValueError("Native support observation is closed."))


def _validate_sensor_cfg(
    cfg: ContactSensorCfg, obj: RigidObject, reference: RigidObject
) -> None:
    if type(cfg) is not ContactSensorCfg:
        raise TypeError("Support sensor declaration must be exactly ContactSensorCfg.")
    cfg.validate()
    _identifier(cfg.uid, "sensor uid")
    if (
        cfg.sensor_type != "ContactSensor"
        or set(cfg.rigid_uid_list) != {obj.uid, reference.uid}
        or len(cfg.rigid_uid_list) != 2
        or (
            cfg.articulation_cfg_list
            or cfg.filter_need_both_actor is not True
            or type(cfg.max_contacts_per_env) is not int
            or cfg.max_contacts_per_env < 1
        )
    ):
        raise ValueError(
            "Support requires an explicit two-rigid-body ContactSensor declaration."
        )


class SupportObservationRegistry:
    """Runtime-owned pair registry for explicit reuse across bridges.

    Instances own only their observer hooks. Native bodies and sensors remain
    simulation-owned, including sensors created from a supplied declaration.
    """

    def __init__(
        self, simulation: SimulationManager, *, acquisition_clock: Callable[[], float]
    ) -> None:
        """Bind the explicit simulation and canonical acquisition clock.

        Args:
            simulation: Owner of the selected native bodies and sensors.
            acquisition_clock: Read-only host acquisition time in seconds.
        """
        if not callable(acquisition_clock):
            raise TypeError("acquisition_clock must be callable.")
        self._simulation, self._acquisition_clock = simulation, acquisition_clock
        self._observers: dict[tuple[str, str], NativeSupportObservation] = {}

    def acquire(
        self,
        obj: RigidObject,
        reference: RigidObject,
        *,
        entity_id: str,
        reference_id: str,
        observer_name: str,
        sensor: ContactSensor | None = None,
        sensor_cfg: ContactSensorCfg | None = None,
    ) -> NativeSupportObservation:
        """Reuse an exact pair, or construct it from one explicit sensor selection.

        Args:
            obj: Exact native payload instance.
            reference: Exact native support instance.
            entity_id: Canonical payload identity.
            reference_id: Canonical support identity.
            observer_name: Unique sensor hook name.
            sensor: Borrowed sensor, mutually exclusive with ``sensor_cfg``.
            sensor_cfg: Strict declaration used to find or create a standard sensor.

        Returns:
            The registry-owned pair observer. Same-UID replacements require rebind.
        """
        if (sensor is None) == (sensor_cfg is None):
            raise ValueError("Select exactly one support sensor or sensor declaration.")
        key = (obj.uid, reference.uid)
        previous = self._observers.get(key)
        if previous is not None:
            if sensor_cfg is not None:
                _validate_sensor_cfg(sensor_cfg, obj, reference)
                if sensor_cfg.to_dict() != previous._sensor_cfg:
                    raise ValueError(
                        "Support sensor declaration changed; explicitly rebind it."
                    )
            expected_sensor_uid = (
                sensor.cfg.uid if sensor is not None else sensor_cfg.uid
            )
            if (
                previous.object is not obj
                or previous.reference is not reference
                or previous.entity_id != entity_id
                or previous.reference_id != reference_id
                or (
                    previous._observer_name != observer_name
                    or previous.sensor.cfg.uid != expected_sensor_uid
                    or sensor is not None
                    and previous.sensor is not sensor
                )
            ):
                raise ValueError(
                    "Support pair ownership changed; explicitly release and rebind it."
                )
            previous._refresh_identity()
            return previous
        if sensor_cfg is not None:
            cfg = deepcopy(sensor_cfg)
            _validate_sensor_cfg(cfg, obj, reference)
            sensor = self._simulation.get_sensor(cfg.uid)
            if sensor is None:
                sensor = self._simulation.add_sensor(cfg)
            elif sensor.cfg.to_dict() != cfg.to_dict():
                raise ValueError(
                    "Existing support sensor declaration differs from the selected one."
                )
        assert sensor is not None
        observer = NativeSupportObservation(
            self._simulation,
            obj,
            reference,
            entity_id=entity_id,
            reference_id=reference_id,
            sensor=sensor,
            observer_name=observer_name,
            acquisition_clock=self._acquisition_clock,
        )
        self._observers[key] = observer
        return observer

    def release(self, object_uid: str, reference_uid: str) -> None:
        """Detach one pair before explicitly rebinding replaced native instances.

        Args:
            object_uid: Native payload UID used when acquiring the pair.
            reference_uid: Native support UID used when acquiring the pair.

        Raises:
            KeyError: This registry does not own the selected pair.
        """
        self._observers.pop((object_uid, reference_uid)).close()

    def close(self) -> None:
        """Detach all registry-owned hooks without destroying borrowed objects."""
        for observer in tuple(self._observers.values()):
            observer.close()
        self._observers.clear()
