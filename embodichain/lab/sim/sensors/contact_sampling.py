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

"""Synchronized contact acquisition using shared physical-evidence channels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch

from embodichain.lab.sim.sensors.contact_sensor import ContactSensor
from embodichain.lab.sim.evidence import (
    PhysicalEvidenceBatch,
    PhysicalEvidenceFrame,
    PhysicalEvidenceRequest,
)


class _ObservationContext(Protocol):
    timestamp: float
    env_ids: torch.Tensor


__all__ = [
    "CONTACT_ACTOR_IDS",
    "CONTACT_VALID_MASK",
    "CONTACT_NORMALS",
    "CONTACT_IMPULSES",
    "CONTACT_POSITIONS",
    "ContactEvidenceSample",
    "ContactEvidenceSampler",
    "ContactSourceStamp",
    "native_contact_revision",
]

CONTACT_ACTOR_IDS = "contact.user_ids"
CONTACT_VALID_MASK = "contact.is_valid"
CONTACT_NORMALS = "contact.normal"
CONTACT_IMPULSES = "contact.impulse"
CONTACT_POSITIONS = "contact.position"
_CHANNELS = {
    CONTACT_ACTOR_IDS: ("user_ids", (2,)),
    CONTACT_VALID_MASK: ("is_valid", ()),
    CONTACT_NORMALS: ("normal", (3,)),
    CONTACT_IMPULSES: ("impulse", ()),
    CONTACT_POSITIONS: ("position", (3,)),
}
_ERRORS = (AttributeError, KeyError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True, slots=True)
class ContactSourceStamp:
    """Source identity captured before or after a contact query fetch.

    Native revision describes the backend's contact sample, independently of
    a runtime observation revision. Missing capabilities remain explicit errors.
    """

    native_revision: int | None
    topology_revision: int | None
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.native_revision, self.topology_revision):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(
                    "Contact revisions must be nonnegative integers or None."
                )
        if type(self.errors) is not tuple or any(
            type(error) is not str or not error for error in self.errors
        ):
            raise ValueError(
                "Contact source errors must be an owned tuple of nonempty strings."
            )


@dataclass(frozen=True, slots=True)
class ContactEvidenceSample:
    """One physical frame with its native acquisition diagnostics.

    The common frame remains invocation-independent. Clause IDs and execution
    ownership are supplied by the effect-evidence consumer, not this sampler.
    """

    frame: PhysicalEvidenceFrame
    native_revision: int | None
    topology_revision: int | None
    dropped_contacts: int
    acquisition_error: str | None = None
    source_device: torch.device | None = None


def native_contact_revision(simulation: object) -> int:
    """Read the pinned Default backend contact revision without sampling.

    Args:
        simulation: Explicit simulation owner exposing its Spawn result and world.

    Returns:
        Nonnegative native contact revision.

    Raises:
        ValueError: If the backend or SDK does not expose this revision contract.
    """
    if simulation.spawn_result.backend != "dexsim":
        raise ValueError("Native contact source revision requires Default physics.")
    physics = simulation.get_world().get_physics_scene()
    getter = getattr(physics, "_get_contact_data_revision", None)
    if not callable(getter):
        raise ValueError("Native contact source revision is unavailable.")
    revision = int(getter())
    if revision < 0:
        raise ValueError("Native contact source revision is invalid.")
    return revision


class ContactEvidenceSampler:
    """Sample a contact sensor into owned, environment-aligned physical channels.

    Only explicit requested channels are copied. Invalid fixed-buffer slots are
    zeroed and remain masked out; they are never treated as measured contacts.
    Positions and normals retain the sensor's Arena frame; impulse values stay
    impulses rather than forces. No contact-frame field assumes a single
    invocation or a task family.
    """

    def __init__(
        self,
        sensor: ContactSensor,
        *,
        native_revision_reader: Callable[[], int] | None = None,
        topology_revision_reader: Callable[[], int] | None = None,
    ) -> None:
        """Bind one sensor and optional explicit source revision readers.

        Args:
            sensor: ContactSensor-compatible source with a fixed environment axis.
            native_revision_reader: Read the backend's current contact revision.
            topology_revision_reader: Read the actor topology revision. A changed
                topology requires a new sampler and refreshed actor bindings.
        """
        if type(sensor.num_instances) is not int or sensor.num_instances < 1:
            raise ValueError(
                "Contact source must declare a positive environment count."
            )
        self.sensor = sensor
        self.num_envs = sensor.num_instances
        self._native_reader = native_revision_reader
        self._topology_reader = topology_revision_reader
        self._initial_topology: int | None = None
        self._initial_errors: tuple[str, ...] = ()
        if topology_revision_reader is not None:
            try:
                self._initial_topology = self._revision(topology_revision_reader())
            except _ERRORS as error:
                self._initial_errors = (str(error) or type(error).__name__,)

    @staticmethod
    def _revision(value: int) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("Contact source revision must be a nonnegative integer.")
        return value

    def source_stamp(self) -> ContactSourceStamp:
        """Capture native/topology identities without refreshing sensor data.

        Returns:
            Source stamp, with capability/read failures preserved as errors.
        """
        errors = list(self._initial_errors)
        native = topology = None
        for reader, name in (
            (self._native_reader, "native"),
            (self._topology_reader, "topology"),
        ):
            if reader is None:
                continue
            try:
                value = self._revision(reader())
                if name == "native":
                    native = value
                else:
                    topology = value
                    if value != self._initial_topology:
                        errors.append(
                            "Contact actor topology changed; rebind the sampler."
                        )
            except _ERRORS as error:
                errors.append(str(error) or type(error).__name__)
        return ContactSourceStamp(native, topology, tuple(dict.fromkeys(errors)))

    def _empty_sample(
        self,
        request: PhysicalEvidenceRequest,
        channels: tuple[str, ...],
        stamp: ContactSourceStamp,
        error: str,
    ) -> ContactEvidenceSample:
        batches = {}
        for channel in channels:
            dtype = (
                torch.bool
                if channel == CONTACT_VALID_MASK
                else torch.int64 if channel == CONTACT_ACTOR_IDS else torch.float32
            )
            values = torch.empty(
                (len(request.env_ids), 0, *_CHANNELS[channel][1]),
                dtype=dtype,
                device=request.env_ids.device,
            )
            batches[channel] = PhysicalEvidenceBatch(
                channel,
                values,
                torch.zeros(
                    len(request.env_ids),
                    dtype=torch.bool,
                    device=request.env_ids.device,
                ),
                (error,) * len(request.env_ids),
                request.timestamp,
                request.env_ids,
                request.observation_revision,
            )
        return ContactEvidenceSample(
            PhysicalEvidenceFrame(
                request.timestamp,
                request.observation_revision,
                request.env_ids,
                batches,
            ),
            stamp.native_revision,
            stamp.topology_revision,
            0,
            error,
        )

    def sample(
        self,
        request: PhysicalEvidenceRequest,
        *,
        refresh: bool = True,
        source_stamp: ContactSourceStamp | None = None,
    ) -> ContactEvidenceSample:
        """Acquire one synchronized sample, retaining loss and nonfinite masks.

        Args:
            request: Timestamp, observation revision, selected native environment
                IDs and requested channel IDs; no channels selects all five.
                The valid-contact mask always accompanies requested raw channels.
            refresh: Fetch through sensor.update, or consume an already completed
                physics-substep sample without fetching a second time.
            source_stamp: Before-fetch stamp for an externally sampled substep.

        Returns:
            Owned raw channels and acquisition diagnostics. Validity is per channel
            and environment; incomplete samples never become measured negatives.
        """
        if type(request) is not PhysicalEvidenceRequest:
            raise TypeError("request must be exactly PhysicalEvidenceRequest.")
        if type(refresh) is not bool:
            raise TypeError("refresh must be a bool.")
        channels = request.channels or tuple(_CHANNELS)
        if CONTACT_VALID_MASK not in channels:
            channels = (CONTACT_VALID_MASK, *channels)
        if any(channel not in _CHANNELS for channel in channels):
            raise ValueError("Unsupported physical contact channel.")
        ids = request.env_ids
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("Contact request contains an out-of-range environment ID.")
        before = self.source_stamp() if refresh else source_stamp
        if before is None:
            before = ContactSourceStamp(
                None, None, ("No before-fetch contact source stamp.",)
            )
        if type(before) is not ContactSourceStamp:
            raise TypeError("source_stamp must be ContactSourceStamp or None.")
        errors = list(before.errors)
        try:
            if refresh:
                self.sensor.update()
            after_fetch = self.source_stamp()
            if after_fetch != before:
                errors.append("Contact source changed during fetch.")
            errors.extend(after_fetch.errors)
            data = self.sensor.get_data()
            capabilities = self.sensor.contact_capabilities
            after_data = self.source_stamp()
            if after_data != before:
                errors.append("Contact source changed during data access.")
            errors.extend(after_data.errors)
            mask = data["is_valid"]
            if (
                mask.dtype is not torch.bool
                or mask.ndim != 2
                or mask.shape[0] != self.num_envs
            ):
                raise ValueError(
                    "Contact validity buffer must have shape (environments, contacts)."
                )
            selected = ids.to(mask.device)
            active = mask.index_select(0, selected).to(ids.device)
            dropped = int(self.sensor.dropped_contacts)
            # ContactSensor keeps interval counts stable during standalone
            # refreshes. Sources without a separate latest-sample diagnostic
            # continue to supply their existing loss count directly.
            latest_dropped = int(
                getattr(self.sensor, "latest_sample_dropped_contacts", dropped)
            )
            if dropped < 0 or latest_dropped < 0:
                raise ValueError("Contact dropped count must be nonnegative.")
            # These counts overlap; summing would double-count substep loss.
            dropped = max(dropped, latest_dropped)
            if dropped:
                errors.append("Contact query or scatter capacity was exceeded.")
            shared_error = "; ".join(dict.fromkeys(errors)) or None
            batches = {}
            for channel in channels:
                key, trailing = _CHANNELS[channel]
                raw = data[key]
                if raw.device != mask.device:
                    raise ValueError(
                        "Contact channels must share the source buffer device."
                    )
                if raw.shape != (*mask.shape, *trailing):
                    raise ValueError(
                        f"Contact channel {channel!r} has an invalid shape."
                    )
                if channel == CONTACT_ACTOR_IDS and raw.dtype not in (
                    torch.int32,
                    torch.int64,
                ):
                    raise ValueError("Contact actor IDs must be integer tensors.")
                if (
                    channel not in (CONTACT_ACTOR_IDS, CONTACT_VALID_MASK)
                    and not raw.is_floating_point()
                ):
                    raise ValueError(
                        "Contact geometric channels must be floating tensors."
                    )
                if channel == CONTACT_VALID_MASK:
                    values = active
                else:
                    values = raw.index_select(0, selected).detach().to(ids.device)
                    enabled = active.reshape((*active.shape, *((1,) * len(trailing))))
                    values = torch.where(enabled, values, torch.zeros_like(values))
                finite = (
                    torch.isfinite(values).reshape(len(ids), -1).all(-1)
                    if values.is_floating_point()
                    else torch.ones(len(ids), dtype=torch.bool, device=ids.device)
                )
                channel_error = shared_error
                if capabilities.geometry is not True:
                    channel_error = (
                        channel_error
                        or "Contact geometry evidence is unsupported by this backend."
                    )
                elif channel == CONTACT_IMPULSES and capabilities.impulse is not True:
                    channel_error = (
                        channel_error
                        or "Contact impulse evidence is unsupported by this backend."
                    )
                valid = finite & (channel_error is None)
                row_errors = tuple(
                    channel_error
                    or (
                        None
                        if bool(ok)
                        else f"Contact channel {channel!r} contains nonfinite active values."
                    )
                    for ok in finite.tolist()
                )
                batches[channel] = PhysicalEvidenceBatch(
                    channel,
                    values,
                    valid,
                    row_errors,
                    request.timestamp,
                    ids,
                    request.observation_revision,
                )
            return ContactEvidenceSample(
                PhysicalEvidenceFrame(
                    request.timestamp, request.observation_revision, ids, batches
                ),
                before.native_revision,
                before.topology_revision,
                dropped,
                shared_error,
                mask.device,
            )
        except _ERRORS as error:
            return self._empty_sample(
                request, channels, before, str(error) or type(error).__name__
            )

    def collect(
        self, context: _ObservationContext, request: PhysicalEvidenceRequest
    ) -> PhysicalEvidenceFrame:
        """Implement the shared physical-evidence provider port.

        Args:
            context: Physical observation owning this request's time and rows.
            request: Explicit physical-contact channel request.

        Returns:
            Synchronized contact channels aligned with the planning context.
        """
        if context.timestamp != request.timestamp or not torch.equal(
            context.env_ids, request.env_ids
        ):
            raise ValueError(
                "Physical contact request does not match its planning context."
            )
        return self.sample(request).frame
