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

"""Native load-bearing support, distinct from geometric containment."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Sequence

import numpy as np
from shapely.geometry import MultiPoint, Point
import torch

from embodichain.lab.sim.sensors.contact_sensor import ContactSensor, ContactSensorCfg

__all__: list[str] = []
_OBSERVATION_ERRORS = (
    AttributeError,
    AssertionError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


class ContactWitness:
    """Keep an actual contact patch only while its physical configuration persists."""

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
            self.value = (obj.clone(), ref.clone(), tuple(v.copy() for v in sample))
            self.sample_info = None
        return None if self.value is None else self.value[2]


class SupportContactSensor(ContactSensor):
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if hasattr(self, "support_reader"):
            reader = self.support_reader
            if hasattr(reader, "_reset_rows"):
                reader._reset_rows(env_ids)
            else:
                for row in range(len(reader.witnesses)) if env_ids is None else env_ids:
                    reader.witnesses[row].value = None

    @property
    def requires_substep_update(self) -> bool:
        return hasattr(self, "support_reader")

    def begin_control_step(self) -> None:
        super().begin_control_step()
        if hasattr(self, "support_reader"):
            self.support_reader._begin_control_step()

    def update_physics_step(self, dt: float) -> None:
        reader = getattr(self, "support_reader", None)
        revision = None if reader is None else reader._revision_or_none()
        super().update_physics_step(dt)
        if reader is not None:
            reader.capture(
                reader.object.get_local_pose(to_matrix=True),
                reader.reference.get_local_pose(to_matrix=True),
                dt=dt,
                native_revision=revision,
            )


def support_from_contacts(
    pairs: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
    impulses: np.ndarray,
    center_of_mass: np.ndarray,
    *,
    object_actor: int,
    reference_actor: int,
    tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    """Require upward load contacts whose measured patch supports the COM."""
    if not all(
        np.isfinite(v).all() for v in (positions, normals, impulses, center_of_mass)
    ):
        raise ValueError("Native support observations must be finite.")
    selected = (
        ((pairs[:, 0] == reference_actor) & (pairs[:, 1] == object_actor))
        | ((pairs[:, 1] == reference_actor) & (pairs[:, 0] == object_actor))
    ) & (impulses > 0)
    directions = np.where(
        (pairs[selected, 0] == reference_actor)[:, None],
        normals[selected],
        -normals[selected],
    )
    lengths = np.linalg.norm(directions, axis=-1)
    if (lengths <= 1e-8).any():
        raise ValueError("Native support normals must be nonzero.")
    upward = directions[:, 2] / lengths >= 0.9
    patch = positions[selected][upward]
    distance = None
    if len(patch):
        distance = float(
            MultiPoint(patch[:, :2]).convex_hull.distance(Point(center_of_mass[:2]))
        )
    accepted = distance is not None and distance <= tolerance
    return accepted, {
        "contacts_world": patch.tolist(),
        "center_of_mass_world": center_of_mass.tolist(),
        "com_patch_distance_m": distance,
        "com_tolerance_m": tolerance,
        "upward_impulse": float(
            (impulses[selected][upward] * directions[upward, 2] / lengths[upward]).sum()
        ),
    }


class NativeSupportObservation:
    """Read exact native rigid-object pairs without stepping or committing state."""

    def __init__(
        self,
        simulation: Any,
        obj: Any,
        reference: Any,
        *,
        entity_id: str,
        reference_id: str,
    ) -> None:
        self.object, self.reference = obj, reference
        self.entity_id, self.reference_id = entity_id, reference_id
        self._simulation = simulation
        uid = f"gen_sim_support__{obj.uid}__{reference.uid}"
        self.sensor = simulation.get_sensor(uid)
        if self.sensor is None:
            factories = simulation.SUPPORTED_SENSOR_TYPES
            simulation.SUPPORTED_SENSOR_TYPES = {
                **factories,
                "GenSimSupportContact": SupportContactSensor,
            }
            try:
                self.sensor = simulation.add_sensor(
                    ContactSensorCfg(
                        uid=uid,
                        sensor_type="GenSimSupportContact",
                        rigid_uid_list=[obj.uid, reference.uid],
                        max_contacts_per_env=512,
                    )
                )
            finally:
                simulation.SUPPORTED_SENSOR_TYPES = factories
        if hasattr(self.sensor, "support_reader"):
            owner = self.sensor.support_reader
            if (
                owner._simulation is not simulation
                or owner.object.uid != obj.uid
                or owner.reference.uid != reference.uid
            ):
                raise ValueError("Native support sensor belongs to another pair.")
            if owner.object is not obj or owner.reference is not reference:
                owner.object, owner.reference = obj, reference
                owner._reset_rows(None)
            self.witnesses = owner.witnesses
        else:
            self.witnesses = [ContactWitness() for _ in range(obj.num_instances)]
            self.sensor.support_reader = self
            self._clock = None
            self._base: tuple[int, float] | None = None
            self._offset, self._substep, self._serial = 0.0, 0, 0
            self._episode_generations = [0] * obj.num_instances
            self._actor_pairs: list[dict[str, Any]] = []
            self._topology_revision: int | None = None
            self._sample_revision: int | None = None
            self._error: str | None = None
        self.object_actors = self.sensor.get_actor_ids(obj.uid)
        self.reference_actors = self.sensor.get_actor_ids(reference.uid)
        # Native rest envelopes bound the contact patch's positional uncertainty.
        rest = []
        for body in (obj, reference):
            attrs = [entity.get_physical_attr() for entity in body._entities]
            rest.append(max(float(a.rest_offset) for a in attrs))
        self.tolerance = sum(rest) + 0.001
        if not math.isfinite(self.tolerance) or not 0 < self.tolerance <= 0.01:
            raise ValueError("Native support rest envelopes are unqualified.")

    def bind_clock(self, clock: Any) -> None:
        """Bind the shared execution clock without advancing it."""
        owner = self.sensor.support_reader
        if owner is not self:
            owner.bind_clock(clock)
            return
        if clock is not self._clock:
            self._clock, self._base = clock, None
            self._reset_rows(None)

    def _revision(self) -> int:
        # Pinned Default SDK dependency: public ContactBuffer has no source tick.
        if self._simulation.spawn_result.backend != "dexsim":
            raise ValueError("Native support source revision requires Default physics.")
        physics = self._simulation.get_world().get_physics_scene()
        getter = getattr(physics, "_get_contact_data_revision", None)
        if not callable(getter):
            raise ValueError("Native support source revision is unavailable.")
        revision = int(getter())
        if revision < 0:
            raise ValueError("Native support source revision is invalid.")
        return revision

    def _revision_or_none(self) -> int | None:
        if self._clock is None:
            return None
        try:
            return self._revision()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def _reset_rows(self, env_ids: Sequence[int] | None) -> None:
        revision = self._revision_or_none()
        for row in range(len(self.witnesses)) if env_ids is None else env_ids:
            witness = self.witnesses[row]
            witness.value, witness.sample_info = None, None
            witness.last_native_revision = revision
            self._episode_generations[row] += 1

    def _unavailable(self, error: Exception) -> None:
        self._error = str(error)
        for witness in self.witnesses:
            witness.value, witness.sample_info = None, None

    def _refresh_identity(self) -> None:
        capabilities = self.sensor.contact_capabilities
        if not capabilities.geometry or not capabilities.impulse:
            raise ValueError("Native support geometry/impulse capability unavailable.")
        topology = int(self._simulation.spawn_result.topology_revision)
        actors = [
            self.sensor.get_actor_ids(body.uid)
            for body in (self.object, self.reference)
        ]
        if any(ids.shape != (len(self.witnesses), 1) for ids in actors):
            raise ValueError("Native support actor mapping is ambiguous.")
        pairs = []
        for row in range(len(self.witnesses)):
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
        if self._actor_pairs and (
            pairs != self._actor_pairs or topology != self._topology_revision
        ):
            self._reset_rows(None)
        self.object_actors, self.reference_actors = actors
        self._actor_pairs, self._topology_revision = pairs, topology

    def _begin_control_step(self) -> None:
        if self._clock is None:
            return
        try:
            index, now, dt = (
                self._clock.step_index,
                self._clock.now(),
                self._clock.step_dt,
            )
            if not math.isfinite(now) or not math.isfinite(dt) or dt <= 0 or index < 0:
                raise ValueError("Native support clock is unqualified.")
            if self._base is not None and (
                index != self._base[0] + 1 or now <= self._base[1]
            ):
                raise ValueError("Native support control clock did not advance once.")
            self._base, self._offset, self._substep = (index, now), 0.0, 0
            self._refresh_identity()
            self._error = None
        except _OBSERVATION_ERRORS as error:
            self._unavailable(error)

    def _start_clocked_capture(
        self, obj: torch.Tensor, ref: torch.Tensor, dt: float, before: int | None
    ) -> bool:
        try:
            if self._error is not None or self._base is None:
                raise ValueError(self._error or "Native support interval is missing.")
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
                raise ValueError("Native support interval exceeds the control cadence.")
            self._refresh_identity()
            revision = self._revision()
            if before is None or revision != before or self.sensor.dropped_contacts:
                raise ValueError(
                    "Native support sample changed during fetch or saturated."
                )
            if not torch.isfinite(obj).all() or not torch.isfinite(ref).all():
                raise ValueError("Native support poses must be finite.")
            self._sample_revision = revision
            return True
        except _OBSERVATION_ERRORS as error:
            self._unavailable(error)
            return False

    def _validate_clocked_observation(self) -> None:
        try:
            if self._clock is None or self._error is not None:
                raise ValueError(
                    self._error or "Native support execution clock unavailable."
                )
            self._refresh_identity()
            if (
                self._revision() != self._sample_revision
                or self.sensor.dropped_contacts
                or not math.isfinite(self._clock.now())
                or self._base is None
                or self._clock.step_index != self._base[0] + 1
                or not math.isclose(
                    self._offset, self._clock.step_dt, rel_tol=0, abs_tol=1e-9
                )
            ):
                raise ValueError(
                    "Native support has an unobserved step or invalid interval."
                )
        except _OBSERVATION_ERRORS as error:
            self._unavailable(error)
            raise ValueError(self._error) from error

    def capture(
        self,
        object_pose: torch.Tensor,
        reference_pose: torch.Tensor,
        *,
        dt: float | None = None,
        native_revision: int | None = None,
    ) -> None:
        clocked = dt is not None and self._clock is not None
        if clocked and not self._start_clocked_capture(
            object_pose, reference_pose, dt, native_revision
        ):
            return
        data = self.sensor.get_data()
        for row, witness in enumerate(self.witnesses):
            mask = data["is_valid"][row]
            pairs = data["user_ids"][row, mask]
            if clocked:
                raw = [
                    data[key][row, mask]
                    for key in ("user_ids", "position", "normal", "impulse")
                ]
                if (
                    any(not torch.isfinite(value).all() for value in raw)
                    or (pairs < 0).any()
                ):
                    self._unavailable(
                        ValueError("Native support data or counterpart is unqualified.")
                    )
                    return
                witness.update(object_pose[row], reference_pose[row], None)
                if witness.last_native_revision == native_revision:
                    continue
                witness.last_native_revision = native_revision
            selected = (
                (
                    (pairs[:, 0] == self.reference_actors[row, 0])
                    & (pairs[:, 1] == self.object_actors[row, 0])
                )
                | (
                    (pairs[:, 1] == self.reference_actors[row, 0])
                    & (pairs[:, 0] == self.object_actors[row, 0])
                )
            ) & (data["impulse"][row, mask] > 0)
            sample = (
                tuple(
                    data[k][row, mask][selected].detach().cpu().numpy()
                    for k in ("user_ids", "position", "normal", "impulse")
                )
                if selected.any()
                else None
            )
            witness.update(
                object_pose[row],
                reference_pose[row],
                sample,
                invalid=self.sensor.dropped_contacts != 0,
            )
            if clocked and sample is not None:
                witness.sample_info = dict(
                    timestamp=self._base[1] + self._offset,
                    control_step=self._base[0] + 1,
                    substep=self._substep,
                    serial=self._serial,
                    episode_generation=self._episode_generations[row],
                    native_revision=native_revision,
                    topology_revision=self._topology_revision,
                )

    def observe(
        self,
        object_pose: torch.Tensor,
        reference_pose: torch.Tensor,
        *,
        support_contract_revision: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        owner = self.sensor.support_reader
        if owner is not self:
            accepted, metadata = owner.observe(
                object_pose,
                reference_pose,
                support_contract_revision=support_contract_revision,
            )
            return accepted, {
                **metadata,
                "entity": self.entity_id,
                "reference": self.reference_id,
            }
        clocked = support_contract_revision is not None
        if clocked:
            if (
                type(support_contract_revision) is not int
                or support_contract_revision != 1
            ):
                raise ValueError("Unsupported native support contract revision.")
            self._validate_clocked_observation()
        else:
            if self._clock is None:
                self.sensor.update()
                self.capture(object_pose, reference_pose)
            else:
                # A legacy consumer cannot erase another declared strong window.
                self._validate_clocked_observation()
        data = self.sensor.get_data()
        com = self.object.body_data.com_pose[:, :3].to(object_pose)
        center = (object_pose[:, :3, :3] @ com[:, :, None]).squeeze(-1) + object_pose[
            :, :3, 3
        ]
        accepted, rows = [], []
        for row in range(object_pose.shape[0]):
            mask = data["is_valid"][row]
            if self.sensor.dropped_contacts:
                raise ValueError("Native support contact query saturated.")
            sample = self.witnesses[row].update(
                object_pose[row], reference_pose[row], None
            )
            info = self.witnesses[row].sample_info
            if clocked:
                if info is not None and (
                    info["timestamp"] > self._clock.now() + 1e-9
                    or info["episode_generation"] != self._episode_generations[row]
                ):
                    raise ValueError(
                        "Native support sample clock/generation is unqualified."
                    )
                if info is None:
                    sample = (
                        np.empty((0, 2)),
                        np.empty((0, 3)),
                        np.empty((0, 3)),
                        np.empty(0),
                    )
            if sample is None:
                sample = tuple(
                    data[k][row, mask].detach().cpu().numpy()
                    for k in ("user_ids", "position", "normal", "impulse")
                )
            result, measurements = support_from_contacts(
                *sample,
                center[row].detach().cpu().numpy(),
                object_actor=int(self.object_actors[row, 0]),
                reference_actor=int(self.reference_actors[row, 0]),
                tolerance=self.tolerance,
            )
            accepted.append(result)
            if clocked:
                measurements.update(
                    sample=None if info is None else dict(info),
                    retained=info is not None and info["serial"] != self._serial,
                    raw_contacts={
                        key: value.tolist()
                        for key, value in zip(
                            ("user_ids", "position", "normal", "impulse"), sample
                        )
                    },
                )
            rows.append(measurements)
        metadata = {
            "scope": "native_contact_support",
            "entity": self.entity_id,
            "reference": self.reference_id,
            "object_pose": object_pose.detach().cpu().tolist(),
            "reference_pose": reference_pose.detach().cpu().tolist(),
            "supported": accepted,
            "env_ids": list(range(len(accepted))),
            "rows": rows,
        }
        if clocked:
            metadata.update(
                support_contract_revision=1,
                clock="environment_step",
                observed_at=self._clock.now(),
                observed_step=self._clock.step_index,
                step_dt=self._clock.step_dt,
                episode_generations=list(self._episode_generations),
                actor_pairs=self._actor_pairs,
            )
        return (
            torch.tensor(accepted, dtype=torch.bool, device=object_pose.device),
            metadata,
        )
