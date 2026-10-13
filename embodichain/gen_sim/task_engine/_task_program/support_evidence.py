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

import math
from typing import Any

import numpy as np
from shapely.geometry import MultiPoint, Point
import torch

from embodichain.lab.sim.sensors.contact_sensor import ContactSensor, ContactSensorCfg

__all__: list[str] = []


class ContactWitness:
    """Keep an actual contact patch only while its physical configuration persists."""

    def __init__(self) -> None:
        self.value: tuple[torch.Tensor, torch.Tensor, tuple[np.ndarray, ...]] | None = (
            None
        )

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
        if not invalid and sample is not None:
            self.value = (obj.clone(), ref.clone(), tuple(v.copy() for v in sample))
        return None if self.value is None else self.value[2]


class SupportContactSensor(ContactSensor):
    @property
    def requires_substep_update(self) -> bool:
        return hasattr(self, "support_reader")

    def update_physics_step(self, dt: float) -> None:
        super().update_physics_step(dt)
        if hasattr(self, "support_reader"):
            reader = self.support_reader
            reader.capture(
                reader.object.get_local_pose(to_matrix=True),
                reader.reference.get_local_pose(to_matrix=True),
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
            self.witnesses = self.sensor.support_reader.witnesses
        else:
            self.witnesses = [ContactWitness() for _ in range(obj.num_instances)]
            self.sensor.support_reader = self
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

    def capture(self, object_pose: torch.Tensor, reference_pose: torch.Tensor) -> None:
        data = self.sensor.get_data()
        for row, witness in enumerate(self.witnesses):
            mask = data["is_valid"][row]
            pairs = data["user_ids"][row, mask]
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

    def observe(
        self, object_pose: torch.Tensor, reference_pose: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.sensor.update()
        self.capture(object_pose, reference_pose)
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
            rows.append(measurements)
        return torch.tensor(accepted, dtype=torch.bool, device=object_pose.device), {
            "scope": "native_contact_support",
            "entity": self.entity_id,
            "reference": self.reference_id,
            "object_pose": object_pose.detach().cpu().tolist(),
            "reference_pose": reference_pose.detach().cpu().tolist(),
            "supported": accepted,
            "env_ids": list(range(len(accepted))),
            "rows": rows,
        }
