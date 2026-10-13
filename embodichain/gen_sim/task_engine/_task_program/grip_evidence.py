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

"""Target-specific native pad evidence for GenSim ordinary manipulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.sensors.contact_sensor import (
    ContactSensorCfg,
    ArticulationContactFilterCfg,
)
from embodichain.lab.task_program.integrations._configured_services import (
    _JointPositionConstraintEvidenceProviderFactory,
)
from embodichain.lab.task_program.semantics.evidence import (
    BinaryEffectObservation,
    BinaryEvidenceKind,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
)

__all__: list[str] = []

SENSOR_UID = "gen_sim_grip_contacts"


def classify_contacts(
    pairs: torch.Tensor,
    normals: torch.Tensor,
    impulses: torch.Tensor,
    *,
    target: int,
    pads: tuple[int, int],
) -> tuple[bool, bool]:
    """Distinguish opposed bilateral contact, clear release, and unresolved contact."""
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


class NativeGripObserver:
    def __init__(
        self,
        sensor: Any,
        object_actors: dict[str, torch.Tensor],
        pad_actors: dict[str, torch.Tensor],
    ) -> None:
        self.sensor = sensor
        self.object_actors = object_actors
        self.pad_actors = pad_actors
        self.trace: deque[dict[str, Any]] = deque(maxlen=4096)
        sensor._gen_sim_grip_observer = self

    def __call__(self, query: Any, context: Any) -> BinaryEffectObservation:
        self.sensor.update()
        data = self.sensor.get_data()
        values, valid, errors, records = [], [], [], []
        uid, part = query.expectation.object_id, query.source.address.control_part
        for env in context.env_ids.tolist():
            mask = data["is_valid"][env]
            pairs = data["user_ids"][env, mask]
            normals = data["normal"][env, mask]
            impulses = data["impulse"][env, mask]
            target = int(self.object_actors[uid][env, 0])
            pads = tuple(self.pad_actors[part][env].tolist())
            value, ok = classify_contacts(
                pairs, normals, impulses, target=target, pads=pads
            )
            ok &= self.sensor.dropped_contacts == 0
            values.append(value)
            valid.append(ok)
            errors.append(
                None
                if ok
                else "Target grip contact is incomplete, one-sided, non-opposing, or saturated."
            )
            selected = (pairs == target).any(-1)
            records.append(
                {
                    "env_id": env,
                    "value": value,
                    "valid": ok,
                    "pairs": pairs[selected].tolist(),
                    "normals": normals[selected].tolist(),
                    "impulses": impulses[selected].tolist(),
                }
            )
        self.trace.append(
            {
                "timestamp": context.timestamp,
                "observation_revision": context.observation_revision,
                "object_id": uid,
                "control_part": part,
                "expected": query.clause.expected,
                "dropped_contacts": self.sensor.dropped_contacts,
                "rows": records,
            }
        )
        return BinaryEffectObservation(
            torch.tensor(values, dtype=torch.bool, device=context.env_ids.device),
            torch.tensor(valid, dtype=torch.bool, device=context.env_ids.device),
            tuple(errors),
        )


@dataclass(frozen=True, slots=True)
class NativeGripEvidenceFactory:
    """Retain the public evidence provider and replace only its grasp observation."""

    source: _JointPositionConstraintEvidenceProviderFactory
    object_bindings: tuple[tuple[str, str], ...]
    algorithm_revision: int = 1
    provider_id: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_REVISION

    def create(
        self,
        *,
        simulation: Any,
        robot: Any,
        scene_registry: Any,
        engine: Any,
        scene_provider: Any,
    ) -> Any:
        provider = self.source.create(
            simulation=simulation,
            robot=robot,
            scene_registry=scene_registry,
            engine=engine,
            scene_provider=scene_provider,
        )
        objects = self.source._resolve_object_ids(scene_registry)
        native = dict(self.object_bindings)
        names_by_part = {
            part: tuple(
                n
                for n in robot.link_names
                if n.startswith(part.removesuffix("_eef") + "_")
                and n.endswith("_finger_pad")
            )
            for part in self.source.control_parts
        }
        if any(len(names) != 2 for names in names_by_part.values()):
            raise ValueError(
                "Native GenSim grip evidence requires two explicitly named finger pads per hand."
            )
        sensor = simulation.get_sensor(SENSOR_UID)
        if sensor is None:
            sensor = simulation.add_sensor(
                ContactSensorCfg(
                    uid=SENSOR_UID,
                    rigid_uid_list=[native[uid] for uid in objects],
                    articulation_cfg_list=[
                        ArticulationContactFilterCfg(
                            articulation_uid=robot.uid,
                            link_name_list=[
                                n for names in names_by_part.values() for n in names
                            ],
                        )
                    ],
                    max_contacts_per_env=1024,
                )
            )
        observer = NativeGripObserver(
            sensor,
            {uid: sensor.get_actor_ids(native[uid]) for uid in objects},
            {
                part: sensor.get_actor_ids(robot.uid, names)
                for part, names in names_by_part.items()
            },
        )
        provider._binary_observers[BinaryEvidenceKind.CONSTRAINT] = observer
        return provider


def with_native_grip_evidence(registration: Any) -> Any:
    source = registration.control_part_evidence_factory
    if not isinstance(source, _JointPositionConstraintEvidenceProviderFactory):
        return registration
    return replace(
        registration,
        relation_grounders=(),
        control_part_evidence_factory=NativeGripEvidenceFactory(
            source,
            tuple(
                (b.entity_id, b.simulation_uid)
                for b in registration.scene_binding.rigid_objects
            ),
        ),
    )
