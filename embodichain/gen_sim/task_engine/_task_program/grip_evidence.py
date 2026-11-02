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
from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Any, ClassVar
import xml.etree.ElementTree as ET

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
from embodichain.lab.task_program.semantics.effects import HeldObjectRelation

__all__: list[str] = []

SENSOR_UID = "gen_sim_grip_contacts"


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
        self.cumulative_query_count = 0
        self.recorded_query_count = 0
        self.evicted_query_count = 0
        self.nonfinite_query_count = 0
        self.dropped_query_count = 0
        self.maximum_dropped_contacts = 0
        self.release_opening: Any = None
        sensor._gen_sim_grip_observer = self

    def snapshot_trace(self) -> dict[str, Any]:
        """Own raw observations without sampling, qualifying success or hiding loss."""
        nonfinite: list[dict[str, str]] = []
        records = _finite_trace_copy(self.trace, "trace", nonfinite)
        return {
            "schema_version": "gen_sim.grip-contact-evidence/v1",
            "source": "native_contact_sensor",
            "observed": self.recorded_query_count > 0,
            "sensor_uid": SENSOR_UID,
            "scope": "observer_lifetime_raw_queries_not_task_success",
            "actor_bindings": {
                "scope": "observer_lifetime_native_ids_not_contact_evidence",
                "env_axis": "native_env_id",
                "object_actor_ids": {
                    uid: actors.detach().cpu().tolist()
                    for uid, actors in self.object_actors.items()
                },
                "pad_actor_ids": {
                    part: actors.detach().cpu().tolist()
                    for part, actors in self.pad_actors.items()
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

    def __call__(self, query: Any, context: Any) -> BinaryEffectObservation:
        self.cumulative_query_count += 1
        self.sensor.update()
        data = self.sensor.get_data()
        values, valid, errors, records = [], [], [], []
        uid, part = query.expectation.object_id, query.source.address.control_part
        finite_query = True
        for env in context.env_ids.tolist():
            mask = data["is_valid"][env]
            pairs = data["user_ids"][env, mask]
            normals = data["normal"][env, mask]
            impulses = data["impulse"][env, mask]
            finite = bool(
                torch.isfinite(normals).all() and torch.isfinite(impulses).all()
            )
            finite_query &= finite
            target = int(self.object_actors[uid][env, 0])
            pads = tuple(self.pad_actors[part][env].tolist())
            value, ok = classify_contacts(
                pairs, normals, impulses, target=target, pads=pads
            )
            opening = None
            if (
                self.release_opening is not None
                and query.clause.expected is False
                and query.expectation.relation is HeldObjectRelation.DETACHED
                and not value
                and not ok
                and finite
                and self.sensor.dropped_contacts == 0
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
                    opening = self.release_opening(part, env)
                    finite &= opening["finite"]
                    finite_query &= finite
                    ok = opening["open"] is True and finite
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
                    "finite": finite,
                    "error": errors[-1],
                    "pairs": pairs[selected].tolist(),
                    "normals": normals[selected].tolist(),
                    "impulses": impulses[selected].tolist(),
                    "raw_contacts": {
                        "pairs": pairs.tolist(),
                        "normals": normals.tolist(),
                        "impulses": impulses.tolist(),
                    },
                    **({"release_opening": opening} if opening is not None else {}),
                }
            )
        if self.trace.maxlen is not None and len(self.trace) == self.trace.maxlen:
            self.evicted_query_count += 1
        self.recorded_query_count += 1
        self.nonfinite_query_count += int(not finite_query)
        self.dropped_query_count += int(self.sensor.dropped_contacts > 0)
        self.maximum_dropped_contacts = max(
            self.maximum_dropped_contacts, int(self.sensor.dropped_contacts)
        )
        self.trace.append(
            {
                "query_index": self.cumulative_query_count,
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
        if self.algorithm_revision == 2:
            observer.release_opening = _open_release_reader(robot, self.source)
        elif self.algorithm_revision != 1:
            raise ValueError("Unsupported native grip algorithm revision.")
        provider._binary_observers[BinaryEvidenceKind.CONSTRAINT] = observer
        return provider


def _open_release_reader(robot: Any, source: Any) -> Any:
    """Observe driving joints, not compliant mimic-joint errors, at release time."""
    joints = {
        j.attrib["name"]: j
        for j in ET.parse(robot.cfg.fpath).getroot().findall("joint")
    }
    drivers = {}
    for part in source.control_parts:
        names = [robot.joint_names[int(i)] for i in robot.get_joint_ids(name=part)]
        if len(names) != len(source.open_qpos) or any(
            name not in joints for name in names
        ):
            raise ValueError(
                "Release opening requires the exact configured joint order."
            )
        indices = [
            i
            for i, name in enumerate(names)
            if joints[name].find("mimic") is None
            and joints[name].attrib["type"] != "fixed"
        ]
        if not indices:
            raise ValueError("Release opening has no independent driving joint.")
        drivers[part] = (names, indices)

    def observe(part: str, env: int) -> dict[str, Any]:
        names, indices = drivers[part]
        qpos = robot.get_qpos(name=part)
        if (
            qpos.ndim != 2
            or qpos.shape[1] != len(names)
            or not 0 <= env < qpos.shape[0]
        ):
            raise ValueError(
                "Release joint observation has an invalid shape or environment."
            )
        measured = qpos[env, indices]
        target = measured.new_tensor(source.open_qpos)[indices]
        finite = bool(torch.isfinite(qpos[env]).all())
        return {
            "finite": finite,
            "open": finite
            and bool(((measured - target).abs() < source.minimum_displacement).all()),
            "driving_joint_names": [names[i] for i in indices],
            "measured_qpos": measured.detach().cpu().tolist(),
            "open_qpos": target.detach().cpu().tolist(),
            "tolerance": source.minimum_displacement,
            "scope": "experimental_open_hand_without_bilateral_grip_not_contact_free_separation",
        }

    return observe


def with_native_grip_evidence(
    registration: Any, *, experimental_open_release: bool = False
) -> Any:
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
            algorithm_revision=2 if experimental_open_release else 1,
        ),
    )
