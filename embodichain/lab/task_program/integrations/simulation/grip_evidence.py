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

"""Declarative native grip bindings, runtime factories and semantic adaptation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.sensors.contact.sensor import (
    ContactSensorCfg,
    ArticulationContactFilterCfg,
)
from embodichain.lab.sim.evidence import PhysicalEvidenceRequest, PhysicalEvidenceBatch
from embodichain.lab.sim.atomic_actions.grip_evidence import (
    GripContactObserver,
    DriverOpeningBinding,
)
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
from embodichain.lab.sim.sensors.contact.sampling import ContactEvidenceSampler

__all__ = [
    "NativeGripEvidenceCfg",
    "NativeGripEvidenceFactory",
    "NativeGripEvidenceProvider",
    "NativeGripObserver",
]


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact identifier.")


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


class NativeGripObserver(GripContactObserver):
    """Adapt semantic constraint queries to the shared physical grip observer."""

    def __call__(
        self, query: BinaryEffectEvidenceQuery, context: EffectEvidenceCollectionContext
    ) -> BinaryEffectObservation:
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
        result = self.observe(
            PhysicalEvidenceRequest(
                context.timestamp, context.observation_revision, context.env_ids
            ),
            object_id=expectation.object_id,
            control_part=address.control_part,
            expected_contact=query.clause.expected,
            releasing=expectation.relation is HeldObjectRelation.DETACHED,
        )
        return BinaryEffectObservation(
            result.values, result.valid, result.acquisition_errors
        )


class NativeGripEvidenceProvider(ControlPartSimulationEvidenceProvider):
    """Standard control-part provider exposing its owned grip observer for audits."""

    def __init__(
        self,
        robot: object,
        observer: NativeGripObserver,
        *,
        scene_provider: object,
        observation_guard: Callable[[], None] | None = None,
    ) -> None:
        """Construct through the public observer-injection boundary.

        Args:
            robot: Exact assembled robot evidence source.
            observer: Instance-owned native grip observer.
            scene_provider: Shared scene observation source.
            observation_guard: Optional actual native-source ownership validator.
        """
        super().__init__(
            robot, scene_provider=scene_provider, constraint_observer=observer
        )
        if observation_guard is not None and not callable(observation_guard):
            raise TypeError("observation_guard must be callable or None.")
        self._grip_observer = observer
        self._observation_guard = observation_guard

    def capture_grip_observation(
        self, request: PhysicalEvidenceRequest, *, object_id: str, control_part: str
    ) -> tuple[PhysicalEvidenceBatch, dict[str, Any]]:
        """Return one bilateral acquisition and its exact owned raw query.

        Args:
            request: Actual current observation identity and native row selection.
            object_id: Explicit target bound by this provider.
            control_part: Explicit hand whose pad actors were declared.

        Returns:
            Owned batch and snapshot from the same single observer acquisition.
            This adds no collect, refresh, or secondary acquisition.
        """
        if self._observation_guard is not None:
            self._observation_guard()
        try:
            query_index = self._grip_observer.cumulative_query_count + 1
            batch = self._grip_observer.observe(
                request,
                object_id=object_id,
                control_part=control_part,
                expected_contact=True,
            )
            sample = self._grip_observer.snapshot_latest_query(
                object_id=object_id, control_part=control_part, query_index=query_index
            )
            return batch.snapshot(), sample
        finally:
            if self._observation_guard is not None:
                self._observation_guard()

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

    def contact_sensor_cfg(self, robot_uid: str) -> ContactSensorCfg:
        """Describe the exact ordinary sensor scope without native resource creation.

        Args:
            robot_uid: Explicit selected physical robot identity.

        Returns:
            Standard contact configuration with ordered declared rigid owners,
            both pad sets, the declared capacity and match-both filtering.
        """
        _identifier(robot_uid, "robot_uid")
        return ContactSensorCfg(
            uid=self.sensor_uid,
            rigid_uid_list=[uid for _, uid in self.object_bindings],
            articulation_cfg_list=[
                ArticulationContactFilterCfg(
                    articulation_uid=robot_uid,
                    link_name_list=[
                        name for _, names in self.pad_bindings for name in names
                    ],
                )
            ],
            max_contacts_per_env=self.max_contacts_per_env,
        )

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
        # Validate calibration before acquiring any sensor resources. Atomic
        # bindings own the measured opening predicate; this layer only routes it.
        opening_readers = (
            {
                binding.control_part: binding.bind(robot)
                for binding in self.opening_bindings
            }
            if self.algorithm_revision == 2
            else {}
        )
        reader = (
            (lambda part, env_id: opening_readers[part](env_id))
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

        def require_observation_owner() -> None:
            if (
                simulation.get_sensor(self.sensor_uid) is not sensor
                or engine.robot is not robot
            ):
                raise ValueError(
                    "Native grip observation ownership changed; rebind the provider."
                )

        return NativeGripEvidenceProvider(
            robot,
            observer,
            scene_provider=scene_provider,
            observation_guard=require_observation_owner,
        )
