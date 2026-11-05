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

"""Episode-owned recorded contact and pose acquisition for explicit windows."""

from __future__ import annotations
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import MISSING
from typing import Any, TYPE_CHECKING
import math

from embodichain.utils import configclass
from embodichain.lab.sim.sensors.contact.timeline import (
    ContactTimeline,
    ContactTimelineCfg,
)
from embodichain.lab.sim.atomic_actions.grip_evidence import DriverOpeningBinding
from embodichain.lab.sim.sensors.joint_opening import create_joint_opening_reader
from .observations import SimulationStateObservationCollector
from ._recorded_observations import _recorded_pose_window

if TYPE_CHECKING:
    from embodichain.lab.sim.sensors.contact.sensor import ContactSensorCfg
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from ..catalog import SimulationTaskProgramRegistration

__all__ = ["SimulationRecordedContactCfg", "SimulationRecordedContactCollector"]


@configclass
class SimulationRecordedContactCfg:
    """Explicit native sensor, calibrated hands and bounded timeline retention.

    The runtime owner supplies ``retention.step_dt`` from its actual control
    cadence. An existing sensor is required; this declaration creates no sensor.
    Its filter must include every registered rigid UID and the explicit pad
    links with ``filter_need_both_actor=False``. Contacts against unknown external
    actors must remain observable; filtering to selected pairs can erase load
    paths and cannot qualify this acquisition contract.
    """

    sensor_uid: str = MISSING
    retention: ContactTimelineCfg = MISSING
    pad_bindings: tuple[tuple[str, tuple[str, str]], ...] = ()
    opening_bindings: tuple[DriverOpeningBinding, ...] = ()
    observer_name: str = "task_program.recorded_contacts"

    def snapshot(self) -> SimulationRecordedContactCfg:
        """Own and validate source selections and bounded retention without simulation."""
        self.validate()
        result = deepcopy(self)
        for value in (result.sensor_uid, result.observer_name):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Recorded contact names must be exact identifiers.")
        if type(result.retention) is not ContactTimelineCfg:
            raise TypeError("Recorded retention requires ContactTimelineCfg.")
        result.retention.validate()
        value = result.retention.step_dt
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(
                "Recorded contacts require a finite positive control cadence."
            )
        for name in (
            "max_control_steps",
            "max_contact_rows",
            "max_substeps_per_control",
            "max_errors_per_env",
        ):
            if (
                type(getattr(result.retention, name)) is not int
                or getattr(result.retention, name) < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(result.pad_bindings, (tuple, list)):
            raise TypeError("Recorded pads require explicit ordered bindings.")
        pads = []
        for row in result.pad_bindings:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("Recorded pads require control-part/link-list pairs.")
            part, names = row
            if (
                type(part) is not str
                or not part
                or part != part.strip()
                or not isinstance(names, (tuple, list))
                or len(names) != 2
                or len(set(names)) != 2
                or any(type(v) is not str or not v or v != v.strip() for v in names)
            ):
                raise ValueError(
                    "Recorded pads require two distinct exact links per control part."
                )
            pads.append((part, tuple(names)))
        if len({part for part, _ in pads}) != len(pads):
            raise ValueError("Recorded pads cannot duplicate control parts.")
        result.pad_bindings = tuple(pads)
        if not isinstance(result.opening_bindings, (tuple, list)) or any(
            type(v) is not DriverOpeningBinding for v in result.opening_bindings
        ):
            raise TypeError(
                "Recorded openings require explicit DriverOpeningBinding values."
            )
        result.opening_bindings = tuple(result.opening_bindings)
        opening_parts = [value.control_part for value in result.opening_bindings]
        if len(set(opening_parts)) != len(opening_parts) or not set(opening_parts) <= {
            part for part, _ in pads
        }:
            raise ValueError(
                "Recorded openings must select unique declared hand parts."
            )
        return result

    def contact_sensor_cfg(
        self, *, robot_uid: str, rigid_uids: Sequence[str]
    ) -> ContactSensorCfg:
        """Declare the complete source single-sided 1024-contact native query.

        Args:
            robot_uid: Exact physical robot UID selected by the host.
            rigid_uids: Complete ordered native rigid roster from scene binding.

        Returns:
            Ordinary ContactSensorCfg; no simulator, sensor or native query is
            created. Explicit pad link order and source retention stay unchanged.
        """
        from embodichain.lab.sim.sensors.contact.sensor import (
            ContactSensorCfg,
            ArticulationContactFilterCfg,
        )

        cfg = self.snapshot()
        if (
            type(robot_uid) is not str
            or not robot_uid
            or robot_uid != robot_uid.strip()
        ):
            raise ValueError("Recorded contact robot UID must be exact.")
        if isinstance(rigid_uids, str) or not isinstance(rigid_uids, Sequence):
            raise TypeError("Recorded rigid UIDs require an ordered sequence.")
        uids = tuple(rigid_uids)
        if len(set(uids)) != len(uids) or any(
            type(v) is not str or not v or v != v.strip() for v in uids
        ):
            raise ValueError("Recorded rigid UIDs must be unique exact identities.")
        links = [name for _, names in cfg.pad_bindings for name in names]
        if len(set(links)) != len(links):
            raise ValueError("Recorded pad links must name distinct native actors.")
        return ContactSensorCfg(
            uid=cfg.sensor_uid,
            sensor_type="ContactSensor",
            rigid_uid_list=list(uids),
            articulation_cfg_list=(
                [
                    ArticulationContactFilterCfg(
                        articulation_uid=robot_uid, link_name_list=links
                    )
                ]
                if links
                else []
            ),
            max_contacts_per_env=1024,
            filter_need_both_actor=False,
        )

    def validate_num_envs(self, num_envs: int) -> None:
        """Reject unsupported batched source acquisition before simulator creation."""
        if type(num_envs) is not int or num_envs != 1:
            raise ValueError("Recorded source contacts require B=1.")


class SimulationRecordedContactCollector(SimulationStateObservationCollector):
    """Register and own one physical timeline for an initialized bridge episode.

    Construct after episode initialization, before the first control step. The
    host closes this collector before replacing the session or resetting the
    episode, then constructs a fresh collector. Sensor reset cannot rebind old
    episode identities. No capture method refreshes contacts or steps physics.
    """

    def __init__(
        self,
        environment: EmbodiedEnv,
        bridge: TaskProgramDemoBridge,
        registration: SimulationTaskProgramRegistration,
        cfg: SimulationRecordedContactCfg,
        *,
        observations: ObservationManager | None = None,
    ) -> None:
        """Bind actual native owners and attach before any sample can be missed.

        Args:
            environment: Initialized recording environment.
            bridge: Actual fresh execution bridge, with control clock at zero.
            registration: Sealed scene/profile declaration used by this runtime.
            cfg: Existing sensor, explicit hand calibration and retention limits.
            observations: Actual canonical owner of recorded raw poses and boundaries.
        """
        import torch
        from embodichain.lab.sim.sensors.contact.sampling import (
            ContactEvidenceSampler,
            native_contact_revision,
        )

        super().__init__(environment, bridge, registration, observations=observations)
        cfg = cfg.snapshot()
        cfg.validate_num_envs(environment.num_envs)
        self._cfg = deepcopy(cfg)
        self._episode_owner = environment.episode_context
        self._simulation_owner = environment.sim
        self._closed = False
        for value in (cfg.sensor_uid, cfg.observer_name):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Recorded contact names must be exact identifiers.")
        if cfg.retention.step_dt != environment.step_dt:
            raise ValueError("Recorded contact cadence differs from the environment.")
        if not isinstance(cfg.pad_bindings, (tuple, list)):
            raise ValueError("Contact pads require explicit part/link declarations.")
        pads = {}
        for part, names in cfg.pad_bindings:
            if (
                type(part) is not str
                or not part
                or part != part.strip()
                or part in pads
                or not isinstance(names, (tuple, list))
                or len(names) != 2
                or len(set(names)) != 2
                or any(
                    type(name) is not str or not name or name != name.strip()
                    for name in names
                )
            ):
                raise ValueError(
                    "Contact pad declarations require distinct named pairs."
                )
            pads[part] = tuple(names)
        if {item.control_part for item in cfg.opening_bindings} != set(pads):
            raise ValueError(
                "Every recorded hand requires exact driver opening calibration."
            )
        sensor = environment.sim.get_sensor(cfg.sensor_uid)
        if (
            sensor is None
            or sensor.num_instances != environment.num_envs
            or getattr(environment, "sensors", {}).get(cfg.sensor_uid) is not sensor
        ):
            raise ValueError(
                "Recorded contact sensor is missing from its ordinary substep suite or native batch."
            )
        required_uids = tuple(
            binding.simulation_uid
            for binding in registration.scene_binding.rigid_objects
        )
        required_links = tuple(name for names in pads.values() for name in names)
        sensor_cfg = sensor.cfg
        articulation_scope = tuple(
            (item.articulation_uid, tuple(item.link_name_list))
            for item in sensor_cfg.articulation_cfg_list
        )
        expected_articulation = (
            ((environment.robot.uid, required_links),) if required_links else ()
        )
        if (
            sensor_cfg.max_contacts_per_env != 1024
            or tuple(sensor_cfg.rigid_uid_list) != required_uids
            or articulation_scope != expected_articulation
            or sensor_cfg.filter_need_both_actor is not False
        ):
            raise ValueError(
                "Recorded contacts require the complete single-sided source sensor scope."
            )
        self._sensor_scope = (
            required_uids,
            articulation_scope,
            sensor_cfg.max_contacts_per_env,
        )
        self._sensor = sensor
        self._robot = environment.robot
        sampler = ContactEvidenceSampler(
            sensor,
            native_revision_reader=lambda: native_contact_revision(environment.sim),
            topology_revision_reader=lambda: int(
                environment.sim.spawn_result.topology_revision
            ),
        )
        stamp = sampler.source_stamp()
        if stamp.errors or stamp.topology_revision is None:
            raise ValueError(
                "Recorded contact native/topology source stamps are unavailable."
            )
        self._topology = stamp.topology_revision
        self._object_actors = {}
        self._body_types = {}
        for entity in self._rigid_owners:
            uid, native = self._native(entity, articulated=False)
            actors = sensor.get_actor_ids(uid)
            if (
                not isinstance(actors, torch.Tensor)
                or actors.shape != (environment.num_envs, 1)
                or actors.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    "Recorded rigid actors require exact integer native rows."
                )
            if native.body_type not in {"dynamic", "static", "kinematic"}:
                raise ValueError("Recorded rigid body type is unavailable.")
            self._object_actors[entity] = actors.detach().cpu().tolist()
            self._body_types[entity] = native.body_type
        self._pad_actors = {}
        for part, names in pads.items():
            actors = sensor.get_actor_ids(self._robot.uid, names)
            if (
                not isinstance(actors, torch.Tensor)
                or actors.shape != (environment.num_envs, 2)
                or actors.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    "Recorded pad actors require exact integer native pairs."
                )
            self._pad_actors[part] = actors.detach().cpu().tolist()
        for row in range(environment.num_envs):
            inventory = [
                actor
                for values in self._object_actors.values()
                for actor in values[row]
            ]
            inventory += [
                actor for values in self._pad_actors.values() for actor in values[row]
            ]
            if len(set(inventory)) != len(inventory) or any(
                type(actor) is not int for actor in inventory
            ):
                raise ValueError(
                    "Recorded native object and pad actors must be disjoint."
                )
        opening = (
            create_joint_opening_reader(self._robot, tuple(cfg.opening_bindings))
            if pads
            else None
        )

        def clock() -> tuple[int, float] | None:
            identities = environment.episode_context.snapshot()
            if (
                self._closed
                or environment.episode_context is not self._episode_owner
                or environment.sim is not self._simulation_owner
                or tuple(row["episode"] for row in identities) != self._episodes
                or not all(row["initialized"] for row in identities)
            ):
                raise ValueError("Recorded contact clock belongs to another episode.")
            return bridge.clock.step_index, bridge.clock.now()

        self.timeline = ContactTimeline(
            sampler,
            cfg.retention,
            clock=clock,
            control_parts=tuple(pads),
            opening_reader=opening,
        )
        self._generations = tuple(
            row["episode_generation"]
            for row in self.timeline.snapshot()["environments"]
        )
        self._closed = False
        sensor.add_sampling_observer(cfg.observer_name, self.timeline)

    def _check_recorded_owner(self) -> None:
        self._check_episode()
        if (
            self._env.episode_context is not self._episode_owner
            or self._env.sim is not self._simulation_owner
        ):
            raise ValueError("Recorded contacts belong to another environment owner.")
        if self._observations is not None:
            self._observations.validate_history_owner(self._episode_owner)

    @property
    def identity(self) -> dict[str, Any]:
        """Return owned program/native identities for an explicit evidence binding."""
        from ..invocation_policy import task_program_fingerprint

        return dict(
            sensor_uid=self._cfg.sensor_uid,
            program_fingerprint=task_program_fingerprint(self._bridge.program),
            step_dt=self._cfg.retention.step_dt,
            object_native_uids={
                key: value[0] for key, value in self._rigid_owners.items()
            },
            object_actors=deepcopy(self._object_actors),
            object_body_types=dict(self._body_types),
            pad_actors=deepcopy(self._pad_actors),
            topology_revision=self._topology,
            timeline_episode_generations=list(self._generations),
        )

    def close(self) -> None:
        """Detach only this instance's observer; repeated close is harmless."""
        if self._closed:
            return
        self._sensor.remove_sampling_observer(
            self._cfg.observer_name, expected=self.timeline
        )
        self._closed = True

    def capture(
        self, object_id: str, reference_id: str, *, start_frame: int, end_frame: int
    ) -> dict[str, Any]:
        """Copy one exact single-row contact/pose window without qualification.

        Args:
            object_id: Registered source rigid object.
            reference_id: Registered reference rigid object.
            start_frame: Explicit inclusive seeded start slot.
            end_frame: Explicit inclusive completed end slot.

        Returns:
            Owned raw timeline, actor inventory and pose acquisition channels.
            The caller owns window selection and normative decisions.
        """
        import torch
        from embodichain.lab.sim.evidence import (
            PhysicalEvidenceFrame,
            PhysicalEvidenceBatch,
        )
        from ..invocation_policy import task_program_fingerprint

        self._check_recorded_owner()
        self._port.validate_observation_scope(self._bridge.clock)
        initial_clock = (self._bridge.clock.step_index, self._bridge.clock.now())
        if self._closed or len(self._episodes) != 1:
            raise ValueError(
                "Source sampled-contact capture requires one active native row."
            )
        if (
            object_id == reference_id
            or type(start_frame) is not int
            or type(end_frame) is not int
            or not 0 <= start_frame < end_frame
        ):
            raise ValueError(
                "Recorded contact capture requires distinct objects and an exact window."
            )
        if (
            self._env.sim.get_sensor(self._cfg.sensor_uid) is not self._sensor
            or getattr(self._env, "sensors", {}).get(self._cfg.sensor_uid)
            is not self._sensor
            or self._env.robot is not self._robot
        ):
            raise ValueError("Recorded contact native owners changed.")
        cfg = self._sensor.cfg
        current_scope = (
            tuple(cfg.rigid_uid_list),
            tuple(
                (item.articulation_uid, tuple(item.link_name_list))
                for item in cfg.articulation_cfg_list
            ),
            cfg.max_contacts_per_env,
        )
        if (
            cfg.filter_need_both_actor is not False
            or current_scope != self._sensor_scope
        ):
            raise ValueError("Recorded contact sensor scope changed after binding.")
        stamp = self.timeline.sampler.source_stamp()
        if stamp.errors or stamp.topology_revision != self._topology:
            raise ValueError("Recorded contact actor topology changed.")
        snapshot = self.timeline.snapshot()
        if (
            tuple(row["episode_generation"] for row in snapshot["environments"])
            != self._generations
        ):
            raise ValueError("Recorded contacts were reset after episode binding.")
        poses, acquisitions = {}, {}
        for entity in (object_id, reference_id):
            uid, _ = self._native(entity, articulated=False)
            rows, metadata = _recorded_pose_window(
                self._observations,
                uid,
                [start_frame],
                [end_frame],
                episode_ids=self._episodes,
            )
            poses[entity] = rows[0].detach().to(device="cpu", dtype=torch.float64)
            acquisitions[entity] = metadata
        self._capture_revision += 1
        timestamp = float(self._env.sim.simulation_time)
        ids = torch.tensor([0], dtype=torch.long)
        frame = PhysicalEvidenceFrame._from_owned_batches(
            timestamp,
            self._capture_revision,
            ids,
            {
                "rigid.pose_window:"
                + entity: PhysicalEvidenceBatch(
                    "rigid.pose_window:" + entity,
                    pose.unsqueeze(0),
                    torch.ones(1, dtype=torch.bool),
                    (None,),
                    timestamp,
                    ids,
                    self._capture_revision,
                )
                for entity, pose in poses.items()
            },
        )
        self._check_recorded_owner()
        for entity, (_, native) in self._rigid_owners.items():
            self._native(entity, articulated=False)
            if native.body_type != self._body_types[entity]:
                raise ValueError("Recorded rigid body type changed during acquisition.")
        self._port.validate_observation_scope(self._bridge.clock)
        if initial_clock != (self._bridge.clock.step_index, self._bridge.clock.now()):
            raise ValueError("Recorded contact clock advanced during capture.")
        final_stamp = self.timeline.sampler.source_stamp()
        if final_stamp.errors or final_stamp != stamp:
            raise ValueError("Recorded native topology changed during acquisition.")
        return dict(
            schema_version="simulation.recorded_contacts/v1",
            sensor_uid=self._cfg.sensor_uid,
            program_fingerprint=task_program_fingerprint(self._bridge.program),
            episode_ids=list(self._episodes),
            env_ids=[0],
            window_start_frames=[start_frame],
            boundary_frames=[end_frame],
            step_dt=self._env.step_dt,
            object_id=object_id,
            reference_id=reference_id,
            object_native_uids={
                key: value[0] for key, value in self._rigid_owners.items()
            },
            object_actors=deepcopy(self._object_actors),
            object_body_types=dict(self._body_types),
            pad_actors=deepcopy(self._pad_actors),
            topology_revision=self._topology,
            timeline_episode_generations=list(self._generations),
            timeline=snapshot,
            recorded_acquisition=acquisitions,
            pose_frame=dict(
                timestamp=frame.timestamp,
                observation_revision=frame.observation_revision,
                env_ids=[0],
                clock_domain="simulation_time",
                revision_kind="recorded_contact_pose_acquisition",
                batches={
                    name: dict(
                        evidence_id=batch.evidence_id,
                        value_dtype=str(batch.values.dtype),
                        values=batch.values.tolist(),
                        valid=batch.valid.tolist(),
                        acquisition_errors=list(batch.acquisition_errors),
                        timestamp=batch.timestamp,
                        observation_revision=batch.observation_revision,
                        env_ids=[0],
                    )
                    for name, batch in frame.batches.items()
                },
            ),
        )

    def capture_timeline(
        self,
        object_id: str,
        reference_id: str,
        *,
        boundary_frame: int,
    ) -> dict[str, Any]:
        """Copy retained native contact history without reading unneeded poses.

        Args:
            object_id: Explicit registered rigid payload.
            reference_id: Explicit registered rigid reference.
            boundary_frame: Exact completed boundary associated with the report.

        Returns:
            Owned actor/episode/program identities and raw packed history. No
            event, acquisition point or release success is selected here.
        """
        self._check_recorded_owner()
        self._port.validate_observation_scope(self._bridge.clock)
        clock = (self._bridge.clock.step_index, self._bridge.clock.now())
        if (
            self._closed
            or self._observations is None
            or len(self._episodes) != 1
            or object_id == reference_id
            or type(boundary_frame) is not int
            or not 0
            < boundary_frame
            <= self._observations.read_history_step_counts(episode_ids=self._episodes)[
                0
            ]
        ):
            raise ValueError(
                "Contact history requires its actual single-row completed boundary."
            )
        if (
            self._env.sim.get_sensor(self._cfg.sensor_uid) is not self._sensor
            or getattr(self._env, "sensors", {}).get(self._cfg.sensor_uid)
            is not self._sensor
            or self._env.robot is not self._robot
        ):
            raise ValueError("Recorded contact native owners changed.")
        cfg = self._sensor.cfg
        scope = (
            tuple(cfg.rigid_uid_list),
            tuple(
                (item.articulation_uid, tuple(item.link_name_list))
                for item in cfg.articulation_cfg_list
            ),
            cfg.max_contacts_per_env,
        )
        if cfg.filter_need_both_actor is not False or scope != self._sensor_scope:
            raise ValueError("Recorded contact sensor scope changed after binding.")
        self._native(object_id, articulated=False)
        self._native(reference_id, articulated=False)
        stamp = self.timeline.sampler.source_stamp()
        if stamp.errors or stamp.topology_revision != self._topology:
            raise ValueError("Recorded contact actor topology changed.")
        snapshot = self.timeline.snapshot()
        if (
            tuple(row["episode_generation"] for row in snapshot["environments"])
            != self._generations
        ):
            raise ValueError("Recorded contacts were reset after episode binding.")
        self._check_recorded_owner()
        for entity, (_, native) in self._rigid_owners.items():
            self._native(entity, articulated=False)
            if native.body_type != self._body_types[entity]:
                raise ValueError("Recorded rigid body type changed during acquisition.")
        self._port.validate_observation_scope(self._bridge.clock)
        if (
            clock != (self._bridge.clock.step_index, self._bridge.clock.now())
            or self.timeline.sampler.source_stamp() != stamp
        ):
            raise ValueError("Recorded contact clock or source changed during capture.")
        return dict(
            schema_version="simulation.recorded_contacts/v1",
            **self.identity,
            object_id=object_id,
            reference_id=reference_id,
            episode_ids=list(self._episodes),
            env_ids=[0],
            boundary_frames=[boundary_frame],
            timeline=snapshot,
        )
