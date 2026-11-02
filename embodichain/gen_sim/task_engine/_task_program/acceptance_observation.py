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

"""Bounded native contact observations on the existing Gym physics clock."""

from __future__ import annotations

from typing import Any

from embodichain.lab.sim.sensors.contact_sensor import (
    ArticulationContactFilterCfg,
    ContactSensor,
    ContactSensorCfg,
)

__all__: list[str] = []
SENSOR_UID = "gen_sim_task_spec_contacts"


class AcceptanceContactSensor(ContactSensor):
    """Record control-boundary samples without issuing commands or stepping."""

    @property
    def requires_substep_update(self) -> bool:
        return hasattr(self, "acceptance_reader")

    def begin_control_step(self) -> None:
        super().begin_control_step()
        self.acceptance_reader.begin()

    def update_physics_step(self, dt: float) -> None:
        revision = self.acceptance_reader.source_revision()
        super().update_physics_step(dt)
        self.acceptance_reader.capture(dt, revision)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if hasattr(self, "acceptance_reader"):
            self.acceptance_reader.reset()


class ContactTimeline:
    """Keep exact actor identities and raw samples, with explicit loss reporting."""

    def __init__(
        self,
        env: Any,
        sensor: Any,
        objects: dict,
        pads: dict,
        opening: Any,
        body_types: dict,
    ) -> None:
        self.env, self.sensor = env, sensor
        self.objects, self.pads = objects, pads
        self.opening = opening
        self.body_types = body_types
        self.rows: dict[int, Any] = {}
        self.errors: list[str] = []
        self.elapsed = 0.0
        self.index: int | None = None
        self.generation = 0
        self.topology = int(env.sim.spawn_result.topology_revision)
        self.contact_rows = 0
        self.substeps: list[dict] = []

    def invalidate(self, reason: str) -> None:
        if reason not in self.errors and len(self.errors) < 16:
            self.errors.append(reason)

    def source_revision(self) -> int | None:
        from .support_evidence import _native_contact_revision

        try:
            if int(self.env.sim.spawn_result.topology_revision) != self.topology:
                raise ValueError("Acceptance native actor topology changed.")
            return _native_contact_revision(self.env.sim)
        except (AttributeError, ValueError, TypeError, RuntimeError) as error:
            self.invalidate(str(error))
            return None

    def reset(self) -> None:
        self.rows.clear()
        self.errors.clear()
        self.index = None
        self.generation += 1
        self.contact_rows = 0
        self.substeps.clear()

    def begin(self) -> None:
        bridge = getattr(self.env, "_active_task_program_bridge", None)
        if bridge is None:
            self.index = None
            return
        self.index = bridge._clock.step_index + 1
        self.elapsed = 0.0
        self.substeps = []
        if abs(bridge._clock.now() - (self.index - 1) * self.env.step_dt) > 1e-8:
            self.invalidate("Acceptance control clock differs from the episode.")
            self.index = None

    def capture(self, dt: float, native_revision: int | None) -> None:
        if self.index is None:
            return
        self.elapsed += dt
        if self.elapsed > self.env.step_dt + 1e-8:
            self.invalidate("Control interval differs from the environment clock.")
            return
        if len(self.rows) >= 30000 or self.contact_rows >= 1_000_000:
            self.invalidate("Contact timeline capacity exceeded.")
            return
        if native_revision is None or self.source_revision() != native_revision:
            self.invalidate("Acceptance contact source changed during fetch.")
            return
        data = self.sensor.get_data()
        if self.source_revision() != native_revision:
            self.invalidate("Acceptance contact source changed during data access.")
            return
        batches = []
        for row in range(self.env.num_envs):
            valid = data["is_valid"][row]
            self.contact_rows += int(valid.sum())
            batches.append(
                {
                    name: data[key][row, valid].detach().cpu().tolist()
                    for name, key in (
                        ("pairs", "user_ids"),
                        ("positions", "position"),
                        ("normals", "normal"),
                        ("impulses", "impulse"),
                    )
                }
            )
        self.substeps.append(
            {
                "offset": self.elapsed,
                "dt": dt,
                "native_revision": native_revision,
                "dropped_contacts": int(self.sensor.dropped_contacts),
                "rows": batches,
            }
        )
        if self.elapsed + 1e-9 < self.env.step_dt:
            return
        self.rows[self.index] = {
            "step": self.index,
            "time": self.index * self.env.step_dt,
            "dropped_contacts": int(self.sensor.dropped_contacts),
            "native_revision": native_revision,
            "topology_revision": self.topology,
            "rows": batches,
            "substeps": self.substeps,
            "hand_opening": {
                part: [self.opening(part, row) for row in range(self.env.num_envs)]
                for part in self.pads
            },
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "gen_sim.acceptance-contact-timeline/v2",
            "scope": "native_substep_samples_not_continuous_contact_proof",
            "step_dt": float(self.env.step_dt),
            "episode_generation": self.generation,
            "topology_revision": self.topology,
            "retained_contact_rows": self.contact_rows,
            "object_actors": {
                key: value.tolist() for key, value in self.objects.items()
            },
            "pad_actors": {key: value.tolist() for key, value in self.pads.items()},
            "object_body_types": dict(self.body_types),
            "errors": list(self.errors),
            "samples": list(self.rows.values()),
        }


def install_contact_timeline(
    env: Any, deployment: Any, graph: dict
) -> ContactTimeline | None:
    """Observe E5 compositions without changing their routes or physical policy."""
    if not any(group["task_type"] == "E5" for group in graph["task_groups"]):
        return None
    target = getattr(env, "unwrapped", env)
    sim, robot = target.sim, target.robot
    opening = _opening_reader(robot, deployment)
    bindings = deployment.integration.registration.scene_binding.rigid_objects
    pads = {
        part: tuple(
            name
            for name in robot.link_names
            if name.startswith(part.removesuffix("_eef") + "_")
            and name.endswith("_finger_pad")
        )
        for part in ("left_eef", "right_eef")
    }
    if any(len(names) != 2 for names in pads.values()):
        raise ValueError("E5 observations require exact paired native finger pads.")
    previous = sim.SUPPORTED_SENSOR_TYPES
    sim.SUPPORTED_SENSOR_TYPES = {
        **previous,
        "GenSimAcceptanceContact": AcceptanceContactSensor,
    }
    try:
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid=SENSOR_UID,
                sensor_type="GenSimAcceptanceContact",
                rigid_uid_list=[value.simulation_uid for value in bindings],
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(
                        articulation_uid=robot.uid,
                        link_name_list=[
                            name for names in pads.values() for name in names
                        ],
                    )
                ],
                max_contacts_per_env=1024,
                filter_need_both_actor=False,
            )
        )
    finally:
        sim.SUPPORTED_SENSOR_TYPES = previous
    reader = ContactTimeline(
        target,
        sensor,
        {
            value.entity_id: sensor.get_actor_ids(value.simulation_uid)
            for value in bindings
        },
        {part: sensor.get_actor_ids(robot.uid, names) for part, names in pads.items()},
        opening,
        {
            value.entity_id: sim.get_rigid_object(value.simulation_uid).cfg.body_type
            for value in bindings
        },
    )
    sensor.acceptance_reader = reader
    target.sensors[SENSOR_UID] = sensor
    return reader


def _opening_reader(robot: Any, deployment: Any) -> Any:
    from .grip_evidence import NativeGripEvidenceFactory, _open_release_reader

    factory = deployment.integration.registration.control_part_evidence_factory
    if not isinstance(factory, NativeGripEvidenceFactory):
        raise ValueError("Coordinated release needs the exact native grip declaration.")
    return _open_release_reader(robot, factory.source)
