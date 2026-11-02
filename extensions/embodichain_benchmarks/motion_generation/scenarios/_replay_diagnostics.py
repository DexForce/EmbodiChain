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

"""Bounded opt-in measurements for physical replay diagnosis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions.plans import CompiledTrajectory
    from embodichain.lab.sim.objects import Robot
    from ..models import BenchmarkCase
    from .atomic_objects import AtomicObjectHandle

__all__: list[str] = []


def resolve_diagnostics_config(physics: Mapping[str, object]) -> dict[str, object]:
    """Validate opt-in collection without changing any scoring channels."""
    value = physics.get("replay_diagnostics")
    if value is None:
        return {"enabled": False}
    if not isinstance(value, Mapping):
        raise TypeError("physics.replay_diagnostics must be a mapping.")
    unknown = set(value) - {"enabled", "samples_per_segment", "max_contacts_per_env"}
    if unknown:
        raise ValueError(f"Unknown replay_diagnostics fields: {sorted(unknown)}.")
    enabled = value.get("enabled", True)
    if type(enabled) is not bool:
        raise TypeError("replay_diagnostics.enabled must be a bool.")
    samples = value.get("samples_per_segment", 8)
    capacity = value.get("max_contacts_per_env", 64)
    if type(samples) is not int or not 2 <= samples <= 64:
        raise ValueError("samples_per_segment must be an integer in [2, 64].")
    if type(capacity) is not int or not 1 <= capacity <= 256:
        raise ValueError("max_contacts_per_env must be an integer in [1, 256].")
    return {
        "enabled": enabled,
        "samples_per_segment": samples,
        "max_contacts_per_env": capacity,
    }


class ReplayDiagnostics:
    """Collect sparse stage snapshots; never modify physics or task scoring."""

    def __init__(
        self,
        config: Mapping[str, object],
        compiled: CompiledTrajectory,
        case: BenchmarkCase,
        simulation: SimulationManager,
        robot: Robot,
        objects: Mapping[str, AtomicObjectHandle],
        control_part: str,
        gripper_parts: tuple[str, ...],
    ) -> None:
        self.config = dict(config)
        self.robot = robot
        self.simulation = simulation
        self.gripper_parts = gripper_parts
        self.arm_parts = tuple(
            dict.fromkeys(
                (control_part, *case.case_parameters.get("arm_waypoints", {}))
            )
        )
        object_ids = dict.fromkeys(
            (case.object_id, *case.case_parameters.get("scene_object_poses", {}))
        )
        self.objects = {
            name: objects[name]
            for name in object_ids
            if name is not None and name in objects
        }
        self.sample_indices: set[int] = set()
        self.segments: list[dict[str, object]] = []
        samples = int(config["samples_per_segment"])
        for action_index, plan in enumerate(compiled.action_plans):
            for segment in plan.segments:
                shifted = compiled.segment(action_index, segment.name)
                if shifted.stop <= shifted.start:
                    continue
                count = min(samples, shifted.stop - shifted.start)
                indices = sorted(
                    {
                        shifted.start
                        + round(
                            index
                            * (shifted.stop - shifted.start - 1)
                            / max(count - 1, 1)
                        )
                        for index in range(count)
                    }
                )
                self.sample_indices.update(indices)
                self.segments.append(
                    {
                        "action_index": action_index,
                        "skill_id": plan.skill_id,
                        "segment": shifted.name,
                        "start": shifted.start,
                        "stop": shifted.stop,
                        "sampled_waypoint_indices": indices,
                    }
                )
        self.snapshots: list[dict[str, object]] = []
        self.actors: dict[str, dict[str, object]] = {}
        self.sensor = None
        self.contact_metadata: dict[str, object] = {
            "available": False,
            "reason": "no_rigid_objects" if not self.objects else "backend_unsupported",
        }
        if self.objects and simulation.physics.supports_contact_sensor:
            from embodichain.lab.sim.sensors import ContactSensor, ContactSensorCfg

            self.sensor = ContactSensor(
                ContactSensorCfg(
                    uid="benchmark_replay_diagnostics",
                    rigid_uid_list=[
                        handle.entity.uid for handle in self.objects.values()
                    ],
                    filter_need_both_actor=False,
                    max_contacts_per_env=int(config["max_contacts_per_env"]),
                ),
                device=robot.device,
                owner=simulation,
            )
            capabilities = self.sensor.contact_capabilities
            self.contact_metadata = {
                "available": True,
                "source": "ContactSensor",
                "frame": "arena",
                "sampling": "instantaneous_at_selected_command_boundaries",
                "geometry": bool(capabilities.geometry),
                "impulse": bool(capabilities.impulse),
                "friction": bool(capabilities.friction),
                "max_contacts_per_env": int(config["max_contacts_per_env"]),
                "absence_is_not_proof_of_no_intermediate_contact": True,
            }

    def capture(
        self,
        *,
        waypoint_index: int,
        elapsed_s: float,
        command: torch.Tensor,
        phase: str,
    ) -> None:
        """Copy one actual state, preserving scene rows and explicit timing."""
        observed = self.robot.get_qpos()
        snapshot = {
            "waypoint_index": waypoint_index,
            "phase": phase,
            "elapsed_sim_time_s": elapsed_s,
            "command_qpos": command.detach().cpu().tolist(),
            "observed_qpos": observed.detach().cpu().tolist(),
            "objects": {
                name: handle.entity.get_local_pose(to_matrix=True)
                .detach()
                .cpu()
                .tolist()
                for name, handle in self.objects.items()
            },
            "tcp_poses": {
                part: self.robot.compute_fk(
                    qpos=self.robot.get_qpos(name=part), name=part, to_matrix=True
                )
                .detach()
                .cpu()
                .tolist()
                for part in self.arm_parts
            },
            "gripper_qpos": {
                part: self.robot.get_qpos(name=part).detach().cpu().tolist()
                for part in self.gripper_parts
            },
            "gripper_command_qpos": {
                part: command[:, list(self.robot.get_joint_ids(name=part))]
                .detach()
                .cpu()
                .tolist()
                for part in self.gripper_parts
            },
        }
        if self.sensor is not None:
            self.sensor.update()
            data = self.sensor.get_data().to("cpu")
            rows = []
            for env_index in range(observed.shape[0]):
                valid = data["is_valid"][env_index]
                values = {
                    key: data[key][env_index][valid].tolist()
                    for key in ("user_ids", "position", "normal", "distance", "impulse")
                }
                if not self.contact_metadata["impulse"]:
                    values["impulse"] = None
                rows.append(values)
                for pair in values["user_ids"]:
                    for actor_id in pair:
                        if actor_id < 0 or str(actor_id) in self.actors:
                            continue
                        actor = self.sensor.get_actor_info(actor_id)
                        self.actors[str(actor_id)] = {
                            "path": actor.path,
                            "link_name": actor.link_name,
                            "env_id": actor.env_id,
                        }
            snapshot["contacts"] = rows
            snapshot["dropped_contact_rows"] = self.sensor.dropped_contacts
        self.snapshots.append(snapshot)

    def finish(self) -> dict[str, object]:
        """Return JSON-native evidence for the measured trial artifact."""
        return {
            "schema_version": 1,
            "configuration": self.config,
            "pose_frame": "arena",
            "timing_scope": "included_in_execution_time_excluded_from_planner_time",
            "segments": self.segments,
            "contacts": self.contact_metadata,
            "contact_actors": self.actors,
            "samples": self.snapshots,
        }
