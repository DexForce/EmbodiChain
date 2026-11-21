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

"""Target-specific, simultaneous finger contact evidence for physical scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from ..evaluation import TOLERANCES

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Robot
    from ..models import BenchmarkCase
    from .atomic_objects import AtomicArticulationHandle, AtomicObjectHandle

__all__: list[str] = []


def contact_configuration(
    track_config: Mapping[str, object],
) -> dict[str, tuple[tuple[str, ...], float]]:
    """Validate explicit finger bodies and per-gripper distance thresholds."""
    entries = track_config.get("grippers")
    if entries is None:
        entry = track_config.get("gripper")
        entries = [] if entry is None else [entry]
    result = {}
    for entry in entries:
        links = entry.get("finger_links", [])
        if not isinstance(links, (list, tuple)) or any(
            not isinstance(link, str) or not link.strip() for link in links
        ):
            raise ValueError("gripper.finger_links must be a list of link names.")
        if links and (len(links) < 2 or len(set(links)) != len(links)):
            raise ValueError("gripper.finger_links requires at least two unique links.")
        threshold = entry.get(
            "contact_distance_threshold_m", TOLERANCES["grasp_contact_distance_m"]
        )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("gripper.contact_distance_threshold_m must be positive.")
        result[str(entry["control_part"])] = (tuple(links), float(threshold))
    return result


def minimum_target_contact_distances(
    data: Mapping[str, torch.Tensor],
    finger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    dropped: torch.Tensor,
) -> torch.Tensor:
    """Reduce one query to (environment, finger) signed distances on device.

    Work and temporary storage are O(B * C * F), where the query bounds C.
    No contacts are accumulated across time or transferred to the host.
    """
    actors = data["user_ids"]
    first = actors[..., 0, None]
    second = actors[..., 1, None]
    fingers = finger_ids[:, None, :]
    target = target_ids[:, None, :]
    matched = data["is_valid"][..., None] & (
        ((first == fingers) & (second == target))
        | ((second == fingers) & (first == target))
    )
    distance = data["distance"][..., None]
    invalid = (matched & ~torch.isfinite(distance)).any(dim=1) | dropped[:, None]
    minimum = torch.where(matched, distance, torch.inf).amin(dim=1)
    return minimum.masked_fill(invalid, torch.nan)


class GraspContacts:
    """Bind contact actor identities once and retain compact physics samples."""

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        track_config: Mapping[str, object],
        case: BenchmarkCase,
        objects: Mapping[str, AtomicObjectHandle],
        parts: Sequence[str],
        articulations: Mapping[str, AtomicArticulationHandle] | None = None,
    ) -> None:
        from embodichain.lab.sim.sensors import (
            ArticulationContactFilterCfg,
            ContactSensor,
            ContactSensorCfg,
        )

        configured = contact_configuration(track_config)
        articulations = {} if articulations is None else articulations
        entities = {**objects, **articulations}
        self.links = {part: configured.get(part, ((), 0.0))[0] for part in parts}
        self.thresholds = {
            part: configured.get(part, ((), TOLERANCES["grasp_contact_distance_m"]))[1]
            for part in parts
        }
        self.targets = {part: case.object_id for part in parts if case.object_id}
        roles = case.case_parameters.get("control_parts", {})
        if case.skill_id == "coordinated_placement" and "support" in roles:
            self.targets[str(roles["support"]["grasp"])] = str(
                case.case_parameters["support_object_id"]
            )
        self.reasons: dict[str, str] = {}
        target_link = case.case_parameters.get("target_link")
        self.target_links = {
            part: (
                (str(target_link),) if target_link and target in articulations else None
            )
            for part, target in self.targets.items()
        }
        self.sensor = None
        self.bindings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.empty = {
            part: torch.full(
                (case.batch_size, max(len(self.links[part]), 2)),
                torch.nan,
                device=robot.device,
            )
            for part in self.targets
        }
        self.dropped = torch.zeros(
            case.batch_size, dtype=torch.bool, device=robot.device
        )
        capacity = dict(track_config.get("physics", {})).get(
            "grasp_contact_max_contacts_per_env", 256
        )
        if type(capacity) is not int or capacity < 1:
            raise ValueError("grasp_contact_max_contacts_per_env must be positive.")
        for part in self.targets:
            if not self.links[part]:
                self.reasons[part] = "finger_links_not_configured"
            elif not set(self.links[part]) <= set(robot.link_names):
                self.reasons[part] = "finger_links_not_found"
            elif self.targets[part] not in entities:
                self.reasons[part] = "target_object_not_found"
            elif (
                self.targets[part] in articulations and self.target_links[part] is None
            ):
                self.reasons[part] = "target_link_not_configured"
            elif not simulation.physics.supports_contact_sensor:
                self.reasons[part] = "backend_unsupported"
        available = set(self.targets) - set(self.reasons)
        if not available:
            return
        self.sensor = ContactSensor(
            ContactSensorCfg(
                uid="benchmark_grasp_contacts",
                rigid_uid_list=list(
                    dict.fromkeys(
                        objects[self.targets[part]].entity.uid
                        for part in sorted(available)
                        if self.targets[part] in objects
                    )
                ),
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(
                        articulation_uid=robot.uid,
                        link_name_list=list(
                            dict.fromkeys(
                                link
                                for part in sorted(available)
                                for link in self.links[part]
                            )
                        ),
                    )
                ]
                + [
                    ArticulationContactFilterCfg(
                        articulation_uid=entities[self.targets[part]].entity.uid,
                        link_name_list=list(self.target_links[part]),
                    )
                    for part in sorted(available)
                    if self.targets[part] in articulations
                ],
                filter_need_both_actor=True,
                max_contacts_per_env=capacity,
            ),
            device=robot.device,
            owner=simulation,
        )
        if not self.sensor.contact_capabilities.geometry:
            self.reasons.update(
                {part: "contact_geometry_unavailable" for part in available}
            )
            self.sensor = None
            return
        for part in sorted(available):
            try:
                fingers = self.sensor.get_actor_ids(robot.uid, self.links[part])
                target = self.sensor.get_actor_ids(
                    entities[self.targets[part]].entity.uid, self.target_links[part]
                )
            except ValueError:
                self.reasons[part] = "contact_actor_not_found"
                continue
            self.bindings[part] = (fingers, target)

    def sample(self) -> dict[str, torch.Tensor]:
        """Read one instantaneous sample; unresolved bindings remain NaN."""
        result = dict(self.empty)
        if self.sensor is not None and self.bindings:
            self.sensor.update()
            data = self.sensor.get_data()
            dropped = self.sensor.dropped_contacts_mask
            self.dropped |= dropped
            for part, (fingers, target) in self.bindings.items():
                result[part] = minimum_target_contact_distances(
                    data, fingers, target, dropped
                )
        return result

    def metadata(self) -> dict[str, object]:
        """Describe the scoring contract and missing evidence in trial artifacts."""
        return {
            "source": "ContactSensor",
            "sampling": "instantaneous_after_physics_update",
            "distance": "signed_separation_m_strictly_less_than_threshold",
            "finger_links": {part: list(links) for part, links in self.links.items()},
            "target_objects": dict(self.targets),
            "target_links": {
                part: None if links is None else list(links)
                for part, links in self.target_links.items()
            },
            "distance_thresholds_m": dict(self.thresholds),
            "unavailable": dict(self.reasons),
            "truncated_env_ids": self.dropped.nonzero().flatten().cpu().tolist(),
        }
