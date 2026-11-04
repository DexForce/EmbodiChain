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

"""Robot-provider contract for planner benchmark suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.sim.objects import Robot

    from ..config import RobotSpecCfg, TrackCfg
    from ..embodiments import EmbodimentResolution

__all__ = ["RobotProvider", "EmbodimentRobotProvider"]


class RobotProvider(ABC):
    """Build one benchmark embodiment behind a stable suite identifier."""

    control_part: str = "arm"

    def __init__(self, spec: "RobotSpecCfg") -> None:
        self.spec = spec

    @abstractmethod
    def build_cfg(self) -> "RobotCfg":
        """Build the robot configuration without mutating a simulation."""

    def add_robot(self, simulation: "SimulationManager") -> "Robot":
        """Add the configured robot to a simulation."""
        return simulation.add_robot(cfg=self.build_cfg())


class EmbodimentRobotProvider(RobotProvider):
    """Build an official robot and bind explicit motion resources.

    Args:
        resolution: Component configuration and bindings resolved for this run.
        resource_id: Select one resource for a single-arm benchmark.
        resource_bindings: Map Atomic slots to distinct resources for coordination.
    """

    def __init__(
        self,
        resolution: "EmbodimentResolution",
        resource_id: str | None = None,
        *,
        resource_bindings: Mapping[str, str] | None = None,
    ) -> None:
        from embodichain.lab.gym.utils.gym_utils import _robot_cfg_from_dict
        from ..config import RobotSpecCfg

        candidates = {
            name: endpoints
            for name, endpoints in resolution.endpoint_bindings.items()
            if "motion" in endpoints
        }
        bindings = dict(resource_bindings or {})
        if bindings:
            if resource_id is not None:
                raise ValueError("Specify resource_id or resource_bindings, not both.")
            if any(
                not isinstance(slot, str)
                or not slot
                or not isinstance(name, str)
                or not name
                for slot, name in bindings.items()
            ) or len(set(bindings.values())) != len(bindings):
                raise ValueError(
                    "resource_bindings need named slots and distinct resources."
                )
        else:
            if resource_id is None:
                if len(candidates) != 1:
                    raise ValueError(
                        "embodiment.resource_id must select one motion resource, or "
                        "resource_bindings must select coordinated resources; "
                        f"available resources: {sorted(candidates)}."
                    )
                resource_id = next(iter(candidates))
            bindings = {"primary": resource_id}
        if unknown := set(bindings.values()) - candidates.keys():
            raise ValueError(f"Unknown motion resources {sorted(unknown)!r}.")
        bindings = {
            slot: bindings[slot]
            for slot in sorted(
                bindings, key=lambda slot: (slot != "source", slot != "primary", slot)
            )
        }
        self.resource_bindings = bindings
        self.slot_control_parts = {
            slot: dict(candidates[name]) for slot, name in bindings.items()
        }
        selected = list(bindings.values())
        motion_parts = [candidates[name]["motion"] for name in selected]
        if len(set(motion_parts)) != len(motion_parts):
            raise ValueError(
                "Coordinated resources must have distinct motion control parts."
            )
        if resolution.simulation is None:
            raise ValueError("The embodiment has no resolved simulation configuration.")
        super().__init__(
            RobotSpecCfg(id=resolution.embodiment_id, provider="embodiment")
        )
        self.control_part = motion_parts[0]
        self._cfg = _robot_cfg_from_dict(deepcopy(dict(resolution.simulation)))
        missing = self._cfg.validate()
        if missing:
            raise ValueError(f"Embodiment robot has missing required fields: {missing}")
        parts = self._cfg.control_parts or {}
        for part in motion_parts:
            if part not in parts:
                raise ValueError(f"Unknown embodiment motion control part {part!r}.")
            solvers = self._cfg.solver_cfg
            if solvers is None or (
                isinstance(solvers, Mapping) and part not in solvers
            ):
                raise ValueError(f"No solver configured for control part {part!r}.")

        self.gripper_configs: list[dict[str, object]] = []
        for name in selected:
            config = self._gripper_commands(resolution, name, candidates[name], parts)
            if config is not None:
                geometry = resolution.runtime_services.get("hand_geometries", {}).get(
                    config["control_part"]
                )
                if geometry is not None:
                    config["finger_links"] = list(geometry["pad_link_names"])
                if any(
                    item["control_part"] == config["control_part"]
                    for item in self.gripper_configs
                ):
                    raise ValueError(
                        "Coordinated resources must have distinct grasp control parts."
                    )
                self.gripper_configs.append(config)
        self.gripper_config = (
            self.gripper_configs[0]
            if len(selected) == 1 and self.gripper_configs
            else None
        )

    def _gripper_commands(
        self,
        resolution: "EmbodimentResolution",
        resource_id: str,
        endpoints: Mapping[str, str],
        parts: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Resolve a selected resource's calibrated command preset."""
        grasp_part = endpoints.get("grasp")
        if grasp_part is not None:
            if grasp_part not in parts or grasp_part == endpoints["motion"]:
                raise ValueError(
                    f"Invalid embodiment grasp control part {grasp_part!r}."
                )
            profile = resolution.skill_profile or {}
            resource = next(
                item
                for item in profile.get("resources", [])
                if item["resource_id"] == resource_id
            )
            endpoint = next(
                item for item in resource["endpoints"] if item["endpoint_id"] == "grasp"
            )
            presets = [
                item
                for item in profile.get("command_presets", [])
                if item.get("preset_id") == endpoint.get("command_preset")
            ]
            if len(presets) != 1 or presets[0].get("control_part") != grasp_part:
                raise ValueError(
                    "The grasp endpoint needs one matching command preset."
                )
            commands = presets[0].get("commands", {})
            values = [commands.get(name) for name in ("open", "grasp")]
            if any(
                not isinstance(value, list)
                or not value
                or any(
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(number)
                    for number in value
                )
                for value in values
            ) or len(values[0]) != len(values[1]):
                raise ValueError(
                    "Gripper open/grasp commands need equal finite vectors."
                )
            return {
                "control_part": grasp_part,
                "open_qpos": list(values[0]),
                "grasp_qpos": list(values[1]),
            }

        return None

    def build_cfg(self) -> "RobotCfg":
        """Return a fresh component-owned robot configuration, including its TCP."""
        return deepcopy(self._cfg)

    def configure_track(self, track: "TrackCfg") -> None:
        """Supply component-owned gripper commands without accepting conflicts.

        Args:
            track: Effective track, including any selected domain overrides.
        """
        if track.scenario != "atomic_skill":
            if len(self.resource_bindings) > 1:
                raise ValueError(
                    "Coordinated resource_bindings require an atomic_skill track."
                )
            return
        if len(self.resource_bindings) > 1:
            # Domains may exchange roles, but must keep the selected endpoint pairs.
            declared_parts = track.config.get("control_parts")
            if declared_parts is not None:
                if (
                    not isinstance(declared_parts, Mapping)
                    or set(declared_parts) != set(self.slot_control_parts)
                    or any(
                        value not in self.slot_control_parts.values()
                        for value in declared_parts.values()
                    )
                    or len(
                        {
                            tuple(sorted(value.items()))
                            for value in declared_parts.values()
                        }
                    )
                    != len(self.slot_control_parts)
                ):
                    raise ValueError(
                        "track.control_parts conflict with selected embodiment resources."
                    )
            else:
                track.config["control_parts"] = deepcopy(self.slot_control_parts)
            key, other, commands = "grippers", "gripper", self.gripper_configs
        else:
            key, other, commands = "gripper", "grippers", self.gripper_config
        if commands is None or commands == []:
            return
        if other in track.config:
            raise ValueError(f"track.{key} conflicts with the selected embodiment.")
        declared = track.config.get(key)
        expected = commands if isinstance(commands, list) else [commands]
        actual = declared if isinstance(declared, list) else [declared]
        if declared is None:
            actual = [{} for _ in expected]
        if len(actual) != len(expected):
            raise ValueError(f"track.{key} conflicts with the selected embodiment.")
        configured = []
        for preset, entry in zip(expected, actual, strict=True):
            if not isinstance(entry, Mapping) or any(
                field in entry and entry[field] != value
                for field, value in preset.items()
            ):
                raise ValueError(f"track.{key} conflicts with the selected embodiment.")
            contacts = {
                field: value for field, value in entry.items() if field not in preset
            }
            if set(contacts) - {"finger_links", "contact_distance_threshold_m"}:
                raise ValueError("Unknown embodiment gripper contact settings.")
            configured.append({**deepcopy(preset), **deepcopy(contacts)})
        track.config[key] = configured if isinstance(commands, list) else configured[0]
