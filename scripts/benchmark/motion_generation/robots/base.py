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
    """Build an official robot and bind one explicit motion resource.

    Args:
        resolution: Component configuration and bindings resolved for this run.
        resource_id: Selected resource, required when several provide motion.
    """

    def __init__(
        self,
        resolution: "EmbodimentResolution",
        resource_id: str | None = None,
    ) -> None:
        from embodichain.lab.gym.utils.gym_utils import _robot_cfg_from_dict
        from ..config import RobotSpecCfg

        candidates = {
            name: endpoints
            for name, endpoints in resolution.endpoint_bindings.items()
            if "motion" in endpoints
        }
        if resource_id is None:
            if len(candidates) != 1:
                raise ValueError(
                    "embodiment.resource_id must select one motion resource; "
                    f"available resources: {sorted(candidates)}."
                )
            resource_id = next(iter(candidates))
        if resource_id not in candidates:
            raise ValueError(f"Unknown motion resource {resource_id!r}.")
        if resolution.simulation is None:
            raise ValueError("The embodiment has no resolved simulation configuration.")
        super().__init__(
            RobotSpecCfg(id=resolution.embodiment_id, provider="embodiment")
        )
        self.control_part = candidates[resource_id]["motion"]
        self._cfg = _robot_cfg_from_dict(deepcopy(dict(resolution.simulation)))
        missing = self._cfg.validate()
        if missing:
            raise ValueError(f"Embodiment robot has missing required fields: {missing}")
        parts = self._cfg.control_parts or {}
        if self.control_part not in parts:
            raise ValueError(
                f"Unknown embodiment motion control part {self.control_part!r}."
            )
        solvers = self._cfg.solver_cfg
        if solvers is None or (
            isinstance(solvers, Mapping) and self.control_part not in solvers
        ):
            raise ValueError(
                f"No solver configured for control part {self.control_part!r}."
            )

        self.gripper_config: dict[str, object] | None = None
        grasp_part = candidates[resource_id].get("grasp")
        if grasp_part is not None:
            if grasp_part not in parts or grasp_part == self.control_part:
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
            self.gripper_config = {
                "control_part": grasp_part,
                "open_qpos": list(values[0]),
                "grasp_qpos": list(values[1]),
            }

    def build_cfg(self) -> "RobotCfg":
        """Return a fresh component-owned robot configuration, including its TCP."""
        return deepcopy(self._cfg)

    def configure_track(self, track: "TrackCfg") -> None:
        """Supply component-owned gripper commands without accepting conflicts.

        Args:
            track: Effective track, including any selected domain overrides.
        """
        if track.scenario != "atomic_task" or self.gripper_config is None:
            return
        declared = track.config.get("gripper")
        if declared is not None and declared != self.gripper_config:
            raise ValueError("track.gripper conflicts with the selected embodiment.")
        track.config["gripper"] = deepcopy(self.gripper_config)
