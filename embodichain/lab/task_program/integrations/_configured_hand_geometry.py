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

"""Closed hand-calibration declarations and deferred Lab configuration factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
    from .simulation.bindings import SimulationRobotSkillProfileBinding

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _RobotHandGeometryCfgFactory:
    """Freeze provider-free calibration and create a fresh config per adapter."""

    motion_part: str
    hand_part: str
    command_preset: str
    hand_root_link: str
    pad_link_names: tuple[str, str]
    open_qpos: tuple[float, ...]
    closed_qpos: tuple[float, ...]
    sample_count: int
    body_max_edge: float
    pad_max_edge: float
    max_subdivisions: int

    def __call__(self) -> RobotHandGeometryCfg:
        from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

        cfg = RobotHandGeometryCfg(
            motion_part=self.motion_part,
            hand_part=self.hand_part,
            hand_root_link=self.hand_root_link,
            pad_link_names=self.pad_link_names,
            open_qpos=self.open_qpos,
            closed_qpos=self.closed_qpos,
            sample_count=self.sample_count,
            body_max_edge=self.body_max_edge,
            pad_max_edge=self.pad_max_edge,
            max_subdivisions=self.max_subdivisions,
        )
        cfg.validate_values()
        return cfg


def _decode_hand_geometry(
    value: object,
    *,
    path: str,
    target_id: str,
    profile: SimulationRobotSkillProfileBinding,
) -> _RobotHandGeometryCfgFactory:
    """Resolve declared commands without importing a robot or sampling geometry."""
    # The owning configured decoder calls this after its helper definitions and
    # robot-profile decoding are complete; keep module initialization acyclic.
    from .configured import _identifier, _identifier_tuple, _integer, _mapping, _real
    from .simulation.bindings import ControlPartResourceBinding

    config = _mapping(
        value,
        path=path,
        required=frozenset(
            {
                "kind",
                "motion_part",
                "hand_part",
                "command_preset",
                "hand_root_link",
                "pad_link_names",
                "sample_count",
                "body_max_edge",
                "pad_max_edge",
                "max_subdivisions",
            }
        ),
    )
    if config["kind"] != "sampled_parallel_jaw":
        raise ValueError(f"{path}.kind must be 'sampled_parallel_jaw'.")
    names = {
        key: _identifier(config[key], path=f"{path}.{key}")
        for key in ("motion_part", "hand_part", "command_preset", "hand_root_link")
    }
    pads = _identifier_tuple(config["pad_link_names"], path=f"{path}.pad_link_names")
    if len(pads) != 2:
        raise ValueError(
            f"{path}.pad_link_names must contain two distinct explicit links."
        )
    presets = {preset.preset_id: preset for preset in profile.command_presets}
    preset = presets.get(names["command_preset"])
    if preset is None or preset.control_part != names["hand_part"]:
        raise ValueError(f"{path}.command_preset must declare commands for hand_part.")
    if not {"open", "grasp"} <= preset.commands.keys():
        raise ValueError(f"{path}.command_preset must provide open and grasp commands.")
    endpoints = [
        endpoint
        for resource in profile.resources
        if isinstance(resource, ControlPartResourceBinding)
        for endpoint in resource.endpoints
    ]
    if target_id != names["hand_part"] or not any(
        endpoint.control_part == target_id
        and endpoint.command_preset == preset.preset_id
        for endpoint in endpoints
    ):
        raise ValueError(
            f"{path} must match a declared hand endpoint target and its command preset."
        )
    if not any(endpoint.control_part == names["motion_part"] for endpoint in endpoints):
        raise ValueError(f"{path}.motion_part must match a declared profile endpoint.")
    factory = _RobotHandGeometryCfgFactory(
        **names,
        pad_link_names=(pads[0], pads[1]),
        open_qpos=preset.commands["open"],
        closed_qpos=preset.commands["grasp"],
        sample_count=_integer(
            config["sample_count"], path=f"{path}.sample_count", minimum=2
        ),
        body_max_edge=_real(
            config["body_max_edge"],
            path=f"{path}.body_max_edge",
            minimum=0,
            strict_minimum=True,
        ),
        pad_max_edge=_real(
            config["pad_max_edge"],
            path=f"{path}.pad_max_edge",
            minimum=0,
            strict_minimum=True,
        ),
        max_subdivisions=_integer(
            config["max_subdivisions"], path=f"{path}.max_subdivisions", minimum=1
        ),
    )
    factory()  # Validate the fully assembled config; this performs no robot reads.
    return factory
