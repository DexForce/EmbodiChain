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

"""Final physical acceptance of configured Task Program environments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import torch

from embodichain.compute.geometry.transforms import box_corners_inside_frame_bounds

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.sim.objects import Articulation, RigidObject
    from embodichain.lab.task_program.integrations.simulation.bindings import (
        ControlPartCommandPreset,
    )

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _FinalAcceptance:
    """Normalized thresholds and scene/embodiment bindings for registration."""

    object_uid: str
    articulation_uid: str
    frame_link: str
    corners: tuple[tuple[float, ...], ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    containment_tolerance: float
    joint: str
    joint_min: float
    joint_max: float
    command_preset: str
    control_part: str
    open_positions: tuple[float, ...]
    finger_links: tuple[str, ...]
    hand_tolerance: float
    min_finger_distance: float

    def _scene_objects(self, env: EmbodiedEnv) -> tuple[RigidObject, Articulation]:
        obj = env.sim.get_rigid_object(self.object_uid)
        articulation = env.sim.get_articulation(self.articulation_uid)
        if obj is None or articulation is None:
            raise ValueError(
                "Final acceptance requires a rigid object "
                f"{self.object_uid!r} and articulation {self.articulation_uid!r}."
            )
        return obj, articulation

    def validate_scene(self, env: EmbodiedEnv) -> None:
        """Validate live scene names and the native width of the hand command."""
        _, articulation = self._scene_objects(env)
        if self.frame_link not in articulation.link_names:
            raise ValueError(
                f"Final acceptance frame link {self.frame_link!r} is missing."
            )
        if self.joint not in articulation.joint_names:
            raise ValueError(
                f"Final acceptance articulation joint {self.joint!r} is missing."
            )
        robot = env.robot
        if robot is None or self.control_part not in (robot.control_parts or {}):
            raise ValueError(
                f"Final acceptance control part {self.control_part!r} is missing."
            )
        joint_ids = robot.get_joint_ids(name=self.control_part)
        if len(joint_ids) != len(self.open_positions) or len(set(joint_ids)) != len(
            joint_ids
        ):
            raise ValueError(
                "Final acceptance open command must match the distinct joints "
                f"of control part {self.control_part!r}."
            )
        missing = set(self.finger_links).difference(robot.link_names)
        if missing:
            raise ValueError(
                f"Final acceptance finger links are missing: {sorted(missing)}."
            )

    def measure(self, env: EmbodiedEnv) -> torch.Tensor:
        """Read current observations and return the conjunction per environment."""
        obj, articulation = self._scene_objects(env)
        pose = obj.get_local_pose(to_matrix=True)
        frame = articulation.get_link_pose(self.frame_link, to_matrix=True)
        contained = box_corners_inside_frame_bounds(
            pose,
            frame,
            pose.new_tensor(self.corners),
            pose.new_tensor(self.lower),
            pose.new_tensor(self.upper),
            self.containment_tolerance,
        )
        joint = articulation.get_qpos(target=False)[
            :, articulation.joint_names.index(self.joint)
        ]
        joint_accepted = (
            torch.isfinite(joint)
            & (joint >= self.joint_min)
            & (joint <= self.joint_max)
        )

        robot = env.robot
        hand = robot.get_qpos(name=self.control_part, target=False)
        released = torch.isfinite(hand).all(dim=1) & (
            (hand - hand.new_tensor(self.open_positions)).abs() <= self.hand_tolerance
        ).all(dim=1)
        accepted = contained & joint_accepted & released
        for link in self.finger_links:
            finger = robot.get_link_pose(link, to_matrix=True)[:, :3, 3]
            accepted &= torch.isfinite(finger).all(dim=1) & (
                torch.linalg.vector_norm(finger - pose[:, :3, 3], dim=1)
                > self.min_finger_distance
            )
        return accepted


def _load_final_acceptance(
    value: object,
    *,
    base_dir: Path,
    command_presets: tuple[ControlPartCommandPreset, ...],
    finger_links: Mapping[str, tuple[str, ...]],
) -> _FinalAcceptance:
    """Decode the trusted component and resolve its embodiment-owned hand."""
    from embodichain.lab.gym.utils._component_composition import (
        _component_path,
        _load_yaml_component,
        _mapping,
        _identifier,
    )
    from embodichain.lab.task_program.integrations.configured import (
        _finite_tuple,
        _real,
    )

    selector = _mapping(value, path="acceptance", required=frozenset({"component"}))
    path = _component_path(
        selector["component"], base_dir=base_dir, path="acceptance.component"
    )
    component = _mapping(
        _load_yaml_component(path, field_name="acceptance component"),
        path="acceptance component",
        required=frozenset({"containment", "articulation_joint", "hand_release"}),
    )
    containment = _mapping(
        component["containment"],
        path="acceptance.containment",
        required=frozenset(
            {
                "object",
                "articulation",
                "frame_link",
                "half_extents",
                "lower",
                "upper",
                "tolerance",
            }
        ),
    )
    half_extents = _finite_tuple(
        containment["half_extents"],
        path="acceptance.containment.half_extents",
        expected_length=3,
    )
    if min(half_extents) <= 0:
        raise ValueError("acceptance.containment.half_extents must be positive.")
    lower = _finite_tuple(
        containment["lower"], path="acceptance.containment.lower", expected_length=3
    )
    upper = _finite_tuple(
        containment["upper"], path="acceptance.containment.upper", expected_length=3
    )
    if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError("acceptance.containment.lower must not exceed upper.")
    joint = _mapping(
        component["articulation_joint"],
        path="acceptance.articulation_joint",
        required=frozenset({"joint", "min", "max"}),
    )
    joint_min = _real(joint["min"], path="acceptance.articulation_joint.min")
    joint_max = _real(
        joint["max"], path="acceptance.articulation_joint.max", minimum=joint_min
    )

    hand = _mapping(
        component["hand_release"],
        path="acceptance.hand_release",
        required=frozenset({"command_preset", "tolerance", "min_finger_distance"}),
    )
    preset_id = _identifier(
        hand["command_preset"], path="acceptance.hand_release.command_preset"
    )
    preset = next(
        (item for item in command_presets if item.preset_id == preset_id), None
    )
    if preset is None or "open" not in preset.commands:
        raise ValueError(
            f"Final acceptance needs an open command in preset {preset_id!r}."
        )
    links = finger_links.get(preset.control_part)
    if not links:
        raise ValueError(
            "Final acceptance needs embodiment finger_links for control part "
            f"{preset.control_part!r}."
        )
    return _FinalAcceptance(
        object_uid=_identifier(
            containment["object"], path="acceptance.containment.object"
        ),
        articulation_uid=_identifier(
            containment["articulation"], path="acceptance.containment.articulation"
        ),
        frame_link=_identifier(
            containment["frame_link"], path="acceptance.containment.frame_link"
        ),
        corners=tuple(product(*((-size, size) for size in half_extents))),
        lower=lower,
        upper=upper,
        containment_tolerance=_real(
            containment["tolerance"], path="acceptance.containment.tolerance", minimum=0
        ),
        joint=_identifier(joint["joint"], path="acceptance.articulation_joint.joint"),
        joint_min=joint_min,
        joint_max=joint_max,
        command_preset=preset_id,
        control_part=preset.control_part,
        open_positions=preset.commands["open"],
        finger_links=tuple(sorted(links)),
        hand_tolerance=_real(
            hand["tolerance"], path="acceptance.hand_release.tolerance", minimum=0
        ),
        min_finger_distance=_real(
            hand["min_finger_distance"],
            path="acceptance.hand_release.min_finger_distance",
            minimum=0,
        ),
    )
