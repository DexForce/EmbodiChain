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
"""Decode task-space goals and bind their frames to a trajectory planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.lab.sim.motion._cartesian_track import (
    _ArcGeometry,
    _plan_cartesian_track,
    _TrackKeyframe,
    _TrackLimits,
)
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.solvers._trajectory_ik import _TrajectoryIK
from embodichain.lab.task_program.semantics import SceneRegistry

from .simulation.targets import AbsolutePoseTarget, SceneEntityTarget

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.task_program.semantics import SceneManifest

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _Arc:
    axis: AbsolutePoseTarget | SceneEntityTarget
    radius_m: float
    start_deg: float
    sweep_deg: float
    height_m: tuple[float, float]
    tool_offset: tuple[float, float, float]
    tool_orientation: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _ArcEndpoint:
    arc: str
    end: bool
    radial_offset_m: float
    tangent_offset_m: float


@dataclass(frozen=True, slots=True)
class _Keyframe:
    step: int
    target: str
    grip: float
    interpolation: str


@dataclass(frozen=True, slots=True)
class _CartesianTrack:
    control_hz: float
    arm: str
    gripper: str
    tool_link: str
    targets: tuple[
        tuple[str, AbsolutePoseTarget | SceneEntityTarget | _ArcEndpoint], ...
    ]
    arcs: tuple[tuple[str, _Arc], ...]
    keyframes: tuple[_Keyframe, ...]
    position_tolerance_m: float
    rotation_tolerance_rad: float

    @property
    def steps(self) -> int:
        return self.keyframes[-1].step

    @property
    def step_dt(self) -> float:
        return 1 / self.control_hz

    def validate_scene(self, scene: SceneManifest) -> None:
        poses = [target for _, target in self.targets]
        poses.extend(arc.axis for _, arc in self.arcs)
        for pose in poses:
            if isinstance(pose, SceneEntityTarget):
                scene.resolve(pose.entity_id)


@dataclass(slots=True)
class _PlannedTrack:
    positions: np.ndarray
    poses: np.ndarray
    targets: dict[str, np.ndarray]
    kinematics: _TrajectoryIK
    arm_ids: tuple[int, ...]
    limits: _TrackLimits


def _resolve_and_plan_track(
    track: _CartesianTrack,
    robot: Robot,
    scene_registry: SceneRegistry,
) -> _PlannedTrack:
    if track.arm not in robot.control_parts or track.gripper not in robot.control_parts:
        raise ValueError(
            "Trajectory planning requires the declared arm and gripper parts."
        )
    arm_ids = robot.get_joint_ids(track.arm)
    gripper_ids = robot.get_joint_ids(track.gripper)
    if not arm_ids or not gripper_ids or set(arm_ids) & set(gripper_ids):
        raise ValueError("Trajectory arm and gripper must control distinct joints.")
    names = tuple(robot.joint_names)
    home = np.asarray(robot.cfg.init_qpos, dtype=float)
    if home.shape != (len(names),) or not np.isfinite(home).all():
        raise ValueError(
            "Trajectory planning requires one finite configured robot home pose."
        )
    kinematics = _TrajectoryIK(
        Path(robot.cfg.fpath),
        track.tool_link,
        tuple(names[i] for i in arm_ids),
        tuple(robot.cfg.init_pos),
        tuple(robot.cfg.init_rot),
    )
    limits = _TrackLimits(
        names,
        robot.get_qpos_limits()[0].cpu().numpy(),
        authored_joint_velocity_limits(robot, None)[0].cpu().numpy(),
        track.step_dt,
    )
    frames: dict[str, np.ndarray] = {}

    def resolve(pose: AbsolutePoseTarget | SceneEntityTarget) -> np.ndarray:
        if isinstance(pose, AbsolutePoseTarget):
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(pose.quaternion_xyzw).as_matrix()
            matrix[:3, 3] = pose.position
            return matrix
        if pose.entity_id not in frames:
            provider = scene_registry.lookup(pose.entity_id).state_provider
            if provider is None:
                raise ValueError(
                    f"Trajectory reference {pose.entity_id!r} has no pose provider."
                )
            measurement = provider.observe(
                timestamp=0.0, env_ids=torch.tensor([0], device=robot.device)
            )
            value = measurement.pose
            if value is None or value.shape != (1, 4, 4):
                raise ValueError(
                    f"Trajectory reference {pose.entity_id!r} requires one arena pose."
                )
            frames[pose.entity_id] = value[0].detach().cpu().double().numpy()
        matrix = frames[pose.entity_id].copy()
        if pose.relative_pose is not None:
            matrix = matrix @ np.asarray(pose.relative_pose).reshape(4, 4)
        if pose.world_orientation is not None:
            matrix[:3, :3] = np.asarray(pose.world_orientation).reshape(3, 3)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Trajectory reference {pose.entity_id!r} must be finite.")
        return matrix

    arcs = {
        name: _ArcGeometry(
            resolve(arc.axis),
            arc.radius_m,
            float(np.deg2rad(arc.start_deg)),
            float(np.deg2rad(arc.start_deg) + np.deg2rad(arc.sweep_deg)),
            arc.height_m,
            np.asarray(arc.tool_offset),
            Rotation.from_quat(arc.tool_orientation).as_matrix(),
        )
        for name, arc in track.arcs
    }
    targets = {"home": kinematics.forward(home[arm_ids])}
    arc_endpoints = {}
    for name, target in track.targets:
        if isinstance(target, _ArcEndpoint):
            targets[name] = arcs[target.arc].endpoint(
                end=target.end,
                radial_offset=target.radial_offset_m,
                tangent_offset=target.tangent_offset_m,
            )
            arc_endpoints[name] = target.arc
        else:
            targets[name] = resolve(target)
    keyframes = tuple(
        _TrackKeyframe(
            key.step,
            targets[key.target],
            key.grip,
            key.interpolation == "joint_home",
            arcs[arc_endpoints[key.target]] if key.interpolation == "arc" else None,
        )
        for key in track.keyframes
    )
    positions, poses = _plan_cartesian_track(
        keyframes,
        home=home,
        arm_ids=arm_ids,
        gripper_ids=gripper_ids,
        control_hz=track.control_hz,
        kinematics=kinematics,
        position_tolerance=track.position_tolerance_m,
        rotation_tolerance=track.rotation_tolerance_rad,
    )
    limits.validate(positions)
    return _PlannedTrack(positions, poses, targets, kinematics, tuple(arm_ids), limits)


def _decode_cartesian_track(value: object, *, path: str) -> _CartesianTrack:
    from .configured import (
        _boolean,
        _decode_goal_pose,
        _finite_tuple,
        _identifier,
        _integer,
        _mapping,
        _real,
        _sequence,
    )

    config = _mapping(
        value,
        path=path,
        required=frozenset(
            {
                "control_hz",
                "arm",
                "gripper",
                "tool_link",
                "constraints",
                "targets",
                "keyframes",
            }
        ),
        optional=frozenset({"arcs"}),
    )
    constraints = _mapping(
        config["constraints"],
        path=f"{path}.constraints",
        required=frozenset({"position_tolerance_m", "rotation_tolerance_rad"}),
    )
    arcs = {}
    raw_arcs = config.get("arcs", {})
    if not isinstance(raw_arcs, Mapping):
        raise TypeError(
            f"{path}.arcs must map names to axis-relative arc declarations."
        )
    for name, item in raw_arcs.items():
        key = f"{path}.arcs.{name}"
        name = _identifier(name, path=key)
        raw = _mapping(
            item,
            path=key,
            required=frozenset(
                {
                    "axis",
                    "radius_m",
                    "start_deg",
                    "sweep_deg",
                    "height_m",
                    "tool_offset",
                    "tool_orientation_xyzw",
                }
            ),
        )
        orientation = AbsolutePoseTarget(
            (0.0, 0.0, 0.0),
            _finite_tuple(
                raw["tool_orientation_xyzw"],
                path=f"{key}.tool_orientation_xyzw",
                expected_length=4,
            ),
        ).quaternion_xyzw
        sweep = _real(raw["sweep_deg"], path=f"{key}.sweep_deg")
        if sweep == 0:
            raise ValueError(f"{key}.sweep_deg must be nonzero.")
        arcs[name] = _Arc(
            _decode_goal_pose(raw["axis"], path=f"{key}.axis"),
            _real(
                raw["radius_m"], path=f"{key}.radius_m", minimum=0, strict_minimum=True
            ),
            _real(raw["start_deg"], path=f"{key}.start_deg"),
            sweep,
            _finite_tuple(raw["height_m"], path=f"{key}.height_m", expected_length=2),
            _finite_tuple(
                raw["tool_offset"], path=f"{key}.tool_offset", expected_length=3
            ),
            orientation,
        )
    targets = {}
    if not isinstance(config["targets"], Mapping):
        raise TypeError(f"{path}.targets must map names to pose declarations.")
    for name, item in config["targets"].items():
        key = f"{path}.targets.{name}"
        name = _identifier(name, path=key)
        if name == "home":
            raise ValueError(f"{key} is reserved for the robot's configured home.")
        if isinstance(item, Mapping) and item.get("kind") == "arc_endpoint":
            raw = _mapping(
                item,
                path=key,
                required=frozenset({"kind", "arc", "end"}),
                optional=frozenset({"radial_offset_m", "tangent_offset_m"}),
            )
            arc = _identifier(raw["arc"], path=f"{key}.arc")
            if arc not in arcs:
                raise ValueError(f"{key}.arc must name a declared arc.")
            targets[name] = _ArcEndpoint(
                arc,
                _boolean(raw["end"], path=f"{key}.end"),
                _real(raw.get("radial_offset_m", 0), path=f"{key}.radial_offset_m"),
                _real(raw.get("tangent_offset_m", 0), path=f"{key}.tangent_offset_m"),
            )
        else:
            targets[name] = _decode_goal_pose(item, path=key)
    keyframes = []
    for index, item in enumerate(
        _sequence(config["keyframes"], path=f"{path}.keyframes")
    ):
        key = f"{path}.keyframes[{index}]"
        raw = _mapping(
            item,
            path=key,
            required=frozenset({"step", "target", "grip"}),
            optional=frozenset({"interpolation"}),
        )
        step = _integer(raw["step"], path=f"{key}.step", minimum=0)
        target = _identifier(raw["target"], path=f"{key}.target")
        mode = raw.get("interpolation", "cartesian")
        if target not in targets and target != "home":
            raise ValueError(f"{key}.target must name a declared pose or home.")
        if not keyframes:
            if step != 0 or target != "home":
                raise ValueError(
                    f"{key} must start at frame zero in the configured home."
                )
        elif step <= keyframes[-1].step:
            raise ValueError(f"{key}.step must increase strictly.")
        if mode not in ("cartesian", "joint_home", "arc"):
            raise ValueError(
                f"{key}.interpolation must be cartesian, joint_home or arc."
            )
        if mode == "joint_home" and (not keyframes or target != "home"):
            raise ValueError(f"{key} joint interpolation must lead to home.")
        if mode == "arc":
            end = targets.get(target)
            start = targets.get(keyframes[-1].target) if keyframes else None
            if not (
                isinstance(start, _ArcEndpoint)
                and isinstance(end, _ArcEndpoint)
                and start.arc == end.arc
                and not start.end
                and end.end
                and not (
                    start.radial_offset_m
                    or start.tangent_offset_m
                    or end.radial_offset_m
                    or end.tangent_offset_m
                )
            ):
                raise ValueError(
                    f"{key}.interpolation requires matching unshifted arc endpoints."
                )
        keyframes.append(
            _Keyframe(step, target, _real(raw["grip"], path=f"{key}.grip"), mode)
        )
    if len(keyframes) < 2:
        raise ValueError(f"{path}.keyframes requires a start and a later target.")
    return _CartesianTrack(
        _real(
            config["control_hz"],
            path=f"{path}.control_hz",
            minimum=0,
            strict_minimum=True,
        ),
        *(
            _identifier(config[name], path=f"{path}.{name}")
            for name in ("arm", "gripper", "tool_link")
        ),
        tuple(targets.items()),
        tuple(arcs.items()),
        tuple(keyframes),
        _real(
            constraints["position_tolerance_m"],
            path=f"{path}.constraints.position_tolerance_m",
            minimum=0,
            strict_minimum=True,
        ),
        _real(
            constraints["rotation_tolerance_rad"],
            path=f"{path}.constraints.rotation_tolerance_rad",
            minimum=0,
            strict_minimum=True,
        ),
    )
