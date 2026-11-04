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

"""Robot-owned hand geometry capture and deferred paired grasp services."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING
import math
from pathlib import Path
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import torch

from embodichain.toolkits.graspkit.pg_grasp.contact_fit import SampledHandGeometry
from embodichain.toolkits.graspkit.pg_grasp.coordinated import (
    CoordinatedGraspCfg,
    CoordinatedGraspSampler,
)
from embodichain.toolkits.graspkit.pg_grasp.pose_generator import (
    AntipodalGraspPoseGenerator,
)
from embodichain.toolkits.graspkit.pg_grasp.urdf_surfaces import (
    sample_urdf_collision_surfaces,
)
from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.sim.objects.robot import Robot

__all__ = [
    "RobotHandGeometryCfg",
    "build_parallel_jaw_hand_geometry",
    "RobotHandGeometryProvider",
    "RobotCoordinatedGraspProvider",
]


@configclass
class RobotHandGeometryCfg:
    """Explicit runtime hand selection, commands and surface sampling bounds."""

    motion_part: str = MISSING
    """Control part whose declared solver end link and TCP define the grasp frame."""
    hand_part: str = MISSING
    """Control part containing every moving joint in the selected hand subtree."""
    hand_root_link: str = MISSING
    """Exact URDF root link of the selected hand."""
    pad_link_names: tuple[str, str] = MISSING
    """Exact two pad links; their geometry determines negative/positive TCP-X order."""
    open_qpos: tuple[float, ...] = MISSING
    """Open joint command in the declared hand control-part joint order."""
    closed_qpos: tuple[float, ...] = MISSING
    """Closed joint command in the same order, including explicit mimic followers."""
    sample_count: int = MISSING
    """Number of uniformly spaced open-to-closed kinematic samples."""
    body_max_edge: float = MISSING
    """Body collision-surface subdivision target in metres."""
    pad_max_edge: float = MISSING
    """Pad collision-surface subdivision target in metres."""
    max_subdivisions: int = MISSING
    """Maximum subdivision depth for each selected collision mesh."""

    def validate_values(self) -> None:
        """Validate and normalize complete static capture settings without FK.

        List-valued link names and commands become owned tuples. Robot topology,
        collision shapes and the fixed-TCP boundary are checked at capture time.

        Raises:
            TypeError: Required configuration values remain unresolved.
            ValueError: A name, command or sampling bound is invalid.
        """
        _validate_cfg_values(self)


def _owned_cfg(cfg: RobotHandGeometryCfg) -> RobotHandGeometryCfg:
    if not isinstance(cfg, RobotHandGeometryCfg):
        raise TypeError("cfg must be RobotHandGeometryCfg.")
    result = deepcopy(cfg)
    result.validate_values()
    return result


def _validate_cfg_values(result: RobotHandGeometryCfg) -> None:
    result.validate()
    for name in ("motion_part", "hand_part", "hand_root_link"):
        if type(getattr(result, name)) is not str or not getattr(result, name):
            raise ValueError(f"{name} must be an explicit nonempty name.")
    pads = result.pad_link_names
    if (
        not isinstance(pads, (tuple, list))
        or len(pads) != 2
        or any(type(name) is not str or not name for name in pads)
        or pads[0] == pads[1]
    ):
        raise ValueError("pad_link_names must identify exactly two distinct links.")
    result.pad_link_names = tuple(pads)
    for name in ("open_qpos", "closed_qpos"):
        values = getattr(result, name)
        if (
            not isinstance(values, (tuple, list))
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            )
        ):
            raise ValueError(f"{name} must contain finite numeric joint commands.")
        setattr(result, name, tuple(float(value) for value in values))
    if len(result.open_qpos) != len(result.closed_qpos):
        raise ValueError("Open and closed commands must have equal joint counts.")
    if type(result.sample_count) is not int or result.sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two.")
    if type(result.max_subdivisions) is not int or result.max_subdivisions < 1:
        raise ValueError("max_subdivisions must be a positive integer.")
    for name in ("body_max_edge", "pad_max_edge"):
        value = getattr(result, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (math.isfinite(value) and value > 0)
        ):
            raise ValueError(f"{name} must be finite and positive.")


def _capture(robot: Robot, cfg: RobotHandGeometryCfg) -> SampledHandGeometry:
    path = Path(robot.cfg.fpath).expanduser().resolve(strict=True)
    document = ET.parse(path).getroot()
    joints = {joint.attrib["name"]: joint for joint in document.findall("joint")}
    all_links = {link.attrib["name"]: link for link in document.findall("link")}
    if cfg.hand_root_link not in all_links:
        raise ValueError("The declared hand root is absent from the runtime URDF.")
    descendants = {cfg.hand_root_link}
    while True:
        children = {
            joint.find("child").attrib["link"]
            for joint in joints.values()
            if joint.find("parent").attrib["link"] in descendants
        }
        if children <= descendants:
            break
        descendants |= children
    if not set(cfg.pad_link_names) <= descendants:
        raise ValueError("Both declared pads must belong to the selected hand subtree.")
    link_names = tuple(name for name in all_links if name in descendants)
    moving = {
        name
        for name, joint in joints.items()
        if joint.attrib["type"] != "fixed"
        and joint.find("child").attrib["link"] in descendants
    }
    joint_names = tuple(robot.cfg.control_parts[cfg.hand_part])
    if len(set(joint_names)) != len(joint_names) or set(joint_names) != moving:
        raise ValueError(
            "Hand commands must cover exactly the moving hand-subtree joints, "
            "including mimic followers."
        )
    if len(cfg.open_qpos) != len(joint_names):
        raise ValueError("Hand command width differs from its declared joint order.")
    solver = robot.cfg.solver_cfg[cfg.motion_part]
    if solver.end_link_name not in all_links:
        raise ValueError("The motion solver end link is absent from the runtime URDF.")
    fixed_neighbors: dict[str, set[str]] = {name: set() for name in all_links}
    for joint in joints.values():
        if joint.attrib["type"] == "fixed":
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            fixed_neighbors[parent].add(child)
            fixed_neighbors[child].add(parent)
    connected, pending = {cfg.hand_root_link}, [cfg.hand_root_link]
    while pending:
        for neighbor in fixed_neighbors[pending.pop()] - connected:
            connected.add(neighbor)
            pending.append(neighbor)
    if solver.end_link_name not in connected:
        raise ValueError(
            "Unsupported hand frame: hand root and motion solver end link must "
            "be connected only by fixed URDF joints."
        )
    current = robot.get_qpos()
    if (
        not isinstance(current, torch.Tensor)
        or current.ndim != 2
        or not len(current)
        or current.shape[1] != len(robot.joint_names)
    ):
        raise ValueError("Robot positions must match the full named joint order.")
    states = current[0:1].detach().repeat(cfg.sample_count, 1)
    opened, closed = (
        states.new_tensor(values) for values in (cfg.open_qpos, cfg.closed_qpos)
    )
    ids = [robot.joint_names.index(name) for name in joint_names]
    states[:, ids] = torch.lerp(
        opened[None],
        closed[None],
        torch.linspace(
            0, 1, cfg.sample_count, dtype=states.dtype, device=states.device
        )[:, None],
    )
    fk = robot.compute_fk(
        qpos=states,
        link_names=[*link_names, solver.end_link_name],
        qpos_joint_names=robot.joint_names,
    )
    if (
        not isinstance(fk, torch.Tensor)
        or fk.shape != (cfg.sample_count, len(link_names) + 1, 4, 4)
        or not torch.isfinite(fk).all()
    ):
        raise ValueError("Named FK must return finite sampled link transforms.")
    tcp = torch.as_tensor(solver.tcp, dtype=fk.dtype, device=fk.device)
    if tcp.shape != (4, 4) or not torch.isfinite(tcp).all():
        raise ValueError("The motion TCP must be an explicit finite 4x4 transform.")
    tools = fk[:, -1] @ tcp
    if not torch.allclose(tools, tools[0:1].expand_as(tools), rtol=0, atol=1e-6):
        raise ValueError(
            "Unsupported hand frame: the declared motion TCP moves during hand actuation."
        )
    relative = torch.linalg.inv(tools)[:, None] @ fk[:, :-1]
    clouds = sample_urdf_collision_surfaces(
        path,
        link_names=link_names,
        max_edge_by_link={
            name: cfg.pad_max_edge if name in cfg.pad_link_names else cfg.body_max_edge
            for name in link_names
        },
        max_subdivisions=cfg.max_subdivisions,
    )
    body, pads = [], []
    for index, name in enumerate(link_names):
        vertices = clouds[name].to(relative)
        if not len(vertices):
            continue
        points = (
            vertices[None] @ relative[:, index, :3, :3].transpose(-1, -2)
            + relative[:, index, None, :3, 3]
        )
        (pads if name in cfg.pad_link_names else body).append(points)
    if len(pads) != 2 or not body or pads[0].shape != pads[1].shape:
        raise ValueError(
            "Hand capture requires two matching collision pads and a body."
        )
    pads.sort(key=lambda points: float(points[0, :, 0].mean()))
    pad_tensor = torch.stack(pads, dim=1)
    gaps = pads[1][..., 0].amin(-1) - pads[0][..., 0].amax(-1)
    return SampledHandGeometry(
        torch.cat(body, dim=1).detach(), pad_tensor.detach(), gaps.detach()
    )


def build_parallel_jaw_hand_geometry(
    robot: Robot, *, cfg: RobotHandGeometryCfg
) -> SampledHandGeometry:
    """Capture a configured hand stroke through read-only named robot FK.

    Args:
        robot: Runtime robot owning the URDF, named joint order and TCP calibration.
        cfg: Explicit hand subtree, pad links, commands and sampling bounds.

    Returns:
        Detached body/pad surfaces in the configured TCP frame, with pad order
        negative/positive X. The first measured robot row supplies the common
        reference posture; this captures model geometry, not physical contacts.

    Raises:
        ValueError: The selected joints, geometry, commands or fixed-TCP contract
            are unsupported. The hand root and solver end link must be connected
            by fixed URDF joints, and the motion TCP must stay fixed during closure.

    No simulator step or robot-state write is performed. Rebuild when the model,
    calibration or hand commands change; runtime ownership belongs to the caller.
    """
    return _capture(robot, _owned_cfg(cfg))


class RobotHandGeometryProvider:
    """Own explicit capture settings for one runtime robot, without eager reads."""

    def __init__(self, robot: Robot, *, cfg: RobotHandGeometryCfg) -> None:
        """Store a robot reference and independently owned validated settings.

        Args:
            robot: Runtime owner whose state will be read only at capture time.
            cfg: Explicit static calibration and sampling settings.
        """
        self._robot = robot
        self._cfg = _owned_cfg(cfg)

    @property
    def runtime_owner(self) -> Robot:
        """Return the exact runtime owner; equal names do not imply equal owners."""
        return self._robot

    def capture(self) -> SampledHandGeometry:
        """Read one current reference posture and return newly owned model samples.

        Returns:
            Closing surfaces and descending pad gaps in the declared fixed TCP.
        """
        return _capture(self._robot, self._cfg)


class RobotCoordinatedGraspProvider:
    """Capture two hands lazily for one owning robot/environment adapter.

    Retain the same provider across bridge, engine or episode resets while its
    owning robot, model, commands and calibration remain unchanged. Construction
    and single-arm queries do not capture geometry; the first paired query does.
    No global model-name cache is used. A different robot/environment needs its
    own providers. GraspKit sees only the numeric query port, while its concrete
    sampler receives explicit geometry data.
    """

    def __init__(
        self,
        left: AntipodalGraspPoseGenerator,
        right: AntipodalGraspPoseGenerator,
        left_geometry: RobotHandGeometryProvider,
        right_geometry: RobotHandGeometryProvider,
        *,
        cfg: CoordinatedGraspCfg,
    ) -> None:
        """Bind explicit generators and same-runtime geometry providers.

        Args:
            left: Existing left-role standalone generator.
            right: Existing right-role standalone generator.
            left_geometry: Deferred left hand capture.
            right_geometry: Deferred right hand capture from the same runtime robot.
            cfg: Caller-selected paired sampling and contact settings.
        """
        if left_geometry.runtime_owner is not right_geometry.runtime_owner:
            raise ValueError(
                "Paired geometry providers must belong to the same runtime robot."
            )
        if not all(
            isinstance(value, AntipodalGraspPoseGenerator) for value in (left, right)
        ):
            raise TypeError("Paired contact sampling requires antipodal generators.")
        if any(value.collision_cfg.filter_ground_collision for value in (left, right)):
            raise ValueError(
                "Paired contact sampling requires filter_ground_collision=False."
            )
        self._left, self._right = left, right
        self._left_geometry, self._right_geometry = left_geometry, right_geometry
        self._cfg = deepcopy(cfg)
        self._cfg.validate_values()
        self._sampler: CoordinatedGraspSampler | None = None

    def clear_cache(self) -> None:
        """Release asset-query caches while retaining any captured hand geometry.

        This performs no capture or robot-state read. Changes to the Robot,
        model, hand commands or calibration still require a new provider.
        """
        if self._sampler is not None:
            self._sampler.clear_cache()

    def get_dual_arm_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
        cargo_bounds: torch.Tensor | None = None,
    ) -> list[dict[str, dict[str, object]] | None]:
        """Capture on the first paired query, then delegate unchanged numeric inputs.

        Args:
            mesh_vertices: Object-local surface vertices.
            mesh_triangles: Matching triangle indices.
            obj_poses: Object-to-world query transforms.
            left_to_right_arm_direction: Explicit world-space pairing direction.
            approach_direction: Explicit world-space approach direction.
            middle_empty_ratio: Existing antipodal middle-region exclusion ratio.
            cargo_bounds: Optional observed world AABBs, shape ``(M, 2, 3)``;
                observation and cohort selection belong to the caller.

        Returns:
            Contact-qualified paired candidates from the cached data-only sampler.
        """
        if self._sampler is None:
            left = self._left_geometry.capture()
            right = self._right_geometry.capture()
            self._sampler = CoordinatedGraspSampler(
                self._left, self._right, left, right, cfg=self._cfg
            )
        return self._sampler.get_dual_arm_valid_grasp_poses(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            obj_poses=obj_poses,
            left_to_right_arm_direction=left_to_right_arm_direction,
            approach_direction=approach_direction,
            middle_empty_ratio=middle_empty_ratio,
            cargo_bounds=cargo_bounds,
        )
