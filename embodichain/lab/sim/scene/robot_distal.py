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

"""Observed distal robot geometry for bounded staging hints."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING
import numpy as np
import torch
from embodichain.compute.geometry.pose_checks import as_rigid_matrix as _pose
from ._native_hand_scene_capture import _descendants

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["RobotDistalMeshSnapshot", "capture_robot_distal_meshes"]

_FK_ATOL = 2e-4


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class RobotDistalMeshSnapshot:
    """Owned observed arm frames and distal collision-input points.

    Args:
        active: Exact active control-part identity.
        passive: Exact disjoint passive arm control-part identity.
        tcp: Observed world TCP matrices by part.
        roots: Observed world arm-root matrices by part.
        points: Ordered transformed distal collision-mesh points by part.
        hand_owner_fk_error: Maximum observed/model hand-owner discrepancy by part;
            diagnostic only, unlike strict arm FK calibration.

    These input-mesh envelopes are staging hints, not native cooked collision
    geometry or a certificate that a future path is collision free.
    """

    active: str
    passive: str
    tcp: Mapping[str, np.ndarray]
    roots: Mapping[str, np.ndarray]
    points: Mapping[str, np.ndarray]
    hand_owner_fk_error: Mapping[str, float]

    def __post_init__(self) -> None:
        parts = {self.active, self.passive}
        if len(parts) != 2 or any(type(part) is not str or not part for part in parts):
            raise ValueError(
                "Distal geometry requires two explicit disjoint arm identities."
            )
        for name in ("tcp", "roots", "points"):
            source = getattr(self, name)
            if set(source) != parts:
                raise ValueError(
                    "Distal geometry maps must contain the exact two arm identities."
                )
            owned = {}
            for part, raw in source.items():
                value = np.asarray(raw).copy()
                if name == "points":
                    if (
                        value.ndim != 2
                        or value.shape[1] != 3
                        or not len(value)
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError(
                            "Distal point envelopes require finite nonempty (N,3) arrays."
                        )
                else:
                    _pose(value)
                value.setflags(write=False)
                owned[part] = value
            object.__setattr__(self, name, MappingProxyType(owned))
        if set(self.hand_owner_fk_error) != parts:
            raise ValueError("Distal hand diagnostics must cover both arm identities.")
        object.__setattr__(
            self,
            "hand_owner_fk_error",
            MappingProxyType(dict(self.hand_owner_fk_error)),
        )


def _calibrate(
    actual: np.ndarray | torch.Tensor, predicted: np.ndarray | torch.Tensor
) -> float:
    error = float(np.max(np.abs(_pose(actual) - _pose(predicted))))
    if error > _FK_ATOL:
        raise ValueError("staging_arm_fk_calibration_failed")
    return error


def capture_robot_distal_meshes(
    robot: Robot,
    *,
    expected_qpos: torch.Tensor,
    active_control_part: str,
    expected_root_pose: torch.Tensor | None = None,
) -> RobotDistalMeshSnapshot:
    """Capture same-observation distal collision-input geometry without stepping.

    Args:
        robot: Exact host-owned B1 robot with assembled URDF and solver frames.
        expected_qpos: Actual full-robot observation, shape ``(1,D)``.
        active_control_part: Explicit active arm; exactly one disjoint passive
            solver must exist in the robot's configured control parts.
        expected_root_pose: Optional actual world-root observation to calibrate.

    Returns:
        Owned points/frames retaining source model and mesh order. Arm chain/FK
        discrepancies reject; distal hand-owner discrepancies remain diagnostic.

    Raises:
        ValueError: Geometry, ownership or measured state cannot be qualified.
            Projected starts must not substitute for the actual observation.
    """
    if expected_qpos.ndim != 2 or expected_qpos.shape[0] != 1:
        raise ValueError("Distal capture requires an actual B1 robot observation.")
    active = active_control_part
    import yourdfpy

    if not torch.equal(robot.get_qpos()[0], expected_qpos[0]):
        raise ValueError("staging_observed_qpos_changed")
    model = yourdfpy.URDF.load(
        robot.cfg.fpath,
        load_meshes=False,
        build_scene_graph=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
    )
    children: dict[str, set[str]] = {}
    parents = {}
    for joint in model.robot.joints:
        children.setdefault(joint.parent, set()).add(joint.child)
        if joint.child in parents:
            raise ValueError("staging_ambiguous_arm_chain")
        parents[joint.child] = joint
    if active not in robot.control_parts or robot.get_solver(active) is None:
        raise ValueError("staging_missing_active_solver")
    active_links = _descendants(children, robot.get_solver(active).root_link_name)
    alternatives = [
        (part, robot.get_solver(part))
        for part in robot.control_parts
        if part != active and robot.get_solver(part) is not None
    ]
    if len(alternatives) != 1 or not _descendants(
        children, alternatives[0][1].root_link_name
    ).isdisjoint(active_links):
        raise ValueError("staging_ambiguous_passive_arm")
    passive, passive_solver = alternatives[0]
    if not set(robot.get_joint_ids(active)).isdisjoint(robot.get_joint_ids(passive)):
        raise ValueError("staging_overlapping_arm_joints")
    world_root = _pose(robot.get_local_pose(to_matrix=True)).copy()
    if expected_root_pose is not None:
        _calibrate(expected_root_pose, world_root)
    model.update_cfg(
        {
            name: float(value)
            for name, value in zip(robot.joint_names, expected_qpos[0])
            if name in model.joint_map
        }
    )
    tcp, roots, distal, chains = {}, {}, {}, {}
    for part, solver in ((active, robot.get_solver(active)), (passive, passive_solver)):
        cursor, moving, chain = solver.end_link_name, [], {solver.root_link_name}
        while cursor != solver.root_link_name:
            if cursor in chain or cursor not in parents:
                raise ValueError("staging_missing_arm_chain")
            chain.add(cursor)
            joint = parents[cursor]
            if joint.type != "fixed":
                moving.append(cursor)
            cursor = joint.parent
        if len(moving) < 2:
            raise ValueError("staging_missing_distal_arm_chain")
        distal[part], chains[part] = _descendants(children, moving[1]), chain
        for link in chain:
            _calibrate(
                robot.get_link_pose(link, env_ids=[0], to_matrix=True),
                world_root @ model.get_transform(link),
            )
        roots[part] = _pose(
            robot.get_link_pose(solver.root_link_name, env_ids=[0], to_matrix=True)
        )
        joint_ids = tuple(robot.get_joint_ids(part))
        if not joint_ids:
            raise ValueError("staging_missing_arm_joints")
        tcp[part] = _pose(
            robot.compute_fk(
                expected_qpos[:, list(joint_ids)],
                name=part,
                env_ids=[0],
                to_matrix=True,
            )
        )
        _calibrate(
            tcp[part],
            world_root
            @ model.get_transform(solver.end_link_name)
            @ _pose(solver.get_tcp()),
        )
    points: dict[str, list[np.ndarray]] = {active: [], passive: []}
    hand_error: dict[str, float] = {active: 0.0, passive: 0.0}
    graph = model.collision_scene.graph
    for node in graph.nodes_geometry:
        owner, visited = node, set()
        while owner not in model.link_map:
            if owner in visited or owner not in graph.transforms.parents:
                raise ValueError("staging_missing_mesh_owner")
            visited.add(owner)
            owner = graph.transforms.parents[owner]
        _, mesh_name = graph[node]
        vertices = np.asarray(model.collision_scene.geometry[mesh_name].vertices)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("staging_invalid_collision_mesh")
        for part in (active, passive):
            if owner not in distal[part]:
                continue
            observed = _pose(robot.get_link_pose(owner, env_ids=[0], to_matrix=True))
            predicted = _pose(world_root @ model.get_transform(owner))
            if owner in chains[part]:
                _calibrate(observed, predicted)
            else:
                hand_error[part] = max(
                    hand_error[part], float(np.max(np.abs(observed - predicted)))
                )
            transform = observed @ _pose(
                model.get_transform(node, frame_from=owner, collision_geometry=True)
            )
            points[part].append(vertices @ transform[:3, :3].T + transform[:3, 3])
    if any(not points[part] for part in (active, passive)):
        raise ValueError("staging_missing_distal_meshes")
    if not torch.equal(robot.get_qpos()[0], expected_qpos[0]) or not np.array_equal(
        _pose(robot.get_local_pose(to_matrix=True)), world_root
    ):
        raise ValueError("staging_observed_robot_changed")
    return RobotDistalMeshSnapshot(
        active=active,
        passive=passive,
        tcp=tcp,
        roots=roots,
        points={part: np.concatenate(values) for part, values in points.items()},
        hand_owner_fk_error=hand_error,
    )
