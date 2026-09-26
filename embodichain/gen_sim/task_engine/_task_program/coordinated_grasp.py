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

"""E5-only URDF hand geometry; no motion planning, execution, or single-arm policy."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

from embodichain.toolkits.graspkit import ParallelJawGraspPoseGenerator
from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator
from embodichain.utils import logger

__all__: list[str] = []

COORDINATED_GRASP_REVISION = 2
_CONTACT_TOLERANCE = 0.0005


@dataclass(frozen=True)
class HandGeometry:
    """URDF surface samples from open to closed, expressed in the configured TCP.

    This is a sampled target-mesh clearance check, not continuous collision
    detection or evidence of physical attachment. Pad order is negative/positive X.
    """

    body: torch.Tensor
    pads: torch.Tensor
    gaps: torch.Tensor

    def at_width(self, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate hand geometry at a full pad gap, not a point-pair contact."""
        gaps = self.gaps.to(widths)
        high = torch.searchsorted(-gaps.contiguous(), -widths).clamp(1, len(gaps) - 1)
        low = high - 1
        alpha = ((gaps[low] - widths) / (gaps[low] - gaps[high])).clamp(0, 1)
        body, pads = self.body.to(widths), self.pads.to(widths)
        return (
            torch.lerp(body[low], body[high], alpha[:, None, None]),
            torch.lerp(pads[low], pads[high], alpha[:, None, None, None]),
        )


def _collision_points(link: ET.Element, directory: Path, *, pad: bool) -> torch.Tensor:
    clouds = []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        box = collision.find("geometry/box")
        if mesh is not None:
            source = Path(mesh.attrib["filename"])
            shape = trimesh.load(directory / source, force="mesh", process=False)
            shape.vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        elif box is not None:
            shape = trimesh.creation.box(np.fromstring(box.attrib["size"], sep=" "))
        else:
            raise ValueError(
                "E5 hand collision geometry requires URDF meshes or boxes."
            )
        pose = np.eye(4)
        origin = collision.find("origin")
        if origin is not None:
            pose[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
            pose[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        # Deterministic samples include faces, not just the eight pad corners.
        vertices, _ = trimesh.remesh.subdivide_to_size(
            shape.vertices,
            shape.faces,
            max_edge=0.004 if pad else 0.01,
            max_iter=10,
        )
        clouds.append(trimesh.transform_points(vertices, pose))
    if not clouds:
        return torch.empty(0, 3)
    return torch.as_tensor(
        np.unique(np.concatenate(clouds), axis=0), dtype=torch.float32
    )


def build_hand_geometry(
    robot: Any, *, motion_part: str, hand_part: str, commands: Any
) -> HandGeometry:
    """Read collision, not visual meshes, and use public named FK without stepping."""
    path = Path(robot.cfg.fpath)
    document = ET.parse(path).getroot()
    joint_names = robot.cfg.control_parts[hand_part]
    joints = {joint.attrib["name"]: joint for joint in document.findall("joint")}
    root = joints[joint_names[0]].find("parent").attrib["link"]
    descendants = {root}
    while True:
        children = {
            joint.find("child").attrib["link"]
            for joint in joints.values()
            if joint.find("parent").attrib["link"] in descendants
        }
        if children <= descendants:
            break
        descendants |= children
    links = [
        link for link in document.findall("link") if link.attrib["name"] in descendants
    ]
    moving = {
        name
        for name, joint in joints.items()
        if joint.attrib["type"] != "fixed"
        and joint.find("child").attrib["link"] in descendants
    }
    if not moving <= set(joint_names):
        raise ValueError(
            "E5 requires explicit commands for all moving hand joints, including mimics."
        )
    solver = robot.cfg.solver_cfg[motion_part]
    states = robot.get_qpos()[0:1].repeat(65, 1)
    opened, closed = (states.new_tensor(commands[key]) for key in ("open", "grasp"))
    ids = [robot.joint_names.index(name) for name in joint_names]
    states[:, ids] = torch.lerp(
        opened[None],
        closed[None],
        torch.linspace(0, 1, 65, device=states.device)[:, None],
    )
    fk = robot.compute_fk(
        qpos=states,
        link_names=[link.attrib["name"] for link in links] + [solver.end_link_name],
        qpos_joint_names=robot.joint_names,
    )
    tools = fk[:, -1] @ torch.as_tensor(solver.tcp, dtype=fk.dtype, device=fk.device)
    relative = torch.linalg.inv(tools)[:, None] @ fk[:, :-1]
    body, pads = [], []
    for index, link in enumerate(links):
        pad = link.attrib["name"].endswith("finger_pad")
        vertices = _collision_points(link, path.parent, pad=pad).to(relative)
        if not len(vertices):
            continue
        points = (
            vertices[None] @ relative[:, index, :3, :3].transpose(-1, -2)
            + relative[:, index, None, :3, 3]
        )
        (pads if pad else body).append(points)
    if len(pads) != 2 or not body or pads[0].shape != pads[1].shape:
        raise ValueError(
            "E5 requires two matching Robotiq collision pads and a hand body."
        )
    pads.sort(key=lambda points: float(points[0, :, 0].mean()))
    pad_tensor = torch.stack(pads, dim=1)
    gaps = pads[1][..., 0].amin(-1) - pads[0][..., 0].amax(-1)
    if not torch.isfinite(pad_tensor).all() or not bool((gaps[:-1] > gaps[1:]).all()):
        raise ValueError(
            "E5 requires a finite, monotonically closing pad gap in TCP X."
        )
    return HandGeometry(torch.cat(body, dim=1), pad_tensor, gaps)


class TriangleTarget:
    """Signed triangle distances preserve cavities lost by a convex approximation."""

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
        import open3d as o3d

        mesh = trimesh.Trimesh(
            vertices.detach().cpu().numpy(), faces.detach().cpu().numpy(), process=True
        )
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError(
                "E5 signed contact queries require a watertight, consistently wound target mesh."
            )
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(
            o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
        )
        self.lower, self.upper = torch.tensor(mesh.bounds, dtype=torch.float32)

    def query_batch_points(
        self, points: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import open3d as o3d

        flat = points.detach().reshape(-1, 3).cpu()
        distance = torch.linalg.vector_norm(
            torch.maximum(self.lower - flat, flat - self.upper).clamp_min(0), dim=-1
        )
        near = ((flat >= self.lower - 0.001) & (flat <= self.upper + 0.001)).all(-1)
        if near.any():
            values = self.scene.compute_signed_distance(
                o3d.core.Tensor(flat[near].contiguous().numpy()), nthreads=2, nsamples=3
            ).numpy()
            distance[near] = torch.from_numpy(values)
        distance = distance.reshape(points.shape[:-1]).to(points)
        return distance <= 0, distance


def _inward_faces(pads: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            (pads[:, 0, :, 0] - pads[:, 0, :, 0].amax(-1, keepdim=True)).abs() < 1e-5,
            (pads[:, 1, :, 0] - pads[:, 1, :, 0].amin(-1, keepdim=True)).abs() < 1e-5,
        ),
        dim=1,
    )


class HandCollisionChecker:
    """Replace only the E5 sampler's three-box query with actual hand geometry."""

    def __init__(self, geometry: HandGeometry, target: Any) -> None:
        self.geometry = geometry
        self.target = target

    def fit_contacts(
        self, obj_pose: torch.Tensor, poses: torch.Tensor, widths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit finite pad surfaces locally; acceptance still requires a full query.

        Antipodal points only propose a neighborhood. Their distance generally
        underestimates the gap at first contact of two finite pads on curved walls.
        """
        result, gaps = poses.clone(), widths.clone()
        valid = torch.isfinite(poses).all(-1).all(-1) & torch.isfinite(widths)
        valid &= (widths > 0) & (widths < self.geometry.gaps[0])
        ids = torch.where(valid)[0]
        if not len(ids):
            return result, gaps
        local_poses, local_gaps = result[ids].clone(), gaps[ids].clone()
        _, initial_pads = self.geometry.at_width(local_gaps)
        # Keep the correction inside the original finite pad neighborhood.
        span = (initial_pads[..., 1].amax(-1) - initial_pads[..., 1].amin(-1)).amin(-1)
        shifts = torch.zeros_like(local_gaps)
        for _ in range(8):
            _, pads = self.geometry.at_width(local_gaps)
            transform = torch.linalg.inv(obj_pose) @ local_poses
            points = (
                pads.flatten(1, 2) @ transform[:, :3, :3].transpose(-1, -2)
                + transform[:, None, :3, 3]
            )
            _, distances = self.target.query_batch_points(points)
            distances = distances.to(pads).reshape(pads.shape[:-1])
            minima = distances.masked_fill(~_inward_faces(pads), torch.inf).amin(-1)
            step = (0.0002 - minima.sum(-1)).clamp(-0.01, 0.01)
            local_gaps = (local_gaps + step).clamp(
                max(float(self.geometry.gaps[-1]), 0.0001),
                float(self.geometry.gaps[0]) - 0.001,
            )
            correction = ((minima[:, 0] - minima[:, 1]) / 2).clamp(-0.005, 0.005)
            next_shift = torch.maximum(
                torch.minimum(shifts + correction, span / 2), -span / 2
            )
            local_poses[:, :3, 3] += (
                local_poses[:, :3, 0] * (next_shift - shifts)[:, None]
            )
            shifts = next_shift
        result[ids], gaps[ids] = local_poses, local_gaps
        return result, gaps

    def query(
        self,
        obj_pose: torch.Tensor,
        grasp_poses: torch.Tensor,
        open_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kwargs.get("is_filter_ground_collision", False):
            raise ValueError(
                "E5 URDF hand query does not implement inferred-ground filtering."
            )
        if kwargs.get("collision_threshold", 0.0) != 0.0:
            raise ValueError(
                "E5 URDF hand query requires the zero collision threshold."
            )
        geometry = self.geometry
        relative = torch.linalg.inv(obj_pose) @ grasp_poses
        valid = torch.isfinite(open_lengths) & (open_lengths > 0)
        valid &= (open_lengths < geometry.gaps[0]) & (open_lengths >= geometry.gaps[-1])
        valid &= torch.isfinite(relative).all(dim=-1).all(dim=-1)
        counts = {
            "input": len(valid),
            "invalid_width_or_pose": int((~valid).sum()),
            "contact_missing": 0,
            "contact_body": 0,
            "contact_pad": 0,
            "open_body": 0,
            "open_pad": 0,
            "closing_body": 0,
            "closing_pad": 0,
        }
        depth = open_lengths.new_zeros(len(open_lengths))
        for start in range(0, len(open_lengths), 16):
            indices = torch.arange(
                start, min(start + 16, len(open_lengths)), device=open_lengths.device
            )
            indices = indices[valid[indices]]
            if not len(indices):
                continue
            widths = open_lengths[indices]
            end_body, end_pads = geometry.at_width(widths)

            def check(
                body: torch.Tensor,
                pads: torch.Tensor,
                *,
                contact: bool,
                active: torch.Tensor,
                phase: str,
            ) -> None:
                selected = active & valid[indices]
                if not selected.any():
                    return
                local = torch.cat((body, pads.flatten(1, 2)), dim=1)[selected]
                transforms = relative[indices[selected]]
                points = (
                    local @ transforms[:, :3, :3].transpose(-1, -2)
                    + transforms[:, None, :3, 3]
                )
                _, distance = self.target.query_batch_points(
                    points, collision_threshold=0.0
                )
                distance = distance.to(open_lengths)
                allowed = torch.zeros_like(distance, dtype=torch.bool)
                if contact:
                    faces = _inward_faces(pads).flatten(1, 2)
                    allowed[:, body.shape[1] :] = faces[selected]
                    pad_distances = distance[:, body.shape[1] :].reshape(
                        -1, 2, pads.shape[2]
                    )
                    nearest = pad_distances.masked_fill(
                        ~_inward_faces(pads)[selected], torch.inf
                    ).amin(-1)
                    touching = (nearest.abs() <= _CONTACT_TOLERANCE).all(-1)
                    counts["contact_missing"] += int((~touching).sum())
                    valid[indices[selected]] &= touching
                limit = torch.where(allowed, -_CONTACT_TOLERANCE, -1e-6)
                body_bad = (
                    distance[:, : body.shape[1]] < limit[:, : body.shape[1]]
                ).any(-1)
                pad_bad = (
                    distance[:, body.shape[1] :] < limit[:, body.shape[1] :]
                ).any(-1)
                counts[phase + "_body"] += int(body_bad.sum())
                counts[phase + "_pad"] += int((pad_bad & ~body_bad).sum())
                valid[indices[selected]] &= (distance >= limit).all(-1)
                depth[indices[selected]] = distance.amin(-1)

            count = len(indices)
            # Validate contact endpoint first, then the actual open-to-contact sweep.
            check(
                end_body,
                end_pads,
                contact=True,
                active=torch.ones(count, dtype=torch.bool, device=widths.device),
                phase="contact",
            )
            for state, gap in enumerate(geometry.gaps):
                if not valid[indices].any():
                    break
                active = gap.to(widths) > widths
                body = geometry.body[state].to(widths)[None].expand(count, -1, -1)
                pads = geometry.pads[state].to(widths)[None].expand(count, -1, -1, -1)
                check(
                    body,
                    pads,
                    contact=False,
                    active=active,
                    phase="open" if state == 0 else "closing",
                )
        counts["accepted"] = int(valid.sum())
        self.last_counts = counts
        logger.log_info(f"GenSim E5 URDF grasp filtering: {counts}")
        return ~valid, depth


class _ProposalQuery:
    """Do not veto proposals with a three-box proxy; no proposal is yet eligible."""

    def query(
        self,
        obj_pose: torch.Tensor,
        poses: torch.Tensor,
        widths: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(
            len(poses), dtype=torch.bool, device=poses.device
        ), widths.new_zeros(len(poses))


class _HandSampler(AntipodalGraspPoseGenerator):
    def __init__(
        self, delegate: AntipodalGraspPoseGenerator, geometry: HandGeometry, role: str
    ) -> None:
        super().__init__(
            delegate.gripper_model,
            algorithm_cfg=delegate.algorithm_cfg,
            collision_cfg=delegate.collision_cfg,
            annotation_cfg=delegate.annotation_cfg,
        )
        self._geometry = geometry
        self._role = role
        self._validators: dict[tuple, HandCollisionChecker] = {}

    def _backend(
        self, mesh_vertices: torch.Tensor, mesh_triangles: torch.Tensor
    ) -> Any:
        # Confine the private GraspKit adapter to this module and a private cache.
        backend = super()._backend(mesh_vertices, mesh_triangles)
        key = self._geometry_key(mesh_vertices, mesh_triangles)
        if key not in self._validators:
            self._validators[key] = HandCollisionChecker(
                self._geometry, TriangleTarget(mesh_vertices, mesh_triangles)
            )
            backend._collision_checker = _ProposalQuery()
        return backend

    def get_dual_arm_valid_grasp_poses(self, **kwargs: Any) -> Any:
        proposals = super().get_dual_arm_valid_grasp_poses(**kwargs)
        key = self._geometry_key(kwargs["mesh_vertices"], kwargs["mesh_triangles"])
        checker = self._validators[key]
        output = []
        for row, proposal in enumerate(proposals):
            if proposal is None:
                output.append(None)
                continue
            item = proposal[self._role]
            poses = item["grasp_poses"]
            empty = {
                "is_success": False,
                "grasp_poses": poses.new_empty((0, 4, 4)),
                "open_lengths": poses.new_empty(0),
                "total_cost": poses.new_empty(0),
            }
            result = {"left": dict(empty), "right": dict(empty)}
            if not item["is_success"]:
                output.append(result)
                continue
            widths, costs = item["open_lengths"], item["total_cost"]
            order = torch.argsort(costs)
            order = order[torch.isfinite(costs[order])]
            accepted_poses, accepted_widths, accepted_costs = [], [], []
            # Match the shared action's bounded reachability population, but
            # validate every returned candidate before it reaches that action.
            for start in range(0, len(order), 32):
                ids = order[start : start + 32]
                unresolved = torch.ones(len(ids), dtype=torch.bool, device=poses.device)
                _, pads = self._geometry.at_width(widths[ids])
                pad_length = (pads[..., 2].amax(-1) - pads[..., 2].amin(-1)).amin(-1)
                insertion = (
                    pads[..., 2].amax(-1).amin(-1) - 0.1 * pad_length
                ).clamp_min(0)
                # Prefer fingertip contact with a pad-length-relative reserve;
                # retain middle/deep alternatives for other object geometries.
                for fraction in (1.0, 0.5, 0.0):
                    subset = torch.where(unresolved)[0]
                    if not len(subset):
                        break
                    selected = ids[subset]
                    trial = poses[selected].clone()
                    trial[:, :3, 3] -= (
                        trial[:, :3, 2] * (fraction * insertion[subset])[:, None]
                    )
                    fitted, gaps = checker.fit_contacts(
                        kwargs["obj_poses"][row], trial, widths[selected]
                    )
                    rejected, _ = checker.query(
                        kwargs["obj_poses"][row],
                        fitted,
                        gaps,
                        is_filter_ground_collision=self.collision_cfg.filter_ground_collision,
                    )
                    kept = ~rejected
                    if kept.any():
                        accepted_poses.append(fitted[kept])
                        accepted_widths.append(gaps[kept])
                        accepted_costs.append(costs[selected[kept]])
                        unresolved[subset[kept]] = False
                if sum(len(value) for value in accepted_poses) >= 32:
                    break
            if accepted_poses:
                combined_costs = torch.cat(accepted_costs)
                chosen = torch.argsort(combined_costs)[:32]
                result[self._role] = {
                    "is_success": True,
                    "grasp_poses": torch.cat(accepted_poses)[chosen],
                    "open_lengths": torch.cat(accepted_widths)[chosen],
                    "total_cost": combined_costs[chosen],
                }
            output.append(result)
        return output

    def get_valid_grasp_poses(self, **kwargs: Any) -> Any:
        raise RuntimeError(
            "The private E5 proposal sampler cannot serve single-arm calls."
        )

    get_best_grasp_poses = get_valid_grasp_poses
    get_grasp_candidates = get_valid_grasp_poses


class _DualSampler:
    def __init__(self, left: _HandSampler, right: _HandSampler) -> None:
        self.left, self.right = left, right

    def get_dual_arm_valid_grasp_poses(self, **kwargs: Any) -> Any:
        device = kwargs["mesh_vertices"].device
        cuda = [device.index or 0] if device.type == "cuda" else []
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device) if cuda else None
        left = self.left.get_dual_arm_valid_grasp_poses(**kwargs)
        with torch.random.fork_rng(devices=cuda):
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
            right = self.right.get_dual_arm_valid_grasp_poses(**kwargs)
        return [
            None if l is None or r is None else {"left": l["left"], "right": r["right"]}
            for l, r in zip(left, right, strict=True)
        ]


class CoordinatedGraspGenerator(ParallelJawGraspPoseGenerator):
    """Select an E5-only service; single-arm calls retain their original delegate."""

    def __init__(
        self, delegate: ParallelJawGraspPoseGenerator, factory: Callable[[], Any]
    ) -> None:
        super().__init__(delegate.gripper_model)
        self._delegate = delegate
        self._factory = lru_cache(maxsize=1)(factory)

    def get_valid_grasp_poses(self, **kwargs: Any) -> Any:
        return self._delegate.get_valid_grasp_poses(**kwargs)

    def get_best_grasp_poses(self, **kwargs: Any) -> Any:
        return self._delegate.get_best_grasp_poses(**kwargs)

    def get_grasp_candidates(self, **kwargs: Any) -> Any:
        return self._delegate.get_grasp_candidates(**kwargs)

    def get_dual_arm_valid_grasp_poses(self, **kwargs: Any) -> Any:
        return self._factory().get_dual_arm_valid_grasp_poses(**kwargs)


def install_coordinated_grasps(
    registration: Any, robot: Any, generators: dict[str, Any]
) -> dict[str, Any]:
    """Resolve named deployment commands lazily and isolate dual-only geometry."""
    profile = registration.robot_profile_binding
    resources = {item.resource_id: item for item in profile.resources}
    commands = {item.preset_id: item.commands for item in profile.command_presets}
    hands = []
    for role in ("left", "right"):
        endpoints = {item.endpoint_id: item for item in resources[role].endpoints}
        hands.append((endpoints["motion"].control_part, endpoints["grasp"]))

    if not all(
        isinstance(generators[endpoint.control_part], AntipodalGraspPoseGenerator)
        and generators[endpoint.control_part].gripper_model.model_id
        == "robotiq_arg2f_140"
        for _, endpoint in hands
    ):
        return dict(generators)

    @lru_cache(maxsize=1)
    def factory() -> _DualSampler:
        samplers = []
        for role, (motion, endpoint) in zip(("left", "right"), hands, strict=True):
            delegate = generators[endpoint.control_part]
            geometry = build_hand_geometry(
                robot,
                motion_part=motion,
                hand_part=endpoint.control_part,
                commands=commands[endpoint.command_preset],
            )
            samplers.append(_HandSampler(delegate, geometry, role))
        return _DualSampler(*samplers)

    result = dict(generators)
    for _, endpoint in hands:
        name = endpoint.control_part
        result[name] = CoordinatedGraspGenerator(generators[name], factory)
    return result
