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

"""Sampled hand contact fitting and discrete target-mesh qualification.

No simulator, task graph or robot is owned here. Geometry must be expressed in
its declared TCP frame; callers keep sample tensors unchanged while in use.
This is neither continuous collision detection nor physical grasp evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np
import torch
import trimesh

from embodichain.utils import logger

__all__ = [
    "SampledHandGeometry",
    "GraspSurfaceQuery",
    "TriangleSurfaceQuery",
    "HandContactChecker",
]


class GraspSurfaceQuery(Protocol):
    """Signed object-frame distances and outward unit normals for grasp queries."""

    def query_batch_points(
        self, points: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return collision masks and signed distances for object-frame points."""
        ...

    def query_surface_normals(self, points: torch.Tensor) -> torch.Tensor:
        """Return outward unit normals at the nearest surfaces, or zero if ambiguous."""
        ...


@dataclass(frozen=True)
class SampledHandGeometry:
    """URDF surface samples from open to closed, expressed in the configured TCP.

    Pad order is negative/positive TCP X. Arrays are caller-owned read-only
    inputs; interpolated results do not alias them. This is input geometry,
    not continuous collision detection or evidence of physical attachment.

    Args:
        body: Body samples with shape ``(G, V, 3)``.
        pads: Pad samples with shape ``(G, 2, P, 3)``.
        gaps: Finite strictly descending pad gaps, shape ``(G,)``.
    """

    body: torch.Tensor
    pads: torch.Tensor
    gaps: torch.Tensor

    def __post_init__(self) -> None:
        """Validate fixed input samples once, outside the candidate hot path."""
        if not all(
            isinstance(value, torch.Tensor)
            for value in (self.body, self.pads, self.gaps)
        ):
            raise TypeError("Hand samples must be tensors.")
        if self.body.ndim != 3 or self.pads.ndim != 4 or self.gaps.ndim != 1:
            raise ValueError("Invalid hand sample dimensions.")
        if (
            len(self.gaps) < 2
            or self.body.shape[0] != len(self.gaps)
            or self.pads.shape[0] != len(self.gaps)
        ):
            raise ValueError("Hand surfaces must share at least two gap samples.")
        if (
            self.body.shape[-1] != 3
            or self.pads.shape[1] != 2
            or self.pads.shape[-1] != 3
            or not self.body.shape[1]
            or not self.pads.shape[2]
        ):
            raise ValueError("Hand samples require a body and two nonempty XYZ pads.")
        if (
            not all(
                torch.isfinite(value).all()
                for value in (self.body, self.pads, self.gaps)
            )
            or not (self.gaps[:-1] > self.gaps[1:]).all()
        ):
            raise ValueError(
                "Hand samples must be finite with strictly descending gaps."
            )

    def at_width(self, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate hand geometry at full pad gaps.

        Args:
            widths: Candidate gaps of shape ``(K,)``, in metres.

        Returns:
            Body and pad samples in the widths' dtype and device. Values outside
            the sampled gap range clamp to its endpoints; eligibility is checked
            separately by the contact checker.
        """
        if widths.ndim != 1:
            raise ValueError("Candidate gaps must be a one-dimensional tensor.")
        gaps = self.gaps.to(widths)
        high = torch.searchsorted(-gaps.contiguous(), -widths).clamp(1, len(gaps) - 1)
        low = high - 1
        alpha = ((gaps[low] - widths) / (gaps[low] - gaps[high])).clamp(0, 1)
        body, pads = self.body.to(widths), self.pads.to(widths)
        return (
            torch.lerp(body[low], body[high], alpha[:, None, None]),
            torch.lerp(pads[low], pads[high], alpha[:, None, None, None]),
        )


class TriangleSurfaceQuery:
    """Signed triangle distances preserve cavities lost by a convex approximation."""

    def __init__(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        *,
        normal_probe_distance: float,
    ) -> None:
        """Build a CPU signed-distance oracle for a closed target mesh.

        Args:
            vertices: Object-frame mesh vertices in metres.
            faces: Triangular vertex indices.
            normal_probe_distance: Explicit positive occupancy-probe offset.

        Raises:
            ValueError: If the probe or closed/winding-consistent mesh is invalid.
        """
        import open3d as o3d

        if (
            isinstance(normal_probe_distance, bool)
            or not math.isfinite(normal_probe_distance)
            or normal_probe_distance <= 0
        ):
            raise ValueError("normal_probe_distance must be finite and positive.")
        self.normal_probe_distance = float(normal_probe_distance)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not vertices.shape[0]
            or not torch.isfinite(vertices).all()
        ):
            raise ValueError("Target vertices must be finite XYZ rows.")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype not in (torch.int32, torch.int64)
            or not faces.shape[0]
            or (faces < 0).any()
            or (faces >= len(vertices)).any()
        ):
            raise ValueError(
                "Target faces must contain valid integer triangle indices."
            )
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

    def query_surface_normals(self, points: torch.Tensor) -> torch.Tensor:
        """Orient nearest triangle normals using local solid occupancy.

        Args:
            points: Object-frame float32 points with final dimension three.

        Returns:
            Outward normals in the input shape/device. Ambiguous equal occupancy
            on both sides produces a zero normal, which cannot pass a facing gate.
        """
        import open3d as o3d

        if not points.numel():
            return torch.empty_like(points)
        flat = points.detach().reshape(-1, 3).cpu().float().contiguous()
        closest = self.scene.compute_closest_points(
            o3d.core.Tensor(flat.numpy()), nthreads=2
        )
        normals = closest["primitive_normals"].numpy()
        surface = closest["points"].numpy()
        # Orient locally using the same solid-interior convention as signed
        # distances. Total mesh volume cannot orient independent or nested shells.
        offset = normals * self.normal_probe_distance
        probes = np.stack((surface + offset, surface - offset), axis=1)
        occupied = self.scene.compute_occupancy(
            o3d.core.Tensor(probes.astype(np.float32)), nthreads=2, nsamples=3
        ).numpy()
        sign = occupied[:, 1] - occupied[:, 0]
        # Equal occupancy on both sides is ambiguous: a zero normal fails the gate.
        return torch.from_numpy(normals * sign[:, None]).to(points).reshape_as(points)

    def query_batch_points(
        self, points: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query source-compatible signed distances using a cached triangle BVH.

        Args:
            points: Object-frame float32 points with final dimension three.
            **kwargs: Compatibility checker keywords; callers use zero threshold.

        Returns:
            Nonpositive-distance flags and signed distances in the input
            shape/device. Far points retain the conservative bounding-box distance.
        """
        import open3d as o3d

        if kwargs.get("collision_threshold", 0.0) != 0.0:
            raise ValueError(
                "Signed triangle queries require zero collision threshold."
            )
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


class HandContactChecker:
    """Qualify sampled hand closing surfaces against one signed object query."""

    def __init__(
        self,
        geometry: SampledHandGeometry,
        target: GraspSurfaceQuery,
        *,
        contact_tolerance: float,
        minimum_normal_cosine: float,
    ) -> None:
        """Bind read-only geometry and explicit contact qualification thresholds.

        Args:
            geometry: Open-to-closed geometry samples in the declared TCP frame.
            target: Signed object-frame surface query service.
            contact_tolerance: Positive contact proximity tolerance in metres.
            minimum_normal_cosine: Required outward-normal alignment in [0, 1].

        Raises:
            ValueError: If a qualification threshold is invalid.
        """
        if (
            isinstance(contact_tolerance, bool)
            or not math.isfinite(contact_tolerance)
            or contact_tolerance <= 0
        ):
            raise ValueError("contact_tolerance must be finite and positive.")
        if (
            isinstance(minimum_normal_cosine, bool)
            or not math.isfinite(minimum_normal_cosine)
            or not 0 <= minimum_normal_cosine <= 1
        ):
            raise ValueError("minimum_normal_cosine must lie in [0, 1].")
        self.contact_tolerance = float(contact_tolerance)
        self.minimum_normal_cosine = float(minimum_normal_cosine)
        self.last_counts: dict[str, int] = {}
        self.geometry = geometry
        self.target = target

    def fit_contacts(
        self, obj_pose: torch.Tensor, poses: torch.Tensor, widths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fit finite pad surfaces through the source's eight bounded iterations.

        Args:
            obj_pose: Object-to-world transform for this target.
            poses: Candidate TCP-to-world transforms, shape ``(K, 4, 4)``.
            widths: Proposed pad gaps, shape ``(K,)``.

        Returns:
            Independent corrected poses and gaps. The correction stays inside the
            original finite pad neighborhood; acceptance still requires ``query``.
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
        """Check the contact endpoint and discrete open-to-contact samples.

        Args:
            obj_pose: Object-to-world transform, shape ``(4, 4)`` or ``(1, 4, 4)``.
            grasp_poses: Candidate TCP-to-world transforms, shape ``(K, 4, 4)``.
            open_lengths: Candidate pad gaps in metres, shape ``(K,)``.
            **kwargs: Existing checker policy keywords; ground filtering and a
                nonzero collision threshold are unsupported.

        Returns:
            Collision flags and source-compatible diagnostic distances. Each
            distance belongs to that candidate's most recently checked sweep
            sample; it is not the minimum over the complete path. Distances
            for candidates rejected before querying have no clearance meaning.
        """
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
            "contact_orientation": 0,
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
                    touching = (nearest.abs() <= self.contact_tolerance).all(-1)
                    counts["contact_missing"] += int((~touching).sum())
                    valid[indices[selected]] &= touching
                    # Point proximity alone also accepts two pads grazing the
                    # same tray face. Require material facing each inward pad.
                    contacts = (
                        _inward_faces(pads)[selected]
                        & (pad_distances.abs() <= self.contact_tolerance)
                        & touching[:, None, None]
                    )
                    facing = torch.zeros_like(contacts)
                    if contacts.any():
                        pad_points = points[:, body.shape[1] :].reshape(
                            -1, 2, pads.shape[2], 3
                        )
                        normals = self.target.query_surface_normals(
                            pad_points[contacts]
                        )
                        axis = transforms[:, :3, 0]
                        outward = torch.stack((-axis, axis), dim=1)
                        outward = outward[:, :, None].expand_as(pad_points)
                        facing[contacts] = (
                            torch.isfinite(normals).all(-1)
                            & (normals != 0).any(-1)
                            & (
                                (normals * outward[contacts]).sum(-1)
                                >= self.minimum_normal_cosine
                            )
                        )
                    opposed = facing.any(-1).all(-1)
                    counts["contact_orientation"] += int((touching & ~opposed).sum())
                    valid[indices[selected]] &= opposed
                limit = torch.where(allowed, -self.contact_tolerance, -1e-6)
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
