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

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from .batch_collision_checker import BatchConvexCollisionChecker
import torch


@dataclass
class SimpleGripperCollisionCfg:
    """ Configuration for the SimpleGripperCollisionChecker. This class defines various parameters related to the gripper geometry, point cloud generation, and collision checking process. Users can customize these parameters based on the specific gripper being modeled and the requirements of the application."""

    max_open_length: float = 0.1
    """ Maximum opening length of the gripper fingers. This should be set according to the specific gripper being modeled, and it defines the maximum distance between the two fingers when fully open."""
    finger_length: float = 0.16
    """ Length of the gripper fingers from the root to the tip. This should be set according to the specific gripper being modeled, and it defines how far the fingers extend from the gripper root frame."""
    y_thickness: float = 0.03
    """ Thickness of the gripper along the Y-axis (the axis perpendicular to the finger opening direction). This should be set according to the specific gripper being modeled, and it defines the width of the gripper's main body and fingers in the Y direction."""
    x_thickness: float = 0.01
    """ Thickness of the gripper along the X-axis (the axis parallel to the finger opening direction). This should be set according to the specific gripper being modeled, and it defines the thickness of the fingers and the root in the X direction."""
    root_z_width: float = 0.06
    """ Width of the gripper root along the Z-axis (the axis along the finger length direction). This should be set according to the specific gripper being modeled, and it defines how far the root extends along the Z direction."""
    device = torch.device("cpu")
    """ Device on which the gripper point cloud will be generated and processed. This should be set according to the computational resources available and the requirements of the application. For example, if using a GPU for collision checking, this should be set to torch.device('cuda'). """
    rough_dense: float = 0.015
    """ Approximate number of points per unit length for the gripper point cloud. Higher values will yield denser point clouds, which can improve collision checking accuracy but also increase computational cost. This should be set based on the desired balance between accuracy and efficiency for the specific application."""
    max_decomposition_hulls: int = 16
    """ Maximum number of convex hulls to decompose the object mesh into for collision checking. This should be set based on the complexity of the object geometry and the desired accuracy of collision checking. More hulls can provide a tighter approximation of the object shape but will increase computational cost."""
    open_check_margin: float = 0.01
    """ Additional margin added to the gripper open length when checking for collisions. This can help account for uncertainties in the gripper pose or object geometry, and can be set based on the specific requirements of the application."""


class SimpleGripperCollisionChecker:
    def __init__(
        self,
        object_mesh_verts: torch.Tensor,
        object_mesh_faces: torch.Tensor,
        cfg: SimpleGripperCollisionCfg = SimpleGripperCollisionCfg(),
    ):
        self._checker = BatchConvexCollisionChecker(
            base_mesh_verts=object_mesh_verts,
            base_mesh_faces=object_mesh_faces,
            max_decomposition_hulls=cfg.max_decomposition_hulls,
        )
        self.device = object_mesh_verts.device
        self.cfg = cfg
        self._init_pc_template()

    def _init_pc_template(self):
        self.root_template = box_surface_grid(
            size=(
                self.cfg.max_open_length,
                self.cfg.y_thickness,
                self.cfg.root_z_width,
            ),
            dense=self.cfg.rough_dense,
            device=self.device,
        )
        self.left_template = box_surface_grid(
            size=(self.cfg.x_thickness, self.cfg.y_thickness, self.cfg.finger_length),
            dense=self.cfg.rough_dense,
            device=self.device,
        )
        self.right_template = box_surface_grid(
            size=(self.cfg.x_thickness, self.cfg.y_thickness, self.cfg.finger_length),
            dense=self.cfg.rough_dense,
            device=self.device,
        )

    def _get_gripper_pc(
        self, grasp_poses: torch.Tensor, open_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            grasp_poses: [B, 4, 4] homogeneous transformation matrix of the gripper root frame.
            open_lengths: [B] opening length of the gripper fingers.
        Returns:
            gripper_pc: [B, P, 3] point cloud of the gripper in the world frame.
        """

        root_grasp_poses = grasp_poses.clone()
        root_grasp_poses[:, :3, 3] -= (
            root_grasp_poses[:, :3, 2]
            * 0.5
            * (self.cfg.finger_length + self.cfg.root_z_width)
        )
        open_lengths_repeat = (
            open_lengths[:, None] + self.cfg.open_check_margin
        ).repeat(1, 3)
        left_finger_poses = grasp_poses.clone()
        left_finger_poses[:, :3, 3] -= left_finger_poses[:, :3, 0] * open_lengths_repeat

        right_finger_poses = grasp_poses.clone()
        right_finger_poses[:, :3, 3] += (
            right_finger_poses[:, :3, 0] * open_lengths_repeat
        )

        root_pc = transform_points_batch(self.root_template, root_grasp_poses)
        left_pc = transform_points_batch(self.left_template, left_finger_poses)
        right_pc = transform_points_batch(self.right_template, right_finger_poses)
        gripper_pc = torch.cat([root_pc, left_pc, right_pc], dim=1)
        return gripper_pc

    def query(
        self,
        obj_pose: torch.Tensor,
        grasp_poses: torch.Tensor,
        open_lengths: torch.Tensor,
        collision_threshold: float = 0.0,
        is_visual: bool = False,
    ) -> torch.Tensor:
        inv_obj_pose = obj_pose.clone()
        inv_obj_pose[:3, :3] = obj_pose[:3, :3].T
        inv_obj_pose[:3, 3] = -obj_pose[:3, 3] @ obj_pose[:3, :3]
        inv_obj_poses = inv_obj_pose[None, :, :].repeat(grasp_poses.shape[0], 1, 1)
        grasp_relative_pose = torch.bmm(inv_obj_poses, grasp_poses)
        gripper_pc = self._get_gripper_pc(grasp_relative_pose, open_lengths)
        return self._checker.query_batch_points(
            gripper_pc, collision_threshold=collision_threshold, is_visual=is_visual
        )


def transform_points_batch(
    points: torch.Tensor, poses: torch.Tensor  # [P, 3]  # [B, 4, 4]
) -> torch.Tensor:
    """
    Apply a batch of rigid transforms to a point cloud.

    Args:
        points: [P, 3] source point cloud.
        poses: [B, 4, 4] batch of homogeneous transformation matrices.

    Returns:
        transformed: [B, P, 3] transformed point cloud for each pose.
    """
    R = poses[:, :3, :3]  # [B, 3, 3]
    t = poses[:, :3, 3]  # [B, 3]
    transformed = torch.einsum("bij, pj -> bpi", R, points) + t.unsqueeze(1)
    return transformed


def box_surface_grid(
    size: Sequence[float] | torch.Tensor,
    dense: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Generate grid-sampled points on the surface of an axis-aligned box.

    Six faces of the box are each sampled independently on a regular 2-D grid.
    Grid resolution per face is derived automatically from ``dense``:
    the number of sample points along an edge of length *L* is
    ``max(2, round(L * dense) + 1)``, so ``dense`` behaves as
    *approximate samples per unit length*.

    Edge and corner points are shared across adjacent faces and are included
    exactly once (no duplicates).

    Args:
        size: Box dimensions ``(sx, sy, sz)``.  Accepts a sequence of three
            floats or a 1-D :class:`torch.Tensor` of length 3.
        dense: Approximate number of grid sample points per unit length along
            each edge.  Higher values yield denser point clouds.
        device: Target PyTorch device for the returned tensor.

    Returns:
        Float tensor of shape ``(N, 3)`` containing surface points expressed
        in the box's local frame (origin at the box centre).

    Example:
        >>> pts = box_surface_grid((0.1, 0.06, 0.03), dense=200.0)
        >>> pts.shape
        torch.Size([..., 3])
    """
    if isinstance(size, torch.Tensor):
        sx, sy, sz = size[0].item(), size[1].item(), size[2].item()
    else:
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    # ── grid resolution per axis (at least 2 points to span the full edge) ──
    nx = max(2, round(sx / dense) + 1)
    ny = max(2, round(sy / dense) + 1)
    nz = max(2, round(sz / dense) + 1)

    xs = torch.linspace(-hx, hx, nx, device=device)
    ys = torch.linspace(-hy, hy, ny, device=device)
    zs = torch.linspace(-hz, hz, nz, device=device)

    # Interior slices (exclude first and last to avoid duplicate edges)
    xs_inner = xs[1:-1]  # length nx-2
    ys_inner = ys[1:-1]  # length ny-2

    def _grid(
        u: torch.Tensor, v: torch.Tensor, axis: int, offset: float
    ) -> torch.Tensor:
        """Build a flat (M, 3) tensor for one face grid.

        Args:
            u: 1-D tensor of coordinates along the first in-plane axis.
            v: 1-D tensor of coordinates along the second in-plane axis.
            axis: Normal axis of the face — 0 (±X), 1 (±Y), or 2 (±Z).
            offset: Signed half-extent along ``axis``.

        Returns:
            Tensor of shape ``(len(u) * len(v), 3)``.
        """
        uu, vv = torch.meshgrid(u, v, indexing="ij")
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)
        cc = torch.full_like(uu, offset)
        if axis == 0:
            return torch.stack([cc, uu, vv], dim=-1)
        elif axis == 1:
            return torch.stack([uu, cc, vv], dim=-1)
        else:
            return torch.stack([uu, vv, cc], dim=-1)

    # ─────────────────────────────────────────────────────────────────────────
    # Build 6 faces.  To avoid duplicate points on shared edges/corners:
    #   ±X faces  → full  NY × NZ  grids
    #   ±Y faces  → (NX-2) × NZ   grids  (x-edges owned by ±X faces)
    #   ±Z faces  → (NX-2) × (NY-2) grids  (x- and y-edges owned above)
    # ─────────────────────────────────────────────────────────────────────────
    faces: list[torch.Tensor] = [
        _grid(ys, zs, axis=0, offset=-hx),  # −X face  (NY × NZ)
        _grid(ys, zs, axis=0, offset=+hx),  # +X face  (NY × NZ)
        _grid(xs_inner, zs, axis=1, offset=-hy),  # −Y face  ((NX-2) × NZ)
        _grid(xs_inner, zs, axis=1, offset=+hy),  # +Y face  ((NX-2) × NZ)
        _grid(xs_inner, ys_inner, axis=2, offset=-hz),  # −Z face
        _grid(xs_inner, ys_inner, axis=2, offset=+hz),  # +Z face
    ]

    return torch.cat(faces, dim=0)
