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

import torch
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from embodichain.toolkits.graspkit.pg_grasp import (
    GraspGenerator,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.common import BatchEntity
    from embodichain.lab.sim.objects import Articulation


@dataclass
class Affordance:
    """Base class for affordance data.

    Represents an object's interaction possibilities. Subclasses carry whatever
    typed fields they need (mesh tensors, interaction points, etc.); the base
    class only carries an object label and a free-form custom_config dict.
    """

    object_label: str = ""
    """Label of the object this affordance belongs to."""

    custom_config: dict[str, Any] = field(default_factory=dict)
    """User-defined configuration payload."""

    def set_custom_config(self, key: str, value: Any) -> None:
        """Set a custom affordance configuration value."""
        self.custom_config[key] = value

    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """Get a custom affordance configuration value."""
        return self.custom_config.get(key, default)

    def get_batch_size(self) -> int:
        """Return the batch size of this affordance data."""
        return 1


@dataclass
class AntipodalAffordance(Affordance):
    """Antipodal grasp affordance for parallel-jaw grippers.

    Geometry may be supplied directly as a triangle mesh, or resolved from one
    link of an articulation.  The articulation form deliberately uses the
    simulation object's public geometry API instead of parsing its source URDF,
    so the sampled mesh matches the geometry instantiated by the simulator.
    """

    mesh_vertices: torch.Tensor | None = None
    """Object mesh vertices, shape [N, 3]."""

    mesh_triangles: torch.Tensor | None = None
    """Object mesh triangle indices, shape [M, 3]."""

    articulation: Articulation | None = None
    """Optional articulation whose link supplies the grasp mesh and live pose."""

    link_name: str | None = None
    """Articulation link passed to ``get_link_vert_face``/``get_link_pose``."""

    generator_cfg: GraspGeneratorCfg | None = None
    """Optional grasp-generator configuration."""

    gripper_collision_cfg: GripperCollisionCfg | None = None
    """Optional gripper-collision configuration."""

    force_reannotate: bool = False
    """If True, recompute the grasp annotation on each access."""

    _generator: GraspGenerator | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve articulation-link geometry while preserving mesh input."""
        has_articulation = self.articulation is not None
        has_link_name = self.link_name is not None
        if has_articulation != has_link_name:
            raise ValueError(
                "articulation and link_name must be provided together for an "
                "articulation AntipodalAffordance."
            )
        if not has_articulation:
            return
        if not isinstance(self.link_name, str) or not self.link_name.strip():
            raise ValueError("link_name must be a non-empty string.")
        if self.mesh_vertices is not None or self.mesh_triangles is not None:
            raise ValueError(
                "Provide either articulation + link_name or mesh_vertices + "
                "mesh_triangles, not both."
            )
        vertices, triangles = self.articulation.get_link_vert_face(self.link_name)
        self.mesh_vertices = torch.as_tensor(vertices)
        self.mesh_triangles = torch.as_tensor(triangles)

    @property
    def is_articulation(self) -> bool:
        """Whether geometry and pose are backed by an articulation link."""
        return self.articulation is not None and self.link_name is not None

    def get_articulation_link_pose(self) -> torch.Tensor:
        """Return the current batched world pose of the configured link."""
        if not self.is_articulation:
            raise ValueError(
                "This AntipodalAffordance is not backed by an articulation link."
            )
        return self.articulation.get_link_pose(self.link_name, to_matrix=True)

    def _init_generator(self) -> None:
        if self.mesh_vertices is None or self.mesh_triangles is None:
            logger.log_error(
                "mesh_vertices and mesh_triangles must be provided to initialize "
                "AntipodalAffordance.",
                ValueError,
            )
        self._generator = GraspGenerator(
            vertices=self.mesh_vertices,
            triangles=self.mesh_triangles,
            cfg=self.generator_cfg,
            gripper_collision_cfg=self.gripper_collision_cfg,
        )
        if self.force_reannotate or self._generator._hit_point_pairs is None:
            self._generator.annotate()

    def _resolve_approach_direction(
        self, approach_direction: torch.Tensor
    ) -> torch.Tensor:
        """Move the approach direction to the grasp generator device."""
        return approach_direction.to(
            device=self._generator.device,
            dtype=torch.float32,
        )

    def get_valid_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
        object_part: str = "center",
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        results = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_poses, _, costs = self._generator.get_valid_grasp_poses(
                object_pose=obj_pose,
                approach_direction=approach_direction,
                object_part=object_part,
            )
            if grasp_poses.shape == (4, 4):
                grasp_poses = grasp_poses.unsqueeze(0)
            if costs.dim() == 0:
                costs = costs.unsqueeze(0)
            if not is_success:
                logger.log_warning(
                    f"Failed to find valid grasp poses for {i}-th object."
                )
                costs = torch.full(
                    (grasp_poses.shape[0],),
                    torch.inf,
                    dtype=torch.float32,
                    device=grasp_poses.device,
                )
            results.append((grasp_poses, costs))
        return results

    def get_dual_arm_valid_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
        middle_empty_ratio: float = 0.4,
    ) -> list[dict | None]:
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        results = []
        for i, obj_pose in enumerate(obj_poses):
            result = self._generator.get_dual_arm_valid_grasp_poses(
                object_pose=obj_pose,
                approach_direction=approach_direction,
                left_to_right_arm_direction=left_to_right_arm_direction,
                middle_empty_ratio=middle_empty_ratio,
            )
            results.append(result)
        return results

    def get_best_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._generator is None:
            self._init_generator()
        approach_direction = self._resolve_approach_direction(approach_direction)
        grasp_xpos_list: list[torch.Tensor] = []
        is_success_list: list[bool] = []
        open_length_list: list[float] = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_xpos, open_length = self._generator.get_grasp_poses(
                obj_pose, approach_direction
            )
            if is_success:
                grasp_xpos_list.append(grasp_xpos.unsqueeze(0))
            else:
                logger.log_warning(f"No valid grasp pose found for {i}-th object.")
                grasp_xpos_list.append(
                    torch.eye(
                        4, dtype=torch.float32, device=self._generator.device
                    ).unsqueeze(0)
                )
            is_success_list.append(is_success)
            open_length_list.append(open_length)
        is_success_t = torch.tensor(
            is_success_list, dtype=torch.bool, device=self._generator.device
        )
        grasp_xpos = torch.concatenate(grasp_xpos_list, dim=0)
        open_length_t = torch.tensor(
            open_length_list, dtype=torch.float32, device=self._generator.device
        )
        return is_success_t, grasp_xpos, open_length_t


@dataclass
class TurnAffordance(Affordance):
    """Geometry and rotation semantics for one articulation link knob."""

    articulation: Articulation | None = None
    """Articulation whose link supplies the knob mesh and live pose."""

    link_name: str = ""
    """Articulation link passed to ``get_link_vert_face``/``get_link_pose``."""

    turn_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 1.0, 0.0])
    )
    """Knob rotation axis expressed in the articulation-link frame."""

    mesh_vertices: torch.Tensor = field(init=False, repr=False)
    """Link-local mesh vertices returned by ``get_link_vert_face``."""

    mesh_triangles: torch.Tensor = field(init=False, repr=False)
    """Link-local mesh triangles returned by ``get_link_vert_face``."""

    def __post_init__(self) -> None:
        articulation = self.articulation
        if articulation is None:
            raise ValueError("TurnAffordance.articulation must be provided.")
        if not isinstance(self.link_name, str) or not self.link_name.strip():
            raise ValueError("TurnAffordance.link_name must be a non-empty string.")
        if (
            not isinstance(self.turn_axis, torch.Tensor)
            or self.turn_axis.shape != (3,)
            or not torch.isfinite(self.turn_axis).all()
        ):
            raise ValueError("TurnAffordance.turn_axis must be a finite (3,) tensor.")
        if torch.linalg.vector_norm(self.turn_axis) <= 1.0e-6:
            raise ValueError("TurnAffordance.turn_axis must be non-zero.")
        self.turn_axis = self.turn_axis.clone()
        vertices, triangles = articulation.get_link_vert_face(self.link_name)
        self.mesh_vertices = torch.as_tensor(vertices)
        self.mesh_triangles = torch.as_tensor(triangles)
        if self.mesh_vertices.dim() != 2 or self.mesh_vertices.shape[1] != 3:
            raise ValueError("TurnAffordance mesh vertices must have shape (N, 3).")
        if self.mesh_vertices.shape[0] == 0:
            raise ValueError("TurnAffordance requires a non-empty link mesh.")

    def get_link_pose(self) -> torch.Tensor:
        """Return the current batched world pose of the configured link."""
        articulation = self.articulation
        if articulation is None:
            raise RuntimeError("TurnAffordance has no articulation.")
        return articulation.get_link_pose(self.link_name, to_matrix=True)

    def get_grasp_pose(self, link_pose: torch.Tensor | None = None) -> torch.Tensor:
        """Construct the deterministic grasp pose at the link mesh center.

        The pose z-axis follows :attr:`turn_axis` transformed into the world
        frame, while its y-axis is fixed to world ``(0, 0, 1)``.

        Returns:
            Batched world-frame grasp poses with shape ``(B, 4, 4)``.

        Raises:
            ValueError: If the world turn axis is parallel to the fixed y-axis.
        """
        link_pose = self.get_link_pose() if link_pose is None else link_pose
        if link_pose.dim() != 3 or link_pose.shape[1:] != (4, 4):
            raise ValueError("Articulation link pose must have shape (B, 4, 4).")
        link_pose = link_pose.to(dtype=torch.float32)
        device = link_pose.device
        center = self.mesh_vertices.to(device=device, dtype=torch.float32).mean(dim=0)
        turn_axis = self.turn_axis.to(device=device, dtype=torch.float32)
        turn_axis = turn_axis / torch.linalg.vector_norm(turn_axis)

        z_axis = torch.matmul(link_pose[:, :3, :3], turn_axis)
        z_axis = torch.nn.functional.normalize(z_axis, dim=1)
        y_axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=device
        ).expand_as(z_axis)
        x_axis = torch.linalg.cross(y_axis, z_axis, dim=1)
        if torch.any(torch.linalg.vector_norm(x_axis, dim=1) <= 1.0e-6):
            raise ValueError(
                "TurnAffordance turn axis must not be parallel to world (0, 0, 1)."
            )
        x_axis = torch.nn.functional.normalize(x_axis, dim=1)

        grasp_pose = torch.eye(4, dtype=torch.float32, device=device).repeat(
            link_pose.shape[0], 1, 1
        )
        grasp_pose[:, :3, 0] = x_axis
        grasp_pose[:, :3, 1] = y_axis
        grasp_pose[:, :3, 2] = z_axis
        grasp_pose[:, :3, 3] = (
            torch.matmul(link_pose[:, :3, :3], center) + link_pose[:, :3, 3]
        )
        return grasp_pose


@dataclass
class PressButtonAffordance(Affordance):
    """Geometry and pressing semantics for one articulation-link button."""

    articulation: Articulation | None = None
    """Articulation whose link supplies the button mesh and live pose."""

    link_name: str = ""
    """Articulation link passed to ``get_link_vert_face``/``get_link_pose``."""

    press_axis: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, 1.0])
    )
    """Press direction expressed in the articulation-link frame."""

    mesh_vertices: torch.Tensor = field(init=False, repr=False)
    """Link-local mesh vertices returned by ``get_link_vert_face``."""

    mesh_triangles: torch.Tensor = field(init=False, repr=False)
    """Link-local mesh triangles returned by ``get_link_vert_face``."""

    def __post_init__(self) -> None:
        articulation = self.articulation
        if articulation is None:
            raise ValueError("PressButtonAffordance.articulation must be provided.")
        if not isinstance(self.link_name, str) or not self.link_name.strip():
            raise ValueError(
                "PressButtonAffordance.link_name must be a non-empty string."
            )
        if (
            not isinstance(self.press_axis, torch.Tensor)
            or self.press_axis.shape != (3,)
            or not torch.isfinite(self.press_axis).all()
        ):
            raise ValueError(
                "PressButtonAffordance.press_axis must be a finite (3,) tensor."
            )
        if torch.linalg.vector_norm(self.press_axis) <= 1.0e-6:
            raise ValueError("PressButtonAffordance.press_axis must be non-zero.")
        self.press_axis = self.press_axis.clone()
        vertices, triangles = articulation.get_link_vert_face(self.link_name)
        self.mesh_vertices = torch.as_tensor(vertices)
        self.mesh_triangles = torch.as_tensor(triangles)
        if self.mesh_vertices.dim() != 2 or self.mesh_vertices.shape[1] != 3:
            raise ValueError(
                "PressButtonAffordance mesh vertices must have shape (N, 3)."
            )
        if self.mesh_vertices.shape[0] == 0:
            raise ValueError("PressButtonAffordance requires a non-empty link mesh.")

    def get_link_pose(self) -> torch.Tensor:
        """Return the current batched world pose of the configured link."""
        articulation = self.articulation
        if articulation is None:
            raise RuntimeError("PressButtonAffordance has no articulation.")
        return articulation.get_link_pose(self.link_name, to_matrix=True)

    def get_press_pose(self, link_pose: torch.Tensor | None = None) -> torch.Tensor:
        """Construct a press pose at the button surface opposite the press axis.

        The end-effector z-axis follows :attr:`press_axis` in world space. The
        position is the center of the mesh's upstream face, so advancing along
        the z-axis moves into the button instead of first targeting its volume
        center.

        Args:
            link_pose: Optional current world pose of the button link with
                shape ``(B, 4, 4)``.

        Returns:
            Batched world-frame press poses with shape ``(B, 4, 4)``.

        Raises:
            ValueError: If the world press axis is parallel to world up.
        """
        link_pose = self.get_link_pose() if link_pose is None else link_pose
        if link_pose.dim() != 3 or link_pose.shape[1:] != (4, 4):
            raise ValueError("Articulation link pose must have shape (B, 4, 4).")
        link_pose = link_pose.to(dtype=torch.float32)
        device = link_pose.device
        vertices = self.mesh_vertices.to(device=device, dtype=torch.float32)
        center = vertices.mean(dim=0)
        press_axis = self.press_axis.to(device=device, dtype=torch.float32)
        press_axis = press_axis / torch.linalg.vector_norm(press_axis)

        center_projection = torch.dot(center, press_axis)
        surface_projection = torch.min(torch.matmul(vertices, press_axis))
        surface_center = center + press_axis * (surface_projection - center_projection)

        z_axis = torch.matmul(link_pose[:, :3, :3], press_axis)
        z_axis = torch.nn.functional.normalize(z_axis, dim=1)
        y_axis = torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=device
        ).expand_as(z_axis)
        x_axis = torch.linalg.cross(y_axis, z_axis, dim=1)
        if torch.any(torch.linalg.vector_norm(x_axis, dim=1) <= 1.0e-6):
            raise ValueError(
                "PressButtonAffordance press axis must not be parallel to world "
                "(0, 0, 1)."
            )
        x_axis = torch.nn.functional.normalize(x_axis, dim=1)

        press_pose = torch.eye(4, dtype=torch.float32, device=device).repeat(
            link_pose.shape[0], 1, 1
        )
        press_pose[:, :3, 0] = x_axis
        press_pose[:, :3, 1] = y_axis
        press_pose[:, :3, 2] = z_axis
        press_pose[:, :3, 3] = (
            torch.matmul(link_pose[:, :3, :3], surface_center) + link_pose[:, :3, 3]
        )
        return press_pose


@dataclass
class InteractionPoints(Affordance):
    """Batch of 3D interaction points on an object surface."""

    points: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 3))
    """Batch of 3D interaction points with shape [B, 3]."""

    normals: torch.Tensor | None = None
    """Optional surface normals at each interaction point with shape [B, 3]."""

    point_types: list[str] = field(default_factory=list)
    """Optional labels for each point's interaction type."""

    def get_points_by_type(self, point_type: str) -> torch.Tensor | None:
        """Get points by their interaction type."""
        if point_type in self.point_types:
            indices = [i for i, t in enumerate(self.point_types) if t == point_type]
            return self.points[indices]
        return None

    def get_batch_size(self) -> int:
        """Return the number of interaction points in this affordance."""
        return self.points.shape[0]

    def get_approach_direction(self, point_idx: int) -> torch.Tensor:
        """Get recommended approach direction for a given point."""
        if self.normals is not None:
            return -self.normals[point_idx]
        return torch.tensor(
            [0, 0, 1], dtype=self.points.dtype, device=self.points.device
        )


@dataclass
class AssembleAffordance(Affordance):
    """Affordance describing how an assemble object fits onto a base object.

    The base object anchors the assembly: its world pose is read at planning
    time from :attr:`base_object_entity` so the target tracks a moved base. The
    assemble object is the part that is picked up and placed; its target pose is
    ``base_pose @ assemble_to_base_pose``.
    """

    base_object_label: str = ""
    """Label of the base object the assemble object is placed onto."""

    base_object_entity: BatchEntity | None = None
    """Simulation entity for the base object; its pose anchors the assembly."""

    assemble_object_label: str = ""
    """Label of the assemble object that is picked up and placed."""

    assemble_object_entity: BatchEntity | None = None
    """Optional simulation entity for the assemble object (reference/logging)."""

    assemble_to_base_pose: torch.Tensor = field(
        default_factory=lambda: torch.eye(4, dtype=torch.float32)
    )
    """Pose of the assemble object relative to the base object frame, shape
    ``(4, 4)`` or ``(n_envs, 4, 4)``."""

    def get_assemble_object_pose(self, base_pose: torch.Tensor) -> torch.Tensor:
        """Return the assemble-object target pose for a given base-object pose.

        The assemble object is placed at ``base_pose @ assemble_to_base_pose``.

        Args:
            base_pose: Base-object pose with shape ``(4, 4)`` or ``(n_envs, 4, 4)``.

        Returns:
            Assemble-object target pose with shape ``(n_envs, 4, 4)``.
        """
        base_pose = base_pose.to(dtype=torch.float32)
        if base_pose.dim() == 2:
            base_pose = base_pose.unsqueeze(0)
        n_envs = base_pose.shape[0]
        rel = self.assemble_to_base_pose.to(
            device=base_pose.device, dtype=torch.float32
        )
        if rel.dim() == 2:
            rel = rel.unsqueeze(0).repeat(n_envs, 1, 1)
        elif rel.shape[0] == 1:
            rel = rel.repeat(n_envs, 1, 1)
        return torch.bmm(base_pose, rel)


__all__ = [
    "Affordance",
    "AntipodalAffordance",
    "PressButtonAffordance",
    "TurnAffordance",
    "InteractionPoints",
    "AssembleAffordance",
]
