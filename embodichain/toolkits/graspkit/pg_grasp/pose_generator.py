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

"""Antipodal grasp-pose service for parallel-jaw grippers."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import math
from typing import Literal

import torch

from embodichain.toolkits.graspkit import (
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)
from embodichain.utils import configclass, logger

from ._antipodal_backend import _AntipodalMeshBackend
from .antipodal_sampler import AntipodalSamplerCfg
from .gripper_collision_checker import GripperCollisionCfg

__all__ = [
    "AntipodalGraspPoseGenerator",
    "AntipodalGraspPoseGeneratorCfg",
    "GraspAnnotationCfg",
    "ParallelJawGraspCollisionCfg",
]


def _real_number(value: float, *, field_name: str, minimum: float) -> float:
    """Return one finite real number at or above ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise ValueError(f"{field_name} must be finite and at least {minimum}.")
    return normalized


def _positive_int(value: int, *, field_name: str) -> int:
    """Return one exact positive integer."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


@configclass
class AntipodalGraspPoseGeneratorCfg:
    """Algorithm-only configuration for antipodal candidate generation."""

    sample_count: int = 20_000
    """Number of surface rays sampled while finding antipodal pairs."""

    ray_deviation_angle: float = math.pi / 12
    """Maximum random ray deviation from a sampled surface normal."""

    approach_deviation_angle: float = math.pi / 6
    """Maximum candidate deviation from the requested approach direction."""

    approach_direction_samples: int = 4
    """Number of approach-direction variants evaluated per antipodal pair."""

    max_candidates: int = 50
    """Maximum number of ranked candidates returned per object pose."""

    def __post_init__(self) -> None:
        self.sample_count = _positive_int(
            self.sample_count,
            field_name="sample_count",
        )
        self.approach_direction_samples = _positive_int(
            self.approach_direction_samples,
            field_name="approach_direction_samples",
        )
        self.max_candidates = _positive_int(
            self.max_candidates,
            field_name="max_candidates",
        )
        self.ray_deviation_angle = _real_number(
            self.ray_deviation_angle,
            field_name="ray_deviation_angle",
            minimum=0.0,
        )
        self.approach_deviation_angle = _real_number(
            self.approach_deviation_angle,
            field_name="approach_deviation_angle",
            minimum=0.0,
        )


@configclass
class ParallelJawGraspCollisionCfg:
    """Collision-check policy independent of physical gripper dimensions."""

    point_sample_density: float = 0.01
    """Sampling density passed to the parallel-jaw collision model."""

    max_decomposition_hulls: int = 16
    """Maximum convex hull count used for target-mesh decomposition."""

    opening_margin: float = 0.01
    """Additional opening used while checking finger collisions in metres."""

    filter_ground_collision: bool = True
    """Whether candidates intersecting the inferred support plane are removed."""

    def __post_init__(self) -> None:
        self.point_sample_density = _real_number(
            self.point_sample_density,
            field_name="point_sample_density",
            minimum=0.0,
        )
        if self.point_sample_density == 0.0:
            raise ValueError("point_sample_density must be positive.")
        self.max_decomposition_hulls = _positive_int(
            self.max_decomposition_hulls,
            field_name="max_decomposition_hulls",
        )
        self.opening_margin = _real_number(
            self.opening_margin,
            field_name="opening_margin",
            minimum=0.0,
        )
        if type(self.filter_ground_collision) is not bool:
            raise TypeError("filter_ground_collision must be a bool.")


@configclass
class GraspAnnotationCfg:
    """Geometry annotation and cache-refresh policy."""

    selection_mode: Literal["whole_mesh", "interactive"] = "whole_mesh"
    """Use the full mesh or select a region through the Viser frontend."""

    viser_port: int = 15531
    """Port used only by interactive region selection."""

    use_largest_connected_component: bool = False
    """Whether an interactive selection keeps only its largest component."""

    force_refresh: bool = False
    """Whether the service recomputes annotations when first seeing a mesh."""

    def __post_init__(self) -> None:
        if self.selection_mode not in ("whole_mesh", "interactive"):
            raise ValueError(
                "selection_mode must be exactly 'whole_mesh' or 'interactive'."
            )
        if type(self.viser_port) is not int or not 1 <= self.viser_port <= 65_535:
            raise ValueError("viser_port must be an integer in [1, 65535].")
        if type(self.use_largest_connected_component) is not bool:
            raise TypeError("use_largest_connected_component must be a bool.")
        if type(self.force_refresh) is not bool:
            raise TypeError("force_refresh must be a bool.")


class AntipodalGraspPoseGenerator(ParallelJawGraspPoseGenerator):
    """Reusable antipodal generator for any parallel-jaw gripper model.

    Target meshes are supplied per call. The service lazily owns one private
    single-mesh backend per tensor-backed mesh, allowing callers to reuse
    sampled annotations without placing live generator state on a scene
    affordance or exposing a second generator API.
    """

    def __init__(
        self,
        gripper_model: ParallelJawGripperModelCfg,
        *,
        algorithm_cfg: AntipodalGraspPoseGeneratorCfg | None = None,
        collision_cfg: ParallelJawGraspCollisionCfg | None = None,
        annotation_cfg: GraspAnnotationCfg | None = None,
    ) -> None:
        super().__init__(gripper_model)
        self._algorithm_cfg = deepcopy(
            AntipodalGraspPoseGeneratorCfg() if algorithm_cfg is None else algorithm_cfg
        )
        self._collision_cfg = deepcopy(
            ParallelJawGraspCollisionCfg() if collision_cfg is None else collision_cfg
        )
        self._annotation_cfg = deepcopy(
            GraspAnnotationCfg() if annotation_cfg is None else annotation_cfg
        )
        if not isinstance(self._algorithm_cfg, AntipodalGraspPoseGeneratorCfg):
            raise TypeError(
                "algorithm_cfg must be an AntipodalGraspPoseGeneratorCfg or None."
            )
        if not isinstance(self._collision_cfg, ParallelJawGraspCollisionCfg):
            raise TypeError(
                "collision_cfg must be a ParallelJawGraspCollisionCfg or None."
            )
        if not isinstance(self._annotation_cfg, GraspAnnotationCfg):
            raise TypeError("annotation_cfg must be a GraspAnnotationCfg or None.")
        if self._collision_cfg.opening_margin >= self._gripper_model.max_opening_width:
            raise ValueError(
                "collision opening_margin must be less than the model's "
                "max_opening_width."
            )
        self._backends: dict[tuple[object, ...], _AntipodalMeshBackend] = {}

    @property
    def algorithm_cfg(self) -> AntipodalGraspPoseGeneratorCfg:
        """Return an owned algorithm-configuration snapshot."""
        return deepcopy(self._algorithm_cfg)

    @property
    def collision_cfg(self) -> ParallelJawGraspCollisionCfg:
        """Return an owned collision-policy snapshot."""
        return deepcopy(self._collision_cfg)

    @property
    def annotation_cfg(self) -> GraspAnnotationCfg:
        """Return an owned annotation-policy snapshot."""
        return deepcopy(self._annotation_cfg)

    @staticmethod
    def _validate_geometry(
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
    ) -> None:
        """Validate one target-local indexed triangle mesh."""
        if (
            not isinstance(mesh_vertices, torch.Tensor)
            or not mesh_vertices.is_floating_point()
            or mesh_vertices.dim() != 2
            or mesh_vertices.shape[1] != 3
            or mesh_vertices.shape[0] == 0
            or not bool(torch.isfinite(mesh_vertices).all().item())
        ):
            raise ValueError(
                "mesh_vertices must be a non-empty finite floating tensor "
                "with shape (N, 3)."
            )
        if (
            not isinstance(mesh_triangles, torch.Tensor)
            or mesh_triangles.dtype == torch.bool
            or mesh_triangles.is_floating_point()
            or mesh_triangles.dim() != 2
            or mesh_triangles.shape[1] != 3
            or mesh_triangles.shape[0] == 0
        ):
            raise ValueError(
                "mesh_triangles must be a non-empty integer tensor with "
                "shape (M, 3)."
            )
        if mesh_triangles.device != mesh_vertices.device:
            raise ValueError("mesh_vertices and mesh_triangles must share a device.")
        if (
            bool((mesh_triangles < 0).any().item())
            or int(mesh_triangles.max().item()) >= mesh_vertices.shape[0]
        ):
            raise ValueError("mesh_triangles reference invalid vertex indices.")

    @staticmethod
    def _geometry_key(
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
    ) -> tuple[object, ...]:
        """Return an in-process identity key that detects tensor mutation."""
        return (
            str(mesh_vertices.device),
            mesh_vertices.dtype,
            tuple(mesh_vertices.shape),
            mesh_vertices.data_ptr(),
            mesh_vertices._version,
            mesh_triangles.dtype,
            tuple(mesh_triangles.shape),
            mesh_triangles.data_ptr(),
            mesh_triangles._version,
        )

    def _backend(
        self,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
    ) -> _AntipodalMeshBackend:
        """Return the lazily prepared backend for one mesh."""
        self._validate_geometry(mesh_vertices, mesh_triangles)
        key = self._geometry_key(mesh_vertices, mesh_triangles)
        backend = self._backends.get(key)
        if backend is not None:
            return backend

        model = self._gripper_model
        algorithm = self._algorithm_cfg
        collision = self._collision_cfg
        annotation = self._annotation_cfg
        backend = _AntipodalMeshBackend(
            vertices=mesh_vertices,
            triangles=mesh_triangles,
            sampler_cfg=AntipodalSamplerCfg(
                n_sample=algorithm.sample_count,
                max_angle=algorithm.ray_deviation_angle,
                max_length=model.max_opening_width,
                min_length=model.min_opening_width,
            ),
            collision_cfg=GripperCollisionCfg(
                max_open_length=model.max_opening_width,
                finger_length=model.finger_length,
                y_thickness=model.finger_width,
                x_thickness=model.finger_thickness,
                root_z_width=model.palm_depth,
                point_sample_dense=collision.point_sample_density,
                max_decomposition_hulls=collision.max_decomposition_hulls,
                open_check_margin=collision.opening_margin,
            ),
            max_deviation_angle=algorithm.approach_deviation_angle,
            approach_direction_samples=algorithm.approach_direction_samples,
            max_candidates=algorithm.max_candidates,
            interactive_annotation=annotation.selection_mode == "interactive",
            viser_port=annotation.viser_port,
            use_largest_connected_component=(
                annotation.use_largest_connected_component
            ),
            filter_ground_collision=collision.filter_ground_collision,
        )
        if annotation.force_refresh or not backend.is_prepared:
            backend.annotate()
        self._backends[key] = backend
        return backend

    def prepare_mesh(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare and return antipodal pairs for one target mesh.

        The configured annotation mode determines whether the whole mesh is
        sampled automatically or a region is selected through Viser. Prepared
        pairs are cached by the private mesh backend and returned as an owned
        tensor snapshot.

        Args:
            mesh_vertices: Target-local vertex positions with shape ``(N, 3)``.
            mesh_triangles: Triangle indices with shape ``(M, 3)``.

        Returns:
            Antipodal contact pairs with shape ``(K, 2, 3)``.
        """
        return self._backend(mesh_vertices, mesh_triangles).antipodal_pairs

    @staticmethod
    def _approach_directions(
        value: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize one shared or batched approach-direction tensor."""
        if not isinstance(value, torch.Tensor):
            raise TypeError("approach_direction must be a torch.Tensor.")
        normalized = value.to(device=device, dtype=torch.float32)
        if normalized.shape == (3,):
            normalized = normalized.unsqueeze(0).expand(batch_size, -1)
        elif normalized.shape != (batch_size, 3):
            raise ValueError(
                "approach_direction must have shape (3,) or "
                f"({batch_size}, 3), got {tuple(normalized.shape)}."
            )
        lengths = torch.linalg.vector_norm(normalized, dim=1, keepdim=True)
        if not bool(torch.isfinite(normalized).all().item()) or bool(
            (lengths <= 1.0e-6).any().item()
        ):
            raise ValueError("approach_direction must contain finite non-zero rows.")
        return normalized / lengths

    @staticmethod
    def _object_poses(
        value: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate batched homogeneous object poses."""
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.dim() != 3
            or value.shape[1:] != (4, 4)
            or value.shape[0] == 0
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(
                "obj_poses must be a non-empty finite floating tensor with "
                "shape (B, 4, 4)."
            )
        return value.to(device=device, dtype=torch.float32)

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
        pose_cost_fn: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None
        ) = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return ranked candidates, optionally from one projected axis end."""
        if pose_cost_fn is not None and not callable(pose_cost_fn):
            raise TypeError("pose_cost_fn must be callable or None.")
        backend = self._backend(mesh_vertices, mesh_triangles)
        poses = self._object_poses(obj_poses, device=backend.device)
        directions = self._approach_directions(
            approach_direction,
            batch_size=poses.shape[0],
            device=backend.device,
        )
        axes: torch.Tensor | None = None
        if obj_longest_axis is not None:
            axes = torch.as_tensor(
                obj_longest_axis,
                dtype=torch.float32,
                device=backend.device,
            )
            if axes.shape == (3,):
                axes = axes.unsqueeze(0).expand(poses.shape[0], -1)
            if axes.shape != (poses.shape[0], 3):
                raise ValueError(
                    "obj_longest_axis must have shape (3,) or "
                    f"({poses.shape[0]}, 3)."
                )
            lengths = torch.linalg.vector_norm(axes, dim=1, keepdim=True)
            if not torch.isfinite(axes).all() or torch.any(lengths <= 1.0e-8):
                raise ValueError("obj_longest_axis must contain finite non-zero rows.")
            axes = axes / lengths
        if isinstance(is_positive_part, bool):
            positive_parts = torch.full(
                (poses.shape[0],),
                is_positive_part,
                dtype=torch.bool,
                device=backend.device,
            )
        else:
            positive_parts = torch.as_tensor(
                is_positive_part,
                device=backend.device,
            )
            if positive_parts.dtype != torch.bool or positive_parts.shape != (
                poses.shape[0],
            ):
                raise ValueError(
                    "is_positive_part must be a bool or a bool tensor with shape "
                    f"({poses.shape[0]},)."
                )
        results: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, object_pose in enumerate(poses):
            success, grasp_poses, _, costs = backend.get_valid_grasp_poses(
                object_pose=object_pose,
                approach_direction=directions[index],
                obj_longest_axis=None if axes is None else axes[index],
                is_positive_part=bool(positive_parts[index].item()),
                pose_cost_fn=(
                    None
                    if pose_cost_fn is None
                    else lambda grasp_poses, costs: pose_cost_fn(
                        object_pose,
                        grasp_poses,
                        costs,
                    )
                ),
            )
            if grasp_poses.shape == (4, 4):
                grasp_poses = grasp_poses.unsqueeze(0)
            if costs.dim() == 0:
                costs = costs.unsqueeze(0)
            if not success:
                logger.log_warning(
                    f"Failed to find valid grasp poses for object row {index}."
                )
                costs = torch.full(
                    (grasp_poses.shape[0],),
                    torch.inf,
                    dtype=torch.float32,
                    device=backend.device,
                )
            results.append((grasp_poses, costs))
        return results

    def get_best_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the lowest-cost antipodal grasp for every object pose."""
        backend = self._backend(mesh_vertices, mesh_triangles)
        poses = self._object_poses(obj_poses, device=backend.device)
        directions = self._approach_directions(
            approach_direction,
            batch_size=poses.shape[0],
            device=backend.device,
        )
        successes: list[bool] = []
        grasp_poses: list[torch.Tensor] = []
        opening_widths: list[float] = []
        for index, object_pose in enumerate(poses):
            success, grasp_pose, opening_width = backend.get_grasp_poses(
                object_pose,
                directions[index],
            )
            successes.append(bool(success))
            if success:
                grasp_poses.append(grasp_pose)
                opening_widths.append(float(opening_width))
            else:
                logger.log_warning(f"No valid grasp pose found for object row {index}.")
                grasp_poses.append(
                    torch.eye(4, dtype=torch.float32, device=backend.device)
                )
                opening_widths.append(0.0)
        return (
            torch.tensor(successes, dtype=torch.bool, device=backend.device),
            torch.stack(grasp_poses),
            torch.tensor(
                opening_widths,
                dtype=torch.float32,
                device=backend.device,
            ),
        )

    def get_dual_arm_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        approach_direction: torch.Tensor,
        middle_empty_ratio: float = 0.4,
    ) -> list[dict[str, dict[str, object]] | None]:
        """Return antipodal candidate sets separated for a left/right pair."""
        if isinstance(middle_empty_ratio, bool) or not isinstance(
            middle_empty_ratio, (int, float)
        ):
            raise TypeError("middle_empty_ratio must be a real number.")
        ratio = float(middle_empty_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
            raise ValueError("middle_empty_ratio must be finite and in [0, 1).")
        backend = self._backend(mesh_vertices, mesh_triangles)
        poses = self._object_poses(obj_poses, device=backend.device)
        directions = self._approach_directions(
            approach_direction,
            batch_size=poses.shape[0],
            device=backend.device,
        )
        arm_direction = self._approach_directions(
            left_to_right_arm_direction,
            batch_size=1,
            device=backend.device,
        )[0]
        return [
            backend.get_dual_arm_valid_grasp_poses(
                object_pose=object_pose,
                approach_direction=directions[index],
                left_to_right_arm_direction=arm_direction,
                middle_empty_ratio=ratio,
            )
            for index, object_pose in enumerate(poses)
        ]
