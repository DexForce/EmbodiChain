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

"""Frozen and case-scoped grasp generators for reproducible skill cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import torch

from embodichain.toolkits.graspkit import (
    GraspPoseGenerator,
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)
from embodichain.utils.math import pose_inv

from ..models import BenchmarkCase

__all__ = [
    "grasp_sample_count",
    "box_surface_mesh",
    "center_grasp_pose",
    "DeterministicCenterGraspPoseGenerator",
    "FrozenArticulationGraspPoseGenerator",
    "ActiveCaseGraspPoseGenerator",
    "parallel_jaw_sampler",
    "CaseSampleCountGraspPoseGenerator",
]


class _AntipodalGraspSearchError(RuntimeError):
    """A bounded grasp pool has no independently solved complete pose chain."""


def grasp_sample_count(case: BenchmarkCase) -> int:
    """Return the grasp sampler size a self-sampling case was frozen with."""
    return int(case.case_parameters.get("grasp_sample_count", 10_000))


def box_surface_mesh(
    size: Sequence[float],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a target-local box mesh for one synthetic graspable handle."""
    half = torch.tensor(size, dtype=torch.float32, device=device) * 0.5
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    vertices *= half
    triangles = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.int64,
        device=device,
    )
    return vertices, triangles


def center_grasp_pose(
    mesh_vertices: torch.Tensor,
    obj_poses: torch.Tensor,
    approach_direction: torch.Tensor,
) -> torch.Tensor:
    """Resolve a deterministic center grasp aligned with the approach axis."""
    directions = approach_direction.to(device=obj_poses.device, dtype=torch.float32)
    if directions.shape == (3,):
        directions = directions.unsqueeze(0).expand(obj_poses.shape[0], -1)
    if directions.shape != (obj_poses.shape[0], 3):
        raise ValueError(
            "approach_direction must have shape (3,) or match the object-pose batch."
        )
    z_axis = torch.nn.functional.normalize(directions, dim=1)
    x_reference = obj_poses[:, :3, 0].to(dtype=torch.float32)
    x_axis = x_reference - (x_reference * z_axis).sum(dim=1, keepdim=True) * z_axis
    degenerate = torch.linalg.vector_norm(x_axis, dim=1) <= 1.0e-6
    if degenerate.any():
        y_reference = obj_poses[:, :3, 1].to(dtype=torch.float32)
        fallback = (
            y_reference - (y_reference * z_axis).sum(dim=1, keepdim=True) * z_axis
        )
        x_axis = torch.where(degenerate[:, None], fallback, x_axis)
    x_axis = torch.nn.functional.normalize(x_axis, dim=1)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=1)

    grasp_pose = torch.eye(4, dtype=torch.float32, device=obj_poses.device).repeat(
        obj_poses.shape[0], 1, 1
    )
    grasp_pose[:, :3, 0] = x_axis
    grasp_pose[:, :3, 1] = y_axis
    grasp_pose[:, :3, 2] = z_axis
    local_center = mesh_vertices.to(device=obj_poses.device, dtype=torch.float32).mean(
        dim=0
    )
    grasp_pose[:, :3, 3] = (
        torch.matmul(obj_poses[:, :3, :3], local_center) + obj_poses[:, :3, 3]
    )
    return grasp_pose


class DeterministicCenterGraspPoseGenerator(GraspPoseGenerator):
    """Return the manifest-defined synthetic handle's center grasp."""

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return one deterministic zero-cost grasp per object pose."""
        del mesh_triangles, obj_longest_axis, is_positive_part
        poses = center_grasp_pose(mesh_vertices, obj_poses, approach_direction)
        return [
            (
                poses[index : index + 1],
                torch.zeros(1, dtype=torch.float32, device=poses.device),
            )
            for index in range(poses.shape[0])
        ]

    def get_best_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the deterministic center grasp for every object pose."""
        del mesh_triangles
        poses = center_grasp_pose(mesh_vertices, obj_poses, approach_direction)
        return (
            torch.ones(poses.shape[0], dtype=torch.bool, device=poses.device),
            poses,
            torch.zeros(poses.shape[0], dtype=torch.float32, device=poses.device),
        )


class FrozenArticulationGraspPoseGenerator(GraspPoseGenerator):
    """Replay manifest-frozen target-to-grasp transforms for Slide cases."""

    def __init__(
        self,
        target_grasps: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        if not target_grasps:
            raise ValueError("At least one frozen Slide grasp is required.")
        normalized: list[tuple[torch.Tensor, torch.Tensor]] = []
        for target, grasp in target_grasps:
            target = target.detach().to(device="cpu", dtype=torch.float32).clone()
            grasp = grasp.detach().to(device="cpu", dtype=torch.float32).clone()
            if target.shape == (4, 4):
                target = target.unsqueeze(0)
            if grasp.shape == (4, 4):
                grasp = grasp.unsqueeze(0)
            if target.dim() != 3 or target.shape[1:] != (4, 4):
                raise ValueError(
                    "Frozen Slide targets must have shape (4, 4) or (B, 4, 4)."
                )
            if grasp.shape != target.shape:
                raise ValueError(
                    "Frozen Slide target and grasp batches must have equal shapes."
                )
            target_to_grasp = torch.bmm(pose_inv(target), grasp)
            for previous_target, previous_grasp in normalized:
                if previous_target.shape != target.shape or not torch.allclose(
                    previous_target, target
                ):
                    continue
                previous_target_to_grasp = torch.bmm(
                    pose_inv(previous_target), previous_grasp
                )
                if not torch.allclose(previous_target_to_grasp, target_to_grasp):
                    raise ValueError(
                        "Frozen Slide cases with identical target batches must "
                        "use the same target-local grasp."
                    )
            normalized.append((target, grasp))
        self._target_grasps = tuple(normalized)

    def _resolve(self, obj_poses: torch.Tensor) -> torch.Tensor:
        """Select the closest frozen batch and preserve each row's local grasp."""
        if obj_poses.dim() != 3 or obj_poses.shape[1:] != (4, 4):
            raise ValueError("Slide object poses must have shape (B, 4, 4).")
        candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        batch_size = obj_poses.shape[0]
        for target, grasp in self._target_grasps:
            if target.shape[0] == 1:
                target = target.expand(batch_size, -1, -1)
                grasp = grasp.expand(batch_size, -1, -1)
            elif target.shape[0] != batch_size:
                continue
            target = target.to(device=obj_poses.device, dtype=torch.float32)
            grasp = grasp.to(device=obj_poses.device, dtype=torch.float32)
            translation_cost = torch.linalg.vector_norm(
                obj_poses[:, :3, 3] - target[:, :3, 3], dim=-1
            )
            rotation_cost = torch.linalg.matrix_norm(
                obj_poses[:, :3, :3] - target[:, :3, :3], dim=(-2, -1)
            )
            candidates.append(
                (
                    (translation_cost + 0.1 * rotation_cost).mean(),
                    target,
                    grasp,
                )
            )
        if not candidates:
            raise ValueError(
                "No frozen Slide grasp batch matches the requested batch size."
            )
        _, target, grasp = min(candidates, key=lambda item: float(item[0].item()))
        target_to_grasp = torch.bmm(pose_inv(target), grasp)
        return torch.bmm(obj_poses.to(dtype=torch.float32), target_to_grasp)

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return one frozen zero-cost grasp candidate per environment."""
        del (
            mesh_vertices,
            mesh_triangles,
            approach_direction,
            obj_longest_axis,
            is_positive_part,
        )
        poses = self._resolve(obj_poses)
        return [
            (
                poses[index : index + 1],
                torch.zeros(1, dtype=torch.float32, device=poses.device),
            )
            for index in range(poses.shape[0])
        ]

    def get_best_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the exact grasp selected while building the case manifest."""
        del mesh_vertices, mesh_triangles, approach_direction
        poses = self._resolve(obj_poses)
        return (
            torch.ones(poses.shape[0], dtype=torch.bool, device=poses.device),
            poses,
            torch.zeros(poses.shape[0], dtype=torch.float32, device=poses.device),
        )


class ActiveCaseGraspPoseGenerator(GraspPoseGenerator):
    """Serve the manifest grasp of the case being compiled.

    Skills such as OpenDoor sample their own grasp and accept no explicit one.
    The scenario activates the case's frozen ``(target, grasp)`` pair around
    compilation; the target-local grasp is replayed for the observed target
    pose.  Without an active pair the request goes to ``fallback``.
    """

    def __init__(self, fallback: GraspPoseGenerator | None = None) -> None:
        self._fallback = fallback
        self._target_to_grasp: torch.Tensor | None = None

    def activate(self, target_pose: torch.Tensor, grasp_pose: torch.Tensor) -> None:
        """Freeze one case's target-local grasp for the next compilation."""
        if target_pose.shape != grasp_pose.shape or target_pose.shape[-2:] != (4, 4):
            raise ValueError("Frozen target and grasp poses must be equal (B, 4, 4).")
        self._target_to_grasp = torch.bmm(
            pose_inv(target_pose.to(dtype=torch.float32)),
            grasp_pose.to(dtype=torch.float32),
        )

    def deactivate(self) -> None:
        """Return to the fallback generator after compilation."""
        self._target_to_grasp = None

    def _resolve(self, obj_poses: torch.Tensor) -> torch.Tensor:
        assert self._target_to_grasp is not None
        target_to_grasp = self._target_to_grasp.to(device=obj_poses.device)
        if target_to_grasp.shape[0] != obj_poses.shape[0]:
            if target_to_grasp.shape[0] != 1:
                raise ValueError("Frozen grasp batch does not match the target batch.")
            target_to_grasp = target_to_grasp.expand(obj_poses.shape[0], -1, -1)
        return torch.bmm(obj_poses.to(dtype=torch.float32), target_to_grasp)

    def _require_fallback(self) -> GraspPoseGenerator:
        if self._fallback is None:
            raise RuntimeError("No frozen benchmark grasp is active for this request.")
        return self._fallback

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return the active frozen grasp as one zero-cost candidate per row."""
        if self._target_to_grasp is None:
            return self._require_fallback().get_valid_grasp_poses(
                mesh_vertices=mesh_vertices,
                mesh_triangles=mesh_triangles,
                obj_poses=obj_poses,
                approach_direction=approach_direction,
                obj_longest_axis=obj_longest_axis,
                is_positive_part=is_positive_part,
            )
        poses = self._resolve(obj_poses)
        return [
            (
                poses[index : index + 1],
                torch.zeros(1, dtype=torch.float32, device=poses.device),
            )
            for index in range(poses.shape[0])
        ]

    def get_best_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the active frozen grasp for every target row."""
        if self._target_to_grasp is None:
            return self._require_fallback().get_best_grasp_poses(
                mesh_vertices=mesh_vertices,
                mesh_triangles=mesh_triangles,
                obj_poses=obj_poses,
                approach_direction=approach_direction,
            )
        poses = self._resolve(obj_poses)
        return (
            torch.ones(poses.shape[0], dtype=torch.bool, device=poses.device),
            poses,
            torch.zeros(poses.shape[0], dtype=torch.float32, device=poses.device),
        )


def parallel_jaw_sampler(
    n_sample: int,
    *,
    gripper_model: ParallelJawGripperModelCfg | None = None,
    root_from_tcp: Sequence[Sequence[float]] | None = None,
) -> ParallelJawGraspPoseGenerator:
    """Build a case sampler with the selected physical hand and runtime TCP.

    Args:
        n_sample: Number of antipodal contact-pair samples.
        gripper_model: Mounted hand geometry shared by preparation and replay.
        root_from_tcp: The bound fixture's calibrated hand-root-to-TCP transform.

    Returns:
        A sampler configured for the selected hand.
    """
    from .._atomic_fixtures import (
        create_parallel_jaw_grasp_pose_generator,
    )

    return create_parallel_jaw_grasp_pose_generator(
        n_sample=n_sample,
        force_refresh=False,
        gripper_model=gripper_model,
        root_from_tcp=root_from_tcp,
    )


class CaseSampleCountGraspPoseGenerator(ParallelJawGraspPoseGenerator):
    """Serve each grasp-sampling case from a sampler with its own sample count.

    Skills such as CoordinatedPickment sample their grasps during compilation
    and the case manifest froze the waypoints those grasps produced with the
    case's ``grasp_sample_count``.  The scenario activates that count around
    compilation so another case in the track cannot change the sampled
    surface.  Without an active count the request goes to ``default_count``.
    """

    def __init__(
        self,
        samplers: Mapping[int, ParallelJawGraspPoseGenerator],
        *,
        default_count: int,
    ) -> None:
        if default_count not in samplers:
            raise ValueError("The default sample count needs its own sampler.")
        super().__init__(samplers[default_count].gripper_model)
        self._samplers = dict(samplers)
        self._default_count = default_count
        self._active_count = default_count

    def activate(self, sample_count: int) -> None:
        """Use the sampler built with ``sample_count`` for the next compilation."""
        if sample_count not in self._samplers:
            raise ValueError(f"No grasp sampler was prepared for {sample_count}.")
        self._active_count = sample_count

    def deactivate(self) -> None:
        """Return to the default sampler after compilation."""
        self._active_count = self._default_count

    @property
    def _sampler(self) -> ParallelJawGraspPoseGenerator:
        return self._samplers[self._active_count]

    def get_valid_grasp_poses(self, **kwargs: object) -> list:
        """Forward the request to the active sampler."""
        return self._sampler.get_valid_grasp_poses(**kwargs)

    def get_best_grasp_poses(self, **kwargs: object) -> tuple:
        """Forward the request to the active sampler."""
        return self._sampler.get_best_grasp_poses(**kwargs)

    def get_grasp_candidates(self, **kwargs: object) -> list:
        """Forward the request to the active sampler."""
        return self._sampler.get_grasp_candidates(**kwargs)

    def get_dual_arm_valid_grasp_poses(self, **kwargs: object) -> list:
        """Forward the request to the active sampler."""
        return self._sampler.get_dual_arm_valid_grasp_poses(**kwargs)
