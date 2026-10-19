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

"""Object-local scene proposals backed by the current robot workspace."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING

import torch

from .validation import SceneCheck, SceneCheckReport
from .expansion import ScenePoseChange

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["RobotSceneWorkspace", "SceneWorkspaceCandidate"]


def _pose_matrix(value: object, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.dtype == torch.bool or tensor.is_complex() or tensor.shape != (4, 4):
        raise ValueError("pose must be a real 4x4 rigid transform")
    validated = ScenePoseChange(
        "pose", tuple(map(tuple, tensor.detach().cpu().tolist()))
    )
    return torch.tensor(validated.arena_pose, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class SceneWorkspaceCandidate:
    """An object proposal derived from one valid cached joint configuration.

    Args:
        pose_change: Proposed object pose in the local arena frame.
        joint_seed: Owned joint values for the explicitly selected control part.
        cache_index: Corresponding valid workspace sample index.
        score: Optional cached reachability/manipulability score.

    Kinematic sampling does not establish object support, collision-free motion
    or task success. Modified proposals require a fresh pose IK check.
    """

    pose_change: ScenePoseChange
    joint_seed: tuple[float, ...]
    cache_index: int
    score: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.pose_change, ScenePoseChange):
            raise TypeError("pose_change must be a ScenePoseChange")
        try:
            values = tuple(self.joint_seed)
        except TypeError as error:
            raise ValueError(
                "joint_seed must contain finite real joint values"
            ) from error
        if not values or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("joint_seed must contain finite real joint values")
        object.__setattr__(self, "joint_seed", tuple(map(float, values)))
        if type(self.cache_index) is not int or self.cache_index < 0:
            raise ValueError("cache_index must be an exact nonnegative integer")
        if self.score is not None:
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, Real)
                or not math.isfinite(self.score)
            ):
                raise ValueError("score must be a finite real number or None")
            object.__setattr__(self, "score", float(self.score))


class RobotSceneWorkspace:
    """Adapt current-base workspace sampling to object-local manipulation poses.

    Args:
        robot: Live robot with workspace sampling and IK support.
        control_part: Explicit solver/workspace control-part name.
        position_tolerance_m: Maximum FK position residual for pose acceptance.
        orientation_tolerance_rad: Maximum FK orientation residual in radians.

    The initial integration supports B=1. All transforms use the local arena
    frame; replication offsets must not be added to Robot FK/IK or object poses.
    """

    def __init__(
        self,
        robot: Robot,
        *,
        control_part: str,
        position_tolerance_m: float = 1e-3,
        orientation_tolerance_rad: float = 1e-2,
    ) -> None:
        if type(control_part) is not str or not control_part.strip():
            raise ValueError("control_part must be an explicit nonempty name")
        if control_part != control_part.strip():
            raise ValueError("control_part must not have outer whitespace")
        for name, value in (
            ("position_tolerance_m", position_tolerance_m),
            ("orientation_tolerance_rad", orientation_tolerance_rad),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        qpos = robot.get_qpos()
        if not isinstance(qpos, torch.Tensor) or qpos.ndim != 2 or qpos.shape[0] != 1:
            raise ValueError("scene workspace currently requires B=1")
        solver = robot.get_solver(name=control_part)
        if solver is None:
            raise ValueError(f"No solver exists for control part {control_part!r}")
        self._robot = robot
        self._control_part = control_part
        self._dof = solver.dof
        self._device = torch.device(robot.device)
        self._position_tolerance_m = position_tolerance_m
        self._orientation_tolerance_rad = orientation_tolerance_rad

    def sample_object_poses(
        self,
        entity_uid: str,
        object_grasp_pose: torch.Tensor | Sequence[Sequence[float]],
        *,
        num_samples: int,
        seed: int,
        position_bounds: Sequence[Sequence[float]] | torch.Tensor | None = None,
        max_attempts: int = 64,
    ) -> tuple[SceneWorkspaceCandidate, ...]:
        """Propose object poses from valid TCP samples and local grasp geometry.

        Args:
            entity_uid: Physical object UID used by the scene host.
            object_grasp_pose: Transform from TCP into the object frame.
            num_samples: Requested maximum number of returned proposals.
            seed: Nonnegative seed for an isolated device-local generator.
            position_bounds: Optional object-origin arena bounds of shape (2,3).
            max_attempts: Sampling budget when filtering object-origin bounds.

        Returns:
            Up to ``num_samples`` valid proposals. Exhausting sampling bounds
            returns fewer samples and does not certify other poses unreachable.

        Raises:
            ValueError: For invalid arguments or malformed workspace results.
            Exception: Cache loading and robot configuration errors propagate.
        """
        ScenePoseChange(entity_uid, torch.eye(4).tolist())
        for name, value in (
            ("num_samples", num_samples),
            ("max_attempts", max_attempts),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_attempts < num_samples:
            raise ValueError("max_attempts must be at least num_samples")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        grasp = _pose_matrix(object_grasp_pose, device=self._device)
        bounds = None
        if position_bounds is not None:
            bounds = torch.as_tensor(
                position_bounds, dtype=torch.float32, device=self._device
            )
            if (
                bounds.shape != (2, 3)
                or not bool(torch.isfinite(bounds).all())
                or bool((bounds[0] > bounds[1]).any())
            ):
                raise ValueError(
                    "position_bounds must be finite ordered bounds of shape (2,3)"
                )
        count = max_attempts if bounds is not None else num_samples
        samples = self._robot.sample_reachable_pose(
            name=self._control_part,
            env_ids=[0],
            num_samples=count,
            max_attempts=max(max_attempts, count),
            generator=torch.Generator(device=self._device).manual_seed(seed),
        )
        if (
            samples.eef_pose.shape != (1, count, 4, 4)
            or samples.qpos.shape != (1, count, self._dof)
            or samples.valid.shape != (1, count)
            or samples.valid.dtype != torch.bool
            or samples.indices.shape != (1, count)
            or samples.indices.dtype != torch.long
            or (samples.score is not None and samples.score.shape != (1, count))
        ):
            raise ValueError(
                "workspace sample shapes do not match the requested control part"
            )
        objects = samples.eef_pose[0] @ torch.linalg.inv(grasp)
        candidates = []
        for index in samples.valid[0].nonzero(as_tuple=False).flatten().tolist():
            if samples.indices[0, index].item() < 0:
                raise ValueError(
                    "valid workspace samples must have nonnegative cache indices"
                )
            pose = _pose_matrix(objects[index], device=self._device)
            seed_values = samples.qpos[0, index]
            if not bool(torch.isfinite(seed_values).all()):
                raise ValueError("valid workspace qpos samples must be finite")
            score = (
                None if samples.score is None else float(samples.score[0, index].item())
            )
            if score is not None and not math.isfinite(score):
                raise ValueError("valid workspace scores must be finite")
            if bounds is not None and not bool(
                ((pose[:3, 3] >= bounds[0]) & (pose[:3, 3] <= bounds[1])).all()
            ):
                continue
            candidates.append(
                SceneWorkspaceCandidate(
                    pose_change=ScenePoseChange(
                        entity_uid, pose.detach().cpu().tolist()
                    ),
                    joint_seed=tuple(seed_values.detach().cpu().tolist()),
                    cache_index=int(samples.indices[0, index].item()),
                    score=score,
                )
            )
            if len(candidates) == num_samples:
                break
        return tuple(candidates)

    def check_object_pose(
        self,
        pose_change: ScenePoseChange,
        object_grasp_pose: torch.Tensor | Sequence[Sequence[float]],
        *,
        joint_seed: Sequence[float] | torch.Tensor | None = None,
    ) -> SceneCheckReport:
        """Run actual pose IK and verify joint limits and full-pose FK residuals.

        Args:
            pose_change: Actual or proposed local-arena object pose.
            object_grasp_pose: Transform from TCP into the object frame.
            joint_seed: Optional selected-control-part seed, width solver DOF.

        Returns:
            Mandatory ``workspace.pose_ik`` evidence. Passing certifies this
            manipulation pose only, independently of motion and task acceptance.
        """
        if not isinstance(pose_change, ScenePoseChange):
            raise TypeError("pose_change must be a ScenePoseChange")
        grasp = _pose_matrix(object_grasp_pose, device=self._device)
        object_pose = torch.tensor(
            pose_change.arena_pose, dtype=torch.float32, device=self._device
        )
        seed = None
        if joint_seed is not None:
            seed = torch.as_tensor(joint_seed, dtype=torch.float32, device=self._device)
            if seed.shape == (self._dof,):
                seed = seed.unsqueeze(0)
            if seed.shape != (1, self._dof) or not bool(torch.isfinite(seed).all()):
                raise ValueError(
                    "joint_seed must contain finite values of selected control-part width"
                )
        target = object_pose @ grasp
        result = self._robot.compute_ik(
            pose=target.unsqueeze(0),
            joint_seed=seed,
            name=self._control_part,
            env_ids=[0],
        )
        if result is None:
            raise RuntimeError("selected control-part IK solver is unavailable")
        success, qpos = result
        if (
            success.shape != (1,)
            or success.dtype != torch.bool
            or qpos.shape != (1, self._dof)
        ):
            raise ValueError("IK returned an invalid success mask or joint shape")
        passed = bool(success[0])
        reason = "IK solver did not find a solution"
        if passed and not bool(torch.isfinite(qpos).all()):
            passed = False
            reason = "IK solution contains nonfinite joint values"
        metrics = {}
        if passed:
            limits = self._robot.get_qpos_limits(name=self._control_part, env_ids=[0])
            if (
                limits.shape != (1, self._dof, 2)
                or bool(torch.isnan(limits).any())
                or bool((limits[..., 0] > limits[..., 1]).any())
            ):
                raise ValueError("Selected control-part joint limits are invalid")
            passed = bool(((qpos >= limits[..., 0]) & (qpos <= limits[..., 1])).all())
            if not passed:
                reason = "IK solution exceeds joint limits"
        if passed:
            fk = self._robot.compute_fk(
                qpos=qpos, name=self._control_part, env_ids=[0], to_matrix=True
            )
            if not isinstance(fk, torch.Tensor) or fk.shape != (1, 4, 4):
                raise ValueError("FK returned an invalid pose shape")
            actual = _pose_matrix(fk[0], device=self._device)
            position_error = float(
                torch.linalg.vector_norm(actual[:3, 3] - target[:3, 3])
            )
            relative_rotation = actual[:3, :3].T @ target[:3, :3]
            cosine = ((torch.trace(relative_rotation) - 1) / 2).clamp(-1.0, 1.0)
            orientation_error = float(torch.acos(cosine))
            metrics = {
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
            }
            passed = (
                position_error <= self._position_tolerance_m
                and orientation_error <= self._orientation_tolerance_rad
            )
            reason = (
                "Full-pose IK/FK check passed"
                if passed
                else "FK position or orientation residual exceeds tolerance"
            )
        return SceneCheckReport(
            (
                SceneCheck(
                    "workspace.pose_ik",
                    "passed" if passed else "failed",
                    f"{reason} for {pose_change.entity_uid!r} using {self._control_part!r}; motion remains unverified",
                    metrics=metrics,
                ),
            )
        )
