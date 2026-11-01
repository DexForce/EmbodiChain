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

"""Workspace-conditioned layout proposals with a retained uniform component."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import math

import numpy as np
import torch

from ._values import count, digest, finite
from .budget import FeasibilityBudget
from .collision import UnsupportedFeasibilityError
from .contracts import TaskFeasibilityRequest
from .search import prepare_feasibility

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["RobotWorkspaceProposalSource"]


class RobotWorkspaceProposalSource:
    """Copy current-model FK samples into an independent numerical proposal prior.

    Args:
        robot: Warm B=1 robot; no live robot reference is retained.
        request: Fixed task and legal layout/installation domain.
        budget: Shared preparation deadline.
        phase_id: Explicit task phase selecting object-local TCP geometry.
        control_part: Explicit solver/cache part, preserving custom names.
        variable_name: Scene group containing that target's frame UID.
        sample_count: Bounded cached-qpos or joint-uniform FK population.
        uniform_fraction: Independent fraction retaining full-domain proposals.

    A missing cache permits finite joint-uniform FK proposals. A configured
    missing/corrupt cache fails explicitly. Projection onto authored height/tilt
    invalidates cached reachability: actual pose IK and full task checks remain
    mandatory. The source owns numeric copies and survives warm-host destruction.
    """

    def __init__(
        self,
        robot: Robot,
        request: TaskFeasibilityRequest,
        budget: FeasibilityBudget,
        *,
        phase_id: str,
        control_part: str,
        variable_name: str,
        sample_count: int = 4096,
        uniform_fraction: float = 0.5,
    ) -> None:
        count(sample_count, "workspace sample_count")
        if sample_count > 65536:
            raise ValueError("workspace sample_count must be <= 65536")
        uniform_fraction = finite((uniform_fraction,), "uniform_fraction")[0]
        if not 0 <= uniform_fraction <= 1:
            raise ValueError("workspace uniform_fraction must lie in [0,1]")
        if robot.get_qpos().shape[0] != 1:
            raise UnsupportedFeasibilityError(
                "Workspace proposal preparation requires B=1"
            )
        self._request = request
        self._prepared = prepare_feasibility(request, budget.cfg)
        self._uid = robot.uid
        self._root = (
            robot.get_local_pose(to_matrix=True)[0].detach().cpu().double().numpy()
        )
        self._variable_index = next(
            (i for i, v in enumerate(request.variables) if v.name == variable_name),
            None,
        )
        if (
            self._variable_index is None
            or request.variables[self._variable_index].kind != "scene"
        ):
            raise ValueError("workspace variable must select a declared scene group")
        variable = request.variables[self._variable_index]
        self._targets = {}
        for mode in request.modes:
            phase = next((p for p in mode.phases if p.phase_id == phase_id), None)
            target = (
                None
                if phase is None
                else next(
                    (t for t in phase.targets if t.control_part == control_part), None
                )
            )
            if target is None or target.frame_uid not in variable.members:
                raise ValueError(
                    "workspace phase/part must select an object-local target in the chosen group"
                )
            self._targets[mode.mode_id] = (
                np.linalg.inv(np.array(target.pose)),
                np.linalg.inv(np.array(variable.members[target.frame_uid])),
            )
        generator = torch.Generator(device=robot.device).manual_seed(budget.cfg.seed)
        budget.check()
        if control_part in (robot.cfg.workspace_cfg or {}):
            workspace = robot.get_workspace(control_part)
            settings = robot.cfg.workspace_cfg[control_part]
            indices = workspace.sample_indices(
                sample_count,
                strategy=settings.strategy,
                min_score=settings.min_score,
                generator=generator,
            )
            if (
                indices.shape != (sample_count,)
                or indices.dtype != torch.int64
                or bool(((indices < 0) | (indices >= workspace.num_samples)).any())
            ):
                raise ValueError("workspace sampler returned invalid cache indices")
            qpos = workspace.qpos[indices].to(device=robot.device)
            origin = "configured_qpos_cache"
        else:
            limits = robot.get_qpos_limits(name=control_part)[0]
            if (
                not bool(torch.isfinite(limits).all())
                or bool((limits[:, 1] < limits[:, 0]).any())
                or bool((limits.abs() > 1e6).any())
            ):
                raise UnsupportedFeasibilityError(
                    "Joint-uniform workspace preparation requires explicit finite joint bounds"
                )
            qpos = limits[:, 0] + torch.rand(
                (sample_count, len(limits)), device=robot.device, generator=generator
            ) * (limits[:, 1] - limits[:, 0])
            origin = "joint_uniform_fk"
        if (
            qpos.ndim != 2
            or len(qpos) != sample_count
            or not bool(torch.isfinite(qpos).all())
        ):
            raise ValueError(
                "workspace proposals require finite selected joint configurations"
            )
        batches = []
        for start in range(0, sample_count, budget.cfg.batch_size):
            budget.check()
            batch = (
                robot.compute_batch_fk(
                    qpos=qpos[None, start : start + budget.cfg.batch_size],
                    name=control_part,
                    env_ids=[0],
                    to_matrix=True,
                )[0]
                .detach()
                .cpu()
                .double()
                .numpy()
                .copy()
            )
            batches.append(batch)
        poses = np.concatenate(batches)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
            raise ValueError("workspace FK proposals must be finite rigid matrices")
        rotations = poses[:, :3, :3]
        if (
            not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5, rtol=0)
            or not np.allclose(
                rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5, rtol=0
            )
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5, rtol=0)
        ):
            raise ValueError("workspace FK proposals must be proper rigid transforms")
        self._local = np.linalg.inv(self._root)[None] @ poses
        self._uniform_fraction = uniform_fraction
        self._rng = np.random.default_rng(budget.cfg.seed)
        self._hits = self._fallbacks = 0
        self.source_id = digest(
            (
                request.robot_profile_id,
                control_part,
                phase_id,
                variable_name,
                origin,
                self._local,
                uniform_fraction,
            )
        )
        self.origin = origin
        budget.check()

    @property
    def metrics(self) -> dict[str, str | int]:
        """Return prior identity, population and actual proposal usage counts."""
        return {
            "source_id": self.source_id,
            "origin": self.origin,
            "workspace_points": len(self._local),
            "guided": self._hits,
            "uniform_or_unavailable": self._fallbacks,
        }

    def propose(
        self,
        parameters: Sequence[float],
        mode_id: str,
        ordinal: int,
        budget: FeasibilityBudget,
    ) -> tuple[float, ...]:
        """Condition object/group XY/yaw on FK samples at the proposed installation.

        Args:
            parameters: Original normalized full-domain low-discrepancy draw.
            mode_id: Allowed operation mode.
            ordinal: Logical source ordinal.
            budget: Shared deadline; no physical execution occurs.
        Returns:
            Full correlated parameter vector, retaining a uniform fallback.
        """
        budget.check()
        original = tuple(float(v) for v in parameters)
        if self._rng.random() < self._uniform_fraction or not len(self._local):
            self._fallbacks += 1
            return original
        candidate = self._prepared.candidate(original, mode_id=mode_id, ordinal=ordinal)
        root = np.array(candidate.deployment.poses.get(self._uid, self._root))
        target_inverse, member_inverse = self._targets[mode_id]
        groups = (root[None] @ self._local @ target_inverse[None]) @ member_inverse[
            None
        ]
        variable = self._request.variables[self._variable_index]
        reference = np.array(variable.reference_pose)
        offsets = np.empty((len(groups), 3))
        offsets[:, :2] = groups[:, :2, 3] - reference[:2, 3]
        relative = groups[:, :3, :3] @ reference[:3, :3].T
        offsets[:, 2] = np.arctan2(relative[:, 1, 0], relative[:, 0, 0])
        for axis, (lo, hi) in enumerate(zip(variable.lower, variable.upper)):
            if lo == hi:
                offsets[:, axis] = lo
        lo, hi = np.array(variable.lower), np.array(variable.upper)
        offsets[:, 2] += (
            2
            * math.pi
            * np.round(((lo[2] + hi[2]) / 2 - offsets[:, 2]) / (2 * math.pi))
        )
        valid = np.flatnonzero(np.all((offsets >= lo) & (offsets <= hi), axis=1))
        if not len(valid):
            self._fallbacks += 1
            return original
        index = valid[int(self._rng.integers(len(valid)))]
        projected = np.divide(
            offsets[index] - lo, hi - lo, out=np.zeros(3), where=hi > lo
        )
        result = list(original)
        result[self._variable_index * 3 : self._variable_index * 3 + 3] = (
            projected.tolist()
        )
        self._hits += 1
        return tuple(result)
