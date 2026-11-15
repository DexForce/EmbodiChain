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

"""Bounded, ordered proposal screening with independent restart streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from ..config import FreeSpaceTrackCfg, SuiteCfg

__all__: list[str] = []


class _ProposalKinematics:
    """Fold proposals into the sample axis without changing arena ownership."""

    def __init__(self, robot: Robot, proposals: int, batch_size: int) -> None:
        self.robot = robot
        self.proposals = proposals
        self.batch_size = batch_size
        self.device = robot.device

    def get_qpos_limits(self, name: str) -> torch.Tensor:
        limits = self.robot.get_qpos_limits(name=name)
        return limits.expand(self.batch_size, -1, -1).repeat(self.proposals, 1, 1)

    def compute_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ) -> torch.Tensor:
        return self.compute_batch_fk(qpos[:, None], name, to_matrix)[:, 0]

    def compute_batch_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ) -> torch.Tensor:
        count, dof = qpos.shape[1:]
        folded = qpos.reshape(self.proposals, self.batch_size, count, dof)
        folded = folded.permute(1, 0, 2, 3).reshape(
            self.batch_size, self.proposals * count, dof
        )
        poses = self.robot.compute_batch_fk(qpos=folded, name=name, to_matrix=to_matrix)
        return (
            poses.reshape(self.batch_size, self.proposals, count, 4, 4)
            .permute(1, 0, 2, 3, 4)
            .reshape(self.proposals * self.batch_size, count, 4, 4)
        )


class _FreeSpaceQualificationError(ValueError):
    """An exhausted input search with reproducible request operands."""

    def __init__(self, message: str, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.qualification_diagnostics = diagnostics


def _sample_request_batched(
    robot: Robot,
    part: str,
    starts: torch.Tensor,
    settings: FreeSpaceTrackCfg,
    seed: int,
    count: int,
    shape: str,
    component: str,
    distance_bin: str,
    batch_mode: str,
    suite: SuiteCfg,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Choose the first passing complete proposal independently for each row.

    Candidate order and starts match serial proposal generation. Each proposal
    owns its CPU restart stream, so chunk size and unrelated RNG use cannot
    alter its random hints. This is an explicit new screening population, not
    bitwise replay of the legacy device RNG. Storage is bounded by the chunk:
    O(C B R D squared + C B K (D + R D)), not the total proposal allowance.
    """
    from ._free_space_sampling import (
        _generator,
        _screen_path,
        _target_path,
        _translation_range,
    )

    batch, dof = starts.shape
    start_pose = robot.compute_fk(qpos=starts, name=part, to_matrix=True)
    targets = start_pose[:, None].expand(-1, count, -1, -1).clone()
    references = starts[:, None].expand(-1, count, -1).clone()
    accepted = torch.zeros(batch, device=starts.device, dtype=torch.bool)
    attempts = torch.zeros(batch, device=starts.device, dtype=torch.int64)
    rng = _generator("target-v1", seed, shape, component, distance_bin)
    translation_range = _translation_range(settings, distance_bin)
    offset = 0
    while offset < settings.max_generation_attempts:
        # Avoid a whole chunk of work for the common first-proposal success.
        chunk = min(
            1 if offset == 0 else settings.screening_proposal_batch_size,
            settings.max_generation_attempts - offset,
        )
        proposed = torch.stack(
            [
                _target_path(
                    start_pose,
                    count,
                    shape,
                    component,
                    distance_bin,
                    rng,
                    batch_mode,
                    translation_range,
                )
                for _ in range(chunk)
            ]
        )
        restart_draws = (
            torch.stack(
                [
                    torch.rand(
                        count,
                        batch,
                        max(0, settings.screening_restarts - 2),
                        dof,
                        generator=_generator(
                            "screen-proposal-v1",
                            seed,
                            shape,
                            component,
                            distance_bin,
                            attempt,
                        ),
                        dtype=starts.dtype,
                    )
                    for attempt in range(offset, offset + chunk)
                ]
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(chunk * batch, count, max(0, settings.screening_restarts - 2), dof)
            .to(starts.device)
        )
        valid, qpos = _screen_path(
            _ProposalKinematics(robot, chunk, batch),
            part,
            starts.repeat(chunk, 1),
            proposed.reshape(chunk * batch, count, 4, 4),
            batch_mode,
            min(0.001, suite.protocol.position_threshold_m * 0.25),
            min(0.01, suite.protocol.rotation_threshold_rad * 0.25),
            min(1.0e-5, suite.protocol.joint_limit_tolerance_rad),
            settings.reachability_solver,
            settings.screening_iterations,
            settings.screening_restarts,
            restart_draws,
            batch,
        )
        valid = valid.reshape(chunk, batch)
        first = valid.to(torch.int64).argmax(dim=0)
        rows = (valid.any(dim=0) & ~accepted).nonzero().flatten()
        targets[rows] = proposed[first[rows], rows]
        references[rows] = qpos.reshape(chunk, batch, count, dof)[first[rows], rows]
        attempts[rows] = offset + first[rows] + 1
        accepted[rows] = True
        offset += chunk
        if bool(accepted.all()):
            return targets, references, attempts.cpu().tolist()

    missing = (~accepted).nonzero().flatten().cpu().tolist()
    raise _FreeSpaceQualificationError(
        f"invalid_case: no independently reachable {component}/{distance_bin}/{shape} "
        f"request for rows {missing} after {settings.max_generation_attempts} attempts "
        f"(seed={seed}, K={count}).",
        {
            "scope": "free_space_endpoint_qualification_only",
            "result": "bounded_search_exhausted_not_infeasibility_proof",
            "seed": seed,
            "num_waypoints": count,
            "path_shape": shape,
            "motion_component": component,
            "distance_bin": distance_bin,
            "batch_mode": batch_mode,
            "unqualified_rows": missing,
            "start_qpos": starts.detach().cpu().tolist(),
            "max_generation_attempts": settings.max_generation_attempts,
            "screening_proposal_batch_size": settings.screening_proposal_batch_size,
            "screening_restart_rule": "proposal_waypoint_cpu_v1",
            "screening_iterations": settings.screening_iterations,
            "screening_restarts": settings.screening_restarts,
        },
    )
