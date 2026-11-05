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

"""Bounded CPU PyTorch IK branch/path search with explicit source frame ownership."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import torch

from embodichain.compute.geometry.pose_checks import fk_pose_error_mask
from embodichain.compute.kinematics.ik_path import resolve_seeded_ik_path
from embodichain.compute.kinematics.seed_selection import (
    select_separated_joint_branches,
)
from embodichain.lab.sim.scene._validation import _owned_b1_pose
from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain
from .limits import authored_joint_velocity_limits
from .search import record_search
from .solvers.pytorch_solver import PytorchSolver
from .solvers.qpos_seed_sampler import QposSeedSampler

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["CpuIKBranchSearch"]


class CpuIKBranchSearch:
    """Source-compatible one-environment CPU solver continuation.

    Args:
        robot: Exact owner of a CPU PytorchSolver without a custom seed sampler.
        control_part: Explicit solver/control group.
        control_dt: Positive unchanged path interval.
        solver_root_pose: Captured CPU world pose of this solver's root, not the
            whole-robot root unless their frame identities actually coincide.

    The fixed endpoint population has 32 seeds, at most 4 separated results and
    a 0.02 joint-distance threshold. The legacy RNG law is deliberately retained:
    each search calls ``torch.manual_seed(0)`` inside a CPU-only fork. PyTorch also
    seeds CUDA generators in that call, and the CPU-only fork does not restore
    them. This source side effect is distinct from a future isolated-seed fix.
    No native scene, physical effect or controller action is created here.
    """

    def __init__(
        self,
        robot: Robot,
        control_part: str,
        control_dt: float,
        *,
        solver_root_pose: torch.Tensor,
    ) -> None:
        if solver_root_pose is None:
            raise ValueError(
                "Pass an explicit solver-root pose, or call capture() to read it."
            )
        self._initialize(robot, control_part, control_dt, solver_root_pose)

    @classmethod
    def capture(
        cls, robot: Robot, control_part: str, control_dt: float
    ) -> CpuIKBranchSearch:
        """Explicitly capture one live solver-root pose, without stepping.

        Args:
            robot: Exact solver owner, paused by the calling planning scope.
            control_part: Declared CPU PytorchSolver group.
            control_dt: Positive path sampling interval.

        Returns:
            A fresh search object. Hosts already owning a same-observation
            solver-root pose use the constructor instead of this live reader.
        """
        result = object.__new__(cls)
        result._initialize(robot, control_part, control_dt, None)
        return result

    @staticmethod
    def supports(robot: Robot, control_part: str) -> bool:
        """Check the preserved solver family/device/seed-sampler boundary.

        Args:
            robot: Solver owner; no pose or IK query is performed.
            control_part: Declared group whose solver is inspected.

        Returns:
            True for a B1 CPU PytorchSolver with no custom seed sampler.
        """
        solver = robot.get_solver(control_part)
        return (
            robot.num_instances == 1
            and isinstance(solver, PytorchSolver)
            and solver.device.type == "cpu"
            and solver._seed_sampler is None
        )

    def _initialize(self, robot, part, dt, pose):
        if type(part) is not str or not part or part != part.strip():
            raise ValueError("CPU IK search requires an exact control group.")
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("CPU IK search requires a finite positive interval.")
        solver = robot.get_solver(part)
        if (
            robot.num_instances != 1
            or not isinstance(solver, PytorchSolver)
            or solver.device.type != "cpu"
            or solver._seed_sampler is not None
        ):
            raise ValueError(
                "CPU IK branch search supports B1 PytorchSolver without a custom seed sampler."
            )
        self._robot, self._part, self._dt, self._solver = robot, part, float(dt), solver
        self._pos_eps, self._rot_eps = solver._pos_eps, solver._rot_eps
        self._lower, self._upper = (
            solver.lower_qpos_limits.clone(),
            solver.upper_qpos_limits.clone(),
        )
        self._limits = authored_joint_velocity_limits(robot, part)[0].cpu().clone()
        root = (
            robot.get_link_pose(solver.root_link_name, to_matrix=True)
            if pose is None
            else pose
        )
        root = _owned_b1_pose(root, "CPU IK solver-root pose")
        if root.device.type != "cpu":
            raise ValueError("CPU IK search requires CPU solver-root observation data.")
        self._inverse = torch.linalg.inv(root)
        self._chain = create_pk_serial_chain(
            urdf_path=solver.urdf_path,
            root_link_name=solver.root_link_name,
            end_link_name=solver.end_link_name,
            device=torch.device("cpu"),
        ).to(dtype=torch.float64)

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed solver owner."""
        return self._robot

    @property
    def part(self) -> str:
        """Return the explicitly selected control group."""
        return self._part

    @property
    def dt(self) -> float:
        """Return the unchanged path interval."""
        return self._dt

    def to_root(self, poses: torch.Tensor) -> torch.Tensor:
        """Project explicitly supplied world poses into the captured solver root.

        Args:
            poses: One or more CPU rigid transforms using the captured dtype.

        Returns:
            Root-relative transforms; no live pose is reread.
        """
        if (
            poses.device != self._inverse.device
            or poses.dtype != self._inverse.dtype
            or poses.shape[-2:] != (4, 4)
        ):
            raise ValueError(
                "CPU IK poses must match the captured root frame dtype/device."
            )
        return self._inverse @ poses

    def pose_valid(self, qpos: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        """Qualify joint rows through the source double-precision FK residual.

        Args:
            qpos: CPU joint rows ``(N,D)``.
            poses: Matching root-relative target transforms or one broadcast pose.

        Returns:
            One positional/angular validity bit per joint row.
        """
        if qpos.device.type != "cpu" or poses.device.type != "cpu":
            raise ValueError("CPU IK qualification does not support CUDA inputs.")
        fk = self._chain.forward_kinematics_tensor(qpos.double())[-1] @ torch.as_tensor(
            self._solver.tcp_xpos, dtype=torch.float64
        )
        return fk_pose_error_mask(fk, poses, self._pos_eps, self._rot_eps)

    def solve_seeds(
        self, pose: torch.Tensor, seeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retain per-seed IK and limit validity rather than an aggregated row mask.

        Args:
            pose: One root-relative TCP target.
            seeds: CPU candidate matrix.

        Returns:
            Per-seed validity and mapped joint rows, retaining source order.
        """
        if pose.device.type != "cpu" or seeds.device.type != "cpu":
            raise ValueError("CPU IK branch search requires CPU inputs.")
        record_search("cpu_ik_seed_batches", budget_checkpoint=False)
        record_search("cpu_ik_seeds", count=len(seeds), budget_checkpoint=False)
        flange = pose @ torch.linalg.inv(
            torch.as_tensor(self._solver.tcp_xpos, dtype=pose.dtype)
        )
        ok, q = self._solver._compute_inverse_kinematics(
            flange.expand(len(seeds), -1, -1), seeds
        )
        valid, q = self._solver._qpos_map_to_limits(q.reshape(-1, seeds.shape[-1]))
        return ok.reshape(-1) & valid & self.pose_valid(q, pose), q

    def path(self, poses: torch.Tensor, start: torch.Tensor) -> torch.Tensor | None:
        """Run the existing bounded continuous path search under its source RNG law.

        Args:
            poses: Root-relative CPU path samples, shape ``(T,4,4)``.
            start: One initial joint posture ``(1,D)``.

        Returns:
            Unchanged-grid joint rows including the start, or None on exhaustion.
        """
        self._validate_start(start)
        if poses.device.type != "cpu":
            raise ValueError("CPU IK paths require CPU targets.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            return resolve_seeded_ik_path(
                poses,
                start,
                solve=lambda pose, seed: self._solver.get_ik(pose, qpos_seed=seed),
                solve_many=self.solve_seeds,
                pose_valid=self.pose_valid,
                lower=self._lower,
                upper=self._upper,
                limits=self._limits,
                dt=self._dt,
                record_operation=lambda operation, count: record_search(
                    operation, count=count
                ),
            )

    def endpoints(
        self, pose: torch.Tensor, start: torch.Tensor
    ) -> list[tuple[int, torch.Tensor]]:
        """Resolve the source fixed population and select up to four stable branches.

        Args:
            pose: One CPU root-relative TCP transform.
            start: One actual controlled-joint posture ``(1,D)``.

        Returns:
            Original seed indices and owned joint rows, in source selection order.
        """
        self._validate_start(start)
        if pose.shape != (1, 4, 4) or pose.device.type != "cpu":
            raise ValueError("CPU IK endpoints require one root-relative pose.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            seeds = QposSeedSampler(32, start.shape[-1], start.device).sample(
                start, self._lower, self._upper, 1
            )
            valid, q = self.solve_seeds(pose, seeds)
        indices = select_separated_joint_branches(q, valid, start)
        record_search(
            "cpu_ik_selected_branches", count=len(indices), budget_checkpoint=False
        )
        return [(index, q[index : index + 1].clone()) for index in indices]

    def _validate_start(self, start):
        if (
            start.ndim != 2
            or start.shape != (1, self._lower.numel())
            or start.device.type != "cpu"
            or not start.is_floating_point()
            or not torch.isfinite(start).all()
        ):
            raise ValueError("CPU IK search requires one finite CPU joint start.")
