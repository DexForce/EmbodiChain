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
"""Configuration-search IK for offset seven-revolute-joint manipulators.

Follows the numerical FEP facade in HolisticMotion (ef044f52), with Torch and
fused Warp correction. No analytical geometry model is assumed.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch

from embodichain.compute.kinematics import _fep
from embodichain.utils import configclass
from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain
from .base_solver import BaseSolver, SolverCfg

__all__ = ["FEPSolverCfg", "FEPSolver"]

_METHODS = {
    "seeded_numerical",
    "configuration",
    "all_configurations",
    "nearest_redundancy",
    "compatibility",
}


@configclass
class FEPSolverCfg(SolverCfg):
    """FEP search and numerical correction settings.

    ``configuration`` preserves the signs of joints 2, 4 and 6. The seventh
    joint is a seed parameter, not a locked redundancy constraint. All methods
    are local numerical searches and may miss reachable solutions.
    """

    class_type: str = "FEPSolver"
    backend: str = "auto"
    """auto uses fused Warp on CUDA and Torch on CPU; torch/warp force a backend."""
    solve_method: str = "seeded_numerical"
    max_iterations: int = 200
    damping: float = 0.01
    max_step: float = 0.35
    position_tolerance: float = 1e-5
    rotation_tolerance: float = 1e-5
    redundancy_step: float = math.pi / 36
    redundancy_range: float = math.pi
    batch_size: int | None = None
    """Target chunk size; None uses 256 CPU targets or up to 16384 CUDA seeds.

    CUDA divides the seed budget by the method's candidate count, bounding
    scratch memory when enumerating eight configurations.
    """

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "torch", "warp"}:
            raise ValueError(f"Unknown FEP backend: {self.backend}")
        if self.solve_method not in _METHODS:
            raise ValueError(f"Unknown FEP solve_method: {self.solve_method}")
        for name in ("max_iterations", "batch_size"):
            if name == "batch_size" and self.batch_size is None:
                continue
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "damping",
            "max_step",
            "position_tolerance",
            "rotation_tolerance",
            "redundancy_step",
            "redundancy_range",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    def init_solver(
        self, device: str | torch.device = "cpu", **kwargs: Any
    ) -> FEPSolver:
        """Construct a solver with the configured chain and TCP.

        Args:
            device: Torch execution device.
            **kwargs: Optional ``pk_serial_chain`` supplied by the robot.

        Returns:
            Configured seven-axis solver.
        """
        solver = FEPSolver(self, device=device, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


class FEPSolver(BaseSolver):
    """Seven-axis FEP configuration search using bounded numerical IK.

    Args:
        cfg: Model, search policy and convergence tolerances.
        device: Torch execution device.
        **kwargs: Optional ``pk_serial_chain`` in solver joint order.
    """

    def __init__(
        self, cfg: FEPSolverCfg, device: str | torch.device = "cpu", **kwargs: Any
    ) -> None:
        cfg.__post_init__()
        chain = kwargs.pop("pk_serial_chain", None)
        if chain is None:
            chain = create_pk_serial_chain(
                urdf_path=cfg.urdf_path,
                end_link_name=cfg.end_link_name,
                root_link_name=cfg.root_link_name,
                device=torch.device(device),
            )
        names = chain.get_joint_parameter_names()
        if len(names) != 7 or any(
            j.joint_type != "revolute" for j in chain.get_joints()
        ):
            raise ValueError(
                "FEP requires a serial chain with exactly seven revolute joints"
            )
        if cfg.joint_names is not None and list(cfg.joint_names) != names:
            raise ValueError("FEP joint_names must match the serial chain order")
        resolved = cfg.copy()
        resolved.joint_names = names
        super().__init__(resolved, device=device, pk_serial_chain=chain, **kwargs)
        # BaseSolver leaves compiled_fk unset for an injected chain.
        self.compiled_fk = chain.forward_kinematics_tensor
        self.set_tcp(cfg._get_tcp_as_numpy())
        self._warp_model = None
        if cfg.backend == "warp" or (
            cfg.backend == "auto" and self.device.type == "cuda"
        ):
            from ._fep_warp import _FEPWarpModel

            self._warp_model = _FEPWarpModel(chain, self.device)

    def get_configuration(self, qpos: torch.Tensor) -> torch.Tensor:
        """Extract the numerical branch descriptor.

        Args:
            qpos: Joint positions ``(..., 7)`` in radians.

        Returns:
            ``(..., 4)`` tensor: shoulder, elbow, wrist signs and joint 7 angle.
            A zero joint angle belongs to the positive branch.
        """
        qpos = torch.as_tensor(qpos, device=self.device, dtype=torch.float32)
        if qpos.ndim < 1 or qpos.shape[-1] != 7 or not bool(torch.isfinite(qpos).all()):
            raise ValueError("Expected finite joint positions ending in seven joints")
        return _fep.configuration(qpos)

    def _solve_seed(
        self, target: torch.Tensor, seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._warp_model is None:
            return self._solve_seed_torch(target, seed)
        _, joints = self._warp_model.solve(
            target,
            seed,
            self.lower_qpos_limits,
            self.upper_qpos_limits,
            self.tcp_xpos,
            self.cfg,
        )
        # Native arithmetic is independently checked with the public FK path.
        error = _fep.pose_error(target, self.get_fk(joints))
        # The kernel targets a stricter residual to leave float32 rounding
        # margin. A budget-exhausted candidate can still meet the public
        # tolerance, so the public FK check is authoritative for acceptance.
        valid = error[:, :3].norm(dim=-1) <= self.cfg.position_tolerance
        valid &= error[:, 3:].norm(dim=-1) <= self.cfg.rotation_tolerance
        valid &= torch.isfinite(joints).all(dim=-1)
        valid &= (
            (joints >= self.lower_qpos_limits) & (joints <= self.upper_qpos_limits)
        ).all(dim=-1)
        failed = (~valid).nonzero().flatten()
        if failed.numel():
            valid[failed], joints[failed] = self._solve_seed_torch(
                target.index_select(0, failed), seed.index_select(0, failed)
            )
        return valid, joints

    def _solve_seed_torch(
        self, target: torch.Tensor, seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tcp = torch.as_tensor(self.tcp_xpos, device=seed.device, dtype=seed.dtype)

        def fk(qpos: torch.Tensor) -> torch.Tensor:
            return self.compiled_fk(qpos)[-1] @ tcp

        def jacobian(qpos: torch.Tensor) -> torch.Tensor:
            transforms = self.compiled_fk(qpos)
            jac = self.pk_serial_chain.jacobian_tensor(qpos, all_transforms=transforms)
            offset = (transforms[-1, :, :3, :3] @ tcp[:3, 3])[:, None, :]
            correction = torch.cross(
                jac[:, 3:, :].transpose(1, 2), offset.expand(-1, 7, -1), dim=-1
            )
            return torch.cat(
                (jac[:, :3] + correction.transpose(1, 2), jac[:, 3:]), dim=1
            )

        return _fep.solve_seed(
            target,
            seed,
            self.lower_qpos_limits.to(seed),
            self.upper_qpos_limits.to(seed),
            fk,
            jacobian,
            max_iterations=self.cfg.max_iterations,
            damping=self.cfg.damping,
            max_step=self.cfg.max_step,
            position_tolerance=self.cfg.position_tolerance,
            rotation_tolerance=self.cfg.rotation_tolerance,
        )

    def _rank(
        self, valid: torch.Tensor, candidates: torch.Tensor, seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cost = (
            (_fep.wrapped_distance(candidates, seed[:, None]) * self.ik_nearest_weight)
            .square()
            .sum(-1)
        )
        order = cost.masked_fill(~valid, torch.inf).argsort(dim=1, stable=True)
        return valid.gather(1, order), candidates.gather(
            1, order[..., None].expand(-1, -1, 7)
        )

    def _search(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        method: str,
        configuration: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(seed)
        if method in {"seeded_numerical", "compatibility"}:
            valid, qpos = self._solve_seed(target, seed)
            return valid[:, None], qpos[:, None]
        if method == "nearest_redundancy":
            valid, qpos = self._solve_seed(target, seed)
            lower = self.lower_qpos_limits.to(seed)
            upper = self.upper_qpos_limits.to(seed)
            previous = seed[:, 6:7].clamp(lower[6], upper[6]).expand(-1, 2).clone()
            for index in range(
                1,
                math.floor(self.cfg.redundancy_range / self.cfg.redundancy_step + 1e-10)
                + 1,
            ):
                pending = ~valid
                if not bool(pending.any()):
                    break
                trial = seed[pending, None].expand(-1, 2, -1).clone()
                delta = index * self.cfg.redundancy_step
                trial[:, :, 6] += seed.new_tensor([-delta, delta])
                trial = trial.clamp(lower, upper)
                fresh = trial[:, :, 6] != previous[pending]
                previous[pending] = trial[:, :, 6]
                if not bool(fresh.any()):
                    continue
                targets = target[pending, None].expand(-1, 2, -1, -1)
                ok = torch.zeros_like(fresh)
                solved = trial.clone()
                ok[fresh], solved[fresh] = self._solve_seed(
                    targets[fresh], trial[fresh]
                )
                ok, solved = self._rank(ok, solved, seed[pending])
                qpos[pending] = solved[:, 0]
                valid[pending] = ok[:, 0]
            return valid[:, None], qpos[:, None]
        if method == "configuration":
            config = (
                _fep.configuration(seed) if configuration is None else configuration
            )[:, None]
        else:
            signs = seed.new_tensor(list(itertools.product((-1, 1), repeat=3)))
            config = torch.cat(
                (signs[None].expand(n, -1, -1), seed[:, None, 6:7].expand(-1, 8, -1)),
                dim=-1,
            )
        k = config.shape[1]
        trial = seed[:, None].expand(-1, k, -1).clone()
        trial[..., [1, 3, 5]] = config[..., :3] * trial[..., [1, 3, 5]].abs().clamp_min(
            0.35
        )
        trial[..., 6] = config[..., 3]
        valid, qpos = self._solve_seed(
            target[:, None].expand(-1, k, -1, -1).reshape(-1, 4, 4),
            trial.reshape(-1, 7),
        )
        qpos, valid = qpos.reshape(n, k, 7), valid.reshape(n, k)
        valid &= (_fep.configuration(qpos)[..., :3] == config[..., :3]).all(-1)
        for i in range(1, k):
            duplicate = (
                _fep.wrapped_distance(qpos[:, i : i + 1], qpos[:, :i]).square().sum(-1)
                < 1e-12
            )
            valid[:, i] &= ~(duplicate & valid[:, :i]).any(-1)
        return self._rank(valid, qpos, seed)

    @torch.no_grad()
    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor | None = None,
        return_all_solutions: bool = False,
        *,
        solve_method: str | None = None,
        configuration: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve TCP poses relative to the chain root.

        Args:
            target_xpos: Target transforms ``(N, 4, 4)`` or ``(4, 4)``.
            qpos_seed: Seeds ``(N, 7)`` or broadcast ``(7,)``; defaults to zero.
            return_all_solutions: Return candidate masks and positions.
            solve_method: Optional override of the configured search method.
            configuration: ``(4,)`` or ``(N, 4)`` descriptor for configuration
                mode, with signs exactly -1/+1 and a finite redundancy seed.
            **kwargs: Reserved common solver arguments.

        Returns:
            Nearest mode: validity ``(N,)`` and joints ``(N, 7)``. All mode:
            validity ``(N, K)`` and joints ``(N, K, 7)``, where K is eight for
            all_configurations and one otherwise. Invalid slots contain the
            clamped input seed. Candidates are ranked by wrapped weighted
            distance; all mode enumerates numerical branches, not all IK roots.

        Raises:
            ValueError: Unsupported method, malformed inputs, or invalid limits.
        """
        method = self.cfg.solve_method if solve_method is None else solve_method
        if method not in _METHODS:
            raise ValueError(f"Unknown FEP solve_method: {method}")
        target = torch.as_tensor(target_xpos, device=self.device, dtype=torch.float32)
        if target.ndim == 2:
            target = target[None]
        if (
            target.ndim != 3
            or target.shape[1:] != (4, 4)
            or not bool(torch.isfinite(target).all())
        ):
            raise ValueError("Expected finite target poses with shape (N, 4, 4)")
        n = len(target)
        seed = (
            torch.zeros(n, 7, device=self.device)
            if qpos_seed is None
            else torch.as_tensor(qpos_seed, device=self.device, dtype=torch.float32)
        )
        if seed.shape == (7,):
            seed = seed.expand(n, -1)
        if seed.shape != (n, 7) or not bool(torch.isfinite(seed).all()):
            raise ValueError("Expected finite seeds with shape (7,) or (N, 7)")
        if configuration is not None:
            configuration = torch.as_tensor(
                configuration, device=self.device, dtype=torch.float32
            )
            if configuration.shape == (4,):
                configuration = configuration.expand(n, -1)
            if (
                method != "configuration"
                or configuration.shape != (n, 4)
                or not bool(torch.isfinite(configuration).all())
                or not bool((configuration[:, :3].abs() == 1).all())
            ):
                raise ValueError(
                    "configuration mode requires finite descriptors (N, 4) with signs -1/+1"
                )
        lower, upper = self.lower_qpos_limits.to(seed), self.upper_qpos_limits.to(seed)
        if (
            lower.shape != (7,)
            or upper.shape != (7,)
            or not bool(
                (torch.isfinite(lower) & torch.isfinite(upper) & (lower <= upper)).all()
            )
        ):
            raise ValueError("FEP requires finite ordered joint limits of shape (7,)")
        k = 8 if method == "all_configurations" else 1
        valid = torch.empty(n, k, device=self.device, dtype=torch.bool)
        qpos = torch.empty(n, k, 7, device=self.device)
        search_width = 2 if method == "nearest_redundancy" else k
        batch_size = self.cfg.batch_size or (
            16384 // search_width if self.device.type == "cuda" else 256
        )
        for start in range(0, n, batch_size):
            end = start + batch_size
            valid[start:end], qpos[start:end] = self._search(
                target[start:end],
                seed[start:end],
                method,
                None if configuration is None else configuration[start:end],
            )
        qpos = torch.where(valid[..., None], qpos, seed.clamp(lower, upper)[:, None])
        if return_all_solutions:
            return valid, qpos
        return valid[:, 0], qpos[:, 0]
