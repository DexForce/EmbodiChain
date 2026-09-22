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
"""Franka-family geometric FEP IK with optional redundancy optimization.

Uses the axis reconstruction in HolisticMotion's FEPGeometry.cpp (64b97a5),
following GeoFIK (https://arxiv.org/abs/2503.03992), and search/pruning ideas
from FEPAuto. Geometry is extracted from the URDF chain; incompatible
seven-axis chains are rejected explicitly.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from embodichain.compute.kinematics import yoshikawa_manipulability
from embodichain.utils import configclass
from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain
from .base_solver import BaseSolver, SolverCfg

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["FEPSolverCfg", "FEPSolver"]


@configclass
class FEPSolverCfg(SolverCfg):
    """Geometric IK tolerances and optional sampled redundancy optimization."""

    class_type: str = "FEPSolver"
    position_tolerance: float = 1e-5
    rotation_tolerance: float = 1e-5
    batch_size: int | None = None
    """Targets per chunk; default 16384, capped at 4096 for search or 1024 for Jacobians."""
    ik_solution_selection: str = "nearest"
    """Select by weighted seed distance or ``"manipulability"``.

    Manipulability uses the shared Yoshikawa metric on the valid fixed-q7
    branches. With redundancy search, it ranks only the eight candidates
    retained by the continuity, arm-angle and limit-margin scores.
    """
    num_samples: int | None = None
    """Unsupported numerical multi-start option; use ``redundancy_search`` instead."""
    redundancy_search: bool = False
    """Search q7 near the seed, expanding and refining when needed."""
    arm_angle: float | None = None
    """Preferred GeoFIK swivel angle in radians; None prefers the seed angle."""
    arm_angle_weight: float = 0.1
    """Weight on squared wrapped arm-angle error during search."""
    joint_limit_weight: float = 1e-4
    """Weight on inverse normalized limit margins during search."""
    max_joint_step: float | None = None
    """Hard per-joint displacement bound from the seed, in radians, during search."""

    def __post_init__(self) -> None:
        if self.ik_solution_selection not in ("nearest", "manipulability"):
            raise ValueError(
                "ik_solution_selection must be 'nearest' or 'manipulability'"
            )
        if self.num_samples is not None:
            raise ValueError("FEP does not support num_samples; use redundancy_search")
        if self.batch_size is not None and (
            type(self.batch_size) is not int or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        for name in ("position_tolerance", "rotation_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.redundancy_search) is not bool:
            raise ValueError("redundancy_search must be a boolean")
        for name in ("arm_angle_weight", "joint_limit_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.arm_angle is not None and not math.isfinite(self.arm_angle):
            raise ValueError("arm_angle must be finite")
        if self.max_joint_step is not None and (
            not math.isfinite(self.max_joint_step) or self.max_joint_step <= 0
        ):
            raise ValueError("max_joint_step must be finite and positive")
        if not self.redundancy_search and (
            self.arm_angle is not None or self.max_joint_step is not None
        ):
            raise ValueError("arm_angle and max_joint_step require redundancy_search")

    def init_solver(
        self, device: str | torch.device = "cpu", **kwargs: Any
    ) -> FEPSolver:
        """Construct a geometric solver with the configured URDF chain and TCP.

        Args:
            device: CPU or CUDA execution device.
            **kwargs: Optional ``pk_serial_chain`` in solver joint order.

        Returns:
            Configured Franka-family solver.
        """
        return FEPSolver(self, device=device, **kwargs)


class FEPSolver(BaseSolver):
    """Geometric IK for the Franka axis layout, with up to eight branches.

    By default the seed fixes q7 and selects the nearest discrete branch.
    Optional q7 sampling scores continuity, arm-angle preference and limit
    margin while preserving geometric FK acceptance. Finite sampling does not
    guarantee a global optimum or establish global unreachability. CPU and CUDA
    share the Warp kernels. No collision or bimanual constraints are enforced.

    Args:
        cfg: Model and FK acceptance tolerances.
        device: CPU or CUDA execution device.
        **kwargs: Optional ``pk_serial_chain`` in solver joint order.
    """

    def __init__(
        self, cfg: FEPSolverCfg, device: str | torch.device = "cpu", **kwargs: Any
    ) -> None:
        import warp as wp

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
        dimensions, scales, base_inverse, tool, origins, axes = _extract_geometry(chain)
        resolved = cfg.copy()
        resolved.joint_names = names
        super().__init__(resolved, device=device, pk_serial_chain=chain, **kwargs)
        self.compiled_fk = chain.forward_kinematics_tensor
        self._dimensions, self._scales, self._local_axes = [
            torch.as_tensor(value, device=self.device, dtype=torch.float64)
            for value in (dimensions, scales, axes)
        ]
        self._model_frames = np.concatenate(
            (base_inverse[None], np.eye(4)[None], origins)
        )
        self._end_tool = tool
        self._ready = None
        self.set_tcp(cfg._get_tcp_as_numpy())
        wp.init()

    def set_tcp(self, xpos: np.ndarray) -> None:
        """Set the TCP and refresh fixed transforms used by geometric IK.

        Args:
            xpos: TCP transform relative to the configured end link.
        """
        super().set_tcp(xpos)
        frames = self._model_frames.copy()
        frames[1] = np.linalg.inv(self._end_tool @ self.tcp_xpos)
        frames[-1] = frames[-1] @ self.tcp_xpos
        self._frames = torch.tensor(frames, device=self.device, dtype=torch.float64)
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            if self._ready is not None:
                current.wait_event(self._ready)
            self._ready = torch.cuda.Event()
            self._ready.record(current)

    def _generate_branches(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        stream: Any,
        guide_limits: bool = False,
        q7_samples: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        score_bounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import warp as wp
        from embodichain.compute.kinematics._warp.fep import solve

        if q7_samples is None:
            q7_samples = seed[:, 6:7]
        n = len(seed) * q7_samples.shape[1]
        # Eight branches per q7 probe, plus one feasible seed per target.
        shape = (len(seed), q7_samples.shape[1] * 8 + 1)
        flags = torch.empty(shape, device=self.device, dtype=torch.int32)
        candidates = seed.new_empty((*shape, 7))
        wp.launch(
            solve,
            dim=(n, 8),
            inputs=[
                wp.from_torch(target, dtype=wp.mat44f),
                wp.from_torch(seed),
                wp.from_torch(q7_samples),
                wp.from_torch(lower),
                wp.from_torch(upper),
                wp.from_torch(self._dimensions),
                wp.from_torch(self._scales),
                wp.from_torch(self._frames, dtype=wp.mat44d),
                wp.from_torch(self._local_axes, dtype=wp.vec3d),
                wp.float64(self.cfg.position_tolerance),
                wp.float64(self.cfg.rotation_tolerance),
                guide_limits,
                wp.from_torch(weights if weights is not None else lower),
                wp.from_torch(
                    score_bounds if score_bounds is not None else self._dimensions
                ),
                wp.float64(self.cfg.max_joint_step or math.inf),
                score_bounds is not None and not guide_limits,
            ],
            outputs=[wp.from_torch(flags), wp.from_torch(candidates)],
            device=str(self.device),
            stream=stream,
        )
        return flags, candidates

    def _solve_branches(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        joints: torch.Tensor,
        stream: Any,
    ) -> None:
        import warp as wp
        from embodichain.compute.kinematics._warp.fep import select

        flags, candidates = self._generate_branches(target, seed, lower, upper, stream)
        wp.launch(
            select,
            dim=len(seed),
            inputs=[
                wp.from_torch(seed),
                wp.from_torch(lower),
                wp.from_torch(upper),
                wp.from_torch(weights),
                wp.from_torch(flags),
                wp.from_torch(candidates),
            ],
            outputs=[wp.from_torch(valid), wp.from_torch(joints)],
            device=str(self.device),
            stream=stream,
        )

    @torch.no_grad()
    def get_arm_angle(self, qpos: torch.Tensor) -> torch.Tensor:
        """Measure the oriented shoulder/elbow/joint-7 plane angle.

        This is GeoFIK's swivel convention in the extracted canonical shoulder
        frame, independent of TCP. It is not the seventh joint angle.

        Args:
            qpos: Joint positions ``(7,)`` or ``(N, 7)`` in solver order.

        Returns:
            Angles ``(N,)`` in radians in [-pi, pi]; NaN where the reference
            plane or elbow plane is singular.

        Raises:
            ValueError: Joint positions are malformed or nonfinite.
        """
        joints = torch.as_tensor(qpos, device=self.device, dtype=torch.float32)
        if joints.shape == (7,):
            joints = joints[None]
        if (
            joints.ndim != 2
            or joints.shape[1] != 7
            or not bool(torch.isfinite(joints).all())
        ):
            raise ValueError(
                "Expected finite joint positions with shape (7,) or (N, 7)"
            )
        stream = None
        if self.device.type == "cuda":
            import warp as wp

            current = torch.cuda.current_stream(self.device)
            current.wait_event(self._ready)
            for tensor in (joints, self._dimensions, self._scales):
                tensor.record_stream(current)
            stream = wp.stream_from_torch(current)
        return self._arm_angles(joints.contiguous(), stream)

    def _arm_angles(self, joints: torch.Tensor, stream: Any) -> torch.Tensor:
        import warp as wp
        from embodichain.compute.kinematics._warp.fep import arm_angles

        angles = joints.new_empty(len(joints))
        if len(joints):
            wp.launch(
                arm_angles,
                dim=len(joints),
                inputs=[
                    wp.from_torch(joints),
                    wp.from_torch(self._dimensions),
                    wp.from_torch(self._scales),
                ],
                outputs=[wp.from_torch(angles)],
                device=str(self.device),
                stream=stream,
            )
        return angles

    def _search_redundancy(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        weights: torch.Tensor,
        reference: torch.Tensor,
        count: int,
        stream: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import warp as wp
        from embodichain.compute.kinematics._warp.fep import (
            score_redundancy,
            select_redundancy,
        )

        n = len(seed)
        valid = torch.zeros((n, count), dtype=torch.bool, device=self.device)
        joints = seed[:, None].expand(-1, count, -1).clamp(lower, upper).clone()
        scores = torch.full(
            (n, count), torch.inf, dtype=torch.float64, device=self.device
        )
        max_step = self.cfg.max_joint_step or math.inf
        lo = torch.maximum(lower[6], seed[:, 6] - max_step)
        hi = torch.minimum(upper[6], seed[:, 6] + max_step)

        def evaluate(
            rows: torch.Tensor, samples: torch.Tensor, guidance: bool = False
        ) -> None:
            m = len(rows)
            if not m:
                return
            # Broadcast target/seed by indexing in the kernel, avoiding a full
            # pose and seven-joint copy for every q7 probe.
            row_seed, row_reference = (
                seed[rows].contiguous(),
                reference[rows].contiguous(),
            )
            flags, candidates = self._generate_branches(
                target[rows].contiguous(),
                row_seed,
                lower,
                upper,
                stream,
                guide_limits=guidance,
                q7_samples=samples.clamp(lower[6], upper[6]),
                weights=weights,
                score_bounds=scores[rows, -1].contiguous(),
            )
            costs = torch.empty(flags.shape, dtype=torch.float64, device=self.device)
            guide_costs = torch.empty_like(costs) if guidance else costs
            wp.launch(
                score_redundancy,
                dim=flags.shape,
                inputs=[
                    wp.from_torch(row_seed),
                    wp.from_torch(candidates),
                    wp.from_torch(flags),
                    wp.from_torch(row_reference),
                    wp.from_torch(lower),
                    wp.from_torch(upper),
                    wp.from_torch(weights),
                    wp.from_torch(self._dimensions),
                    wp.from_torch(self._scales),
                    wp.float64(self.cfg.arm_angle_weight),
                    wp.float64(self.cfg.joint_limit_weight),
                    wp.float64(max_step),
                    guidance,
                ],
                outputs=[wp.from_torch(costs), wp.from_torch(guide_costs)],
                device=str(self.device),
                stream=stream,
            )
            # Merge previous winners and update guidance directly in the full
            # batch. Placeholder guide arrays are untouched outside recovery.
            wp.launch(
                select_redundancy,
                dim=m,
                inputs=[
                    wp.from_torch(rows),
                    wp.from_torch(seed),
                    wp.from_torch(lower),
                    wp.from_torch(upper),
                    wp.from_torch(candidates),
                    wp.from_torch(costs),
                    wp.from_torch(guide_costs),
                    guidance,
                ],
                outputs=[
                    wp.from_torch(valid),
                    wp.from_torch(joints),
                    wp.from_torch(scores),
                    wp.from_torch(guide_center if guidance else seed[:, 6]),
                    wp.from_torch(guide_score if guidance else scores[:, 0]),
                ],
                device=str(self.device),
                stream=stream,
            )

        rows = torch.arange(n, device=self.device)
        center = seed[:, 6].clamp(lower[6], upper[6])
        radius = min(0.2, max_step)
        grid = torch.linspace(-1, 1, 9, device=self.device, dtype=seed.dtype)
        evaluate(
            rows,
            torch.maximum(
                lo[:, None], torch.minimum(hi[:, None], center[:, None] + radius * grid)
            ),
        )
        width = (upper - lower).clamp_min(1e-8)
        margin = torch.minimum(joints[:, 0] - lower, upper - joints[:, 0]) / width
        margin[:, upper == lower] = 1
        expand = (
            (~valid[:, 0])
            | (margin.amin(-1) < 0.05)
            | ((joints[:, 0] - seed).norm(dim=-1) > 0.5)
        )
        global_rows = expand.nonzero().flatten()
        global_grid = torch.linspace(0, 1, 33, device=self.device, dtype=seed.dtype)
        evaluate(
            global_rows,
            lo[global_rows, None] + (hi - lo)[global_rows, None] * global_grid,
        )
        # Two finer grids around the best sampled q7; never relax hard bounds.
        spacing = seed.new_full((n,), radius / 4)
        spacing[global_rows] = (hi - lo)[global_rows] / 32
        # Reuse the device grid's +/-1, +/-0.5 points; the center was already
        # evaluated when it became the retained solution/guide.
        fine_grid = torch.cat((grid[:4:2], grid[6::2]))
        for _ in range(2):
            samples = joints[:, 0, 6, None] + spacing[:, None] * fine_grid
            evaluate(
                rows, torch.maximum(lo[:, None], torch.minimum(hi[:, None], samples))
            )
            spacing *= 0.5
        # A narrow feasible interval can fall between all coarse q7 probes.
        # Refine only failed targets, guided by normalized constraint violation.
        failed = (~valid[:, 0]).nonzero().flatten()
        if not len(failed):
            return valid, joints
        # Allocate guidance only when needed; the ordinary path neither reads
        # these closure variables nor performs the extra candidate reductions.
        guide_center = seed[:, 6].clamp(lower[6], upper[6]).clone()
        guide_score = torch.full(
            (n,), torch.inf, device=self.device, dtype=torch.float64
        )
        guide_grid = torch.linspace(0, 1, 65, device=self.device, dtype=seed.dtype)
        evaluate(
            failed,
            lo[failed, None] + (hi - lo)[failed, None] * guide_grid,
            guidance=True,
        )
        spacing = (hi - lo) / 64
        # Like HolisticMotion's denser global coverage, but spend the extra
        # probes only where no geometric branch can guide local refinement.
        # Insert midpoints rather than recomputing the existing grid nodes.
        for intervals in (64, 128):
            missing = (
                ((~valid[:, 0]) & ~torch.isfinite(guide_score)).nonzero().flatten()
            )
            if not len(missing):
                break
            midpoints = (
                torch.arange(intervals, device=self.device, dtype=seed.dtype) + 0.5
            ) / intervals
            evaluate(
                missing,
                lo[missing, None] + (hi - lo)[missing, None] * midpoints,
                guidance=True,
            )
            spacing[missing] = (hi - lo)[missing] / (2 * intervals)
        for _ in range(8):
            failed = ((~valid[:, 0]) & torch.isfinite(guide_score)).nonzero().flatten()
            if not len(failed):
                break
            samples = guide_center[failed, None] + spacing[failed, None] * fine_grid
            evaluate(
                failed,
                torch.maximum(
                    lo[failed, None], torch.minimum(hi[failed, None], samples)
                ),
                guidance=True,
            )
            spacing *= 0.5
        return valid, joints

    @torch.no_grad()
    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor | None = None,
        return_all_solutions: bool = False,
        arm_angle: float | torch.Tensor | None = None,
        num_samples: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve geometric branches, optionally optimizing the redundant q7.

        Args:
            target_xpos: Rigid TCP targets ``(N, 4, 4)`` or ``(4, 4)`` in the
                chain-root frame.
            qpos_seed: ``(N, 7)``, ``(1, 7)`` or ``(7,)``. Defaults to the
                joint-limit midpoint. q7 stays fixed unless search is enabled.
            return_all_solutions: Return up to eight branches at fixed q7, or
                the eight best distinct sampled solutions during search.
            arm_angle: Optional scalar or ``(N,)`` preferred swivel angle in
                radians, overriding ``cfg.arm_angle`` for this call. Requires
                redundancy search. If both are None, use the seed's arm angle;
                disable that score where the seed angle is undefined. Undefined
                candidate angles are rejected when an angle preference is active.
            num_samples: Must be None. Numerical multi-start sampling is not
                equivalent to FEP's adaptive q7 search; use ``redundancy_search``.
            **kwargs: Reserved common solver arguments.

        Returns:
            Validity ``(N,)`` and joints ``(N, 7)``; with all solutions, shapes
            are ``(N, 8)`` and ``(N, 8, 7)``. Invalid slots contain the clamped
            seed. Failure at one q7 does not prove the target unreachable at
            other q7 values. Singular continuous families use seed representatives.
            ``ik_solution_selection="manipulability"`` selects the highest
            Yoshikawa score from the valid branches, or the eight retained
            search candidates. All-solutions ordering remains by seed distance
            or the search score, regardless of the selection mode.

        Raises:
            ValueError: Malformed inputs/configuration or unsupported ``num_samples``.
        """
        import warp as wp

        self.cfg.__post_init__()
        if num_samples is not None:
            raise ValueError("FEP does not support num_samples; use redundancy_search")
        if arm_angle is not None and not self.cfg.redundancy_search:
            raise ValueError("arm_angle requires redundancy_search")
        target = torch.as_tensor(target_xpos, device=self.device, dtype=torch.float32)
        if target.ndim == 2:
            target = target[None]
        if target.ndim != 3 or target.shape[1:] != (4, 4):
            raise ValueError("Expected finite target poses with shape (N, 4, 4)")
        n = len(target)
        seed = torch.as_tensor(
            self.get_default_qpos_seed() if qpos_seed is None else qpos_seed,
            device=self.device,
            dtype=torch.float32,
        )
        if seed.shape in {(7,), (1, 7)}:
            seed = seed.expand(n, 7)
        if seed.shape != (n, 7):
            raise ValueError("Expected finite seeds with shape (7,), (1, 7) or (N, 7)")
        lower, upper = (
            self.lower_qpos_limits.to(seed).contiguous(),
            self.upper_qpos_limits.to(seed).contiguous(),
        )
        if lower.shape != (7,) or upper.shape != (7,):
            raise ValueError("FEP requires finite ordered joint limits of shape (7,)")
        target, seed = target.contiguous(), seed.contiguous()
        weights = torch.as_tensor(
            self.ik_nearest_weight, device=self.device, dtype=torch.float32
        ).contiguous()
        if weights.shape != (7,):
            raise ValueError("FEP joint weights must have shape (7,)")
        stream = None
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            current.wait_event(self._ready)
            # Warp accesses are invisible to Torch's caching allocator.
            for tensor in (
                target,
                seed,
                lower,
                upper,
                weights,
                self._dimensions,
                self._scales,
                self._local_axes,
                self._frames,
            ):
                tensor.record_stream(current)
            stream = wp.stream_from_torch(current)
        from embodichain.compute.kinematics._warp.fep import validate_inputs

        errors = torch.zeros(1, device=self.device, dtype=torch.int32)
        wp.launch(
            validate_inputs,
            dim=max(n, 1),
            inputs=[
                wp.from_torch(target, dtype=wp.mat44f),
                wp.from_torch(seed),
                wp.from_torch(lower),
                wp.from_torch(upper),
                wp.from_torch(weights),
            ],
            outputs=[wp.from_torch(errors)],
            device=str(self.device),
            stream=stream,
        )
        # One scalar transfer per call, regardless of batch or chunk count.
        code = int(errors.item())
        messages = (
            "Expected finite target poses with shape (N, 4, 4)",
            "Expected finite seeds with shape (7,), (1, 7) or (N, 7)",
            "FEP requires finite ordered joint limits of shape (7,)",
            "FEP target poses must be rigid transforms",
            "FEP joint weights must be finite and nonnegative",
        )
        for bit, message in enumerate(messages):
            if code & (1 << bit):
                raise ValueError(message)
        reference = None
        if self.cfg.redundancy_search:
            preferred = self.cfg.arm_angle if arm_angle is None else arm_angle
            if preferred is None:
                reference = self._arm_angles(seed, stream)
            else:
                reference = torch.as_tensor(
                    preferred, device=self.device, dtype=torch.float32
                )
                if not bool(torch.isfinite(reference).all()):
                    raise ValueError("arm_angle must be finite")
                if reference.ndim == 0 or reference.shape == (1,):
                    reference = reference.expand(n)
                if reference.shape != (n,):
                    raise ValueError("arm_angle must be scalar or shape (N,)")
                reference = reference.contiguous()
                if self.device.type == "cuda":
                    reference.record_stream(torch.cuda.current_stream(self.device))
        select_manipulability = (
            self.cfg.ik_solution_selection == "manipulability"
            and not return_all_solutions
        )
        count = 8 if return_all_solutions or select_manipulability else 1
        valid = torch.empty((n, count), device=self.device, dtype=torch.bool)
        joints = seed.new_empty((n, count, 7))
        batch_size = self.cfg.batch_size or 16384
        if self.cfg.redundancy_search:
            # Bound candidate memory even when q7 expands to 33 probes.
            batch_size = min(batch_size, 4096)
        if select_manipulability:
            # Bound the batched Jacobian workspace for the eight-candidate pool.
            batch_size = min(batch_size, 1024)
        for start in range(0, n, batch_size):
            end = start + batch_size
            if self.cfg.redundancy_search:
                valid[start:end], joints[start:end] = self._search_redundancy(
                    target[start:end],
                    seed[start:end],
                    lower,
                    upper,
                    weights,
                    reference[start:end],
                    count,
                    stream,
                )
            else:
                self._solve_branches(
                    target[start:end],
                    seed[start:end],
                    lower,
                    upper,
                    weights,
                    valid[start:end],
                    joints[start:end],
                    stream,
                )
            if select_manipulability:
                mask, candidates = valid[start:end], joints[start:end]
                scores = candidates.new_full(mask.shape, -torch.inf)
                feasible = candidates[mask]
                if len(feasible):
                    scores[mask] = yoshikawa_manipulability(self.get_jacobian(feasible))
                # Mirror-symmetric branches have the same theoretical score,
                # but determinant rounding can differ across devices/chunks.
                # Treat relative 1e-5 score differences as ties and preserve
                # continuity by selecting the closest tied branch to the seed.
                maximum = scores.amax(dim=1, keepdim=True)
                tolerance = 1e-8 + 1e-5 * maximum.abs()
                tied = mask & (scores >= maximum - tolerance)
                distances = (
                    (candidates - seed[start:end, None]).square() * weights[None, None]
                ).sum(dim=-1)
                best = distances.masked_fill(~tied, torch.inf).argmin(dim=1)
                # All-invalid rows retain slot zero's clamped seed fallback.
                selected = candidates[torch.arange(len(mask), device=self.device), best]
                valid[start:end, 0] = mask.any(dim=1)
                joints[start:end, 0] = selected
        if return_all_solutions:
            return valid, joints
        return valid[:, 0], joints[:, 0]


def _extract_geometry(
    chain: pk.SerialChain,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover and validate Franka screw-axis lines in the URDF home pose.

    Fixed frames are accumulated before recording each revolute axis. The
    1e-6 structural tolerance accommodates the chain's float32 transforms;
    solving still requires FK acceptance against that actual chain.
    """
    home, pending = np.eye(4), np.eye(4)
    points, axes, origins, local_axes = [], [], [], []
    for frame in chain._serial_frames:
        for offset in (frame.link.offset, frame.joint.offset):
            if offset is not None:
                transform = (
                    offset.get_matrix()[0].detach().cpu().numpy().astype(np.float64)
                )
                home = home @ transform
                pending = pending @ transform
        if frame.joint.joint_type == "revolute":
            origins.append(pending)
            pending = np.eye(4)
            local_axes.append(
                frame.joint.axis.detach().cpu().numpy().astype(np.float64)
            )
            points.append(home[:3, 3].copy())
            axis = home[:3, :3] @ frame.joint.axis.detach().cpu().numpy()
            axes.append(axis / np.linalg.norm(axis))
    points, axes = np.asarray(points), np.asarray(axes)
    z = axes[0].copy()
    shoulder = points[0] + z * np.dot(z, points[1] - points[0])
    elbow = points[3] + axes[3] * np.dot(axes[3], shoulder - points[3])
    if np.dot(z, elbow - shoulder) < 0:
        z = -z
    x = elbow - shoulder - z * np.dot(z, elbow - shoulder)
    message = "FEP requires Franka-compatible screw-axis geometry; use a numerical solver for other chains"
    if np.linalg.norm(x) <= 1e-6:
        raise ValueError(message)
    x /= np.linalg.norm(x)
    base = np.eye(4)
    base[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    base[:3, 3] = shoulder
    inverse = np.linalg.inv(base)
    wrist = points[4] + z * np.dot(z, points[5] - points[4])
    local_elbow = inverse[:3, :3] @ (elbow - shoulder)
    local_wrist = inverse[:3, :3] @ (wrist - shoulder)
    a4, d3 = local_elbow[0], local_elbow[2]
    a5, d5 = a4 - local_wrist[0], local_wrist[2] - d3
    a7 = np.dot(x, points[6] - wrist)
    dimensions = np.array([d3, d5, a4, a5, a7])
    if not np.isfinite(dimensions).all() or dimensions.min() <= 1e-6:
        raise ValueError(message)
    canonical_axes = np.array(
        [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]]
    )
    canonical_points = np.array(
        [[0, 0, 0]] * 3
        + [
            [a4, 0, d3],
            [a4 - a5, 0, d3 + d5],
            [a4 - a5, 0, d3 + d5],
            [a4 - a5 + a7, 0, d3 + d5],
        ]
    )
    directions = canonical_axes @ base[:3, :3].T
    offsets = canonical_points @ base[:3, :3].T + shoulder - points
    if (
        np.linalg.norm(np.cross(axes, directions), axis=-1).max() > 1e-6
        or np.linalg.norm(np.cross(offsets, directions), axis=-1).max() > 1e-6
    ):
        raise ValueError(message)
    scales = np.sign(np.sum(directions * axes, axis=-1))
    canonical_home = np.diag([1.0, -1.0, -1.0, 1.0])
    canonical_home[:3, 3] = canonical_points[6]
    tool = np.linalg.inv(canonical_home) @ inverse @ home
    dimensions = np.concatenate(
        (
            dimensions,
            [
                np.hypot(d3, a4),
                np.hypot(d5, a5),
                np.arctan2(a4, d3),
                np.arctan2(a5, d5),
            ],
        )
    )
    origins.append(pending)
    return dimensions, scales, inverse, tool, np.stack(origins), np.stack(local_axes)
