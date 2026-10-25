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

"""Model-derived Warp IK and an optional pinned SSIK CPU reference."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import torch

from embodichain.compute.kinematics._ssik_model import _extract_model
from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain
from embodichain.utils import configclass
from embodichain.utils.math import inv_transform
from .base_solver import BaseSolver, SolverCfg
from ._buffers import _with_ik_buffers

__all__ = ["SSIKSolverCfg", "SSIKSolver"]


@configclass
class SSIKSolverCfg(SolverCfg):
    """Configure geometric, prepared algebraic or optional reference IK.

    Warp supports three-parallel-axis 6R and concurrent shoulder/wrist SRS 7R
    models. Explicit ``near_parallel`` mode adds bounded original-model
    polishing for small 6R axis deviations. ``cuda_rr`` evaluates a verified
    offline algebraic bundle on CUDA. The optional CPU reference requires
    ``requirements/ssik-reference.txt``. Failure never invokes another solver.
    """

    class_type: str = "SSIKSolver"
    backend: Literal["warp", "reference", "cuda_rr"] = "warp"
    geometry_mode: Literal["strict", "near_parallel"] = "strict"
    axis_alignment_tolerance: float = 1e-5
    srs_search: Literal["uniform", "seed_local"] = "uniform"
    rr_search: Literal["fixed", "seed_local"] = "fixed"
    num_samples: int = 128
    position_tolerance: float = 1e-5
    rotation_tolerance: float = 1e-5
    max_joint_step: float | None = None
    chunk_size: int = 512
    reference_artifact: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("warp", "reference", "cuda_rr"):
            raise ValueError("SSIK backend must be 'warp', 'reference' or 'cuda_rr'")
        if self.geometry_mode not in ("strict", "near_parallel"):
            raise ValueError("geometry_mode must be 'strict' or 'near_parallel'")
        if self.srs_search not in ("uniform", "seed_local"):
            raise ValueError("srs_search must be 'uniform' or 'seed_local'")
        if self.rr_search not in ("fixed", "seed_local"):
            raise ValueError("rr_search must be 'fixed' or 'seed_local'")
        if self.rr_search != "fixed" and self.backend != "cuda_rr":
            raise ValueError("rr_search='seed_local' requires backend='cuda_rr'")
        if not math.isfinite(self.axis_alignment_tolerance) or not (
            0 < self.axis_alignment_tolerance <= 1e-3
        ):
            raise ValueError("axis_alignment_tolerance must be in (0, 1e-3]")
        for name in ("num_samples", "chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("position_tolerance", "rotation_tolerance", "max_joint_step"):
            value = getattr(self, name)
            if (value is None and name != "max_joint_step") or (
                value is not None and (not math.isfinite(value) or value <= 0)
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.backend != "reference" and self.reference_artifact is not None:
            raise ValueError(
                "reference_artifact is only valid for the reference backend"
            )

    def init_solver(
        self, device: str | torch.device = "cpu", **kwargs: Any
    ) -> SSIKSolver:
        """Build and validate the chain once, then initialize its chosen backend.

        Args:
            device: CPU/CUDA for Warp, CUDA for RR, CPU for the reference.
            **kwargs: Optional ``pk_serial_chain`` in configured joint order.

        Returns:
            Initialized geometric solver with its configured TCP.
        """
        solver = SSIKSolver(self, device, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


class SSIKSolver(BaseSolver):
    """Enumerate finite IK candidates without switching execution backends.

    SRS candidates condition on the seed swivel and a fixed global swivel
    grid, plus a seed-local redundancy search when requested. Prepared RR
    candidates condition on a fixed joint-lock grid, optionally extended by
    seed-local algebraic lock samples. Sampling failure does not establish
    global unreachability. The
    nearest result uses weighted distance to the limit-projected caller seed.
    Public tensors remain owned by the caller across subsequent calls.

    Args:
        cfg: Geometry, backend and acceptance settings.
        device: Execution device; reference is CPU-only and RR is CUDA-only.
        **kwargs: Optional preconstructed PK serial chain.
    """

    def __init__(
        self, cfg: SSIKSolverCfg, device: str | torch.device = "cpu", **kwargs: Any
    ) -> None:
        cfg.__post_init__()
        if cfg.backend == "reference" and torch.device(device).type != "cpu":
            raise ValueError("SSIK reference backend accepts CPU devices only")
        if cfg.backend == "cuda_rr" and torch.device(device).type != "cuda":
            raise ValueError("SSIK cuda_rr backend accepts CUDA devices only")
        if cfg.backend != "reference":
            import warp as wp

            # Base initialization dispatches to CUDA-fenced setters, which
            # borrow Warp streams before this solver's model is packed.
            wp.init()
        chain = kwargs.pop("pk_serial_chain", None)
        if chain is None:
            chain = create_pk_serial_chain(
                urdf_path=cfg.urdf_path,
                root_link_name=cfg.root_link_name,
                end_link_name=cfg.end_link_name,
                device=torch.device(device),
            )
        self._model = _extract_model(
            chain,
            geometry_mode=cfg.geometry_mode,
            axis_alignment_tolerance=cfg.axis_alignment_tolerance,
        )
        if cfg.joint_names is not None and tuple(cfg.joint_names) != self._model.names:
            raise ValueError("SSIK joint_names must match the serial-chain order")
        if cfg.backend == "warp" and self._model.family == "unsupported":
            raise ValueError(
                "SSIK Warp requires three-parallel 6R or strict SRS 7R geometry"
            )
        self._ready: torch.cuda.Event | None = None
        resolved = cfg.copy()
        resolved.joint_names = list(self._model.names)
        super().__init__(resolved, device=device, pk_serial_chain=chain, **kwargs)
        self.compiled_fk = chain.forward_kinematics_tensor
        self.family = self._model.family
        self.model_fingerprint = self._model.fingerprint
        self._reference = None
        self._rr = None
        self._near = None
        self._geometric = None
        if cfg.backend == "reference":
            from ._ssik_reference import _Reference

            self._reference = _Reference(self._model, resolved)
        else:
            import warp as wp
            from embodichain.compute.kinematics._warp.ssik import _Model
            from embodichain.compute.kinematics._warp.ssik_6r import mat73d, mat83d

            wp.init()
            model = self._model
            self._warp_model = _Model()
            self._warp_model.h = mat73d(model.axes)
            self._warp_model.p = mat83d(model.offsets)
            self._warp_model.home = wp.mat33d(model.home_rotation)
            self._warp_model.shoulder = wp.vec3d(model.shoulder)
            self._warp_model.wrist = wp.vec3d(model.wrist)
            self._warp_model.dof = model.dof
            self._model_frames = torch.tensor(
                model.frames, dtype=torch.float64, device=self.device
            )
            self._model_local_axes = torch.tensor(
                model.local_axes, dtype=torch.float64, device=self.device
            )
            self._warp_model.frames = wp.from_torch(self._model_frames)
            self._warp_model.local_axes = wp.from_torch(self._model_local_axes)
            self._axes = torch.tensor(
                model.axes, dtype=torch.float64, device=self.device
            )
            self._offsets = torch.tensor(
                model.offsets, dtype=torch.float64, device=self.device
            )
            if (
                self.device.type == "cuda"
                and cfg.backend == "warp"
                and model.family in ("three_parallel", "srs")
            ):
                from embodichain.compute.kinematics._ssik_geometric import (
                    _GeometricCandidates,
                )

                self._geometric = _GeometricCandidates(
                    model, self.device, cfg.num_samples, cfg.srs_search
                )
            if model.family == "near_parallel" and cfg.backend == "warp":
                from embodichain.compute.kinematics._ssik_near import (
                    _NearParallelCandidates,
                )

                self._near = _NearParallelCandidates(
                    model, self.device, cfg.axis_alignment_tolerance
                )
            if cfg.backend == "cuda_rr":
                from embodichain.compute.kinematics._ssik_rr import (
                    _PreparedRRCandidates,
                )

                self._rr = _PreparedRRCandidates(
                    model, self.device, search=cfg.rr_search
                )
        self.set_tcp(resolved._get_tcp_as_numpy())

    def _wait_ready(self) -> None:
        if self.device.type == "cuda" and self._ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self._ready)

    def _record_ready(self) -> None:
        if self.device.type == "cuda":
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(self.device))

    def _record_warp_inputs(self, *tensors: torch.Tensor) -> None:
        # Warp borrows storage without notifying the Torch allocator. Record
        # consumers so input/model storage survives cross-stream destruction.
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            for tensor in (
                self._model_frames,
                self._model_local_axes,
                self._axes,
                self._offsets,
                self._tcp_inverse_tensor,
                *tensors,
            ):
                tensor.record_stream(stream)

    @_with_ik_buffers
    def set_qpos_limits(self, lower_qpos_limits: Any, upper_qpos_limits: Any) -> bool:
        """Update effective limits with ordering across the solver's CUDA streams.

        Args:
            lower_qpos_limits: Lower bounds in solver joint order.
            upper_qpos_limits: Upper bounds in solver joint order.

        Returns:
            Whether the ordered bounds were accepted by the base solver.
        """
        self._wait_ready()
        result = super().set_qpos_limits(lower_qpos_limits, upper_qpos_limits)
        self._record_ready()
        return result

    @_with_ik_buffers
    def set_ik_nearest_weight(
        self, ik_weight: np.ndarray, joint_ids: np.ndarray | None = None
    ) -> bool:
        """Update selection weights after preceding calls have finished reading.

        Args:
            ik_weight: Weights in the selected joint order.
            joint_ids: Optional subset of solver joint indices.

        Returns:
            The base solver's update result.
        """
        self._wait_ready()
        result = super().set_ik_nearest_weight(ik_weight, joint_ids)
        self._record_ready()
        return result

    @_with_ik_buffers
    def update_with_robot_limit(self, robot_qpos_limits: torch.Tensor) -> None:
        """Intersect robot limits while preserving CUDA stream dependencies.

        Args:
            robot_qpos_limits: Robot hard limits shaped ``(dof,2)``.
        """
        self._wait_ready()
        super().update_with_robot_limit(robot_qpos_limits)
        self._record_ready()

    @_with_ik_buffers
    def set_tcp(self, xpos: np.ndarray) -> None:
        """Update the tool transform without rebuilding chain geometry.

        Args:
            xpos: Valid end-link-to-TCP rigid transform.
        """
        super().set_tcp(xpos)
        self._tcp_translation = tuple(float(value) for value in self.tcp_xpos[:3, 3])
        self._tcp_inverse = inv_transform(np.asarray(xpos, dtype=np.float64))
        self._tcp_inverse_tensor = torch.tensor(
            self._tcp_inverse, dtype=torch.float64, device=self.device
        )
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            if self._ready is not None:
                stream.wait_event(self._ready)
            self._ready = torch.cuda.Event()
            self._ready.record(stream)

    @_with_ik_buffers
    def prepare_buffers(self, max_batch: int) -> None:
        """Reserve bounded scratch; returned all-candidate tensors are separate.

        Args:
            max_batch: Expected positive target count.
        """
        provider = self._rr if self._rr is not None else self._near
        if provider is None:
            provider = self._geometric
        capacity = min(max_batch, self.cfg.chunk_size)
        if provider is not None:
            capacity = min(capacity, provider.max_chunk_size)
        self._ik_buffers.reserve(capacity)
        if provider is not None:
            provider.prepare_buffers(capacity)

    @property
    def supports_continuous_batch_ik(self) -> bool:
        """Whether candidate generation and previous-frame selection are supported."""
        return True

    def _get_continuous_ik_candidates(
        self, target_xpos: torch.Tensor, qpos_seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_ik(
            target_xpos,
            qpos_seed,
            return_all_solutions=True,
            _continuous_candidates=True,
        )

    @_with_ik_buffers
    def _get_continuous_ik_path(
        self, target_xpos: torch.Tensor, qpos_seed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (
            self._near is not None
            or (self._rr is not None and self.cfg.rr_search == "seed_local")
            or (
                self.cfg.backend == "warp"
                and self.family == "srs"
                and self.cfg.srs_search == "seed_local"
            )
        ):
            return super()._get_continuous_ik_path(target_xpos, qpos_seed)
        if target_xpos.ndim != 4 or target_xpos.shape[-2:] != (4, 4):
            raise ValueError("Continuous targets must have shape (B,N,4,4)")
        b, n = target_xpos.shape[:2]
        if qpos_seed.shape != (b, self.dof):
            raise ValueError("Continuous initial seeds must have shape (B,dof)")
        import warp as wp
        from embodichain.compute.kinematics._warp.ssik_srs_path import (
            srs_continuous_path,
        )

        self._wait_ready()
        target = target_xpos.to(device=self.device, dtype=torch.float64).contiguous()
        seed = qpos_seed.to(device=self.device, dtype=torch.float32).contiguous()
        self._record_warp_inputs(target, seed, self.ik_nearest_weight)
        valid = torch.empty((b, n), dtype=torch.bool, device=self.device)
        joints = torch.empty((b, n, self.dof), dtype=torch.float32, device=self.device)
        if not b or not n:
            return valid, joints
        if self.device.type == "cuda" or self._near is not None:
            # Frames depend on the preceding selection, but candidates within
            # each frame remain parallel. Near-parallel correction also needs
            # the current seed, including its appropriate periodic winding.
            finite_initial = torch.isfinite(seed).all(dim=-1)
            previous = torch.where(
                torch.isfinite(seed),
                seed,
                (self.lower_qpos_limits + self.upper_qpos_limits) / 2,
            ).clamp(min=self.lower_qpos_limits, max=self.upper_qpos_limits)
            for frame in range(n):
                accepted, result = self.get_ik(target[:, frame], previous)
                accepted = accepted & finite_initial
                previous = torch.where(accepted[:, None], result, previous)
                valid[:, frame].copy_(accepted)
                joints[:, frame].copy_(previous)
            return valid, joints
        if b and n:
            wp.launch(
                srs_continuous_path,
                dim=b,
                inputs=[
                    self._warp_model,
                    wp.from_torch(self._axes, dtype=wp.vec3d),
                    wp.from_torch(self._offsets, dtype=wp.vec3d),
                    wp.from_torch(target, dtype=wp.mat44d),
                    wp.mat44d(self.tcp_xpos),
                    wp.mat44d(self._tcp_inverse),
                    wp.from_torch(seed),
                    wp.from_torch(self.lower_qpos_limits.double().contiguous()),
                    wp.from_torch(self.upper_qpos_limits.double().contiguous()),
                    wp.from_torch(self.ik_nearest_weight.float().contiguous()),
                    self.cfg.num_samples,
                    self.cfg.position_tolerance,
                    self.cfg.rotation_tolerance,
                    self.cfg.max_joint_step or math.inf,
                ],
                outputs=[wp.from_torch(valid), wp.from_torch(joints)],
                device=str(self.device),
                block_dim=1,
            )
        return valid, joints

    @_with_ik_buffers
    def get_ik(
        self,
        target_xpos: torch.Tensor,
        qpos_seed: torch.Tensor | None = None,
        return_all_solutions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve root-relative TCP targets and return validity before positions.

        Args:
            target_xpos: Poses shaped ``(N, 4, 4)``.
            qpos_seed: Optional seed shaped ``(N, dof)`` or ``(1, dof)``.
            return_all_solutions: Return ``(N,K)`` and ``(N,K,dof)`` rather
                than ``(N,)`` and ``(N,dof)``. SRS K depends on num_samples;
                invalid slots contain a projected seed and a false mask.
            **kwargs: Internal continuous candidate-generation context.

        Returns:
            Boolean validity and float32 joint positions on the solver device.
            All-candidate 7R results cover sampled redundancy values only.
        """
        if target_xpos.ndim != 3 or target_xpos.shape[1:] != (4, 4):
            raise ValueError("SSIK targets must have shape (N,4,4)")
        if not target_xpos.is_floating_point():
            raise ValueError("SSIK targets must be floating point")
        self._wait_ready()
        n = len(target_xpos)
        target = target_xpos.to(device=self.device, dtype=torch.float64).contiguous()
        if qpos_seed is None:
            qpos_seed = self.get_default_qpos_seed()[None]
        if (
            qpos_seed.ndim != 2
            or qpos_seed.shape[-1] != self.dof
            or qpos_seed.shape[0] not in (1, n)
        ):
            raise ValueError("SSIK seeds must have shape (N,dof) or (1,dof)")
        seed = (
            qpos_seed.to(device=self.device, dtype=torch.float64)
            .expand(n, -1)
            .contiguous()
        )
        lower = self.lower_qpos_limits.to(dtype=torch.float64).contiguous()
        upper = self.upper_qpos_limits.to(dtype=torch.float64).contiguous()
        weights = self.ik_nearest_weight.to(dtype=torch.float32).contiguous()
        max_step = self.cfg.max_joint_step or math.inf
        if kwargs.get("_continuous_candidates", False):
            max_step = math.inf
        if self._reference is not None:
            return self._reference.solve(
                self,
                target,
                seed,
                lower,
                upper,
                weights,
                return_all_solutions,
                max_step,
            )
        import warp as wp
        from embodichain.compute.kinematics._warp.ssik import (
            accept_raw,
            select_nearest,
        )

        self._record_warp_inputs(target, seed, lower, upper, weights)
        if self._ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self._ready)
        samples = self.cfg.num_samples if self.dof == 7 and self._rr is None else 0
        provider = self._rr if self._rr is not None else self._near
        if provider is None:
            provider = self._geometric
        local_search = int(self.dof == 7 and self.cfg.srs_search == "seed_local")
        sample_count = samples + 1 + local_search
        count = provider.candidate_count if provider is not None else 8 * sample_count
        chunk = self.cfg.chunk_size
        if provider is not None:
            chunk = min(chunk, provider.max_chunk_size)
        shape = (n, count) if return_all_solutions else (n,)
        valid = torch.empty(shape, dtype=torch.bool, device=self.device)
        joints = torch.empty(
            (*shape, self.dof), dtype=torch.float32, device=self.device
        )
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            m = end - start
            flags = wp.to_torch(
                self._ik_buffers.zeros("ssik_flags", m, count, torch.int32)
            ).reshape(m, count)
            candidates = wp.to_torch(
                self._ik_buffers.zeros("ssik_qpos", m, count * self.dof, torch.float32)
            ).reshape(m, count, self.dof)
            if provider is not None:
                goal = target[start:end] @ self._tcp_inverse_tensor
                safe = torch.where(
                    torch.isfinite(goal).all(dim=(-2, -1))[:, None, None],
                    goal,
                    torch.eye(4, device=self.device, dtype=torch.float64)[None],
                ).contiguous()
                if self._rr is not None and self.cfg.rr_search == "fixed":
                    raw_flags, raw = self._rr.generate(safe)
                else:
                    projected_seed = torch.where(
                        torch.isfinite(seed[start:end]),
                        seed[start:end],
                        (lower + upper) / 2,
                    ).clamp(min=lower, max=upper)
                    if self._rr is not None:
                        raw_flags, raw = self._rr.generate(
                            safe, projected_seed, lower, upper, max_step
                        )
                    elif self._near is not None:
                        raw_flags, raw = self._near.generate(
                            safe,
                            projected_seed,
                            self._tcp_translation,
                            self.cfg.position_tolerance,
                            self.cfg.rotation_tolerance,
                        )
                    else:
                        raw_flags, raw = self._geometric.generate(
                            safe,
                            projected_seed,
                            lower,
                            upper,
                            weights,
                            self.cfg.position_tolerance,
                        )
                wp.launch(
                    accept_raw,
                    dim=(m, count),
                    inputs=[
                        self._warp_model,
                        wp.from_torch(raw),
                        wp.from_torch(raw_flags),
                        wp.from_torch(target[start:end], dtype=wp.mat44d),
                        wp.mat44d(self.tcp_xpos),
                        wp.from_torch(seed[start:end]),
                        wp.from_torch(lower),
                        wp.from_torch(upper),
                        self.cfg.position_tolerance,
                        self.cfg.rotation_tolerance,
                        max_step,
                    ],
                    outputs=[wp.from_torch(flags), wp.from_torch(candidates)],
                    device=str(self.device),
                )
            else:
                from embodichain.compute.kinematics._warp.ssik_legacy import generate

                wp.launch(
                    generate,
                    dim=(m, sample_count),
                    inputs=[
                        self._warp_model,
                        wp.from_torch(self._axes, dtype=wp.vec3d),
                        wp.from_torch(self._offsets, dtype=wp.vec3d),
                        wp.from_torch(target[start:end], dtype=wp.mat44d),
                        wp.mat44d(self._tcp_inverse),
                        wp.mat44d(self.tcp_xpos),
                        wp.from_torch(seed[start:end]),
                        wp.from_torch(lower),
                        wp.from_torch(upper),
                        wp.from_torch(weights),
                        samples,
                        local_search,
                        self.cfg.position_tolerance,
                        self.cfg.rotation_tolerance,
                        max_step,
                    ],
                    outputs=[wp.from_torch(flags), wp.from_torch(candidates)],
                    device=str(self.device),
                )
            if return_all_solutions:
                valid[start:end].copy_(flags != 0)
                joints[start:end].copy_(candidates)
            else:
                wp.launch(
                    select_nearest,
                    dim=m,
                    inputs=[
                        wp.from_torch(candidates),
                        wp.from_torch(flags),
                        wp.from_torch(seed[start:end]),
                        wp.from_torch(lower),
                        wp.from_torch(upper),
                        wp.from_torch(weights),
                        count,
                        self.dof,
                    ],
                    outputs=[
                        wp.from_torch(valid[start:end]),
                        wp.from_torch(joints[start:end]),
                    ],
                    device=str(self.device),
                )
        return valid, joints

    @_with_ik_buffers
    def _select_continuous_ik_path(
        self,
        candidate_qpos: torch.Tensor,
        candidate_valid: torch.Tensor,
        qpos_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import warp as wp
        from embodichain.compute.kinematics._warp.ssik import select_path

        if candidate_qpos.ndim != 4 or candidate_qpos.shape[-1] != self.dof:
            raise ValueError("candidate_qpos must have shape (B,N,K,dof)")
        self._wait_ready()
        b, n, k, _ = candidate_qpos.shape
        if candidate_valid.shape != (b, n, k) or qpos_seed.shape != (b, self.dof):
            raise ValueError(
                "Candidate masks and initial path seeds must match path axes"
            )
        candidates = candidate_qpos.to(
            device=self.device, dtype=torch.float32
        ).contiguous()
        validity = candidate_valid.to(device=self.device, dtype=torch.bool).contiguous()
        seed = qpos_seed.to(device=self.device, dtype=torch.float32).contiguous()
        valid = torch.empty((b, n), dtype=torch.bool, device=self.device)
        result = torch.empty((b, n, self.dof), dtype=torch.float32, device=self.device)
        if b and n:
            wp.launch(
                select_path,
                dim=b,
                inputs=[
                    wp.from_torch(candidates),
                    wp.from_torch(validity),
                    wp.from_torch(seed),
                    wp.from_torch(self.lower_qpos_limits.double().contiguous()),
                    wp.from_torch(self.upper_qpos_limits.double().contiguous()),
                    wp.from_torch(self.ik_nearest_weight.float().contiguous()),
                    n,
                    k,
                    self.dof,
                    self.cfg.max_joint_step or math.inf,
                ],
                outputs=[wp.from_torch(valid), wp.from_torch(result)],
                device=str(self.device),
            )
        return valid, result
