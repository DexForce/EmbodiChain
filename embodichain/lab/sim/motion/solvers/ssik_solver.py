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

"""Model-derived geometric IK and an optional pinned SSIK CPU reference."""

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
    """Configuration for strict geometric IK derived from the actual chain.

    Warp supports three-parallel-axis 6R and concurrent shoulder/wrist SRS 7R
    models. The optional CPU reference requires the pinned ``ssik-reference``
    extra. Failure never invokes another solver.
    """

    class_type: str = "SSIKSolver"
    backend: Literal["warp", "reference"] = "warp"
    num_samples: int = 128
    position_tolerance: float = 1e-5
    rotation_tolerance: float = 1e-5
    max_joint_step: float | None = None
    chunk_size: int = 512
    reference_artifact: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("warp", "reference"):
            raise ValueError("SSIK backend must be 'warp' or 'reference'")
        for name in ("num_samples", "chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("position_tolerance", "rotation_tolerance", "max_joint_step"):
            value = getattr(self, name)
            if (value is None and name != "max_joint_step") or (
                value is not None and (not math.isfinite(value) or value <= 0)
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.backend == "warp" and self.reference_artifact is not None:
            raise ValueError(
                "reference_artifact is only valid for the reference backend"
            )

    def init_solver(
        self, device: str | torch.device = "cpu", **kwargs: Any
    ) -> SSIKSolver:
        """Build and validate the chain once, then initialize its chosen backend.

        Args:
            device: CPU or CUDA for Warp; CPU only for the reference.
            **kwargs: Optional ``pk_serial_chain`` in configured joint order.

        Returns:
            Initialized geometric solver with its configured TCP.
        """
        solver = SSIKSolver(self, device, **kwargs)
        solver.set_tcp(self._get_tcp_as_numpy())
        return solver


class SSIKSolver(BaseSolver):
    """Enumerate finite geometric branches without numerical IK fallback.

    SRS candidates condition on the seed swivel and a fixed global swivel
    grid. Sampling failure does not establish global unreachability. The
    nearest result uses weighted distance to the limit-projected caller seed.
    Public tensors remain owned by the caller across subsequent calls.

    Args:
        cfg: Geometry, backend and acceptance settings.
        device: Execution device; reference mode accepts CPU only.
        **kwargs: Optional preconstructed PK serial chain.
    """

    def __init__(
        self, cfg: SSIKSolverCfg, device: str | torch.device = "cpu", **kwargs: Any
    ) -> None:
        cfg.__post_init__()
        if cfg.backend == "reference" and torch.device(device).type != "cpu":
            raise ValueError("SSIK reference backend accepts CPU devices only")
        chain = kwargs.pop("pk_serial_chain", None)
        if chain is None:
            chain = create_pk_serial_chain(
                urdf_path=cfg.urdf_path,
                root_link_name=cfg.root_link_name,
                end_link_name=cfg.end_link_name,
                device=torch.device(device),
            )
        self._model = _extract_model(chain)
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
        if cfg.backend == "reference":
            from ._ssik_reference import _Reference

            self._reference = _Reference(self._model, resolved)
        else:
            import warp as wp
            from embodichain.compute.kinematics._warp.ssik import _Model, _mat812d
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
            self._warp_model.frames = _mat812d(model.frames)
            self._warp_model.local_axes = mat73d(model.local_axes)
            self._axes = torch.tensor(
                model.axes, dtype=torch.float64, device=self.device
            )
            self._offsets = torch.tensor(
                model.offsets, dtype=torch.float64, device=self.device
            )
        self.set_tcp(resolved._get_tcp_as_numpy())

    def _wait_ready(self) -> None:
        if self.device.type == "cuda" and self._ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self._ready)

    def _record_ready(self) -> None:
        if self.device.type == "cuda":
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(self.device))

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
        self._tcp_inverse = inv_transform(np.asarray(xpos, dtype=np.float64))
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            if self._ready is not None:
                stream.wait_event(self._ready)
            self._ready = torch.cuda.Event()
            self._ready.record(stream)

    def prepare_buffers(self, max_batch: int) -> None:
        """Reserve bounded scratch; returned all-candidate tensors are separate.

        Args:
            max_batch: Expected positive target count.
        """
        self._ik_buffers.reserve(min(max_batch, self.cfg.chunk_size))

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
        from embodichain.compute.kinematics._warp.ssik import generate, select_nearest

        if self._ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self._ready)
        samples = self.cfg.num_samples if self.dof == 7 else 0
        count = 8 * (samples + 1)
        shape = (n, count) if return_all_solutions else (n,)
        valid = torch.empty(shape, dtype=torch.bool, device=self.device)
        joints = torch.empty(
            (*shape, self.dof), dtype=torch.float32, device=self.device
        )
        for start in range(0, n, self.cfg.chunk_size):
            end = min(n, start + self.cfg.chunk_size)
            m = end - start
            flags = wp.to_torch(
                self._ik_buffers.zeros("ssik_flags", m, count, torch.int32)
            ).reshape(m, count)
            candidates = wp.to_torch(
                self._ik_buffers.zeros("ssik_qpos", m, count * self.dof, torch.float32)
            ).reshape(m, count, self.dof)
            wp.launch(
                generate,
                dim=(m, samples + 1),
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
                    samples,
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
