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

"""Parameterized POE RR candidates with explicit seed-local redundancy search.

CPU and CUDA evaluate the same sixteen global and nine local locked
coordinates using Warp. The independent SciPy QZ implementation remains an
explicit private oracle; runtime dispatch never calls it or a pose IK fallback.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp

from ._ssik_model import _SSIKModel

__all__: list[str] = []

_SIDECAR_SHA256 = "deda0b3e150d2f7c7b56ea3a255bb76a758b2f6bb406397faf1acd74e1329477"
_FIRST_PAIRS = ((5, 8), (2, 5), (11, 2), (4, 7), (10, 1), (3, 6), (9, 0))
_SECOND_PAIRS = ((7, 8), (6, 7), (1, 2), (4, 5), (10, 11), (0, 1), (3, 4), (9, 10))


def _condition_target(target: np.ndarray) -> np.ndarray | None:
    """Condition a valid pose for the exact SO(3) coefficient identities.

    Float32 frame conversion can leave a rotation a few ulps off SO(3), which
    an ill-conditioned RR chart can amplify during elimination. Two fixed
    Newton--Schulz polar steps restore the rotation in float64. This changes
    only matrix preparation; the caller still accepts joints against its
    original, unmodified target and original-model FK.
    """
    if not np.isfinite(target).all():
        return None
    rotation = target[:3, :3]
    identity = np.eye(3)
    if (
        np.any(np.abs(rotation.T @ rotation - identity) > 1e-5)
        or np.linalg.det(rotation) <= 0
        or np.any(np.abs(target[3, :3]) > 1e-6)
        or abs(target[3, 3] - 1) > 1e-6
    ):
        return None
    conditioned = target.copy()
    for _ in range(2):
        rotation = rotation @ (1.5 * identity - 0.5 * (rotation.T @ rotation))
    conditioned[:3, :3] = rotation
    return conditioned


def _rz(angle: float) -> np.ndarray:
    sine, cosine = np.sin(angle), np.cos(angle)
    return np.array(
        [[cosine, -sine, 0, 0], [sine, cosine, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


def _projective(v: np.ndarray, pairs: tuple[tuple[int, int], ...]) -> float:
    first, second = max(
        pairs, key=lambda indices: max(abs(v[indices[0]]), abs(v[indices[1]]))
    )
    numerator, denominator = v[first], v[second]
    if denominator < 0 or (denominator == 0 and numerator < 0):
        numerator, denominator = -numerator, -denominator
    return float(2 * np.arctan2(numerator, denominator))


def _embed(matrix: np.ndarray) -> np.ndarray:
    result = np.zeros((12, 12))
    result[:6, :9] = matrix
    result[6:, :6] = matrix[:, 3:]
    result[6:, 9:] = matrix[:, :3]
    return result


def _lock_predictor(
    frames: np.ndarray,
    local_axes: np.ndarray,
    target: np.ndarray,
    seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_step: float,
) -> tuple[float, float]:
    """Choose two redundancy coordinates from one original-model tangent.

    These values are sampling parameters only. No Cartesian Newton iterate or
    tangent joint vector is returned as IK; every emitted joint vector still
    comes from a newly evaluated RR polynomial system and its exact roots.
    """
    rotation, position = np.eye(3), np.zeros(3)
    origins, axes = [], []
    for index in range(8):
        fixed = frames[index].reshape(3, 4)
        position += rotation @ fixed[:, 3]
        rotation = rotation @ fixed[:, :3]
        if index < 7:
            axis = local_axes[index]
            origins.append(position.copy())
            axes.append(rotation @ axis)
            magnitude = np.linalg.norm(axis)
            direction = axis / magnitude
            skew = np.array(
                [
                    [0, -direction[2], direction[1]],
                    [direction[2], 0, -direction[0]],
                    [-direction[1], direction[0], 0],
                ]
            )
            angle = seed[index] * magnitude
            rotation = rotation @ (
                np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
            )
    axes = np.array(axes)
    jacobian = np.vstack((np.cross(axes, position - np.array(origins)).T, axes.T))
    relative = target[:3, :3] @ rotation.T
    angular = 0.5 * np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    error = np.r_[target[:3, 3] - position, angular]
    trust = min(max_step, 0.2)
    interval_lo = max(lower[2] - seed[2], -trust)
    interval_hi = min(upper[2] - seed[2], trust)
    first = second = 0.0
    if interval_lo > interval_hi:
        return first, second
    free = (0, 1, 3, 4, 5, 6)
    try:
        matrix = jacobian[:, free]
        if np.linalg.cond(matrix) > 1e12:
            raise np.linalg.LinAlgError
        uv = np.linalg.solve(matrix, np.column_stack((error, jacobian[:, 2])))
        u, v = uv[:, 0], uv[:, 1]
        unconstrained = float(v @ u / (1 + v @ v))
        first = float(np.clip(unconstrained, interval_lo, interval_hi))
        second = first * 0.5
        feasible_lo, feasible_hi = interval_lo, interval_hi
        for index, joint in enumerate(free):
            lo = max(lower[joint] - seed[joint], -trust)
            hi = min(upper[joint] - seed[joint], trust)
            if abs(v[index]) > 1e-14:
                a, b = (u[index] - lo) / v[index], (u[index] - hi) / v[index]
                feasible_lo = max(feasible_lo, min(a, b))
                feasible_hi = min(feasible_hi, max(a, b))
            elif u[index] < lo or u[index] > hi:
                feasible_hi = feasible_lo - 1
        if feasible_lo <= feasible_hi:
            first = float(np.clip(unconstrained, feasible_lo, feasible_hi))
            second = float((feasible_lo + feasible_hi) * 0.5)
    except np.linalg.LinAlgError:
        damped = jacobian @ jacobian.T + 1e-10 * np.eye(6)
        first = float(
            np.clip(
                jacobian[:, 2] @ np.linalg.solve(damped, error),
                interval_lo,
                interval_hi,
            )
        )
        second = first * 0.5
    return first, second


def _back_substitute(
    x: float,
    monomial: np.ndarray,
    p_sin: np.ndarray,
    p_cos: np.ndarray,
    p_one: np.ndarray,
    q_inverse: np.ndarray,
    factors: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    first, second = _projective(monomial, _FIRST_PAIRS), _projective(
        monomial, _SECOND_PAIRS
    )
    q = np.zeros(6)
    q[1], q[2], q[3] = 2 * np.arctan(x), first, second
    sa, ca, sb, cb = np.sin(first), np.cos(first), np.sin(second), np.cos(second)
    left = np.array([sa * sb, sa * cb, ca * sb, ca * cb, sa, ca, sb, cb, 1])
    right = q_inverse @ ((p_sin * np.sin(q[1]) + p_cos * np.cos(q[1]) + p_one) @ left)
    q[0], q[5] = np.arctan2(right[4], right[5]), np.arctan2(right[6], right[7])
    before = np.eye(4)
    for index in range(4):
        before = before @ _rz(q[index]) @ factors[index]
    after = _rz(q[5]) @ factors[5]
    residual = (
        np.linalg.solve(before, target)
        @ np.linalg.inv(after)
        @ np.linalg.inv(factors[4])
    )
    q[4] = np.arctan2(residual[1, 0], residual[0, 0])
    return q


class _ParametricRRCandidates:
    """One prepared POE coefficient system evaluated at arbitrary lock angles."""

    sample_count = 25
    candidate_count = sample_count * 24
    max_chunk_size = 32

    def __init__(self, model: _SSIKModel, device: torch.device | str) -> None:
        wp.init()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        path = Path(__file__).parent / "_warp/data/ssik_rr_rizon4_parametric.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != _SIDECAR_SHA256:
            raise ValueError(
                "Parameterized SSIK RR coefficient sidecar failed integrity check"
            )
        with np.load(path, allow_pickle=False) as archive:
            self._constants = {key: archive[key] for key in archive.files}
        c = self._constants
        if (
            model.dof != 7
            or np.max(np.linalg.norm(model.axes - c["source_axes"], axis=-1)) > 5e-7
            or np.max(
                np.linalg.norm(model.offsets[:7] - c["source_offsets"][:7], axis=-1)
            )
            > 5e-7
        ):
            raise ValueError(
                "Parameterized SSIK RR bundle does not match this model's joint prefix"
            )
        home = np.eye(4)
        home[:3, :3], home[:3, 3] = model.home_rotation, model.offsets.sum(axis=0)
        self.source_to_artifact = np.linalg.solve(home, c["canonical_home"])
        self._model_frames = model.frames.copy()
        self._model_axes = model.local_axes.copy()
        self._owned: list[torch.Tensor] = []
        self._workspace: dict[str, torch.Tensor] = {}
        self._capacity = 0
        self._data = None
        self._prepare_constants()

    def _prepare_constants(self) -> None:
        # Shared device-resident coefficient data, independent of the QZ oracle.
        from ._warp.ssik_rr_parametric import _ParametricData

        self._data = _ParametricData()
        for name, key in (
            ("p_affine", "p_affine"),
            ("factor_affine", "factor_affine"),
            ("mono_factors", "mono_factors"),
            ("coefficient_values", "coefficient_values"),
            ("global_angles", "global_lock_angles"),
            ("local_offsets", "local_offsets"),
        ):
            tensor = torch.tensor(self._constants[key], device=self.device)
            self._owned.append(tensor)
            setattr(self._data, name, wp.from_torch(tensor))
        indices = self._constants["coefficient_indices"]
        cells = np.where(
            indices[:, 0] == 0,
            indices[:, 1] * 9 + indices[:, 2],
            126 + indices[:, 1] * 8 + indices[:, 2],
        )
        if np.any(np.diff(cells) < 0):
            raise ValueError(
                "Parameterized RR coefficients must be sorted by output cell"
            )
        offsets = np.r_[0, np.cumsum(np.bincount(cells, minlength=238))].astype(
            np.int32
        )
        for name, array in (
            ("cell_offsets", offsets),
            ("coefficient_monomials", indices[:, 3]),
        ):
            tensor = torch.tensor(array, device=self.device)
            self._owned.append(tensor)
            setattr(self._data, name, wp.from_torch(tensor))
        for name, array in (
            ("model_frames", self._model_frames),
            ("model_axes", self._model_axes),
        ):
            tensor = torch.tensor(array, device=self.device)
            self._owned.append(tensor)
            setattr(self._data, name, wp.from_torch(tensor))
        from ._ssik_rr import _W_TRIG_TO_X

        tensor = torch.tensor(_W_TRIG_TO_X, device=self.device)
        self._owned.append(tensor)
        self._data.w = wp.from_torch(tensor)
        self._data.pre_inverse = wp.mat44d(self._constants["pre_inverse"])
        self._data.source_to_artifact = wp.mat44d(self.source_to_artifact)

    def prepare_buffers(self, size: int) -> None:
        size = min(size, self.max_chunk_size)
        if size <= self._capacity:
            return
        self._capacity = size
        samples, slots = self.sample_count, self.candidate_count
        shapes = {
            "goal": (size, 4, 4),
            "p_target": (size, 14, 9),
            "q_basis": (size, 6, 14),
            "q_inverse": (size, 8, 14),
            "pencil": (size, samples, 36, 12),
            "real": (size, samples, 24, 24),
            "roots": (size, samples, 24),
            "angles": (size, samples),
            "predictions": (size, 2),
            "back_matrix": (size, slots, 12, 12),
            "back_vectors": (size, slots, 2, 12),
        }
        if self.device.type == "cpu":
            shapes.update(
                lhs=(size, samples, 24, 24),
                rhs=(size, samples, 24, 24),
                imag=(size, samples, 24, 24),
                eigen_vectors=(size, samples, 4, 24),
                root_imag=(size, samples, 24),
            )
        for name, shape in shapes.items():
            self._workspace[name] = torch.empty(
                shape, dtype=torch.float64, device=self.device
            )
        self._workspace["target_valid"] = torch.empty(
            (size,), dtype=torch.int32, device=self.device
        )
        self._workspace["root_valid"] = torch.empty(
            (size, samples, 24), dtype=torch.int32, device=self.device
        )

    def generate(
        self,
        target_end: torch.Tensor,
        seed: torch.Tensor | None = None,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
        max_step: float = math.inf,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            target_end.ndim != 3
            or target_end.shape[1:] != (4, 4)
            or target_end.dtype != torch.float64
            or target_end.device != self.device
        ):
            raise ValueError(
                "Parameterized RR targets must be float64 (N,4,4) on the prepared device"
            )
        if (
            seed is None
            or lower is None
            or upper is None
            or seed.shape != (len(target_end), 7)
            or lower.shape != (7,)
            or upper.shape != (7,)
        ):
            raise ValueError(
                "Parameterized RR requires seed (N,7) and effective lower/upper (7,)"
            )
        if any(
            value.device != self.device or value.dtype != torch.float64
            for value in (seed, lower, upper)
        ):
            raise ValueError(
                "Parameterized RR seeds and limits must be float64 on the prepared device"
            )
        if self.device.type == "cpu":
            return self._warp_cpu(target_end, seed, lower, upper, max_step)
        return self._cuda(target_end, seed, lower, upper, max_step)

    def _warp_cpu(
        self,
        target_end: torch.Tensor,
        seed: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        max_step: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ._warp.ssik_rr_parametric_cpu import (
            parametric_target_prepare_cpu,
            parametric_lock_predictor_cpu,
            parametric_pencil_roots_cpu,
        )
        from ._warp.ssik_rr_parametric import parametric_back_substitute

        n = len(target_end)
        validity = torch.empty(
            (n, self.candidate_count), dtype=torch.int32, device=self.device
        )
        candidates = torch.empty(
            (n, self.candidate_count, 7), dtype=torch.float64, device=self.device
        )
        self.prepare_buffers(min(n, self.max_chunk_size))
        for start in range(0, n, self.max_chunk_size):
            stop = min(start + self.max_chunk_size, n)
            count = stop - start
            w = {
                name: wp.from_torch(
                    tensor[:count], dtype=wp.mat44d if name == "goal" else None
                )
                for name, tensor in self._workspace.items()
            }
            target = wp.from_torch(target_end[start:stop].contiguous(), dtype=wp.mat44d)
            initial = wp.from_torch(seed[start:stop])
            bounds = [wp.from_torch(lower), wp.from_torch(upper)]
            wp.launch(
                parametric_target_prepare_cpu,
                dim=count,
                inputs=[
                    self._data,
                    target,
                    w["goal"],
                    w["p_target"],
                    w["q_basis"],
                    w["q_inverse"],
                    w["target_valid"],
                ],
                device="cpu",
            )
            wp.launch(
                parametric_lock_predictor_cpu,
                dim=count,
                inputs=[
                    self._data,
                    target,
                    initial,
                    *bounds,
                    max_step,
                    w["predictions"],
                ],
                device="cpu",
            )
            wp.launch(
                parametric_pencil_roots_cpu,
                dim=(count, self.sample_count),
                inputs=[
                    self._data,
                    w["p_target"],
                    w["q_basis"],
                    w["target_valid"],
                    initial,
                    *bounds,
                    max_step,
                    w["predictions"],
                    w["angles"],
                    w["pencil"],
                    w["lhs"],
                    w["rhs"],
                    w["real"],
                    w["imag"],
                    w["eigen_vectors"],
                    w["roots"],
                    w["root_imag"],
                    w["root_valid"],
                ],
                device="cpu",
            )
            wp.launch(
                parametric_back_substitute,
                dim=(count, self.sample_count, 24),
                inputs=[
                    self._data,
                    w["goal"],
                    w["p_target"],
                    w["q_inverse"],
                    w["pencil"],
                    w["roots"],
                    w["root_valid"],
                    w["angles"],
                    w["back_matrix"],
                    w["back_vectors"],
                    wp.from_torch(candidates[start:stop]),
                    wp.from_torch(validity[start:stop]),
                ],
                device="cpu",
            )
        return validity, candidates

    def _cpu_qz(
        self,
        target_end: torch.Tensor,
        seed: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        max_step: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit independent test oracle; never selected by runtime dispatch."""
        from scipy.linalg import eig
        from ._ssik_rr import _W_TRIG_TO_X

        n = len(target_end)
        validity = np.zeros((n, self.candidate_count), dtype=np.int32)
        candidates = np.zeros((n, self.candidate_count, 7), dtype=np.float64)
        c = self._constants
        lower_phi, upper_phi = float(lower[2]), float(upper[2])
        for target_index, (target, initial) in enumerate(
            zip(target_end.detach().numpy(), seed.detach().numpy())
        ):
            conditioned = _condition_target(target)
            if conditioned is None:
                continue
            goal = c["pre_inverse"] @ conditioned @ self.source_to_artifact
            values = np.r_[goal[:3].reshape(-1), 1.0]
            monomials = values[
                np.where(c["mono_factors"] < 0, 12, c["mono_factors"])
            ].prod(axis=1)
            p_target, q = np.zeros((14, 9)), np.zeros((14, 8))
            for kind, matrix in enumerate((p_target, q)):
                selected = c["coefficient_indices"][:, 0] == kind
                indices = c["coefficient_indices"][selected]
                np.add.at(
                    matrix,
                    (indices[:, 1], indices[:, 2]),
                    c["coefficient_values"][selected] * monomials[indices[:, 3]],
                )
            u, _, _ = np.linalg.svd(q, full_matrices=True)
            basis, q_inverse = u[:, -6:].T, np.linalg.pinv(q)
            offsets = c["local_offsets"].copy()
            offsets[-2:] = _lock_predictor(
                self._model_frames,
                self._model_axes,
                target,
                initial,
                lower.detach().numpy(),
                upper.detach().numpy(),
                max_step,
            )
            angles = np.r_[c["global_lock_angles"], initial[2] + offsets]
            for sample, raw in enumerate(angles):
                if sample >= 16:
                    raw = np.clip(raw, lower_phi, upper_phi)
                first, last = np.ceil((lower_phi - 1e-6 - raw) / (2 * np.pi)), np.floor(
                    (upper_phi + 1e-6 - raw) / (2 * np.pi)
                )
                turns = np.clip(
                    np.floor((initial[2] - raw) / (2 * np.pi) + 0.5), first, last
                )
                phi = np.clip(raw + 2 * np.pi * turns, lower_phi, upper_phi)
                if first > last or abs(float(np.float32(phi)) - initial[2]) > max_step:
                    continue
                affine = (
                    c["p_affine"][:, 0]
                    + np.sin(phi) * c["p_affine"][:, 1]
                    + np.cos(phi) * c["p_affine"][:, 2]
                )
                ps, pc, po = affine[0], affine[1], p_target + affine[2]
                es, ec, eo = [basis @ matrix for matrix in (ps, pc, po)]
                aa, bb, cc = [
                    _embed(matrix @ _W_TRIG_TO_X)
                    for matrix in (eo - ec, 2 * es, eo + ec)
                ]
                left = np.block([[np.zeros((12, 12)), np.eye(12)], [-cc, -bb]])
                right = np.block(
                    [[np.eye(12), np.zeros((12, 12))], [np.zeros((12, 12)), aa]]
                )
                pairs = eig(left, right, homogeneous_eigvals=True, right=False)
                factors = (
                    c["factor_affine"][0]
                    + np.sin(phi) * c["factor_affine"][1]
                    + np.cos(phi) * c["factor_affine"][2]
                )
                for branch, (alpha, beta) in enumerate(pairs.T):
                    root = alpha / beta if abs(beta) > 1e-13 else complex(1e30)
                    if abs(root.imag) > 1e-3 * max(1.0, abs(root.real)) or (
                        abs(abs(root.imag) - 1) < 0.1 and abs(root.real) < 0.1
                    ):
                        continue
                    x = float(root.real)
                    norm = 1 + x * x
                    matrix = aa * (x * x / norm) + bb * (x / norm) + cc / norm
                    v = np.linalg.svd(matrix)[2][-1]
                    q6 = _back_substitute(x, v, ps, pc, po, q_inverse, factors, goal)
                    slot = sample * 24 + branch
                    candidates[target_index, slot] = np.insert(q6, 2, phi)
                    validity[target_index, slot] = int(np.isfinite(q6).all())
        return torch.from_numpy(validity), torch.from_numpy(candidates)

    def _cuda(
        self,
        target_end: torch.Tensor,
        seed: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        max_step: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from ._warp.ssik_rr_parametric_cuda import (
            parametric_pencil_prepare,
            parametric_target_prepare,
            parametric_lock_predictor,
        )
        from ._warp.ssik_rr_cuda import rr_eigen_cooperative
        from ._warp.ssik_rr_parametric import parametric_back_substitute

        n = len(target_end)
        validity = torch.empty(
            (n, self.candidate_count), dtype=torch.int32, device=self.device
        )
        candidates = torch.empty(
            (n, self.candidate_count, 7), dtype=torch.float64, device=self.device
        )
        self.prepare_buffers(min(n, self.max_chunk_size))
        torch_stream = torch.cuda.current_stream(self.device)
        for tensor in (
            *self._owned,
            *self._workspace.values(),
            target_end,
            seed,
            lower,
            upper,
        ):
            tensor.record_stream(torch_stream)
        stream = wp.stream_from_torch(torch_stream)
        for start in range(0, n, self.max_chunk_size):
            stop = min(start + self.max_chunk_size, n)
            count = stop - start
            w = {
                name: wp.from_torch(
                    tensor[:count], dtype=wp.mat44d if name == "goal" else None
                )
                for name, tensor in self._workspace.items()
            }
            target = wp.from_torch(target_end[start:stop].contiguous(), dtype=wp.mat44d)
            wp.launch(
                parametric_target_prepare,
                dim=(count, 32),
                inputs=[
                    self._data,
                    target,
                    w["goal"],
                    w["p_target"],
                    w["q_basis"],
                    w["q_inverse"],
                    w["target_valid"],
                ],
                device=str(self.device),
                stream=stream,
                block_dim=32,
            )
            wp.launch(
                parametric_lock_predictor,
                dim=(count, 32),
                inputs=[
                    self._data,
                    target,
                    wp.from_torch(seed[start:stop]),
                    wp.from_torch(lower),
                    wp.from_torch(upper),
                    max_step,
                    w["predictions"],
                ],
                device=str(self.device),
                stream=stream,
                block_dim=32,
            )
            wp.launch(
                parametric_pencil_prepare,
                dim=(count, self.sample_count, 32),
                inputs=[
                    self._data,
                    w["p_target"],
                    w["q_basis"],
                    w["target_valid"],
                    wp.from_torch(seed[start:stop]),
                    wp.from_torch(lower),
                    wp.from_torch(upper),
                    max_step,
                    w["predictions"],
                    w["angles"],
                    w["pencil"],
                    w["real"],
                    w["root_valid"],
                ],
                device=str(self.device),
                stream=stream,
                block_dim=32,
            )
            wp.launch(
                rr_eigen_cooperative,
                dim=(count, self.sample_count, 32),
                inputs=[w["real"], w["roots"], w["root_valid"]],
                device=str(self.device),
                stream=stream,
                block_dim=32,
            )
            wp.launch(
                parametric_back_substitute,
                dim=(count, self.sample_count, 24),
                inputs=[
                    self._data,
                    w["goal"],
                    w["p_target"],
                    w["q_inverse"],
                    w["pencil"],
                    w["roots"],
                    w["root_valid"],
                    w["angles"],
                    w["back_matrix"],
                    w["back_vectors"],
                    wp.from_torch(candidates[start:stop]),
                    wp.from_torch(validity[start:stop]),
                ],
                device=str(self.device),
                stream=stream,
            )
        return validity, candidates
