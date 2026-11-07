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

"""Prepared algebraic candidates with a device-resident Warp matrix-pencil solve.

This module owns numerical constants and temporary arrays only. Runtime joint
limits, TCP conversion, float32 rounding, original-model FK acceptance and seed
selection remain with the caller. Preparation imports neither SSIK nor SymPy.
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

_SIDECAR_SHA256 = "067f5cf1c57fb93868b09dcac62fd4126c1d00def4bdb15eb8d3bc9b1d302c16"
_W_TRIG_TO_X = np.array(
    [
        [0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 0, 0, -2, 0, 2, 0, 0, 0],
        [0, -2, 0, 0, 0, 0, 0, 2, 0],
        [1, 0, -1, 0, 0, 0, -1, 0, 1],
        [0, 0, 0, 2, 0, 2, 0, 0, 0],
        [-1, 0, -1, 0, 0, 0, 1, 0, 1],
        [0, 2, 0, 0, 0, 0, 0, 2, 0],
        [-1, 0, 1, 0, 0, 0, -1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1],
    ],
    dtype=np.float64,
)


class _FixedRRCandidates:
    """Verify a prepared joint prefix and enumerate every fixed-grid RR slot.

    The first prepared bundle is the pinned Flexiv Rizon4 geometry. Matching is
    numerical: model names do not choose a solver, joint limits do not change
    geometry, and a changed end frame is bridged explicitly. The 16 fixed locked
    coordinates are part of the baked derivation. They do not describe the full
    continuous 7R solution manifold.
    """

    candidate_count = 16 * 24
    max_chunk_size = 32

    def __init__(self, model: _SSIKModel, device: torch.device | str) -> None:
        from ._warp.ssik_rr import _RRData

        wp.init()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        path = Path(__file__).parent / "_warp" / "data" / "ssik_rr_rizon4.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != _SIDECAR_SHA256:
            raise ValueError(
                "Prepared SSIK RR coefficient sidecar failed integrity check"
            )
        with np.load(path, allow_pickle=False) as archive:
            constants = {key: archive[key] for key in archive.files}
        if model.dof != 7:
            raise ValueError(
                "Prepared SSIK RR bundle requires its verified seven-joint prefix"
            )
        expected_axes = constants["axes"]
        expected_offsets = constants["t_left"][:, :3, 3]
        if (
            np.max(np.linalg.norm(model.axes - expected_axes, axis=-1)) > 5e-7
            or np.max(np.linalg.norm(model.offsets[:7] - expected_offsets, axis=-1))
            > 5e-7
        ):
            raise ValueError(
                "Prepared SSIK RR bundle does not match this model's joint prefix"
            )
        actual_home = np.eye(4)
        actual_home[:3, :3] = model.home_rotation
        actual_home[:3, 3] = model.offsets.sum(axis=0)
        artifact_home = np.eye(4)
        for left, right in zip(constants["t_left"], constants["t_right"]):
            artifact_home = artifact_home @ left @ right
        correction = np.linalg.solve(actual_home, artifact_home)
        self.source_to_artifact = correction
        self._data = _RRData()
        self._owned: list[torch.Tensor] = []
        self._workspace: dict[str, torch.Tensor] = {}
        self._capacity = 0

        def bind(name: str, value: np.ndarray, dtype: object = None) -> None:
            tensor = torch.as_tensor(np.ascontiguousarray(value), device=self.device)
            self._owned.append(tensor)
            setattr(self._data, name, wp.from_torch(tensor, dtype=dtype))

        bind("p_sin", constants["p_sin"])
        bind("p_cos", constants["p_cos"])
        factors, po_indices, q_indices, po_coeff, q_coeff = [], [], [], [], []
        po_offsets, q_offsets = [0], [0]
        mono_offset = 0
        for sample in range(16):
            factors.append(constants[f"mono_factors_{sample}"])
            for prefix, index_parts, coefficients, offsets in (
                ("po", po_indices, po_coeff, po_offsets),
                ("q", q_indices, q_coeff, q_offsets),
            ):
                rc = constants[f"{prefix}_rc_{sample}"]
                mono = constants[f"{prefix}_mono_{sample}"] + mono_offset
                index_parts.append(np.column_stack((rc, mono)).astype(np.int32))
                coefficients.append(constants[f"{prefix}_coeff_{sample}"])
                offsets.append(offsets[-1] + len(rc))
            mono_offset += len(factors[-1])
        bind("factors", np.concatenate(factors))
        bind("po_indices", np.concatenate(po_indices))
        bind("q_indices", np.concatenate(q_indices))
        bind("po_coeff", np.concatenate(po_coeff))
        bind("q_coeff", np.concatenate(q_coeff))
        bind("po_offsets", np.array(po_offsets, dtype=np.int32))
        bind("q_offsets", np.array(q_offsets, dtype=np.int32))
        bind("w", _W_TRIG_TO_X)
        bind(
            "dh",
            np.concatenate([constants[key] for key in ("alpha", "a", "d")], axis=1),
        )
        bind("theta_offset", constants["theta_offset"])
        bind("t_pre_inv", constants["t_pre_inv"], wp.mat44d)
        bind("t_post_inv", constants["t_post_inv"], wp.mat44d)
        metadata = np.column_stack(
            [
                constants[key]
                for key in (
                    "linearity_joint",
                    "left_bilinear",
                    "right_bilinear",
                    "drop_joint",
                )
            ]
        ).astype(np.int32)
        bind("metadata", metadata)
        bind("lock_angles", constants["q_lock"])
        self._data.lock_joint = int(constants["lock_idx"])
        self._data.source_to_artifact = wp.mat44d(correction)

    def prepare_buffers(self, size: int) -> None:
        """Reserve a bounded workspace; caller-owned candidate outputs are separate."""
        size = min(size, self.max_chunk_size)
        if size <= self._capacity:
            return
        self._capacity = size
        shapes = {
            "p_one": (size, 16, 14, 9),
            "q": (size, 16, 14, 8),
            "qt": (size, 16, 14, 14),
            "q_inverse": (size, 16, 8, 14),
            "pencil": (size, 16, 36, 12),
            "lhs": (size, 16, 24, 24),
            "rhs": (size, 16, 24, 24),
            "real": (size, 16, 24, 24),
            "imag": (size, 16, 24, 24),
            "vectors": (size, 16, 4, 24),
            "roots": (size, 16, 24),
            "root_imag": (size, 16, 24),
            "back_matrix": (size, 384, 12, 12),
            "back_vectors": (size, 384, 2, 12),
        }
        if self.device.type == "cuda":
            for key in ("q", "qt", "lhs", "rhs", "imag", "vectors", "root_imag"):
                del shapes[key]
        for key, shape in shapes.items():
            # Adjacent CUDA threads own adjacent (target, slice) problems.
            # Keep this axis contiguous so QR matrix-element accesses coalesce.
            if len(shape) == 4:
                tensor = torch.empty(
                    (*shape[2:], *shape[:2]), dtype=torch.float64, device=self.device
                ).permute(2, 3, 0, 1)
            else:
                tensor = torch.empty(
                    (*shape[2:], *shape[:2]), dtype=torch.float64, device=self.device
                ).permute(1, 2, 0)
            self._workspace[key] = tensor
        self._workspace["root_valid"] = torch.empty(
            (size, 16, 24), dtype=torch.int32, device=self.device
        )

    def generate(self, target_end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw validity ``(N,384)`` and float64 joints ``(N,384,7)``.

        Inputs are actual-model end-link poses; TCP must already be removed.
        The caller filters every returned slot using its actual joint limits
        and independent original-chain FK. No seed, limiting or hidden solve is
        performed here. Returned arrays are never reused by subsequent calls.
        """
        from ._warp.ssik_rr import rr_back_substitute, rr_roots

        if target_end.ndim != 3 or target_end.shape[1:] != (4, 4):
            raise ValueError("RR targets must have shape (N,4,4)")
        if target_end.device != self.device or target_end.dtype != torch.float64:
            raise ValueError("RR targets must be float64 on the prepared device")
        n = len(target_end)
        validity = torch.empty(
            (n, self.candidate_count), dtype=torch.int32, device=self.device
        )
        candidates = torch.empty(
            (n, self.candidate_count, 7), dtype=torch.float64, device=self.device
        )
        self.prepare_buffers(min(n, self.max_chunk_size))
        stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
            if self.device.type == "cuda"
            else None
        )
        if self.device.type == "cuda":
            torch_stream = torch.cuda.current_stream(self.device)
            for tensor in (*self._owned, *self._workspace.values()):
                # Warp launches use zero-copy Torch allocations. Record their
                # consumer stream so growth or object destruction cannot return
                # live storage to the allocator on another stream.
                tensor.record_stream(torch_stream)
        for start in range(0, n, self.max_chunk_size):
            stop = min(start + self.max_chunk_size, n)
            count = stop - start
            workspace = {
                key: wp.from_torch(value[:count])
                for key, value in self._workspace.items()
            }
            targets = wp.from_torch(
                target_end[start:stop].contiguous(), dtype=wp.mat44d
            )
            back_arguments = [self._data, targets] + [
                workspace[key]
                for key in (
                    "p_one",
                    "q_inverse",
                    "pencil",
                    "roots",
                    "root_valid",
                    "back_matrix",
                    "back_vectors",
                )
            ]
            if self.device.type == "cuda":
                from ._warp.ssik_rr_cuda import (
                    rr_eigen_cooperative,
                    rr_prepare_cooperative,
                )

                wp.launch(
                    rr_prepare_cooperative,
                    dim=(count, 16, 32),
                    inputs=[self._data, targets]
                    + [
                        workspace[key]
                        for key in (
                            "p_one",
                            "q_inverse",
                            "pencil",
                            "real",
                            "root_valid",
                        )
                    ],
                    device=str(self.device),
                    stream=stream,
                    block_dim=32,
                )
                wp.launch(
                    rr_eigen_cooperative,
                    dim=(count, 16, 32),
                    inputs=[
                        workspace["real"],
                        workspace["roots"],
                        workspace["root_valid"],
                    ],
                    device=str(self.device),
                    stream=stream,
                    block_dim=32,
                )
            else:
                root_arguments = [self._data, targets] + [
                    workspace[key]
                    for key in (
                        "p_one",
                        "q",
                        "qt",
                        "q_inverse",
                        "pencil",
                        "lhs",
                        "rhs",
                        "real",
                        "imag",
                        "vectors",
                        "roots",
                        "root_imag",
                        "root_valid",
                    )
                ]
                wp.launch(
                    rr_roots,
                    dim=(count, 16),
                    inputs=root_arguments + [0],
                    device=str(self.device),
                    stream=stream,
                )
            wp.launch(
                rr_back_substitute,
                dim=(count, 16, 24),
                inputs=back_arguments
                + [
                    wp.from_torch(candidates[start:stop]),
                    wp.from_torch(validity[start:stop]),
                ],
                device=str(self.device),
                stream=stream,
            )
        return validity, candidates


class _PreparedRRCandidates:
    """Select a prepared fixed or parameterized algebraic RR candidate pool.

    ``fixed`` retains the original sixteen-chart implementation. ``seed_local``
    evaluates a separately derived POE polynomial system at sixteen global and
    nine seed-local locked-joint coordinates. Both return every current sample's
    twenty-four pencil slots; callers retain the original-model acceptance gate.
    """

    def __init__(
        self,
        model: _SSIKModel,
        device: torch.device | str,
        search: str = "fixed",
    ) -> None:
        if search == "fixed":
            self._provider = _FixedRRCandidates(model, device)
        elif search == "seed_local":
            from ._ssik_rr_parametric import _ParametricRRCandidates

            self._provider = _ParametricRRCandidates(model, device)
        else:
            raise ValueError("RR search must be 'fixed' or 'seed_local'")
        self.search = search

    @property
    def candidate_count(self) -> int:
        return self._provider.candidate_count

    @property
    def max_chunk_size(self) -> int:
        return self._provider.max_chunk_size

    @property
    def device(self) -> torch.device:
        return self._provider.device

    @property
    def source_to_artifact(self) -> np.ndarray:
        return self._provider.source_to_artifact

    @property
    def _workspace(self) -> dict[str, torch.Tensor]:
        return self._provider._workspace

    @property
    def _owned(self) -> list[torch.Tensor]:
        return self._provider._owned

    @property
    def _capacity(self) -> int:
        return self._provider._capacity

    @property
    def _data(self) -> object:
        return self._provider._data

    def prepare_buffers(self, size: int) -> None:
        self._provider.prepare_buffers(size)

    def generate(
        self,
        target_end: torch.Tensor,
        seed: torch.Tensor | None = None,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
        max_step: float = math.inf,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.search == "fixed":
            return self._provider.generate(target_end)
        return self._provider.generate(target_end, seed, lower, upper, max_step)
