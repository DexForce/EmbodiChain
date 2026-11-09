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

"""Borrowed strict-geometry candidates from isolated, bounded Warp kernels."""

from __future__ import annotations

import torch
import warp as wp

from ._ssik_model import _SSIKModel

__all__: list[str] = []


class _GeometricCandidates:
    """Generate complete branch slots without a runtime-family CUDA kernel.

    Raw outputs are borrowed until the next call. The host owns serialization,
    original-model acceptance, final output ownership and stream dependencies.
    """

    max_chunk_size = 512

    def __init__(
        self,
        model: _SSIKModel,
        device: torch.device | str,
        num_samples: int = 128,
        srs_search: str = "uniform",
    ) -> None:
        from ._warp.ssik_6r import mat73d, mat83d

        if model.family not in ("three_parallel", "srs"):
            raise ValueError(
                "Geometric candidates require a certified strict6R or SRS chain"
            )
        if srs_search not in ("uniform", "seed_local") or num_samples < 1:
            raise ValueError("Invalid geometric redundancy search configuration")
        wp.init()
        self.device = torch.device(device)
        self._dof = model.dof
        self._h = mat73d(model.axes)
        self._p = mat83d(model.offsets)
        self._home = wp.mat33d(model.home_rotation)
        self._identity = wp.mat44d(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        self._shoulder = wp.vec3d(model.shoulder)
        self._wrist = wp.vec3d(model.wrist)
        self._axes = torch.as_tensor(
            model.axes.copy(), dtype=torch.float64, device=self.device
        )
        self._offsets = torch.as_tensor(
            model.offsets.copy(), dtype=torch.float64, device=self.device
        )
        self._samples = num_samples if model.dof == 7 else 0
        self._local = int(model.dof == 7 and srs_search == "seed_local")
        self.candidate_count = 8 * (self._samples + 1 + self._local)
        self._capacity = 0
        self._workspace: dict[str, torch.Tensor] = {}

    def prepare_buffers(self, batch_size: int) -> None:
        """Reserve a reusable high-water mark bounded to512targets."""
        from ._warp.ssik_srs_geometry import CONTEXT_SIZE

        capacity = min(batch_size, self.max_chunk_size)
        if capacity <= self._capacity:
            return
        self._workspace = {
            "valid": torch.empty(
                (capacity, self.candidate_count), dtype=torch.int32, device=self.device
            ),
            "raw": torch.empty(
                (capacity, self.candidate_count, self._dof),
                dtype=torch.float64,
                device=self.device,
            ),
        }
        if self._dof == 7:
            self._workspace["context"] = torch.empty(
                (2 * capacity, CONTEXT_SIZE), dtype=torch.float64, device=self.device
            )
        self._capacity = capacity

    def generate(
        self,
        targets: torch.Tensor,
        seeds: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        weights: torch.Tensor,
        position_tolerance: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return every borrowed raw branch for flange goals and projected seeds."""
        count = len(targets)
        if count > self.max_chunk_size:
            raise ValueError("Geometric candidates require chunks of at most512targets")
        if not count:
            return (
                torch.empty(
                    (0, self.candidate_count), dtype=torch.int32, device=self.device
                ),
                torch.empty(
                    (0, self.candidate_count, self._dof),
                    dtype=torch.float64,
                    device=self.device,
                ),
            )
        self.prepare_buffers(count)
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            for value in (
                *self._workspace.values(),
                self._axes,
                self._offsets,
                targets,
                seeds,
                lower,
                upper,
                weights,
            ):
                value.record_stream(stream)
        valid, raw = self._workspace["valid"][:count], self._workspace["raw"][:count]
        if self._dof == 6:
            from ._warp.ssik_strict_6r import three_parallel_raw

            wp.launch(
                three_parallel_raw,
                count,
                inputs=[
                    self._h,
                    self._p,
                    self._home,
                    wp.from_torch(targets, dtype=wp.mat44d),
                    self._identity,
                    wp.from_torch(seeds),
                    wp.from_torch(lower),
                    wp.from_torch(upper),
                ],
                outputs=[wp.from_torch(valid), wp.from_torch(raw)],
                device=str(self.device),
            )
        else:
            from ._warp.ssik_srs_geometry import prepare_srs_context
            from ._warp.ssik_srs_optimize import optimize_srs_context, srs_raw

            context = wp.from_torch(self._workspace["context"][: 2 * count])
            axes = wp.from_torch(self._axes, dtype=wp.vec3d)
            wp.launch(
                prepare_srs_context,
                count,
                inputs=[
                    axes,
                    wp.from_torch(self._offsets, dtype=wp.vec3d),
                    self._home,
                    self._shoulder,
                    self._wrist,
                    wp.from_torch(targets, dtype=wp.mat44d),
                    self._identity,
                    wp.from_torch(seeds),
                    wp.from_torch(lower),
                    wp.from_torch(upper),
                    position_tolerance,
                    self._local,
                ],
                outputs=[context],
                device=str(self.device),
            )
            if self._local:
                optimizer, dim, block = optimize_srs_context, count, 256
                if self.device.type == "cuda":
                    from ._warp.ssik_srs_parallel import optimize_srs_parallel

                    optimizer, dim, block = optimize_srs_parallel, (count, 32), 32
                wp.launch(
                    optimizer,
                    dim,
                    inputs=[context, axes, wp.from_torch(weights), count],
                    device=str(self.device),
                    block_dim=block,
                )
            wp.launch(
                srs_raw,
                (count, self.candidate_count),
                inputs=[context, axes, self._samples, self._local, count],
                outputs=[wp.from_torch(valid), wp.from_torch(raw)],
                device=str(self.device),
            )
        return valid, raw
