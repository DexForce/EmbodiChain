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
"""Static serial-chain adaptation and stream-safe fused FEP launches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from embodichain.compute.kinematics._warp.fep import solve

if TYPE_CHECKING:
    import pytorch_kinematics as pk
    from .fep_solver import FEPSolverCfg

__all__ = []


class _FEPWarpModel:
    """Fold fixed frames into eight origins without assuming a DH geometry."""

    def __init__(self, chain: pk.SerialChain, device: torch.device) -> None:
        origins = []
        axes = []
        pending = np.eye(4)
        for frame in chain._serial_frames:
            for offset in (frame.link.offset, frame.joint.offset):
                if offset is not None:
                    pending = pending @ offset.get_matrix()[
                        0
                    ].detach().cpu().numpy().astype(np.float64)
            if frame.joint.joint_type == "revolute":
                origins.append(pending)
                axes.append(frame.joint.axis.detach().cpu().numpy())
                pending = np.eye(4)
        origins.append(pending)
        self.device = device
        self.origins = torch.tensor(
            np.stack(origins), device=device, dtype=torch.float64
        )
        self.axes = torch.tensor(np.stack(axes), device=device, dtype=torch.float64)
        self.ready = None
        if device.type == "cuda":
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(device))
        wp.init()

    def solve(
        self,
        target: torch.Tensor,
        seed: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        tcp: np.ndarray,
        cfg: FEPSolverCfg,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch on the caller's Torch stream; return independently owned tensors."""
        stream = None
        target = target.contiguous()
        seed = seed.contiguous()
        lower = lower.to(seed).contiguous()
        upper = upper.to(seed).contiguous()
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            current.wait_event(self.ready)
            # Warp launches are invisible to Torch's caching allocator. Keep
            # inputs alive through work queued on a non-creation stream.
            for tensor in (target, seed, lower, upper, self.origins, self.axes):
                tensor.record_stream(current)
            stream = wp.stream_from_torch(current)
        tool = torch.as_tensor(tcp, device=self.device, dtype=torch.float64).reshape(
            1, 4, 4
        )
        valid = torch.empty(len(seed), device=self.device, dtype=torch.int32)
        joints = torch.empty_like(seed)
        wp.launch(
            solve,
            dim=len(seed),
            inputs=[
                wp.from_torch(target, dtype=wp.mat44f),
                wp.from_torch(seed),
                wp.from_torch(lower),
                wp.from_torch(upper),
                wp.from_torch(self.origins, dtype=wp.mat44d),
                wp.from_torch(self.axes, dtype=wp.vec3d),
                wp.from_torch(tool, dtype=wp.mat44d),
                cfg.max_iterations,
                wp.float64(cfg.damping),
                wp.float64(cfg.max_step),
                # Leave rounding margin for the float32 public FK verifier.
                wp.float64(cfg.position_tolerance * 0.5),
                wp.float64(cfg.rotation_tolerance * 0.5),
            ],
            outputs=[wp.from_torch(valid), wp.from_torch(joints)],
            device=str(self.device),
            stream=stream,
        )
        return valid.bool(), joints
