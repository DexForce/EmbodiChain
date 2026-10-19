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

"""Simulation-independent serial-chain FK and TCP Jacobian adapter."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pytorch_kinematics import SerialChain

__all__ = ["SerialChainKinematics"]


class SerialChainKinematics:
    """Own a PK chain without mutable per-call tensors or simulation objects.

    Args:
        chain: Serial chain to copy. The caller's model is never moved/mutated.
        device: Tensor device.
        dtype: Float32 for training or float64 for numerical validation.
    """

    def __init__(
        self,
        chain: SerialChain,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Kinematics supports float32 and float64")
        device = torch.empty(0, device=device).device
        self.chain = deepcopy(chain).to(device=device, dtype=dtype)
        self.joint_names = tuple(self.chain.get_joint_parameter_names())
        self.dof = len(self.joint_names)
        if self.dof < 1:
            raise ValueError("A kinematics chain must have at least one joint")
        # A checkpoint must be loaded into the same fixed geometry. Hash the
        # serial frames in a canonical precision independently of runtime dtype.
        digest = hashlib.sha256(repr(self.joint_names).encode())
        for frame in self.chain._serial_frames:
            digest.update(repr((frame.name, frame.joint.joint_type)).encode())
            for tensor in (
                frame.joint.axis,
                None if frame.link.offset is None else frame.link.offset.get_matrix(),
                None if frame.joint.offset is None else frame.joint.offset.get_matrix(),
            ):
                if tensor is not None:
                    digest.update(
                        tensor.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                        .numpy()
                        .tobytes()
                    )
        self.fingerprint = digest.hexdigest()

    @classmethod
    def from_urdf(
        cls,
        urdf: str,
        end_link_name: str,
        root_link_name: str | None = None,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> SerialChainKinematics:
        """Build from URDF text; parsing and model construction happen once.

        Args:
            urdf: URDF XML text, not a path.
            end_link_name: Controlled flange frame.
            root_link_name: Optional chain root frame.
            device: Tensor device.
            dtype: Tensor precision.

        Returns:
            An independent serial-chain model.
        """
        import pytorch_kinematics as pk

        chain = pk.build_serial_chain_from_urdf(
            urdf, end_link_name, root_link_name=root_link_name or ""
        )
        return cls(chain, device, dtype)

    def to(
        self, device: torch.device | str, dtype: torch.dtype
    ) -> SerialChainKinematics:
        """Return a model copy on the requested device and precision."""
        return type(self)(self.chain, device, dtype)

    def evaluate(
        self, qpos: torch.Tensor, tcp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute TCP transforms and geometric Jacobians using one FK pass.

        Args:
            qpos: Joint positions, shape ``(..., dof)``.
            tcp: Flange-to-tool transforms, shape ``(4, 4)`` or ``(..., 4, 4)``.

        Returns:
            TCP poses ``(..., 4, 4)`` and root-frame Jacobians ``(..., 6, dof)``.
            Their backward paths include both FK and Jacobian derivatives.
        """
        if (
            qpos.ndim < 1
            or qpos.shape[-1] != self.dof
            or tcp.shape not in ((4, 4), (*qpos.shape[:-1], 4, 4))
        ):
            raise ValueError("Invalid qpos or TCP shape")
        if (
            qpos.device != self.chain.device
            or qpos.dtype != self.chain.dtype
            or tcp.device != qpos.device
            or tcp.dtype != qpos.dtype
        ):
            raise ValueError("qpos, TCP and kinematics model must share dtype/device")
        batch = qpos.shape[:-1]
        q = qpos.reshape(-1, self.dof)
        transforms = self.chain.forward_kinematics_tensor(q, analytical_grad=False)
        jacobian, flange = self.chain.jacobian_tensor(
            q, ret_eef_pose=True, all_transforms=transforms
        )
        tool = flange @ tcp.reshape(-1, 4, 4)
        offset = tool[:, :3, 3] - flange[:, :3, 3]
        correction = torch.cross(
            jacobian[:, 3:, :].transpose(-1, -2),
            offset[:, None, :].expand(-1, self.dof, -1),
            dim=-1,
        ).transpose(-1, -2)
        tool_jacobian = torch.cat(
            (jacobian[:, :3, :] + correction, jacobian[:, 3:, :]), dim=-2
        )
        return tool.reshape(*batch, 4, 4), tool_jacobian.reshape(*batch, 6, self.dof)
