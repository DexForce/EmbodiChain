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

"""URDF-based gravity feedforward from encoder coordinates."""

from __future__ import annotations

from collections.abc import Sequence
import xml.etree.ElementTree as ET

import torch
import pytorch_kinematics as pk

__all__ = ["URDFGravityCompensation"]


class URDFGravityCompensation:
    """Cache URDF kinematics and compute gravity compensation in joint order.

    Mass/COM values come from the supplied URDF. The model does not read a
    simulator, configuration registry, or asset cache. Returned torques are
    detached feedforward values, computed as the gradient of potential energy.

    Args:
        urdf_xml: Complete URDF document, not a filesystem path.
        gravity: Three-component acceleration in the robot root frame.
        device: Device used by cached kinematics and COM tensors.
        dtype: Floating dtype used by the model and its input coordinates.
    """

    @torch.inference_mode(False)
    def __init__(
        self,
        urdf_xml: str | bytes,
        gravity: Sequence[float] | torch.Tensor = (0.0, 0.0, -9.81),
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._chain = pk.build_chain_from_urdf(urdf_xml).to(device=device, dtype=dtype)
        self.joint_names: tuple[str, ...] = tuple(
            self._chain.get_joint_parameter_names()
        )
        self._gravity = (
            torch.as_tensor(gravity, dtype=dtype, device=device).detach().clone()
        )
        if self._gravity.shape != (3,):
            raise ValueError("Gravity must contain three root-frame components.")
        self._links: list[tuple[str, float, torch.Tensor]] = []
        for link in ET.fromstring(urdf_xml).findall("link"):
            inertia = link.find("inertial")
            if inertia is None:
                continue
            origin = inertia.find("origin")
            com = [
                float(value)
                for value in (
                    origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
                ).split()
            ]
            self._links.append(
                (
                    link.attrib["name"],
                    float(inertia.find("mass").attrib["value"]),
                    torch.tensor(com, dtype=dtype, device=device),
                )
            )

    def torques(self, qpos: torch.Tensor) -> torch.Tensor:
        """Return gravity compensation for each coordinate in ``joint_names``.

        Args:
            qpos: Root-frame encoder coordinates shaped ``[batch, joints]``
                matching the cached model dtype/device and joint order.

        Returns:
            Owned, detached joint torques shaped like ``qpos``.

        Raises:
            ValueError: Input layout, dtype or device differs from the model.
        """
        if qpos.ndim != 2 or qpos.shape[1] != len(self.joint_names):
            raise ValueError(
                "Encoder coordinates must have shape [batch, URDF joints]."
            )
        if qpos.device != self._gravity.device or qpos.dtype != self._gravity.dtype:
            raise ValueError(
                "Encoder coordinates must match the gravity model dtype/device."
            )
        if not self._links or qpos.numel() == 0:
            return torch.zeros_like(qpos)
        # Differentiate a local potential even when the caller is doing
        # inference. Backward can only save ordinary tensors.
        with torch.inference_mode(False), torch.enable_grad():
            q = qpos.detach()
            if q.is_inference():
                q = q.clone()
            q.requires_grad_(True)
            transforms = self._chain.forward_kinematics(q)
            potential = torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
            for name, mass, com in self._links:
                transform = transforms[name].get_matrix()
                position = transform[:, :3, :3] @ com + transform[:, :3, 3]
                potential = potential - mass * (position * self._gravity).sum(dim=-1)
            if not potential.requires_grad:
                return torch.zeros_like(qpos)
            torque = torch.autograd.grad(potential.sum(), q)[0]
        return torque.detach()
