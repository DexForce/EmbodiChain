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
"""CPU tool-pose IK shared by ordered Cartesian planning and grasp feedback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

__all__: list[str] = []


class _TrajectoryIK:
    """Keep one ordered URDF chain for planning and feedback in the same arena."""

    def __init__(
        self,
        urdf: Path,
        tool_link: str,
        joint_names: tuple[str, ...],
        root_position: tuple[float, ...],
        root_rotation: tuple[float, ...],
    ) -> None:
        import pytorch_kinematics as pk

        self._chain = pk.build_serial_chain_from_urdf(urdf.read_bytes(), tool_link).to(
            dtype=torch.float64, device="cpu"
        )
        if tuple(self._chain.get_joint_parameter_names()) != joint_names:
            raise ValueError(
                "The selected arm joints differ from the ordered TCP chain."
            )
        limits = np.asarray(self._chain.get_joint_limits(), dtype=float).T
        self._lower = limits[:, 0] + 1e-5
        self._upper = limits[:, 1] - 1e-5
        if not np.isfinite(limits).all() or np.any(self._lower >= self._upper):
            raise ValueError("The TCP chain requires finite ordered joint limits.")
        self._root_position = np.asarray(root_position, dtype=float)
        self._root_rotation = (
            Rotation.from_euler("xyz", root_rotation, degrees=True).as_matrix()
            if any(root_rotation)
            else None
        )
        self._damping = np.eye(6) * 1e-8

    def forward(self, qpos: np.ndarray) -> np.ndarray:
        pose = (
            self._chain.forward_kinematics(torch.as_tensor(qpos, dtype=torch.float64))
            .get_matrix()
            .numpy()
        )
        if self._root_rotation is not None:
            pose[..., :3, :] = self._root_rotation @ pose[..., :3, :]
        pose[..., :3, 3] += self._root_position
        return pose[0] if qpos.ndim == 1 else pose

    def _delta(self, q: np.ndarray, error: np.ndarray) -> np.ndarray:
        jacobian = self._chain.jacobian(torch.as_tensor(q, dtype=torch.float64))[
            0
        ].numpy()
        if self._root_rotation is not None:
            jacobian[:3] = self._root_rotation @ jacobian[:3]
            jacobian[3:] = self._root_rotation @ jacobian[3:]
        jacobian[3:] *= 0.15
        return jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + self._damping, error
        )

    def _pose_error(
        self, target: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        current = self.forward(q)
        linear = target[:3, 3] - current[:3, 3]
        angular = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
        return linear, angular

    def plan_sample(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        *,
        position_tolerance: float,
        rotation_tolerance: float,
    ) -> np.ndarray:
        """Fit one nominal sample using the preceding sample as its IK seed."""
        q = seed.copy()
        position_stop = min(2e-7, position_tolerance)
        rotation_stop = min(2e-6, rotation_tolerance)
        for _ in range(45):
            linear, angular = self._pose_error(target, q)
            if (
                np.linalg.norm(linear) < position_stop
                and np.linalg.norm(angular) < rotation_stop
            ):
                return q
            delta = self._delta(q, np.r_[linear, 0.15 * angular])
            scale = min(1.0, 0.1 / max(np.max(np.abs(delta)), 1e-9))
            q = np.clip(q + scale * delta, self._lower, self._upper)

        def residual(candidate: np.ndarray) -> np.ndarray:
            linear, angular = self._pose_error(target, candidate)
            return np.r_[linear, 0.15 * angular]

        result = least_squares(
            residual,
            q,
            bounds=(self._lower, self._upper),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=160,
        )
        linear = np.linalg.norm(result.fun[:3])
        angular = np.linalg.norm(result.fun[3:]) / 0.15
        if linear > position_tolerance or angular > rotation_tolerance:
            raise ValueError(
                f"Unreachable TCP target {target[:3, 3]}: {linear:.6g} m, "
                f"{angular:.6g} rad."
            )
        return result.x

    def feedback_sample(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Apply a bounded local correction to one feedback sample."""
        q = seed.astype(float)
        for _ in range(20):
            linear, angular = self._pose_error(target, q)
            error = np.r_[linear, 0.15 * angular]
            if np.linalg.norm(error) < 1e-7:
                break
            q = np.clip(q + self._delta(q, error), self._lower, self._upper)
        return q.astype(np.float32)
