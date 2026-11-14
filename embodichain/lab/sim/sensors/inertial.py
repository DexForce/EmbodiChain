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

"""Pose, velocity, acceleration and IMU sensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import torch

from embodichain.utils import configclass
from embodichain.utils.math import matrix_from_quat, quat_mul

from .base_sensor import SensorCfg
from .state import _ReadOnlySensor

if TYPE_CHECKING:
    from .context import SensorContext

__all__ = ["PVA", "PVACfg", "IMU", "IMUCfg"]


@configclass
class PVACfg(SensorCfg):
    """Measure a named rigid-body or link origin with optional sensor offset."""

    sensor_type: str = "PVA"
    sample_phase: Literal["physics"] = "physics"
    asset_uid: str = MISSING
    link_name: str | None = None
    offset: SensorCfg.OffsetCfg = SensorCfg.OffsetCfg()

    acceleration_source: Literal["finite_difference", "native", "auto"] = (
        "finite_difference"
    )
    """Acceleration acquisition mode. Auto selects the source at bind time.

    Native sampling currently requires a dynamic rigid object on CUDA Newton
    MuJoCo Warp with RNE enabled. Auto falls back to finite differences when
    the backend or target is unsupported. Native mode raises during binding.
    """

    def get_data_types(self) -> list[str]:
        """Return measured PVA fields and acceleration validity."""
        return [
            "pose",
            "linear_velocity",
            "angular_velocity",
            "linear_acceleration",
            "angular_acceleration",
            "acceleration_valid",
        ]


@configclass
class IMUCfg(PVACfg):
    """Approximate a gyroscope and accelerometer at a rigid sensor mount."""

    sensor_type: str = "IMU"

    def get_data_types(self) -> list[str]:
        """Return local angular velocity, specific force and its validity."""
        return ["angular_velocity", "specific_force", "acceleration_valid"]


class PVA(_ReadOnlySensor):
    """Sample pose, velocity and acceleration at a body-mounted point.

    Args:
        config: Frame and acquisition configuration.
        context: Manager-owned state and time providers.
    """

    def __init__(self, config: PVACfg, *, context: SensorContext) -> None:
        super().__init__(config, context=context)

    def on_bind(self) -> None:
        self.context.state.get_frame_state(self.cfg.asset_uid, self.cfg.link_name)
        offset = self.cfg.offset.transformation.to(self.device)
        self._offset_position = offset[:3, 3]
        self._offset_quat = torch.as_tensor(self.cfg.offset.quat, device=self.device)
        self._previous_linear = torch.zeros((self.num_instances, 3), device=self.device)
        self._previous_angular = torch.zeros_like(self._previous_linear)
        self._previous_valid = torch.zeros(
            self.num_instances, dtype=torch.bool, device=self.device
        )
        source = self.cfg.acceleration_source
        if source not in ("finite_difference", "native", "auto"):
            raise ValueError(f"Unknown acceleration_source {source!r}.")
        factory = self.context.acceleration_factory
        self._read_native = (
            factory(self.cfg.asset_uid, self.cfg.link_name)
            if source != "finite_difference" and factory is not None
            else None
        )
        if source == "native" and self._read_native is None:
            raise NotImplementedError(
                f"Native acceleration is unavailable for {self.cfg.asset_uid!r} "
                f"(link={self.cfg.link_name!r}). Use a supported dynamic rigid body "
                "on CUDA Newton MuJoCo Warp with RNE enabled, or choose auto."
            )
        self._native_after_step = self.context.clock.physics_step_index
        self._last_time = None
        for name in self.cfg.get_data_types():
            if name == "acceleration_valid":
                self._data_buffer[name] = torch.zeros_like(self._previous_valid)
            else:
                self._data_buffer[name] = torch.zeros(
                    (self.num_instances, 7 if name == "pose" else 3), device=self.device
                )

    def on_unbind(self) -> None:
        """Release the acceleration reader for the previous topology."""
        self._read_native = None

    def update(self, **kwargs: object) -> None:
        """Sample the configured point using the source selected at bind time."""
        if self._bound_topology is None:
            return
        now = self.context.clock.now()
        if now == self._last_time or (
            self._read_native is not None
            and self.context.clock.physics_step_index <= self._native_after_step
        ):
            return
        state = self.context.state.get_frame_state(
            self.cfg.asset_uid, self.cfg.link_name
        )
        pose = state["pose"]
        rotation = matrix_from_quat(pose[:, 3:])
        offset = rotation @ self._offset_position
        orientation = quat_mul(
            pose[:, 3:], self._offset_quat.expand(self.num_instances, -1)
        )
        angular = state["angular_velocity"]
        if self._read_native is not None:
            dv, dw, com_offset = self._read_native()
            arm = (
                rotation @ (self._offset_position - com_offset).unsqueeze(-1)
            ).squeeze(-1)
            point_velocity = torch.linalg.cross(angular, arm)
            linear = state["linear_velocity"] + point_velocity
            dv = (
                dv
                + torch.linalg.cross(dw, arm)
                + torch.linalg.cross(angular, point_velocity)
            )
            valid = torch.isfinite(dv).all(dim=1) & torch.isfinite(dw).all(dim=1)
        else:
            linear = state["linear_velocity"] + torch.linalg.cross(angular, offset)
            elapsed = now - self._last_time if self._last_time is not None else 0.0
            if elapsed > 0:
                dv = (linear - self._previous_linear) / elapsed
                dw = (angular - self._previous_angular) / elapsed
                valid = self._previous_valid
            else:
                dv = dw = torch.zeros_like(linear)
                valid = torch.zeros_like(self._previous_valid)

        self._data_buffer["acceleration_valid"].copy_(valid)
        self._publish(
            pose[:, :3] + offset,
            orientation,
            linear,
            angular,
            torch.where(valid[:, None], dv, 0),
            torch.where(valid[:, None], dw, 0),
        )
        if self._read_native is None:
            self._previous_linear.copy_(linear)
            self._previous_angular.copy_(angular)
            self._previous_valid.fill_(True)
        self._last_time = now

    def _publish(self, position, orientation, linear, angular, dv, dw) -> None:
        for name, value in (
            ("pose", torch.cat((position, orientation), dim=-1)),
            ("linear_velocity", linear),
            ("angular_velocity", angular),
            ("linear_acceleration", dv),
            ("angular_acceleration", dw),
        ):
            self._data_buffer[name].copy_(value)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Invalidate acceleration and history for the selected environments."""
        super().reset(env_ids)
        rows = (
            slice(None)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self._previous_valid[rows] = False
        if self._read_native is not None:
            self._native_after_step = self.context.clock.physics_step_index

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Return arena-relative kinematic quantities and their units."""
        fields = super().get_data_descriptors()
        for name, unit in {
            "pose": "m and xyzw",
            "linear_velocity": "m/s",
            "angular_velocity": "rad/s",
            "linear_acceleration": "m/s²",
            "angular_acceleration": "rad/s²",
        }.items():
            if name in fields:
                fields[name].update(unit=unit, frame="arena")
        if self._bound_topology is not None:
            fields["acceleration_valid"]["source"] = (
                "native" if self._read_native is not None else "finite_difference"
            )
        return fields


class IMU(PVA):
    """Sample local angular velocity and specific force from rigid-body state."""

    def on_bind(self) -> None:
        super().on_bind()
        self._gravity = self.context.state.get_gravity()

    def _publish(self, position, orientation, linear, angular, dv, dw) -> None:
        rotation_t = matrix_from_quat(orientation).transpose(-2, -1)
        self._data_buffer["angular_velocity"].copy_(
            (rotation_t @ angular.unsqueeze(-1)).squeeze(-1)
        )
        specific_force = (rotation_t @ (dv - self._gravity).unsqueeze(-1)).squeeze(-1)
        self._data_buffer["specific_force"].copy_(
            torch.where(
                self._data_buffer["acceleration_valid"][:, None], specific_force, 0
            )
        )

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Return sensor-frame angular velocity and specific force descriptors."""
        fields = super().get_data_descriptors()
        fields["angular_velocity"].update(frame="sensor")
        fields["specific_force"].update(unit="m/s²", frame="sensor")
        return fields
