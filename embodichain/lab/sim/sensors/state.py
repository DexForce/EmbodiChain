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

"""Reference sensors for joint coordinates and relative frame transforms."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import torch

from embodichain.utils import configclass
from embodichain.utils.math import matrix_from_quat

from .base_sensor import BaseSensor, SensorCfg

if TYPE_CHECKING:
    from .context import SensorContext

__all__ = ["JointEncoder", "JointEncoderCfg", "FrameTransform", "FrameTransformCfg"]


@configclass
class JointEncoderCfg(SensorCfg):
    """Select measured generalized coordinates in declared joint order."""

    sensor_type: str = "JointEncoder"
    sample_phase: Literal["physics"] = "physics"
    asset_uid: str = MISSING
    joint_names: tuple[str, ...] | None = None

    def get_data_types(self) -> list[str]:
        """Return the position and velocity field names."""
        return ["qpos", "qvel"]


@configclass
class FrameTransformCfg(SensorCfg):
    """Observe an offset target frame in a named source frame or the arena."""

    sensor_type: str = "FrameTransform"
    sample_phase: Literal["physics"] = "physics"
    target_uid: str = MISSING
    target_link: str | None = None
    source_uid: str | None = None
    source_link: str | None = None
    target_offset: SensorCfg.OffsetCfg = SensorCfg.OffsetCfg()
    source_offset: SensorCfg.OffsetCfg = SensorCfg.OffsetCfg()

    def get_data_types(self) -> list[str]:
        """Return the relative transform field name."""
        return ["transform"]


class _ReadOnlySensor(BaseSensor):
    def _build_sensor_from_config(
        self, config: SensorCfg, device: torch.device
    ) -> None:
        self._data_buffer.update(
            {
                name: torch.empty((self.num_instances, 0), device=device)
                for name in config.get_data_types()
            }
        )

    def get_arena_pose(self, to_matrix: bool = False) -> torch.Tensor | None:
        if self._mount_pose is not None:
            from embodichain.utils.math import quat_from_matrix

            matrix = self.get_mount_pose()
            if to_matrix:
                return matrix
            return torch.cat(
                (matrix[:, :3, 3], quat_from_matrix(matrix[:, :3, :3])), dim=-1
            )
        return None

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor | None:
        return self.get_arena_pose(to_matrix)

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: list[int] | None = None
    ) -> None:
        raise NotImplementedError("Configure the local sensor mount in its SensorCfg.")

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        rows = (
            slice(None)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        for value in self._data_buffer.values():
            value[rows] = 0
        self._invalidate_samples(env_ids)


class JointEncoder(_ReadOnlySensor):
    """Sample joint position and velocity through the sensor state provider.

    Args:
        config: Articulation identity and optional named joint selection.
        context: Explicit runtime state services.
    """

    def __init__(self, config: JointEncoderCfg, *, context: SensorContext) -> None:
        super().__init__(config, context=context)

    def on_bind(self) -> None:
        names = self.context.state.get_joint_names(self.cfg.asset_uid)
        selected = self.cfg.joint_names
        self._indices = None
        if selected is not None:
            if len(set(selected)) != len(selected) or any(
                name not in names for name in selected
            ):
                raise ValueError(
                    f"Invalid joint selection {selected!r}; available joints: {names!r}."
                )
            self._indices = torch.tensor(
                [names.index(name) for name in selected],
                device=self.device,
                dtype=torch.long,
            )
        width = len(names) if selected is None else len(selected)
        for field in self.cfg.get_data_types():
            self._data_buffer[field] = torch.zeros(
                (self.num_instances, width), device=self.device
            )

    def update(self, **kwargs: object) -> None:
        """Acquire the currently selected joint position and velocity."""
        values = self.context.state.get_joint_state(self.cfg.asset_uid)
        for field in self.cfg.get_data_types():
            value = values[field]
            if self._indices is not None:
                value = value.index_select(1, self._indices)
            self._data_buffer[field].copy_(value)

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Return descriptors with generalized angular/linear joint units."""
        fields = super().get_data_descriptors()
        fields["qpos"]["unit"] = "rad or m"
        fields["qvel"]["unit"] = "rad/s or m/s"
        return fields


def _pose_matrix(pose: torch.Tensor) -> torch.Tensor:
    result = (
        torch.eye(4, device=pose.device, dtype=pose.dtype)
        .expand(pose.shape[0], -1, -1)
        .clone()
    )
    result[:, :3, :3] = matrix_from_quat(pose[:, 3:])
    result[:, :3, 3] = pose[:, :3]
    return result


class FrameTransform(_ReadOnlySensor):
    """Sample the transform from source coordinates to target pose.

    Args:
        config: Source and target identities with optional local offsets.
        context: Explicit runtime state services.
    """

    def __init__(self, config: FrameTransformCfg, *, context: SensorContext) -> None:
        super().__init__(config, context=context)

    def on_bind(self) -> None:
        if self.cfg.source_uid is None and self.cfg.source_link is not None:
            raise ValueError("A source link requires source_uid.")
        self._target_offset = self.cfg.target_offset.transformation.to(self.device)
        self._source_offset = self.cfg.source_offset.transformation.to(self.device)
        self._data_buffer["transform"] = torch.zeros(
            (self.num_instances, 4, 4), device=self.device
        )
        self.update()

    def update(self, **kwargs: object) -> None:
        """Acquire the target transform expressed relative to the source."""
        provider = self.context.state
        target = (
            _pose_matrix(
                provider.get_frame_state(self.cfg.target_uid, self.cfg.target_link)[
                    "pose"
                ]
            )
            @ self._target_offset
        )
        source = self._source_offset
        if self.cfg.source_uid is not None:
            source = (
                _pose_matrix(
                    provider.get_frame_state(self.cfg.source_uid, self.cfg.source_link)[
                        "pose"
                    ]
                )
                @ source
            )
        rotation_t = source[..., :3, :3].transpose(-2, -1)
        result = self._data_buffer["transform"]
        result[:, :3, :3] = rotation_t @ target[:, :3, :3]
        result[:, :3, 3] = (
            rotation_t @ (target[:, :3, 3] - source[..., :3, 3]).unsqueeze(-1)
        ).squeeze(-1)
        result[:, 3, 3] = 1

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Return the homogeneous-transform descriptor."""
        fields = super().get_data_descriptors()
        fields["transform"].update(unit="m for translation", frame="source")
        return fields
