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

"""A sensor plugin that consumes only the public SensorContext contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from embodichain.lab.sim.sensors import (
    BaseSensor,
    SensorCfg,
    SensorContext,
    SensorRegistration,
)
from embodichain.utils import configclass

__all__ = ["JointEncoderCfg", "JointEncoder", "SENSOR"]


@configclass
class JointEncoderCfg(SensorCfg):
    """Observe all generalized joint positions and velocities of one asset."""

    sensor_type: str = "ExampleJointEncoder"
    asset_uid: str = MISSING

    def get_data_types(self) -> list[str]:
        """Describe the encoder output fields.

        Returns:
            Joint position and velocity field names.
        """
        return ["qpos", "qvel"]


class JointEncoder(BaseSensor):
    """Joint encoder backed by the public state acquisition service.

    Args:
        config: Robot/articulation UID to observe.
        context: Explicit simulation ownership and state services.
    """

    def __init__(self, config: JointEncoderCfg, *, context: SensorContext) -> None:
        config.validate()
        super().__init__(config, context=context)

    def _build_sensor_from_config(
        self, config: JointEncoderCfg, device: torch.device
    ) -> None:
        self._asset_uid = config.asset_uid
        self.update()

    def update(self, **kwargs: object) -> None:
        """Acquire owned joint-state snapshots without advancing physics.

        Args:
            **kwargs: Optional host acquisition hints, including fetch_only.
        """
        self._data_buffer.update(self.context.state.get_joint_state(self._asset_uid))

    def get_arena_pose(self, to_matrix: bool = False) -> torch.Tensor | None:
        """Return None because joint encoders have no spatial measurement frame.

        Args:
            to_matrix: Accepted for BaseSensor compatibility.

        Returns:
            None for this non-spatial sensor.
        """
        return None

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor | None:
        """Return None for a non-spatial sensor.

        Args:
            to_matrix: Accepted for BatchEntity compatibility.

        Returns:
            None because the encoder measures joint coordinates.
        """
        return None

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Reject spatial pose writes for a joint-state sensor.

        Args:
            pose: Unused spatial transform.
            env_ids: Unused environment selection.

        Raises:
            NotImplementedError: Encoders have no spatial measurement frame.
        """
        raise NotImplementedError("Joint encoders have no spatial measurement frame.")

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear selected output rows; the next update reacquires joint state.

        Args:
            env_ids: Environment rows to clear, or all rows when omitted.
        """
        rows = (
            slice(None)
            if env_ids is None
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        for value in self._data_buffer.values():
            value[rows] = 0


SENSOR = SensorRegistration("ExampleJointEncoder", JointEncoderCfg, JointEncoder)
