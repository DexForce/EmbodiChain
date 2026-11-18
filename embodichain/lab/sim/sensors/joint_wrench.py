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

"""Six-axis parent reaction at a mounted point on an articulation link."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import math

import torch

from embodichain.utils import configclass

from .base_sensor import SensorCfg
from .state import _ReadOnlySensor

if TYPE_CHECKING:
    from .context import SensorContext

__all__ = ["JointWrench", "JointWrenchCfg"]


@configclass
class JointWrenchCfg(SensorCfg):
    """Select the reaction exerted across a named parent/child joint."""

    sensor_type: str = "JointWrench"
    sample_phase: Literal["physics"] = "physics"
    asset_uid: str = MISSING
    joint_name: str = MISSING
    offset: SensorCfg.OffsetCfg = SensorCfg.OffsetCfg()
    """Measurement frame relative to the joint's child link origin."""

    def get_data_types(self) -> list[str]:
        """Return force, torque and report validity field names."""
        return ["force", "torque", "wrench_valid"]


class JointWrench(_ReadOnlySensor):
    """Measure the parent-on-child wrench from the final solver substep.

    Force is in newtons. Torque is in newton-metres about the sensor origin.
    Both are expressed in the axes of the sensor mounted on the child link.
    """

    def __init__(self, config: JointWrenchCfg, *, context: SensorContext) -> None:
        if config.sample_phase != "physics":
            raise ValueError("JointWrench samples completed physics steps.")
        if config.mount is not None or config.offset.parent is not None:
            raise ValueError(
                "JointWrench takes an offset relative to its joint's child link."
            )
        if not all(
            map(math.isfinite, (*config.offset.pos, *config.offset.quat))
        ) or not any(config.offset.quat):
            raise ValueError(
                "JointWrench requires a finite offset and nonzero quaternion."
            )
        self._reader = None
        super().__init__(config, context=context)

    def on_bind(self) -> None:
        """Resolve the joint and initialize the per-environment measurements."""
        factory = self.context.joint_wrench_factory
        if factory is None:
            raise NotImplementedError(
                "JointWrench requires a native joint wrench provider."
            )
        self._last_step = self.context.clock.physics_step_index
        for name in ("force", "torque"):
            self._data_buffer[name] = torch.zeros(
                (self.num_instances, 3), dtype=torch.float32, device=self.device
            )
        self._data_buffer["wrench_valid"] = torch.zeros(
            self.num_instances, dtype=torch.bool, device=self.device
        )

        self._reader = factory(
            self.cfg.asset_uid,
            self.cfg.joint_name,
            self.cfg.offset,
            self._data_buffer["force"],
            self._data_buffer["torque"],
            self._data_buffer["wrench_valid"],
        )

    def on_unbind(self) -> None:
        """Release the native reader for the previous topology."""
        self._reader = None

    def update(self, **kwargs: object) -> None:
        """Transport the latest parent reaction to the configured mount."""
        step = self.context.clock.physics_step_index
        if self._reader is None or step <= self._last_step:
            return
        self._reader()
        self._last_step = step

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Invalidate the selected rows until a new physics step completes."""
        super().reset(env_ids)
        self._last_step = self.context.clock.physics_step_index

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Return the components' physical units and expression frame."""
        fields = super().get_data_descriptors()
        fields["force"].update(unit="N", frame="sensor")
        fields["torque"].update(unit="N·m", frame="sensor")
        return fields
