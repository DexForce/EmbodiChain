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

"""Batch contact-resultant reference sensor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import torch

from embodichain.utils import configclass

from .base_sensor import SensorCfg
from .contact import ArticulationContactFilterCfg, ContactSensorCfg
from .state import _ReadOnlySensor

if TYPE_CHECKING:
    from .context import SensorContext

__all__ = ["ContactResultant", "ContactResultantCfg"]


@configclass
class ContactResultantCfg(SensorCfg):
    """Measure summed contact forces acting on one rigid body or link."""

    sensor_type: str = "ContactResultant"
    # Use every physics boundary so the native impulse interval is unambiguous.
    sample_phase: Literal["physics"] = "physics"
    asset_uid: str = MISSING
    link_name: str | None = None
    max_contacts_per_env: int = 64

    def get_data_types(self) -> list[str]:
        """Return force components and their individual validity fields."""
        return [
            "normal_force",
            "friction_force",
            "normal_valid",
            "friction_valid",
            "contact",
        ]


class ContactResultant(_ReadOnlySensor):
    """Sum available arena-frame impulse vectors for a monitored body.

    Args:
        config: Target identity and query capacity.
        context: Explicit runtime contact acquisition service.
    """

    def __init__(self, config: ContactResultantCfg, *, context: SensorContext) -> None:
        if config.sample_phase != "physics" or config.sample_period != 0:
            raise ValueError("ContactResultant samples every completed physics step.")
        self._reader = None
        super().__init__(config, context=context)

    def on_bind(self) -> None:
        if self.context.contact_factory is None:
            raise NotImplementedError(
                "ContactResultant requires a public contact provider."
            )
        target = self.cfg.asset_uid
        link = self.cfg.link_name
        self._reader = self.context.contact_factory(
            ContactSensorCfg(
                uid=f"{self.uid}_reader",
                rigid_uid_list=[target] if link is None else [],
                articulation_cfg_list=(
                    []
                    if link is None
                    else [
                        ArticulationContactFilterCfg(
                            articulation_uid=target, link_name_list=[link]
                        )
                    ]
                ),
                filter_need_both_actor=False,
                max_contacts_per_env=self.cfg.max_contacts_per_env,
            )
        )
        self._actors = self._reader.get_actor_ids(
            target, None if link is None else [link]
        )
        for field in self.cfg.get_data_types():
            is_force = field.endswith("_force")
            self._data_buffer[field] = torch.zeros(
                (self.num_instances, 3) if is_force else (self.num_instances,),
                dtype=torch.float32 if is_force else torch.bool,
                device=self.device,
            )

    def on_unbind(self) -> None:
        if self._reader is not None:
            self._reader.destroy()
            self._reader = None

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear selected outputs and the corresponding contact reader rows."""
        super().reset(env_ids)
        if self._reader is not None:
            self._reader.reset(env_ids)

    def update(self, **kwargs: object) -> None:
        """Read the current contact snapshot and reduce only measured rows."""
        reader = self._reader
        dt = float(kwargs["sample_dt"])
        if dt <= 0:
            raise ValueError(
                "Contact sampling requires a completed positive physics step."
            )
        reader.update()
        reader._wait_for_sample()
        rows = reader.get_data()
        ids = rows["user_ids"]
        # Actor 0 receives -normal but the native tangential impulse is on actor 0.
        a = ids[..., 0] == self._actors
        b = ids[..., 1] == self._actors
        valid = rows["is_valid"] & (a ^ b)
        sign = (b.to(torch.float32) - a.to(torch.float32)).unsqueeze(-1)
        self._data_buffer["contact"].copy_(valid.any(dim=1))
        complete = ~reader.get_latest_contact_loss_mask()
        duration = reader.get_impulse_duration(dt)
        quantities = (
            (
                "normal",
                rows["normal"] * rows["impulse"].reshape_as(valid).unsqueeze(-1) * sign,
                rows["impulse_valid"],
                reader.normal_impulse_available and reader.contact_capabilities.impulse,
            ),
            (
                "friction",
                rows["friction"] * -sign,
                rows["friction_valid"],
                reader.contact_capabilities.friction,
            ),
        )
        for name, values, valid_values, supported in quantities:
            valid_values = valid_values.reshape_as(valid)
            measured = valid & valid_values
            result_valid = complete & ~(valid & ~valid_values).any(dim=1)
            result_valid &= measured.any(dim=1) | bool(supported)
            if name == "normal" and not reader.normal_impulse_available:
                result_valid.fill_(False)
            force = torch.where(measured[..., None], values, 0).sum(dim=1) / duration
            self._data_buffer[f"{name}_force"].copy_(
                torch.where(result_valid[:, None], force, 0)
            )
            self._data_buffer[f"{name}_valid"].copy_(result_valid)

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Return component-wise observability and arena force units."""
        fields = super().get_data_descriptors()
        for name in ("normal_force", "friction_force"):
            fields[name].update(unit="N", frame="arena")
        return fields
