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


"""Planar contact arrays with separate validity for native impulse quantities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import math
from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from embodichain.utils import configclass
from embodichain.utils.device_utils import current_warp_stream

from .base_sensor import SensorCfg
from .contact import ArticulationContactFilterCfg, ContactSensor, ContactSensorCfg
from .contact._warp.grid import scatter_grid
from .state import _ReadOnlySensor

if TYPE_CHECKING:
    from .context import SensorContext

__all__ = ["ContactGrid", "ContactGridCfg"]


@configclass
class ContactGridCfg(SensorCfg):
    """Place a metric XY grid in a sensor mount to observe one rigid actor.

    Each cell counts contact points within the grid footprint and within
    `max_plane_distance` of its plane. The lower XY edges belong to the
    grid and the upper edges do not. Grid rows follow local +Y and columns +X.
    Set `mount.parent` to attach the array to a moving asset or link.
    """

    sensor_type: str = "ContactGrid"
    sample_phase: Literal["physics"] = "physics"
    mount: SensorCfg.OffsetCfg = SensorCfg.OffsetCfg()
    asset_uid: str = MISSING
    link_name: str | None = None
    size: tuple[float, float] = (0.02, 0.02)
    resolution: tuple[int, int] = (8, 8)
    max_plane_distance: float = 0.005
    max_contacts_per_env: int = 64

    def get_data_types(self) -> list[str]:
        """Return contact counts, impulse quantities and validity fields."""
        return [
            "count",
            "complete",
            "impulse_duration",
            "impulse_magnitude",
            "normal_impulse",
            "friction_impulse",
            "magnitude_valid",
            "normal_valid",
            "friction_valid",
        ]


class ContactGrid(_ReadOnlySensor):
    """Bin contact impulses at a completed physics boundary.

    Impulse vectors act on the selected actor and use the mounted sensor frame.
    The native scalar magnitude is total on Default CPU and normal on qualified
    backends. A quantity's validity is false when any contributing row lacks
    it, or when the native query or per-environment buffer dropped rows.
    Incomplete data can still have nonzero measured counts and sums.
    """

    def __init__(self, config: ContactGridCfg, *, context: SensorContext) -> None:
        if config.sample_phase != "physics" or config.sample_period != 0:
            raise ValueError("ContactGrid samples every completed physics step.")
        if config.mount is None:
            raise ValueError("ContactGrid requires a mount.")
        if len(config.size) != 2 or any(
            not math.isfinite(v) or v <= 0 for v in config.size
        ):
            raise ValueError("ContactGrid size must contain two positive lengths.")
        if len(config.resolution) != 2 or any(
            type(n) is not int or n <= 0 for n in config.resolution
        ):
            raise ValueError(
                "ContactGrid resolution must contain positive width and height."
            )
        if (
            not math.isfinite(config.max_plane_distance)
            or config.max_plane_distance < 0
        ):
            raise ValueError("max_plane_distance must be finite and nonnegative.")
        if (
            type(config.max_contacts_per_env) is not int
            or config.max_contacts_per_env <= 0
        ):
            raise ValueError("max_contacts_per_env must be a positive integer.")
        self._reader: ContactSensor | None = None
        super().__init__(config, context=context)

    def on_bind(self) -> None:
        wp.get_device(str(self.device))
        factory = self.context.contact_factory
        if factory is None:
            raise NotImplementedError("ContactGrid requires a contact provider.")
        target = self.cfg.asset_uid
        link = self.cfg.link_name
        reader = factory(
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
        self._reader = reader
        self._actors = reader.get_actor_ids(
            target, None if link is None else [link]
        ).reshape(-1)
        width, height = self.cfg.resolution
        batch = (self.num_instances,)
        grid = (*batch, height, width)
        data = self._data_buffer
        data["count"] = torch.zeros(grid, dtype=torch.int32, device=self.device)
        data["complete"] = torch.zeros(batch, dtype=torch.bool, device=self.device)
        data["impulse_duration"] = torch.zeros(batch, device=self.device)
        data["impulse_magnitude"] = torch.zeros(grid, device=self.device)
        for name in ("normal_impulse", "friction_impulse"):
            data[name] = torch.zeros((*grid, 3), device=self.device)
        for name in ("magnitude_valid", "normal_valid", "friction_valid"):
            data[name] = torch.zeros(grid, dtype=torch.bool, device=self.device)
        self._measured = torch.zeros((*grid, 3), dtype=torch.int32, device=self.device)

    def on_unbind(self) -> None:
        if self._reader is not None:
            self._reader.destroy()
            self._reader = None

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Invalidate selected rows and clear their accumulated impulses."""
        super().reset(env_ids)
        if self._reader is not None:
            self._reader.reset(env_ids)

    def update(self, **kwargs: object) -> None:
        """Sample native contacts and bin measured impulses in the mount frame."""
        reader = self._reader
        if reader is None:
            raise RuntimeError("ContactGrid must be bound before sampling.")
        dt = float(kwargs["sample_dt"])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("ContactGrid requires a completed positive physics step.")
        reader.update()
        rows = reader.get_data()
        data = self._data_buffer
        for name in (
            "count",
            "impulse_magnitude",
            "normal_impulse",
            "friction_impulse",
        ):
            data[name].zero_()
        self._measured.zero_()
        use_normal = reader.normal_impulse_available
        width, height = self.cfg.resolution
        wp.launch(
            scatter_grid,
            dim=rows["is_valid"].shape,
            inputs=[
                wp.from_torch(rows["position"], dtype=wp.vec3),
                wp.from_torch(rows["normal"], dtype=wp.vec3),
                wp.from_torch(rows["friction"], dtype=wp.vec3),
                wp.from_torch(rows["impulse"]),
                wp.from_torch(rows["user_ids"], dtype=wp.vec2i),
                wp.from_torch(rows["is_valid"]),
                wp.from_torch(rows["impulse_valid"]),
                wp.from_torch(rows["friction_valid"]),
                wp.from_torch(self._actors),
                wp.from_torch(self.get_mount_pose().contiguous(), dtype=wp.mat44),
                *self.cfg.size,
                self.cfg.max_plane_distance,
                width,
                height,
                use_normal,
            ],
            outputs=[
                wp.from_torch(data["count"]),
                wp.from_torch(self._measured),
                wp.from_torch(data["impulse_magnitude"]),
                wp.from_torch(data["normal_impulse"], dtype=wp.vec3),
                wp.from_torch(data["friction_impulse"], dtype=wp.vec3),
            ],
            device=str(self.device),
            stream=current_warp_stream(self.device),
        )
        data["impulse_duration"].fill_(reader.get_impulse_duration(dt))
        data["complete"].copy_(~reader.get_latest_contact_loss_mask())
        capabilities = reader.contact_capabilities
        count = data["count"]
        for i, (name, supported) in enumerate(
            (
                ("magnitude_valid", capabilities.impulse),
                ("normal_valid", use_normal and capabilities.impulse),
                ("friction_valid", capabilities.friction),
            )
        ):
            measured = self._measured[..., i]
            data[name].copy_(
                data["complete"][:, None, None]
                & (measured == count)
                & ((measured > 0) | supported)
            )

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Describe the impulse units and the source of the scalar magnitude."""
        fields = super().get_data_descriptors()
        fields["impulse_duration"].update(unit="s")
        fields["impulse_magnitude"].update(unit="N·s")
        if self._reader is not None:
            fields["impulse_magnitude"].update(
                quantity="normal" if self._reader.normal_impulse_available else "total"
            )
        for name in ("normal_impulse", "friction_impulse"):
            fields[name].update(unit="N·s", frame="mount")
        return fields
