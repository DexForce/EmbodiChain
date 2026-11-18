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


"""Geometric indentation of planar or curved tactile pads."""

from __future__ import annotations

from dataclasses import MISSING
import math
from typing import TYPE_CHECKING, Literal

import torch

from embodichain.compute.geometry._indentation import _IndentationQuery
from embodichain.utils import configclass

from .base_sensor import SensorCfg
from .ray import RayScene
from .state import _ReadOnlySensor

if TYPE_CHECKING:
    from .context import SensorContext

__all__ = ["TactileIndentation", "TactileIndentationCfg"]


@configclass
class TactileIndentationCfg(SensorCfg):
    """Configure a normal-aligned geometric tactile layout.

    Resolution is (width, height). By default probe centers fill the specified
    local XY size and point toward +Z. A positive curvature radius bends the
    layout into a spherical cap. Supply row-major points and outward normals
    together to use an explicitly calibrated curved layout.
    """

    sensor_type: str = "TactileIndentation"
    sample_phase: Literal["physics"] = "physics"
    mount: SensorCfg.OffsetCfg = SensorCfg.OffsetCfg()
    target_uids: tuple[str, ...] = MISSING
    geometry_source: Literal["collision", "visual"] = "collision"
    resolution: tuple[int, int] = (16, 16)
    size: tuple[float, float] = (0.02, 0.02)
    curvature_radius: float | None = None
    points: tuple[tuple[float, float, float], ...] | None = None
    normals: tuple[tuple[float, float, float], ...] | None = None
    max_depth: float = 0.005
    use_candidates: bool = False
    """Enable conservative geometric culling for workloads that benefit from it."""

    def get_data_types(self) -> list[str]:
        """Return published layout-aligned fields."""
        return ["depth", "hit", "shape_id"]


class TactileIndentation(_ReadOnlySensor):
    """Measure geometric overlap against closed, outward-wound target meshes.

    Two opposite ray queries from each undeformed surface sample must both
    exit the same mesh. The outward exit distance is clipped to the sensing
    interval `[0, max_depth]`. Across targets, retain the deepest result; ties
    favor the first mesh. No hit gives zero depth and shape index -1.
    """

    def __init__(
        self, config: TactileIndentationCfg, *, context: SensorContext
    ) -> None:
        if config.sample_phase != "physics" or config.mount is None:
            raise ValueError(
                "TactileIndentation requires a physics sample and a mount."
            )
        if not config.target_uids or len(set(config.target_uids)) != len(
            config.target_uids
        ):
            raise ValueError("TactileIndentation requires unique closed target UIDs.")
        if len(config.resolution) != 2 or any(
            type(n) is not int or n <= 0 for n in config.resolution
        ):
            raise ValueError(
                "Indentation resolution must contain positive width and height."
            )
        if not math.isfinite(config.max_depth) or config.max_depth <= 0:
            raise ValueError("max_depth must be finite and positive.")
        self._scene: RayScene | None = None
        self._query: _IndentationQuery | None = None
        super().__init__(config, context=context)

    def _layout(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        width, height = cfg.resolution
        if cfg.points is not None or cfg.normals is not None:
            if (
                cfg.points is None
                or cfg.normals is None
                or cfg.curvature_radius is not None
            ):
                raise ValueError(
                    "Supply both points and normals, or a generated layout."
                )
            points = torch.tensor(cfg.points, dtype=torch.float32, device=self.device)
            normals = torch.tensor(cfg.normals, dtype=torch.float32, device=self.device)
            if points.shape != (width * height, 3) or normals.shape != points.shape:
                raise ValueError("Expected one point and normal per grid cell.")
            lengths = normals.norm(dim=-1, keepdim=True)
            if not bool(
                torch.isfinite(points).all()
                & torch.isfinite(lengths).all()
                & (lengths > 0).all()
            ):
                raise ValueError("Probe geometry must be finite and normals nonzero.")
            return points, normals / lengths

        if len(cfg.size) != 2 or any(not math.isfinite(v) or v <= 0 for v in cfg.size):
            raise ValueError("Indentation size must contain two positive lengths.")
        axes = [
            ((torch.arange(n, device=self.device, dtype=torch.float32) + 0.5) / n - 0.5)
            * s
            for s, n in zip(cfg.size, (width, height), strict=True)
        ]
        y, x = torch.meshgrid(axes[1], axes[0], indexing="ij")
        x, y = x.flatten(), y.flatten()
        if cfg.curvature_radius is None:
            zero = torch.zeros_like(x)
            return torch.stack((x, y, zero), dim=-1), torch.stack(
                (zero, zero, zero + 1), dim=-1
            )
        radius = cfg.curvature_radius
        if not math.isfinite(radius) or radius <= math.hypot(*cfg.size) * 0.5:
            raise ValueError("curvature_radius must exceed the pad half-diagonal.")
        z = (radius * radius - x.square() - y.square()).sqrt()
        return (
            torch.stack((x, y, z - radius), dim=-1),
            torch.stack((x, y, z), dim=-1) / radius,
        )

    def on_bind(self) -> None:
        factory = self.context.ray_factory
        if factory is None:
            raise NotImplementedError(
                "TactileIndentation requires a ray scene provider."
            )
        scene = factory(tuple(self.cfg.target_uids), self.cfg.geometry_source)
        if not scene._meshes:
            raise ValueError("Indentation targets contain no triangle geometry.")
        points, normals = self._layout()
        width, height = self.cfg.resolution
        shape = (self.num_instances, height, width)
        data = self._data_buffer
        data["depth"] = torch.zeros(shape, device=self.device)
        data["shape_id"] = torch.full(shape, -1, dtype=torch.int32, device=self.device)
        data["hit"] = torch.zeros(shape, dtype=torch.bool, device=self.device)
        query = _IndentationQuery(
            points,
            normals,
            data["depth"].flatten(1),
            data["shape_id"].flatten(1),
            self.cfg.max_depth,
        )
        query.bind(tuple(mesh for mesh, _ in scene._meshes))
        self._scene = scene
        self._query = query

    def on_unbind(self) -> None:
        self._scene = None
        self._query = None

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        """Invalidate selected depth rows and restore the miss sentinel."""
        super().reset(env_ids)
        rows = (
            slice(None)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self._data_buffer["shape_id"][rows] = -1

    def update(self, **kwargs: object) -> None:
        """Publish the bounded outward depth and validity for each probe."""
        if self._scene is None or self._query is None:
            raise RuntimeError("TactileIndentation must be bound before sampling.")
        meshes = self._scene._posed_meshes()
        self._query.cast(
            self.get_mount_pose(),
            [pose for _, pose in meshes],
            use_candidates=self.cfg.use_candidates,
        )
        torch.ge(self._data_buffer["shape_id"], 0, out=self._data_buffer["hit"])

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Describe depth units, local direction and geometry provenance."""
        fields = super().get_data_descriptors()
        fields["depth"].update(
            unit="m", frame="surface_normal", source=self.cfg.geometry_source
        )
        return fields
