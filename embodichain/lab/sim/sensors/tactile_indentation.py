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
from embodichain.compute.geometry._tactile_shear import _TactileShear
from embodichain.compute.image._tactile_rgb import _TactileRGB
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

    @configclass
    class RGBCfg:
        """First-order optical response for a regular or calibrated pad grid."""

        x_response: tuple[float, float, float] = MISSING
        """RGB gain per unit depth slope toward increasing columns."""
        y_response: tuple[float, float, float] = MISSING
        """RGB gain per unit depth slope toward increasing rows."""
        background: (
            tuple[float, float, float] | tuple[tuple[float, float, float], ...]
        ) = (
            0.5,
            0.5,
            0.5,
        )
        """Normalized no-contact RGB, either uniform or one triple per row-major cell."""

    @configclass
    class ShearCfg:
        """Per-cell penalty contact and shear with a Coulomb slip limit."""

        normal_stiffness: float = MISSING
        """Compressive stiffness per probe, in N/m."""
        shear_stiffness: float = MISSING
        """Tangential elastic stiffness per probe, in N/m."""
        friction: float = MISSING
        """Nonnegative Coulomb coefficient."""
        marker_indices: tuple[int, ...] = ()
        """Optional ordered, row-major probe indices for sparse marker readout."""

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

    rgb: RGBCfg | None = None
    """Enable calibrated optical output on the same probe grid."""

    shear: ShearCfg | None = None
    """Enable modeled contact loads and elastic displacement."""

    def get_data_types(self) -> list[str]:
        """Return published layout-aligned fields."""
        fields = ["depth", "hit", "shape_id"]
        if self.rgb is not None:
            fields.append("rgb")
        if self.shear is not None:
            fields.extend(("normal_force", "shear_force", "shear_displacement"))
            if self.shear.marker_indices:
                fields.append("marker_displacement")
        return fields


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
        self._shear: _TactileShear | None = None
        self._markers: torch.Tensor | None = None
        self._rgb: _TactileRGB | None = None
        self._rgb_background: torch.Tensor | None = None
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
        if self.cfg.rgb is not None:
            self._bind_rgb(self.cfg.rgb, points)
        if self.cfg.shear is not None:
            self._bind_shear(self.cfg.shear, points, normals, len(scene._meshes))
        self._scene = scene
        self._query = query

    def _bind_rgb(
        self, cfg: TactileIndentationCfg.RGBCfg, points: torch.Tensor
    ) -> None:
        cfg.validate()
        if any(
            len(gain) != 3 or not all(math.isfinite(value) for value in gain)
            for gain in (cfg.x_response, cfg.y_response)
        ):
            raise ValueError("RGB gains must contain three finite values.")
        data = self._data_buffer
        _, height, width = data["depth"].shape
        background = torch.as_tensor(
            cfg.background, dtype=torch.float32, device=self.device
        )
        if background.shape == (3,):
            background = background.expand(height, width, 3).contiguous()
        elif background.shape == (height * width, 3):
            background = background.reshape(height, width, 3)
        else:
            raise ValueError("RGB background needs one color or one per probe.")
        if not bool(((background >= 0.0) & (background <= 1.0)).all()):
            raise ValueError("RGB background must be in [0, 1].")
        rgb = background.expand(self.num_instances, -1, -1, -1).clone()
        data["rgb"] = rgb
        self._rgb = _TactileRGB(
            data["depth"],
            data["hit"],
            rgb,
            background,
            points,
            cfg.x_response,
            cfg.y_response,
        )
        self._rgb_background = background

    def _bind_shear(
        self,
        cfg: TactileIndentationCfg.ShearCfg,
        points: torch.Tensor,
        normals: torch.Tensor,
        num_meshes: int,
    ) -> None:
        cfg.validate()
        if (
            any(
                not math.isfinite(value) or value <= 0
                for value in (cfg.normal_stiffness, cfg.shear_stiffness)
            )
            or not math.isfinite(cfg.friction)
            or cfg.friction < 0
        ):
            raise ValueError(
                "Shear stiffnesses must be finite and positive, and friction nonnegative."
            )
        marker_indices = cfg.marker_indices
        if any(
            type(index) is not int or not 0 <= index < len(points)
            for index in marker_indices
        ) or len(set(marker_indices)) != len(marker_indices):
            raise ValueError("Marker indices must be unique cells of the probe grid.")

        data = self._data_buffer
        depth = data["depth"]
        data["normal_force"] = torch.zeros_like(depth)
        data["shear_force"] = depth.new_zeros((*depth.shape, 2))
        data["shear_displacement"] = torch.zeros_like(data["shear_force"])
        if marker_indices:
            self._markers = torch.as_tensor(marker_indices, device=self.device)
            data["marker_displacement"] = depth.new_zeros(
                (self.num_instances, len(marker_indices), 2)
            )
        self._shear = _TactileShear(
            points,
            normals,
            depth,
            data["shape_id"],
            data["normal_force"],
            data["shear_force"],
            data["shear_displacement"],
            num_meshes=num_meshes,
            normal_stiffness=cfg.normal_stiffness,
            shear_stiffness=cfg.shear_stiffness,
            friction=cfg.friction,
        )

    def on_unbind(self) -> None:
        self._shear = None
        self._markers = None
        self._rgb = None
        self._rgb_background = None
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
        if self._shear is not None:
            self._shear.reset(rows)
        if self._rgb_background is not None:
            self._data_buffer["rgb"][rows] = self._rgb_background

    def update(self, **kwargs: object) -> None:
        """Publish the bounded outward depth and validity for each probe."""
        if self._scene is None or self._query is None:
            raise RuntimeError("TactileIndentation must be bound before sampling.")
        poses = [pose for _, pose in self._scene._posed_meshes()]
        mount = self.get_mount_pose()
        self._query.cast(mount, poses, use_candidates=self.cfg.use_candidates)
        torch.ge(self._data_buffer["shape_id"], 0, out=self._data_buffer["hit"])
        if self._rgb is not None:
            self._rgb.render()
        if self._shear is not None:
            self._shear.sample(mount, poses)
            if self._markers is not None:
                torch.index_select(
                    self._data_buffer["shear_displacement"].flatten(1, 2),
                    1,
                    self._markers,
                    out=self._data_buffer["marker_displacement"],
                )

    def get_data_descriptors(self) -> dict[str, dict[str, object]]:
        """Describe depth units, local direction and geometry provenance."""
        fields = super().get_data_descriptors()
        fields["depth"].update(
            unit="m", frame="surface_normal", source=self.cfg.geometry_source
        )
        if "rgb" in fields:
            fields["rgb"].update(
                unit="normalized", frame="probe_grid", source="calibrated_depth_slope"
            )
        for name, unit, frame in (
            ("normal_force", "N", "surface_normal"),
            ("shear_force", "N", "surface_tangent"),
            ("shear_displacement", "m", "surface_tangent"),
            ("marker_displacement", "m", "surface_tangent"),
        ):
            if name in fields:
                fields[name].update(
                    unit=unit, frame=frame, source="indentation_penalty"
                )
        return fields
