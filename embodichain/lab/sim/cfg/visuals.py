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

"""Typed visual resources independent of physical geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
from typing import Literal
import math
import numpy as np

from embodichain.utils import configclass
from ..shapes import ShapeCfg

__all__ = ["GaussianSplatCfg", "VisualElementCfg", "GaussianRenderCfg"]


@configclass
class GaussianSplatCfg:
    """Reference captured-radiance Gaussian data; no physical geometry is inferred.

    File decoding and immutable resource residency belong to DexSim. Use
    ``UsdSourceCfg`` or ``SimulationManager.add_usd`` for a USD stage whose
    visual Xform references an external Gaussian PLY.
    """

    file_path: str = ""
    lod_path: str | None = None
    lod_enabled: bool = False
    lod_detail_scale: float = 1.0
    color_gain: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Independent captured-radiance RGB gains before native tone mapping."""
    opacity_gain: float = 1.0
    """Independent opacity gain before native alpha culling and compositing."""

    point_opacities: np.ndarray | None = None
    """Optional activated per-point opacity (N,) for an immutable flat-PLY variant."""
    sh_coefficients: np.ndarray | None = None
    """Optional basis-major SH (N,K,3) for an immutable flat-PLY variant."""

    type: Literal["gaussian_splat"] = "gaussian_splat"
    """Serialized representation tag retained by nested configuration writers."""

    def __post_init__(self) -> None:
        if self.type != "gaussian_splat":
            raise ValueError("Gaussian representation type must be 'gaussian_splat'.")
        for name in ("point_opacities", "sh_coefficients"):
            value = getattr(self, name)
            if value is None:
                continue
            if self.lod_path or Path(self.file_path).suffix.lower() != ".ply":
                raise ValueError(
                    "Point attribute overrides require flat PLY; rebuild LoD offline for a changed asset."
                )
            array = np.asarray(value, dtype=np.float32)
            if array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"{name} requires nonempty finite float32 values.")
            if name == "point_opacities":
                if array.ndim != 1 or np.any(array < 0) or np.any(array > 1):
                    raise ValueError(
                        "point_opacities requires (N,) activated values in [0,1]."
                    )
            elif (
                array.ndim != 3
                or array.shape[1] not in (1, 4, 9, 16)
                or array.shape[2] != 3
            ):
                raise ValueError("sh_coefficients requires (N,K,3), K in {1,4,9,16}.")
        gain = np.asarray(self.color_gain, dtype=np.float64)
        if (
            gain.shape != (3,)
            or not np.isfinite(gain).all()
            or np.any(gain < 0)
            or np.any(gain > np.finfo(np.float32).max)
            or not math.isfinite(self.opacity_gain)
            or self.opacity_gain < 0
            or self.opacity_gain > np.finfo(np.float32).max
        ):
            raise ValueError(
                "Gaussian appearance gains must be finite nonnegative float32 values."
            )
        self.color_gain = tuple(float(v) for v in gain)
        if not self.file_path or Path(self.file_path).suffix.lower() not in {
            ".ply",
            ".gsq",
        }:
            raise ValueError(
                "GaussianSplatCfg.file_path requires .ply or .gsq; use UsdSourceCfg or add_usd for USD visual references."
            )
        if self.lod_path and Path(self.lod_path).suffix.lower() != ".dgslod":
            raise ValueError("Gaussian LoD requires a .dgslod sidecar.")
        if Path(self.file_path).suffix.lower() == ".gsq" and self.lod_path:
            raise ValueError("Compressed .gsq resources cannot use a LoD sidecar.")
        if not math.isfinite(self.lod_detail_scale) or self.lod_detail_scale <= 0:
            raise ValueError("lod_detail_scale must be finite and positive.")
        if not self.lod_path and (self.lod_enabled or self.lod_detail_scale != 1.0):
            raise ValueError("Gaussian instance LoD settings require a sidecar.")


@configclass
class VisualElementCfg:
    """One owner-local visual, with a stable ID and no collision semantics.

    ``representation`` accepts an existing ShapeCfg or GaussianSplatCfg.
    ``local_pose`` is an SE(3) matrix in metres; ``scale`` is a positive
    uniform geometry multiplier, independent of LoD detail. Configuration
    ownership selects the root, body or named link driving this element.
    """

    id: str = "appearance"
    representation: ShapeCfg | GaussianSplatCfg | None = None
    local_pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    scale: float = 1.0
    visible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or "/" in self.id:
            raise ValueError("Visual element ID must be nonempty and contain no '/'.")
        if not isinstance(self.representation, (ShapeCfg, GaussianSplatCfg)):
            raise TypeError(
                "Visual representation requires ShapeCfg or GaussianSplatCfg."
            )
        if isinstance(self.representation, GaussianSplatCfg):
            self.representation.__post_init__()
        pose = np.asarray(self.local_pose, dtype=np.float32)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
        ):
            raise ValueError("Visual local_pose must be a finite rigid 4x4 matrix.")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Visual scale must be finite, positive and uniform.")
        if not isinstance(self.visible, bool):
            raise TypeError("Visual visible must be a bool.")
        self.local_pose = pose.copy()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> VisualElementCfg:
        """Decode a tagged Gaussian or existing shape representation.

        Args:
            data: Element fields with a tagged representation mapping or config.

        Returns:
            Validated visual element configuration.
        """
        values = dict(data)
        rep = values.get("representation")
        if isinstance(rep, Mapping):
            rep = dict(rep)
            if rep.pop("type", None) == "gaussian_splat":
                rep = GaussianSplatCfg(**rep)
            else:
                rep = ShapeCfg.from_dict(rep)
        values["representation"] = rep
        return cls(**values)


def _visual_elements(value: object) -> list[VisualElementCfg] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("visuals must be a list or None.")
    elements = [
        VisualElementCfg.from_dict(v) if isinstance(v, Mapping) else v for v in value
    ]
    for element in elements:
        if not isinstance(element, VisualElementCfg):
            raise TypeError("visuals requires VisualElementCfg entries.")
        element.__post_init__()
    if len({e.id for e in elements}) != len(elements):
        raise ValueError("Visual element IDs must be unique within their owner.")
    return elements


@configclass
class GaussianRenderCfg:
    """Native CUDA Gaussian raster settings; disabled for existing scenes.

    Budgets retain DexSim semantics: resource_attribute_budget_mb is per new
    resource and workspace_memory_budget_mb excludes resource residency.
    A smaller LoD frontier does not imply reduced resident memory.
    """

    enabled: bool = False
    enable_render_profiler: bool = False
    """Enable native render counters for residency and workspace measurements."""
    lod_mode: Literal["disabled", "dynamic_hierarchy"] = "disabled"
    max_resident_nodes: int = 0
    resource_attribute_budget_mb: int = 0
    workspace_memory_budget_mb: int = 0
    workspace_slot_count: int = 0
    lod_target_pixel_size: float = 2.0
    lod_max_nodes_per_camera: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.enable_render_profiler) is not bool
        ):
            raise TypeError("Gaussian enable flags must be booleans.")
        if self.lod_mode not in {"disabled", "dynamic_hierarchy"}:
            raise ValueError("Unsupported Gaussian lod_mode.")
        for name in (
            "max_resident_nodes",
            "resource_attribute_budget_mb",
            "workspace_memory_budget_mb",
            "workspace_slot_count",
            "lod_max_nodes_per_camera",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Gaussian {name} must be a nonnegative integer.")
        if (
            not math.isfinite(self.lod_target_pixel_size)
            or self.lod_target_pixel_size <= 0
        ):
            raise ValueError("lod_target_pixel_size must be finite and positive.")

    def apply(self, world_config: object) -> None:
        """Configure the native World before startup; require APIs only when enabled.

        Args:
            world_config: Native DexSim WorldConfig to configure before creation.
        """
        self.__post_init__()
        if not self.enabled:
            return
        import dexsim

        if not hasattr(dexsim, "GaussianSplatConfig"):
            raise RuntimeError(
                "Gaussian rendering requires a DexSim build with GaussianSplatConfig."
            )
        native = dexsim.GaussianSplatConfig()
        world_config.enable_render_profiler = self.enable_render_profiler
        native.enabled = True
        native.approach = dexsim.GaussianRenderApproach.RASTER_CUDA
        native.lod_mode = (
            dexsim.GaussianLodMode.DISABLED
            if self.lod_mode == "disabled"
            else dexsim.GaussianLodMode.DYNAMIC_HIERARCHY
        )
        for name in (
            "max_resident_nodes",
            "resource_attribute_budget_mb",
            "workspace_memory_budget_mb",
            "workspace_slot_count",
            "lod_max_nodes_per_camera",
        ):
            setattr(native, name, getattr(self, name))
        native.lod_target_pixel_size = self.lod_target_pixel_size
        world_config.gaussian_config = native
