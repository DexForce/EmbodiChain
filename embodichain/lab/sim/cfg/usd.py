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

"""Configuration for a complete USD import into a simulation manager."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any

import numpy as np

from embodichain.utils import configclass
from .source import UsdSourceCfg

if TYPE_CHECKING:
    from .articulation import ArticulationCfg
    from .deformable import DeformableObjectCfg
    from .rigid_object import RigidObjectCfg
    from .robot import RobotCfg
    from .visual_object import VisualObjectCfg

__all__ = ["UsdImportCfg"]


@configclass
class UsdImportCfg:
    """Describe one source selection, placement and set of asset overrides.

    Source prim paths identify override targets and returned assets. Runtime
    names are assigned independently through ``namespace`` and per-asset UIDs.
    Replication, backend and device are owned by the simulation manager.
    """

    name: str = MISSING
    """Import label, also used to disambiguate conflicting visual names."""

    source: UsdSourceCfg = MISSING
    """USD source file, optional subtree and variant selections."""

    namespace: str | None = None
    """Prefix passed to DexSim for imported entity and material names."""

    pose: np.ndarray | None = None
    """Finite 4x4 scene transform in meters/Z-up, left-multiplied onto the source."""

    asset_cfgs: dict[
        str,
        RigidObjectCfg
        | ArticulationCfg
        | DeformableObjectCfg
        | VisualObjectCfg
        | dict[str, Any],
    ] = field(default_factory=dict)
    """Sparse overrides or typed asset configurations keyed by source prim path."""

    robot_cfgs: dict[str, RobotCfg] = field(default_factory=dict)
    """Robot metadata bindings; source physics and placement are preserved."""

    def __post_init__(self) -> None:
        if isinstance(self.source, Mapping):
            self.source = UsdSourceCfg.from_dict(self.source)
        if isinstance(self.robot_cfgs, Mapping):
            from .robot import RobotCfg

            self.robot_cfgs = {
                path: RobotCfg.from_dict(value) if isinstance(value, Mapping) else value
                for path, value in self.robot_cfgs.items()
            }
        if isinstance(self.asset_cfgs, Mapping):
            normalized_assets = dict(self.asset_cfgs)
            for path, value in self.asset_cfgs.items():
                if isinstance(value, Mapping) and "class_type" in value:
                    if set(value) != {"class_type", "config"}:
                        raise ValueError(
                            "Serialized typed USD overrides require class_type and config."
                        )
                    config_type = _asset_config_types().get(value["class_type"])
                    if config_type is None:
                        raise ValueError(
                            f"Unsupported USD override class_type {value['class_type']!r}."
                        )
                    decoded = config_type.from_dict(value["config"])
                    if (
                        "init_local_pose" in value["config"]
                        and value["config"]["init_local_pose"] is None
                    ):
                        decoded.init_local_pose = None
                    normalized_assets[path] = decoded
            self.asset_cfgs = normalized_assets

    def to_dict(self) -> dict[str, Any]:
        """Serialize the recipe while retaining typed override semantics.

        Returns:
            Recipe fields with tagged typed overrides and unchanged sparse mappings.
        """
        from embodichain.utils.configclass import class_to_dict

        data = class_to_dict(self)
        data["robot_cfgs"] = {
            path: value.to_dict() for path, value in self.robot_cfgs.items()
        }
        types = _asset_config_types()
        data["asset_cfgs"] = {}
        for path, value in self.asset_cfgs.items():
            if isinstance(value, Mapping):
                data["asset_cfgs"][path] = class_to_dict(dict(value))
            elif (
                type(value).__name__ in types
                and type(value) is types[type(value).__name__]
            ):
                data["asset_cfgs"][path] = {
                    "class_type": type(value).__name__,
                    "config": value.to_dict(),
                }
            else:
                raise TypeError(
                    f"Cannot serialize USD override type {type(value).__name__}."
                )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UsdImportCfg:
        """Decode a portable import recipe without creating native resources.

        Args:
            data: Import fields, including a source mapping and optional robot metadata.

        Returns:
            Import configuration; ``validate_import()`` checks the assembled recipe.
        """
        unknown = set(data) - {
            "name",
            "source",
            "namespace",
            "pose",
            "asset_cfgs",
            "robot_cfgs",
        }
        if unknown:
            raise ValueError(f"Unknown UsdImportCfg fields: {sorted(unknown)}.")
        values = dict(data)
        return cls(**values)

    def validate_import(self) -> None:
        """Validate the assembled recipe before reading or declaring its source."""
        from .source import _validate_prim_path

        self.validate()
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("USD import name must be a non-empty string.")
        if not isinstance(self.source, UsdSourceCfg):
            raise TypeError("USD import source must be a UsdSourceCfg.")
        self.source.validate_source()
        if self.source.source_prim_path is not None:
            raise ValueError(
                "Scene imports select a subtree with source.root_prim_path; source_prim_path is an exact singleton selector."
            )
        if self.source.namespace is not None or self.source.pose_mode != "auto":
            raise ValueError(
                "Scene namespace and placement belong to UsdImportCfg, not its source."
            )
        if self.namespace is not None and (
            not isinstance(self.namespace, str)
            or not self.namespace
            or "/" in self.namespace
        ):
            raise ValueError("USD namespace must be non-empty and cannot contain '/'.")
        if self.pose is not None:
            pose = np.asarray(self.pose, dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("USD scene pose must be a finite (4, 4) transform.")
            if not np.allclose(pose[3], (0, 0, 0, 1)):
                raise ValueError("USD scene pose must be an affine transform.")
            rotation = pose[:3, :3]
            if not np.allclose(
                rotation.T @ rotation, np.eye(3), atol=1e-5
            ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
                raise ValueError(
                    "USD scene pose must contain a rigid rotation without scale."
                )
        for label, mapping in (
            ("asset_cfgs", self.asset_cfgs),
            ("robot_cfgs", self.robot_cfgs),
        ):
            if not isinstance(mapping, Mapping):
                raise TypeError(
                    f"{label} must map source prim paths to configurations."
                )
            for path in mapping:
                _validate_prim_path(
                    path, allow_root=False, field_name=f"{label} target"
                )
        overlap = self.robot_cfgs.keys() & self.asset_cfgs.keys()
        if overlap:
            raise ValueError(f"Duplicate USD configuration targets: {sorted(overlap)}.")


def _asset_config_types() -> dict[str, type]:
    from .articulation import ArticulationCfg
    from .deformable import VolumeDeformableObjectCfg, SurfaceDeformableObjectCfg
    from .rigid_object import RigidObjectCfg
    from .robot import RobotCfg
    from .visual_object import VisualObjectCfg

    return {
        config_type.__name__: config_type
        for config_type in (
            ArticulationCfg,
            RigidObjectCfg,
            RobotCfg,
            VisualObjectCfg,
            VolumeDeformableObjectCfg,
            SurfaceDeformableObjectCfg,
        )
    }


def _normalize_usd_import(
    cfg: UsdImportCfg | str | None,
    *,
    file_path: str | None,
    name: str | None,
    pose: np.ndarray | None,
    robot_cfgs: dict | None,
    usd: UsdSourceCfg | None,
    asset_cfgs: dict | None,
) -> UsdImportCfg:
    from copy import deepcopy
    import warnings

    if isinstance(cfg, UsdImportCfg):
        if any(
            value is not None
            for value in (file_path, name, pose, robot_cfgs, usd, asset_cfgs)
        ):
            raise ValueError(
                "UsdImportCfg cannot be combined with legacy add_usd arguments."
            )
        return deepcopy(cfg)
    if cfg is not None and not isinstance(cfg, str):
        raise TypeError("add_usd() requires a UsdImportCfg.")
    if cfg is not None and name is not None:
        raise ValueError("Specify the legacy USD import name once.")
    label = cfg if cfg is not None else name
    if label is None or file_path is None:
        raise TypeError(
            "add_usd() requires UsdImportCfg(name=..., source=UsdSourceCfg(file_path=...))."
        )
    warnings.warn(
        "add_usd(name, file_path, ...) is deprecated; pass UsdImportCfg instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    source = deepcopy(usd) if usd is not None else UsdSourceCfg(file_path=file_path)
    if not isinstance(source, UsdSourceCfg):
        raise TypeError("usd must be a UsdSourceCfg.")
    if isinstance(source.file_path, str) and source.file_path != file_path:
        raise ValueError("add_usd() file_path and usd.file_path must match.")
    source.file_path = file_path
    namespace = source.namespace
    source.namespace = None
    return UsdImportCfg(
        name=label,
        source=source,
        namespace=namespace,
        pose=pose,
        robot_cfgs=robot_cfgs or {},
        asset_cfgs=asset_cfgs or {},
    )
