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

"""Base asset configuration and file-backed physics policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Literal

import numpy as np

from embodichain.utils import configclass, is_configclass, logger

__all__ = ["AssetPhysicsMode", "ObjectBaseCfg", "UsdSourceCfg"]

AssetPhysicsMode = Literal["preserve", "overlay"]
"""Policy for applying EmbodiChain physics to a file-backed asset."""


def _resolve_asset_physics_mode(
    mode: AssetPhysicsMode,
) -> AssetPhysicsMode:
    """Validate and return a source-agnostic asset-physics policy."""
    if mode not in ("preserve", "overlay"):
        raise ValueError(
            f"asset_physics_mode must be 'preserve' or 'overlay', got {mode!r}."
        )
    return mode


def _is_usd_source(path: object) -> bool:
    return path is not None and str(path).lower().endswith(
        (".usd", ".usda", ".usdc", ".usdz")
    )


@configclass
class UsdSourceCfg:
    """Selection and placement intent for a USD source.

    Source lengths, mass and world axes are normalized by DexSim. Physics
    overlays remain owned by the containing rigid/articulation configuration.
    """

    prim_path: str | None = None
    """Absolute source subtree; None selects the default prim."""

    variants: dict[str, dict[str, str]] | None = None
    """Absolute prim paths mapped to variant-set names and choices."""

    namespace: str | None = None
    """Optional prefix for imported scene entity/material names."""

    pose_mode: Literal["auto", "config", "preserve", "offset"] = "auto"
    """Root placement policy, independent of the physics overlay policy.

    Auto preserves the source pose unless init_local_pose or a nonzero
    init_pos/init_rot is configured. Config replaces it, preserve retains it,
    and offset left-multiplies the configured transform onto the source pose.
    Use config or an explicit identity init_local_pose to place at the origin.
    """

    def __post_init__(self) -> None:
        if self.pose_mode not in {"auto", "config", "preserve", "offset"}:
            raise ValueError(f"Unsupported USD pose_mode {self.pose_mode!r}.")
        if self.prim_path is not None and (
            not isinstance(self.prim_path, str) or not self.prim_path.startswith("/")
        ):
            raise ValueError("USD prim_path must be an absolute path.")
        if self.namespace is not None and (
            not isinstance(self.namespace, str)
            or not self.namespace
            or "/" in self.namespace
        ):
            raise ValueError("USD namespace must be non-empty and cannot contain '/'.")
        if self.variants is not None:
            if not isinstance(self.variants, Mapping):
                raise ValueError(
                    "USD variants must map prim paths to variant selections."
                )
            for path, choices in self.variants.items():
                if (
                    not isinstance(path, str)
                    or not path.startswith("/")
                    or not isinstance(choices, Mapping)
                ):
                    raise ValueError(
                        "USD variants require absolute prim paths and mappings."
                    )
                if any(
                    not isinstance(name, str) or not isinstance(choice, str)
                    for name, choice in choices.items()
                ):
                    raise ValueError(
                        "USD variant-set names and choices must be strings."
                    )

    @classmethod
    def from_dict(cls, init_dict: Mapping[str, Any]) -> UsdSourceCfg:
        """Parse source options, rejecting misspelled fields.

        Args:
            init_dict: USD source configuration fields.

        Returns:
            Validated source configuration.
        """
        unknown = set(init_dict) - {"prim_path", "variants", "namespace", "pose_mode"}
        if unknown:
            raise ValueError(f"Unknown USD source fields: {sorted(unknown)}.")
        return cls(**dict(init_dict))

    def parse_kwargs(self) -> dict[str, Any]:
        """Return configured DexSim parser options.

        Returns:
            Non-None selection options; placement is applied by EmbodiChain.
        """
        return {
            name: getattr(self, name)
            for name in ("prim_path", "variants", "namespace")
            if getattr(self, name) is not None
        }


@configclass
class ObjectBaseCfg:
    """Base configuration for an asset in the simulation.

    This class defines the basic properties of an asset, such as its type, initial state, and collision group.
    It is used as a base class for specific asset configurations.
    """

    uid: str | None = None

    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    init_rot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Euler angles (in degree) of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""

    init_local_pose: np.ndarray | None = None
    """4x4 transformation matrix of the root in local frame. If specified, it will override init_pos and init_rot."""

    @classmethod
    def from_dict(cls, init_dict: Dict[str, str | float | tuple]) -> ObjectBaseCfg:
        """Initialize the configuration from a dictionary."""
        cfg = cls()  # Create a new instance of the class (cls)
        for key, value in init_dict.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if key == "attrs" and isinstance(value, Mapping):
                    # Keep the base module independent of rigid schemas at
                    # import time; only rigid-derived configs expose this key.
                    from .rigid import _rigid_body_physics_from_dict

                    setattr(cfg, key, _rigid_body_physics_from_dict(value))
                elif is_configclass(attr):
                    setattr(
                        cfg, key, attr.from_dict(value)
                    )  # Call from_dict on the attribute
                else:
                    setattr(cfg, key, value)
            else:
                logger.log_warning(
                    f"Key '{key}' not found in {cfg.__class__.__name__}."
                )

        # Automatically infer init_local_pose if not provided
        if cfg.init_local_pose is None:
            # If only init_pos or init_rot are provided, generate the 4x4 pose matrix
            from scipy.spatial.transform import Rotation as R

            T = np.eye(4)
            T[:3, 3] = np.array(cfg.init_pos)
            T[:3, :3] = R.from_euler("xyz", np.deg2rad(cfg.init_rot)).as_matrix()
            cfg.init_local_pose = T
        else:
            # If only init_local_pose is provided, extract init_pos and init_rot
            from scipy.spatial.transform import Rotation as R

            T = np.array(cfg.init_local_pose)
            cfg.init_pos = tuple(T[:3, 3])
            cfg.init_rot = tuple(R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True))

        return cfg
