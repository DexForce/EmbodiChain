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

"""Source USD asset addressing, independent of runtime IDs and placement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING

from embodichain.utils import configclass

__all__ = ["UsdSourceCfg"]


@configclass
class UsdSourceCfg:
    """Select a typed asset from a source USD stage.

    Supply this as ``source`` on a visual-object, rigid-object or articulation
    configuration. Runtime identity and placement belong to the containing
    configuration. This source does not bind an already imported scene entity.
    """

    file_path: str = MISSING
    """Source USD, USDA, USDC or USDZ file."""

    source_prim_path: str | None = None
    """Exact absolute prim path in the source stage.

    Omit only when the source contains a single candidate of the requested
    asset type. This is not a destination path, regex or parent/link binding.
    """

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> UsdSourceCfg:
        """Decode a source mapping, rejecting unknown source options.

        Args:
            data: Source file and optional exact source prim selector.

        Returns:
            Source configuration; call ``validate()`` after assembly.
        """
        unknown = set(data) - {"file_path", "source_prim_path"}
        if unknown:
            raise ValueError(f"Unknown UsdSourceCfg fields: {sorted(unknown)}.")
        return cls(**data)


def _usd_source_from_value(value: object) -> UsdSourceCfg | None:
    if value is None or isinstance(value, UsdSourceCfg):
        return value
    if isinstance(value, Mapping):
        return UsdSourceCfg.from_dict(value)
    raise TypeError("source must be a UsdSourceCfg, a source mapping, or None.")


def _raise_removed_source_fields(data: Mapping[str, object]) -> None:
    if "prim_path" in data or "source_prim_path" in data:
        raise ValueError(
            "Use source=UsdSourceCfg(file_path=..., source_prim_path=...) "
            "instead of a top-level prim selector."
        )


def _asset_file_path(cfg: object) -> str | None:
    """Resolve one file owner and validate an explicit USD source."""
    source = getattr(cfg, "source", None)
    shape = getattr(cfg, "shape", None)
    legacy_path = (
        getattr(shape, "fpath", None)
        if hasattr(cfg, "shape")
        else getattr(cfg, "fpath", None)
    )
    if source is None:
        if hasattr(cfg, "shape") and shape is None:
            raise ValueError("Mesh objects require either shape or source.")
        return legacy_path
    if not isinstance(source, UsdSourceCfg):
        raise TypeError("source must be a UsdSourceCfg.")
    if shape is not None or legacy_path is not None:
        raise ValueError("source cannot be combined with shape or fpath.")
    source.validate()
    path = source.file_path
    if not isinstance(path, str) or not path.strip():
        raise ValueError("UsdSourceCfg.file_path must be a non-empty string.")
    if not path.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
        raise ValueError("UsdSourceCfg.file_path requires a USD source file.")
    selector = source.source_prim_path
    if selector is not None:
        from pxr import Sdf

        if not isinstance(selector, str) or not Sdf.Path.IsValidPathString(selector):
            raise ValueError("source_prim_path must be an absolute USD prim path.")
        prim_path = Sdf.Path(selector)
        if (
            not prim_path.IsAbsolutePath()
            or not prim_path.IsPrimPath()
            or prim_path.ContainsPrimVariantSelection()
        ):
            raise ValueError("source_prim_path must be an absolute USD prim path.")
    return path


def _source_prim_path(cfg: object) -> str | None:
    source = getattr(cfg, "source", None)
    return None if source is None else source.source_prim_path
