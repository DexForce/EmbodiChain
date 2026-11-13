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
from typing import Any, Literal

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

    root_prim_path: str | None = None
    """Absolute subtree to parse; omission uses the source default prim/stage."""

    prim_path: str | None = None
    """Compatibility spelling of :attr:`root_prim_path`."""

    variants: dict[str, dict[str, str]] | None = None
    """Variant selections keyed by absolute source prim path."""

    namespace: str | None = None
    """Legacy singleton namespace; scene namespace belongs to UsdImportCfg."""

    pose_mode: Literal["auto", "config", "preserve", "offset"] = "auto"
    """Legacy singleton placement; prefer the containing asset's source_pose_mode."""

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> UsdSourceCfg:
        """Decode a source mapping, rejecting unknown source options.

        Args:
            data: Source file and optional exact source prim selector.

        Returns:
            Source configuration; call ``validate()`` after assembly.
        """
        unknown = set(data) - {
            "file_path",
            "source_prim_path",
            "root_prim_path",
            "prim_path",
            "variants",
            "namespace",
            "pose_mode",
        }
        if unknown:
            raise ValueError(f"Unknown UsdSourceCfg fields: {sorted(unknown)}.")
        return cls(**data)

    def parse_kwargs(self) -> dict[str, Any]:
        """Return source selection options accepted by DexSim.

        Returns:
            Configured parser options.
        """
        if self.pose_mode not in {"auto", "config", "preserve", "offset"}:
            raise ValueError(f"Unsupported USD pose_mode {self.pose_mode!r}.")
        if self.root_prim_path is not None and self.prim_path is not None:
            raise ValueError("Specify root_prim_path or legacy prim_path, not both.")
        root = (
            self.root_prim_path if self.root_prim_path is not None else self.prim_path
        )
        if root is not None:
            _validate_prim_path(root, allow_root=True, field_name="root_prim_path")
        if self.namespace is not None and (
            not isinstance(self.namespace, str)
            or not self.namespace
            or "/" in self.namespace
        ):
            raise ValueError("USD namespace must be non-empty and cannot contain '/'.")
        if self.variants is not None:
            if not isinstance(self.variants, Mapping):
                raise ValueError(
                    "USD variants require absolute prim paths and mappings."
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
        options = {
            name: getattr(self, name)
            for name in ("variants", "namespace")
            if getattr(self, name) is not None
        }
        if root is not None:
            options["prim_path"] = root
        return options

    def validate_source(self) -> None:
        """Validate one fully assembled source without opening its stage."""
        self.validate()
        path = self.file_path
        if not isinstance(path, str) or not path.strip():
            raise ValueError("UsdSourceCfg.file_path must be a non-empty string.")
        if not path.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
            raise ValueError("UsdSourceCfg.file_path requires a USD source file.")
        if self.source_prim_path is not None:
            _validate_prim_path(
                self.source_prim_path, allow_root=False, field_name="source_prim_path"
            )
        self.parse_kwargs()


def _validate_prim_path(value: object, *, allow_root: bool, field_name: str) -> None:
    from pxr import Sdf

    if not isinstance(value, str) or not Sdf.Path.IsValidPathString(value):
        raise ValueError(f"{field_name} must be an absolute USD prim path.")
    path = Sdf.Path(value)
    if (
        not path.IsAbsolutePath()
        or path.ContainsPrimVariantSelection()
        or not (path.IsPrimPath() or (allow_root and path == Sdf.Path.absoluteRootPath))
    ):
        raise ValueError(f"{field_name} must be an absolute USD prim path.")


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
        if (
            hasattr(cfg, "shape")
            and shape is None
            and getattr(cfg, "visuals", None) is None
        ):
            raise ValueError("Mesh objects require either shape or source.")
        return legacy_path
    if not isinstance(source, UsdSourceCfg):
        raise TypeError("source must be a UsdSourceCfg.")
    if shape is not None or legacy_path is not None:
        raise ValueError("source cannot be combined with shape or fpath.")
    source.validate_source()
    path = source.file_path
    return path


def _source_prim_path(cfg: object) -> str | None:
    source = getattr(cfg, "source", None)
    return None if source is None else source.source_prim_path
