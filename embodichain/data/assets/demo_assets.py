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

"""Registered downloadable asset bundles for standalone demos."""

from __future__ import annotations

import hashlib
import json
import open3d as o3d
import os
from pathlib import Path
from typing import ClassVar

from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

demo_assets = "demo"

__all__ = [
    "CoordinatedPlacementAndPickment",
    "DeformableDemoData",
    "MultiW1Data",
    "RubiksCube",
    "ScoopIceNewEnv",
    "TabletopFourBases",
    "Real2SimCarton",
]


class ScoopIceNewEnv(EmbodiChainDataset):
    """Downloadable meshes and robot assets for the scoop-ice demo."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the scoop-ice asset bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, demo_assets, "ScoopIceNewEnv.zip"
            ),
            "e92734a9de0f64be33a11fbda0fbd3b6",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class MultiW1Data(EmbodiChainDataset):
    """Downloadable scene assets for multi-W1 manipulation demos."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the multi-W1 demo asset bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, demo_assets, "multi_w1_demo.zip"),
            "984e8fa3aa05cb36a1fd973a475183ed",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class DeformableDemoData(EmbodiChainDataset):
    """Shared cloth-twist and W1 T-shirt-folding demo assets."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the downloadable deformable-demo bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX,
                demo_assets,
                "deformable_demo_assets.zip",
            ),
            "cdb1d1b105f0e96f46945052296da4d3",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class CoordinatedPlacementAndPickment(EmbodiChainDataset):
    """Downloadable meshes for coordinated placement and pickment tutorials."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the coordinated manipulation asset bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX,
                demo_assets,
                "coordinated_placement_and_pickment.zip",
            ),
            "297c10b386a4d7a8ccb68926d69425e9",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class RubiksCube(EmbodiChainDataset):
    """Downloadable articulated Rubik's-cube asset for manipulation demos."""

    def __init__(self, data_root: str | None = None) -> None:
        """Initialize the Rubik's-cube asset bundle.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, demo_assets, "RubiksCube.zip"),
            "2f77b313cd4d14dd41cc29d740e8fdb6",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(prefix, data_descriptor, path)


class TabletopFourBases(EmbodiChainDataset):
    """Portable tabletop scene for grouped and intra-group layout demos.

    The bundle contains ``scene_export/scene_config.json``, the explicit support
    graph and twelve GLB assets. Layouts, preparation caches and previews are
    generated in a caller-owned workspace rather than in this asset cache.
    """

    def __init__(self, data_root: str | None = None) -> None:
        """Download and extract the four-support tabletop scene.

        Args:
            data_root: Optional cache root overriding the EmbodiChain default.
        """
        filename = "demo/tabletop_four_bases.zip"
        hub_prefix = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
        )
        urls = [f"{EMBODICHAIN_DOWNLOAD_PREFIX}{filename}"]
        if EMBODICHAIN_DOWNLOAD_PREFIX != hub_prefix:
            urls.append(f"{hub_prefix}{filename}")
        descriptor = o3d.data.DataDescriptor(urls, "103d891625d1f9f90e1393d048ea87c2")
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(type(self).__name__, descriptor, root)


class Real2SimCarton(EmbodiChainDataset):
    """Pinned real and synthetic inputs for the directional-carton demo.

    ``raw/`` contains one unchanged DROID episode and three original videos;
    ``annotations/``, ``inputs/`` and ``evidence/`` retain manual labels, frozen
    motion prototypes and typed state/command evidence. ``scene/`` contains
    metre-scale estimated cuboid proxies. The bundle creates no collection
    identity and includes no generated training episodes. DROID files and
    derivatives retain CC BY 4.0; independently synthetic data is Apache-2.0.

    Args:
        data_root: Optional root for the standard locked download cache.

    Raises:
        ValueError: The pinned archive, manifest or an extracted file differs.
    """

    cache_prefix: ClassVar[str] = "Real2SimCartonV1"
    """Immutable revision-specific cache entry used by the download CLI."""
    _archive_name: ClassVar[str] = "real2sim_carton_v1.zip"
    _archive_md5: ClassVar[str] = "cef38008a3bae2d17a422bf2698cb4fb"
    _archive_sha256: ClassVar[str] = (
        "169fff766e6c9e4af1ef3b890a52203867e75fd60f20c4ee5dedb0f6fcf6a11a"
    )
    _manifest_sha256: ClassVar[str] = (
        "f29a4a60f14ba450a251d2ddd3642bfc73e2e3e383a211bea3f0ac3fa3084d06"
    )

    def __init__(self, data_root: str | None = None) -> None:
        filename = "demo/real2sim_carton/" + self._archive_name
        hub_prefix = (
            "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
        )
        urls = [EMBODICHAIN_DOWNLOAD_PREFIX + filename]
        if EMBODICHAIN_DOWNLOAD_PREFIX != hub_prefix:
            urls.append(hub_prefix + filename)
        descriptor = o3d.data.DataDescriptor(urls, self._archive_md5)
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__(self.cache_prefix, descriptor, root)
        if (
            self._digest(Path(self.download_dir) / self._archive_name)
            != self._archive_sha256
        ):
            raise ValueError("Real2SimCarton archive differs from its pinned SHA-256")
        self._verify_files(Path(self.extract_dir))

    @staticmethod
    def _digest(path: Path) -> str:
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        return checksum.hexdigest()

    @classmethod
    def _verify_files(cls, root: Path) -> None:
        root = root.resolve()
        manifest = root / "manifest.json"
        if not manifest.is_file() or cls._digest(manifest) != cls._manifest_sha256:
            raise ValueError("Real2SimCarton resource manifest is missing or changed")
        for entry in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
            path = (root / entry["path"]).resolve()
            if (
                root not in path.parents
                or not path.is_file()
                or path.stat().st_size != entry["size"]
                or cls._digest(path) != entry["sha256"]
            ):
                raise ValueError(
                    f"Real2SimCarton resource is missing or changed: {entry['path']}"
                )
