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
"""Pinned Franka narrow jaws and RJ45 fixture for the insertion-cycle scene."""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

from embodichain.data import EmbodiChainDataset, get_data_path
from embodichain.data.assets.task_assets import Rj45Fixture
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

from .assets import _FR3_WITH_HAND_SHA256, _franka_v4_archive, _write_derivatives
from .rope_pads import apply_rope_pads

__all__ = ["FrankaRj45", "Rj45Fixture", "get_rj45_models"]

_ROBOT_MODEL = "Panda/Fr3Rj45V1.urdf"
_FIXTURE_ROOT = "dexuni_cable_rj45_assets_v1/assets/rj45"


class FrankaRj45(EmbodiChainDataset):
    """Derive the narrow jaws from the pinned FR3 model while retaining its hand.

    Args:
        data_root: Optional root of the shared EmbodiChain asset cache.
    """

    def __init__(self, data_root: str | None = None) -> None:
        root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        super().__init__("Franka", _franka_v4_archive(), root)
        source = Path(self.extract_dir) / "Panda/PandaWithHand.urdf"
        if _sha256(source) != _FR3_WITH_HAND_SHA256:
            raise ValueError("Pinned FR3 with hand differs from the RJ45 scene source.")
        model = ET.parse(source).getroot()
        apply_rope_pads(
            model,
            stem_size=(0.014, 0.008, 0.04),
            center=(0, 0.002, 0.041),
            size=(0.009, 0.004, 0.012),
        )
        _write_derivatives(
            source,
            root,
            {
                Path(_ROBOT_MODEL).name: ET.tostring(
                    model, encoding="utf-8", xml_declaration=True
                )
            },
        )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def get_rj45_models() -> dict[str, Path]:
    """Resolve the three shared URDFs through the EmbodiChain asset cache.

    Returns:
        Paths for the `robot`, `plug` and `socket` models, each with meshes
        relative to its containing directory.
    """
    return {
        "robot": Path(get_data_path(f"FrankaRj45/{_ROBOT_MODEL}")).resolve(),
        "plug": Path(
            get_data_path(f"Rj45Fixture/{_FIXTURE_ROOT}/rj45_plug_v1.urdf")
        ).resolve(),
        "socket": Path(
            get_data_path(f"Rj45Fixture/{_FIXTURE_ROOT}/rj45_socket_v1.urdf")
        ).resolve(),
    }
