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

"""Asset source and cache identity for Franka."""

from __future__ import annotations

import os
import hashlib
import json
import math
from pathlib import Path
import tempfile
from xml.etree import ElementTree
import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset, _dataset_download_lock
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)

__all__ = ["Franka", "FrankaPandaArm"]

robot_assets = "robot_assets"


class Franka(EmbodiChainDataset):
    """Dataset class for the Franka robots.

    Reference:
        https://franka.de/franka-research-3

    Directory structure:
        Franka/
            Panda/
                Panda.urdf
                PandaWithHand.urdf
            FR3/
                fr3.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Franka
        >>> dataset = Franka()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Franka/Panda/Panda.urdf"))
        >>> print(get_data_path("Franka/PandaWithHand/PandaWithHand.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "FrankaV4.zip"),
            "0c613d884798e8b084604381641cc8e1",
        )
        prefix = type(self).__name__
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class FrankaPandaArm(EmbodiChainDataset):
    """Pinned original Panda arm with Panda limits, distinct from FR3 assets.

    The original archive preserves ``Joint1`` through ``Joint7`` and the
    ``ee_link`` flange. Existing Franka asset IDs keep their current defaults.

    Args:
        data_root: Optional ordinary EmbodiChain asset cache root.
    """

    def __init__(self, data_root: str | None = None) -> None:
        descriptor = o3d.data.DataDescriptor(
            [
                os.path.join(
                    EMBODICHAIN_DOWNLOAD_PREFIX, "robot_assets", "FrankaV2.zip"
                ),
                "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/robot_assets/FrankaV2.zip",
            ],
            "f0675b9da98126bc3d4e18c98ef5e06c",
        )
        super().__init__(
            type(self).__name__,
            descriptor,
            EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root,
        )

        # The legacy CAD export rounded quarter-turns to 1.5708. Materialize a
        # model-owned exact-geometry variant while retaining the original file.
        root = Path(self.extract_dir)
        original = root / "Panda/Panda.urdf"
        source_bytes = original.read_bytes()
        expected_source_sha = (
            "ec88eb5b8bbbe2bd9ceaa0b1784a9ff0408fac118be6bf312904b7fbdb570b04"
        )
        if hashlib.sha256(source_bytes).hexdigest() != expected_source_sha:
            raise ValueError("pinned Panda arm source changed in the asset cache")
        parser = ElementTree.XMLParser(
            target=ElementTree.TreeBuilder(insert_comments=True)
        )
        document = ElementTree.fromstring(source_bytes, parser=parser)
        corrections = []
        for i in range(2, 8):
            joint = document.find(f"joint[@name='Joint{i}']")
            origin = joint.find("origin")
            angles = [float(value) for value in origin.get("rpy").split()]
            if abs(abs(angles[0]) - math.pi / 2) > 5e-6 or angles[1:] != [0.0, 0.0]:
                raise ValueError("unexpected Panda CAD axis convention")
            angle = math.copysign(math.pi / 2, angles[0])
            origin.set("rpy", f"{angle:.17g} 0 0")
            corrections.append(
                {
                    "joint": f"Joint{i}",
                    "original_rpy": angles,
                    "exact_rpy": [angle, 0.0, 0.0],
                }
            )
        exact = ElementTree.tostring(document, encoding="utf-8", xml_declaration=True)
        report = json.dumps(
            {
                "schema": "panda-cad-axis-precision/v1",
                "original_sha256": expected_source_sha,
                "exact_sha256": hashlib.sha256(exact).hexdigest(),
                "corrections": corrections,
            },
            sort_keys=True,
            indent=2,
        ).encode()
        cache_root = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root
        with _dataset_download_lock(cache_root, type(self).__name__):
            for name, content in (
                ("PandaExactV1.urdf", exact),
                ("PandaExactV1.normalization.json", report),
            ):
                destination = original.parent / name
                if destination.exists():
                    if destination.read_bytes() != content:
                        raise ValueError(
                            "derived Panda model differs from its pinned normalization"
                        )
                    continue
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=destination.parent, delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.link(temporary, destination)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
