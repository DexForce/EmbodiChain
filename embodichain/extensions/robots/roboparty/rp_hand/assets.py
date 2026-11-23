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

"""Standalone hand URDFs derived from the pinned articulated RP1 model."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from embodichain.data.dataset import _dataset_download_lock

from .._assets import _publish_urdf
from ..rp1.assets import RoboPartyRP1

__all__ = ["RoboPartyRPHand"]


class RoboPartyRPHand(RoboPartyRP1):
    """Resolve left and right RP Hand URDFs in the shared RP1 archive cache."""

    def __init__(self, data_root: str | None = None) -> None:
        """Prepare the hands with the same cache root as the RP1 source.

        Args:
            data_root: Optional cache root. Defaults to the configured data root.
        """
        super().__init__(data_root)
        with _dataset_download_lock(self.data_root, self.cache_prefix):
            directory = Path(self.extract_dir)
            source = ET.parse(directory / "rp1_hand.urdf")
            for side, prefix in (("left", "lh_"), ("right", "rh_")):
                hand = ET.Element("robot", {"name": f"rp_hand_{side}"})
                hand.extend(
                    link
                    for link in source.findall("link")
                    if link.attrib["name"].startswith(prefix)
                )
                # The incoming wrist mount has a parent outside this hand.
                hand.extend(
                    joint
                    for joint in source.findall("joint")
                    if joint.find("parent").attrib["link"].startswith(prefix)
                )
                _publish_urdf(
                    directory / f"rp_hand_{side}.urdf",
                    ET.tostring(hand, encoding="utf-8", xml_declaration=True),
                )
