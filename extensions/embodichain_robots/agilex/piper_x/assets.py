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

"""Pinned source URDF and meshes for the AgileX PiPER-X arm."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .._arm_assets import _AgilexArmAsset, _REVISION

__all__ = ["AgilexPiperX"]

_ARM_URDF = "piper_x/urdf/piper_x_description.urdf"
_MOUNT_XACRO = "piper_x/urdf/piper_x_with_gripper_description.xacro"
_SOURCE_FILES = {
    _ARM_URDF: "93eb2a84d01df81566a3c1ed275626d93a1c4275564a30c7899a35a672e265bb",
    _MOUNT_XACRO: "88561fe45a3f4e639c7a233b13574d88d7707b545110cb30b983e360cdb57420",
    "piper_x/meshes/base_link.stl": "b732cf8a1f8435f51e3ec860f11189c3ec63744d3df008b7d3e9b6505dfc483d",
    "piper_x/meshes/dae/base_link.dae": "706008b1e425227d42fb490febb6a54a2a294a39911b8cf4e9ca4b8c61eb42bd",
    "piper_x/meshes/dae/flange.dae": "421c3e20c6109df3c96c5ce18ff0b2609b6604135ff790e4bfc1d9a281b0701a",
    "piper_x/meshes/dae/link1.dae": "6448c874d7106738d20cdd044e0e88f86262935f0bc3bdc072c5025bcf47ecfd",
    "piper_x/meshes/dae/link2.dae": "05ad1975346a330129dbce02131405dba6f51d74d0e5f3a70ec154ef219d9163",
    "piper_x/meshes/dae/link3.dae": "001b0b97dd514a9f8e6b321e12a8005d17198eb8b60ec443b9bd67251a3790ce",
    "piper_x/meshes/dae/link4.dae": "cda6aab6dfefb13d5409a691d7486c4253737f114cca9ada67da36a10fb08123",
    "piper_x/meshes/dae/link5.dae": "8c2a2da46dfe9d13fc8841a990694c77106c27be292fac4fc42a86cd78a2dee1",
    "piper_x/meshes/dae/link6.dae": "7e6076e2574f042f5067dd431e96486249cffbc8ab292fb66cb68848ad5bab9a",
    "piper_x/meshes/flange.stl": "f9ce752f93a6598858a779e3d8755a09e466b588efcffc3a1c3953773753a22b",
    "piper_x/meshes/link1.stl": "60c53c4d5d20c5b663cc4ced2f5e322d2a857baa6e79cef04a33b5f9f7c1e985",
    "piper_x/meshes/link2.stl": "0316b881cd0bae6c9f8f794fabdee7fbf041f83f2f75980d34b67555fe8a3561",
    "piper_x/meshes/link3.stl": "c51994fa4ca6a0577cfc65fe5a666ad47ad6af8dc98fbe81b71c4a7fe8ad09a5",
    "piper_x/meshes/link4.stl": "10b69bde5305c53fbb4a6dc4e910e8604d54d250a969e6cda8128d8244989831",
    "piper_x/meshes/link5.stl": "fb7ab27270e6b5fa0b2e6df9fa37722062f4ff397cd4f89883d23551bedd0801",
    "piper_x/meshes/link6.stl": "8c391520fd963fe4f9b46afcde82ceeab2a93b73c01fc5debe039fe632f597e6",
}


class AgilexPiperX(_AgilexArmAsset):
    """Resolve the bare arm and its manufacturer-authored mounting flange.

    Args:
        data_root: Optional override for the EmbodiChain asset cache root.
    """

    cache_prefix = f"AgilexPiperX_{_REVISION[:12]}_v1"

    def __init__(self, data_root: str | None = None) -> None:
        super().__init__(
            data_root,
            source_urdf=_ARM_URDF,
            urdf_md5="b541ce064d50fffb23dec4f0584d757c",
            source_files=_SOURCE_FILES,
            output_urdf="piper_x.urdf",
        )

    def _portable_urdf(self, extracted: Path, source_urdf: str) -> bytes:
        arm = ET.parse(extracted / source_urdf).getroot()
        # The simulator owns the world. The component must root at base_link.
        arm.remove(arm.find("link[@name='world']"))
        arm.remove(arm.find("joint[@name='world_to_base_link']"))
        # AgileX declares the physical flange in the tool Xacro. Copy its
        # fixed link and joint so both configurations use the same interface.
        mount = ET.parse(extracted / _MOUNT_XACRO).getroot()
        arm.append(mount.find("link[@name='flange_link']"))
        arm.append(mount.find("joint[@name='flange_joint']"))
        ET.indent(arm)
        return ET.tostring(arm, encoding="utf-8", xml_declaration=True)
