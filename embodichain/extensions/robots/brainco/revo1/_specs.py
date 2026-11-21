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

"""Native Revo1 asset names, shared by standalone and mounted configurations."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = []


@dataclass(frozen=True)
class _Revo1SideSpec:
    urdf_path: str
    joint_names: tuple[str, ...]
    base_link_name: str
    reference_link_name: str


def get_side_spec(side: str) -> _Revo1SideSpec:
    """Return native asset identifiers without applying host naming or mounting."""
    if side not in ("left", "right"):
        raise ValueError(f"Revo1 side must be 'left' or 'right', got {side!r}")
    title = side.title()
    return _Revo1SideSpec(
        urdf_path=f"BrainCoHandRevo1/BrainCo{title}Hand/BrainCo{title}Hand.urdf",
        joint_names=tuple(
            f"{side.upper()}_HAND_{finger}"
            for finger in ("THUMB1", "THUMB2", "INDEX", "MIDDLE", "RING", "PINKY")
        ),
        base_link_name=f"{side}_hand_base",
        reference_link_name=f"{side}_thumb_dist",
    )
