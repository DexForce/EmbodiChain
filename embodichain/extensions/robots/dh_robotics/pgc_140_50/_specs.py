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

"""Native PGC asset identifiers, independent of the mounting robot."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = []


@dataclass(frozen=True)
class _PGCSpec:
    urdf_path: str
    joint_names: tuple[str, ...]
    base_link_name: str
    reference_link_name: str


def get_asset_spec(modified: bool = False) -> _PGCSpec:
    """Select the original archive or the separately distributed DexForce edit."""
    suffix = "_M" if modified else ""
    return _PGCSpec(
        urdf_path=f"DH_PGC_140_50{suffix}/DH_PGC_140_50{suffix}.urdf",
        joint_names=(
            ("FINGER1", "FINGER2") if modified else ("finger1_joint", "finger2_joint")
        ),
        # W1 retains its historical renamed base_link_1 in its own adapter.
        base_link_name="base_link",
        reference_link_name="finger2" if modified else "finger2_link",
    )
