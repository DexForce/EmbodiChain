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

"""Asset and mounting specifications for W1 external end effectors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation as R

from .types import (
    DexforceW1ArmSide,
    DexforceW1HandBrand,
    DexforceW1HandVersion,
)

__all__ = [
    "W1HandSideSpec",
    "W1HandSpec",
    "get_default_w1_hand_version",
    "get_w1_hand_spec",
]


def _attach_xpos(
    rotation_xyz_degrees: tuple[float, float, float],
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, ...], ...]:
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler(
        "xyz", rotation_xyz_degrees, degrees=True
    ).as_matrix()
    transform[:3, 3] = translation
    return tuple(tuple(float(value) for value in row) for row in transform)


@dataclass(frozen=True)
class W1HandSideSpec:
    """Side-specific asset metadata for one hand release."""

    urdf_path: str
    joint_names: tuple[str, ...]
    end_link_name: str
    root_link_name: str
    attach_xpos: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class W1HandSpec:
    """A hand release independent of the W1 body/arm version."""

    brand: DexforceW1HandBrand
    version: DexforceW1HandVersion
    sides: Mapping[DexforceW1ArmSide, W1HandSideSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sides", MappingProxyType(dict(self.sides)))

    def for_side(self, side: DexforceW1ArmSide | str) -> W1HandSideSpec:
        side = DexforceW1ArmSide.parse(side)
        try:
            return self.sides[side]
        except KeyError as exc:
            raise ValueError(
                f"Hand {self.brand.value} {self.version.value} has no "
                f"{side.value} asset registered"
            ) from exc


def _brainco_side(side: DexforceW1ArmSide) -> W1HandSideSpec:
    prefix = "LEFT" if side == DexforceW1ArmSide.LEFT else "RIGHT"
    side_name = "Left" if side == DexforceW1ArmSide.LEFT else "Right"
    rotation = (90.0, 0.0, 180.0 if side == DexforceW1ArmSide.LEFT else 0.0)
    return W1HandSideSpec(
        urdf_path=(
            f"BrainCoHandRevo1/BrainCo{side_name}Hand/" f"BrainCo{side_name}Hand.urdf"
        ),
        joint_names=(
            f"{prefix}_HAND_THUMB1",
            f"{prefix}_HAND_THUMB2",
            f"{prefix}_HAND_INDEX",
            f"{prefix}_HAND_MIDDLE",
            f"{prefix}_HAND_RING",
            f"{prefix}_HAND_PINKY",
        ),
        end_link_name=f"{side.value}_hand_base",
        root_link_name=f"{side.value}_thumb_dist",
        attach_xpos=_attach_xpos(rotation),
    )


def _dh_gripper_side(
    side: DexforceW1ArmSide,
    modified: bool,
) -> W1HandSideSpec:
    prefix = "LEFT" if side == DexforceW1ArmSide.LEFT else "RIGHT"
    suffix = "_M" if modified else ""
    joint_suffixes = (
        ("FINGER1", "FINGER2")
        if modified
        else (
            "FINGER1_JOINT",
            "FINGER2_JOINT",
        )
    )
    return W1HandSideSpec(
        urdf_path=f"DH_PGC_140_50{suffix}/DH_PGC_140_50{suffix}.urdf",
        joint_names=tuple(f"{prefix}_{name}" for name in joint_suffixes),
        end_link_name=f"{side.value}_base_link_1",
        root_link_name=(
            f"{side.value}_finger2" if modified else f"{side.value}_finger2_link"
        ),
        attach_xpos=_attach_xpos(
            (0.0, 0.0, 90.0),
            (0.0, 0.0, 0.0 if modified else 0.015),
        ),
    )


_W1_HAND_SPECS = {
    (
        DexforceW1HandBrand.BRAINCO_HAND,
        DexforceW1HandVersion.V021,
    ): W1HandSpec(
        brand=DexforceW1HandBrand.BRAINCO_HAND,
        version=DexforceW1HandVersion.V021,
        sides={side: _brainco_side(side) for side in DexforceW1ArmSide},
    ),
    (
        DexforceW1HandBrand.DH_PGC_GRIPPER,
        DexforceW1HandVersion.V021,
    ): W1HandSpec(
        brand=DexforceW1HandBrand.DH_PGC_GRIPPER,
        version=DexforceW1HandVersion.V021,
        sides={side: _dh_gripper_side(side, False) for side in DexforceW1ArmSide},
    ),
    (
        DexforceW1HandBrand.DH_PGC_GRIPPER_M,
        DexforceW1HandVersion.V021,
    ): W1HandSpec(
        brand=DexforceW1HandBrand.DH_PGC_GRIPPER_M,
        version=DexforceW1HandVersion.V021,
        sides={side: _dh_gripper_side(side, True) for side in DexforceW1ArmSide},
    ),
}

_DEFAULT_W1_HAND_VERSIONS = MappingProxyType(
    {
        DexforceW1HandBrand.BRAINCO_HAND: DexforceW1HandVersion.V021,
        DexforceW1HandBrand.DH_PGC_GRIPPER: DexforceW1HandVersion.V021,
        DexforceW1HandBrand.DH_PGC_GRIPPER_M: DexforceW1HandVersion.V021,
    }
)


def get_default_w1_hand_version(
    brand: DexforceW1HandBrand | str,
) -> DexforceW1HandVersion:
    """Return the explicitly selected default release for a hand brand."""
    brand = DexforceW1HandBrand.parse(brand)
    try:
        return _DEFAULT_W1_HAND_VERSIONS[brand]
    except KeyError as exc:
        raise ValueError(
            f"No default hand version registered for {brand.value}"
        ) from exc


def get_w1_hand_spec(
    brand: DexforceW1HandBrand | str,
    version: DexforceW1HandVersion | str,
) -> W1HandSpec:
    """Return an explicitly registered hand release."""
    brand = DexforceW1HandBrand.parse(brand)
    version = DexforceW1HandVersion.parse(version)
    try:
        return _W1_HAND_SPECS[(brand, version)]
    except KeyError as exc:
        raise ValueError(
            f"No hand asset registered for brand={brand.value}, "
            f"version={version.value}"
        ) from exc
