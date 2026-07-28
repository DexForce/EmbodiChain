# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------

"""Single source of truth for released Dexforce W1 hardware revisions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .types import (
    DexforceW1ArmKind,
    DexforceW1ArmSide,
    DexforceW1HandBrand,
    DexforceW1Type,
    DexforceW1Version,
)

__all__ = [
    "W1VersionSpec",
    "get_w1_version_spec",
    "normalize_component_versions",
]


_INDUSTRIAL_TCP = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.15),
    (0.0, 0.0, 0.0, 1.0),
)
_LEFT_ANTHROPOMORPHIC_TCP = (
    (-1.0, 0.0, 0.0, 0.012),
    (0.0, 0.0, 1.0, 0.0675),
    (0.0, 1.0, 0.0, 0.127),
    (0.0, 0.0, 0.0, 1.0),
)
_RIGHT_ANTHROPOMORPHIC_TCP = (
    (1.0, 0.0, 0.0, 0.012),
    (0.0, 0.0, -1.0, -0.0675),
    (0.0, 1.0, 0.0, 0.127),
    (0.0, 0.0, 0.0, 1.0),
)
_DEFAULT_TCP = {
    (DexforceW1ArmKind.INDUSTRIAL, DexforceW1ArmSide.LEFT): _INDUSTRIAL_TCP,
    (DexforceW1ArmKind.INDUSTRIAL, DexforceW1ArmSide.RIGHT): _INDUSTRIAL_TCP,
    (
        DexforceW1ArmKind.ANTHROPOMORPHIC,
        DexforceW1ArmSide.LEFT,
    ): _LEFT_ANTHROPOMORPHIC_TCP,
    (
        DexforceW1ArmKind.ANTHROPOMORPHIC,
        DexforceW1ArmSide.RIGHT,
    ): _RIGHT_ANTHROPOMORPHIC_TCP,
}


@dataclass(frozen=True)
class W1VersionSpec:
    """Asset layout and calibrated defaults belonging to one W1 revision."""

    version: DexforceW1Version
    supported_arm_kinds: frozenset[DexforceW1ArmKind]
    component_urdfs: dict[DexforceW1Type, str]
    full_robot_urdfs: dict[DexforceW1ArmKind, str]
    arm_d_lists: dict[DexforceW1ArmKind, tuple[float, ...]]
    arm_base_z: dict[DexforceW1ArmKind, float]
    solver_tcp: dict[tuple[DexforceW1ArmKind, DexforceW1ArmSide], tuple]
    eyes_attach_xpos: tuple[tuple[float, ...], ...]
    wrist_camera_rpy: tuple[float, float, float]
    wrist_camera_xyz: tuple[float, float, float]
    head_contains_eyes: bool = False
    local_asset_root_env: str | None = None

    @property
    def assembly_name(self) -> str:
        return f"DexforceW1V{self.version.value.removeprefix('v')}"

    def validate_arm_kind(self, arm_kind: DexforceW1ArmKind) -> None:
        if arm_kind not in self.supported_arm_kinds:
            supported = ", ".join(kind.value for kind in self.supported_arm_kinds)
            raise ValueError(
                f"W1 {self.version.value} does not provide a "
                f"{arm_kind.value} arm asset. Supported arm kinds: {supported}."
            )

    def component_urdf(self, component_type: DexforceW1Type) -> str:
        try:
            urdf_path = self.component_urdfs[component_type]
        except KeyError as exc:
            raise ValueError(
                f"W1 {self.version.value} has no registered "
                f"{component_type.value} component asset."
            ) from exc
        return self._resolve_local_urdf(urdf_path)

    def full_robot_urdf(self, arm_kind: DexforceW1ArmKind) -> str:
        self.validate_arm_kind(arm_kind)
        return self._resolve_local_urdf(self.full_robot_urdfs[arm_kind])

    def _resolve_local_urdf(self, registered_path: str) -> str:
        """Resolve a version-specific local asset override when configured."""
        if not self.local_asset_root_env:
            return registered_path
        local_root_value = os.getenv(self.local_asset_root_env)
        if not local_root_value:
            return registered_path

        local_root = Path(local_root_value).expanduser()
        relative_path = registered_path.split("/w1/", maxsplit=1)[-1]
        candidates = [
            local_root / relative_path,
            local_root / "w1" / relative_path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())

        expected = " or ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"{self.local_asset_root_env} is set to '{local_root}', but "
            f"'{registered_path}' was not found. Expected {expected}."
        )

    def tcp(
        self, arm_kind: DexforceW1ArmKind, arm_side: DexforceW1ArmSide
    ) -> np.ndarray:
        self.validate_arm_kind(arm_kind)
        return np.asarray(self.solver_tcp[(arm_kind, arm_side)], dtype=float).copy()

    def hand_attach_xpos(
        self,
        brand: DexforceW1HandBrand,
        arm_kind: DexforceW1ArmKind,
        arm_side: DexforceW1ArmSide,
    ) -> np.ndarray:
        """Return the default transform from the arm flange to an external EEF."""
        self.validate_arm_kind(arm_kind)
        is_left = arm_side == DexforceW1ArmSide.LEFT
        result = np.eye(4)
        if brand == DexforceW1HandBrand.BRAINCO_HAND:
            rotations = {
                (DexforceW1ArmKind.INDUSTRIAL, True): [90, 0, 0],
                (DexforceW1ArmKind.INDUSTRIAL, False): [90, 0, 180],
                (DexforceW1ArmKind.ANTHROPOMORPHIC, True): [90, 0, 180],
                (DexforceW1ArmKind.ANTHROPOMORPHIC, False): [90, 0, 0],
            }
            result[:3, :3] = R.from_euler(
                "xyz", rotations[(arm_kind, is_left)], degrees=True
            ).as_matrix()
        elif brand == DexforceW1HandBrand.DH_PGC_GRIPPER:
            result[2, 3] = 0.015
            result[:3, :3] = R.from_rotvec([0, 0, 90], degrees=True).as_matrix()
        elif brand == DexforceW1HandBrand.DH_PGC_GRIPPER_M:
            result[:3, :3] = R.from_rotvec([0, 0, 90], degrees=True).as_matrix()
        else:
            raise ValueError(f"Unknown hand brand: {brand}")
        return result

    def eyes_xpos(self) -> np.ndarray:
        return np.asarray(self.eyes_attach_xpos, dtype=float).copy()

    def wrist_camera_xpos(self, arm_side: DexforceW1ArmSide) -> np.ndarray:
        attach_xpos = np.eye(4)
        attach_xpos[:3, :3] = R.from_euler("xyz", self.wrist_camera_rpy).as_matrix()
        attach_xpos[:3, 3] = self.wrist_camera_xyz

        frame_xpos = np.eye(4)
        yaw = -90 if arm_side == DexforceW1ArmSide.LEFT else 90
        frame_xpos[:3, :3] = R.from_rotvec([0, 0, yaw], degrees=True).as_matrix()
        return frame_xpos @ attach_xpos


_V021_COMPONENT_URDFS = {
    DexforceW1Type.CHASSIS: "DexforceW1ChassisV021/chassis.urdf",
    DexforceW1Type.TORSO: "DexforceW1TorsoV021/torso.urdf",
    DexforceW1Type.EYES: "DexforceW1EyesV021/eyes.urdf",
    DexforceW1Type.HEAD: "DexforceW1HeadV021/head.urdf",
    DexforceW1Type.LEFT_ARM1: "DexforceW1LeftArm1V021/left_arm.urdf",
    DexforceW1Type.RIGHT_ARM1: "DexforceW1RightArm1V021/right_arm.urdf",
    DexforceW1Type.LEFT_ARM2: "DexforceW1LeftArm2V021/left_arm.urdf",
    DexforceW1Type.RIGHT_ARM2: "DexforceW1RightArm2V021/right_arm.urdf",
}
_V025_COMPONENT_URDFS = {
    DexforceW1Type.CHASSIS: "DexforceW1V025/w1/chassis.urdf",
    DexforceW1Type.TORSO: "DexforceW1V025/w1/torso.urdf",
    DexforceW1Type.HEAD: "DexforceW1V025/w1/head.urdf",
    DexforceW1Type.LEFT_ARM1: "DexforceW1V025/w1/left_arm.urdf",
    DexforceW1Type.RIGHT_ARM1: "DexforceW1V025/w1/right_arm.urdf",
}
_SHARED_D_LIST = (0.0, 0.0, 0.260, 0.0, 0.166, 0.098, 0.0)
_EYES_ATTACH_XPOS = (
    (-0.0, 0.25959, -0.96572, 0.091),
    (0.0, -0.96572, -0.25959, -0.051),
    (-1.0, -0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_WRIST_CAMERA_RPY = (2.79252648, 0.0, 1.57079633)
_WRIST_CAMERA_XYZ = (0.08, 0.0, 0.06)

_W1_VERSION_SPECS = {
    DexforceW1Version.V021: W1VersionSpec(
        version=DexforceW1Version.V021,
        supported_arm_kinds=frozenset(DexforceW1ArmKind),
        component_urdfs=_V021_COMPONENT_URDFS,
        full_robot_urdfs={
            DexforceW1ArmKind.ANTHROPOMORPHIC: ("DexforceW1V021/DexforceW1_v02_1.urdf"),
            DexforceW1ArmKind.INDUSTRIAL: "DexforceW1V021/DexforceW1_v02_2.urdf",
        },
        arm_d_lists={
            DexforceW1ArmKind.ANTHROPOMORPHIC: _SHARED_D_LIST,
            DexforceW1ArmKind.INDUSTRIAL: _SHARED_D_LIST,
        },
        arm_base_z={
            DexforceW1ArmKind.ANTHROPOMORPHIC: 0.1025,
            DexforceW1ArmKind.INDUSTRIAL: 0.1025,
        },
        solver_tcp=_DEFAULT_TCP,
        eyes_attach_xpos=_EYES_ATTACH_XPOS,
        wrist_camera_rpy=_WRIST_CAMERA_RPY,
        wrist_camera_xyz=_WRIST_CAMERA_XYZ,
    ),
    DexforceW1Version.V025: W1VersionSpec(
        version=DexforceW1Version.V025,
        supported_arm_kinds=frozenset({DexforceW1ArmKind.ANTHROPOMORPHIC}),
        component_urdfs=_V025_COMPONENT_URDFS,
        full_robot_urdfs={
            DexforceW1ArmKind.ANTHROPOMORPHIC: "DexforceW1V025/w1/robot.urdf",
        },
        arm_d_lists={
            # Verified against left_arm.urdf/right_arm.urdf in the V025 release.
            DexforceW1ArmKind.ANTHROPOMORPHIC: _SHARED_D_LIST,
        },
        arm_base_z={DexforceW1ArmKind.ANTHROPOMORPHIC: 0.1025},
        solver_tcp={
            key: value
            for key, value in _DEFAULT_TCP.items()
            if key[0] == DexforceW1ArmKind.ANTHROPOMORPHIC
        },
        eyes_attach_xpos=_EYES_ATTACH_XPOS,
        wrist_camera_rpy=_WRIST_CAMERA_RPY,
        wrist_camera_xyz=_WRIST_CAMERA_XYZ,
        head_contains_eyes=True,
        local_asset_root_env="EMBODICHAIN_W1_V025_ROOT",
    ),
}


def get_w1_version_spec(version: DexforceW1Version | str) -> W1VersionSpec:
    if not isinstance(version, DexforceW1Version):
        version = DexforceW1Version(version.lower())
    return _W1_VERSION_SPECS[version]


def normalize_component_versions(
    versions: dict[DexforceW1Type | str, DexforceW1Version | str] | None,
) -> dict[DexforceW1Type, DexforceW1Version]:
    normalized = {}
    for component_type, version in (versions or {}).items():
        if not isinstance(component_type, DexforceW1Type):
            component_type = DexforceW1Type(component_type)
        if not isinstance(version, DexforceW1Version):
            version = DexforceW1Version(version.lower())
        normalized[component_type] = version
    return normalized
