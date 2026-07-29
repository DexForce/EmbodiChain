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


_LEFT_TCP = (
    (-1.0, 0.0, 0.0, 0.012),
    (0.0, 0.0, 1.0, 0.0675),
    (0.0, 1.0, 0.0, 0.127),
    (0.0, 0.0, 0.0, 1.0),
)
_RIGHT_TCP = (
    (1.0, 0.0, 0.0, 0.012),
    (0.0, 0.0, -1.0, -0.0675),
    (0.0, 1.0, 0.0, 0.127),
    (0.0, 0.0, 0.0, 1.0),
)
_DEFAULT_TCP = {
    DexforceW1ArmSide.LEFT: _LEFT_TCP,
    DexforceW1ArmSide.RIGHT: _RIGHT_TCP,
}


@dataclass(frozen=True)
class W1VersionSpec:
    """Asset layout and calibrated defaults belonging to one W1 revision."""

    version: DexforceW1Version
    component_urdfs: dict[DexforceW1Type, str]
    full_robot_urdf_path: str
    arm_d_list: tuple[float, ...]
    arm_base_z: float
    default_eef_attach_xpos: dict[DexforceW1ArmSide, tuple]
    solver_tcp: dict[DexforceW1ArmSide, tuple]
    eyes_attach_xpos: tuple[tuple[float, ...], ...]
    wrist_camera_rpy: tuple[float, float, float]
    wrist_camera_xyz: tuple[float, float, float]
    head_contains_eyes: bool = False
    local_asset_root_env: str | None = None

    @property
    def assembly_name(self) -> str:
        return f"DexforceW1V{self.version.value.removeprefix('v')}"

    def component_urdf(self, component_type: DexforceW1Type) -> str:
        try:
            urdf_path = self.component_urdfs[component_type]
        except KeyError as exc:
            raise ValueError(
                f"W1 {self.version.value} has no registered "
                f"{component_type.value} component asset."
            ) from exc
        return self._resolve_local_urdf(urdf_path)

    def full_robot_urdf(self) -> str:
        return self._resolve_local_urdf(self.full_robot_urdf_path)

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

    def eef_attach_xpos(self, arm_side: DexforceW1ArmSide) -> np.ndarray:
        """Return the version-owned transform applied before every EEF."""
        return np.asarray(self.default_eef_attach_xpos[arm_side], dtype=float).copy()

    def compose_eef_attach_xpos(
        self,
        arm_side: DexforceW1ArmSide,
        eef_xpos: np.ndarray,
    ) -> np.ndarray:
        """Compose the W1 revision offset with an EEF-specific transform."""
        eef_xpos = np.asarray(eef_xpos, dtype=float)
        if eef_xpos.shape != (4, 4):
            raise ValueError(
                f"EEF transform must have shape (4, 4), got {eef_xpos.shape}."
            )
        return self.eef_attach_xpos(arm_side) @ eef_xpos

    def tcp(self, arm_side: DexforceW1ArmSide) -> np.ndarray:
        """Return the final EE-to-TCP transform for this W1 revision."""
        return self.compose_eef_attach_xpos(
            arm_side,
            np.asarray(self.solver_tcp[arm_side], dtype=float),
        )

    def hand_attach_xpos(
        self,
        brand: DexforceW1HandBrand,
        arm_side: DexforceW1ArmSide,
    ) -> np.ndarray:
        """Return the default transform from the arm flange to an external EEF."""
        is_left = arm_side == DexforceW1ArmSide.LEFT
        result = np.eye(4)
        if brand == DexforceW1HandBrand.BRAINCO_HAND:
            rotation = [90, 0, 180] if is_left else [90, 0, 0]
            result[:3, :3] = R.from_euler("xyz", rotation, degrees=True).as_matrix()
        elif brand == DexforceW1HandBrand.DH_PGC_GRIPPER:
            result[2, 3] = 0.015
            result[:3, :3] = R.from_rotvec([0, 0, 90], degrees=True).as_matrix()
        elif brand == DexforceW1HandBrand.DH_PGC_GRIPPER_M:
            result[:3, :3] = R.from_rotvec([0, 0, 90], degrees=True).as_matrix()
        else:
            raise ValueError(f"Unknown hand brand: {brand}")
        return self.compose_eef_attach_xpos(arm_side, result)

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
    DexforceW1Type.LEFT_ARM: "DexforceW1LeftArmV021/left_arm.urdf",
    DexforceW1Type.RIGHT_ARM: "DexforceW1RightArmV021/right_arm.urdf",
}
_V025_COMPONENT_URDFS = {
    DexforceW1Type.CHASSIS: "DexforceW1V025/w1/chassis.urdf",
    DexforceW1Type.TORSO: "DexforceW1V025/w1/torso.urdf",
    DexforceW1Type.HEAD: "DexforceW1V025/w1/head.urdf",
    DexforceW1Type.LEFT_ARM: "DexforceW1V025/w1/left_arm.urdf",
    DexforceW1Type.RIGHT_ARM: "DexforceW1V025/w1/right_arm.urdf",
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
_IDENTITY_XPOS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_V021_DEFAULT_EEF_ATTACH_XPOS = {
    DexforceW1ArmSide.LEFT: _IDENTITY_XPOS,
    DexforceW1ArmSide.RIGHT: _IDENTITY_XPOS,
}
# Calibrated outward flange offset shared by every V025 end effector.
_V025_EEF_ATTACH_XPOS = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.012),
    (0.0, 0.0, 0.0, 1.0),
)
_V025_DEFAULT_EEF_ATTACH_XPOS = {
    DexforceW1ArmSide.LEFT: _V025_EEF_ATTACH_XPOS,
    DexforceW1ArmSide.RIGHT: _V025_EEF_ATTACH_XPOS,
}

_W1_VERSION_SPECS = {
    DexforceW1Version.V021: W1VersionSpec(
        version=DexforceW1Version.V021,
        component_urdfs=_V021_COMPONENT_URDFS,
        full_robot_urdf_path="DexforceW1V021/DexforceW1_v02_1.urdf",
        arm_d_list=_SHARED_D_LIST,
        arm_base_z=0.1025,
        default_eef_attach_xpos=_V021_DEFAULT_EEF_ATTACH_XPOS,
        solver_tcp=_DEFAULT_TCP,
        eyes_attach_xpos=_EYES_ATTACH_XPOS,
        wrist_camera_rpy=_WRIST_CAMERA_RPY,
        wrist_camera_xyz=_WRIST_CAMERA_XYZ,
    ),
    DexforceW1Version.V025: W1VersionSpec(
        version=DexforceW1Version.V025,
        component_urdfs=_V025_COMPONENT_URDFS,
        full_robot_urdf_path="DexforceW1V025/w1/robot.urdf",
        # Verified against left_arm.urdf/right_arm.urdf in the V025 release.
        arm_d_list=_SHARED_D_LIST,
        arm_base_z=0.1025,
        default_eef_attach_xpos=_V025_DEFAULT_EEF_ATTACH_XPOS,
        solver_tcp=_DEFAULT_TCP,
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
