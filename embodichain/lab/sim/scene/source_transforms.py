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
"""Declared physical pose/scale transformations without live simulation."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import math
import warnings

__all__ = ["normalize_source_pose", "apply_source_body_scale", "rotate_source_world_z"]


def normalize_source_pose(config: dict[str, Any]) -> None:
    """Normalize owned pose fields, retaining matrix authority and XYZ degrees."""
    if config.get("init_local_pose") is not None:
        import numpy as np
        from scipy.spatial.transform import Rotation

        pose = np.asarray(config["init_local_pose"], dtype=float)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)
        ):
            raise ValueError("init_local_pose must be a finite rigid 4x4 pose.")
        config["init_local_pose"] = pose.tolist()
        config["init_pos"] = pose[:3, 3].tolist()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected")
            config["init_rot"] = (
                Rotation.from_matrix(pose[:3, :3])
                .as_euler("XYZ", degrees=True)
                .tolist()
            )
    else:
        config["init_pos"] = _vector3(config.get("init_pos", [0.0, 0.0, 0.0]))
        config["init_rot"] = _vector3(config.get("init_rot", [0.0, 0.0, 0.0]))
    if "body_scale" in config:
        scale = _vector3(config["body_scale"])
        if any(value <= 0.0 for value in scale):
            raise ValueError(
                f"Object {config.get('uid')!r} has non-positive body_scale."
            )
        config["body_scale"] = scale


def apply_source_body_scale(
    config: dict[str, Any],
    *,
    policy: str,
    requested: Sequence[float],
) -> None:
    """Apply preserve/multiply/absolute scale to an owned source declaration."""
    if policy not in {"preserve", "multiply", "absolute"}:
        raise ValueError(
            "Source body scale policy must be preserve, multiply, or absolute."
        )
    source = _vector3(config.get("body_scale", [1.0, 1.0, 1.0]))
    if policy == "preserve":
        result = source
    elif policy == "multiply":
        result = [left * right for left, right in zip(source, requested)]
    else:
        result = list(requested)
    config["body_scale"] = [_clean_float(value) for value in result]


def rotate_source_world_z(config: dict[str, Any], degrees: float) -> None:
    """Rotate the declared world pose about Z, preserving source rounding."""
    if math.isclose(degrees, 0.0, abs_tol=1e-12):
        return
    theta = math.radians(degrees)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    x, y, z = _vector3(config["init_pos"])
    config["init_pos"] = [
        _clean_float(x * cos_theta - y * sin_theta),
        _clean_float(x * sin_theta + y * cos_theta),
        _clean_float(z),
    ]

    # EmbodiChain and Prompt2Scene both interpret these values as intrinsic XYZ.
    from scipy.spatial.transform import Rotation

    original = Rotation.from_euler("XYZ", config["init_rot"], degrees=True)
    world_z = Rotation.from_rotvec([0.0, 0.0, theta])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        rotated = (world_z * original).as_euler("XYZ", degrees=True)
    config["init_rot"] = [_clean_float(value) for value in rotated]
    if config.get("init_local_pose") is not None:
        import numpy as np

        pose = np.asarray(config["init_local_pose"], dtype=float).copy()
        pose[:3, :3] = world_z.as_matrix() @ pose[:3, :3]
        pose[:3, 3] = config["init_pos"]
        config["init_local_pose"] = pose.tolist()


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Expected a finite xyz vector, got: {value!r}")
    values = [float(item) for item in value]
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"Expected a finite xyz vector, got: {value!r}")
    return values


def _clean_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 1e-12 else rounded
