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

"""Import pinned robot assets into an independent Newton teacher builder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

__all__: list[str] = []

_SOURCE_COMMIT = "476dc67fccc6a99be293bf611f897b1c06f13e30"
_ASSETS = {"cartpole": "cartpole_single.urdf", "ant": "ant.xml"}
_JOINTS = {
    "cartpole": ("slider_to_cart", "cart_to_pole"),
    "ant": (
        "hip_1",
        "ankle_1",
        "hip_2",
        "ankle_2",
        "hip_3",
        "ankle_3",
        "hip_4",
        "ankle_4",
    ),
}


def _ant_reference_friction() -> dict[str, Any]:
    """Describe the coupled old Euclidean-to-Newton Huber friction mapping.

    Newton uses ``H=delta*(speed-delta/2)`` above ``delta`` in both
    normalization and the viscous/Coulomb minimum. Scaling both materials'
    Coulomb coefficients by the same ``delta`` approximates the original
    Euclidean law without changing the installed solver. This is a parameter
    semantics conversion, not equality of the numerical material arrays.
    """
    delta = 1e-3
    return {
        "schema": "ant-warp-euclidean-to-newton-huber-friction-v1",
        "friction_smoothing": delta,
        "material_mu_scale": delta,
        "reference_shape_mu": 0.75,
        "reference_ground_mu": 0.5,
        "shape_mu": 0.75 * delta,
        "ground_mu": 0.5 * delta,
        "interpretation": (
            "Coupled Huber/Coulomb parameter mapping; approximate original "
            "Euclidean friction above the smoothing scale. Actual Newton "
            "coefficients differ from the original coefficients."
        ),
    }


def _asset(robot: str) -> Path:
    if robot not in _ASSETS:
        raise ValueError(f"Unsupported NeRD reference robot: {robot}")
    from .._resources import prepare_reference_assets

    return prepare_reference_assets() / _ASSETS[robot]


def reference_asset_hash(robot: str) -> str:
    """Return the SHA256 of the pinned downloaded reference robot source."""
    return hashlib.sha256(_asset(robot).read_bytes()).hexdigest()


def reference_joint_names(robot: str) -> tuple[str, ...]:
    """Return actuated/scalar joints in the official generalized-state order."""
    _asset(robot)
    return _JOINTS[robot]


def reference_profile(robot: str) -> dict[str, Any]:
    """Return source identity, state dimensions, controls and default reset state."""
    _asset(robot)
    return {
        "robot": robot,
        "dt": 1 / 60,
        "source_commit": _SOURCE_COMMIT,
        "asset_sha256": reference_asset_hash(robot),
        "joint_names": list(reference_joint_names(robot)),
        "q_dim": 2 if robot == "cartpole" else 15,
        "qd_dim": 2 if robot == "cartpole" else 14,
        "default_q": (
            [0.0, 0.0]
            if robot == "cartpole"
            else [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                -1.0,
                0.0,
                -1.0,
                0.0,
                1.0,
            ]
        ),
        "default_qd": [0.0] * (2 if robot == "cartpole" else 14),
        "effort_limits": [1500.0, 0.0] if robot == "cartpole" else [400.0] * 8,
        "codec": {
            "robot": robot,
            "base_com": [0.0, 0.0, 0.0],
            "continuous_q_indices": [1] if robot == "cartpole" else [],
        },
        "teacher_profile": "current_newton_import_generalized_effort",
        "source_actuators_imported": False,
        "source_solver_options_imported": False,
    }


def _import_reference(robot: str, physics_profile: str = "native") -> Any:
    """Interpret the pinned asset with upstream environment import defaults."""
    import newton
    import warp as wp

    if physics_profile not in {"native", "warp_reference"}:
        raise ValueError(f"Unsupported reference physics profile: {physics_profile}")
    legacy = physics_profile == "warp_reference"
    builder = newton.ModelBuilder()
    if robot == "cartpole":
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.01, limit_ke=1e4, limit_kd=10.0
        )
        if legacy:
            builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
                density=1000.0, mu=0.25
            )
        builder.add_urdf(
            str(_asset(robot)),
            xform=wp.transform(
                (0.0, 0.0, 0.0),
                (
                    wp.quat_identity()
                    if legacy
                    else wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)
                ),
            ),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=legacy,
        )
        if legacy:
            # Warp 1.8 URDF imports ignore authored inertials by default and
            # add armature to BOTH body inertia and generalized joint inertia.
            for index, inertia in enumerate(builder.body_inertia):
                builder.body_inertia[index] = wp.mat33(
                    np.asarray(inertia).reshape(3, 3) + 0.01 * np.eye(3)
                )
                builder.body_inv_inertia[index] = wp.inverse(
                    builder.body_inertia[index]
                )
    else:
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.05, limit_ke=1e3, limit_kd=1e2
        )
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=4e4, kd=1e2, kf=3e3, mu=0.75
        )
        source = str(_asset(robot))
        if legacy:
            # Warp 1.5.1 honors density for spheres but uses the parser's
            # density=1000 for capsules, ignoring MJCF geom density=5.
            # Express those measured semantics in a temporary XML string;
            # the pinned, packaged asset remains byte-for-byte unchanged.
            root = ET.parse(source).getroot()
            for geom in root.findall(".//body/geom"):
                if geom.get("type", "sphere") == "capsule":
                    geom.set("density", "1000")
                    # Old Y-axis capsules emit +Y then -Y contact slots. The
                    # modern fromto importer directs +Z toward the other end.
                    # Reverse the equivalent capsule frame to preserve the
                    # network's ordered contact features, not just geometry.
                    endpoints = geom.get("fromto", "").split()
                    if len(endpoints) == 6:
                        geom.set("fromto", " ".join(endpoints[3:] + endpoints[:3]))
                geom.set("friction", "0.75 0 0")
                geom.set("margin", "0")
            source = ET.tostring(root, encoding="unicode")
        builder.add_mjcf(
            source,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_names=["floor", "ground"],
            parse_mujoco_options=not legacy,
        )
        if legacy:
            # The original root authors armature=0; scalar joints author .01.
            # Old force mode ignores target_kd=.1. Newton now imports that
            # value as passive damping, so leaving it active changes physics.
            for index in range(1, builder.body_count):
                builder.body_inertia[index] = wp.mat33(
                    np.asarray(builder.body_inertia[index]).reshape(3, 3)
                    + 0.01 * np.eye(3)
                )
                builder.body_inv_inertia[index] = wp.inverse(
                    builder.body_inertia[index]
                )
            builder.joint_damping[:] = [0.0] * len(builder.joint_damping)
            builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
            builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)
    return builder
