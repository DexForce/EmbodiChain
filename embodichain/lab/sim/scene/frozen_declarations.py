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
"""Effective authored articulation declarations, without creating a World."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["effective_articulation_declaration", "validate_frozen_pose_fields"]


def effective_articulation_declaration(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a copied USD declaration through the existing Spawn reader.

    Args:
        config: Physical articulation fields accepted by ``ArticulationCfg``.
            Serialized pose sequences are decoded without reshaping or repair.
            Non-USD declarations return an independent copy unchanged.

    Returns:
        Owned effective scale, pose and fixed-base declarations. The original
        configured scale precision is retained when Spawn selects override or
        overlay mode. No World or articulation instance is constructed.
    """
    result = deepcopy(dict(config))
    if Path(str(config.get("fpath", ""))).suffix.lower() not in {
        ".usd",
        ".usda",
        ".usdc",
    }:
        return result
    from embodichain.lab.sim.cfg import ArticulationCfg
    from embodichain.lab.sim.spawn.usd import articulation_desc_from_usd

    # Use the same declaration translation as spawn, without creating a World.
    configured = deepcopy(dict(config))
    if configured.get("init_local_pose") is not None:
        configured["init_local_pose"] = np.asarray(configured["init_local_pose"])
    cfg = ArticulationCfg.from_dict(configured)
    descriptor, _ = articulation_desc_from_usd(cfg, per_env=True)
    # The translator selects source scale for preserve mode. Overlay keeps its
    # declared precision; round-tripping the float32 descriptor breaks lineage.
    result["body_scale"] = [float(value) for value in cfg.body_scale]
    result["init_local_pose"] = np.asarray(descriptor.pose).tolist()
    result["init_pos"] = np.asarray(descriptor.pose)[:3, 3].tolist()
    result.setdefault("root_props", {})["fixed_base"] = bool(descriptor.fixed_base)
    return result


def validate_frozen_pose_fields(
    merged: dict[str, Any], config: Mapping[str, Any]
) -> None:
    """Check raw frozen pose redundancy before descriptor translation.

    Args:
        merged: Owned declaration to receive position/Euler projections after a
            successful check; rejected input is not repaired or mutated.
        config: Authoritative raw matrix/position/intrinsic-XYZ degree fields.

    Raises:
        ValueError: Matrix shape, finiteness, rigidity or redundant fields disagree.
    """
    uid = config.get("uid")
    if merged.get("init_local_pose") is not None:
        pose = np.asarray(merged["init_local_pose"], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Frozen scene pose fields disagree for {uid!r}.")
        rotation = pose[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0):
            raise ValueError(f"Frozen scene pose is not rigid for {uid!r}.")
        expected = np.eye(4)
        expected[:3, :3] = (
            Rotation.from_euler("XYZ", config["init_rot"], degrees=True).as_matrix()
            if "init_rot" in config
            else pose[:3, :3]
        )
        expected[:3, 3] = config.get("init_pos", pose[:3, 3])
        if not np.allclose(pose, expected, atol=1e-12, rtol=0):
            raise ValueError(f"Frozen scene pose fields disagree for {uid!r}.")
        merged["init_pos"] = pose[:3, 3].tolist()
        if "init_rot" not in config:
            merged["init_rot"] = (
                Rotation.from_matrix(pose[:3, :3])
                .as_euler("XYZ", degrees=True)
                .tolist()
            )
