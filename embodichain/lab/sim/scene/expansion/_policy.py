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

"""Source-independent policies for bounded scene layout sampling."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import MISSING
import math
from typing import Any
from embodichain.utils import configclass

__all__ = [
    "MemberRandomizationCfg",
    "SupportOverrideCfg",
    "IntraGroupRandomizationCfg",
    "SceneRandomizationPolicyCfg",
    "SceneRandomizationJobCfg",
]


@configclass
class MemberRandomizationCfg:
    """One non-root member's displacement bounds in its parent support frame.

    ``xy_offset_bounds_m`` is ``[[min_x, min_y], [max_x, max_y]]`` relative
    to the original entity origin; ``None`` samples the complete support area.
    Yaw is a continuous increment around the support normal, retaining tilt.
    """

    xy_offset_bounds_m: list[list[float]] | None = None
    yaw_delta_range_deg: list[float] = [-180.0, 180.0]


@configclass
class SupportOverrideCfg:
    """Explicit parent-local support frame and GeoJSON Polygon/MultiPolygon.

    The proper rigid ``parent_from_support`` matrix locates the support plane
    at support Z=0 in the parent local Z-up geometry after asset scaling. Its polygon must be covered by actual coplanar mesh faces.
    """

    parent_from_support: list[list[float]] = MISSING
    polygon: dict = MISSING


@configclass
class IntraGroupRandomizationCfg:
    """Explicit member selection and conservative support geometry settings.

    Fixed members follow their parent. Clearances and grid resolution are in
    metres. Overrides are keyed by parent UID, independently of target order.
    """

    targets: dict[str, MemberRandomizationCfg] = {}
    boundary_margin_m: float = 0.002
    sibling_margin_m: float = 0.002
    support_gap_m: float = 0.001
    grid_resolution_m: float = 0.001
    support_overrides: dict[str, SupportOverrideCfg] = {}


def _validate_backend(*, backend: str, device: str, batch_size: int) -> None:
    """Validate execution controls without initializing CUDA or reading assets."""
    if backend not in ("slsqp", "batched"):
        raise ValueError("backend must be slsqp or batched.")
    if not isinstance(device, str) or not (
        device == "cpu"
        or device == "cuda"
        or (
            device.startswith("cuda:") and device[5:].isascii() and device[5:].isdigit()
        )
    ):
        raise ValueError("device must be cpu, cuda, or cuda:N.")
    if backend == "slsqp" and device != "cpu":
        raise ValueError(
            "The slsqp backend runs on CPU; use backend: batched for CUDA."
        )
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")


def _validate_generation_parameters(
    *, seed: int, candidate_count: int, max_attempts: int, collision_margin_m: float
) -> None:
    """Validate YAML values and Python overrides before accessing scene files."""
    for name, value, minimum in (
        ("seed", seed, 0),
        ("candidate_count", candidate_count, 1),
        ("max_attempts", max_attempts, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if candidate_count > max_attempts:
        raise ValueError("max_attempts must be at least candidate_count.")
    if type(collision_margin_m) not in (int, float) or collision_margin_m < 0:
        raise ValueError("collision_margin_m must be finite and nonnegative.")
    try:
        finite = math.isfinite(collision_margin_m)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("collision_margin_m must be finite and nonnegative.")


def _mapping(value: Any, *, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}.")
    return value


def _decode_intra(value: Any) -> IntraGroupRandomizationCfg | None:
    """Strictly decode nested YAML mappings, also used for persisted policies."""
    if value is None:
        return None
    root = _mapping(
        value,
        name="intra_group",
        allowed={
            "targets",
            "boundary_margin_m",
            "sibling_margin_m",
            "support_gap_m",
            "grid_resolution_m",
            "support_overrides",
        },
    )
    targets = root.get("targets", {})
    if not isinstance(targets, dict):
        raise ValueError("targets must be a mapping.")
    decoded = {}
    for uid, target in targets.items():
        if not isinstance(uid, str) or not uid.strip():
            raise ValueError("Target UIDs must be nonempty strings.")
        item = _mapping(
            target,
            name=f"target {uid}",
            allowed={"xy_offset_bounds_m", "yaw_delta_range_deg"},
        )
        decoded[uid] = MemberRandomizationCfg(**item)
    overrides = root.get("support_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("support_overrides must be a mapping.")
    override_cfg = {}
    for uid, override in overrides.items():
        item = _mapping(
            override,
            name=f"support override {uid}",
            allowed={"parent_from_support", "polygon"},
        )
        if set(item) != {"parent_from_support", "polygon"}:
            raise ValueError("Support overrides require a transform and polygon.")
        override_cfg[uid] = SupportOverrideCfg(**item)
    cfg = IntraGroupRandomizationCfg(
        **{k: v for k, v in root.items() if k not in ("targets", "support_overrides")},
        targets=decoded,
        support_overrides=override_cfg,
    )
    _validate_intra(cfg)
    return cfg


def _validate_intra(cfg: IntraGroupRandomizationCfg) -> None:
    import numpy as np

    if not isinstance(cfg, IntraGroupRandomizationCfg):
        raise ValueError("intra_group must be an IntraGroupRandomizationCfg.")
    cfg.validate()
    if not isinstance(cfg.targets, dict) or not isinstance(cfg.support_overrides, dict):
        raise ValueError("targets and support_overrides must be mappings.")
    for field in (
        "boundary_margin_m",
        "sibling_margin_m",
        "support_gap_m",
        "grid_resolution_m",
    ):
        value = getattr(cfg, field)
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0 or (field == "grid_resolution_m" and value == 0):
            raise ValueError(
                f"{field} must be finite and {'positive' if field == 'grid_resolution_m' else 'nonnegative'}."
            )
    for uid, item in cfg.targets.items():
        if (
            not isinstance(uid, str)
            or not uid
            or not isinstance(item, MemberRandomizationCfg)
        ):
            raise ValueError("targets must map UIDs to MemberRandomizationCfg.")
        yaw = np.asarray(item.yaw_delta_range_deg)
        if (
            yaw.shape != (2,)
            or yaw.dtype.kind not in "fi"
            or not np.isfinite(yaw).all()
            or yaw[0] > yaw[1]
            or not np.isfinite(yaw[1] - yaw[0])
        ):
            raise ValueError(
                "yaw_delta_range_deg requires finite ordered bounds and width."
            )
        if item.xy_offset_bounds_m is not None:
            xy = np.asarray(item.xy_offset_bounds_m)
            if (
                xy.shape != (2, 2)
                or xy.dtype.kind not in "fi"
                or not np.isfinite(xy).all()
                or np.any(xy[0] > xy[1])
            ):
                raise ValueError(
                    "xy_offset_bounds_m requires finite ordered lower/upper XY corners."
                )
    for uid, override in cfg.support_overrides.items():
        if (
            not isinstance(uid, str)
            or not uid
            or not isinstance(override, SupportOverrideCfg)
        ):
            raise ValueError("support_overrides must map UIDs to SupportOverrideCfg.")
        override.validate()
        from embodichain.lab.sim.scene._support_grid import _read_support_polygon

        try:
            _read_support_polygon(override.polygon)
        except ValueError as exc:
            raise ValueError(f"Support override {uid!r}: {exc}") from exc
        try:
            matrix = np.asarray(override.parent_from_support, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "parent_from_support must contain finite numeric values."
            ) from exc
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0, 0, 0, 1])
            or not np.allclose(
                matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8, rtol=0
            )
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError("parent_from_support must be a proper rigid transform.")


def _intra_payload(cfg: IntraGroupRandomizationCfg) -> dict:
    """Serialize only documented settings, independent of configclass internals."""
    return deepcopy(
        {
            "targets": {
                uid: {
                    "xy_offset_bounds_m": item.xy_offset_bounds_m,
                    "yaw_delta_range_deg": list(item.yaw_delta_range_deg),
                }
                for uid, item in sorted(cfg.targets.items())
            },
            **{
                key: float(getattr(cfg, key))
                for key in (
                    "boundary_margin_m",
                    "sibling_margin_m",
                    "support_gap_m",
                    "grid_resolution_m",
                )
            },
            "support_overrides": {
                uid: {
                    "parent_from_support": item.parent_from_support,
                    "polygon": item.polygon,
                }
                for uid, item in sorted(cfg.support_overrides.items())
            },
        }
    )


@configclass
class SceneRandomizationPolicyCfg:
    """Geometric policy independent of scene files and runtime lifecycles.

    Args:
        backend: Explicit numeric backend; the shared interface uses batched sampling.
        device: CPU or CUDA tensor device.
        batch_size: Maximum logical proposals evaluated together.
        collision_margin_m: Minimum group AABB separation in metres.
        randomize_groups: Sample permitted group-root XY translations.
        group_targets: Root UIDs to move; None selects all permitted roots.
        group_xy_offset_bounds_m: Optional per-root arena XY displacement bounds.
        intra_group: Selected non-root XY/yaw changes and support settings.
    """

    backend: str = "batched"
    device: str = "cpu"
    batch_size: int = 4096
    collision_margin_m: float = 0.02
    randomize_groups: bool = True
    group_targets: tuple[str, ...] | None = None
    group_xy_offset_bounds_m: dict[str, list[list[float]]] = {}
    intra_group: IntraGroupRandomizationCfg | None = None


@configclass
class SceneRandomizationJobCfg:
    """Bounded caller-owned generation settings and output representation.

    Args:
        policy: Source-independent geometric sampling policy.
        seed: Addressed generation seed.
        candidate_count: Requested valid layouts.
        max_attempts: Whole-layout proposal budget.
        layout_storage: JSON or packed candidate storage.
    """

    policy: SceneRandomizationPolicyCfg = SceneRandomizationPolicyCfg()
    seed: int = 0
    candidate_count: int = 10
    max_attempts: int = 1000
    layout_storage: str = "packed"
