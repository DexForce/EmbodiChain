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

"""Strict configuration for scene baselines and grouped layout generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING
import math
from pathlib import Path
from typing import Any

import yaml

from embodichain.utils import configclass

__all__ = [
    "MemberRandomizationCfg",
    "SupportOverrideCfg",
    "IntraGroupRandomizationCfg",
    "SceneRandomizationCfg",
    "load_randomization_config",
]

_GENERATION_FIELDS = (
    "seed",
    "candidate_count",
    "max_attempts",
    "collision_margin_m",
)
_BATCH_FIELDS = ("backend", "device", "batch_size")
_BATCH_GENERATOR = "scene_engine.group_batch_rejection/v1"
_LEGACY_GENERATOR = "scene_engine.group_randomization/v1"


_INTRA_GENERATOR = "scene_engine.hierarchical_batch_rejection/v2"
_INTRA_GENERATORS = ("scene_engine.hierarchical_batch_rejection/v1", _INTRA_GENERATOR)


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


@configclass
class SceneRandomizationCfg:
    """Resolved scene paths and grouped layout generation parameters.

    Attributes:
        scene_path: Absolute path to a portable ``scene_export`` directory.
        base_data_path: Absolute destination for the baseline YAML file.
        schema_version: Normalized baseline configuration schema, currently 1.
        seed: Nonnegative seed for reproducible layout proposals.
        candidate_count: Positive number of successful layouts requested.
        max_attempts: Total proposal budget, at least ``candidate_count``.
        collision_margin_m: Finite, nonnegative clearance between group AABBs.
        backend: ``slsqp`` preserves the original solver; ``batched`` samples
            independent bounded layouts and rejects intersecting group AABBs.
        device: Tensor device for the batched backend, ``cpu`` or ``cuda[:N]``.
        batch_size: Maximum proposals evaluated together by the batched backend.
        randomize_groups: Whether to sample group-root XY translations.
        intra_group: Optional explicit non-root member XY/yaw selection. Requires
            the batched backend. Empty targets retain the original group path.
    """

    scene_path: Path = MISSING
    base_data_path: Path = MISSING
    schema_version: int = 1
    seed: int = 0
    candidate_count: int = 10
    max_attempts: int = 1000
    collision_margin_m: float = 0.02
    backend: str = "slsqp"
    device: str = "cpu"
    batch_size: int = 4096
    randomize_groups: bool = True
    intra_group: IntraGroupRandomizationCfg | None = None


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


def _validate_policy(policy: dict) -> None:
    """Validate persisted policies while retaining the exact legacy hash schema."""
    version = policy.get("schema_version")
    fields = set(_GENERATION_FIELDS)
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("Unsupported resolved policy schema.")
    if version >= 2:
        fields.update(_BATCH_FIELDS)
    if version == 3:
        fields.update(("randomize_groups", "intra_group"))
    if set(policy) != {"schema_version", *fields}:
        raise ValueError("Unsupported resolved policy fields.")
    _validate_generation_parameters(**{key: policy[key] for key in _GENERATION_FIELDS})
    if version >= 2:
        _validate_backend(**{key: policy[key] for key in _BATCH_FIELDS})
        if policy["backend"] != "batched":
            raise ValueError("Batch policies require the batched backend.")
    if version == 3:
        _validate_scopes(
            policy["randomize_groups"],
            _decode_intra(policy["intra_group"]),
            policy["backend"],
        )


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


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate and non-string keys before decoding configuration."""

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[str, Any]:
        if not isinstance(node, yaml.nodes.MappingNode):
            raise ValueError("Configuration entries must be mappings.")
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Configuration mapping keys must be strings.")
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key!r}.")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _mapping(value: Any, *, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}.")
    return value


def load_randomization_config(
    path: str | Path, *, output_root: str | Path
) -> SceneRandomizationCfg:
    """Load a closed version-1, version-2 or hierarchical version-3 policy.

    Version 3 adds ``randomize_groups`` and a nested ``intra_group`` selection.
    Targets are explicit non-root UIDs; XY offsets refer to original entity
    origins in the parent support frame. Its defaults use 2 mm boundary/sibling
    margins, 1 mm support gap and 1 mm conservative raster cells. At least one
    scope must be enabled, and selected members require ``backend: batched``.

    The original ``schema_version: 1`` accepts optional
    ``seed``, ``candidate_count``, ``max_attempts`` and ``collision_margin_m``.
    Optional ``backend``, ``device`` and ``batch_size`` select batch rejection
    sampling; its distribution and acceptance rate differ from SLSQP repair.
    A saved version-2 batch policy can be reused directly; its fields are
    validated before normalizing the baseline configuration version to 1.
    Omitted parameters use :class:`SceneRandomizationCfg` defaults. The scene
    workspace is supplied separately through ``output_root``: its ``scene_export/``
    directory is always the source, and ``scene_randomization_base.yaml`` in
    that directory is always the baseline destination. Export contents are
    validated during baseline preparation.

    Args:
        path: Configuration YAML path.
        output_root: Scene workspace, resolved relative to the current working
            directory. Pass the workspace containing ``scene_export/``, rather
            than the export directory itself.

    Returns:
        Configuration with absolute input and output paths.

    Raises:
        OSError: The configuration cannot be read.
        ValueError: YAML syntax, schema, keys, or values are invalid.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.load(
            config_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid randomization configuration YAML: {exc}") from exc

    if isinstance(payload, dict) and {"input", "output"} & payload.keys():
        raise ValueError(
            "Configuration input/output paths are no longer supported; "
            "use --output_root to select the scene workspace. "
            "The baseline is written to "
            "scene_export/scene_randomization_base.yaml within that workspace."
        )
    root = _mapping(
        payload,
        name="configuration",
        allowed={
            "schema_version",
            *_GENERATION_FIELDS,
            *_BATCH_FIELDS,
            "randomize_groups",
            "intra_group",
        },
    )
    version = root.get("schema_version")
    if type(version) is int and version == 2 and root.get("backend") == "batched":
        _validate_policy(root)
        version = 1
    if type(version) is not int or version not in (1, 3):
        raise ValueError("schema_version must be the integer 1, 2, or 3.")
    if version != 3 and ({"randomize_groups", "intra_group"} & root.keys()):
        raise ValueError("Intra-group fields require schema_version: 3.")
    if not isinstance(output_root, (str, Path)) or (
        isinstance(output_root, str) and not output_root.strip()
    ):
        raise ValueError("--output_root must be a non-empty workspace path.")
    scene_path = Path(output_root).expanduser().resolve() / "scene_export"
    # Keep the final path unresolved so persistence can reject a symlink file.
    base_data_path = scene_path / "scene_randomization_base.yaml"
    if base_data_path.resolve() == config_path:
        raise ValueError("Baseline output must not overwrite the input configuration.")
    cfg = SceneRandomizationCfg(
        scene_path=scene_path,
        base_data_path=base_data_path,
        schema_version=1,
        randomize_groups=root.get("randomize_groups", True),
        intra_group=_decode_intra(root.get("intra_group")),
        **{
            key: root[key]
            for key in (*_GENERATION_FIELDS, *_BATCH_FIELDS)
            if key in root
        },
    )
    cfg.validate()
    _validate_generation_parameters(
        seed=cfg.seed,
        candidate_count=cfg.candidate_count,
        max_attempts=cfg.max_attempts,
        collision_margin_m=cfg.collision_margin_m,
    )
    _validate_backend(backend=cfg.backend, device=cfg.device, batch_size=cfg.batch_size)
    _validate_scopes(cfg.randomize_groups, cfg.intra_group, cfg.backend)
    return cfg


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


def _validate_scopes(
    randomize_groups: bool, intra: IntraGroupRandomizationCfg | None, backend: str
) -> None:
    if type(randomize_groups) is not bool:
        raise ValueError("randomize_groups must be a boolean.")
    if intra is not None:
        _validate_intra(intra)
    enabled = intra is not None and bool(intra.targets)
    if enabled and backend != "batched":
        raise ValueError("Intra-group randomization requires backend: batched.")
    if not randomize_groups and not enabled:
        raise ValueError("At least one randomization scope must be enabled.")


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
