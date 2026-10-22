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

"""Auditable qpos normalization using explicit robot-layout and phase contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass
import hashlib
import json
import math
from typing import Literal

import numpy as np
import torch

from embodichain.utils import configclass
from embodichain.compute.trajectory import (
    AnchoredTimeMap,
    resample_with_anchors,
    differentiate_positions,
)
from .contracts import (
    Real2SimTrajectory,
    RobotJointLayout,
    SourcePhase,
    _array,
    _json,
    _plain,
)

__all__ = ["Real2SimNormalizationCfg", "NormalizedReal2SimUnit", "normalize_real2sim"]


@configclass
class Real2SimNormalizationCfg:
    """Explicit source-to-host normalization settings.

    Args:
        control_dt: Host-owned command interval in seconds; required.
        authority: Source authority. This first implementation supports qpos only.
        joint_name_map: Source-to-target aliases. Unlisted names map identically.
        hold_joint_names: Missing independent joints explicitly held at the supplied initial state.
        initial_tolerance: Initial-position tolerance in each target joint's canonical unit.
        mimic_tolerance: Redundant mimic observation tolerance in canonical joint units.
        max_anchor_delay: Permitted forward anchor timing error; None means one control tick.
        max_samples: Maximum normalized output length.
        allow_unverified: Allow diagnostic normalization, never verified execution eligibility.
    """

    control_dt: float = MISSING
    authority: Literal["qpos", "eef_pose", "hybrid"] = "qpos"
    joint_name_map: dict[str, str] = {}
    hold_joint_names: tuple[str, ...] = ()
    initial_tolerance: float = 1e-4
    mimic_tolerance: float = 1e-5
    max_anchor_delay: float | None = None
    max_samples: int = 1_000_000
    allow_unverified: bool = False


@dataclass(frozen=True)
class NormalizedReal2SimUnit:
    """Owned canonical trajectory plus original evidence and bidirectional timing.

    Args:
        source: Original immutable source evidence.
        joint_names: Canonical full robot joint order.
        joint_units: Canonical rad/m units.
        positions: Normalized ``(T,D)`` full-joint positions.
        velocities: Velocities recomputed on the complete normalized path.
        dt: Per-sample seconds, with zero at the first sample.
        phases: Full semantic phases on the normalized grid.
        expandable_joint_indices: Explicit independent-joint expansion permissions.
        time_map: Source/execution time and index correspondence.
        eef_poses: Optional base-frame TCP positions and xyzw quaternions.
        report: Immutable normalization evidence; physical success is not implied.
    """

    source: Real2SimTrajectory
    joint_names: tuple[str, ...]
    joint_units: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    dt: np.ndarray
    phases: tuple[SourcePhase, ...]
    expandable_joint_indices: tuple[int, ...]
    time_map: AnchoredTimeMap
    eef_poses: np.ndarray | None
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Normalized joint names must be nonempty and unique")
        positions = _array(self.positions, (None, len(names)), "normalized positions")
        n = len(positions)
        if (
            n < 2
            or not isinstance(self.source, Real2SimTrajectory)
            or not isinstance(self.time_map, AnchoredTimeMap)
        ):
            raise ValueError("Invalid normalized unit source or time map")
        dt = _array(self.dt, (n,), "normalized dt")
        if dt[0] != 0 or np.any(dt[1:] <= 0) or len(self.time_map.execution_times) != n:
            raise ValueError(
                "Normalized intervals require zero first dt and positive later dt"
            )
        if not np.allclose(
            np.cumsum(dt), self.time_map.execution_times, atol=1e-10, rtol=1e-10
        ):
            raise ValueError("Normalized dt does not match the execution clock")
        if len(self.joint_units) != len(names) or any(
            u not in ("rad", "m") for u in self.joint_units
        ):
            raise ValueError("Normalized units must be rad/m for every joint")
        if any(
            type(i) is not int or not 0 <= i < len(names)
            for i in self.expandable_joint_indices
        ):
            raise ValueError("Invalid expandable joint indices")
        if len(set(self.expandable_joint_indices)) != len(
            self.expandable_joint_indices
        ):
            raise ValueError("Expandable joint indices must be unique")
        previous = 0
        for phase in self.phases:
            if (
                not isinstance(phase, SourcePhase)
                or phase.start < previous
                or phase.stop > n
            ):
                raise ValueError("Invalid normalized phase partition")
            previous = phase.stop
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_units", tuple(self.joint_units))
        object.__setattr__(self, "positions", positions)
        object.__setattr__(
            self,
            "velocities",
            _array(self.velocities, positions.shape, "normalized velocities"),
        )
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(
            self, "expandable_joint_indices", tuple(self.expandable_joint_indices)
        )
        object.__setattr__(
            self,
            "eef_poses",
            (
                None
                if self.eef_poses is None
                else _array(self.eef_poses, (n, 7), "normalized EEF")
            ),
        )
        object.__setattr__(self, "report", _json(self.report))

    @property
    def revision(self) -> str:
        """Content-derived normalization revision used by SourceContext."""
        return str(self.report["normalized_revision"])

    @property
    def eligible(self) -> bool:
        """Whether reviewed input prerequisites passed, not physical qualification."""
        return self.report["input_prerequisites_verified"] is True

    def report_dict(self) -> dict:
        """Return an independent JSON-compatible normalization report."""
        return _plain(self.report)


def _normalize_eef(
    source: Real2SimTrajectory, mapping: AnchoredTimeMap
) -> np.ndarray | None:
    eef = source.eef
    if eef is None:
        return None
    from scipy.spatial.transform import Rotation, Slerp

    poses = eef.poses.astype(np.float64)
    quat = poses[:, 3:]
    if eef.quaternion_order == "wxyz":
        quat = quat[:, [1, 2, 3, 0]]
    norm = np.linalg.norm(quat, axis=1)
    if not np.allclose(norm, 1.0, atol=1e-4, rtol=0):
        raise ValueError("EEF quaternions must be normalized")
    matrices = np.tile(np.eye(4), (len(poses), 1, 1))
    matrices[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    matrices[:, :3, 3] = poses[:, :3] * (0.001 if eef.position_unit == "mm" else 1.0)
    canonical = eef.reference_to_base @ matrices @ eef.link_to_tcp
    q = mapping.source_query_times
    positions = np.stack(
        [np.interp(q, source.seconds, canonical[:, i, 3]) for i in range(3)], axis=1
    )
    rotations = Slerp(source.seconds, Rotation.from_matrix(canonical[:, :3, :3]))(
        q
    ).as_quat()
    for i in range(1, len(rotations)):
        if rotations[i - 1] @ rotations[i] < 0:
            rotations[i] *= -1
    return np.c_[positions, rotations]


def normalize_real2sim(
    source: Real2SimTrajectory,
    layout: RobotJointLayout,
    cfg: Real2SimNormalizationCfg,
) -> NormalizedReal2SimUnit:
    """Normalize one qpos-authoritative unit without live robot or scene access.

    Args:
        source: Immutable raw trajectory and evidence declarations.
        layout: Full target robot joint order, initial state and mimic graph.
        cfg: Explicit host clock, aliases and missing-joint policy.

    Returns:
        Canonical full-joint positions/velocities, optional base-frame TCP evidence,
        exact event anchors, source/execution mappings and an auditable report.
        Eligibility only establishes reviewed input prerequisites.
    """
    if not isinstance(source, Real2SimTrajectory) or not isinstance(
        layout, RobotJointLayout
    ):
        raise TypeError("source and layout must use the real2sim value contracts")
    if not isinstance(cfg, Real2SimNormalizationCfg):
        raise TypeError("cfg must be Real2SimNormalizationCfg")
    cfg.validate()
    if cfg.authority not in ("qpos", "eef_pose", "hybrid"):
        raise ValueError("Unknown real2sim authority")
    if cfg.authority != "qpos":
        raise NotImplementedError(
            "eef_pose and hybrid require an explicit IK/replanning provider"
        )
    if type(cfg.allow_unverified) is not bool:
        raise ValueError("allow_unverified must be boolean")
    for name in ("initial_tolerance", "mimic_tolerance"):
        value = getattr(cfg, name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (source.robot_id, source.robot_revision) != (
        layout.robot_id,
        layout.robot_revision,
    ):
        raise ValueError("qpos authority requires the same robot ID and revision")
    statuses = (source.calibration_status, source.annotation_status)
    if "rejected" in statuses:
        raise ValueError("Rejected calibration or annotations cannot be normalized")
    eligible = all(s == "verified" for s in statuses) and bool(source.phases)
    if not eligible and not cfg.allow_unverified:
        raise ValueError(
            "Unverified calibration/phase annotations require explicit diagnostic mode"
        )
    aliases = dict(cfg.joint_name_map)
    if not set(aliases) <= set(source.joint_names):
        raise ValueError("Joint aliases include unknown source names")
    mapped = tuple(aliases.get(n, n) for n in source.joint_names)
    if len(set(mapped)) != len(mapped) or not set(mapped) <= set(layout.joint_names):
        raise ValueError("Joint aliases must map uniquely into the target layout")
    hold = tuple(cfg.hold_joint_names)
    mimic_names = {m.joint for m in layout.mimics}
    if (
        len(set(hold)) != len(hold)
        or not set(hold) <= set(layout.joint_names) - set(mapped) - mimic_names
    ):
        raise ValueError("Hold policy must name only missing independent joints")
    missing = set(layout.joint_names) - set(mapped) - mimic_names - set(hold)
    if missing:
        raise ValueError(
            f"Missing independent joints without explicit hold policy: {sorted(missing)}"
        )
    if not set(layout.expandable_joint_names) <= set(mapped) - mimic_names:
        raise ValueError("Only observed independent joints may be expanded")
    n, d = len(source.positions), len(layout.joint_names)
    full = np.empty((n, d), dtype=np.float64)
    factors = {
        "rad": ("rad", 1.0),
        "deg": ("rad", np.pi / 180),
        "m": ("m", 1.0),
        "mm": ("m", 0.001),
    }
    column_mapping = []
    for column, target in enumerate(mapped):
        target_index = layout.joint_names.index(target)
        unit, scale = factors[source.joint_units[column]]
        if unit != layout.joint_units[target_index]:
            raise ValueError(f"Joint dimension mismatch for {target}")
        full[:, target_index] = source.positions[:, column] * scale
        column_mapping.append(
            {
                "source": source.joint_names[column],
                "target": target,
                "target_index": target_index,
                "scale": scale,
            }
        )
    held = {}
    for name in hold:
        index = layout.joint_names.index(name)
        full[:, index] = layout.initial_positions[index]
        held[name] = float(layout.initial_positions[index])
    available = set(mapped) - mimic_names | set(hold)
    remaining = list(layout.mimics)
    resolved_mimics = []
    while remaining:
        ready = [m for m in remaining if m.parent in available]
        for mimic in ready:
            j = layout.joint_names.index(mimic.joint)
            parent = layout.joint_names.index(mimic.parent)
            derived = full[:, parent] * mimic.multiplier + mimic.offset
            if mimic.joint in mapped and not np.allclose(
                full[:, j], derived, atol=cfg.mimic_tolerance, rtol=0
            ):
                raise ValueError(
                    f"Observed mimic joint {mimic.joint} conflicts with its driver"
                )
            full[:, j] = derived
            available.add(mimic.joint)
            remaining.remove(mimic)
            resolved_mimics.append(mimic)
        if not ready:
            raise ValueError("Mimic graph could not resolve its independent drivers")
    error = np.abs(full[0] - layout.initial_positions)
    if np.any(error > cfg.initial_tolerance):
        raise ValueError(
            "Source initial positions do not match the declared restored initial state"
        )
    anchors = {0, n - 1}
    for phase in source.phases:
        anchors.update((phase.start, phase.stop - 1))
    positions, time_map = resample_with_anchors(
        full,
        source.seconds,
        cfg.control_dt,
        sorted(anchors),
        max_anchor_delay=cfg.max_anchor_delay,
        max_samples=cfg.max_samples,
    )
    for mimic in resolved_mimics:
        positions[:, layout.joint_names.index(mimic.joint)] = (
            positions[:, layout.joint_names.index(mimic.parent)] * mimic.multiplier
            + mimic.offset
        )
    index_map = dict(
        zip(
            time_map.anchor_source_indices.tolist(),
            time_map.anchor_execution_indices.tolist(),
        )
    )
    phases = tuple(
        SourcePhase(
            p.name,
            index_map[p.start],
            index_map[p.stop - 1] + 1,
            p.kind,
            p.allowed_operators,
        )
        for p in source.phases
    )
    dt = np.r_[0.0, np.full(len(positions) - 1, cfg.control_dt)]
    velocities = differentiate_positions(
        torch.from_numpy(positions[None].copy()), torch.from_numpy(dt[None])
    ).numpy()[0]
    eef = _normalize_eef(source, time_map)
    report = {
        "schema_version": 1,
        "normalization_version": "qpos-anchors-v1",
        "authority": "qpos",
        "source_digest": source.fingerprint(),
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "unit_id": source.unit_id,
        "source_description": {
            "joint_names": source.joint_names,
            "joint_units": source.joint_units,
            "time_unit": source.time_unit,
            "velocity_time_unit": "s",
            "provenance": _plain(source.provenance),
            "eef": (
                None
                if source.eef is None
                else {
                    "reference_frame": source.eef.reference_frame,
                    "observed_link": source.eef.observed_link,
                    "tcp_id": source.eef.tcp_id,
                    "reference_to_base": source.eef.reference_to_base.tolist(),
                    "link_to_tcp": source.eef.link_to_tcp.tolist(),
                    "origin": source.eef.origin,
                    "quaternion_order": source.eef.quaternion_order,
                    "position_unit": source.eef.position_unit,
                }
            ),
        },
        "robot_id": layout.robot_id,
        "robot_revision": layout.robot_revision,
        "joint_names": layout.joint_names,
        "joint_units": layout.joint_units,
        "column_mapping": column_mapping,
        "held_positions": held,
        "expandable_joint_names": layout.expandable_joint_names,
        "normalization_options": {
            "initial_tolerance": cfg.initial_tolerance,
            "mimic_tolerance": cfg.mimic_tolerance,
            "max_anchor_delay": cfg.max_anchor_delay,
            "max_samples": cfg.max_samples,
        },
        "mimics": [vars(m) for m in layout.mimics],
        "initial_max_error": float(error.max()),
        "control_dt": cfg.control_dt,
        "source_samples": n,
        "execution_samples": len(positions),
        "source_duration_s": float(source.seconds[-1]),
        "execution_duration_s": float(time_map.execution_times[-1]),
        "max_anchor_delay_s": float(
            np.max(
                time_map.execution_times[time_map.anchor_execution_indices]
                - source.seconds[time_map.anchor_source_indices]
            )
        ),
        "calibration_revision": source.calibration_revision,
        "calibration_status": source.calibration_status,
        "annotation_revision": source.annotation_revision,
        "annotation_status": source.annotation_status,
        "input_prerequisites_verified": eligible,
        "eef_origin": None if source.eef is None else source.eef.origin,
        "independent_eef_measurement": source.eef is not None
        and source.eef.origin == "measured",
        "source_phases": [vars(p) for p in source.phases],
        "execution_phases": [vars(p) for p in phases],
        "unannotated_interphase_samples": "locked",
        "qvel_policy": "recomputed_after_full_path_resampling",
        "time_mapping_arrays": {
            name: {
                "key": "map_" + name,
                "shape": list(getattr(time_map, name).shape),
                "dtype": getattr(time_map, name).dtype.str,
                "sha256": hashlib.sha256(getattr(time_map, name).tobytes()).hexdigest(),
            }
            for name in time_map.__dataclass_fields__
        },
    }
    digest = hashlib.sha256(
        json.dumps(report, sort_keys=True, allow_nan=False).encode()
    )
    for array in (positions, velocities, dt, *(() if eef is None else (eef,))):
        digest.update(array.tobytes())
    report["normalized_revision"] = "real2sim-v1:" + digest.hexdigest()
    return NormalizedReal2SimUnit(
        source,
        layout.joint_names,
        layout.joint_units,
        positions,
        velocities,
        dt,
        phases,
        tuple(layout.joint_names.index(n) for n in layout.expandable_joint_names),
        time_map,
        eef,
        report,
    )
