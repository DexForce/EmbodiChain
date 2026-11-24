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

"""Shared NMG observation layout and metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

import torch

_WAYPOINT_OBSERVATION_LAYOUT = "unified_constraint_tokens"
_WAYPOINT_OBSERVATION_DTYPE = "float32"
_WAYPOINT_OBSERVATION_QUATERNION_ORDER = "xyzw"

_MASK_SEMANTICS = {
    "active_onehot": "one_at_clamped_active_waypoint",
    "valid_mask": "one_for_episode_waypoints_zero_for_padding",
    "pos_mask": "one_when_position_is_constrained",
    "rot_mask": "one_when_orientation_is_constrained",
    "joint_mask": "one_when_joint_position_is_constrained",
}
_MASKED_VALUE_SEMANTICS = {
    "position": "zero",
    "quaternion": "identity_xyzw",
    "joint": "zero",
    "relative_quaternion": "target_times_inverse_eef_xyzw",
}


@dataclass(frozen=True)
class _OnnxObservationDimensions:
    """Joint count and waypoint capacity declared by the policy export."""

    num_controlled_joints: int
    num_waypoints: int

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> _OnnxObservationDimensions:
        """Read the two required dimensions from an NMG ONNX export."""
        return cls(
            _read_observation_dimension(metadata, "nmg.num_controlled_joints"),
            _read_observation_dimension(metadata, "nmg.num_waypoints"),
        )

    def check_expectations(
        self, *, num_arm_joints: int | None, num_waypoints: int | None
    ) -> None:
        """Validate optional caller expectations against the policy export."""
        for name, configured, exported in (
            ("num_arm_joints", num_arm_joints, self.num_controlled_joints),
            ("num_waypoints", num_waypoints, self.num_waypoints),
        ):
            if configured is not None and configured != exported:
                raise ValueError(
                    f"Configured {name}={configured!r} does not match the "
                    f"ONNX export value {exported}."
                )


def _read_observation_dimension(metadata: Mapping[str, str], key: str) -> int:
    raw_value = metadata.get(key)
    try:
        value = int(raw_value) if raw_value is not None else 0
    except (TypeError, ValueError):
        value = 0
    if value < 1:
        raise ValueError(
            f"ONNX observation layout metadata mismatch: {key} must be a "
            f"positive integer, got {raw_value!r}."
        )
    return value


@dataclass(frozen=True)
class _WaypointObservationLayout:
    """Concrete NMG observation layout derived from planner configuration."""

    num_waypoints: int
    num_controlled_joints: int
    use_relative_obs: bool
    canonicalize_quat_obs: bool = True
    joint_periodic_features: bool = False

    def __post_init__(self) -> None:
        if self.num_waypoints < 1 or self.num_controlled_joints < 1:
            raise ValueError("num_waypoints and num_controlled_joints must be positive")

    @property
    def block_widths(self) -> tuple[tuple[str, int], ...]:
        """Return ordered block names and widths for this policy capacity."""
        n = int(self.num_waypoints)
        j = int(self.num_controlled_joints)
        widths = [
            ("joint", j),
            ("eef", 7),
            ("waypoint_pos", 3 * n),
            ("waypoint_quat", 4 * n),
            ("waypoint_joint", j * n),
            ("active_onehot", n),
            ("valid_mask", n),
            ("pos_mask", n),
            ("rot_mask", n),
            ("joint_mask", n),
            ("last_action", j),
        ]
        if self.use_relative_obs:
            widths.extend(
                [
                    ("waypoint_rel_pos", 3 * n),
                    ("waypoint_rel_quat", 4 * n),
                    ("waypoint_joint_err", j * n),
                ]
            )
        if self.joint_periodic_features:
            widths.extend((("joint_sin", j), ("joint_cos", j)))
        return tuple(widths)

    @property
    def slices(self) -> dict[str, slice]:
        """Return a named slice for every flat observation block."""
        result = {}
        cursor = 0
        for name, width in self.block_widths:
            result[name] = slice(cursor, cursor + width)
            cursor += width
        return result

    @property
    def dim(self) -> int:
        """Return the flat observation width."""
        return sum(width for _, width in self.block_widths)

    @property
    def fingerprint(self) -> str:
        """Return the NMG-compatible fingerprint for this concrete layout."""
        payload = {
            "blocks": self.block_widths,
            "canonicalize_quat_obs": bool(self.canonicalize_quat_obs),
            "dtype": _WAYPOINT_OBSERVATION_DTYPE,
            "layout": _WAYPOINT_OBSERVATION_LAYOUT,
            "mask_semantics": _MASK_SEMANTICS,
            "masked_value_semantics": _MASKED_VALUE_SEMANTICS,
            "num_controlled_joints": int(self.num_controlled_joints),
            "num_waypoints": int(self.num_waypoints),
            "quaternion_order": _WAYPOINT_OBSERVATION_QUATERNION_ORDER,
            "use_relative_obs": bool(self.use_relative_obs),
        }
        if self.joint_periodic_features:
            payload["joint_periodic_encoding"] = "sin_cos_radians"
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def onnx_metadata(self) -> dict[str, str]:
        """Return metadata expected from an NMG ONNX export."""
        metadata = {
            "nmg.observation_layout": _WAYPOINT_OBSERVATION_LAYOUT,
            "nmg.observation_fingerprint": self.fingerprint,
            "nmg.num_waypoints": str(int(self.num_waypoints)),
            "nmg.num_controlled_joints": str(int(self.num_controlled_joints)),
            "nmg.use_relative_obs": str(bool(self.use_relative_obs)).lower(),
            "nmg.canonicalize_quat_obs": str(bool(self.canonicalize_quat_obs)).lower(),
            "nmg.quaternion_order": _WAYPOINT_OBSERVATION_QUATERNION_ORDER,
        }
        if self.joint_periodic_features:
            metadata["nmg.joint_periodic_encoding"] = "sin_cos_radians"
        return metadata

    def concatenate(self, blocks: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Validate and concatenate named observation blocks in contract order."""
        expected_names = tuple(name for name, _ in self.block_widths)
        if set(blocks) != set(expected_names):
            missing = sorted(set(expected_names) - set(blocks))
            extra = sorted(set(blocks) - set(expected_names))
            raise ValueError(
                f"Observation blocks do not match layout; missing={missing}, extra={extra}."
            )
        values = []
        batch_size = None
        for name, width in self.block_widths:
            value = blocks[name]
            if value.ndim != 2 or value.shape[-1] != width:
                raise ValueError(
                    f"Observation block {name!r} must have shape [batch, {width}], "
                    f"got {tuple(value.shape)}."
                )
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError(
                    f"Observation block {name!r} has batch size {value.shape[0]}, "
                    f"expected {batch_size}."
                )
            values.append(value)
        return torch.cat(values, dim=-1)
