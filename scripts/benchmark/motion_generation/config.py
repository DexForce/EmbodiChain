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

"""Configuration loading and hashing for motion-generation benchmark suites."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from embodichain.utils import configclass

from .models import AlgorithmRole

__all__ = [
    "BENCHMARK_ROOT",
    "FreeSpaceTrackCfg",
    "PlannerSpecCfg",
    "ProtocolCfg",
    "RobotSpecCfg",
    "SuiteCfg",
    "TrackCfg",
    "load_suite",
    "stable_hash",
    "suite_to_dict",
]

BENCHMARK_ROOT = Path(__file__).resolve().parent
_FREE_SPACE_TRACK_ID = "free-space-common"
_FREE_SPACE_SCENARIO = "free_space"
_SUPPORTED_PATH_SHAPES = {
    "direct",
    "l_turn",
    "s_curve",
    "orientation_only",
    "combined",
}
_SUPPORTED_START_STATE_BINS = {
    "nominal",
    "random_reachable",
    "near_limit",
    "near_singularity",
}


@configclass
class PlannerSpecCfg:
    """Configuration for one registered planner adapter."""

    id: str = ""
    adapter: str = ""
    role: str = AlgorithmRole.DIAGNOSTIC_BASELINE.value
    enabled: bool = False
    config: dict[str, Any] = {}


@configclass
class ProtocolCfg:
    """Common timing and external-validation protocol."""

    warmup_trials: int = 1
    measured_trials: int = 3
    sample_interval: int = 40
    validation_samples: int = 128
    position_threshold_m: float = 0.01
    rotation_threshold_rad: float = 0.1
    joint_limit_tolerance_rad: float = 1.0e-5


@configclass
class RobotSpecCfg:
    """Robot provider selected for every track in one suite run."""

    id: str = "franka_panda"
    provider: str = "franka_panda"
    config: dict[str, Any] = {}


@configclass
class FreeSpaceTrackCfg:
    """Case matrix for the ``free-space-common`` track."""

    batch_sizes: list[int] = [1]
    waypoint_counts: list[int] = [1, 3, 5]
    path_shapes: list[str] = ["direct", "l_turn", "s_curve"]
    start_state_bins: list[str] = ["nominal"]
    seeds: list[int] = [11]


@configclass
class TrackCfg:
    """One enabled benchmark track and its scenario provider."""

    id: str = ""
    scenario: str = ""
    enabled: bool = True
    config: dict[str, Any] = {}


@configclass
class SuiteCfg:
    """Resolved benchmark suite configuration."""

    schema_version: int = 1
    name: str = "free_space_common"
    suite_version: str = "free_space_common_v2"
    profile: str = "smoke"
    planners: list[PlannerSpecCfg] = []
    robot: RobotSpecCfg = RobotSpecCfg()
    protocol: ProtocolCfg = ProtocolCfg()
    tracks: list[TrackCfg] = []
    free_space: FreeSpaceTrackCfg = FreeSpaceTrackCfg()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuiteCfg":
        """Build and validate a suite from a YAML-compatible mapping."""
        planners = [PlannerSpecCfg(**item) for item in data.get("planners", [])]
        tracks, free_space = _resolve_tracks_and_free_space(data)
        suite = cls(
            schema_version=int(data.get("schema_version", 1)),
            name=str(data.get("name", "free_space_common")),
            suite_version=str(data.get("suite_version", "free_space_common_v2")),
            profile=str(data.get("profile", "smoke")),
            planners=planners,
            robot=RobotSpecCfg(**data.get("robot", {})),
            protocol=ProtocolCfg(**data.get("protocol", {})),
            tracks=tracks,
            free_space=free_space,
        )
        suite.validate_benchmark()
        return suite

    def enabled_tracks(self) -> list[TrackCfg]:
        """Return enabled tracks in suite order."""
        return [track for track in self.tracks if track.enabled]

    def sync_track_configs(self) -> None:
        """Copy typed free-space settings into the matching track config."""
        for track in self.tracks:
            if track.scenario == _FREE_SPACE_SCENARIO:
                track.id = track.id or _FREE_SPACE_TRACK_ID
                track.config = asdict(self.free_space)

    def validate_benchmark(self) -> None:
        """Validate values that affect benchmark correctness."""
        self.sync_track_configs()
        missing_fields = self.validate()
        if missing_fields:
            raise ValueError(
                "Benchmark suite has missing required fields: "
                + ", ".join(missing_fields)
            )
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported suite schema_version={self.schema_version}; expected 1."
            )
        if not self.planners:
            raise ValueError("The benchmark suite must declare at least one planner.")
        planner_ids = [spec.id for spec in self.planners]
        if len(planner_ids) != len(set(planner_ids)):
            raise ValueError("Planner ids must be unique within a suite.")
        for spec in self.planners:
            if not spec.id or not spec.adapter:
                raise ValueError(
                    "Every planner must define a non-empty id and adapter."
                )
            AlgorithmRole(spec.role)
        if not self.robot.id or not self.robot.provider:
            raise ValueError("robot must define non-empty id and provider values.")
        if not self.tracks:
            raise ValueError("The benchmark suite must declare at least one track.")
        track_ids = [track.id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Track ids must be unique within a suite.")
        if not self.enabled_tracks():
            raise ValueError("At least one track must be enabled.")
        for track in self.tracks:
            if not track.id or not track.scenario:
                raise ValueError("Every track must define a non-empty id and scenario.")
        if self.protocol.warmup_trials < 0:
            raise ValueError("warmup_trials must be >= 0.")
        if self.protocol.measured_trials < 1:
            raise ValueError("measured_trials must be >= 1.")
        if self.protocol.sample_interval < 2:
            raise ValueError("sample_interval must be >= 2.")
        if self.protocol.validation_samples < 2:
            raise ValueError("validation_samples must be >= 2.")
        if self.protocol.position_threshold_m <= 0.0:
            raise ValueError("position_threshold_m must be > 0.")
        if self.protocol.rotation_threshold_rad <= 0.0:
            raise ValueError("rotation_threshold_rad must be > 0.")
        if self.protocol.joint_limit_tolerance_rad < 0.0:
            raise ValueError("joint_limit_tolerance_rad must be >= 0.")
        if any(track.scenario == _FREE_SPACE_SCENARIO for track in self.tracks):
            _validate_free_space(self.free_space)
        nmg = next((spec for spec in self.planners if spec.id == "nmg"), None)
        if nmg is not None:
            if float(nmg.config.get("pos_eps", 0.01)) <= 0.0:
                raise ValueError("NMG pos_eps must be > 0.")
            if float(nmg.config.get("rot_eps", 0.1)) <= 0.0:
                raise ValueError("NMG rot_eps must be > 0.")


def _resolve_tracks_and_free_space(
    data: dict[str, Any],
) -> tuple[list[TrackCfg], FreeSpaceTrackCfg]:
    """Accept either ``tracks`` or legacy top-level ``free_space``."""
    tracks_data = data.get("tracks")
    free_space_data = dict(data.get("free_space", {}) or {})
    if tracks_data is None:
        tracks = [
            TrackCfg(
                id=_FREE_SPACE_TRACK_ID,
                scenario=_FREE_SPACE_SCENARIO,
                enabled=True,
                config=dict(free_space_data),
            )
        ]
    else:
        if not isinstance(tracks_data, list):
            raise TypeError("tracks must be a list of track mappings.")
        tracks = [TrackCfg(**item) for item in tracks_data]
        for track in tracks:
            if track.scenario == _FREE_SPACE_SCENARIO and track.config:
                free_space_data = {**free_space_data, **dict(track.config)}
    return tracks, FreeSpaceTrackCfg(**free_space_data)


def _validate_free_space(free_space: FreeSpaceTrackCfg) -> None:
    """Validate free-space case-matrix fields."""
    if not free_space.batch_sizes or any(value < 1 for value in free_space.batch_sizes):
        raise ValueError("batch_sizes must contain positive integers.")
    if not free_space.waypoint_counts or any(
        value < 1 for value in free_space.waypoint_counts
    ):
        raise ValueError("waypoint_counts must contain positive integers.")
    if not free_space.seeds:
        raise ValueError("seeds must not be empty.")
    if not free_space.path_shapes:
        raise ValueError("path_shapes must not be empty.")
    unknown_shapes = set(free_space.path_shapes) - _SUPPORTED_PATH_SHAPES
    if unknown_shapes:
        raise ValueError(f"Unsupported path_shapes: {sorted(unknown_shapes)}.")
    if not free_space.start_state_bins:
        raise ValueError("start_state_bins must not be empty.")
    unknown_bins = set(free_space.start_state_bins) - _SUPPORTED_START_STATE_BINS
    if unknown_bins:
        raise ValueError(f"Unsupported start_state_bins: {sorted(unknown_bins)}.")
    for name, values in (
        ("batch_sizes", free_space.batch_sizes),
        ("waypoint_counts", free_space.waypoint_counts),
        ("path_shapes", free_space.path_shapes),
        ("start_state_bins", free_space.start_state_bins),
        ("seeds", free_space.seeds),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"{name} must not contain duplicate values.")


def load_suite(name_or_path: str = "smoke") -> SuiteCfg:
    """Load a suite by short name or explicit YAML path."""
    requested = Path(name_or_path)
    path = (
        requested
        if requested.is_file()
        else BENCHMARK_ROOT / "suites" / f"{name_or_path}.yaml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark suite not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a mapping in suite {path}, got {type(data).__name__}."
        )
    return SuiteCfg.from_dict(data)


def suite_to_dict(suite: SuiteCfg) -> dict[str, Any]:
    """Convert a resolved suite to plain YAML/JSON-compatible values."""
    suite.sync_track_configs()
    data = asdict(suite)
    data.pop("free_space", None)
    return data


def stable_hash(value: object) -> str:
    """Return a stable SHA256 hash for JSON-compatible configuration data."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
