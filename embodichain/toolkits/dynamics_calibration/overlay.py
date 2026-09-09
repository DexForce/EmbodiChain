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

"""Reviewable drive-overlay construction and serialization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .schema import CalibrationConfig

_OVERLAY_KIND = "embodichain.dynamics_calibration.drive_overlay"


def build_drive_overlay(
    config: CalibrationConfig, candidate: Mapping[str, float]
) -> dict[str, Any]:
    """Build a non-destructive RobotCfg-compatible drive-property overlay.

    Args:
        config: Validated calibration configuration and asset identities.
        candidate: Exact parameter-name to candidate-value mapping.

    Returns:
        Serializable overlay containing ``drive_pros`` and provenance.

    Raises:
        ValueError: If candidate names or bounds do not match the configuration.
    """
    expected = {parameter.name for parameter in config.parameters}
    if set(candidate) != expected:
        missing = sorted(expected - set(candidate))
        extra = sorted(set(candidate) - expected)
        raise ValueError(f"candidate keys mismatch; missing={missing}, extra={extra}")
    drive_properties: dict[str, dict[str, float]] = {}
    normalized_candidate: dict[str, float] = {}
    for parameter in config.parameters:
        value = float(candidate[parameter.name])
        if not parameter.lower <= value <= parameter.upper:
            raise ValueError(
                f"candidate {parameter.name!r}={value:g} is outside "
                f"[{parameter.lower:g}, {parameter.upper:g}]"
            )
        drive_properties.setdefault(parameter.field, {})[parameter.selector] = value
        normalized_candidate[parameter.name] = value
    return {
        "schema_version": 1,
        "kind": _OVERLAY_KIND,
        "assets": config.asset_records(),
        "backend": config.backend,
        "device": config.device,
        "physics_dt": config.physics_dt,
        "control_frequency_hz": config.control_frequency_hz,
        "drive_pros": drive_properties,
        "calibration": {
            "claim": "effective_drive_tuning",
            "seed": config.seed,
            "candidate_count": config.candidate_count,
            "candidate": normalized_candidate,
        },
    }


def write_overlay(path: str | Path, overlay: Mapping[str, Any]) -> None:
    """Write an overlay as deterministic, human-reviewable YAML.

    Args:
        path: Destination YAML path.
        overlay: Overlay payload to serialize.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(dict(overlay), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_overlay(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a dynamics-calibration overlay.

    Args:
        path: Existing YAML overlay path.

    Returns:
        Parsed V1 overlay payload.

    Raises:
        ValueError: If the payload has an unsupported schema or shape.
    """
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("drive overlay must contain a mapping")
    if loaded.get("schema_version") != 1 or loaded.get("kind") != _OVERLAY_KIND:
        raise ValueError("unsupported dynamics-calibration drive overlay")
    if not isinstance(loaded.get("drive_pros"), dict):
        raise ValueError("drive overlay must contain drive_pros")
    return loaded


__all__ = ["build_drive_overlay", "load_overlay", "write_overlay"]
