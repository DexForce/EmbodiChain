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

from dataclasses import MISSING
import math
from pathlib import Path
from typing import Any

import yaml

from embodichain.utils import configclass

__all__ = ["SceneRandomizationCfg", "load_randomization_config"]

_GENERATION_FIELDS = (
    "seed",
    "candidate_count",
    "max_attempts",
    "collision_margin_m",
)


@configclass
class SceneRandomizationCfg:
    """Resolved scene paths and grouped layout generation parameters.

    Attributes:
        scene_path: Absolute path to a portable ``scene_export`` directory.
        base_data_path: Absolute destination for the baseline YAML file.
        schema_version: Supported input configuration schema, currently 1.
        seed: Nonnegative seed for reproducible layout proposals.
        candidate_count: Positive number of successful layouts requested.
        max_attempts: Total proposal budget, at least ``candidate_count``.
        collision_margin_m: Finite, nonnegative clearance between group AABBs.
    """

    scene_path: Path = MISSING
    base_data_path: Path = MISSING
    schema_version: int = 1
    seed: int = 0
    candidate_count: int = 10
    max_attempts: int = 1000
    collision_margin_m: float = 0.02


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
    """Load the closed version-1 YAML configuration.

    The configuration requires ``schema_version: 1`` and accepts optional
    ``seed``, ``candidate_count``, ``max_attempts`` and ``collision_margin_m``.
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
        payload, name="configuration", allowed={"schema_version", *_GENERATION_FIELDS}
    )
    version = root.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError("schema_version must be the integer 1.")
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
        schema_version=version,
        **{key: root[key] for key in _GENERATION_FIELDS if key in root},
    )
    cfg.validate()
    _validate_generation_parameters(
        seed=cfg.seed,
        candidate_count=cfg.candidate_count,
        max_attempts=cfg.max_attempts,
        collision_margin_m=cfg.collision_margin_m,
    )
    return cfg
