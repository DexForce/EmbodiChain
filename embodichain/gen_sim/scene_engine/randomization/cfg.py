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

"""Strict configuration for preparing a scene randomization baseline."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
from typing import Any

import yaml

from embodichain.utils import configclass


@configclass
class SceneRandomizationCfg:
    """Resolved paths used to prepare a reusable scene baseline.

    Attributes:
        scene_path: Absolute path to a portable ``scene_export`` directory.
        base_data_path: Absolute destination for the baseline YAML file.
        schema_version: Supported input configuration schema, currently 1.
    """

    scene_path: Path = MISSING
    base_data_path: Path = MISSING
    schema_version: int = 1


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

    The configuration accepts only ``schema_version: 1``. The scene workspace
    is supplied separately through ``output_root``: its ``scene_export/``
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
    root = _mapping(payload, name="configuration", allowed={"schema_version"})
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
        scene_path=scene_path, base_data_path=base_data_path, schema_version=version
    )
    cfg.validate()
    return cfg
