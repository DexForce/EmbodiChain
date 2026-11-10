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
from typing import Any, TYPE_CHECKING
from importlib import import_module

import yaml

from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.sim.scene.expansion._policy import (
        IntraGroupRandomizationCfg,
        MemberRandomizationCfg,
        SupportOverrideCfg,
    )


def _shared_policy() -> Any:
    # Keep legacy configuration/help imports independent of simulation geometry.
    return import_module("embodichain.lab.sim.scene.expansion._policy")


def __getattr__(name: str) -> Any:
    if name in {
        "IntraGroupRandomizationCfg",
        "MemberRandomizationCfg",
        "SupportOverrideCfg",
    }:
        value = getattr(_shared_policy(), name)
        globals()[name] = value
        return value
    raise AttributeError(name)


def _mapping(value: Any, *, name: str, allowed: set[str]) -> dict[str, Any]:
    # This lightweight structural check is also used before loading geometry.
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}.")
    return value


def _validate_backend(*, backend: str, device: str, batch_size: int) -> None:
    _shared_policy()._validate_backend(
        backend=backend, device=device, batch_size=batch_size
    )


def _validate_generation_parameters(
    *, seed: int, candidate_count: int, max_attempts: int, collision_margin_m: float
) -> None:
    _shared_policy()._validate_generation_parameters(
        seed=seed,
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        collision_margin_m=collision_margin_m,
    )


def _decode_intra(value: Any) -> IntraGroupRandomizationCfg | None:
    return _shared_policy()._decode_intra(value)


def _validate_intra(cfg: IntraGroupRandomizationCfg) -> None:
    _shared_policy()._validate_intra(cfg)


def _intra_payload(cfg: IntraGroupRandomizationCfg) -> dict:
    return _shared_policy()._intra_payload(cfg)


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
