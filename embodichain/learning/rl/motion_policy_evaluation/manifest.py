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

"""Minimal index connecting a training run to motion-policy evaluation."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = ["RUN_MANIFEST_NAME", "RunManifest", "write_run_manifest"]

RUN_MANIFEST_NAME = "run-manifest.json"


@dataclass(frozen=True)
class RunManifest:
    """Resolved paths from one EmbodiChain training run."""

    root: Path
    motion_profile: str | None
    configs: Mapping[str, Path]
    checkpoints: Mapping[str, Path | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "configs", MappingProxyType(dict(self.configs)))
        object.__setattr__(
            self,
            "checkpoints",
            MappingProxyType(dict(self.checkpoints)),
        )

    @classmethod
    def load(cls, run: str | Path) -> RunManifest:
        """Load ``run-manifest.json`` and resolve its referenced files.

        Args:
            run: EmbodiChain training run directory.

        Returns:
            Resolved manifest.
        """
        root = Path(run).expanduser().resolve()
        path = root / RUN_MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"Run manifest does not exist: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise ValueError(f"Unsupported run manifest: {path}")
        configs = _resolve_group(root, value.get("configs"), "configs")
        checkpoints = _resolve_group(
            root,
            value.get("checkpoints"),
            "checkpoints",
            allow_none=True,
        )
        profile = value.get("motion_profile")
        if profile is not None and not isinstance(profile, str):
            raise TypeError("Run manifest motion_profile must be a string or null")
        return cls(root, profile, configs, checkpoints)

    def select_checkpoint(self, requested: str = "best") -> tuple[str, Path]:
        """Select ``best`` or ``latest`` and return its resolved path.

        Args:
            requested: Checkpoint role.

        Returns:
            Selected role and checkpoint path. ``best`` uses ``latest`` when
            the training run has no best checkpoint.
        """
        if requested not in {"best", "latest"}:
            raise ValueError("checkpoint role must be best or latest")
        selected = requested
        checkpoint = self.checkpoints.get(selected)
        if checkpoint is None and requested == "best":
            selected = "latest"
            checkpoint = self.checkpoints.get(selected)
        if checkpoint is None:
            raise FileNotFoundError(
                f"Run manifest has no {requested} checkpoint: {self.root}"
            )
        return selected, checkpoint


def write_run_manifest(
    run: str | Path,
    *,
    train_config: str | Path,
    latest_checkpoint: str | Path,
    best_checkpoint: str | Path | None = None,
    gym_config: str | Path | None = None,
    motion_profile: str | None = None,
) -> Path:
    """Snapshot training configs and write the minimal run manifest.

    Args:
        run: Training run directory containing the checkpoints.
        train_config: Training config used for the run.
        latest_checkpoint: Final saved checkpoint.
        best_checkpoint: Best checkpoint when evaluation selected one.
        gym_config: Referenced task config when the trainer uses one.
        motion_profile: Default Motion Profile for visual evaluation.

    Returns:
        Written manifest path.
    """
    root = Path(run).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_dir = root / "configs"
    config_dir.mkdir(exist_ok=True)
    configs = {
        "train": _snapshot_config(train_config, config_dir, "train"),
    }
    if gym_config is not None:
        configs["gym"] = _snapshot_config(gym_config, config_dir, "gym")
    checkpoints = {
        "best": _relative_file(root, best_checkpoint),
        "latest": _relative_file(root, latest_checkpoint),
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "motion_profile": motion_profile,
        "configs": configs,
        "checkpoints": checkpoints,
    }
    path = root / RUN_MANIFEST_NAME
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _snapshot_config(source: str | Path, target: Path, name: str) -> str:
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {path}")
    suffix = path.suffix.lower() if path.suffix else ".yaml"
    destination = target / f"{name}{suffix}"
    shutil.copyfile(path, destination)
    return destination.relative_to(target.parent).as_posix()


def _relative_file(root: Path, value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training checkpoint does not exist: {path}")
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Training checkpoint is outside its run: {path}") from error


def _resolve_group(
    root: Path,
    value: object,
    field: str,
    *,
    allow_none: bool = False,
) -> dict[str, Path | None]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Run manifest {field} must be a mapping")
    result: dict[str, Path | None] = {}
    for name, reference in value.items():
        if reference is None and allow_none:
            result[str(name)] = None
            continue
        if not isinstance(reference, str) or not reference:
            raise TypeError(f"Run manifest {field}.{name} must be a path")
        relative = Path(reference)
        if relative.is_absolute():
            raise ValueError(f"Run manifest {field}.{name} must be relative")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"Run manifest {field}.{name} escapes the run directory"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"Run manifest file does not exist: {path}")
        result[str(name)] = path
    return result
