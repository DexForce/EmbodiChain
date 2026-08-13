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

"""Motion Profile providers for task-specific model and control semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

__all__ = [
    "MotionProfile",
    "MotionProfileRequest",
    "build_motion_profile",
    "get_motion_profile_names",
    "register_motion_profile",
]


@dataclass(frozen=True)
class MotionProfileRequest:
    """Checkpoint, configs, and runtime choices supplied to a provider."""

    checkpoint: Path
    device: torch.device
    configs: Mapping[str, Path] = field(default_factory=dict)
    resource_root: Path | None = None
    renderer: str = "hybrid"

    def __post_init__(self) -> None:
        checkpoint = Path(self.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Motion checkpoint does not exist: {checkpoint}")
        configs = {
            name: Path(path).expanduser().resolve()
            for name, path in self.configs.items()
        }
        for name, path in configs.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Motion config {name!r} does not exist: {path}"
                )
        root = (
            None
            if self.resource_root is None
            else Path(self.resource_root).expanduser().resolve()
        )
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "configs", MappingProxyType(configs))
        object.__setattr__(self, "resource_root", root)


@dataclass(frozen=True)
class MotionProfile:
    """DexSim Policy Spec and report metadata built by one provider."""

    profile_id: str
    checkpoint: Path
    policy_spec: Mapping[str, Any]
    provider_version: int = 1
    provenance: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        checkpoint = Path(self.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Motion checkpoint does not exist: {checkpoint}")
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(
            self,
            "policy_spec",
            MappingProxyType(dict(self.policy_spec)),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))


MotionProfileProvider = Callable[[MotionProfileRequest], MotionProfile]
_PROVIDERS: dict[str, MotionProfileProvider] = {}


def register_motion_profile(name: str, provider: MotionProfileProvider) -> None:
    """Register a Motion Profile provider under its CLI name.

    Args:
        name: Stable profile name.
        provider: Callable that builds one :class:`MotionProfile`.
    """
    if not name:
        raise ValueError("Motion profile name must not be empty")
    if name in _PROVIDERS:
        raise ValueError(f"Motion profile is already registered: {name}")
    _PROVIDERS[name] = provider


def get_motion_profile_names() -> tuple[str, ...]:
    """Return the registered profile names."""
    return tuple(sorted(_PROVIDERS))


def build_motion_profile(
    name: str,
    request: MotionProfileRequest,
) -> MotionProfile:
    """Build one profile with its registered provider.

    Args:
        name: Registered profile name.
        request: Checkpoint, configs, and runtime choices.

    Returns:
        Provider-built Motion Profile.
    """
    try:
        provider = _PROVIDERS[name]
    except KeyError:
        available = ", ".join(sorted(_PROVIDERS)) or "none"
        raise ValueError(
            f"Unknown motion profile {name!r}; available: {available}"
        ) from None
    profile = provider(request)
    if profile.profile_id != name:
        raise ValueError(
            f"Motion provider {name!r} returned profile {profile.profile_id!r}"
        )
    return profile
