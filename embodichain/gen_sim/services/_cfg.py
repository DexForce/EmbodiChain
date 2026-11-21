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

"""Public configclass definition, loaded only for explicit model profiles."""

from __future__ import annotations
from dataclasses import field
from typing import Any
from embodichain.utils import configclass
from .errors import ServiceConfigurationError

__all__: list[str] = []


@configclass
class GenSimServicesCfg:
    """Named adapter profiles and role bindings, with no implicit fallbacks.

    A profile contains ``adapter``, optional ``model``, ``base_url``, ``api_key``
    or ``api_key_env``, positive ``timeout_s``/``max_attempts`` and ``options``.
    Connection fields can instead reference environment values via
    ``{"env": "VARIABLE_NAME"}``. Profiles are resolved only when used.
    """

    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.profiles, dict) or not isinstance(self.roles, dict):
            raise ServiceConfigurationError("profiles and roles must be mappings")
        for role, profile in self.roles.items():
            if (
                not isinstance(role, str)
                or not isinstance(profile, str)
                or profile not in self.profiles
            ):
                raise ServiceConfigurationError("Every role must bind a named profile")
        allowed = {
            "adapter",
            "model",
            "base_url",
            "api_key",
            "api_key_env",
            "timeout_s",
            "max_attempts",
            "options",
        }
        for profile in self.profiles.values():
            if (
                not isinstance(profile, dict)
                or set(profile) - allowed
                or (
                    not isinstance(profile.get("adapter"), str)
                    or not profile.get("adapter")
                )
            ):
                raise ServiceConfigurationError("Invalid model service profile")
            if not isinstance(profile.get("options", {}), dict):
                raise ServiceConfigurationError("Profile options must be a mapping")
            if "api_key" in profile and "api_key_env" in profile:
                raise ServiceConfigurationError(
                    "Select api_key or api_key_env, not both"
                )
