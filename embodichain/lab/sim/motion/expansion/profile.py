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

"""Strict loading for the canonical combined Expansion Profile schema."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from embodichain.utils.utility import load_config

from .combined import CombinedExpansionProfile

__all__ = ["load_expansion_profile"]


def load_expansion_profile(path: str | Path) -> CombinedExpansionProfile:
    """Load one callable-free combined Expansion Profile.

    Args:
        path: YAML or JSON profile path.

    Returns:
        Strictly decoded and semantically validated combined profile.

    Raises:
        TypeError: If the serialized root is not a mapping.
        ValueError: If the profile contains unknown or invalid fields.
    """
    try:
        data = load_config(path)
        if not isinstance(data, Mapping):
            raise TypeError("profile root must be a mapping")
        if data.get("schema_version") != 1 or "trajectory" not in data:
            raise ValueError(
                "only schema_version=1 CombinedExpansionProfile is supported"
            )
        return CombinedExpansionProfile.from_mapping(data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Expansion Profile {path}: {error}") from error
